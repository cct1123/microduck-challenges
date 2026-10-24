"""Explore/exploit controls and fail-closed Sprint promotion contracts."""

import pytest
import torch
from copy import deepcopy
import os
import json
import subprocess
import sys
from typing import Any, cast
from rsl_rl.modules.distribution import GaussianDistribution

from microduck_challenges.sprint_2m import env as sprint_env
from microduck_challenges.sprint_2m import rewards
from microduck_challenges.sprint_2m import promotion
from microduck_challenges.sprint_2m import campaign


def test_default_exploration_is_a_bounded_free_leg_screen():
    assert sprint_env.RL_CFG.max_iterations == 500
    assert sprint_env.TRAINING_PHASE == "explore"
    assert sprint_env.TRAINING_RECIPE["max_iterations"] == 500
    assert sprint_env.RL_CFG.save_interval == 50
    algorithm = sprint_env.RL_CFG.algorithm
    assert algorithm.learning_rate == pytest.approx(5e-5)
    assert algorithm.entropy_coef == pytest.approx(0.01)
    assert algorithm.clip_param == pytest.approx(0.2)
    assert getattr(algorithm, "exploration_std_scale") == 1.0
    cfg = sprint_env.make_env_cfg()
    assert cfg.rewards["pose"].weight == 0.0
    assert cfg.actions["joint_pos"].clip is None


@pytest.mark.parametrize("source_scale,target_scale,factor", [
    (1.0, 1.0, 1.0), (1.0, 0.5, 0.5), (0.5, 0.5, 1.0), (0.5, 1.0, 2.0),
])
def test_phase_transition_changes_only_restored_training_noise(source_scale, target_scale, factor):
    algorithm = cast(Any, object.__new__(rewards.SprintRefinePPO))
    algorithm.actor, algorithm.critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    algorithm.actor.add_module("distribution", GaussianDistribution(1, init_std=0.4))
    algorithm.rnd = None
    algorithm.exploration_std_scale = target_scale
    payload = {"actor_state_dict": deepcopy(algorithm.actor.state_dict()),
               "critic_state_dict": deepcopy(algorithm.critic.state_dict()),
               "infos": {"env_state": {"common_step_counter": 222648}},
               "sprint_exploration_std_scale": source_scale}
    original = deepcopy(payload)
    assert algorithm.load(payload) is False
    for key, value in algorithm.actor.state_dict().items():
        expected = original["actor_state_dict"][key]
        torch.testing.assert_close(value, expected * factor if "std_param" in key else expected)
    for key, value in payload["actor_state_dict"].items():
        torch.testing.assert_close(value, original["actor_state_dict"][key], rtol=0, atol=0)
    assert payload["infos"] == original["infos"]


def test_exploitation_cannot_start_without_a_passing_promotion_report():
    env = {**os.environ, "MICRODUCK_SPRINT_PHASE": "exploit"}
    env.pop("MICRODUCK_SPRINT_PROMOTION", None)
    result = subprocess.run([sys.executable, "-c", "import microduck_challenges"],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "approved promotion report" in result.stderr


@pytest.fixture
def race_evidence():
    """Synthetic unit-test races only; never used as real training approval."""
    gate = dict(sprint_env.STAGE_SETTINGS["gate"])
    def rows(seeds, time):
        return [{"seed": seed, "status": "finish", "time_s": time,
                 "command_vx": gate["command_vx"]} for seed in seeds]
    return {"schema": 1, "status": "complete", "gate": gate, "checkpoints": [0, 50, 99],
            "candidate_checkpoint": 50,
            "screen": {str(n): rows(range(1, 6), 1.1) for n in (0, 50, 99)},
            "candidate_heldout": rows(range(6, 26), 1.15),
            "incumbent_rows": rows(range(1, 26), 1.2)}


def test_promotion_needs_a_stable_neighborhood_and_full_heldout_success(race_evidence):
    report = race_evidence
    result = promotion.promotion_decision(report, report["gate"])
    assert result["approved"] and result["stable_neighbors"] == [0, 50, 99]
    assert not result["target_reached"]
    report["screen"]["99"][4].update(status="fall", time_s=None)
    result = promotion.promotion_decision(report, report["gate"])
    assert not result["approved"]
    assert "neighborhood" in result["reasons"][0]


@pytest.mark.parametrize("fault", ["missing_seed", "duplicate_seed", "wrong_command", "nan_time", "missing_checkpoint"])
def test_incomplete_or_invalid_evidence_is_never_promoted(race_evidence, fault):
    report = race_evidence
    rows = report["candidate_heldout"]
    if fault == "missing_seed":
        rows.pop()
    elif fault == "duplicate_seed":
        rows[-1]["seed"] = 6
    elif fault == "wrong_command":
        rows[-1]["command_vx"] = 0.4
    elif fault == "nan_time":
        rows[-1]["time_s"] = float("nan")
    else:
        del report["screen"]["99"]
    with pytest.raises(ValueError):
        promotion.promotion_decision(report, report["gate"])


def test_heldout_falls_are_scored_as_fifty_seconds_not_dropped(race_evidence):
    report = race_evidence
    report["candidate_heldout"][-1].update(status="fall", time_s=None)
    result = promotion.promotion_decision(report, report["gate"])
    assert not result["approved"]
    assert result["candidate_heldout"]["finishes"] == 19
    assert result["candidate_heldout"]["mean_s"] == pytest.approx((19 * 1.15 + 50) / 20)


def test_promotion_cannot_bind_another_checkpoint_to_passing_seed_rows(tmp_path, race_evidence):
    report = race_evidence
    run = tmp_path / "candidate"
    (run / "params").mkdir(parents=True)
    for n in report["checkpoints"]:
        (run / f"model_{n}.pt").write_bytes(b"unit-test placeholder, never a real policy")
    for name in ("params/env.yaml", "params/agent.yaml", "provenance.json"):
        (run / name).write_text("{}")
    incumbent = tmp_path / "incumbent.pt"
    incumbent.write_bytes(b"unit-test placeholder")
    wrong = run / "model_99.pt"
    report.update(candidate_path=str(wrong), incumbent_path=str(incumbent),
                  artifact_hashes={str(p): promotion.digest(p)
                                   for p in [*run.rglob("*.*"), incumbent] if p.is_file()})
    evidence = tmp_path / "report.json"
    evidence.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identifier"):
        promotion.require_promotion(evidence, wrong, report["gate"])


def test_launch_plan_is_bounded_explicit_and_does_not_launch():
    make_plan = cast(Any, getattr(campaign, "make_plan", None))
    assert callable(make_plan), "the strategy needs a bounded launcher"
    plan = cast(dict[str, Any], make_plan("explore"))
    assert plan["max_iterations"] == 500
    assert plan["environment"]["MICRODUCK_SPRINT_PHASE"] == "explore"
    assert plan["source"].endswith("model_8250.pt")
    assert "--gpu-ids" in plan["command"] and "all" in plan["command"]
    command = cast(list[str], plan["command"])
    assert command[command.index("--agent.upload-model") + 1] == "False"
    with pytest.raises(ValueError, match="budget"):
        make_plan("explore", iterations=501)


def test_actor_export_uses_actual_checkpoint_and_preserves_source(tmp_path):
    export_actor = cast(Any, getattr(campaign, "export_actor", None))
    assert callable(export_actor), "assessment needs an isolated verified actor export"
    source = sprint_env.REFINE_CHECKPOINT["path"]
    before = promotion.digest(source)
    output = tmp_path / "policy.onnx"
    metadata = cast(dict[str, Any], export_actor(source, output))
    assert output.is_file() and metadata["parity_cases"] >= 3
    assert metadata["input_shape"] == [1, 61] and metadata["output_shape"] == [1, 14]
    assert promotion.digest(source) == before


def test_bounded_launcher_refuses_a_second_trainer_before_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "active_trainers", lambda: [12345], raising=False)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("preflight must not start"))
    with pytest.raises(ValueError, match="trainer"):
        campaign.run_bounded({"requires_promotion": False, "max_wall_minutes": 40}, tmp_path / "run")


@pytest.fixture
def passing_report_file(tmp_path, race_evidence):
    """Artifact-binding tests use explicit dummy files, never trained policies."""
    report = race_evidence
    run = tmp_path / "source"
    (run / "params").mkdir(parents=True)
    for n in report["checkpoints"]:
        (run / f"model_{n}.pt").write_bytes(b"unit-test-only checkpoint placeholder")
    (run / "params/env.yaml").write_text("{}")
    (run / "params/agent.yaml").write_text('{"max_iterations":100,"save_interval":50}')
    (run / "provenance.json").write_text("{}")
    incumbent = tmp_path / "incumbent.pt"
    incumbent.write_bytes(b"unit-test-only incumbent")
    source = run / "model_50.pt"
    report.update(candidate_path=str(source), incumbent_path=str(incumbent),
                  artifact_hashes={str(p): promotion.digest(p)
                                   for p in [*run.rglob("*.*"), incumbent] if p.is_file()})
    report["runtime"] = promotion.runtime_signature()
    path = tmp_path / "assessment.json"
    path.write_text(json.dumps(report))
    return path, source, report


def test_promotion_rechecks_hashes_not_the_saved_approved_flag(passing_report_file):
    path, source, report = passing_report_file
    report["decision"] = {"approved": False}
    path.write_text(json.dumps(report))
    assert promotion.require_promotion(path, source, report["gate"])
    source.write_bytes(b"changed unit-test placeholder")
    with pytest.raises(ValueError, match="changed"):
        promotion.require_promotion(path, source, report["gate"])


def test_duplicate_json_keys_cannot_approve_exploitation(passing_report_file):
    path, source, report = passing_report_file
    path.write_text('{"schema":1,' + json.dumps(report)[1:])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        promotion.require_promotion(path, source, report["gate"])


def test_deleted_checkpoints_cannot_create_a_fake_adjacent_neighborhood(passing_report_file):
    path, source, report = passing_report_file
    agent = source.parent / "params/agent.yaml"
    agent.write_text('{"max_iterations":200,"save_interval":50}')
    report["artifact_hashes"][str(agent)] = promotion.digest(agent)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="[Ii]ncomplete|inventory"):
        promotion.require_promotion(path, source, report["gate"])


def test_changed_evaluator_invalidates_old_promotion(passing_report_file):
    path, source, report = passing_report_file
    report["runtime"]["versions"]["torch"] = "different-unit-test-runtime"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="runtime"):
        promotion.require_promotion(path, source, report["gate"])


def test_refinement_rejects_inconsistent_saved_noise_phase(tmp_path):
    selected = deepcopy(sprint_env.REFINE_CHECKPOINT)
    artifact = tmp_path / "unit-test-noise-marker.pt"
    torch.save({"infos": {"env_state": {"common_step_counter": 222648}},
                "sprint_exploration_std_scale": 0.5}, artifact)
    selected["path"] = artifact
    with pytest.raises(ValueError, match="noise"):
        rewards.configure_refine_locomotion(sprint_env.make_env_cfg(stage="baseline"),
                                            sprint_env.REFINE_SETTINGS, selected)


@pytest.mark.parametrize("outcome,expected", [(0, "completed"), (7, "failed"), ("timeout", "wall_budget_exhausted")])
def test_bounded_process_lifecycle_is_journaled(tmp_path, monkeypatch, outcome, expected):
    """Only the subprocess boundary is faked; this is not a GPU training result."""
    plan = campaign.make_plan("explore")
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/sprint_2m")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "smoke", lambda *a: None)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: None)
    signals = []
    class Process:
        pid = 12345
        returncode = None
        calls = 0
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            self.calls += 1
            self.returncode = -15 if outcome == "timeout" else outcome
            return self.returncode
    if outcome == "timeout":
        clock = iter((0.0, 3000.0))
        monkeypatch.setattr(campaign, "monotonic", lambda: next(clock))
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(campaign.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    output = tmp_path / "run"
    result = campaign.run_bounded(plan, output)
    assert result["state"] == expected
    assert json.loads((output / "run.json").read_text())["state"] == expected
    assert bool(signals) == (outcome == "timeout")
