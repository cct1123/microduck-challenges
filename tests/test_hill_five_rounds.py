"""Five-round policy contracts; runtime boundaries remain guarded/mocked."""
import importlib
import importlib.util
import time

import pytest

from microduck_challenges.hill_climb_1m5 import campaign, overnight


def feature():
    name = "microduck_challenges.hill_climb_1m5.five_rounds"
    # The legacy policy is the actual pre-feature behavior, not a fake oracle.
    return importlib.import_module(name) if importlib.util.find_spec(name) else overnight


def state():
    return {"state": "prepared", "policy": "five-rounds-v1", "start_epoch": time.time(),
            "training_deadline_epoch": time.time()+21600, "hard_deadline_epoch": time.time()+28800,
            "earliest_speed_epoch": 1792923355., "screens": [], "current_screen": None,
            "anchor": None, "best_candidate": None, "best_reliable": None,
            "reliable_sources": [], "final": None, "original_source": {"path": "initial"},
            "maximum_new_screens": 5, "p1_disabled": False}


def test_five_not_legacy_two_without_gate_and_seeds_three_to_seven():
    module, data = feature(), state()
    for seed in range(3, 8):
        assert module.next_action(data, time.time()) == ("S0", seed)
        data["screens"].append({"ordinal": seed-2, "profile": "S0", "seed": seed, "state": "completed"})
    assert module.next_action(data, time.time()) == ("finalize", None)
    assert overnight.next_action({**data, "screens": data["screens"][:2]}, time.time())[0] == "stop_no_stability"


@pytest.fixture
def module():
    return feature()


@pytest.fixture
def local_auth():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "logs/campaigns/hill-five-rounds-20261025-044013/authorization.json"
    if not path.exists():
        pytest.skip("local pinned campaign evidence absent")
    return path


def controller(module, tmp_path, data=None):
    result = module.Controller(tmp_path / "auth.json", tmp_path / "controller")
    result.journal = module.Journal.create(result.output / "campaign.json", data or state())
    return result


def test_prepare_normal_source_keeps_legacy_source_compatible_and_anchor_null(module, tmp_path, local_auth, monkeypatch):
    auth = overnight.validate_authorization(local_auth)
    monkeypatch.setattr(module.time, "time", lambda: auth["start_epoch"] + 60)
    source = auth["initial_source"]["path"]
    before = campaign.selected_source(source, phase="explore")
    ctl = module.Controller(local_auth, tmp_path / "controller")
    ctl.journal = ctl.prepare()
    ctl.verify_inputs()
    assert ctl.state["best_reliable"]["path"] == source
    assert ctl.state["best_reliable"]["development"]["finishes"] == 5
    assert ctl.state["anchor"] is None and ctl.state["best_candidate"] is None
    assert ctl.state["start_epoch"] == 1792921213
    assert ctl.state["training_deadline_epoch"] == 1792942813
    assert campaign.selected_source(source, phase="explore")["source_identity"] == before["source_identity"]
    inputs = overnight.read_report(ctl.state["inputs_path"])
    assert str(__import__("pathlib").Path(module.__file__).resolve()) in inputs["files"]
    assert str(__import__("pathlib").Path(__file__).resolve()) in inputs["files"]


@pytest.mark.parametrize("field", ["policy", "requested_additional_rounds", "training_seeds", "speed_not_before_epoch",
                                   "sha256", "profile", "prior_summary_sha256", "fixed_evaluation_sha256", "development"])
def test_prepare_refuses_changed_initial_normal_source_and_fixed_pins(module, tmp_path, local_auth, field):
    auth = overnight.read_report(local_auth)
    target = auth if field in ("policy", "requested_additional_rounds", "training_seeds", "speed_not_before_epoch") else auth["initial_source"]
    target[field] = "tampered"
    path = tmp_path / "bad-auth.json"
    overnight.durable_json(path, auth)
    ctl = module.Controller(path, tmp_path / "controller")
    with pytest.raises((ValueError, OSError, TypeError)):
        ctl.prepare()
    assert not (ctl.output / "campaign.json").exists()


def test_policy_gates_clock_rollback_and_reservations(module):
    data = state()
    data["screens"] = [{"profile": "S0", "state": "completed"}]
    data["anchor"] = {"path": "R"}
    assert module.next_action(data, 1792923354.) == ("S0", 4)
    assert module.next_action(data, 1792923355.) == ("P1", 4)
    data["p1_disabled"] = True
    assert module.next_action(data, 1792923355.) == ("S0", 4)
    data["current_screen"] = {"ordinal": 2, "state": "reserved"}
    with pytest.raises(ValueError, match="reservation"):
        module.next_action(data, 1792923355.)
    data["current_screen"] = None
    with pytest.raises(TimeoutError, match="deadline"):
        module.next_action(data, data["training_deadline_epoch"])


def test_monotonic_best_is_enforced_before_journal_persistence(module, tmp_path):
    ctl = controller(module, tmp_path)
    r = {"path": "R", "profile": "S0", "development": {"mean_s": 5.}}
    fast = {"path": "fast", "profile": "P1", "development": {"mean_s": 3.}}
    slow = {"path": "slow", "profile": "P1", "development": {"mean_s": 4.}}
    ctl.journal.record("reliable_anchor_frozen", anchor=r, best_candidate=r)
    ctl.journal.record("speed_candidate_frozen_for_final", best_candidate=fast)
    ctl.journal.record("speed_candidate_frozen_for_final", best_candidate=slow)
    assert module.Journal.load(ctl.journal.path).data["best_candidate"] == fast
    with pytest.raises(ValueError, match="anchor"):
        ctl.journal.record("bad_replacement", anchor=slow)
    assert ctl.state["anchor"] == r


def test_final_heldouts_forbidden_until_all_five_reconciled(module, tmp_path, monkeypatch):
    ctl = controller(module, tmp_path)
    monkeypatch.setattr(ctl, "evaluate", lambda *a, **kw: pytest.fail("early heldouts"))
    with pytest.raises(ValueError, match="five"):
        ctl.finalize()
    ctl.state["screens"] = [{"state": "completed", "profile": "S0"} for _ in range(5)]
    ctl.finalize()
    assert ctl.state["state"] == "stability_unmet" and not ctl.state["final"]


def test_run_five_rounds_and_restart_never_launches_sixth(module, tmp_path, monkeypatch):
    ctl = module.Controller(tmp_path / "auth.json", tmp_path / "controller")
    monkeypatch.setattr(ctl, "prepare", lambda: module.Journal.create(ctl.output / "campaign.json", state()))
    monkeypatch.setattr(ctl, "verify_inputs", lambda: None)
    monkeypatch.setattr(ctl, "reconcile_preexisting", lambda: None)
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    calls = []
    def screen(profile, seed):
        calls.append((profile, seed))
        return {"ordinal": len(calls), "profile": profile, "seed": seed, "state": "completed", "reliable": []}
    monkeypatch.setattr(ctl, "run_screen", screen)
    result = ctl.run()
    assert calls == [("S0", s) for s in range(3, 8)]
    assert result["state"] == "stability_unmet" and result["round_counts"]["completed"] == 5
    assert result["round_counts"]["reserved_or_uncertain"] == 0
    ctl.run()
    assert len(calls) == 5


def test_uncertain_reserved_restart_counts_attempt_without_duplicate(module, tmp_path, monkeypatch):
    data = state()
    data["current_screen"] = {"ordinal": 1, "profile": "S0", "state": "reserved"}
    ctl = controller(module, tmp_path, data)
    monkeypatch.setattr(ctl, "verify_inputs", lambda: None)
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(ctl, "run_screen", lambda *a: pytest.fail("duplicate launch"))
    result = ctl.run()
    assert result["state"] == "interrupted"
    assert result["round_counts"]["reserved_or_uncertain"] == 1


def test_deadline_and_competing_trainer_fail_closed_before_launch(module, tmp_path, monkeypatch):
    ctl = module.Controller(tmp_path / "auth.json", tmp_path / "controller")
    data = state()
    data["training_deadline_epoch"] = 1.
    monkeypatch.setattr(ctl, "prepare", lambda: module.Journal.create(ctl.output / "campaign.json", data))
    monkeypatch.setattr(ctl, "verify_inputs", lambda: None)
    monkeypatch.setattr(ctl, "reconcile_preexisting", lambda: None)
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(ctl, "run_screen", lambda *a: pytest.fail("expired launch"))
    result = ctl.run()
    assert result["state"] == "deadline_exhausted" and result["round_counts"]["attempts"] == 0


# Reuse existing synthetic runtime-boundary fixtures, not a second evidence verifier.
from test_hill_overnight import stable_evidence, worker_report_factory


@pytest.mark.parametrize("prior_p1", [False, True])
def test_p1_plan_requires_own_gate_and_preserves_bounded_controls(module, tmp_path, monkeypatch, stable_evidence, prior_p1):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import evidence, learning
    gate, report = stable_evidence
    source = Path(report["candidate_path"])
    selected = {"path": source, "profile": "P1" if prior_p1 else "S0", "partial_exception": None,
                "clock": 24024, "curriculum_clock": 24024, "reward_origin": 24024,
                "base_knobs": learning.APPROVED_PARTIAL["base_knobs"],
                "source_identity": {"path": str(source)},
                "result_path": source.parent / "workshop/model_0.json"}
    monkeypatch.setattr(campaign, "selected_source", lambda *a, **kw: selected)
    ctl = controller(module, tmp_path)
    for path in (ctl.authorization, ctl.output / "inputs.json", ctl.output / "authorization.json"):
        overnight.durable_json(path, {})
    record = {"path": str(source), "sha256": evidence.digest(source), "profile": selected["profile"],
              "gate_path": str(gate), "gate_sha256": evidence.digest(gate), "development": {"mean_s": 5.}}
    ctl.state.update(anchor=record, best_candidate=record,
                     screens=[{"profile": "S0", "state": "completed"},
                              *([{"profile": "P1", "state": "completed"}] if prior_p1 else [])])
    seed = 5 if prior_p1 else 4
    monkeypatch.setattr(module.time, "time", lambda: 1792923354.)
    with pytest.raises(ValueError, match="P1"):
        ctl.launch_plan("P1", seed, source)
    monkeypatch.setattr(module.time, "time", lambda: 1792923355.)
    plan = ctl.launch_plan("P1", seed, source)
    assert plan["max_iterations"] == 500 and plan["max_wall_minutes"] == 50
    assert plan["controls"]["num_envs"] == 2048 and plan["controls"]["save_interval"] == 50
    assert plan["environment"]["MICRODUCK_HILL_STABILITY_GATE"] == str(gate)
    assert plan["environment"]["MICRODUCK_HILL_STABILITY_GATE_SHA256"] == evidence.digest(gate)
    assert plan["target_reward_weights"]["horizontal_forward_speed"] == 1.
    assert plan["target_algorithm"]["learning_rate"] == .00005
    assert plan["command"][plan["command"].index("--agent.upload-model")+1] == "False"
    assert "MICRODUCK_HILL_STABILITY_GATE" not in __import__("os").environ
    with pytest.raises(ValueError, match="exact source"):
        ctl.launch_plan("P1", seed, source.parent / "model_50.pt")
    ctl.state["p1_disabled"] = True
    with pytest.raises(ValueError, match="P1"):
        ctl.launch_plan("P1", seed, source)
    ctl.state["screens"] = [{"profile": "S0", "state": "completed"} for _ in range(5)]
    with pytest.raises(ValueError, match="exhausted"):
        ctl.launch_plan("S0", 8, source)


def test_real_run_screen_no_smoke_reserves_before_runtime_and_rollback_reverts(module, tmp_path, monkeypatch, stable_evidence):
    gate, evidence = stable_evidence
    source = evidence["candidate_path"]
    ctl = controller(module, tmp_path)
    initial = {"path": source, "sha256": overnight.digest(source), "profile": "S0",
               "development": {"mean_s": 5., "finishes": 5, "total": 5}}
    ctl.state.update(best_reliable=initial, reliable_sources=[initial],
                     screens=[{"ordinal": 1, "profile": "S0", "state": "completed"}])
    r = {**initial, "gate_path": str(gate), "gate_sha256": overnight.digest(gate)}
    ctl.journal.record("reliable_anchor_frozen", anchor=r, best_candidate=r)
    monkeypatch.setattr(ctl, "verify_inputs", lambda: None)
    monkeypatch.setattr(ctl, "launch_plan", lambda *a: {"reward_profile": "P1"})
    class Monitor:
        run = None
        def __init__(self, *a): pass
        def close(self): pass
    monkeypatch.setattr(overnight, "ScreenMonitor", Monitor)
    def launch(plan, output, **kw):
        on_disk = module.Journal.load(ctl.journal.path).data
        assert on_disk["current_screen"]["state"] == "reserved"
        assert kw["no_smoke"] is True
        assert kw["global_deadline"] == ctl.state["training_deadline_epoch"]
        assert kw["monitor"] is not None
        return {"state": "branch_rollback", "exit_code": -15}
    monkeypatch.setattr(campaign, "run_bounded", launch)
    record = ctl.run_screen("P1", 4)
    assert record["state"] == "branch_rollback"
    assert ctl.state["p1_disabled"] is True
    assert ctl.state["best_reliable"] == initial and ctl.state["best_candidate"] == r
    assert ctl.state["anchor"] == r
    assert ctl.state["reliable_sources"] == [initial]
    with pytest.raises(ValueError, match="reservation"):
        ctl.run_screen("S0", 3)


@pytest.mark.parametrize("tamper", [False, True])
def test_after_five_final_opens_one_candidate_paired_with_original_r_once(module, tmp_path, monkeypatch, worker_report_factory, tamper):
    r, b = tmp_path / "R.pt", tmp_path / "B.pt"
    r.write_bytes(b"R"); b.write_bytes(b"B")
    anchor = {"path": str(r), "sha256": overnight.digest(r), "profile": "S0", "development": {"mean_s": 4.}}
    candidate = {"path": str(b), "sha256": overnight.digest(b), "profile": "P1", "development": {"mean_s": 3.}}
    ctl = controller(module, tmp_path)
    ctl.state.update(anchor=anchor, best_candidate=candidate,
                     screens=[{"state": "completed", "profile": "S0"} for _ in range(5)])
    cutoff = ctl.state["training_deadline_epoch"]
    monkeypatch.setattr(ctl, "verify_inputs", lambda: None)
    monkeypatch.setattr(ctl, "wait_until", lambda *a: pytest.fail("unnecessary six-hour wait"))
    calls = []
    def evaluate(path, out, seeds, **kw):
        assert ctl.state["final"]["heldouts_opened"] and seeds == list(range(6, 26))
        calls.append(str(path))
        if str(path) == str(r): assert ctl.state["final"]["candidate_rows"] is not None
        result = worker_report_factory(path, out, seeds, **kw)
        if tamper and str(path) == str(r):
            (ctl.output / "final-candidate/rows.jsonl").write_text("tampered after paired race")
        return result
    monkeypatch.setattr(ctl, "evaluate", evaluate)
    if tamper:
        with pytest.raises(ValueError): ctl.finalize()
        assert ctl.state["state"] != "complete"
    else:
        ctl.finalize()
        assert ctl.state["state"] == "complete"
    assert calls == [str(b), str(r)]
    assert ctl.state["anchor"] == anchor and ctl.state["training_deadline_epoch"] < cutoff
    with pytest.raises(ValueError, match="opened"):
        ctl.finalize()
    assert calls == [str(b), str(r)]


def test_cli_default_is_nonlaunching(module, tmp_path, local_auth, capsys):
    module.main(["run", "--authorization", str(local_auth), "--output", str(tmp_path / "unused")])
    assert '"launching": false' in capsys.readouterr().out
    assert not (tmp_path / "unused").exists()


def test_retained_global_source_is_reverified_not_just_previously_five_of_five(module, tmp_path, stable_evidence, worker_report_factory):
    from pathlib import Path
    gate, evidence = stable_evidence
    source = Path(evidence["candidate_path"])
    assessment = tmp_path / "assessment"
    for number in evidence["checkpoints"]:
        worker_report_factory(source.parent / f"model_{number}.pt", assessment / f"model-{number}", overnight.DEV)
    ctl = controller(module, tmp_path)
    record = {"path": str(source), "sha256": overnight.digest(source), "profile": "S0",
              "development": {"finishes": 5, "total": 5, "mean_s": 3.},
              "official": overnight.official_summary(source.parent / f"workshop/{source.stem}.json",
                                                     int(source.stem.removeprefix("model_")))}
    ctl.state.update(screens=[{
        "ordinal": 1, "profile": "S0", "state": "completed", "run": str(source.parent),
        "assessment": str(assessment), "expected_checkpoints": evidence["checkpoints"],
        "evaluated_checkpoints": evidence["checkpoints"], "pending_checkpoints": []}])
    ctl.journal.record("global_reliable_sources_updated", reliable_sources=[record], best_reliable=record)
    assert str(ctl.choose_source("S0")) == str(source)
    (assessment / "model-50/rows.jsonl").write_text("changed after complete assessment")
    with pytest.raises(ValueError, match="hash|tamper|rows"):
        ctl.choose_source("S0")


def test_competing_trainer_stops_new_campaign_without_any_launch(module, tmp_path, monkeypatch):
    ctl = module.Controller(tmp_path / "auth.json", tmp_path / "controller")
    monkeypatch.setattr(ctl, "prepare", lambda: module.Journal.create(ctl.output / "campaign.json", state()))
    monkeypatch.setattr(ctl, "verify_inputs", lambda: None)
    monkeypatch.setattr(ctl, "reconcile_preexisting", lambda: None)
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [12345])
    monkeypatch.setattr(ctl, "run_screen", lambda *a: pytest.fail("competing launch"))
    result = ctl.run()
    assert result["state"] == "blocked" and result["round_counts"]["attempts"] == 0


def test_failed_launch_is_counted_failed_not_merely_reserved(module, tmp_path):
    ctl = controller(module, tmp_path)
    launch = ctl.output / "launch/run.json"
    overnight.durable_json(launch, {"state": "failed", "source_files_unchanged": True,
                                    "source_artifacts_unchanged": True, "plan": {}})
    ctl.journal.record("screen_reserved", current_screen={"ordinal": 1, "seed": 3, "profile": "S0",
                       "state": "reserved", "launch_journal": str(launch)}, state="blocked")
    result = ctl.write_summary()
    assert result["round_counts"]["failed_or_incomplete"] == 1
    assert result["round_counts"]["reserved_or_uncertain"] == 0


def test_retained_descendant_rejects_rewritten_official_rows_still_five_of_five(module, tmp_path, stable_evidence, worker_report_factory):
    from pathlib import Path
    gate, evidence = stable_evidence
    source = Path(evidence["candidate_path"])
    assessment = tmp_path / "assessment"
    for number in evidence["checkpoints"]:
        worker_report_factory(source.parent / f"model_{number}.pt", assessment / f"model-{number}", overnight.DEV)
    number = int(source.stem.removeprefix("model_"))
    official = source.parent / f"workshop/model_{number}.json"
    record = {"path": str(source), "sha256": overnight.digest(source), "profile": "S0",
              "development": {"finishes": 5, "total": 5, "mean_s": 3.},
              "official": overnight.official_summary(official, number)}
    ctl = controller(module, tmp_path)
    ctl.state["screens"] = [{"ordinal": 1, "profile": "S0", "state": "completed", "run": str(source.parent),
        "assessment": str(assessment), "expected_checkpoints": evidence["checkpoints"],
        "evaluated_checkpoints": evidence["checkpoints"], "pending_checkpoints": []}]
    ctl.journal.record("global_reliable_sources_updated", reliable_sources=[record], best_reliable=record)
    assert str(ctl.choose_source("S0")) == str(source)
    rewritten = overnight.read_report(official)
    for row in rewritten["per_seed"]:
        row["time_s"] += .1
    rewritten["score"] += .1
    overnight.durable_json(official, rewritten)
    assert overnight.official_summary(official, number)["finishes"] == 5
    with pytest.raises(ValueError, match="official|hash|changed"):
        ctl.choose_source("S0")


def test_real_prepare_refuses_expired_authorization_without_creating_reservation(module, tmp_path, local_auth, monkeypatch):
    auth = overnight.validate_authorization(local_auth)
    monkeypatch.setattr(module.time, "time", lambda: auth["training_deadline_epoch"])
    ctl = module.Controller(local_auth, tmp_path / "expired")
    with pytest.raises(TimeoutError, match="deadline"):
        ctl.prepare()
    assert not ctl.output.exists()
