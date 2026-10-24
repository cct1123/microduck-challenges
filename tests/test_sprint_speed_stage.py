"""CPU contracts for reliable Sprint selection and warm-started speed tuning."""

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml

import microduck_challenges  # noqa: F401
from microduck_challenges.sprint_2m import rewards
from microduck_challenges.sprint_2m.env import RL_CFG, SPEED_SETTINGS, make_env_cfg


@pytest.fixture
def source_cfg():
    return make_env_cfg(stage="source")


def _run(root, name, cfg):
    run = root / name
    (run / "params").mkdir(parents=True)
    (run / "workshop").mkdir()
    (run / "params/env.yaml").write_text(yaml.dump(asdict(cfg), sort_keys=False))
    (run / "params/agent.yaml").write_text(yaml.dump(asdict(RL_CFG), sort_keys=False))
    (run / "provenance.json").write_text(json.dumps({"task": "Mjlab-Sprint2m-MicroDuck"}))
    return run


def _result(run, checkpoint, times, statuses=None, **overrides):
    """Write synthetic unit-test metadata; the placeholder .pt is never loaded."""
    statuses = statuses or ["finish"] * 5
    result = {
        "checkpoint": checkpoint,
        "score": sum(times) / 5,
        "seeds_finished": statuses.count("finish"),
        "seeds_total": 5,
        "command_vx": 0.9,
        "per_seed": [{"seed": seed, "status": status, "time_s": time}
                     for seed, status, time in zip(range(1, 6), statuses, times, strict=True)],
        **overrides,
    }
    (run / f"model_{checkpoint}.pt").write_bytes(b"unit-test placeholder, not a model")
    (run / "workshop" / f"model_{checkpoint}.json").write_text(json.dumps(result))


def test_selection_requires_five_finishes_then_minimizes_time_not_iteration(tmp_path, source_cfg):
    run = _run(tmp_path, "baseline", source_cfg)
    _result(run, 750, [20.0] * 5)
    _result(run, 2000, [21.0] * 5)
    _result(run, 2500, [1.0] * 4 + [50.0], ["finish"] * 4 + ["fall"])
    best = rewards.select_sprint_checkpoint(
        tmp_path, expected_env=source_cfg, expected_agent=RL_CFG,
    )
    assert best["checkpoint"] == 750
    assert best["score"] == 20.0


@pytest.mark.parametrize("group", ["actor", "critic"])
def test_selection_rejects_reordered_features_even_if_definitions_match(tmp_path, source_cfg, group):
    reordered = deepcopy(source_cfg)
    reordered.observations[group].terms = dict(reversed(list(reordered.observations[group].terms.items())))
    run = _run(tmp_path, "reordered", reordered)
    _result(run, 750, [2.0] * 5)
    with pytest.raises(RuntimeError, match=f"incompatible {group} observation order"):
        rewards.select_sprint_checkpoint(tmp_path, expected_env=source_cfg, expected_agent=RL_CFG)


def test_selection_rejects_padded_critic_for_unpadded_source(tmp_path, source_cfg):
    run = _run(tmp_path, "padded", make_env_cfg(stage="baseline"))
    _result(run, 750, [2.0] * 5)
    with pytest.raises(RuntimeError, match="critic observation order"):
        rewards.select_sprint_checkpoint(tmp_path, expected_env=source_cfg, expected_agent=RL_CFG)


@pytest.mark.parametrize("overrides", [
    {"score": 0.5}, {"score": float("nan")}, {"error": "export failed"},
    {"seeds_finished": 4}, {"command_vx": 2.0},
    {"per_seed": [{"seed": 1, "status": "finish", "time_s": 2.0}] * 5},
])
def test_selection_refuses_incomplete_or_invalid_evaluation(tmp_path, source_cfg, overrides):
    run = _run(tmp_path, "invalid", source_cfg)
    _result(run, 750, [2.0] * 5, **overrides)
    with pytest.raises(RuntimeError, match="Refusing a cold start"):
        rewards.select_sprint_checkpoint(tmp_path, expected_env=source_cfg, expected_agent=RL_CFG)


def test_selection_refuses_missing_checkpoint(tmp_path, source_cfg):
    run = _run(tmp_path, "missing", source_cfg)
    _result(run, 750, [2.0] * 5)
    (run / "model_750.pt").unlink()
    with pytest.raises(RuntimeError, match="missing/empty training checkpoint"):
        rewards.select_sprint_checkpoint(tmp_path, expected_env=source_cfg, expected_agent=RL_CFG)


def test_malformed_saved_config_does_not_hide_valid_candidate(tmp_path, source_cfg):
    malformed = _run(tmp_path, "a-malformed", source_cfg)
    doc = rewards._config_data(source_cfg)
    doc["observations"]["critic"]["terms"] = None
    (malformed / "params/env.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    valid = _run(tmp_path, "b-valid", source_cfg)
    _result(valid, 750, [2.0] * 5)
    assert rewards.select_sprint_checkpoint(
        tmp_path, expected_env=source_cfg, expected_agent=RL_CFG,
    )["checkpoint"] == 750


def test_source_profile_preserves_guidance_and_original_critic(source_cfg):
    assert "race_state" not in source_cfg.observations["critic"].terms
    assert "heading_error" in source_cfg.observations["critic"].terms
    assert "heading_error" not in source_cfg.observations["actor"].terms
    assert source_cfg.rewards["upright"].weight == 1.0
    assert source_cfg.rewards["heading_hold"].weight == 1.0
    assert source_cfg.rewards["air_time"].func is rewards.productive_air_time
    assert source_cfg.rewards["air_time"].weight == 6.0
    assert source_cfg.rewards["forward_progress"].func is rewards.forward_progress
    assert source_cfg.rewards["forward_progress"].weight == 0.5
    assert source_cfg.actions["joint_pos"].scale == 1.0
    assert source_cfg.actions["joint_pos"].clip is None


def test_speed_stage_uses_selected_checkpoint_regularization_before_ramp(source_cfg):
    selected = {"source_env": rewards._config_data(source_cfg),
                "source_agent": {"num_steps_per_env": 24}, "checkpoint": 750}
    cfg = rewards.configure_speed_locomotion(source_cfg, SPEED_SETTINGS, selected)
    params = cfg.curriculum["speed_rewards"].params
    assert params["targets"]["action_rate_l2"] == (-0.4, -0.3)
    assert params["targets"]["track_linear_velocity"] == (5.0, 0.5)
    assert params["targets"]["speed_forward"] == (0.0, 5.0)
    assert params["hold_steps"] == 1200
    assert params["ramp_steps"] == 8400
    assert "standing_envs" not in cfg.curriculum
    assert "action_rate_weight" not in cfg.curriculum
    assert cfg.commands["twist"].rel_standing_envs == 0.0
    assert cfg.rewards["upright"].weight == 1.0
    assert cfg.rewards["heading_hold"].weight == 1.0
    assert cfg.rewards["air_time"].weight == 6.0
    assert "race_state" not in cfg.observations["critic"].terms


@pytest.mark.parametrize("step,fraction", [(0, 0.0), (1200, 0.0), (5400, 0.5), (9600, 1.0), (12000, 1.0)])
def test_curriculum_blends_live_weights_without_exceeding_targets(step, fraction):
    terms = {"speed_forward": SimpleNamespace(weight=0.0),
             "action_rate_l2": SimpleNamespace(weight=-0.4)}
    env = SimpleNamespace(common_step_counter=step, device="cpu",
                          reward_manager=SimpleNamespace(get_term_cfg=terms.__getitem__))
    value = rewards.speed_reward_curriculum(
        env, None, {"speed_forward": (0.0, 5.0), "action_rate_l2": (-0.4, -0.3)},
        hold_steps=1200, ramp_steps=8400,
    )
    assert value.item() == pytest.approx(fraction)
    assert terms["speed_forward"].weight == pytest.approx(5.0 * fraction)
    assert terms["action_rate_l2"].weight == pytest.approx(-0.4 + 0.1 * fraction)


def test_warm_start_keeps_weights_and_normalizers_but_not_optimizer_or_clocks():
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    actor.register_buffer("normalizer_mean", torch.zeros(2))
    critic.register_buffer("normalizer_mean", torch.zeros(2))
    algorithm = cast(Any, object.__new__(rewards.SprintSpeedPPO))
    algorithm.actor, algorithm.critic, algorithm.rnd = actor, critic, None
    algorithm.optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-4)
    actor_state, critic_state = deepcopy(actor.state_dict()), deepcopy(critic.state_dict())
    actor_state["normalizer_mean"].fill_(3.0)
    critic_state["normalizer_mean"].fill_(4.0)
    actor_state["weight"].fill_(0.5)
    payload = {"actor_state_dict": actor_state, "critic_state_dict": critic_state,
               "optimizer_state_dict": {"must_not_be_loaded": True}, "iter": 750,
               "infos": {"env_state": {"common_step_counter": 18024}}}
    assert algorithm.load(payload) is False
    torch.testing.assert_close(actor.weight, torch.full_like(actor.weight, 0.5))
    torch.testing.assert_close(actor.normalizer_mean, torch.full((2,), 3.0))
    torch.testing.assert_close(critic.normalizer_mean, torch.full((2,), 4.0))
    assert algorithm.optimizer.param_groups[0]["lr"] == 1e-4
    assert not algorithm.optimizer.state
    assert payload["infos"]["env_state"]["common_step_counter"] == 0
    algorithm.train_mode()
    assert actor.training and critic.training


def test_speed_algorithm_refuses_random_policy_training():
    algorithm = object.__new__(rewards.SprintSpeedPPO)
    with pytest.raises(RuntimeError, match="requires a loaded locomotion checkpoint"):
        algorithm.train_mode()
