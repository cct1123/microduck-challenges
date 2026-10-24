"""CPU contracts for reliable Sprint selection and warm-started speed tuning."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
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


def test_inference_restores_actor_without_loading_incompatible_critic():
    algorithm = cast(Any, object.__new__(rewards.SprintSpeedPPO))
    algorithm.actor = torch.nn.Linear(2, 1)
    algorithm.critic = torch.nn.Linear(3, 1)
    algorithm.rnd = None
    algorithm._sprint_inference_only = True
    original_critic = deepcopy(algorithm.critic.state_dict())
    actor_state = deepcopy(algorithm.actor.state_dict())
    actor_state["weight"].fill_(0.75)
    checkpoint = {"actor_state_dict": actor_state,
                  "critic_state_dict": torch.nn.Linear(4, 1).state_dict(),
                  "infos": {}, "iter": 750}
    assert algorithm.load(checkpoint) is False
    torch.testing.assert_close(algorithm.actor.weight, torch.full_like(algorithm.actor.weight, 0.75))
    for key, value in original_critic.items():
        torch.testing.assert_close(algorithm.critic.state_dict()[key], value)
    with pytest.raises(RuntimeError, match="inference-only"):
        algorithm.train_mode()


def test_training_still_rejects_incompatible_critic():
    algorithm = cast(Any, object.__new__(rewards.SprintSpeedPPO))
    algorithm.actor = torch.nn.Linear(2, 1)
    algorithm.critic = torch.nn.Linear(3, 1)
    algorithm.rnd = None
    checkpoint = {"actor_state_dict": algorithm.actor.state_dict(),
                  "critic_state_dict": torch.nn.Linear(4, 1).state_dict(),
                  "infos": {}, "iter": 750}
    with pytest.raises(RuntimeError, match="size mismatch"):
        algorithm.load(checkpoint)


def test_only_play_configs_mark_actor_only_inference(source_cfg):
    assert "actor_only_inference" not in source_cfg.events
    play = make_env_cfg(play=True, stage="source")
    marker = play.events["actor_only_inference"]
    assert marker.mode == "startup"
    probe = SimpleNamespace()
    marker.func(probe)
    assert probe._sprint_inference_only is True


def _drive_env(yaw=0.0, pitch=0.0, roll=0.0, speed=1.0):
    # Rz(yaw) Ry(pitch) Rx(roll); positive pitch tips the forward axis down.
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    data = SimpleNamespace(
        root_link_quat_w=torch.tensor([[cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                                       cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]]),
        projected_gravity_b=torch.tensor([[math.sin(pitch), -math.sin(roll)*math.cos(pitch),
                                           -math.cos(roll)*math.cos(pitch)]]),
        root_link_lin_vel_w=torch.tensor([[speed*math.cos(yaw), speed*math.sin(yaw), 0.0]]),
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                           episode_length_buf=torch.tensor([10]),
                           _heading_ref=torch.tensor([yaw]), common_step_counter=9600)


def test_course_speed_rewards_signed_spawn_direction_not_body_direction_or_command():
    env = _drive_env(yaw=1.2)
    expected = math.tanh(1 / 1.5)
    assert rewards.course_forward_speed(env).item() == pytest.approx(expected)
    # Turning the body must not rotate the course or pay for sideways motion.
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([[1.0, 0, 0, 0]])
    assert rewards.course_forward_speed(env).item() == pytest.approx(expected)
    env.scene["robot"].data.root_link_lin_vel_w *= -1
    assert rewards.course_forward_speed(env).item() == pytest.approx(-expected)
    env.scene["robot"].data.root_link_lin_vel_w.zero_()
    assert rewards.course_forward_speed(env).item() == 0.0


@pytest.mark.parametrize("bad", ["velocity", "heading", "quaternion", "fallen"])
def test_course_speed_is_finite_and_never_pays_invalid_or_fallen_states(bad):
    env = _drive_env()
    data = env.scene["robot"].data
    if bad == "velocity":
        data.root_link_lin_vel_w[0, 0] = float("nan")
    elif bad == "heading":
        env._heading_ref[:] = float("nan")
    elif bad == "quaternion":
        data.root_link_quat_w[:] = float("nan")
    else:
        data.projected_gravity_b[0, 2] = 0.0
    assert rewards.course_forward_speed(env).item() == 0.0


def test_drive_lean_relaxes_only_forward_pitch_while_moving_and_after_hold():
    angle = math.radians(25)
    env = _drive_env(pitch=angle)
    baseline = math.exp(-math.sin(angle)**2 / 0.05)
    relaxed = math.exp(-math.sin(angle)**2 / 0.30)
    params = dict(std=math.sqrt(0.05), forward_variance=0.30,
                  hold_steps=1200, ramp_steps=8400, motion_speed=0.5)
    for step, fraction in [(0, 0.0), (1200, 0.0), (5400, 0.5), (9600, 1.0)]:
        env.common_step_counter = step
        actual = rewards.drive_upright(env, **params).item()
        expected = math.exp(-math.sin(angle)**2 / (0.05 + fraction*0.25))
        assert actual == pytest.approx(expected)
    assert relaxed > 10*baseline
    for speed in [0.0, -1.0]:
        stationary = _drive_env(pitch=angle, speed=speed)
        assert rewards.drive_upright(stationary, **params).item() == pytest.approx(baseline)
    for other in [_drive_env(pitch=-angle), _drive_env(roll=angle)]:
        assert rewards.drive_upright(other, **params).item() == pytest.approx(baseline)


def test_drive_lean_never_rewards_a_lean_more_than_upright_or_pays_fallen_states():
    params = dict(std=math.sqrt(0.05), forward_variance=0.30,
                  hold_steps=1200, ramp_steps=8400, motion_speed=0.5)
    assert rewards.drive_upright(_drive_env(), **params).item() == 1.0
    for angle in [10, 25, 45]:
        assert 0 < rewards.drive_upright(_drive_env(pitch=math.radians(angle)), **params) < 1
    assert rewards.drive_upright(_drive_env(pitch=math.radians(70)), **params).item() == 0
    invalid = _drive_env()
    invalid.scene["robot"].data.projected_gravity_b[:] = float("nan")
    assert rewards.drive_upright(invalid, **params).item() == 0


@pytest.mark.parametrize("key,value", [("std", 0), ("forward_variance", float("nan")),
                                      ("ramp_steps", 0), ("hold_steps", -1), ("motion_speed", 0)])
def test_drive_lean_rejects_invalid_scales(key, value):
    params = dict(std=math.sqrt(0.05), forward_variance=0.30,
                  hold_steps=1200, ramp_steps=8400, motion_speed=0.5)
    params[key] = value
    with pytest.raises(ValueError):
        rewards.drive_upright(_drive_env(), **params)


def test_drive_stage_preserves_81d_source_safety_and_ramps_motion_freedom():
    cfg = make_env_cfg(stage="baseline")
    original = deepcopy(cfg)
    source = rewards._config_data(cfg)
    # Restore selected-run knobs, even if the user has since moved a UI slider.
    source["rewards"]["track_linear_velocity"]["weight"] = 4.8
    selected = {"source_env": source, "source_agent": {"num_steps_per_env": 24},
                "checkpoint": 2250}
    settings = {**SPEED_SETTINGS, "forward_variance": 0.30, "motion_speed": 0.5}
    cfg = rewards.configure_drive_locomotion(cfg, settings, selected)
    assert cfg.observations == original.observations
    assert cfg.actions == original.actions
    assert cfg.terminations == original.terminations
    assert cfg.scene.entities == original.scene.entities
    assert cfg.rewards["heading_hold"] == original.rewards["heading_hold"]
    assert cfg.rewards["self_collisions"] == original.rewards["self_collisions"]
    assert cfg.rewards["dof_pos_limits"] == original.rewards["dof_pos_limits"]
    assert cfg.rewards["air_time"] == original.rewards["air_time"]
    assert cfg.rewards["speed_forward"].func is rewards.course_forward_speed
    assert cfg.rewards["upright"].func is rewards.drive_upright
    assert cfg.rewards["track_linear_velocity"].weight == 4.8
    targets = cfg.curriculum["speed_rewards"].params["targets"]
    assert targets["action_rate_l2"] == (-1.0, -0.3)
    assert targets["pose"] == (1.0, 0.6)
    assert targets["head_pose_tracking"] == (2.0, 0.5)
    assert targets["head_pose_bias"] == (3.0, 1.0)
    assert targets["track_linear_velocity"] == pytest.approx((4.8, 0.48))
    assert targets["speed_forward"] == (0.0, 5.0)
    assert cfg.commands["twist"].rel_standing_envs == 0
    # The checkpoint clock restarts, but source physical randomization does not.
    assert cfg.curriculum["com_range"].params["range_stages"][-1]["step"] == -1


def test_drive_stage_refuses_a_source_with_different_reward_semantics():
    cfg = make_env_cfg(stage="baseline")
    source = rewards._config_data(cfg)
    source["rewards"]["air_time"]["func"] = "unknown.changed_reward"
    selected = {"source_env": source, "source_agent": {"num_steps_per_env": 24},
                "checkpoint": 2250}
    with pytest.raises(ValueError, match="source reward"):
        rewards.configure_drive_locomotion(
            cfg, {**SPEED_SETTINGS, "forward_variance": 0.30, "motion_speed": 0.5}, selected,
        )


def test_drive_factory_uses_baseline_layout_and_refuses_a_cold_start(monkeypatch):
    from microduck_challenges.sprint_2m import env as sprint_env

    source = make_env_cfg(stage="baseline")
    selected = {"source_env": rewards._config_data(source),
                "source_agent": {"num_steps_per_env": 24}, "checkpoint": 2250}
    monkeypatch.setattr(sprint_env, "DRIVE_CHECKPOINT", selected, raising=False)
    monkeypatch.setattr(sprint_env, "DRIVE_SETTINGS",
                        {**SPEED_SETTINGS, "forward_variance": 0.30, "motion_speed": 0.5},
                        raising=False)
    cfg = sprint_env.make_env_cfg(stage="drive")
    assert cfg.observations == source.observations
    assert cfg.rewards["speed_forward"].func is rewards.course_forward_speed
    assert "actor_only_inference" not in cfg.events
    assert "actor_only_inference" in sprint_env.make_env_cfg(play=True, stage="drive").events
    monkeypatch.setattr(sprint_env, "DRIVE_CHECKPOINT", None)
    with pytest.raises(RuntimeError, match="compatible 5/5"):
        sprint_env.make_env_cfg(stage="drive")
