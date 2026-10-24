"""Active dense Sprint baseline and warm-started speed reward contracts."""

from dataclasses import asdict
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import microduck_challenges  # noqa: F401 (register tasks before upstream imports)
from microduck_challenges.sprint_2m import env as sprint_env
from microduck_challenges.sprint_2m import rewards
from mjlab_microduck import challenge
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_baseline_uses_dense_reference_rewards_without_changing_arena_commands():
    cfg = sprint_env.make_env_cfg(stage="baseline")
    knobs = challenge.params(sprint_env.__file__)
    speed = cfg.rewards["track_linear_velocity"]
    assert speed.func is mdp.track_linear_velocity
    assert speed.weight == knobs["forward_reward"]
    assert speed.params["std"] == pytest.approx(math.sqrt(0.1))
    assert cfg.rewards["upright"].func is mdp.upright
    assert cfg.rewards["upright"].weight == knobs["upright_reward"]
    assert cfg.rewards["air_time"].func is mdp.feet_air_time
    assert cfg.rewards["air_time"].weight == knobs["step_reward"]
    assert cfg.rewards["air_time"].params["threshold_min"] == 0.125
    assert cfg.rewards["air_time"].params["threshold_max"] == 0.3
    assert cfg.rewards["heading_hold"].func is microduck_mdp.heading_hold_reward
    assert cfg.rewards["heading_hold"].weight == 1.0
    assert cfg.rewards["heading_hold"].params["std"] == 0.4
    assert "heading_error" not in cfg.observations["actor"].terms
    assert "heading_error" in cfg.observations["critic"].terms
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (0.2, 1.2)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert sprint_env.RL_CFG.experiment_name == "sprint_2m"


def test_baseline_restores_standing_smoothing_and_locomotion_episodes():
    cfg = sprint_env.make_env_cfg(stage="baseline")
    assert cfg.rewards["pose"].weight == 1.0
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    smoothing = cfg.curriculum["action_rate_weight"]
    assert smoothing.func is microduck_mdp.reward_weight
    assert smoothing.params["reward_name"] == "action_rate_l2"
    assert smoothing.params["weight_stages"] == [
        {"step": iteration * sprint_env.NUM_STEPS_PER_ENV, "weight": weight}
        for iteration, weight in (
            (0, -0.1), (500, -0.2), (750, -0.4),
            (1000, -0.6), (1250, -0.8), (1500, -1.0),
        )
    ]
    assert cfg.commands["twist"].rel_standing_envs == 0.02
    standing = cfg.curriculum["standing_envs"]
    assert standing.func is microduck_mdp.standing_envs_curriculum
    assert standing.params["command_name"] == "twist"
    assert standing.params["standing_stages"][-1] == {
        "step": 2000 * sprint_env.NUM_STEPS_PER_ENV, "rel_standing_envs": 0.25,
    }
    assert "push_robot" in cfg.events
    assert cfg.episode_length_s == 20.0
    assert cfg.is_finite_horizon is False
    assert cfg.scale_rewards_by_dt
    assert cfg.actions["joint_pos"].clip is None
    assert cfg.actions["joint_pos"].scale == 1.0
    for name in ("race_time", "discovery_finish", "discovery_failure", "head_posture_cost"):
        assert name not in cfg.rewards
    assert "race_complete" not in cfg.terminations
    assert sprint_env.RL_CFG.algorithm.gamma == 0.99


def test_baseline_retains_zero_only_critic_padding_not_legacy_race_state():
    cfg = sprint_env.make_env_cfg(stage="baseline")
    assert "race_state" not in cfg.observations["actor"].terms
    term = cfg.observations["critic"].terms["race_state"]
    assert term.func is rewards.reference_critic_padding
    probe = SimpleNamespace(num_envs=3, device="cpu", _sprint_race=SimpleNamespace(
        progress=torch.tensor([0.0, 1.0, 2.0]), elapsed=torch.tensor([1.0, 2.0, 3.0]),
    ))
    torch.testing.assert_close(term.func(probe), torch.zeros(3, 4))


def test_speed_source_restores_0109_guidance_without_baseline_critic_padding():
    baseline = sprint_env.make_env_cfg(stage="baseline")
    source = sprint_env.make_env_cfg(stage="source")
    assert source.rewards["track_linear_velocity"].func is mdp.track_linear_velocity
    assert source.rewards["track_linear_velocity"].weight == 5.0
    assert source.rewards["upright"].weight == 1.0
    assert source.rewards["air_time"].func is rewards.productive_air_time
    assert source.rewards["air_time"].weight == 6.0
    assert source.rewards["air_time"].params["lateral_std"] == 0.2
    assert source.rewards["forward_progress"].func is rewards.forward_progress
    assert source.rewards["forward_progress"].weight == 0.5
    assert source.rewards["forward_progress"].params["lateral_std"] == 0.2
    assert list(source.observations["actor"].terms) == list(baseline.observations["actor"].terms)
    assert list(source.observations["critic"].terms) == [
        name for name in baseline.observations["critic"].terms if name != "race_state"
    ]
    assert source.actions == baseline.actions
    assert source.rewards["heading_hold"].weight == 1.0


def test_speed_starts_from_source_guidance_and_retains_it_during_the_hold(monkeypatch):
    source = sprint_env.make_env_cfg(stage="source")
    # Selection/compatibility validation has separate tests; model a selected
    # mature source checkpoint here without reading or loading training weights.
    monkeypatch.setattr(sprint_env, "SPEED_CHECKPOINT", {
        "checkpoint": 750, "source_env": asdict(source),
        "source_agent": asdict(sprint_env.RL_CFG),
    })
    cfg = sprint_env.make_env_cfg(stage="speed")
    assert cfg.rewards["track_linear_velocity"].weight == 5.0
    assert cfg.rewards["upright"].weight == 1.0
    assert cfg.rewards["air_time"].func is rewards.productive_air_time
    assert cfg.rewards["air_time"].weight == 6.0
    assert cfg.rewards["forward_progress"].func is rewards.forward_progress
    assert cfg.rewards["forward_progress"].weight == 0.5
    assert cfg.rewards["heading_hold"].weight == 1.0
    assert cfg.rewards["speed_forward"].func is microduck_mdp.forward_speed_reward
    assert cfg.rewards["speed_forward"].weight == 0.0
    assert cfg.rewards["speed_forward"].params["vel_ref"] == sprint_env.SPEED_SETTINGS["vel_ref"]
    assert cfg.commands["twist"].rel_standing_envs == 0.0
    assert "race_state" not in cfg.observations["critic"].terms
    assert "race_time" not in cfg.rewards
    for name in ("action_rate_weight", "head_pose_bias_weight", "standing_envs"):
        assert name not in cfg.curriculum
    assert cfg.rewards["action_rate_l2"].weight == -0.4
    ramp = cfg.curriculum["speed_rewards"]
    assert ramp.func is rewards.speed_reward_curriculum
    assert ramp.params["targets"]["track_linear_velocity"] == (5.0, 0.5)
    assert ramp.params["targets"]["speed_forward"] == (0.0, sprint_env.SPEED_SETTINGS["forward_reward"])
    assert ramp.params["hold_steps"] == sprint_env.SPEED_SETTINGS["hold_iterations"] * sprint_env.NUM_STEPS_PER_ENV
    assert ramp.params["ramp_steps"] == sprint_env.SPEED_SETTINGS["ramp_iterations"] * sprint_env.NUM_STEPS_PER_ENV
