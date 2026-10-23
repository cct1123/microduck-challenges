"""Active Sprint race configuration and Arena policy contract."""

import importlib
import importlib.util
from types import SimpleNamespace

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import microduck_challenges  # noqa: F401 (register tasks before upstream imports)


def _rewards():
    name = "microduck_challenges.sprint_2m.rewards"
    assert importlib.util.find_spec(name) is not None, "Sprint custom rewards are missing"
    return importlib.import_module(name)


def test_sprint_config_wires_the_new_rewards_without_changing_the_arena_contract():
    from microduck_challenges.sprint_2m.env import RL_CFG, make_env_cfg

    cfg = make_env_cfg()
    assert cfg.rewards["track_linear_velocity"].func is _rewards().PotentialDifference
    assert cfg.rewards["track_linear_velocity"].params["potential"] is _rewards().progress_potential
    assert cfg.rewards["track_linear_velocity"].weight == 5.0
    assert cfg.rewards["air_time"].func is _rewards().PotentialDifference
    assert cfg.rewards["air_time"].params["potential"] is _rewards().race_air_time_potential
    assert cfg.rewards["air_time"].weight == 1.0
    assert cfg.rewards["upright"].weight == 0.1
    assert cfg.rewards["heading_hold"].weight == 0.0
    assert "heading_error" not in cfg.observations["actor"].terms
    assert "heading_error" in cfg.observations["critic"].terms
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (0.2, 1.2)
    assert RL_CFG.experiment_name == "sprint_2m"


def test_aggressive_recipe_replaces_overspeed_penalty_and_opens_stride_freedom():
    from microduck_challenges.sprint_2m.env import RL_CFG, make_env_cfg

    cfg = make_env_cfg()
    speed = cfg.rewards["track_linear_velocity"]
    assert speed.func is _rewards().PotentialDifference
    probe = SimpleNamespace(_sprint_race=SimpleNamespace(progress=torch.tensor([0.0, 0.5, 1.0])))
    values = speed.params["potential"](probe)
    torch.testing.assert_close(values, torch.tensor([0.0, 0.25, 0.5]))
    assert cfg.rewards["air_time"].params["threshold_min"] == 0.06
    assert cfg.rewards["race_time"].weight == -0.1
    assert RL_CFG.algorithm.gamma == 1.0
    for name in ("swing_drive", "head_forward", "pose", "action_rate_l2"):
        assert name not in cfg.rewards
    assert "action_rate_weight" not in cfg.curriculum
    assert "standing_envs" not in cfg.curriculum
    assert "push_robot" not in cfg.events
