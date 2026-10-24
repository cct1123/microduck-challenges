"""Leg motion freedom must not change Arena hardware or head regularization."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import microduck_challenges  # noqa: F401
from mjlab.envs.mdp.actions import JointPositionActionCfg
from microduck_challenges.sprint_2m import env as sprint_env
from microduck_challenges.sprint_2m import rewards


HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    *HEAD_JOINTS,
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def test_default_refinement_removes_leg_pose_and_leg_smoothing():
    cfg = sprint_env.make_env_cfg(stage="refine")
    assert cfg.rewards["pose"].weight == 0.0
    assert cfg.curriculum["speed_rewards"].params["targets"]["pose"] == (0.0, 0.0)
    assert cfg.rewards["action_rate_l2"].func is rewards.HeadActionRateL2
    assert cfg.rewards["action_rate_l2"].weight == pytest.approx(-0.3)


def _action_env(names=JOINTS, width=None):
    width = len(names) if width is None else width
    term = SimpleNamespace(target_names=list(names))
    return SimpleNamespace(
        device="cpu",
        action_manager=SimpleNamespace(
            action=torch.zeros(2, width), prev_action=torch.zeros(2, width),
            active_terms=["joint_pos"], get_term=lambda name: term,
        ),
    )


def test_head_action_rate_ignores_every_leg_and_keeps_head_cost():
    env = _action_env()
    term = rewards.HeadActionRateL2(None, env)
    legs = [i for i, name in enumerate(JOINTS) if name not in HEAD_JOINTS]
    env.action_manager.action[:, legs] = 100.0
    torch.testing.assert_close(term(env), torch.zeros(2))
    env.action_manager.action[0, 5:9] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    env.action_manager.action[1, 5:9] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    torch.testing.assert_close(term(env), torch.full((2,), 30.0))
    env.action_manager.prev_action.copy_(env.action_manager.action)
    torch.testing.assert_close(term(env), torch.zeros(2))


def test_head_action_rate_resolves_names_instead_of_fixed_output_indices():
    names = tuple(reversed(JOINTS))
    env = _action_env(names)
    term = rewards.HeadActionRateL2(None, env)
    env.action_manager.action[:, names.index("head_yaw")] = 3.0
    torch.testing.assert_close(term(env), torch.full((2,), 9.0))


@pytest.mark.parametrize("names,width", [(JOINTS[:-6], None), (JOINTS, 15)])
def test_head_action_rate_rejects_incomplete_or_ambiguous_layout(names, width):
    with pytest.raises(ValueError, match="head|action|layout"):
        rewards.HeadActionRateL2(None, _action_env(names, width))


def test_leg_freedom_preserves_every_other_contract_and_cannot_reramp_pose():
    selected = sprint_env.REFINE_CHECKPOINT
    legacy_settings = {**sprint_env.REFINE_SETTINGS, "free_legs": False}
    legacy = rewards.configure_refine_locomotion(
        sprint_env.make_env_cfg(stage="baseline"), legacy_settings, selected,
    )
    cfg = sprint_env.make_env_cfg(stage="refine")
    before, after = rewards._config_data(legacy), rewards._config_data(cfg)
    for key in ("actions", "observations", "scene", "sim", "commands", "events", "terminations"):
        assert before[key] == after[key]
    for group in ("actor", "critic"):
        assert list(before["observations"][group]["terms"]) == list(after["observations"][group]["terms"])
    assert list(before["actions"]) == list(after["actions"])
    for name in cfg.rewards:
        if name not in ("pose", "action_rate_l2"):
            assert before["rewards"][name] == after["rewards"][name]
    terms = {name: SimpleNamespace(weight=cfg.rewards[name].weight)
             for name in cfg.curriculum["speed_rewards"].params["targets"]}
    probe = SimpleNamespace(common_step_counter=10**9, device="cpu",
                            reward_manager=SimpleNamespace(get_term_cfg=terms.__getitem__))
    rewards.speed_reward_curriculum(probe, None, **cfg.curriculum["speed_rewards"].params)
    assert terms["pose"].weight == 0.0
    assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
    assert cfg.actions["joint_pos"].scale == 1.0 and cfg.actions["joint_pos"].clip is None
    assert sprint_env.RL_CFG.clip_actions is None


def test_leg_free_descendant_is_validated_against_independent_known_recipe():
    selected = deepcopy(sprint_env.REFINE_CHECKPOINT)
    free = sprint_env.make_env_cfg(stage="refine")
    selected["source_env"] = rewards._config_data(free)
    # The existing source PT supplies a real inherited environment age.
    cfg = rewards.configure_refine_locomotion(
        sprint_env.make_env_cfg(stage="baseline"), sprint_env.REFINE_SETTINGS, selected,
    )
    assert cfg.rewards["pose"].weight == 0.0
    assert cfg.rewards["action_rate_l2"].func is rewards.HeadActionRateL2
    selected["source_env"]["rewards"]["pose"]["weight"] = 0.01
    with pytest.raises(ValueError, match="Invalid source reward weight"):
        rewards.configure_refine_locomotion(
            sprint_env.make_env_cfg(stage="baseline"), sprint_env.REFINE_SETTINGS, selected,
        )
