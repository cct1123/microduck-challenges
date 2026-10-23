"""Small sustained head preference, with the real camera's optical convention."""

import math
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import microduck_challenges  # noqa: F401 (register before upstream imports)
from microduck_challenges.sprint_2m.env import make_env_cfg
from microduck_challenges.sprint_2m import rewards


def test_head_cost_is_small_direct_cost_with_ordered_named_selector():
    cfg = make_env_cfg()
    assert "head_posture_cost" in cfg.rewards
    term = cfg.rewards["head_posture_cost"]
    assert term.func is rewards.head_posture_cost
    assert term.weight == -0.05
    selector = term.params["asset_cfg"]
    assert selector.name == "robot"
    assert selector.joint_names == ("head_pitch", "head_yaw", "head_roll")
    assert selector.preserve_order is True
    assert cfg.scale_rewards_by_dt
    assert cfg.actions["joint_pos"].clip is None
    assert cfg.actions["joint_pos"].scale == 1.0


def _probe(offsets):
    # Deliberately nonstandard, noncontiguous pitch/yaw/roll IDs; other joints
    # (including the neck) have large errors and must not enter this cost.
    ids = [8, 2, 5]
    default = torch.full((len(offsets), 12), 0.3491)
    biased = default + 9.0
    biased[:, ids] = default[:, ids] + default.new_tensor(offsets)
    data = SimpleNamespace(joint_pos_biased=biased, default_joint_pos=default,
                           joint_pos=torch.zeros_like(default))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    selector = SimpleNamespace(name="robot", joint_ids=ids)
    return env, selector


def test_target_is_free_and_each_head_error_has_a_bounded_sustained_cost():
    target = -math.pi / 4
    env, selector = _probe([
        [target, 0, 0], [target + 0.35, 0, 0],
        [target, 0.2, 0], [target, 0, -0.2], [1000, 1000, 1000],
    ])
    assert hasattr(rewards, "head_posture_cost")
    cost = rewards.head_posture_cost(env, asset_cfg=selector)
    torch.testing.assert_close(cost, torch.tensor([0.0, 1-math.exp(-1/3),
        1-math.exp(-1/3), 1-math.exp(-1/3), 1.0]))
    torch.testing.assert_close(rewards.head_posture_cost(env, asset_cfg=selector), cost)
    assert torch.all((cost >= 0) & (cost <= 1))
    assert torch.all((-0.05 * cost >= -0.05) & (-0.05 * cost <= 0))


def test_nonfinite_head_rows_are_zero_without_hiding_valid_costs():
    env, selector = _probe([[float("nan"), 0, 0], [0, float("inf"), 0],
                            [0, 0, -float("inf")], [0, 0, 0]])
    cost = rewards.head_posture_cost(env, asset_cfg=selector)
    assert torch.isfinite(cost).all()
    assert torch.equal(cost[:3], torch.zeros(3))
    assert cost[3] > 0


def test_real_camera_target_points_45_degrees_down_with_home_neck():
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG

    cfg = MICRODUCK_WALK_ROBOT_CFG
    model = cfg.spec_fn().compile()
    data = mujoco.MjData(model)
    for index in range(model.njnt):
        name = model.joint(index).name
        for pattern, value in cfg.init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[index]] = value
    mujoco.mj_forward(model, data)
    camera_id = model.camera("head_camera").id
    np.testing.assert_allclose(-data.cam_xmat[camera_id].reshape(3, 3)[:, 2],
                               [-1, 0, 0], atol=1e-6)
    joint = model.joint("head_pitch")
    home = data.qpos[joint.qposadr[0]]
    target = home - math.pi / 4
    assert joint.range[0] < target < joint.range[1]
    # Use the configured term's zero-cost pose, not a camera site frame.
    env, selector = _probe([[target - home, 0, 0]])
    assert rewards.head_posture_cost(env, asset_cfg=selector).item() == pytest.approx(0)
    data.qpos[joint.qposadr[0]] = target
    mujoco.mj_forward(model, data)
    gaze = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(gaze, [-math.sqrt(0.5), 0, -math.sqrt(0.5)], atol=1e-6)
    print(f"home={home:.7f}, target={target:.7f}, limits={joint.range}, gaze={gaze}")
