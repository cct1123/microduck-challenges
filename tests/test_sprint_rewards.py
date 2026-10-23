"""Sprint reward semantics: real progress, bounded payouts, and no observation changes."""

import importlib
import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import microduck_challenges  # noqa: F401 (register tasks before upstream imports)


def _rewards():
    name = "microduck_challenges.sprint_2m.rewards"
    assert importlib.util.find_spec(name) is not None, "Sprint custom rewards are missing"
    return importlib.import_module(name)


def _env(velocities, commands=None, headings=None, gravity_z=None):
    velocity = torch.tensor(velocities, dtype=torch.float32)
    count = len(velocity)
    yaw = torch.zeros(count) if headings is None else torch.tensor(headings)
    zero = torch.zeros(count)
    quat = torch.stack([(yaw / 2).cos(), zero, zero, (yaw / 2).sin()], dim=-1)
    gravity = torch.zeros(count, 3)
    gravity[:, 2] = -1.0 if gravity_z is None else torch.tensor(gravity_z)
    command = torch.zeros(count, 3)
    command[:, 0] = 0.6 if commands is None else torch.tensor(commands)
    data = SimpleNamespace(
        root_link_lin_vel_w=velocity,
        root_link_quat_w=quat,
        projected_gravity_b=gravity,
        site_lin_vel_w=velocity.unsqueeze(1).repeat(1, 2, 1),
        joint_pos_biased=torch.zeros(count, 14),
        default_joint_pos=torch.zeros(count, 14),
    )
    return SimpleNamespace(
        scene={
            "robot": SimpleNamespace(data=data),
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(
                    current_air_time=torch.tensor([[0.2, 0.0]]).repeat(count, 1),
                    found=torch.tensor([[0.0, 1.0]]).repeat(count, 1),
                )
            ),
            "foot_height_scan": SimpleNamespace(data=SimpleNamespace(heights=torch.full((count, 2), 0.04))),
        },
        command_manager=SimpleNamespace(get_command=lambda name: command),
        episode_length_buf=torch.full((count,), 10),
        num_envs=count,
        _heading_ref=yaw.clone(),
        extras={"log": {}},
    )


def test_forward_progress_is_signed_and_capped_at_the_command():
    env = _env([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [1.2, 0, 0], [-0.3, 0, 0]])
    reward = _rewards().forward_progress(env)
    torch.testing.assert_close(reward, torch.tensor([0.0, 0.5, 1.0, 1.0, -0.5]))
    assert reward.shape == (5,)
    assert reward.dtype == torch.float32
    assert reward.device == env.scene["robot"].data.root_link_lin_vel_w.device


def test_aggressive_speed_gain_keeps_paying_beyond_the_nominal_command():
    env = _env([[1.2, 0, 0], [2.4, 0, 0], [3.0, 0, 0], [4.0, 0, 0]], commands=[1.2] * 4)
    reward = _rewards().forward_progress(env, speed_gain=2.5)
    torch.testing.assert_close(reward, torch.tensor([0.4, 0.8, 1.0, 1.0]))


@pytest.mark.parametrize("gain", [0.0, -1.0, float("inf"), float("nan")])
def test_speed_gain_must_be_finite_and_positive(gain):
    with pytest.raises(ValueError, match="speed_gain"):
        _rewards().forward_progress(_env([[0.6, 0, 0]]), speed_gain=gain)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, math.pi - 0.01])
def test_progress_uses_spawn_heading_even_after_the_robot_turns(yaw):
    env = _env([[0.6 * math.cos(yaw), 0.6 * math.sin(yaw), 0.0]], headings=[yaw])
    # Keep the spawn reference but turn the current body by 90 degrees.
    angle = yaw + math.pi / 2
    env.scene["robot"].data.root_link_quat_w = torch.tensor(
        [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]]
    )
    torch.testing.assert_close(_rewards().forward_progress(env), torch.ones(1))


def test_positive_progress_requires_upright_low_drift_motion():
    env = _env(
        [[0.6, 0.0, 0], [0.6, 0.2, 0], [0.6, 0.0, 0], [0.6, 0.0, 0], [-0.6, 0, 0]],
        gravity_z=[-1.0, -1.0, -0.75, 1.0, 1.0],
    )
    reward = _rewards().forward_progress(env, lateral_std=0.2)
    torch.testing.assert_close(reward, torch.tensor([1.0, math.exp(-1), 0.5, 0.0, -1.0]))


def test_progress_is_zero_for_idle_commands_and_nonfinite_states():
    env = _env(
        [[0.6, 0, 0], [0.6, 0, 0], [float("nan"), 0, 0], [float("inf"), 0, 0]],
        commands=[0.0, -0.6, 0.6, 0.6],
    )
    torch.testing.assert_close(_rewards().forward_progress(env), torch.zeros(4))


def test_air_time_only_pays_for_productive_stepping():
    env = _env(
        [[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [-0.6, 0, 0], [0.6, 0, 0], [0.6, 0.2, 0]],
        gravity_z=[-1, -1, -1, -1, 1, -1],
    )
    reward = _rewards().productive_air_time(env, sensor_name="feet_ground_contact")
    torch.testing.assert_close(reward, torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0, math.exp(-1)]))
    # Stepping in place cannot earn this payout; two feet airborne still cap at 2.
    env.scene["feet_ground_contact"].data.current_air_time[:] = 0.2
    reward = _rewards().productive_air_time(env, sensor_name="feet_ground_contact")
    assert reward[0] == 0
    assert reward[2] == 2


def test_supported_swing_rewards_relative_forward_foot_drive_not_flailing():
    env = _env([[0.6, 0, 0], [0, 0, 0], [-0.6, 0, 0], [0.6, 0, 0], [0.6, 0, 0], [0.6, 0, 0]])
    env.scene["robot"].data.site_lin_vel_w[:, 0, 0] += 2.0
    env.scene["feet_ground_contact"].data.found[3] = 0  # both feet flying
    env.scene["feet_ground_contact"].data.found[4] = 1  # both feet grounded
    env.scene["robot"].data.site_lin_vel_w[5, 0, 0] += 100  # cap, no jackpot
    reward = _rewards().swing_drive(env, sensor_name="feet_ground_contact", height_sensor_name="foot_height_scan")
    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_air_time_can_require_support_without_imposing_a_gait_clock():
    env = _env([[0.6, 0, 0], [0.6, 0, 0]])
    env.scene["feet_ground_contact"].data.current_air_time[:] = 0.2
    env.scene["feet_ground_contact"].data.found[1] = 0
    result = _rewards().productive_air_time(env, sensor_name="feet_ground_contact", require_support=True)
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0]))


def test_sprint_config_wires_the_new_rewards_without_changing_the_arena_contract():
    from microduck_challenges.sprint_2m.env import RL_CFG, make_env_cfg

    cfg = make_env_cfg()
    assert "forward_progress" in cfg.rewards
    assert cfg.rewards["forward_progress"].func is _rewards().forward_progress
    assert cfg.rewards["forward_progress"].weight == 0.5
    assert cfg.rewards["air_time"].func is _rewards().productive_air_time
    assert cfg.rewards["air_time"].params["lateral_std"] == 0.2
    assert cfg.rewards["heading_hold"].weight > 0
    assert "heading_error" not in cfg.observations["actor"].terms
    assert "heading_error" in cfg.observations["critic"].terms
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.ranges.lin_vel_x == (0.2, 1.2)
    assert RL_CFG.experiment_name == "sprint_2m"


@pytest.mark.parametrize("parameter", ["lateral_std", "command_threshold"])
@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_progress_rejects_invalid_scales(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        _rewards().forward_progress(_env([[0.6, 0, 0]]), **{parameter: value})


def test_reset_rows_ignore_stale_heading_without_mutating_the_shared_reference():
    env = _env([[0, 0.6, 0]] * 3, headings=[math.pi / 2] * 3)
    env._heading_ref[:] = 0.0  # reference from the previous episodes
    env.episode_length_buf = torch.tensor([0, 1, 10])
    before = env._heading_ref.clone()
    torch.testing.assert_close(_rewards().forward_progress(env), torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(env._heading_ref, before)


def test_progress_has_a_signal_at_the_top_command_before_speed_tracking_is_useful():
    env = _env([[0, 0, 0], [0.12, 0, 0]], commands=[1.2, 1.2])
    torch.testing.assert_close(_rewards().forward_progress(env), torch.tensor([0.0, 0.1]))


@pytest.mark.parametrize("air_time", [0.0, 0.125, 0.3, 0.5, float("nan")])
def test_productive_steps_preserve_the_stock_air_time_window(air_time):
    env = _env([[0.6, 0, 0]])
    env.scene["feet_ground_contact"].data.current_air_time[:] = air_time
    reward = _rewards().productive_air_time(env, sensor_name="feet_ground_contact")
    torch.testing.assert_close(reward, torch.zeros(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_complete_custom_reward_outputs_match_on_cpu_and_cuda():
    velocities = [[0, 0, 0], [0.3, 0.05, 0], [0.6, 0, 0], [-0.3, 0, 0], [1.2, 0.2, 0]]
    cpu = _env(velocities)
    gpu = _env(velocities)
    data = gpu.scene["robot"].data
    for name, value in vars(data).items():
        setattr(data, name, value.to("cuda:0"))
    gpu.scene["feet_ground_contact"].data.current_air_time = (
        gpu.scene["feet_ground_contact"].data.current_air_time.to("cuda:0")
    )
    gpu._heading_ref = gpu._heading_ref.to("cuda:0")
    gpu.episode_length_buf = gpu.episode_length_buf.to("cuda:0")
    command = torch.tensor([[0.6, 0.0, 0.0]] * len(velocities), device="cuda:0")
    gpu.command_manager = SimpleNamespace(get_command=lambda name: command)
    for function, kwargs in [
        (_rewards().forward_progress, {}),
        (_rewards().productive_air_time, {"sensor_name": "feet_ground_contact"}),
    ]:
        actual = function(gpu, **kwargs)
        expected = function(cpu, **kwargs)
        assert actual.device.type == "cuda"
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)
