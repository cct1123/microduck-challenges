"""Sprint-specific reward shaping; the policy observation/action contract is unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FOOT_CFG = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot"), preserve_order=True
)
_DEFAULT_HEAD_CFG = SceneEntityCfg(
    "robot", joint_names=("head_yaw", "head_roll"), preserve_order=True
)


def _spawn_heading(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Read heading_hold's reference without changing its reset lifecycle."""
    w, x, y, z = env.scene[asset_cfg.name].data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.where(env.episode_length_buf <= 1, yaw, getattr(env, "_heading_ref", yaw))


def _valid_physics(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reject numerical-invalid steps, not ordinary falls or timeouts.

    The live termination manager has already checked all physical fields and
    configured contact forces before rewards run. Reuse that result. Outside
    the managed step, check the available physical state tensors directly.
    """
    manager = getattr(env, "termination_manager", None)
    if manager is not None and "nan_state" in manager.active_terms:
        return ~manager.get_term("nan_state")
    data = env.scene[asset_cfg.name].data
    valid = torch.ones_like(data.root_link_lin_vel_w[:, 0], dtype=torch.bool)
    for name in (
        "joint_pos", "joint_vel", "root_link_pos_w", "root_link_quat_w",
        "root_link_lin_vel_w", "root_link_ang_vel_w",
    ):
        value = getattr(data, name, None)
        if value is not None:
            valid &= torch.isfinite(value).all(dim=-1)
    return valid


def forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    lateral_std: float = 0.2,
    speed_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bounded spawn-frame progress, with upright/low-drift positive payouts.

    Standing still pays zero. Reverse motion stays negative even when fallen;
    only positive progress is gated. Speed is capped at speed_gain * command;
    gain > 1 treats the command as sprint intensity, not a speed limit. This
    avoids penalising faster-than-command motion in a timed race.
    The retained heading_hold term owns _heading_ref. On the first step after
    reset use current yaw, matching its capture rule regardless of term order.
    This is training-only privileged state, not an additional actor observation.
    """
    if not math.isfinite(lateral_std) or lateral_std <= 0.0:
        raise ValueError("lateral_std must be finite and positive")
    if not math.isfinite(command_threshold) or command_threshold <= 0.0:
        raise ValueError("command_threshold must be finite and positive")
    if not math.isfinite(speed_gain) or speed_gain <= 0.0:
        raise ValueError("speed_gain must be finite and positive")
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    data = env.scene[asset_cfg.name].data
    reference = _spawn_heading(env, asset_cfg)
    velocity = data.root_link_lin_vel_w
    speed = reference.cos() * velocity[:, 0] + reference.sin() * velocity[:, 1]
    sideways = -reference.sin() * velocity[:, 0] + reference.cos() * velocity[:, 1]
    target = command[:, 0]
    progress = (speed / (speed_gain * target.clamp_min(command_threshold))).clamp(-1.0, 1.0)
    # Zero positive payout at >=60 degrees tilt, including upside-down poses.
    upright = ((-data.projected_gravity_b[:, 2] - 0.5) / 0.5).clamp(0.0, 1.0)
    quality = upright * torch.exp(-torch.square(sideways / lateral_std))
    reward = torch.where(progress > 0.0, progress * quality, progress)
    valid = (
        (target > command_threshold)
        & torch.isfinite(command).all(dim=-1)
        & torch.isfinite(velocity).all(dim=-1)
        & torch.isfinite(data.root_link_quat_w).all(dim=-1)
        & torch.isfinite(data.projected_gravity_b).all(dim=-1)
        & torch.isfinite(reference)
        & _valid_physics(env, asset_cfg)
    )
    return torch.where(valid, reward, 0.0)


def productive_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.125,
    threshold_max: float = 0.3,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    lateral_std: float = 0.2,
    require_support: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward productive stepping, optionally requiring another foot's support.

    Returns a nonnegative reward (positive weight). No gait clock, foot
    ordering, or policy input is imposed. Flight phases remain possible but
    earn no air-time bonus when require_support is enabled.
    """
    data = env.scene[sensor_name].data
    assert data.current_air_time is not None
    valid = torch.isfinite(data.current_air_time).all(dim=-1) & _valid_physics(env, asset_cfg)
    if require_support:
        assert data.found is not None
        valid &= torch.isfinite(data.found).all(dim=-1)

    # Reject the entire sensor row before either reward or logging reductions.
    # NaN * 0 is still NaN, so multiplying a late gate is not sufficient.
    safe_time = torch.where(valid.unsqueeze(-1), data.current_air_time, 0.0)
    eligible = (safe_time > threshold_min) & (safe_time < threshold_max)
    in_air = safe_time > 0.0
    env.extras["log"]["Metrics/air_time_mean"] = (
        (safe_time * in_air).sum() / in_air.float().sum().clamp_min(1.0)
    )
    if require_support:
        contact = data.found > 0
        eligible &= ~contact
        air_time = eligible.float().sum(dim=-1) * contact.any(dim=-1) * valid
    else:
        air_time = eligible.float().sum(dim=-1) * valid
    progress = forward_progress(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        lateral_std=lateral_std,
        asset_cfg=asset_cfg,
    ).clamp_min(0.0)
    return air_time * progress


def swing_drive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    swing_speed: float = 2.0,
    clearance: float = 0.04,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    lateral_std: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_CFG,
) -> torch.Tensor:
    """Bounded forward swing drive, paid only while another foot supports progress.

    Foot speed is relative to the trunk: translating a frozen pose does not
    count as a swing. Stance sliding, backward swings, and unsupported flailing
    earn zero. No external clock or fixed phase is supplied to the policy.
    """
    for name, value in (("swing_speed", swing_speed), ("clearance", clearance)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    data = env.scene[asset_cfg.name].data
    feet = data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    relative = feet - data.root_link_lin_vel_w.unsqueeze(1)
    heading = _spawn_heading(env, asset_cfg).unsqueeze(-1)
    forward = relative[..., 0] * heading.cos() + relative[..., 1] * heading.sin()
    found = env.scene[sensor_name].data.found
    assert found is not None
    heights = env.scene[height_sensor_name].data.heights
    contact = found > 0
    swing = (forward / swing_speed).clamp(0.0, 1.0)
    swing = swing * (heights / clearance).clamp(0.0, 1.0) * (~contact)
    drive = swing.sum(dim=-1).clamp_max(1.0) * contact.any(dim=-1)
    progress = forward_progress(
        env, command_name=command_name, command_threshold=command_threshold,
        lateral_std=lateral_std, asset_cfg=asset_cfg,
    ).clamp_min(0.0)
    valid = (
        torch.isfinite(feet).all(dim=(-1, -2))
        & torch.isfinite(heights).all(dim=-1)
        & torch.isfinite(found).all(dim=-1)
        & torch.isfinite(drive)
    )
    return torch.where(valid, drive * progress, 0.0)


def head_forward_cost(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    deadband: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_HEAD_CFG,
) -> torch.Tensor:
    """Nonnegative yaw/roll misalignment cost; pair with a negative weight.

    Respect the head command and encoder view used by the actor. Do not price
    pitch oscillation here: it is required for a heavy head during fast strides.
    Body heading is independently held by heading_hold.
    """
    if not math.isfinite(deadband) or deadband < 0.0:
        raise ValueError("deadband must be finite and nonnegative")
    data = env.scene[asset_cfg.name].data
    actual = data.joint_pos_biased[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    error = actual - command[:, [2, 3]]
    error = torch.atan2(error.sin(), error.cos())
    cost = (error.abs() - deadband).clamp_min(0.0).square().mean(dim=-1).clamp_max(1.0)
    valid = (
        torch.isfinite(actual).all(dim=-1)
        & torch.isfinite(command).all(dim=-1)
        & _valid_physics(env, asset_cfg)
    )
    return torch.where(valid, cost, 0.0)
