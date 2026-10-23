"""Sprint-specific reward shaping; the policy observation/action contract is unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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
    gain > 1 explicitly treats the command as a sprint intensity, not a speed
    limit. This avoids penalising faster-than-command motion in a timed race.
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
    w, x, y, z = data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    reference = torch.where(
        env.episode_length_buf <= 1, yaw, getattr(env, "_heading_ref", yaw)
    )
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
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Keep the stock air-time window, but pay only for productive forward motion.

    Returns a nonnegative reward (positive weight). No new gait clock, foot
    ordering, support pattern, or policy input is imposed.
    """
    air_time = mdp.feet_air_time(
        env,
        sensor_name=sensor_name,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        command_name=command_name,
        command_threshold=command_threshold,
    )
    progress = forward_progress(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        lateral_std=lateral_std,
        asset_cfg=asset_cfg,
    ).clamp_min(0.0)
    return air_time * progress
