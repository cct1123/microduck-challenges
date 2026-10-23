"""Sprint-specific reward shaping; the policy observation/action contract is unchanged."""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
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


# Finite race objective. The helpers above remain available for old checkpoints
# and optional potential guidance; they are not additive gait/style objectives.
DISTANCE_M = 2.0
TIME_LIMIT_S = 50.0
SETTLE_S = 0.5
FALL_GRAVITY_Z = -0.5
FALL_HOLD_S = 0.3


class RaceEpisode:
    """Arena-style first passage: 0.5 s settle, then 2 m along world +x.

    A sustained fall takes precedence over a simultaneous crossing. A finish
    on the 50 s deadline wins over timeout. The independent clock cannot be
    skipped by the generic runner's randomised initial episode_length_buf.
    """

    def __init__(self, cfg, env):
        del cfg
        self.settle_steps = round(SETTLE_S / env.step_dt)
        self.limit_steps = round(TIME_LIMIT_S / env.step_dt)
        self.fall_steps = round(FALL_HOLD_S / env.step_dt)
        self.start_x = torch.zeros(env.num_envs, device=env.device)
        self.progress = torch.zeros_like(self.start_x)
        self.elapsed = torch.zeros_like(self.start_x)
        self.steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.tipped_steps = torch.zeros_like(self.steps)
        self.finished = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.failed = torch.zeros_like(self.finished)
        self.timed_out = torch.zeros_like(self.finished)
        self.done = torch.zeros_like(self.finished)
        env._sprint_race = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (
            self.start_x, self.progress, self.elapsed, self.steps, self.tipped_steps,
            self.finished, self.failed, self.timed_out, self.done,
        ):
            value[ids] = 0

    def __call__(self, env):
        data = env.scene["robot"].data
        self.steps += 1
        timed = self.steps > self.settle_steps
        valid = (
            torch.isfinite(data.root_link_pos_w).all(dim=-1)
            & torch.isfinite(data.root_link_quat_w).all(dim=-1)
            & torch.isfinite(data.projected_gravity_b).all(dim=-1)
        )
        manager = getattr(env, "termination_manager", None)
        if manager is not None and "nan_state" in manager.active_terms:
            valid &= ~manager.get_term("nan_state")
        x = torch.nan_to_num(data.root_link_pos_w[:, 0])
        self.start_x.copy_(torch.where(self.steps <= self.settle_steps, x, self.start_x))
        self.progress.copy_(torch.where(timed, x - self.start_x, 0.0))
        self.elapsed.copy_((self.steps - self.settle_steps).clamp(0, self.limit_steps) * env.step_dt)
        tipped = timed & (data.projected_gravity_b[:, 2] > FALL_GRAVITY_Z)
        self.tipped_steps.copy_(torch.where(tipped, self.tipped_steps + 1, 0))
        self.failed.copy_(~valid | (self.tipped_steps >= self.fall_steps))
        self.finished.copy_(timed & (self.progress >= DISTANCE_M) & ~self.failed)
        self.timed_out.copy_(timed & (self.steps - self.settle_steps >= self.limit_steps) & ~self.finished)
        self.done.copy_(self.finished | self.failed | self.timed_out)
        return self.done


def race_time_cost(env):
    """Cost rate: with weight -1 and gamma=1, total return is -T or -50.

    Early failure pays the remaining seconds immediately, so falling early
    cannot avoid clock costs. mjlab multiplies this rate by step_dt.
    """
    race = env._sprint_race
    timed = race.steps > race.settle_steps
    remaining = (TIME_LIMIT_S - race.elapsed).clamp_min(0.0)
    return timed.float() + (race.failed | race.timed_out).float() * remaining / env.step_dt


class PotentialDifference:
    """Zero-net episode guidance: Phi(next)-Phi(previous), terminal Phi=0.

    Terminal clawback preserves the complete-episode race-time ranking. This
    does not guarantee that finite-rollout PPO escapes every local optimum.
    """

    def __init__(self, cfg, env):
        self.previous = torch.zeros(env.num_envs, device=env.device)
        self._potential = None
        if cfg is not None:
            params = dict(cfg.params)
            potential = params.pop("potential")
            # mjlab instantiates only the top-level term. A nested class-based
            # reward must receive its own cfg/env before it can be called.
            self._potential = (
                potential(cfg=RewardTermCfg(func=potential, weight=1.0, params=params), env=env)
                if inspect.isclass(potential) else potential
            )

    def reset(self, env_ids=None):
        self.previous[slice(None) if env_ids is None else env_ids] = 0.0
        if hasattr(self._potential, "reset"):
            self._potential.reset(env_ids=env_ids)

    def __call__(self, env, potential, **params):
        function = self._potential if self._potential is not None else potential
        current = torch.nan_to_num(function(env, **params), nan=0.0, posinf=0.0, neginf=0.0)
        current = torch.where(env._sprint_race.done, 0.0, current)
        rate = (current - self.previous) / env.step_dt
        self.previous.copy_(current)
        return rate


def progress_potential(env):
    """Position guidance without a speed cap, posture gate, or support gate."""
    return (env._sprint_race.progress / DISTANCE_M).clamp(-1.0, 1.0)


def race_air_time_potential(
    env, sensor_name="feet_ground_contact", threshold_min=0.06, threshold_max=0.22,
):
    """Bounded airborne-foot guidance, including flight; no posture/heading gate.

    Use only through PotentialDifference: holding or cycling an airborne pose
    cannot earn lasting episode return. Each foot contributes half the potential.
    """
    times = env.scene[sensor_name].data.current_air_time
    assert times is not None
    valid = torch.isfinite(times).all(dim=-1)
    eligible = (times > threshold_min) & (times < threshold_max)
    return torch.where(valid, eligible.float().mean(dim=-1), 0.0)


def race_state(env):
    """Privileged clock/progress/fall state for the critic, never the actor."""
    race = getattr(env, "_sprint_race", None)
    if race is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return torch.stack((
        race.elapsed / TIME_LIMIT_S,
        (race.progress / DISTANCE_M).clamp(-1.0, 1.0),
        race.tipped_steps / race.fall_steps,
        race.steps / (race.settle_steps + race.limit_steps),
    ), dim=-1)


def add_lane_walls(spec):
    """Arena wall geometry, shared across isolated Warp physics worlds."""
    lane = spec.worldbody.add_body(name="sprint_lane")
    for name, y in (("wall_left", 0.425), ("wall_right", -0.425)):
        lane.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(1.0, y, 0.075), size=(1.5, 0.025, 0.075),
            friction=(1.0, 0.005, 0.0001), contype=1, conaffinity=1,
        )


def configure_race(cfg, knobs):
    """Replace walker incentives while preserving robot and policy contracts.

    Knob IDs remain workshop-compatible but control zero-net potentials only.
    No lasting pose, head, effort, smoothing, joint-proximity, or swing tax
    remains. Physical actuator limits are unchanged; outputs are unclipped.
    """
    old_rewards = cfg.rewards
    cfg.rewards = {
        "race_time": RewardTermCfg(func=race_time_cost, weight=-1.0),
        "track_linear_velocity": RewardTermCfg(
            func=PotentialDifference, weight=knobs["forward_reward"],
            params={"potential": progress_potential},
        ),
        "air_time": RewardTermCfg(
            func=PotentialDifference, weight=knobs["step_reward"],
            params={"potential": race_air_time_potential,
                    "sensor_name": "feet_ground_contact",
                    "threshold_min": 0.06, "threshold_max": 0.22},
        ),
    }
    for name, weight in (
        ("upright", knobs["upright_reward"]),
        ("heading_hold", 0.0),  # free yaw: finish-line progress, not facing direction
    ):
        term = old_rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=PotentialDifference, weight=weight,
            params={"potential": term.func, **term.params},
        )
    cfg.terminations = {
        "nan_state": cfg.terminations["nan_state"],
        "race_complete": TerminationTermCfg(func=RaceEpisode, time_out=False),
    }
    cfg.episode_length_s = SETTLE_S + TIME_LIMIT_S
    cfg.is_finite_horizon = True
    cfg.scale_rewards_by_dt = True
    cfg.scene.env_spacing = 0.0
    cfg.scene.terrain.env_spacing = 0.0
    cfg.scene.spec_fn = add_lane_walls
    cfg.observations["critic"].terms["race_state"] = ObservationTermCfg(func=race_state)
    for name in ("action_rate_weight", "head_pose_bias_weight", "standing_envs"):
        cfg.curriculum.pop(name, None)
    for command in cfg.commands.values():
        command.resampling_time_range = (cfg.episode_length_s + 1.0,) * 2
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.12, 0.12),
        "yaw": (-math.radians(2.0), math.radians(2.0)),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
    cfg.actions["joint_pos"].clip = None
    cfg.actions["joint_pos"].scale = 1.0
    return cfg
