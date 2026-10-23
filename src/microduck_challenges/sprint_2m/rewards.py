"""Sprint-specific reward shaping; the policy observation/action contract is unchanged."""

from __future__ import annotations

import inspect
import math

import mujoco
import torch
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
    """Cost rate: with weight -0.1 and gamma=1, return is -0.1*T or -5.

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


def head_posture_cost(env, asset_cfg):
    """Bounded sustained preference, not potential shaping or an action limit.

    Ordered pitch/yaw/roll encoder offsets match the actor's biased view.
    Negative pitch delta tips the real camera down: -pi/4 gives 45 degrees
    below horizontal with the trunk/neck at HOME, not during all movement.
    Broad tolerances and a small negative weight leave balance motion cheap.
    """
    data = env.scene[asset_cfg.name].data
    offsets = (data.joint_pos_biased[:, asset_cfg.joint_ids]
               - data.default_joint_pos[:, asset_cfg.joint_ids])
    target = offsets.new_tensor((-math.pi / 4, 0.0, 0.0))
    std = offsets.new_tensor((0.35, 0.2, 0.2))
    cost = 1.0 - torch.exp(-((offsets - target) / std).square().mean(dim=-1))
    return torch.where(torch.isfinite(offsets).all(dim=-1), cost, 0.0)


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
    Only a modest bounded head-posture cost adds a lasting style preference.
    No effort, smoothing, joint-proximity, or swing tax remains. Physical
    actuator limits are unchanged; outputs are unclipped.
    """
    old_rewards = cfg.rewards
    cfg.rewards = {
        "race_time": RewardTermCfg(func=race_time_cost, weight=-0.1),
        "head_posture_cost": RewardTermCfg(
            func=head_posture_cost, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=("head_pitch", "head_yaw", "head_roll"),
                preserve_order=True,
            )},
        ),
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
