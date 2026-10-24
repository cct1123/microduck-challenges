"""Reference locomotion configuration and legacy Sprint checkpoint helpers.

Only configure_reference_locomotion is selected by the current env factory.
The former race functions remain importable for saved experiment configurations;
their clock, mastery gate and terminal costs are not active in the new recipe.
"""

from __future__ import annotations

import inspect
import math

import mujoco
import torch
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab_microduck.tasks import mdp as microduck_mdp

DISTANCE_M = 2.0
TIME_LIMIT_S = 50.0
SETTLE_S = 0.5
FALL_GRAVITY_Z = -0.5
FALL_HOLD_S = 0.3
TIME_COST_WEIGHT = 0.1
MASTERY_FINISH_RATE = 0.8
MASTERY_MIN_EPISODES = 32
COMMAND_RESAMPLE_S = (SETTLE_S + TIME_LIMIT_S + 1.0,) * 2


def reference_critic_padding(env):
    """Keep four legacy critic-only slots; never add these to the actor."""
    return torch.zeros(env.num_envs, 4, device=env.device)


def configure_reference_locomotion(cfg, knobs, rollout_steps=24):
    """Port electricshaka/MicroDuck's Sprint recipe at da146b3944dc80d894f09781455d827925ab07a0.

    Its Sprint settings match the Pollen reference: dense balance/velocity/gait
    rewards, gentle-to-strong smoothing and standing practice. Keep our existing
    compute budget and four critic padding slots; this is not an exact run
    reproduction or a promise of a particular Arena score.
    """
    standing = {
        r".*hip_yaw.*": 0.1, r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.15, r".*knee.*": 0.15, r".*ankle.*": 0.1,
    }
    walking = {
        r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.4, r".*knee.*": 0.4, r".*ankle.*": 0.25,
    }
    cfg.rewards["pose"].params.update(
        std_standing=standing, std_walking=walking, std_running=walking,
        walking_threshold=0.01,
        asset_cfg=SceneEntityCfg(
            "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",),
        ),
    )
    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["upright"].weight = knobs["upright_reward"]
    cfg.rewards["upright"].params.update(
        std=math.sqrt(0.05),
        asset_cfg=SceneEntityCfg("robot", body_names=("trunk_base",)),
    )
    cfg.rewards["track_linear_velocity"].weight = knobs["forward_reward"]
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward, weight=1.0,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["air_time"].weight = knobs["step_reward"]
    cfg.rewards["air_time"].params.update(
        command_threshold=0.01, threshold_min=0.125, threshold_max=0.300,
    )
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = ["left_foot", "right_foot"]
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params.update(command_threshold=0.01, target_height=0.02)
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking, weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d, weight=0.0,
        params={
            "command_name": "body_pose", "nominal_height": 0.095,
            "xy_std": 0.05, "z_std": 0.02, "angle_std": math.radians(15),
        },
    )
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty, weight=0.0,
        params={"command_name": "head_pose", "tau_s": 1.0},
    )
    for term, reward_name, stages in (
        ("action_rate_weight", "action_rate_l2", (
            (0, -0.1), (500, -0.2), (750, -0.4),
            (1000, -0.6), (1250, -0.8), (1500, -1.0),
        )),
        ("head_pose_bias_weight", "head_pose_bias", (
            (0, 0.0), (600, 1.0), (1000, 2.0), (1500, 3.0),
        )),
    ):
        cfg.curriculum[term] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": reward_name,
                "weight_stages": [
                    {"step": iteration * rollout_steps, "weight": weight}
                    for iteration, weight in stages
                ],
            },
        )
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": iteration * rollout_steps, "rel_standing_envs": fraction}
                for iteration, fraction in (
                    (0, 0.02), (500, 0.05), (750, 0.1),
                    (1000, 0.15), (1500, 0.2), (2000, 0.25),
                )
            ],
        },
    )
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)
    cfg.episode_length_s = 20.0
    cfg.is_finite_horizon = False
    cfg.actions["joint_pos"].clip = None
    cfg.observations["critic"].terms["race_state"] = ObservationTermCfg(
        func=reference_critic_padding,
    )
    return cfg


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
        self.spawn_heading = torch.zeros_like(self.start_x)
        self.progress = torch.zeros_like(self.start_x)
        self.elapsed = torch.zeros_like(self.start_x)
        self.steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.tipped_steps = torch.zeros_like(self.steps)
        self.finished = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.failed = torch.zeros_like(self.finished)
        self.timed_out = torch.zeros_like(self.finished)
        self.done = torch.zeros_like(self.finished)
        self.invalid = torch.zeros_like(self.finished)
        # Distance-first discovery; the latest outcome per world is retained
        # across resets. Once established, time optimisation stays enabled.
        self.last_finished = torch.zeros_like(self.finished)
        self.completed = torch.zeros((), device=env.device, dtype=torch.long)
        self.mastered = torch.zeros((), device=env.device, dtype=torch.bool)
        self.time_scale = torch.zeros_like(self.start_x)
        env._sprint_race = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        completed = self.done[ids]
        self.last_finished[ids] = torch.where(
            completed, self.finished[ids], self.last_finished[ids],
        )
        self.completed += completed.sum()
        # Unfinished/unobserved worlds count as non-finishes, so a handful of
        # fast successful worlds cannot prematurely certify the whole batch.
        finish_rate = self.last_finished.float().mean()
        self.mastered |= (
            (self.completed >= MASTERY_MIN_EPISODES)
            & (finish_rate >= MASTERY_FINISH_RATE)
        )
        # Fix the phase for the whole episode: a mid-race phase change would
        # invalidate the progress term's terminal correction.
        self.time_scale[ids] = self.mastered.float()
        for value in (
            self.start_x, self.spawn_heading, self.progress, self.elapsed, self.steps, self.tipped_steps,
            self.finished, self.failed, self.timed_out, self.done, self.invalid,
        ):
            value[ids] = 0

    def __call__(self, env):
        data = env.scene["robot"].data
        self.steps += 1
        w, x, y, z = data.root_link_quat_w.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.spawn_heading.copy_(torch.where(self.steps == 1, yaw, self.spawn_heading))
        timed = self.steps > self.settle_steps
        valid = (
            torch.isfinite(data.root_link_pos_w).all(dim=-1)
            & torch.isfinite(data.root_link_quat_w).all(dim=-1)
            & torch.isfinite(data.projected_gravity_b).all(dim=-1)
        )
        manager = getattr(env, "termination_manager", None)
        if manager is not None and "nan_state" in manager.active_terms:
            valid &= ~manager.get_term("nan_state")
        self.invalid.copy_(~valid)
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
        env.extras["log"]["Curriculum/race_time_enabled"] = self.time_scale.mean()
        env.extras["log"]["Metrics/recent_finish_rate"] = self.last_finished.float().mean()
        return self.done


def race_time_cost(env):
    """Raw cost rate: zero in discovery, then T or 50 when integrated.

    Early failure pays the remaining seconds immediately, so falling early
    cannot avoid clock costs. mjlab multiplies this rate by step_dt and the
    configured negative TIME_COST_WEIGHT; this function does not negate itself.
    """
    race = env._sprint_race
    timed = race.steps > race.settle_steps
    remaining = (TIME_LIMIT_S - race.elapsed).clamp_min(0.0)
    cost = timed.float() + (race.failed | race.timed_out).float() * remaining / env.step_dt
    return race.time_scale * cost


class DistanceProgress:
    """Pay reductions in distance-to-finish, not occupancy near the finish.

    Discovery retains terminal progress, so a farther unsuccessful attempt is
    better than immediate collapse, but a separate failure cost keeps falls
    net negative. Backtracking gives the credit back. Once
    the episode's time phase is enabled, terminal clawback restores zero-net
    shaping and lets race time, rather than partial-distance credit, rank it.
    """

    def __init__(self, cfg, env):
        del cfg
        self.previous = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.previous[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env):
        race = env._sprint_race
        current = torch.where(race.invalid, self.previous, progress_potential(env))
        current = torch.where(race.done, current * (1.0 - race.time_scale), current)
        rate = (current - self.previous) / env.step_dt
        self.previous.copy_(current)
        return rate


def discovery_finish_bonus(env):
    """One terminal payout for a legal finish; no reward for a failed crossing."""
    race = env._sprint_race
    return race.finished.float() * (1.0 - race.time_scale) / env.step_dt


def discovery_failure_cost(env):
    """One terminal fall/invalid-state cost, not a clock or motion tax.

    Weight this nonnegative rate by -2 * forward_reward. Since bounded progress
    earns at most forward_reward and a failed crossing earns no finish bonus,
    every fallen discovery episode has negative main return. Partial-distance
    differences remain useful; a safe timeout keeps its progress credit.
    The timed phase already charges the Arena's full failure score, so this
    additional cost is disabled there rather than double-counting failure.
    """
    race = env._sprint_race
    return race.failed.float() * (1.0 - race.time_scale) / env.step_dt


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
    """Privileged clock/progress/fall/learning-phase state, never actor inputs.

    Keep four features: replace the redundant episode-length fraction with
    the per-episode phase so the critic can distinguish discovery from timing.
    Negative clock values distinguish the untimed settle from the race start.
    """
    race = getattr(env, "_sprint_race", None)
    if race is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return torch.stack((
        (race.steps - race.settle_steps).clamp(max=race.limit_steps) * env.step_dt / TIME_LIMIT_S,
        (race.progress / DISTANCE_M).clamp(-1.0, 1.0),
        race.tipped_steps / race.fall_steps,
        race.time_scale,
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

    Forward guidance retains partial-distance credit during discovery, with
    falls net negative, then becomes zero-net shaping after finishing is
    established. Other guidance remains zero-net. Head posture is only a small
    tie-breaker, not a reason to choose a slower scored finish.
    No effort, smoothing, joint-proximity, or swing tax remains. Physical
    actuator limits are unchanged; outputs are unclipped.
    """
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep
    # Even over the maximum episode, posture costs less than half the clock
    # return gap of one 20 ms tick averaged across the Arena's five seeds.
    posture_budget = 0.5 * TIME_COST_WEIGHT * control_dt / 5
    cfg.rewards = {
        "race_time": RewardTermCfg(func=race_time_cost, weight=-TIME_COST_WEIGHT),
        "head_posture_cost": RewardTermCfg(
            func=head_posture_cost, weight=-posture_budget / (SETTLE_S + TIME_LIMIT_S),
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=("head_pitch", "head_yaw", "head_roll"),
                preserve_order=True,
            )},
        ),
        "track_linear_velocity": RewardTermCfg(
            func=DistanceProgress, weight=knobs["forward_reward"],
        ),
        "discovery_finish": RewardTermCfg(
            func=discovery_finish_bonus, weight=knobs["forward_reward"],
        ),
        "discovery_failure": RewardTermCfg(
            func=discovery_failure_cost, weight=-2.0 * knobs["forward_reward"],
        ),
        "air_time": RewardTermCfg(
            func=PotentialDifference, weight=knobs["step_reward"],
            params={"potential": race_air_time_potential,
                    "sensor_name": "feet_ground_contact",
                    "threshold_min": 0.06, "threshold_max": 0.22},
        ),
    }
    cfg.rewards["upright"] = RewardTermCfg(
        func=PotentialDifference, weight=knobs["upright_reward"],
        # SteveTarter's dash6 (1d048512e): a 20-degree tilt retains about 56%
        # of the potential instead of 10%, leaving acceleration lean available.
        # This widens tilt tolerance, not a target pose or the Arena fall guard.
        # Keep guidance weak and terminal-corrected, not a standing reward.
        params={
            "potential": mdp.upright, "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # Keep the documented term name, but retain the selected free-yaw behavior.
    # The critic's spawn reference is captured independently by RaceEpisode.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=PotentialDifference, weight=0.0,
        params={
            "potential": microduck_mdp.heading_hold_reward, "std": 0.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
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
    for command in cfg.commands.values():
        command.resampling_time_range = COMMAND_RESAMPLE_S
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.12, 0.12),
        "yaw": (-math.radians(2.0), math.radians(2.0)),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
    cfg.actions["joint_pos"].clip = None
    cfg.actions["joint_pos"].scale = 1.0
    return cfg
