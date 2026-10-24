"""Reference locomotion, warm-started fine-tuning and Sprint checkpoint selection.

The TOML defaults to refinement of a compatible, fully-ramped 5/5 drive policy.
Refinement preserves source environment age and validates mature reward weights;
the explicit free_legs option then removes leg pose and leg smoothing guidance.
The older speed/drive stages restart their reward ramps on a new stage clock.
The former race functions remain importable for saved experiment configurations;
their clock, mastery gate and terminal costs are not active in the new recipe.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import mujoco
import torch
import yaml
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution

DISTANCE_M = 2.0
TIME_LIMIT_S = 50.0
SETTLE_S = 0.5
FALL_GRAVITY_Z = -0.5
FALL_HOLD_S = 0.3
TIME_COST_WEIGHT = 0.1
MASTERY_FINISH_RATE = 0.8
MASTERY_MIN_EPISODES = 32
COMMAND_RESAMPLE_S = (SETTLE_S + TIME_LIMIT_S + 1.0,) * 2
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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


def forward_progress(
    env,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    lateral_std: float = 0.2,
    speed_gain: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Restore 6ac3c45's bounded, upright/low-drift spawn-frame progress."""
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


def _course_velocity(env, asset_cfg, max_tilt_deg=60.0):
    """Match heading_hold's first-step reference without depending on reward order."""
    if not math.isfinite(max_tilt_deg) or not 60.0 <= max_tilt_deg <= 70.0:
        raise ValueError("max_tilt_deg must be finite and between 60 and 70 degrees")
    data = env.scene[asset_cfg.name].data
    w, x, y, z = data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    reference = torch.where(
        env.episode_length_buf <= 1, yaw, getattr(env, "_heading_ref", yaw),
    )
    velocity = data.root_link_lin_vel_w
    speed = reference.cos() * velocity[:, 0] + reference.sin() * velocity[:, 1]
    guard = getattr(env, "_sprint_tilt_guard", None)
    # Deep lean needs a live held-fall guard; legacy and unguarded calls keep 60 degrees.
    cutoff = (FALL_GRAVITY_Z if guard is None or max_tilt_deg == 60.0
              else -math.cos(math.radians(max_tilt_deg)))
    valid = (torch.isfinite(velocity).all(-1)
             & torch.isfinite(data.root_link_quat_w).all(-1)
             & torch.isfinite(data.projected_gravity_b).all(-1)
             & torch.isfinite(reference)
             & (data.projected_gravity_b[:, 2] < cutoff))
    if guard is not None:
        valid &= ~guard.failed
    return torch.where(valid, speed, 0.0), valid


def course_forward_speed(env, vel_ref=1.5, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
                         max_tilt_deg=60.0):
    """Bounded signed spawn-course speed; no command target or payout while fallen."""
    if not math.isfinite(vel_ref) or vel_ref <= 0:
        raise ValueError("vel_ref must be finite and positive")
    speed, _ = _course_velocity(env, asset_cfg, max_tilt_deg)
    return torch.tanh(speed / vel_ref)


def drive_upright(env, std, forward_variance, hold_steps, ramp_steps, motion_speed,
                  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, max_tilt_deg=60.0):
    """Permit nose-down pitch during forward travel, not roll/backward lean or camping.

    On the flat Sprint course projected gravity x is positive for nose-down
    pitch. Keep the original upright optimum; only widen its forward tolerance
    as the stage advances AND course speed rises from zero to motion_speed.
    """
    for name, value in (("std", std), ("forward_variance", forward_variance),
                        ("ramp_steps", ramp_steps), ("motion_speed", motion_speed)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(hold_steps) or hold_steps < 0 or forward_variance < std**2:
        raise ValueError("hold_steps must be nonnegative and forward variance cannot tighten upright")
    speed, valid = _course_velocity(env, asset_cfg, max_tilt_deg)
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    fraction = min(1.0, max(0.0, (env.common_step_counter - hold_steps) / ramp_steps))
    blend = fraction * (speed / motion_speed).clamp(0.0, 1.0)
    variance = std**2 + blend * (forward_variance - std**2)
    pitch_variance = torch.where(gravity[:, 0] > 0, variance, std**2)
    value = torch.exp(-gravity[:, 0].square() / pitch_variance
                      -gravity[:, 1].square() / std**2)
    return torch.where(valid, value, 0.0)


class ArenaTiltGuard:
    """End sustained >60-degree tilts after 0.3 s; allow brief recovery excursions.

    Terminations run before rewards, so the shared per-world failed mask stops
    deep-lean rewards on the held-fall step, before auto-reset.
    """

    def __init__(self, cfg, env):
        del cfg
        self.hold_steps = max(1, round(FALL_HOLD_S / env.step_dt))
        self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._sprint_tilt_guard = self

    def __call__(self, env):
        gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
        tipped = gravity_z > FALL_GRAVITY_Z
        self.steps[:] = torch.where(tipped, (self.steps + 1).clamp_max(self.hold_steps), 0)
        self.failed[:] = (self.steps >= self.hold_steps) | ~torch.isfinite(gravity_z)
        return self.failed

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self.steps[rows] = 0
        self.failed[rows] = False


def productive_air_time(
    env,
    sensor_name: str,
    threshold_min: float = 0.125,
    threshold_max: float = 0.3,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    lateral_std: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Restore 6ac3c45's air-time window gated on productive forward motion."""
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


def configure_0109_locomotion(cfg):
    """Restore the selected run's reward guidance and original 77D critic."""
    cfg.observations["critic"].terms.pop("race_state")
    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["air_time"].func = productive_air_time
    cfg.rewards["air_time"].weight = 6.0
    cfg.rewards["air_time"].params["lateral_std"] = 0.2
    cfg.rewards["forward_progress"] = RewardTermCfg(
        func=forward_progress, weight=0.5,
        params={"command_name": "twist", "lateral_std": 0.2},
    )
    return cfg


class _ConfigLoader(yaml.SafeLoader):
    """Read saved config structure without constructing Python objects."""


def _config_tag(loader, suffix, node):
    del suffix
    if node.tag.startswith("tag:yaml.org,2002:python/name:"):
        return node.tag.removeprefix("tag:yaml.org,2002:python/name:")
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _config_tag)


def _config_data(config):
    return yaml.load(yaml.dump(asdict(config), sort_keys=False), Loader=_ConfigLoader)


def select_sprint_checkpoint(log_root, *, run_pattern=".*", stage="any", checkpoint=None,
                             expected_env=None, expected_agent=None):
    """Choose the fastest verified 5/5 saved result, never the latest checkpoint.

    Incomplete/corrupt results, missing artifacts and legacy race critics are
    ineligible. Optional current configs make a training handoff fail closed on
    observation, action, robot, physics or model/normalizer incompatibility.
    This reads metadata only: it neither evaluates policies nor loads .pt files.
    """
    if stage not in ("any", "baseline", "speed"):
        raise ValueError(f"Unknown source stage: {stage!r}")
    if checkpoint is not None and (type(checkpoint) is not int or checkpoint < 0):
        raise ValueError("checkpoint must be a nonnegative integer")
    pattern = re.compile(run_pattern)
    root = Path(log_root).resolve()
    wanted_env = _config_data(expected_env) if expected_env is not None else None
    wanted_agent = _config_data(expected_agent) if expected_agent is not None else None
    eligible = []
    rejected = []
    for run in sorted(root.iterdir()) if root.is_dir() else ():
        if not run.is_dir() or not pattern.fullmatch(run.name):
            continue
        try:
            source_env = yaml.load((run / "params/env.yaml").read_text(), Loader=_ConfigLoader)
            source_agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=_ConfigLoader)
            provenance = json.loads((run / "provenance.json").read_text())
            if not all(isinstance(doc, dict) for doc in (source_env, source_agent, provenance)):
                raise ValueError("saved config/provenance must be mappings")
            if provenance.get("task") != "Mjlab-Sprint2m-MicroDuck":
                continue
            if source_agent["experiment_name"] != "sprint_2m":
                continue
            critic_terms = source_env["observations"]["critic"]["terms"]
            if not isinstance(critic_terms, dict):
                raise ValueError("saved critic terms must be a mapping")
            padding = critic_terms.get("race_state")
            if padding is not None and padding["func"] != "microduck_challenges.sprint_2m.rewards.reference_critic_padding":
                continue
            if critic_terms["heading_error"]["func"] != "microduck_challenges.sprint_2m.env.heading_error":
                continue
            source_stage = "speed" if "speed_forward" in source_env["rewards"] else "baseline"
            if stage != "any" and source_stage != stage:
                continue
            if wanted_env is not None:
                # Managers concatenate in insertion order; dict equality alone
                # accepts reordered observations/actions with the wrong semantics.
                for group in ("actor", "critic"):
                    if (list(source_env["observations"][group]["terms"])
                            != list(wanted_env["observations"][group]["terms"])):
                        raise ValueError(f"incompatible {group} observation order")
                if list(source_env["actions"]) != list(wanted_env["actions"]):
                    raise ValueError("incompatible action term order")
                for key in ("observations", "actions", "sim", "decimation"):
                    if source_env[key] != wanted_env[key]:
                        raise ValueError(f"incompatible {key}")
                if source_env["scene"]["entities"]["robot"] != wanted_env["scene"]["entities"]["robot"]:
                    raise ValueError("incompatible robot")
            if wanted_agent is not None:
                for key in ("actor", "critic", "obs_groups", "clip_actions"):
                    if source_agent[key] != wanted_agent[key]:
                        raise ValueError(f"incompatible {key}")
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            rejected.append(f"{run.name}: {error}")
            continue
        for result_file in sorted((run / "workshop").glob("model_*.json")):
            try:
                result = json.loads(result_file.read_text())
                number = result["checkpoint"]
                if type(number) is not int or number < 0 or result_file.stem != f"model_{number}":
                    raise ValueError("invalid checkpoint identifier")
                if checkpoint is not None and number != checkpoint:
                    continue
                artifact = run / f"model_{number}.pt"
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError("missing/empty training checkpoint")
                if result.get("error") or result.get("seeds_finished") != 5 or result.get("seeds_total") != 5:
                    continue
                seeds = result["per_seed"]
                if (len(seeds) != 5 or any(type(s["seed"]) is not int for s in seeds)
                        or {s["seed"] for s in seeds} != {1, 2, 3, 4, 5}):
                    raise ValueError("expected exactly seeds 1–5")
                if any(s["status"] != "finish" for s in seeds):
                    continue
                times = [s["time_s"] for s in seeds]
                score = result["score"]
                command = result["command_vx"]
                if any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t <= TIME_LIMIT_S
                       for t in [score, *times]):
                    raise ValueError("invalid finish time")
                if not math.isclose(score, sum(times) / 5, abs_tol=1e-6, rel_tol=1e-8):
                    raise ValueError("score disagrees with five finished seed times")
                if (type(command) not in (int, float) or not math.isfinite(command)
                        or not 0.2 - 1e-8 <= command <= 1.2 + 1e-8):
                    raise ValueError("invalid Sprint command")
                eligible.append({
                    "path": artifact, "result_path": result_file, "score": score,
                    "command_vx": command, "checkpoint": number, "stage": source_stage,
                    "source_env": source_env, "source_agent": source_agent,
                })
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append(f"{result_file}: {error}")
    if not eligible:
        details = "; ".join(rejected[-5:]) or "no complete 5/5 evaluations with compatible saved configs"
        raise RuntimeError(f"No eligible Sprint checkpoint in {root}: {details}. Refusing a cold start.")
    return min(eligible, key=lambda item: (item["score"], item["checkpoint"], str(item["path"])))


def mark_actor_only_inference(env, env_ids=None):
    """Explicit play/export marker; never infer mode from tensor dimensions."""
    del env_ids
    env._sprint_inference_only = True


class SprintPPO(PPO):
    """Load only the actor for explicit play/export; keep training loads strict."""

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        setattr(
            algorithm, "_sprint_inference_only",
            bool(getattr(getattr(env, "unwrapped", env), "_sprint_inference_only", False)),
        )
        return algorithm

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if getattr(self, "_sprint_inference_only", False):
            load_cfg = {"actor": True, "critic": False, "optimizer": False,
                        "iteration": False, "rnd": False}
        return super().load(loaded_dict, load_cfg, strict)

    def train_mode(self):
        if getattr(self, "_sprint_inference_only", False):
            raise RuntimeError("The actor-only Sprint play/export model is inference-only.")
        super().train_mode()


class SprintSpeedPPO(SprintPPO):
    """Warm-start weights/normalizers, not the old optimizer or iteration clock."""

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if getattr(self, "_sprint_inference_only", False):
            return super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": False,
                        "iteration": False, "rnd": False}
            # Mjlab's runner restores this independently of load_iteration.
            # Change the in-memory load payload only, never the source .pt file.
            infos = loaded_dict.get("infos") or {}
            loaded_dict["infos"] = {
                **infos, "env_state": {**infos.get("env_state", {}), "common_step_counter": 0},
            }
        restored_iteration = super().load(loaded_dict, load_cfg, strict)
        self._sprint_warm_started = bool(load_cfg.get("actor") and load_cfg.get("critic"))
        return restored_iteration

    def train_mode(self):
        if getattr(self, "_sprint_inference_only", False):
            raise RuntimeError("The actor-only Sprint play/export model is inference-only.")
        if not getattr(self, "_sprint_warm_started", False):
            raise RuntimeError("The Sprint speed stage requires a loaded locomotion checkpoint.")
        super().train_mode()


class SprintRefinePPO(SprintSpeedPPO):
    """Fresh optimizer/runner iteration while preserving the source environment age."""

    def __init__(self, *args, exploration_std_scale=1.0, **kwargs):
        self.exploration_std_scale = self._noise_scale(exploration_std_scale)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _noise_scale(value):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("exploration_std_scale must be finite and in (0, 1]")
        return float(value)

    def save(self):
        return {**super().save(),
                "sprint_exploration_std_scale": getattr(self, "exploration_std_scale", 1.0)}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if getattr(self, "_sprint_inference_only", False):
            return super().load(loaded_dict, load_cfg, strict)
        required = {"actor": True, "critic": True, "optimizer": False,
                    "iteration": False, "rnd": False}
        if not strict or (load_cfg is not None and load_cfg != required):
            raise ValueError("Refine requires a strict full actor/critic warm start")
        clock = (loaded_dict.get("infos") or {}).get("env_state", {}).get("common_step_counter")
        if type(clock) is not int or clock <= 0:
            raise ValueError("Refine requires the saved source environment clock")
        target_scale = self._noise_scale(getattr(self, "exploration_std_scale", 1.0))
        source_scale = self._noise_scale(loaded_dict.get("sprint_exploration_std_scale", 1.0))
        # An explicit load config bypasses SprintSpeedPPO's new-stage clock reset.
        restored = super().load(loaded_dict, required, strict=True)
        if target_scale != source_scale:
            distribution = self.actor.distribution
            if not isinstance(distribution, GaussianDistribution) or distribution.std_type != "scalar":
                raise ValueError("Phase transitions require the known scalar Gaussian distribution")
            with torch.no_grad():
                distribution.std_param.mul_(target_scale / source_scale)
            print(f"[sprint phase] restored training-noise scale {source_scale} -> {target_scale}; "
                  "actor mean, normalizers and physical action range unchanged")
        return restored


def speed_reward_curriculum(env, env_ids, targets, hold_steps, ramp_steps, initial_fraction=0.0):
    """Blend live reward weights on the new stage's clock, on curriculum resets."""
    del env_ids
    if not math.isfinite(initial_fraction) or not 0.0 <= initial_fraction <= 1.0:
        raise ValueError("initial_fraction must be between zero and one")
    fraction = min(1.0, max(initial_fraction, (env.common_step_counter - hold_steps) / ramp_steps))
    for name, (initial, final) in targets.items():
        env.reward_manager.get_term_cfg(name).weight = initial + fraction * (final - initial)
    return torch.tensor(fraction, device=env.device)


def configure_speed_locomotion(cfg, settings, selected, rollout_steps=24):
    """Fine-tune a competent legged policy without removing balance guidance."""
    source = selected["source_env"]
    source_steps = (selected["checkpoint"] + 1) * selected["source_agent"]["num_steps_per_env"]

    def source_weight(name):
        weight = source["rewards"][name]["weight"]
        for curriculum in source["curriculum"].values():
            params = curriculum["params"]
            if params.get("reward_name") == name:
                for entry in params.get("weight_stages", []):
                    if source_steps > entry["step"]:
                        weight = entry["weight"]
        return weight

    # Start at the selected policy's scheduled regularization, not a random
    # beginner's -0.1 smoothing. Retain upright, heading, gait and fall guidance.
    cfg.curriculum.pop("action_rate_weight", None)
    cfg.curriculum.pop("head_pose_bias_weight", None)
    cfg.curriculum.pop("standing_envs", None)
    cfg.commands["twist"].rel_standing_envs = 0.0
    targets = {}
    for name, final in (
        ("track_linear_velocity", 0.1 * source_weight("track_linear_velocity")),
        ("action_rate_l2", max(source_weight("action_rate_l2"), -0.3)),
        ("pose", 0.6 * source_weight("pose")),
        ("head_pose_bias", source_weight("head_pose_bias") / 3),
        ("body_ang_vel", 0.5 * source_weight("body_ang_vel")),
        ("angular_momentum", 0.5 * source_weight("angular_momentum")),
    ):
        initial = source_weight(name)
        cfg.rewards[name].weight = initial
        targets[name] = (initial, final)
    cfg.rewards["speed_forward"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward, weight=0.0,
        params={"vel_ref": settings["vel_ref"], "asset_cfg": SceneEntityCfg("robot")},
    )
    targets["speed_forward"] = (0.0, settings["forward_reward"])
    cfg.curriculum["speed_rewards"] = CurriculumTermCfg(
        func=speed_reward_curriculum,
        params={"targets": targets,
                "hold_steps": settings["hold_iterations"] * rollout_steps,
                "ramp_steps": settings["ramp_iterations"] * rollout_steps},
    )
    # Do not silently rewind mature CoM randomization when resetting PPO's clock.
    for curriculum in cfg.curriculum.values():
        for entry in curriculum.params.get("range_stages", []):
            entry["step"] = max(-1, entry["step"] - source_steps)
    return cfg


class HeadActionRateL2:
    """Keep head/neck smoothing without taxing any of the ten leg actions."""

    def __init__(self, cfg, env):
        del cfg
        manager = env.action_manager
        names = manager.get_term("joint_pos").target_names
        head_names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
        if (list(manager.active_terms) != ["joint_pos"] or len(names) != 14
                or manager.action.shape[-1] != len(names) or len(set(names)) != len(names)
                or any(names.count(name) != 1 for name in head_names)):
            raise ValueError("head smoothing requires an unambiguous 14-action joint_pos layout")
        self.indices = torch.tensor([names.index(name) for name in head_names],
                                    dtype=torch.long, device=env.device)

    def __call__(self, env):
        manager = env.action_manager
        delta = manager.action[:, self.indices] - manager.prev_action[:, self.indices]
        return delta.square().sum(dim=-1)


def configure_leg_freedom(cfg):
    """Remove leg pose/action-rate preferences, not mechanical limits or guards."""
    cfg.rewards["pose"].weight = 0.0
    cfg.curriculum["speed_rewards"].params["targets"]["pose"] = (0.0, 0.0)
    cfg.rewards["action_rate_l2"].func = HeadActionRateL2
    return cfg


def configure_refine_locomotion(cfg, settings, selected, rollout_steps=24):
    """Validate a drive or mature refine source against an independent recipe."""
    # Construction metadata only: this does not select or invent policy evidence.
    baseline = {"source_env": _config_data(cfg),
                "source_agent": {"num_steps_per_env": rollout_steps},
                "checkpoint": settings["source_baseline_checkpoint"]}
    cfg = configure_drive_locomotion(cfg, settings, baseline, rollout_steps)
    free_legs = settings.get("free_legs", False)
    if type(free_legs) is not bool:
        raise ValueError("training.refine.free_legs must be boolean")
    source = selected["source_env"]
    source_class = selected["source_agent"]["algorithm"]["class_name"]
    source_profile = deepcopy(cfg)
    if source_class == "microduck_challenges.sprint_2m.rewards:SprintRefinePPO":
        # Build the expected mature recipe from local baseline/settings, never
        # from the source being validated (including its weights and guard).
        source_profile = _mature_refine_recipe(source_profile, settings)
        if source["rewards"]["action_rate_l2"]["func"] == f"{__name__}.HeadActionRateL2":
            # Reconstruct an allowlisted profile; validate all its scalars and
            # semantics below rather than trusting the source's reward weights.
            source_profile = configure_leg_freedom(source_profile)
    elif source_class != "microduck_challenges.sprint_2m.rewards:SprintSpeedPPO":
        raise ValueError("Incompatible source algorithm for refine stage")
    canonical = _config_data(source_profile)
    for group in ("actor", "critic"):
        if list(source["observations"][group]["terms"]) != list(canonical["observations"][group]["terms"]):
            raise ValueError(f"Incompatible source {group} observation order")
    if list(source["actions"]) != list(canonical["actions"]):
        raise ValueError("Incompatible source action order")
    for key in ("observations", "actions", "sim", "decimation", "events", "commands",
                "terminations", "episode_length_s", "is_finite_horizon", "scale_rewards_by_dt"):
        if source[key] != canonical[key]:
            raise ValueError(f"Incompatible source {key} for refine stage")
    # Batch size is an execution choice, not a physics change.
    source_scene, canonical_scene = deepcopy(source["scene"]), deepcopy(canonical["scene"])
    for scene in (source_scene, canonical_scene):
        scene.pop("num_envs", None)
        if scene.get("terrain") is not None:
            scene["terrain"].pop("num_envs", None)
    if source_scene != canonical_scene:
        raise ValueError("Incompatible source scene for refine stage")
    if source["rewards"].keys() != canonical["rewards"].keys():
        raise ValueError("Incompatible source reward set for refine stage")
    for name, term in canonical["rewards"].items():
        saved = source["rewards"][name]
        if saved["func"] != term["func"] or saved["params"] != term["params"]:
            raise ValueError(f"Incompatible source reward semantics: {name}")
        weight = saved["weight"]
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight != term["weight"]:
            raise ValueError(f"Invalid source reward weight: {name}")
    if source["curriculum"] != canonical["curriculum"]:
        raise ValueError("Incompatible source curriculum for refine stage")
    params = source_profile.curriculum["speed_rewards"].params
    # Refine descendants restart PPO numbering, but inherit environment age.
    payload = torch.load(selected["path"], map_location="cpu", weights_only=False)
    source_steps = (payload.get("infos") or {}).get("env_state", {}).get("common_step_counter")
    if type(source_steps) is not int or source_steps <= 0:
        raise ValueError("Refine requires the saved source environment clock")
    if source_steps < params["hold_steps"] + params["ramp_steps"]:
        raise ValueError("Refine requires a fully-ramped source checkpoint")
    declared_noise = selected["source_agent"]["algorithm"].get("exploration_std_scale", 1.0)
    saved_noise = payload.get("sprint_exploration_std_scale", 1.0)
    if SprintRefinePPO._noise_scale(saved_noise) != SprintRefinePPO._noise_scale(declared_noise):
        raise ValueError("Incompatible saved noise phase and source agent configuration")
    cfg = _mature_refine_recipe(cfg, settings)
    return configure_leg_freedom(cfg) if free_legs else cfg


def _mature_refine_recipe(cfg, settings):
    """Apply only the explicit refinement recipe, independent of saved metadata."""
    params = cfg.curriculum["speed_rewards"].params
    # Environment construction resets before runner.load restores the source
    # clock. Keep both weights and the logged phase mature during that reset.
    params["initial_fraction"] = 1.0
    for name, (initial, final) in params["targets"].items():
        cfg.rewards[name].weight = initial + (final - initial)
    # Retain the original drive-to-refine transition as well as consolidation.
    lean_variance = settings["lean_forward_variance"]
    max_tilt = settings["lean_max_tilt_deg"]
    if (not math.isfinite(lean_variance) or lean_variance < settings["forward_variance"]
            or not math.isfinite(max_tilt) or not 60.0 <= max_tilt <= 70.0):
        raise ValueError("Invalid refinement forward-lean settings")
    cfg.rewards["upright"].params.update(forward_variance=lean_variance, max_tilt_deg=max_tilt)
    cfg.rewards["speed_forward"].params["max_tilt_deg"] = max_tilt
    cfg.terminations["arena_tilt_hold"] = TerminationTermCfg(func=ArenaTiltGuard, time_out=False)
    return cfg


def configure_drive_locomotion(cfg, settings, selected, rollout_steps=24):
    """Specialize the current 81D baseline: course drive, mobile head and forward lean."""
    source = selected["source_env"]
    current_rewards = _config_data(cfg)["rewards"]
    if current_rewards.keys() != source["rewards"].keys():
        raise ValueError("Incompatible source reward set for drive stage")
    source_steps = (selected["checkpoint"] + 1) * selected["source_agent"]["num_steps_per_env"]
    for name, term in current_rewards.items():
        saved = source["rewards"][name]
        if term["func"] != saved["func"] or term["params"] != saved["params"]:
            raise ValueError(f"Incompatible source reward semantics: {name}")
        weight = saved["weight"]
        for curriculum in source["curriculum"].values():
            params = curriculum["params"]
            if params.get("reward_name") == name:
                for entry in params.get("weight_stages", []):
                    if source_steps > entry["step"]:
                        weight = entry["weight"]
        cfg.rewards[name].weight = weight
    cfg = configure_speed_locomotion(cfg, settings, selected, rollout_steps)
    params = cfg.curriculum["speed_rewards"].params
    initial = cfg.rewards["head_pose_tracking"].weight
    params["targets"]["head_pose_tracking"] = (initial, 0.25 * initial)
    cfg.rewards["speed_forward"].func = course_forward_speed
    cfg.rewards["upright"].func = drive_upright
    cfg.rewards["upright"].params.update(
        forward_variance=settings["forward_variance"], motion_speed=settings["motion_speed"],
        hold_steps=params["hold_steps"], ramp_steps=params["ramp_steps"],
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select a saved Sprint policy by 5/5 finishes, then time")
    parser.add_argument("command", choices=("select",))
    parser.add_argument("--log-root", default="logs/rsl_rl/sprint_2m")
    parser.add_argument("--run", default=".*", help="full-match run directory regex")
    parser.add_argument("--stage", choices=("any", "baseline", "speed"), default="any")
    args = parser.parse_args()
    best = select_sprint_checkpoint(args.log_root, run_pattern=args.run, stage=args.stage)
    print(json.dumps({
        "checkpoint": str(best["path"]), "evaluation": str(best["result_path"]),
        "stage": best["stage"], "seeds_finished": 5, "seeds_total": 5,
        "score": best["score"], "command_vx": best["command_vx"],
    }, indent=2))
