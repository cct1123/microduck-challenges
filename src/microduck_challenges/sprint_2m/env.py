"""Sprint locomotion recovery and warm-started speed/forward-drive specialization.

Learn sustained balance and walking with dense reference rewards before tuning
race speed. Training uses 20-second locomotion episodes, not the experimental
distance/time mastery objective. Soft posture, heading and smoothing guidance
is intentional; all 14 actuator outputs and physical robot limits are unchanged.

The actor remains Arena-compatible (61 observations). Four legacy critic slots
are retained as zeros for checkpoint shape compatibility, not race state. Start
the baseline as a fresh run; matching shapes do not make legacy race critics
suitable. The drive stage selects the current baseline run's best compatible
5/5 checkpoint, preserving its 81D critic, actor and observation normalizers.
The older speed stage restores run-1023-0109's 77D critic and reward guidance.
No source changes alter a training process that is already running.
"""

import math
import os
import re
import tomllib
from copy import deepcopy
from pathlib import Path

NUM_STEPS_PER_ENV = 24
POSE_COMMAND_RESAMPLE_S = (2.0, 5.0)

# Match the reference baseline before adding experimental actor regularizers.
ENABLE_SYMMETRY = False

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = True  # Reference training uses mild recovery pushes
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initially; trunk curriculum reaches ±15mm
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initially; head curriculum reaches ±10mm
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was ±0.5 — an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
# letting a calmer gait be optimal.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # ±0.86° per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

from .rewards import (
    configure_0109_locomotion,
    configure_drive_locomotion,
    configure_reference_locomotion,
    configure_refine_locomotion,
    configure_speed_locomotion,
    mark_actor_only_inference,
    select_sprint_checkpoint,
)

TRAINING_RECIPE = challenge.load(__file__).recipe
STAGE_SETTINGS = tomllib.loads(Path(__file__).with_name("challenge.toml").read_text())["training"]
TRAINING_STAGE = os.environ.get("MICRODUCK_SPRINT_STAGE", STAGE_SETTINGS["stage"])
if TRAINING_STAGE not in ("baseline", "speed", "drive", "refine"):
    raise ValueError(f"Unknown Sprint training stage: {TRAINING_STAGE!r}")
SPEED_SETTINGS = STAGE_SETTINGS["speed"]
DRIVE_SETTINGS = STAGE_SETTINGS["drive"]
REFINE_SETTINGS = STAGE_SETTINGS["refine"]
SPEED_CHECKPOINT = None
DRIVE_CHECKPOINT = None
REFINE_CHECKPOINT = None
for _stage in ("speed", "drive", "refine"):
    _settings = STAGE_SETTINGS[_stage]
    for _key in ("max_iterations", "save_interval", "hold_iterations", "ramp_iterations"):
        _value = _settings[_key]
        if type(_value) is not int or _value < (0 if _key == "hold_iterations" else 1):
            raise ValueError(f"training.{_stage}.{_key} must be a positive integer (hold may be zero)")
    for _key in ("learning_rate", "vel_ref", "forward_reward"):
        _value = _settings[_key]
        if type(_value) not in (int, float) or not math.isfinite(_value) or _value <= 0:
            raise ValueError(f"training.{_stage}.{_key} must be finite and positive")
    if _settings["hold_iterations"] + _settings["ramp_iterations"] > _settings["max_iterations"]:
        raise ValueError(f"The {_stage} reward ramp must fit within the stage budget")
for _key in ("forward_variance", "motion_speed"):
    _value = DRIVE_SETTINGS[_key]
    if type(_value) not in (int, float) or not math.isfinite(_value) or _value <= 0:
        raise ValueError(f"training.drive.{_key} must be finite and positive")
if DRIVE_SETTINGS["forward_variance"] < 0.05:
    raise ValueError("training.drive.forward_variance must not tighten forward lean")


def heading_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic-only error against the reference captured by heading_hold."""
    w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    reference = torch.where(
        env.episode_length_buf <= 1, yaw, getattr(env, "_heading_ref", yaw),
    )
    error = yaw - reference
    return torch.atan2(torch.sin(error), torch.cos(error)).unsqueeze(-1)


def make_env_cfg(
    play: bool = False,
    *, stage: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """The sprint's training environment."""

    # The knobs a novice tunes, from challenge.toml beside this file.
    knobs = challenge.params(__file__)

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT foot first, RIGHT foot second
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",), reduce="none", num_slots=1,
    )

    # Used by the reference's foot-height observations and clearance rewards.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()
    cfg.scene.num_envs = int(TRAINING_RECIPE["num_envs"])
    if not ENABLE_VELOCITY_PUSHES:
        cfg.events.pop("push_robot", None)  # also remove any inherited base event

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # Events
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # Grippier footpad — narrowed from (0.3, 1.2)
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # Velocity-based pushes for robustness training
    if ENABLE_VELOCITY_PUSHES:
        # In play mode, use shorter interval for better visibility
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # required by domain_randomization=True
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.

    # Base orientation randomization (forces reactive behavior)
    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # Observations
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Add base_lin_vel to critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # Privileged only: the actor is never told its heading.
    cfg.observations["critic"].terms["heading_error"] = ObservationTermCfg(func=heading_error)

    # Determine gravity/accelerometer term name based on flag
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Commands — deepcopy to avoid shared-state corruption from other env cfgs
    # (make_velocity_env_cfg() returns objects with shared mutable references;
    # standup/ground_pick envs mutate commands["twist"] in place, zeroing ranges)
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    command.rel_standing_envs = 0.02  # Reference balance practice, ramped by curriculum
    command.rel_heading_envs = 0.0
    # The ranges are the Arena event's own (arena/events/sprint-2m.toml):
    # straight ahead, the event's sweep of commanded speeds, never sideways or
    # turning. History: the source task trained ±0.4 m/s, and the Arena's
    # reference walker then peaked at exactly 0.40 m/s — the top of that range.
    command.ranges.lin_vel_x = (0.20, 1.20)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Preserve the reference's command slots and soft head-tracking targets.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=POSE_COMMAND_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — much smaller mechanical range)
        ),
    )
    # Body command slot: [x, y, z, roll, pitch, yaw]; tracking weight remains zero.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=POSE_COMMAND_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # Append head + body command obs terms to both policy and critic groups.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # Terrain: the sprint's lane is flat.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # CoM randomization range curriculum - start small, ramp up
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    # Capped at ±15 mm (2026-07 audit): the previous ramp to ±30 mm
                    # exceeded the foot support polygon (heel is only 20 mm behind
                    # the ankle) — the randomized CoM could sit entirely outside
                    # support, forcing a wide/fast hyper-reactive gait and making
                    # BACKWARD balance untrainable. Regression timeline matched the
                    # ramp increases: 0.015 → 0.02 → 0.03 as policies got worse.
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "range": 0.015},
                ],
            },
        )

    # Head CoM randomization range curriculum - start small, ramp up
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    # Capped at ±10 mm (2026-07 audit — same over-conservatism
                    # concern as trunk CoM; head is a large lever arm).
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
                ],
            },
        )

    # Disable default curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Baseline behavior is unchanged. A speed stage is a separately launched run.
    cfg = configure_reference_locomotion(cfg, knobs, NUM_STEPS_PER_ENV)
    active_stage = TRAINING_STAGE if stage is None else stage
    if active_stage in ("speed", "source"):
        cfg = configure_0109_locomotion(cfg)
    if active_stage == "speed":
        if SPEED_CHECKPOINT is None:
            raise RuntimeError("Select a compatible 5/5 baseline before speed tuning")
        cfg = configure_speed_locomotion(cfg, SPEED_SETTINGS, SPEED_CHECKPOINT, NUM_STEPS_PER_ENV)
    elif active_stage == "drive":
        if DRIVE_CHECKPOINT is None:
            raise RuntimeError("Select a compatible 5/5 baseline before forward-drive tuning")
        cfg = configure_drive_locomotion(cfg, DRIVE_SETTINGS, DRIVE_CHECKPOINT, NUM_STEPS_PER_ENV)
    elif active_stage == "refine":
        if REFINE_CHECKPOINT is None:
            raise RuntimeError("Select a compatible 5/5 drive checkpoint before refinement")
        cfg = configure_refine_locomotion(cfg, REFINE_SETTINGS, REFINE_CHECKPOINT, NUM_STEPS_PER_ENV)
    elif active_stage not in ("baseline", "source"):
        raise ValueError(f"Unknown Sprint environment stage: {active_stage!r}")
    if play:
        cfg.events["actor_only_inference"] = EventTermCfg(
            func=mark_actor_only_inference, mode="startup",
        )
    return cfg


RL_CFG = RslRlOnPolicyRunnerCfg(
    seed=int(TRAINING_RECIPE["seed"]),
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            # Reference exploration scale; no change to physical action range.
            "init_std": 1.0,
            "std_type": "scalar",  # independent learned std for each of 14 outputs
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        class_name="microduck_challenges.sprint_2m.rewards:SprintPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=deepcopy(SYMMETRY_CFG) if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="sprint_2m",  # logs/rsl_rl/sprint_2m/
    run_name="sprint_2m",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=int(TRAINING_RECIPE["max_iterations"]),
)

if TRAINING_STAGE in ("speed", "drive", "refine"):
    _settings = STAGE_SETTINGS[TRAINING_STAGE]
    _selected = select_sprint_checkpoint(
        Path("logs/rsl_rl/sprint_2m"),
        run_pattern=re.escape(_settings["source_run"]),
        stage="speed" if TRAINING_STAGE == "refine" else "baseline",
        checkpoint=_settings.get("source_checkpoint"),
        expected_env=make_env_cfg(stage="source" if TRAINING_STAGE == "speed" else "baseline"),
        expected_agent=RL_CFG,
    )
    if TRAINING_STAGE == "speed":
        SPEED_CHECKPOINT = _selected
    elif TRAINING_STAGE == "refine":
        REFINE_CHECKPOINT = _selected
    else:
        DRIVE_CHECKPOINT = _selected
    _checkpoint_path = _selected["path"]
    RL_CFG.resume = True
    # Exact regexes prevent the generic runner from substituting the latest model.
    RL_CFG.load_run = "^" + re.escape(_checkpoint_path.parent.name) + "$"
    RL_CFG.load_checkpoint = "^" + re.escape(_checkpoint_path.name) + "$"
    RL_CFG.run_name = f"sprint_2m_{TRAINING_STAGE}"
    RL_CFG.max_iterations = _settings["max_iterations"]
    RL_CFG.save_interval = _settings["save_interval"]
    RL_CFG.algorithm.class_name = "microduck_challenges.sprint_2m.rewards:SprintSpeedPPO"
    if TRAINING_STAGE == "refine":
        RL_CFG.algorithm.class_name = "microduck_challenges.sprint_2m.rewards:SprintRefinePPO"
    RL_CFG.algorithm.learning_rate = _settings["learning_rate"]
    RL_CFG.algorithm.schedule = "fixed"
    print(f"[sprint {TRAINING_STAGE} config] selected training source {_checkpoint_path}; "
          f"5/5 in {_selected['score']:.3f}s at "
          f"{_selected['command_vx']:.2f}m/s; "
          "the runner logs the checkpoint actually loaded")
