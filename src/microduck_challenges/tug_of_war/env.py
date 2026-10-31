"""The Tug of War's training environment: the 2 m Sprint's recipe made the tug's.

The duck trains in the Arena's bout: a 3 m ring cut by a pit, two ducks tied neck to neck by a 1 m
string, for a bout's length. The learner (`robot`) is always the west duck; the `opponent` is the
east one, turned half a way about the pit's vertical axis (Plan F, Decision 1: the east side is
the same bout turned). The HF job installs no Arena, so the event's numbers and the course's
arithmetic (arena/course.py, compile_tug) are copied here; tests/test_tug_of_war.py pins each
copy against the Arena's own model.

`make_env_cfg` is the sprint's recipe (sprint_2m/env.py), the shortest way to a duck that steps
backwards, with the knobs a novice tunes (the three reward weights and stay upright) read from challenge.toml ([params]). Everything
here is yours to change; the robot (MICRODUCK_STANDUP_ROBOT_CFG) and the shared reward functions
(microduck_mdp) come from mjlab_microduck, the tug's own terms from rewards.py. It carries these
decisions, each commented where it is made:

  D1  no mjlab terrain: `tug_scene` builds the course (the ring: its ropes and posts, its
      backstops, its platforms and the pit across it) and every env origin is 0, so each world
      holds it at the Arena's coordinates (no mjlab terrain has a pit)
  D2  the course body is named `terrain`, as the foot contact sensor expects
  D3  `tug_scene` adds both ducks' string ends (a site at the base of each neck) and the string
      tied to them (`add_string`), so the robot cfg stays untouched
  D4  two entities, `robot` and `opponent`, both the standup robot, tied by the string:
      `compile_tug`'s bout. The string is a rope, the Arena's (arena.course._add_string): sixteen
      capsules, 80 g and 8 mm thick, two hinges a joint, tied neck to neck; it sags, pulls only
      when taut and can be stepped or lain on. Its numbers are the Arena's, measured by the rope
      study (2026-10-21), and never lowered: each step down from them was unstable there. Its
      armature, on every string dof, keeps mujoco_warp finite: without it every world turned NaN
      within 0.4 to 0.7 s, the rope's twist inertia (3e-8 kg m²) beside the ducks' terms being
      beyond float32's precision once the rope touches a duck (study §5). The training feet grip the
      string harder than the Arena's: mjlab gives them contact priority 1 (the Arena's feet have 0)
      and the string has 1, so a foot on it mixes friction to max(foot, 1.0), 1.0 to 1.3 with the
      learner's randomised foot friction, against 1.0 in the Arena; neither number changes (the
      string's are measured, the feet's priority keeps floor contact equal)
  D5  the Arena's numbers are copied below, pinned by the parity tests
  D6  the twist is the event's fixed command, -0.30 m/s backwards, in every env
  D7  the Arena's collision set (the standup robot: a dragged duck touches with its body) and
      solver: implicitfast, 10/20 iterations, 0.005 s x 4
  D9  either pit ends an episode, as on the Arena (owner, 2026-10-20: going down to hold is a
      fair way to hold, the point is to drag the other duck in): the learner's costs `out`
      (-50), the opponent's pays `win` (+50); both in on one step pay nothing, the rarest end. A
      fall ends one too, TRAINING ONLY, with `out`: the Arena's fall rule (ArenaFall) ends a
      training episode so the duck keeps learning to stand and pull; the Arena judges only the
      pit. Without it an untrained duck falls at 1 s and lies on its side for the rest of the
      episode, collecting the pit margin, and never learns to stand (t2, t3). NaN physics on
      either duck ends one too, paying nothing: the critic reads both
  D11 an episode is the settle plus a bout: `BOUT_S`
  D12 the starts: west x [west_x, west_x + 2 cm], east x [east_x - 2 cm, east_x], y ±1 cm, yaw
      the Arena's ±YAW_PERTURBATION_DEG (east 180° ± it): never farther apart than the Arena's
      taut start, since a stretched start jerks both ducks
  D13 the critic alone reads the pit margin, the opponent's trunk (its position from the
      learner's, its gravity, its velocities), the string's length and the marker; the actor
      reads the Arena's 61 numbers
  D14 `marker_ground` pays the new ground the marker wins towards the learner, the knob a
      novice tunes as "Win ground": the Arena rates a bout by the marker, and only ground past
      the episode's best counts, so a sway earns nothing
  D15 walking back pays three times more (`back_reward` 3.0, std √0.05: standing still keeps a
      sixth of it). t1-t5 never discovered the walk back: each dove just under the fall line
      and earned at standstill 0.41 of what walking back pays
  D16 the opponent is the Arena's reference duck, frozen (opponent.py): an action term with no
      agent dimension observes it as the Arena does and drives its own servos at each step's
      start, on the state the learner's observation saw. Nothing is randomised on it, as on the
      Arena: no DR event or push names it, and its servos (OPPONENT_CFG) run on the Arena's
      nominal battery, VIN_NOMINAL with VIN_DROP_GAIN's sag, where the learner's draw U(6.5, 8.2)
      V and U(0, 0.2) per world. Their command delay stays drawn per substep, as the Arena's
  D17 removed, locomotion first (a slack string for the first 250 iterations): a rope has no
      limit to lift, and every training starts as the reference (D19), which already walks back:
      t11, taut from step 0, won sooner than t10, which kept the slack phase
  D18 the course's boxes (platforms, pit floor, backstops) come in tiles at most TILE_M (2 m)
      along x and y, their tops and contacts the Arena's; its capsules (ropes, posts) come in
      pieces at most TILE_M long, end to end (D18's rule for every course geom; a cut capsule is
      the same union). Where a duck lies flat on a box, mujoco_warp's box-mesh collision can
      put the contact off the duck, along an edge of that box, by several times its length: under
      the Arena's 21.85 x 40.9 m platforms it landed 60 to 130 m away and the solver turned the
      whole world NaN (t8, t9: nan_state on both ducks at once, the frozen opponent lying head
      down for most of a bout). On 2 m tiles it lands within 8 m and the world stays finite.
      The tiles also give back the Arena's ground: on the 40 m boxes mujoco_warp made about 38 %
      fewer duck-course contacts than the Arena's MuJoCo and a pushed duck slid further; tiled,
      both agree (t1-t9 trained on the untiled boxes)
  D19 the learner starts as the reference duck (RL_CFG's actor, opponent.ReferenceActor). From
      scratch it plateaued (t8r +12.6 cm at 1000) or learned to lie flat and hold (t9r -18.3 cm);
      started as a duck that already backs and pulls, it dragged the reference into the pit in 10
      bouts of 10 by 500 (t10 +139.6 cm), then decayed to lying prone by 1000 (-17.6 cm): the
      recipe (challenge.toml) stops at 1000 and the best checkpoint is kept, not the last
  D20 the ring is the Arena's (Plan G Task 2): ropes and posts are rigid capsules, backstops
      close the gaps between them, the pit crosses the ring from rope to rope
  D21 every reset lays the string straight between the ducks' actual string ends, centred on
      them (`reset_string`): mujoco_warp's reset_data puts it back straight and still between the
      nominal starts, up to 2 cm off the jittered ducks (D12), and the event moves its root onto
      the ends the ducks' resets drew. A straight 1.00 m string overshoots ends closer than
      1 m by (1 - gap) / 2 at each end, at most 2 cm, and the ties take it in on the first steps
      (Plan H Task 4's probe watches them for NaN)
  D22 stay_upright, a knob (challenge.toml): 0, what users get, lets a duck crouch, sit, lie or
      plant its head, as the Arena allows, and keeps D9's fall. Above 0, `down` replaces `fell`:
      the learner out of the walking-upright bands (rewards.is_upright: the trunk UPRIGHT_Z_M up
      or more, pitched within UPRIGHT_PITCH_DEG, rolled UPRIGHT_ROLL_DEG at most, the head's
      centre of mass UPRIGHT_HEAD_Z_M up or more, the NECK joints within UPRIGHT_NECK_DEG of
      STAND), or any body but its feet on the course (the BODY_GROUND sensor), for FALL_HOLD_S
      ends the training episode and costs that many pits (-50 each, `down`). The bands are the
      walker's envelope with margin: BEST_alpha_walking over six runs held the trunk at
      11.0–12.4 cm, pitch -2.7..+2.1°, roll 4.7° at most, head 22.6–24.1 cm, neck -4.7..+11°,
      head pitch -8.3..+5°, only the feet down. The 20° of leaning back is sized for a ~2.5 N
      string held at the 15 cm tie, a physics estimate. The bow it ends (t15-250, t17-250 and
      t19-250, probe 2026-10-22): trunk 7–8 cm, pitched 33–44° towards the opponent, neck
      50–85° down, head on the floor in 77–83 % of frames. Why the sensor: posed at random
      within every band, a shin touched the floor in 2.2 % of 200,000 poses, the trunk at
      10.0–11.1 cm with a foot down. The owner's ducks train with it on (owner, 2026-10-22);
      check.py reads the same rule
"""

import math
from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

from .opponent import OpponentActionCfg
from .rewards import (
    FEET,
    HEAD,
    MIDDLE,
    NECK,
    STRING,
    ArenaFall,
    Down,
    MarkerGround,
    ended_by,
    marker_x,
    opponent_state,
    pit_margin,
    pit_margin_reward,
    string_length,
)

# Copied from the Arena (the HF job cannot import it); tests/test_tug_of_war.py pins each one.
RING_M, PIT_M, PIT_DEPTH_M, STRING_M = 3.0, 0.30, 0.30, 1.00  # arena/events/tug-of-war.toml
RING_ROPES_M = (0.06, 0.12, 0.18)                          # arena/course.py: the ropes' heights, bottom to top
RING_ROPE_RADIUS_M, RING_POST_RADIUS_M, RING_POST_TOP_M = 0.006, 0.022, 0.23
COMMAND_VX = -0.30
COMMAND = (COMMAND_VX,) + (0.0,) * 12                      # [command], the 13 numbers a duck reads
BOUT_S, SETTLE_S = 45.0, 0.5
JOINT_PERTURBATION_RAD, YAW_PERTURBATION_DEG = 0.01, 2.0
FALL_GRAVITY_Z, FALL_HOLD_S = -0.5, 0.3                    # [termination]: TRAINING ONLY here (D9)
WALL_THICKNESS_M, PLATFORM_OVERRUN_M = 0.05, 20.0           # arena/course.py
STRING_OFFSET = (0.026, 0.0145, 0.0324215)                 # string_offset("legs")
PIT_DROP_M = 0.05                                          # arena/tug.py
VIN_NOMINAL, VIN_DROP_GAIN = 7.35, 0.1                     # arena/actuator.py: the reference duck's battery
STRING_SITE = "string_end"
SPAWN_HEIGHT_M = 0.12                                      # arena/course.py: the string is built at it

# The string (D4), arena/course.py's: the rope study's numbers, never lowered.
STRING_SEGMENTS = 16                                       # capsules of 6.25 cm
STRING_RADIUS_M = 0.004
STRING_MASS_KG = 0.080
STRING_END_MASS_KG = 0.015                                 # each end piece: a tie is as stiff as its tip is heavy
STRING_DAMPING = 0.003                                     # N m s/rad a hinge
STRING_ARMATURE = 1e-6                                     # on every string dof, the free root's too (D4)
STRING_TIE_SOLREF = (0.01, 1.0)                            # two timesteps, the stiffest MuJoCo allows
STRING_TIE_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)           # MuJoCo's default, as the study measured it
STRING_FRICTION = (1.0, 0.005, 0.0001)

TILE_M = 2.0  # the longest a course tile is along x or y (D18)
BODY_GROUND = "body_ground_contact"  # the learner's bodies but its feet on the course (D22)

ROBOT = "robot/"  # the prefix mjlab attaches the learner entity under
OPPONENT = "opponent/"  # and the opponent's

# The opponent: the standup robot on the Arena's servos (D16), its battery VIN_NOMINAL with
# VIN_DROP_GAIN's sag in every world; the learner's are drawn per world.
OPPONENT_CFG = replace(MICRODUCK_STANDUP_ROBOT_CFG, articulation=replace(
    MICRODUCK_STANDUP_ROBOT_CFG.articulation,
    actuators=tuple(replace(servos, vin_range=(VIN_NOMINAL,) * 2, vin_drop_gain_range=(VIN_DROP_GAIN,) * 2)
                    for servos in MICRODUCK_STANDUP_ROBOT_CFG.articulation.actuators),
))

# The sprint's recipe switches and ranges (sprint_2m/env.py), the ones it uses.
NUM_STEPS_PER_ENV = 24

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)

# Head/body pose commands. Head pose: 4D deltas-from-HOME on neck/head joints,
# tracked by head_pose_tracking. Body pose: 6D delta in [x, y, z, roll, pitch,
# yaw], tracked by nothing: it keeps the actor's command slot (6 of its 61
# numbers) alive. The Arena sends zeros in both.
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped to ±8mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% applied to BOTH mass and inertia together.
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was ±0.5 — an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
# letting a calmer gait be optimal.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # ±0.86° per-joint encoder offset (constant per env)


def course_boxes() -> list[tuple[str, tuple, tuple]]:
    """The tug's ground and backstops, `(name, pos, half_extents)`: the Arena's
    `pit_boxes(course) + backstops(course)`, same order, same arithmetic. Two platforms with
    their top face at z = 0 either side of the pit, the pit's floor at z = -PIT_DEPTH_M, and a
    wall flush behind each side's ropes, up to the top rope; no plane, so the pit is a hole."""
    half_t = WALL_THICKNESS_M / 2
    half_y = RING_M / 2 + WALL_THICKNESS_M + PLATFORM_OVERRUN_M
    x_out = RING_M / 2 + PLATFORM_OVERRUN_M
    half_pit = PIT_M / 2
    slab = PIT_DEPTH_M + WALL_THICKNESS_M
    platform = ((x_out - half_pit) / 2, half_y, slab / 2)
    centre = (x_out + half_pit) / 2
    top = RING_ROPES_M[-1] + RING_ROPE_RADIUS_M
    reach, along = RING_M / 2 + RING_ROPE_RADIUS_M + half_t, RING_M / 2 + RING_ROPE_RADIUS_M + WALL_THICKNESS_M
    return [
        ("platform_west", (-centre, 0.0, -slab / 2), platform),
        ("platform_east", (centre, 0.0, -slab / 2), platform),
        ("pit_floor", (0.0, 0.0, -PIT_DEPTH_M - half_t), (half_pit, half_y, half_t)),
        ("backstop_west", (-reach, 0.0, top / 2), (half_t, along, top / 2)),
        ("backstop_east", (reach, 0.0, top / 2), (half_t, along, top / 2)),
        ("backstop_north", (0.0, reach, top / 2), (along, half_t, top / 2)),
        ("backstop_south", (0.0, -reach, top / 2), (along, half_t, top / 2)),
    ]


def course_capsules() -> list[tuple[str, tuple, tuple, float]]:
    """The ring's ropes then its posts, `(name, from, to, radius)`: the Arena's `ring_ropes(course)`
    and `ring_posts(course)`, same order, same arithmetic. A side runs post to post at the ring's
    edge, its ropes bottom to top; a post runs from the canvas to RING_POST_TOP_M."""
    h = RING_M / 2
    sides = {"west": ((-h, -h), (-h, h)), "east": ((h, -h), (h, h)),
             "north": ((-h, h), (h, h)), "south": ((-h, -h), (h, -h))}
    corners = {"nw": (-h, h), "ne": (h, h), "sw": (-h, -h), "se": (h, -h)}
    return [
        (f"rope_{side}_{k}", (*start, z), (*end, z), RING_ROPE_RADIUS_M)
        for side, (start, end) in sides.items() for k, z in enumerate(RING_ROPES_M)
    ] + [
        (f"post_{corner}", (x, y, 0.0), (x, y, RING_POST_TOP_M), RING_POST_RADIUS_M)
        for corner, (x, y) in corners.items()
    ]


def tiles(boxes: list[tuple[str, tuple, tuple]]) -> list[tuple[str, tuple, tuple]]:
    """Each box `(name, pos, half_extents)` cut into equal tiles at most TILE_M along x and y, named
    `name_i_j`, each as high as the box (D18)."""
    out = []
    for name, (x, y, z), (hx, hy, hz) in boxes:
        nx, ny = math.ceil(2 * hx / TILE_M), math.ceil(2 * hy / TILE_M)
        out += [(f"{name}_{i}_{j}", (x - hx + (2 * i + 1) * hx / nx, y - hy + (2 * j + 1) * hy / ny, z),
                 (hx / nx, hy / ny, hz)) for i in range(nx) for j in range(ny)]
    return out


def pieces(capsules: list[tuple[str, tuple, tuple, float]]) -> list[tuple[str, tuple, tuple, float]]:
    """Each capsule `(name, from, to, radius)` cut into equal pieces at most TILE_M long, end to end,
    named `name_k` (D18)."""
    out = []
    for name, start, end, radius in capsules:
        n = math.ceil(math.dist(start, end) / TILE_M)
        point = [tuple(a + (b - a) * k / n for a, b in zip(start, end)) for k in range(n + 1)]
        out += [(f"{name}_{k}", point[k], point[k + 1], radius) for k in range(n)]
    return out


def west_start() -> tuple[float, float]:
    """The west duck's trunk (x, y), facing east: its string end on y = 0, half the string
    west of the pit's centre."""
    ox, oy, _ = STRING_OFFSET
    half = STRING_M / 2
    return -half - ox, -oy


def east_start() -> tuple[float, float]:
    """The east duck's trunk (x, y), facing west: the west start turned half a way about the
    pit's vertical axis."""
    x, y = west_start()
    return -x, -y


def tug_scene(spec: mujoco.MjSpec) -> None:
    """mjlab's `SceneCfg.spec_fn`: the Arena's bout, built on the scene once both ducks are in
    (`arena.course.compile_tug`, Plan F Decisions 1 and 2).

    The course is a static body named `terrain`, as mjlab's foot contact sensor expects, its boxes
    in tiles and its ropes and posts in pieces (D18). Each duck, `robot` and `opponent`, carries a
    string end, a site at the base of its neck, and the string (`add_string`) is built straight
    between the two as they stand at their nominal starts.
    """
    terrain = spec.worldbody.add_body(name="terrain")
    for name, pos, size in tiles(course_boxes()):
        terrain.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
    for name, start, end, radius in pieces(course_capsules()):
        terrain.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*start, *end], size=[radius, 0, 0])
    for prefix in (ROBOT, OPPONENT):
        spec.body(prefix + "trunk_base").add_site(name=prefix + STRING_SITE, pos=STRING_OFFSET)
    (west_x, west_y), (east_x, east_y) = west_start(), east_start()
    ox, oy, oz = STRING_OFFSET  # the east duck is turned half a way: its offset points west
    add_string(spec, (west_x + ox, west_y + oy, SPAWN_HEIGHT_M + oz), (east_x - ox, east_y - oy, SPAWN_HEIGHT_M + oz))


def add_string(spec: mujoco.MjSpec, a, b) -> None:
    """The Arena's string (arena.course._add_string), straight from world point `a` to `b` due east
    of it: STRING_SEGMENTS capsules in one tree, rooted by a free joint at the middle segment, two
    chains out to the ends. Each joint is two hinges, y then z: no twist, which a round chain cannot
    show. Segment k carries the site p{k} at its west end, the last one p{N} at its east end too.
    The first point is tied to the learner's string end, the last to the opponent's, each by a
    `connect`, and each end piece is left out of the collisions with the trunk it is tied to. The
    string never collides with itself (contype 0) and touches everything else, its own contact
    numbers winning (priority 1; MuJoCo's default solref and solimp)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, mid = STRING_SEGMENTS, MIDDLE
    seg = float(np.linalg.norm(b - a)) / n
    inner = (STRING_MASS_KG - 2 * STRING_END_MASS_KG) / (n - 2)

    def segment(parent: mujoco.MjsBody, k: int, pos, west: float) -> mujoco.MjsBody:
        """Segment k at `pos` in `parent`, its capsule running east from `west` on its x."""
        body = parent.add_body(name=f"{STRING}seg{k}", pos=list(pos))
        if k == mid:
            body.add_freejoint(name=f"{STRING}root").armature = STRING_ARMATURE
        else:
            for axis, direction in (("y", [0, 1, 0]), ("z", [0, 0, 1])):
                body.add_joint(
                    name=f"{STRING}j{k}{axis}",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=direction,
                    damping=STRING_DAMPING,
                    armature=STRING_ARMATURE,
                )
        body.add_geom(
            name=f"{STRING}cap{k}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[west, 0, 0, west + seg, 0, 0],
            size=[STRING_RADIUS_M, 0, 0],
            mass=STRING_END_MASS_KG if k in (0, n - 1) else inner,
            contype=0,
            conaffinity=1,
            priority=1,
            friction=list(STRING_FRICTION),
        )
        body.add_site(name=f"{STRING}p{k}", pos=[west, 0, 0])
        if k == n - 1:
            body.add_site(name=f"{STRING}p{n}", pos=[west + seg, 0, 0])
        return body

    root = segment(spec.worldbody, mid, (a + b) / 2, 0.0)
    parent = root
    for k in range(mid + 1, n):  # the east half: each segment hangs from its west end
        parent = segment(parent, k, (seg, 0, 0), 0.0)
    parent = root
    for k in range(mid - 1, -1, -1):  # the west half: each segment hangs from its east end
        parent = segment(parent, k, (0 if k == mid - 1 else -seg, 0, 0), -seg)

    for prefix, tip in ((ROBOT, 0), (OPPONENT, n)):  # the learner to the first point, the opponent to the last
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=prefix + STRING_SITE,
            name2=f"{STRING}p{tip}",
            solref=list(STRING_TIE_SOLREF),
            solimp=list(STRING_TIE_SOLIMP),
        )
    for prefix, piece in ((ROBOT, 0), (OPPONENT, n - 1)):
        spec.add_exclude(bodyname1=prefix + "trunk_base", bodyname2=f"{STRING}seg{piece}")


def reset_string(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """A reset event, after both ducks' (D21): the string straight between their string ends as
    their resets placed them. mjlab's reset (mujoco_warp's reset_data) has already put every hinge
    at 0, the string straight, and every velocity at 0, so this only places the root, the middle
    point, centred between the two ends. Each end is the duck's free-joint position plus
    STRING_OFFSET turned by its quaternion, read off qpos: the sites' kinematics are stale until
    the reset's forward. The root's quaternion turns its +x onto the west end to the east end, a
    yaw then a pitch."""
    data = env.sim.data
    ends = []
    for name in ("robot", "opponent"):
        pose = data.qpos[env_ids][:, env.scene[name].indexing.free_joint_q_adr]
        ends.append(pose[:, :3] + quat_apply(pose[:, 3:], pose.new_tensor(STRING_OFFSET).expand_as(pose[:, :3])))
    west, east = ends
    direction = (east - west) / (east - west).norm(dim=-1, keepdim=True)
    yaw, pitch = torch.atan2(direction[:, 1], direction[:, 0]), -torch.asin(direction[:, 2])
    q = int(env.sim.mj_model.joint(f"{STRING}root").qposadr[0])
    data.qpos[env_ids, q:q + 3] = (west + east) / 2
    data.qpos[env_ids, q + 3:q + 7] = quat_from_euler_xyz(torch.zeros_like(yaw), pitch, yaw)


def heading_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """How far the trunk has turned from the heading heading_hold holds (rad, wrapped).

    Critic-only: the Arena never tells the actor its heading. Reads the spawn
    heading heading_hold_reward captures (env._heading_ref); zero on an
    episode's first observation, which comes before that reward captures it.
    """
    w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    reference = torch.where(env.episode_length_buf == 0, yaw, getattr(env, "_heading_ref", yaw))
    error = yaw - reference
    return torch.atan2(torch.sin(error), torch.cos(error)).unsqueeze(-1)


def make_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """The Tug of War's training environment."""

    # The knobs a novice tunes, from challenge.toml beside this file.
    knobs = challenge.params(__file__)
    stay_upright = knobs["stay_upright"]

    std_standing = {
        # Lower body — tighter to keep the robot in home pose when standing
        r".*hip_yaw.*": 0.1,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance (sole sits flat), stop leg splay
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.1,
    }

    std_walking = {
        # Lower body
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 — hold the 5°-inward stance, stop the leg splay to vertical
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25, # was 0.15
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order. Its secondary is the body tug_scene
    # names `terrain` (D2): the course's platforms and pit.
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
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # No foot_height_scan (the sprint's terrain-height rays): its readers, the
    # foot_clearance and foot_swing_height rewards and the critic's foot_height,
    # are dropped below.

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()

    # Robot setup: the standup robot, the Arena's collision set (D7): a duck
    # dragged towards the pit touches the ground with its body, not only its feet.
    # The learner and the opponent (D4), the opponent on the Arena's battery (D16);
    # every sensor, term and event below names `robot` but the opponent's action
    # term (D16) and its two resets (D12).
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG, "opponent": OPPONENT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # The course and the string are the Arena's tug, built on the scene by tug_scene
    # (D1–D4): no mjlab terrain has a pit, and without one every env origin is 0, so
    # each world holds the course and the string ends at the Arena's coordinates. Its
    # boxes come in 2 m tiles, which keep the world finite (D18).
    cfg.scene.terrain = None
    cfg.scene.spec_fn = tug_scene

    # The Arena's solver (D7, arena/course.py): implicitfast, 10/20 iterations,
    # 0.005 s physics steps, 4 per 0.02 s control step. nconmax and njmax are
    # buffers, not physics: with the string, a whole world (both ducks, the
    # string, the course) peaks at 31 to 40 contacts (42 in the Arena), under
    # 200, and at 196 constraint rows, under 1500 (the rope study). Every world's
    # CCD scratch is sized from nconmax: 400 asked 4.4 GB more of the L4 at 4096
    # envs (t8, t9).
    cfg.sim.nconmax = 200
    cfg.sim.njmax = 1500
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20
    cfg.decimation = 4
    ctrl_dt = cfg.sim.mujoco.timestep * cfg.decimation

    # An episode is the Arena's settle plus a bout (D11).
    cfg.episode_length_s = SETTLE_S + BOUT_S

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # The opponent, the reference duck frozen (D16), on the event's command. After
    # joint_pos: its agent dimension is 0, so the learner's 14 numbers are the action.
    cfg.actions["opponent"] = OpponentActionCfg(command=COMMAND)

    # === REWARDS ===
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = std_standing  # tight when command=0
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    # Pose reward operates on LEG joints only. Head/neck are command-driven
    # (head_pose_tracking) — if they were in this reward too, it would pull
    # them to HOME while head_pose_tracking pulls them to the command, and the
    # policy converges to "ignore the command" because pose reward dominates
    # once head_pose_tracking's gradient dies at large commands.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Body-specific reward configurations
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # upright: std 0.5, far looser than the sprint's √0.05. Leaning back is how a
    # duck holds a pull, so a lean must stay cheap; the training's fall
    # termination (ArenaFall, or `down` with stay_upright, below) is the hard line.
    cfg.rewards["upright"].weight = knobs["upright_reward"]
    cfg.rewards["upright"].params["std"] = 0.5

    # Foot-specific configuration: foot_slip's asset_cfg site_names select the feet.
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

    # Body-specific configurations
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip weak (-0.1): under a pull a planted foot may slide,
    # and keeping its feet under it matters more than keeping them still.
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    # Dropped: foot_clearance and foot_swing_height (the sprint's 2 cm foot-lift
    # targets) and soft_landing (the sprint drops it too). A duck holding a pull
    # plants its feet and shuffles backwards; air_time alone keeps it stepping.
    for name in ("foot_clearance", "foot_swing_height", "soft_landing"):
        del cfg.rewards[name]

    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # air_time window [0.125, 0.300] s, a fixed 1.0 (no knob): it keeps the duck
    # stepping backwards. The tug's command is never zero, so every env steps.
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # Velocity tracking rewards: the twist is walking backwards at the event's
    # speed (D6), the knob a novice tunes as "Walk backwards". std √0.05, the
    # sprint's √0.1 halved (D15): standing still keeps e^-1.8 = 0.17 of what
    # walking back at -0.30 m/s earns, not e^-0.9 = 0.41.
    cfg.rewards["track_linear_velocity"].weight = knobs["back_reward"]
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.05)
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # Pull straight. The Arena commands yaw 0 and never tells the policy its
    # heading, and nothing above prices a slow drift: at this std a 0.07 rad/s
    # turn keeps 99% of track_angular_velocity. A duck that turns away from the
    # string's line pulls across it and walks into the ring's ropes.
    # heading_hold prices the yaw ANGLE against the spawn heading (the roller's
    # term and std): the policy cannot see its heading, but it can learn to
    # null its gait's mean yaw rate. The critic sees the heading error
    # (heading_error, below) so it can value what this term pays.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )

    # The new ground the marker wins towards the learner (D14), the knob a novice
    # tunes as "Win ground". Only ground west of the episode's best counts, so
    # swaying earns nothing. Scale 0.05 m/s: a marker won at 5 cm/s pays tanh(1),
    # 0.76 of it, each step; backing at the event's 0.30 m/s against a still
    # opponent moves the marker at 0.15 m/s, tanh(3).
    cfg.rewards["marker_ground"] = RewardTermCfg(
        func=MarkerGround,
        weight=knobs["pull_reward"],
        params={"scale": 0.05},
    )
    # The ground kept between the string end and the pit's west edge; the scale
    # is the margin the duck starts with (0.35 m), so giving ground costs from
    # the first step.
    cfg.rewards["pit_margin_reward"] = RewardTermCfg(
        func=pit_margin_reward,
        weight=1.0,
        params={"offset": STRING_OFFSET, "pit_edge_x": -PIT_M / 2, "scale": 0.35},
    )
    # Either pit ends the episode (terminations below), as it ends a bout on the
    # Arena (D9): the learner's costs `out`, and so does its training-only fall
    # unless stay_upright is on, where going down pays its own `down` (D22); the
    # opponent's pays `win`. Both in on one step pay nothing, the rarest end.
    cfg.rewards["out"] = RewardTermCfg(func=ended_by, weight=-50.0, params={"terms": ("in_pit",) if stay_upright else ("in_pit", "fell")})
    if stay_upright:
        cfg.rewards["down"] = RewardTermCfg(func=ended_by, weight=-50.0 * stay_upright, params={"terms": ("down",)})
    cfg.rewards["win"] = RewardTermCfg(func=ended_by, weight=50.0, params={"terms": ("opponent_in_pit",)})

    # Action smoothness: stage-0 value; the action_rate_weight curriculum below
    # ramps it -0.1 → -1.0 by iter 1500.
    cfg.rewards["action_rate_l2"].weight = -0.1

    # NOTE: no neck-only action-rate term — the shared action_rate_l2 sums over
    # ALL action dims (neck included), and head_pose_tracking below gives the
    # 4 neck/head DOFs a position objective, so the neck is fully shaped.

    # === TERMINATIONS ===
    # The Arena's one end of a bout (D9), a pit, either duck's, plus the
    # learner's training-only fall, the time limit and NaN physics on either
    # duck. A pit: the trunk below -PIT_DROP_M, as arena.tug.in_pit judges it.
    # The template's instantaneous fell_over gives way to the Arena's own fall
    # rule (ArenaFall, below).
    # out_of_terrain_bounds reads mjlab's terrain, and there is none (D1).
    del cfg.terminations["fell_over"], cfg.terminations["out_of_terrain_bounds"]
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    # The same for the opponent, whose state the critic reads (D13); it pays
    # nothing, like nan_state.
    cfg.terminations["opponent_nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("opponent")},
    )
    cfg.terminations["in_pit"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -PIT_DROP_M},
    )
    cfg.terminations["opponent_in_pit"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -PIT_DROP_M, "asset_cfg": SceneEntityCfg("opponent")},
    )
    # TRAINING ONLY (D9, D22): going down ends a training episode so the duck
    # keeps learning to stand and pull; the Arena judges only the pit. The
    # Arena's fall rule (gravity z above FALL_GRAVITY_Z for FALL_HOLD_S of
    # straight control steps), or, with stay_upright, walking upright (D22): its
    # bands, read off the head and the neck, and no body but the feet on the course.
    hold_steps = round(FALL_HOLD_S / ctrl_dt)
    if stay_upright:
        cfg.scene.sensors += (ContactSensorCfg(
            name=BODY_GROUND,
            primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=FEET),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),)
        cfg.terminations["down"] = TerminationTermCfg(
            func=Down,
            params={"sensor_name": BODY_GROUND, "hold_steps": hold_steps,
                    "asset_cfg": SceneEntityCfg("robot", body_names=(HEAD,), joint_names=NECK)},
        )
    else:
        cfg.terminations["fell"] = TerminationTermCfg(
            func=ArenaFall,
            params={"fall_gravity_z": FALL_GRAVITY_Z, "hold_steps": hold_steps},
        )

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

    # The starts are the Arena's (arena.course.tug_layout, D12): the west duck
    # faces east, the east duck west, each with its string end on y = 0, the two
    # STRING_M apart and centred on the pit. The offsets only ever bring them closer
    # (west x up to +2 cm, east x down to -2 cm), the ±1 cm and the Arena's yaw
    # perturbation instead of the template's ±0.5 m and ±π, which would start ducks
    # over the pit or facing away from each other. The joints get the Arena's
    # perturbation.
    west_x, west_y = west_start()
    east_x, east_y = east_start()
    yaw = math.radians(YAW_PERTURBATION_DEG)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (west_x, west_x + 0.02),
        "y": (west_y - 0.01, west_y + 0.01),
        "z": (0.12, 0.13),
        "yaw": (-yaw, yaw),
    }
    cfg.events["reset_opponent_base"] = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (east_x - 0.02, east_x),
                "y": (east_y - 0.01, east_y + 0.01),
                "z": (0.12, 0.13),
                "yaw": (math.pi - yaw, math.pi + yaw),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("opponent"),
        },
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (
        -JOINT_PERTURBATION_RAD,
        JOINT_PERTURBATION_RAD,
    )
    cfg.events["reset_opponent_joints"] = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-JOINT_PERTURBATION_RAD, JOINT_PERTURBATION_RAD),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("opponent", joint_names=(".*",)),
        },
    )
    # The string, straight between both ducks as their resets above placed them (D21).
    cfg.events["reset_string"] = EventTermCfg(func=reset_string, mode="reset")

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

    # Observations
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # foot_height reads foot_height_scan, which the tug does not build.
    del cfg.observations["critic"].terms["foot_height"]

    # Add base_lin_vel to critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # Privileged too: the heading error heading_hold prices.
    cfg.observations["critic"].terms["heading_error"] = ObservationTermCfg(func=heading_error)
    # And what the tug's rewards pay on (D13): the string end's margin to the
    # pit, the opponent's trunk, the string's length and the marker. The actor
    # reads none of it: on the Arena it gets the 61 numbers every legs policy gets.
    cfg.observations["critic"].terms["pit_margin"] = ObservationTermCfg(
        func=pit_margin,
        params={"offset": STRING_OFFSET, "pit_edge_x": -PIT_M / 2},
    )
    cfg.observations["critic"].terms["opponent"] = ObservationTermCfg(func=opponent_state)
    for name, func in (("string_length", string_length), ("marker_x", marker_x)):
        cfg.observations["critic"].terms[name] = ObservationTermCfg(func=func)

    # Projected gravity, the slot the Arena feeds the policy.
    cfg.observations["actor"].terms["projected_gravity"] = deepcopy(
        cfg.observations["actor"].terms["projected_gravity"]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read contact sensor data,
    # which MuJoCo can return non-finite for while the state is still clean). A
    # single NaN here kills the whole run via rsl_rl's check_nan — that is the
    # 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so sanitizing costs
    # the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms["projected_gravity"]
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

    # Commands. The twist is the event's own command (arena/events/tug-of-war.toml
    # [command], D6): walk backwards at COMMAND_VX, never sideways or turning, in
    # every env. No standing envs (the Arena never sends a zero twist here), and
    # none of the template's forward envs, which would turn it into +0.30 m/s.
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_forward_envs = 0.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (COMMAND_VX, COMMAND_VX)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # Head pose command (4D deltas from HOME, in joint order:
    #   neck_pitch, head_pitch, head_yaw, head_roll). Tracked by
    # head_pose_tracking, added below. Ranges are small non-zero
    # (not zero) so the input neurons stay alive, and held there for the whole
    # run — the Arena commands no head motion, so there is no widening
    # curriculum here (the head stays still).
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — much smaller mechanical range)
        ),
    )
    # Body pose command (6D delta from nominal standing: [x, y, z, roll, pitch, yaw]).
    # It fills 6 of the actor's 61 numbers, where the Arena sends zeros. Small
    # non-zero ranges keep those input neurons alive; nothing rewards tracking it.
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
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

    # === Pose tracking rewards ===
    # head_pose: hold the head on its command (the Arena's is zero).
    # std=0.5 with per-joint Gaussian (see head_pose_tracking in mdp.py). Final
    # reward is the mean over 4 joints, so partial tracking is partial reward
    # (no all-or-nothing).
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # Head droop fix (2026-08-20). The head walks pitched ~15° down (measured:
    # run ww1g2198 head_pose_tracking 1.544/2.0 → 14.6° mean joint error).
    # DO NOT fix this by tightening head_pose_tracking's std: run 5yay13u4 tried
    # fine_std=0.1 and the policy stopped walking entirely by iter 300 (air_time
    # 1.01 → 0.02, peak foot height 15 mm → 2 mm, entropy collapsed 10.9 → 1.9).
    # An instantaneous tight tolerance taxes walking 0.77/step — 76% of the whole
    # air_time reward — and is UNESCAPABLE, since a 280 g head (38% of robot
    # mass) must oscillate while stepping. Standing still scored higher, so it
    # stood still.
    # The DC bias, unlike the oscillation, IS escapable (bias the neck command up
    # to cancel gravity sag), so price only that: L1 on a 1 s EMA of the error.
    # At the optimum this costs a walking policy nothing.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by the head_pose_bias_weight curriculum below
        params={"command_name": "head_pose", "tau_s": 1.0},
    )

    # action_rate weight ramp: gentle smoothing while the gait bootstraps, then
    # tighten to -1.0 by iter 1500.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.2},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -0.4},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.6},
                {"step": 1250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # NOTE: no standing_envs curriculum (the sprint's): the tug never commands a
    # standstill (D6). No velocity-command-range curriculum either — the twist is
    # fixed (see the command section above). Same for head_pose: no range
    # curriculum — the head stays at its small initial range for the whole run
    # (see the head_pose command section above).

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
                    {"step": 0, "range": 0.003},
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
                    {"step": 0, "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
                ],
            },
        )

    # Disable default curriculum
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # head_pose_bias ramp: OFF until iter 600, then 1.0 → 3.0 by iter 1500.
    # Held at 0 early because a posture-precision term is a distraction before
    # a gait exists. At weight 3.0 a 15° residual bias costs 0.79/step and a
    # 2° bias costs 0.10/step.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},
            ],
        },
    )

    return cfg


RL_CFG = RslRlOnPolicyRunnerCfg(
    # The learner starts as the reference duck (D19): ReferenceActor loads its layers and normalizer
    # and starts its action noise at WARM_START_STD; init_std is MLPModel's, a start from scratch.
    actor=RslRlModelCfg(
        class_name="microduck_challenges.tug_of_war.opponent:ReferenceActor",
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
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
    ),
    wandb_project="mjlab_microduck",
    experiment_name="tug_of_war",  # logs/rsl_rl/tug_of_war/
    run_name="tug_of_war",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=50_000,
)
