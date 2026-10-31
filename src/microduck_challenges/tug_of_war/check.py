"""Which of two Tug of War policies drags the other into the pit?

    uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx [B.onnx] [--bouts 10]

Plays bouts between the two (B defaults to the reference duck) on the Arena's own physics
(arena.tug.run_bout), bout k on seed k, sides swapped after each (A starts on the west side),
and prints a line per bout (who lost, how A pulled and how it walked), then the tally.
"""

import argparse
import itertools
import statistics
import sys
from pathlib import Path

import mujoco
import numpy as np
from arena.constants import CTRL_DT, DEFAULT_POSE, JOINT_NAMES
from arena.course import compile_tug, duck_prefix
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.tug import BoutOutcome, marker_x, run_bout

from .env import PIT_M
from .opponent import REFERENCE
from .rewards import FEET, HEAD, NECK, is_upright

EVENT = "tug-of-war"
SIDES = ("west", "east")
TAUT_M = 0.98  # the ends this far apart: taut; the string goes taut when the other duck stands and resists, and stays short of it when that duck lies down or gives way
BACKED_S = 2.0  # where A's retreat is read
LIP_M = 0.08  # A's frames with its trunk this near the pit's lip are left out of the posture reading
STEP_AIR_FRAMES = 2  # a clear step: a foot back in contact after at least this many frames out of it...
STEP_LIFT_M = 0.005  # ...during which its site rose this far above its height on its last frame in contact
STILL_CM_S = 1.0  # both feet travelling less than this on average: the duck stands still


def play(manifest, model, a, b, bouts: int) -> list[tuple[str, BoutOutcome]]:
    """(A's side, outcome) per bout on `model`, the event's tug compiled once: bout k on seed k + 1,
    A west when k is even."""
    results = []
    for k in range(bouts):
        side = SIDES[k % 2]
        west, east = (a, b) if side == "west" else (b, a)
        results.append((side, run_bout(manifest, west, east, k + 1, model=model)))
    return results


def pull(outcome: BoutOutcome, side: str) -> str:
    """How A pulled: when the string's ends first stood TAUT_M apart, how far A's end of the string
    (its first point west, its last east) moved away from the pit from the settle's last frame to
    BACKED_S, and where the marker ended, in cm towards A."""
    towards_a = 1.0 if side == "west" else -1.0
    end = 0 if side == "west" else -1
    string = outcome.string
    taut = np.flatnonzero(np.linalg.norm(string[:, -1] - string[:, 0], axis=1) >= TAUT_M)
    taut_at = f"taut at {(taut[0] + 1) * CTRL_DT:.2f} s" if taut.size else "never taut"
    start = outcome.string_settle[-1, end, 0]
    at = min(round(BACKED_S / CTRL_DT), len(string)) - 1
    backed = -towards_a * (string[at, end, 0] - start) * 100
    marker = -towards_a * marker_x(string[-1:])[0] * 100
    return f"{taut_at} · A backed {round(backed, 1) + 0.0:.1f} cm by {BACKED_S:.0f} s · marker {round(marker, 1) + 0.0:.1f} cm to A"


def _qpos(model, prefix: str) -> list[int]:
    """A duck's qpos addresses in a trace frame's order: its trunk's free joint, then JOINT_NAMES."""
    free = model.jnt_qposadr[model.body(prefix + "trunk_base").jntadr[0]]
    return [*range(free, free + 7), *(model.joint(prefix + name).qposadr[0] for name in JOINT_NAMES)]


def gait(contact: np.ndarray, feet: np.ndarray, counted: np.ndarray) -> str:
    """How the duck stepped, from the pairs of consecutive counted frames: `contact` is (n, 2) bool (left, right
    foot on the course), `feet` the foot sites (n, 2, 3). Steps a second is the feet's clear steps (STEP_AIR_FRAMES,
    STEP_LIFT_M; a foot starting in the air has no clear first step) over the counted time; slide is the share of
    the feet's horizontal travel between two frames both in contact."""
    pair = counted[1:] & counted[:-1]
    seconds = counted.sum() * CTRL_DT
    travel = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) * pair[:, None]
    if travel.sum() <= STILL_CM_S / 100 * seconds:
        return "standing still"
    slid = (travel * (contact[1:] & contact[:-1])).sum()
    steps = 0
    for foot in range(2):
        on, height = contact[:, foot], feet[:, foot, 2]
        left = None
        for i in range(1, len(on)):
            if not pair[i - 1]:
                left = None
            elif on[i - 1] and not on[i]:
                left = i
            elif on[i] and not on[i - 1] and left is not None:
                steps += i - left >= STEP_AIR_FRAMES and height[left:i].max() - height[left - 1] >= STEP_LIFT_M
    return f"{steps / seconds:.1f} steps/s · slide {100 * slid / travel.sum():.0f} %"


def posture(outcome: BoutOutcome, side: str, model) -> str:
    """How A walked, over its frames whose trunk is more than half the pit and LIP_M from the centre
    line: the share upright (rewards.is_upright, the training's rule, and only its feet on the course),
    its longest straight run out, and the share with only its feet on the course. Each frame poses
    both ducks from the trace in `model`, the event's tug, the string left at rest, and recomputes
    their contacts: the head's height is its centre of mass's, the gravity is the trace quaternion's."""
    a = SIDES.index(side)
    prefix, trace = duck_prefix(a), outcome.trace.astype(np.float64)
    ducks = [_qpos(model, duck_prefix(k)) for k in (0, 1)]
    head, feet = model.body(prefix + HEAD).id, [model.body(prefix + name).id for name in FEET]
    soles = [model.site(prefix + name).id for name in ("left_foot", "right_foot")]
    body = np.flatnonzero((model.body_rootid[model.geom_bodyid] == model.body(prefix + "trunk_base").id)
                          & ~np.isin(model.geom_bodyid, feet))
    shoes = [np.flatnonzero(model.geom_bodyid == foot) for foot in feet]
    course = np.flatnonzero(model.geom_bodyid == model.body("course").id)
    data = mujoco.MjData(model)
    frames = trace.shape[1]
    head_z, feet_only = np.empty(frames), np.empty(frames, dtype=bool)
    contact, sites = np.empty((frames, 2), dtype=bool), np.empty((frames, 2, 3))
    for f in range(frames):
        for k, qpos in enumerate(ducks):
            data.qpos[qpos] = trace[k, f]
        mujoco.mj_fwdPosition(model, data)
        head_z[f] = data.xipos[head, 2]
        pairs = data.contact.geom
        sites[f] = data.site_xpos[soles]
        on_course = np.isin(pairs[:, ::-1], course)   # a pair in either order: (ours, course) or (course, ours)
        contact[f] = [(np.isin(pairs, shoe) & on_course).any() for shoe in shoes]
        feet_only[f] = not (np.isin(pairs, body) & on_course).any()
    w, x, y, z = trace[a, :, 3:7].T
    gravity_b = -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    neck = [JOINT_NAMES.index(name) for name in NECK]
    upright = is_upright(trace[a, :, 2], gravity_b, head_z, trace[a, :, 7:][:, neck] - DEFAULT_POSE[neck]) & feet_only
    counted = np.abs(trace[a, :, 0]) > PIT_M / 2 + LIP_M
    runs = [len(list(run)) for out, run in itertools.groupby(counted & ~upright) if out]

    def share(kept):
        return 100 * (kept & counted).sum() / counted.sum() if counted.any() else 100.0

    return (f"upright {share(upright):.0f} % · longest out {max(runs, default=0) * CTRL_DT:.2f} s"
            f" · feet only {share(feet_only):.0f} % · {gait(contact, sites, counted)}")


def summary(results, model) -> list[str]:
    lines = []
    for k, (side, outcome) in enumerate(results, 1):
        if outcome.loser is None:
            verdict = "drawn"
        else:
            loser = "A" if SIDES[outcome.loser] == side else "B"
            verdict = f"{loser} loses by {outcome.how}"
        lines.append(
            f"bout {k} · seed {outcome.seed} · A {side} · {verdict} at {outcome.time_s:.2f} s"
            f" · {pull(outcome, side)} · {posture(outcome, side, model)}"
        )
    outcomes = [outcome for _, outcome in results]
    decided = [(side, o) for side, o in results if o.loser is not None]
    a_loses = sum(SIDES[o.loser] == side for side, o in decided)
    pits = [sum(o.how == "pit" and (SIDES[o.loser] == side) == a for side, o in results) for a in (True, False)]
    lines.append(
        f"A {len(decided) - a_loses} · B {a_loses} · draws {len(results) - len(decided)}"
        f" · A in the pit {pits[0]} · B in the pit {pits[1]} · by marker {sum(o.how == 'marker' for o in outcomes)}"
        f" · median decision {statistics.median(o.time_s for o in outcomes):.1f} s"
    )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play Tug of War bouts between two policies.")
    parser.add_argument("a")
    parser.add_argument("b", nargs="?", default=str(REFERENCE))
    parser.add_argument("--bouts", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        a, b = load_policy(Path(args.a)), load_policy(Path(args.b))
    except PolicyError as exc:
        print(exc, file=sys.stderr)
        return 1
    manifest = load_event(EVENT)
    model = compile_tug(manifest)
    print("\n".join(summary(play(manifest, model, a, b, args.bouts), model)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
