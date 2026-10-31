"""Why did each Sock Heist seed end?

    uv run --frozen --group workshop python -m microduck_challenges.sock_heist_6m.check policy.onnx [--speed V] [--seeds N]

Replays the policy on the Arena's own physics (arena.runner.run_once) over seeds 1 to N (default 5, the
event's), at V m/s or, by default, at the command `arena score` scores it at, and prints a line per seed
(caught still moving or creeping, a fall on a stop, a start or green, out of time, a duck that never left
the line, with the reds met, the worst drift from a mark and the speed on green), then the tally.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from arena.constants import CTRL_DT
from arena.course import compile_course
from arena.evaluate import evaluate
from arena.lights import Red, timetable
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.runner import RunOutcome, run_once

EVENT = "sock-heist-6m"
CAUGHT_EARLY_S = 0.2  # caught within this of the watch's start: still moving when the head finished turning
STOP_WINDOW_S = 0.6  # a fall tipping within this (inclusive) of a red's start is on a stop, of its end on a start
LEFT_THE_LINE_M = 0.5  # a duck that ended nearer its line than this never left it


def _steps(seconds: float) -> int:
    return round(seconds / CTRL_DT)


def judge(outcome: RunOutcome, reds: tuple[Red, ...], turn: int, fall_hold: int) -> tuple[str, str]:
    """(kind, line): why the run ended. Step i of the trace ends at (i + 1) * CTRL_DT; k is the last timed
    step, the catch's for a caught duck, whose trace runs on past it."""
    k = outcome.caught_frame if outcome.status == "caught" else len(outcome.trace) - 1
    x, xy = outcome.trace[:, 0].astype(np.float64), outcome.trace[:, :2].astype(np.float64)
    start = float(outcome.settle[-1, 0])
    t = (k + 1) * CTRL_DT

    in_red = sum(max(0, min(r.onset + r.length, k + 1) - r.onset) for r in reds)
    green_speed = (x[k] - start) / ((k + 1 - in_red) * CTRL_DT)
    drift = max(
        (np.hypot(*(xy[j] - xy[r.onset + turn - 1])) for r in reds if r.onset + turn - 1 < k
         for j in range(r.onset + turn, min(r.onset + r.length, k))),
        default=0.0,
    )
    met = sum(r.onset <= k for r in reds)
    tail = f"{met} red{'' if met == 1 else 's'} met · worst drift {drift * 100:.1f} cm · {green_speed:.2f} m/s on green"

    if outcome.status == "finish":
        return "finish", f"finish {outcome.time_s:.2f} s · {tail}"
    if outcome.status == "caught":
        red = max((r for r in reds if r.onset <= k), key=lambda r: r.onset)
        into = k - (red.onset + turn)
        p = len(outcome.settle) + red.onset - 1  # the frame the red's first step starts from, settle included
        before = np.concatenate([outcome.settle[:, 0], x]).astype(np.float64)
        speed = (before[p] - before[p - 5]) / (5 * CTRL_DT)
        kind = "caught early" if into <= _steps(CAUGHT_EARLY_S) else "caught late"
        return kind, f"caught at {t:.2f} s, {into * CTRL_DT:.2f} s into the watch, {speed:.2f} m/s as the red began · {tail}"
    if outcome.status == "fall":
        tip, window = k - fall_hold + 1, _steps(STOP_WINDOW_S)
        for r in reds:
            if r.onset <= tip <= r.onset + window:
                return "fell on a stop", f"fell at {t:.2f} s on a stop, tipping {(tip - r.onset) * CTRL_DT:.2f} s after a red began · {tail}"
            end = r.onset + r.length
            if end <= tip <= end + window:
                return "fell on a start", f"fell at {t:.2f} s on a start, tipping {(tip - end) * CTRL_DT:.2f} s after a red ended · {tail}"
        under = any(r.onset <= tip < r.onset + r.length for r in reds)
        where = "under a red" if under else "on green"
        return f"fell {where}", f"fell at {t:.2f} s {where} · {tail}"
    progress = x[k] - start
    if progress < LEFT_THE_LINE_M:
        return "never left the line", f"never left the line ({progress:.2f} m) · {tail}"
    return "out of time", f"out of time at {progress:.2f} m · {tail}"


def tally(kinds: list[str], times: list[float]) -> str:
    """The seeds' kinds in one line; `times` holds an unfinished seed at the time limit, as the Arena scores."""
    count = kinds.count
    caught, fell = [k for k in kinds if k.startswith("caught")], [k for k in kinds if k.startswith("fell")]
    return (
        f"finished {count('finish')}/{len(kinds)} · caught {len(caught)} (early {count('caught early')})"
        f" · fell {len(fell)} (on a stop {count('fell on a stop')}, on a start {count('fell on a start')})"
        f" · out of time {count('out of time') + count('never left the line')} (never left the line {count('never left the line')})"
        f" · mean {sum(times) / len(times):.2f} s"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Say why each Sock Heist seed of a policy ended.")
    parser.add_argument("policy")
    parser.add_argument("--speed", type=float, help="the commanded speed, m/s (default: the one the Arena scores it at)")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(exc, file=sys.stderr)
        return 1
    manifest = load_event(EVENT)
    speed = args.speed
    if speed is None:
        speed = evaluate(manifest, policy).command_vx
        print(f"scored at {speed:.2f} m/s")
    model = compile_course(manifest)
    steps, turn = _steps(manifest.time_limit_s), _steps(manifest.lights.turn_s)
    fall_hold = _steps(manifest.termination.fall_hold_s)
    kinds, times = [], []
    for seed in range(1, args.seeds + 1):
        outcome = run_once(manifest, policy, seed, model, command=manifest.command_at(speed))
        kind, line = judge(outcome, timetable(manifest.lights, manifest.id, seed, steps), turn, fall_hold)
        print(f"seed {seed} · {line}")
        kinds.append(kind)
        times.append(outcome.time_s if outcome.status == "finish" else manifest.time_limit_s)
    print(tally(kinds, times))
    return 0


if __name__ == "__main__":
    sys.exit(main())
