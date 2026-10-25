# Hill Climb: bootstrap first, refine only verified Hill policies

This transfers Sprint's **software/selection discipline**, not a demonstrated
faster hill gait. No compatible trained Hill checkpoint was available when this
was implemented. Sprint checkpoints, mature reward endpoints and its unproven
C1 combination are **not** Hill starting points.

## Two distinct entry points

| | Bootstrap (default Workshop / `train`) | Refinement explore | Refinement exploit |
|---|---:|---:|---:|
| Start | Random weights | Explicit compatible Hill `.pt`, official 5/5 | Exact candidate approved by paired assessment |
| New updates (maximum) | 3000 | 500 | 500 |
| Environments | 4096 | 2048 | 2048 |
| Save interval | 250 | 50 | 50 |
| Learning rate | 0.001, adaptive | 0.00005, fixed | 0.00001, fixed |
| Entropy / PPO clip | 0.01 / 0.2 | 0.01 / 0.2 | 0.001 / 0.1 |
| Noise | Original initialization | Restored noise × relative scale 1 | Restored noise × relative scale 0.5 |
| Wrapper training wall cap | 180 minutes | 40 minutes | 40 minutes |

One invocation is one bounded run; there is **no automatic switch, retry,
promotion, incumbent replacement or publication**. `plan` and `run` without
`--execute` only print a plan. Planning is not authorization to train.

Bootstrap retains the original hill rewards, beginner curricula and domain
randomization. Generated heightfield terrain, 8–15° slope knob (default 10°),
30/50 solver, heading_hold, critic-only heading_error, robot/BAM actuators,
14 un-clipped actions and ordered 61D actor / 77D critic remain intact. Commands
still cover the complete straight 0.15–1.00 m/s sweep; head/body command slots
remain present. There are no new reward formulas, broad physics changes, course
speed terms, lean relaxations or leg-freedom changes in this adaptation.

Refinement rejects missing or foreign checkpoints instead of cold-starting. It
checks the literal filename, Hill task/provenance, ordered layouts, full robot,
physics/terrain/reward/curriculum configuration, actor/critic architectures,
both normalizers and all state tensor shapes/finiteness, saved profile/noise
markers, runner iteration and environment age. Loads inspect tensors on CPU.
Only known runtime terrain hydration (world count and generated sub-patch size)
is normalized; arbitrary terrain changes are rejected.

Refinement loads the complete actor/critic and normalizers, starts fresh Adam
and runner numbering, and retains the real environment clock. Its curricula are
frozen at the source's age-appropriate stages, **not blindly at mature endpoints**.
The frozen curriculum clock is recorded separately and inherited unchanged by
refinement descendants; installed strict `step > threshold` boundaries are
preserved. Phase noise scaling applies to restored learned Gaussian deviations,
not mean actions, `init_std` or mechanical limits, and does not compound when
continuing the same scale. `MICRODUCK_WARM_START`'s upstream clock reset is refused.

## Independent, explicitly Hill-local reward hypotheses

A0 is the safe default for all entry points. Experiments require strict refinement:

- **A0:** unchanged Hill reward recipe, frozen at the source stage.
- **A1:** only `track_linear_velocity` weight → 0.
- **B1:** only `air_time` weight → 1.5 (Hill control is 3). It does **not** include A1.

These are hypotheses, not proven gains. B1's Hill coefficient deliberately differs
from Sprint's namesake; unsupported C1/B2 are refused. Source recipe/profile is
validated independently **before** applying a target override, and matching
curriculum endpoints are frozen. Exploitation must retain the approved source's
profile; a changed reward requires new exploration and assessment. Larger training
return is not evidence of a faster legal hill finish. Episode reward terms and
last-100 completed episode returns use different accounting/windows; inspect the
logged reward mass, not just coefficients, before proposing more ablations.

## Commands

Run at the repository root. The separate, currently edited Sprint recipe can
require archived Sprint files during package registration; the portable commands
below explicitly keep that unrelated task in baseline mode. Hill's own default
does not search for archives.

```bash
export MICRODUCK_SPRINT_STAGE=baseline

# Non-launching; original bootstrap optimization, not a 500-update refinement.
uv run --group workshop python -m microduck_challenges.hill_climb_1m5.campaign plan

# Required before ANY training, including a CPU smoke.
uv run --group workshop pytest tests/test_hill_lite.py
uv run --group workshop pytest

# Exactly two CPU environments and two 24-step updates; explicit seed, fresh path.
# Tests must pass first. The ONNX includes normalization and checks finite parity.
uv run --group workshop python -m microduck_challenges.hill_climb_1m5.campaign smoke \
  --seed 4 --output /absolute/new/output-directory

# Later, only after a REAL compatible Hill checkpoint has a consistent 5/5 result:
uv run --group workshop python -m microduck_challenges.hill_climb_1m5.campaign plan \
  --stage refine --phase explore --source /absolute/Hill/run/model_N.pt --reward-profile A0

# Optional refinement software smoke still requires the genuine source above.
uv run --group workshop python -m microduck_challenges.hill_climb_1m5.campaign smoke \
  --stage refine --source /absolute/Hill/run/model_N.pt --seed 4 --output /absolute/new/refine-smoke

# After Workshop has judged EVERY checkpoint, freeze a command BEFORE assessment.
# --source is the retained incumbent to beat, not a candidate selected on held-outs.
uv run --group workshop python -m microduck_challenges.hill_climb_1m5.campaign assess \
  --run logs/rsl_rl/hill_climb_1m5/COMPLETE_RUN \
  --source /absolute/Hill/incumbent/model_N.pt --command 0.55 \
  --output /absolute/new/assessment-directory

# A passing assessment enables planning this EXACT candidate/profile, not publishing:
uv run --group workshop python -m microduck_challenges.hill_climb_1m5.campaign plan \
  --stage refine --phase exploit --promotion /absolute/assessment-directory/assessment.json
```

`run ... --output NEW_DIRECTORY --execute` is the separately authorized launch
form. It reruns tests and CPU smoke under an exclusive local launch lock, refuses
concurrent recognized trainers, then uses the recognized `mjlab.scripts.train`
entry point on GPU 0 so installed provenance recording works. The watchdog
terminates/reaps only its own child process group on budget, termination signals
or post-spawn errors. Source/config hashes are checked before and after training.
It does not submit paid jobs, stop pre-existing jobs, commit, push or publish.
Raw Workshop/`train` uses the bootstrap recipe but does **not** inherit the
wrapper's watchdog/lock; arbitrary CLI overrides are not a bounded campaign.

## Assessment gates

1. Reconcile **every** expected checkpoint using the saved budget/cadence,
   including checkpoint 0 and the final update; require a consistent Workshop
   result for every file. Missing/corrupt/contradictory results block assessment.
2. Freeze one valid sweep command explicitly before any fresh assessment races.
   Export each actor to a new directory outside source runs/Workshop caches and
   check finite CPU ONNX/Torch parity, `[1,61] → [1,14]`, and the 64 MiB limit.
3. Screen every checkpoint on seeds 1–5. Require three adjacent entries in the
   **unfiltered complete inventory** to each finish 5/5. Never skip failed neighbors.
4. Select exactly the fastest development mean in a stable window; persist its
   identity/hashes **before opening held-outs**. A failed candidate does not trigger
   testing another candidate against those same held-outs.
5. Compare that candidate and the explicitly retained incumbent at the same command
   on paired seeds 6–25. Require candidate 20/20, strictly faster development mean,
   and held-out failure-penalized mean no worse than incumbent. Every fall/timeout
   counts **60 seconds**, never a dropped row.
6. Recompute approval from complete raw rows, bind reports to their persisted JSONL,
   exports/checkpoints/config hashes, event/evaluator/version signature and unchanged
   inventory. An `approved=true` field by itself authorizes nothing.

Held-outs can lose independence through repeated reuse. Passing this software
gate is neither a statistical guarantee, an Arena leaderboard result nor physical
robot validation. A first bootstrap has no incumbent yet: inspect all its official
scores and explicitly retain an initial compatible 5/5 policy; it does not
self-promote. Assessment/refinement remain blocked if such evidence is absent.

## Verification and deliberate limits

Real bootstrap CPU training/export evidence is under
`logs/verification/hill-strategy-20261024/`: two environments/two updates, seed 4,
61D actor / 77D critic, 49 live reward checks, finite TensorBoard values and eight
CPU ONNX parity cases. Unit tests use generated local fixtures, not invented
training/evaluation results; they cover strict restore, independent profiles,
source/layout/clock rejection, score integrity, adjacency, locks and owned-child
cleanup. Shared tests and the seven pre-existing dirty Sprint files are preserved.

There was no genuine reliable Hill source, so no end-to-end trained refinement,
real paired Arena assessment, long/GPU training or racing improvement is claimed.
The untrained smoke checkpoint is **not** installed as an incumbent or given
fabricated 5/5 evidence. Bootstrap must first learn a reliable hill gait.
