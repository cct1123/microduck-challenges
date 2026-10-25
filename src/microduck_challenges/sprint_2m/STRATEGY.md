# Sprint: discover freely, consolidate only verified gains

## Opt-in reward experiments and plateau audit

The next Workshop/train run defaults to **C4**, selected explicitly by
`training.reward_profile` in `challenge.toml`. Campaign plan/run/smoke use the
same default. The `--reward-profile` option selects one explicitly named recipe:

| Profile | Difference from the mature A0 recipe |
|---|---|
| `A0` | No reward override (control) |
| `A1` | `track_linear_velocity` weight -> 0 |
| `B1` | `air_time` weight -> 3; does NOT include A1 |
| `B2` | `speed_forward` weight -> 18; does NOT include A1 or B1 |
| `C1` | Explicit combination: tracking -> 0 AND air time -> 3; only air time differs from A1 |
| `C2` | C1 plus course-speed weight -> 18; only speed weight differs from C1 |
| `C3` | C2 plus `upright.motion_speed` -> 0.1 m/s; only the lean-speed parameter differs from C2 |
| `C4` | Same weights as C3; `upright.motion_speed` -> 0 explicitly disables all velocity gating |

These are hypotheses, not demonstrated speed gains. The C2 comparison starts
from the same retained model_8250 and keeps
C1's seed 2, 2,048 environments and 500-update budget. Course-speed weight
12 -> 18 is its ONLY reward change relative to C1: tracking remains 0, air time
remains 3, and heading, upright, safety, physics and PPO controls are unchanged.
B1/B2 keep their original independent semantics. Do not infer a gain from a
larger total reward. Source metadata and
checkpoint markers are validated against an independently built source recipe
BEFORE target overrides; applicable curriculum endpoints are frozen so resets
cannot undo the experiment. Saved agent configuration and `.pt` files record the
reward profile. A legacy checkpoint without a marker means A0. Promotion to
exploitation must retain the source's reward profile; changing it needs a new
exploration/assessment, not an old approval applied to a new reward.

### Earlier forward-lean tolerance (C3)

C3 changes only `upright.params.motion_speed`: 0.5 -> 0.1 m/s. At the mature
refinement clock, forward-pitch variance ramps linearly from its standing value
at zero actual course speed to the existing full variance 0.80 at 0.1 m/s,
instead of 0.5 m/s. This is full allowance BY 0.1, not a new hard switch ABOVE
0.1. The input is measured spawn-course velocity, not the commanded speed.
Stationary/reverse travel, roll and backward lean retain their previous
penalties. The upright optimum remains upright; forward lean is less costly,
not directly rewarded. Brief tilt cap 70 degrees and the independent held-fall
rule (>60 degrees for 0.3 seconds) remain unchanged.

Do NOT edit `training.drive.motion_speed` or the refinement construction scalar
in place: these reconstruct archived source recipes for strict validation.
C3's explicit parameter override is applied only after validating the source,
and its identity/parameter are checked in saved descendants and live smoke
steps. A0–C2 keep their old semantics. Logged weight-reweight estimates do not
rescore a changed lean parameter or predict its learned gait.

Prepare this change outside the live source while a Sprint trainer is active.
Apply the prepared patch only after that run exits and the protected source
hashes still match. Do not stop/restart the active run, change PPO or budgets,
promote a checkpoint, or launch another run as part of preparing C3.

### No velocity requirement for forward lean (C4)

C4 makes the full existing forward-pitch variance 0.80 available at every valid
velocity, including standing and reverse motion, at the mature refinement
clock. `motion_speed = 0` is an explicit no-gating mode, not a tiny threshold
or division by zero. Positive thresholds retain their original formulas and
all A0–C3 meanings. Upright still scores highest; leaning is tolerated, not
given a separate bonus. Roll/backward-pitch penalties remain unchanged.

The user explicitly retained the official fall limits: the 75-degree/0.5-second
proposal is NOT applied. The held-fall guard remains >60 degrees for 0.3 s;
the existing immediate 70-degree cutoff, finite-state checks, mechanical
limits, event/evaluator, observations and PPO controls remain unchanged.
Air time is unchanged: weight 3, current airborne time strictly between
0.125 and 0.300 s, commanded-motion norm >0.01, no actual-progress or landing
requirement. Looser lean may make idle leaning more attractive; this is an
unvalidated reward preference, not proof of faster or always-legal finishes.

Prepare and test C4 outside the live source while a trainer is running. Apply
only after it exits, protected source hashes still match and no replacement
trainer has started. Do not restart training or publish automatically.

The CLI is non-launching without the existing explicit `run --execute` switch:

```bash
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign plan --phase explore --reward-profile C4
```

`--reward-profile A0` selects the unchanged control. An explicit
`MICRODUCK_SPRINT_REWARD_PROFILE` overrides the TOML default for raw training
(refine stage only). Baseline/speed/drive diagnostic stages retain A0 unless an
invalid non-A0 override is supplied, which is refused. The campaign remains the
bounded, locked, provenance-recorded path. A running trainer retains its loaded
recipe; changing the file does not alter it. Startup prints the actual selected
profile and overrides. Commit the implementation before pressing Train (Workshop
refuses a dirty checkout), then refresh the page before starting the next run.

The completed `2026-10-24_17-22-58_run-1024-1722` was A0, not A1: its saved
tracking weight stayed 0.4927176. Its best Workshop result was 1.208 s at
checkpoint 0; the final checkpoint scored 1.272 s, with reliability regressions
at intermediate checkpoints. That run did not test A1.

The later A1 GPU run `2026-10-24_18-07-46_explore-A1-s2-20261024-180704`
completed all 500 updates with unchanged source files. All 11 checkpoints were
evaluated: best was model_0, 1.212 s and 5/5, while model_499 scored 11.008 s
with 4/5 finishes. The source is not replaced by the failed final model. Over
iterations 479–499, logged air-time contribution averaged 4.671 versus speed
6.542 (about 71%). This motivates testing lower air-time weight, not a claim
that air time is the proven cause or that C1 is already faster.

The completed C1 GPU run `2026-10-24_19-47-22_explore-C1-s2-20261024-194631`
also finished 500 updates with unchanged source files and 11/11 results. Best
reliable was model_50 (tied with model_150), 1.220 s, 5/5; final model_499 was
1.256 s, 5/5. Only 6/11 C1 checkpoints finished 5/5, versus 8/11 for A1. These
variable-command sweep scores do not establish improved reliability or speed.
Over iterations 479–499, C1's weighted air-time contribution was 2.263 versus
A1's 4.671, but weighted speed was essentially unchanged (6.508 versus 6.542).
Lower air-time weight did not produce a clear racing gain. C2 tests stronger
course-speed incentive without weakening safety terms or changing the speed
function/vel_ref. This dense 20-second objective is still not exact race-time
optimization; neither a larger reward nor C2's CPU smoke can validate the
hypothesis. Keep model_8250 and compare legal times and completion before any
promotion. No new GPU run is authorized by preparing these rewards.

Read existing logs without inference, rollout or training:

```bash
uv run --group workshop python -m microduck_challenges.sprint_2m.diagnostics \
  --run logs/rsl_rl/sprint_2m/2026-10-24_16-50-33_run-1024-1650 \
  --iteration 264 --radius 10 --output logs/verification/my-reward-audit
```

The audit requires the EXACT requested iteration, includes all saved reward
terms, reports missing data rather than inventing zeros, and writes JSON/Markdown
to a new directory outside the source run. `Episode_Reward` values use weighted
episode sums divided by maximum episode seconds, while `Train/mean_reward` uses
a rolling 100-completed-episode window. They cannot be added together to explain
the headline return. Saved startup weights are not assumed live for a changing
curriculum; affected terms are excluded from unweighting estimates. The optional
fixed-data weight estimates are not retrained behaviour or a full trajectory
counterfactual. One reward point is not evidence of a plateau.

### Slim post-development checks

The seven legacy Sprint-only test files are replaced by ONE
`tests/test_sprint_lite.py`, written after development. Shared/package/actuator
and other-challenge tests remain unchanged. The slim suite uses generated local
fixtures, no historical policy inventory, no simulator rollout and no GPU/network
requests. It covers profiles/curricula, strict restore, safety gates, ONNX parity,
score integrity and audit accounting. To avoid the training registry selecting an
archived warm-start source at import on a clean clone, set
`MICRODUCK_SPRINT_STAGE=baseline` for portable test collection.

```bash
uv run --group workshop pytest tests/test_sprint_lite.py
uv run --group workshop pytest
```

Before a long training, run the suite and one separate two-env/two-update CPU
smoke/export check in a fresh directory:

```bash
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign smoke \
  --phase explore --reward-profile C4 --output logs/verification/my-c4-smoke
```

This verifies software/restore/export only, not improved racing performance.
Finite-race objective, teacher imitation/DAgger, and model compression remain
later separately authorized stages, not automatically started by this feature.

## Goal and invariant

Target: a legal **2 m finish averaging 0.8 s**, without sacrificing reliable completion.
The current best-known exploration source is `2026-10-24_22-11-29_run-1024-2211/model_350.pt` (Workshop 1.180 s, 5/5, command 0.35 m/s). Keep model_8250 archived as a fallback. Before each improvement run, recheck completed compatible 5/5 candidates and explicitly pin the best known source; do not reset to the original source or replace it with a worse final checkpoint. These are development sweep results, not held-out certification or exploitation approval.

All stages retain the real robot, actuator limits, scale 1.0/no action clipping, ordered 61D actor and 14D actions, the source's 81D critic, heading guidance, head smoothing, and independent fall guards. Leg pose preference and leg action-rate costs remain off in **both** phases; conservatism comes from learning controls and selection, not locking the legs. The existing 20-second dense locomotion objective is unchanged. This is not yet an exact finite race-time objective.

## Two separate launches, not a blind iteration-based switch

The single recipe source is `challenge.toml` → `training.phases`.

| Control | Explore (default) | Exploit (requires approval) |
|---|---:|---:|
| New updates, maximum | 500 | 500 |
| Environments | 2048 | 2048 |
| Save every | 50 updates | 50 updates |
| Fixed learning rate | 0.00005 | 0.00001 |
| Entropy coefficient | 0.01 | 0.001 |
| PPO clipping | 0.2 | 0.1 |
| Restored training-noise scale | 1.0 | 0.5 |
| Wall-time limit per GPU run | 40 minutes | 40 minutes |

Both load the complete compatible actor/critic and normalizers, preserve environment age and mature physical curricula, and start fresh Adam and runner numbering. The noise scale changes the **restored learned Gaussian standard deviations**, not `init_std`, mean actions or mechanical range. Checkpoints record their scale: continuing the same phase does not halve/double noise again. Returning from exploitation to exploration restores the relative exploration scale. Noise scaling is an explicit phase-boundary change, not a within-run annealing schedule.

These are experimental controls, not a claim of faster learning or a stable trained gait. Smaller PPO clipping is not a hard KL bound. The fixed-LR path does not activate upstream KL-adaptive scheduling. Exported policies already use deterministic mean actions, so lower training noise alone cannot fix deterministic falls.

## Promotion gate

Freeze command **0.30 m/s before assessment**. Do not select a command using held-outs.

1. Wait for every checkpoint in the run's saved recipe to exist and for Workshop to judge all of them. Invalid/missing exports or contradictory seed totals block assessment.
2. Re-export each saved actor into a **new assessment directory**, using the installed MLP/normalizer exporter. Restore strictly and verify finite `[1,61] → [1,14]` ONNX/Torch parity. Never trust a Workshop export merely because its mtime matches.
3. Evaluate **every saved checkpoint** on seeds 1–5 at the frozen command. Require a window of **three adjacent entries in the complete checkpoint inventory**, each finishing 5/5. Filtering to passing checkpoints before finding neighbors is forbidden.
4. Choose exactly one candidate by the fastest development-seed mean within a stable window. Freeze its identity/hashes **before** opening held-outs. Do not try alternative candidates against the same held-outs after a failed gate.
5. Run candidate and explicitly retained incumbent on the same validation seeds **6–25**, at the same frozen command. Require candidate 20/20, a strictly faster seeds-1–5 mean, and a held-out failure-penalized mean no worse than the incumbent. Any failure counts **50 s**, never a dropped row.
6. Recheck complete checkpoint coverage, artifact hashes, event/evaluator files and runtime versions. Recompute approval from raw rows; a saved `approved=true` flag is not authority.

A rejected screen does not consume the additional validation starts. All assessment exports, raw seed rows and reports live outside Workshop's official cache. Approval is specific to the exact source and criteria. It enables an exploitation launch; it does **not** automatically publish, overwrite the incumbent, or authorize another round. After exploitation, assess again against the current retained checkpoint, explicitly passed with `--source`.

Seeds 6–25 are held out from the five-seed sweep, not guaranteed pristine across all historical experiments. Repeated reuse can overfit selection; use fresh confirmation starts and another training seed before calling an improvement robust or publishing it. Passing this gate is not a statistical guarantee.

## One CLI

Run from the repository root with the Workshop dependency group. The default action is non-launching unless `run --execute` is explicit.

```bash
# Inspect exact controls, source hashes, environment and parsed training command.
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign plan --phase explore

# Reproducible software checks: two CPU environments, two real updates and ONNX parity.
# Use a NEW output directory each time. Exploit smoke tests controls only; it does not
# bypass the approval requirement for a long exploitation run.
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign smoke --phase explore --output logs/verification/my-explore-smoke
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign smoke --phase exploit --output logs/verification/my-exploit-smoke

# Explicitly launch ONE local GPU screen. Preflight reruns tests and the CPU smoke.
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign run --phase explore --seed 2 --output logs/campaigns/my-explore --execute

# Once Workshop has judged every checkpoint: substitute the actual completed run.
# --source means the incumbent to BEAT, not an alternate candidate to cherry-pick.
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign assess --run logs/rsl_rl/sprint_2m/<completed-run> --source /absolute/path/to/retained/model_N.pt --output logs/verification/my-assessment

# Only a passing report permits this exact candidate's exploitation launch.
uv run --group workshop python -m microduck_challenges.sprint_2m.campaign run --phase exploit --promotion logs/verification/my-assessment/assessment.json --output logs/campaigns/my-exploit --execute
```

The bounded launcher refuses concurrent trainers, holds a launch lock, records its source/controls/process, checks source hashes, and terminates/reaps only its own process group on its wall budget, SIGINT/SIGTERM/SIGHUP, or a post-spawn error. It uses the recognized `mjlab.scripts.train` entry point so the runner records the provenance Workshop needs. Test locks are isolated from the operational lock, including during nested preflight. It neither submits paid jobs nor publishes, commits or pushes. Each invocation is one run: there is no unattended infinite campaign or automatic retry after failed validation. Keep Workshop open (`uv run --group workshop workshop sprint-2m`) to evaluate ordinary GPU checkpoints; assessment refuses missing official results rather than manufacturing them.

Normal Workshop/`train` defaults also select **explore**, 500 updates; refresh an already-open Workshop form so an old 2000-update UI value does not override the file. The 40-minute watchdog and launch lock belong to the `campaign run` wrapper, not arbitrary raw `train` launches.

## Verified implementation, not a racing improvement claim

Verification evidence is under `logs/verification/sprint-explore-exploit-20261024/`:
- Explore and exploit each completed two real CPU updates with 25 restored state tensors checked, correct fresh optimizer/source age, finite logged scalars and eight ONNX parity cases.
- Exploitation smoke measured mean training standard deviation about 0.19 versus about 0.38 in exploration. This verifies the control, not racing reliability.
- The latest completed low-LR run was screened across all 11 saved checkpoints at the frozen command and rejected for lacking a three-checkpoint stable neighborhood. No long run or publication was launched as part of this revision.
- Launch-path regressions exercise the real provenance hook, preflight under a held sandbox lock, and real non-training child-process cleanup after SIGTERM or an injected journal error. These are launcher checks, not GPU learning results.
