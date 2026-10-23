# Aggressive Sprint reward candidate

## Status

Implemented and integration-tested, **not yet demonstrated to train a faster
policy**. The 3.0 m/s reward saturation is a design target, not an achieved
speed. This is an intentionally combined speed/stride/head experiment rather
than a causal one-setting ablation. Do not deploy it to physical hardware
without separate actuator, impact, and sim-to-real validation.

## Reference inspection and research

The live Arena leaderboard's leading entries report 0.84 s, five finishes from
five seeds, at command 1.15 m/s.[4] All three user-supplied ONNX files were also
run through the installed Arena's complete speed sweep and finalist evaluation:

| Reference | SHA-256 prefix | Local mean time | Finishes | Best command |
|---|---|---:|---:|---:|
| `policy (2).onnx` | `df443e8aaa0b` | 0.84 s | 5/5 | 1.15 m/s |
| `policy (1).onnx` | `3bd79f53654e` | 0.84 s | 5/5 | 1.15 m/s |
| `policy.onnx` | `6eabd8527cbb` | 0.84 s | 5/5 | 1.15 m/s |

Their selected best traces are identical despite distinct file hashes. Treat
this as one demonstrated behaviour, not three independent successful gaits.
The stored traces give a mean sampled forward speed of 2.40063 m/s and a peak
sampled speed of 3.35779 m/s. The head also moves substantially; these are not
examples of a rigidly fixed head.

Graph inspection found 512/256/128 neural layers plus lookup tables, signature
comparisons, and a counter encoded through previous action slot 42 (head-roll
output slot 8). A graph and its weights do not reveal the original reward or
PPO recipe. This candidate does not copy the action tables, seed signatures,
or encoded counter, and it does not claim to reproduce those policy designs.

Forward kinematics of the selected trace, using the Arena model, gave foot
height median 0.03729 m, 90th percentile 0.07222 m, and maximum 0.09765 m.
Finite differences at the 50 Hz trace sampling gave absolute trunk-relative
forward foot-speed 90th percentile 1.99599 m/s, with about 0.20 m relative
forward excursion. These are coarse sampled diagnostics, not exact contact
phase or peak instantaneous velocity measurements.

Machine-readable local evidence, with all three hashes, scans, and seed results:
`logs/20261023_sprint_reference_analysis.json`.

Upstream MicroDuck guidance stresses correct penalty signs, avoiding rewards
for bad resting states, allowing motion required by a dynamic skill, and
introducing smoothing cautiously.[1] Rudin et al., section 3.2, combine velocity
objectives, undesired-axis controls, regularisation, and step rewards without
prescribing a gait clock.[3] Installed mjlab air time counts currently airborne
feet inside a time window; legged_gym instead pays its air-time term on contact,
so the implementations' coefficients are not interchangeable.[2]

## Changes in the three files

### challenge.toml

- Parallel environments: 2048; iterations: 3000; training seed: 2.
- Forward weight: 5.0; upright weight: 1.5; productive-step weight: 6.0.
- Existing knob identifiers and allowed min/max ranges are retained.
- Labels now describe sprint intensity and productive steps.

The workshop Settings form can overwrite the file. Reload it and check the
seed and values before starting another training. An already-running process
uses its loaded configuration, not these edits.

### rewards.py

**Dense sprint speed.** Rotate horizontal world velocity into the spawn-heading
frame. For the main speed term, define
`p = clip(v_forward / (2.5 * command_vx), -1, 1)`.
Positive progress is multiplied by
`clip((-gravity_z - 0.5)/0.5, 0, 1) * exp(-(v_side/0.2)^2)`.
Reverse progress remains negative rather than hiding behind an upright gate.
Idle/non-forward commands and invalid state produce no payout.

The 2.5 gain makes the reward saturation 0.5–3.0 m/s across the required
0.20–1.20 command sweep. Unlike exact Gaussian command tracking, the term does
not penalise a 2.4 m/s sprint at command 1.15. A separate small weight-0.5
bootstrap term still saturates at the unscaled command and never penalises
higher speed. These are bounded velocity-rate objectives, not potential-based
shaping; policy invariance or general immunity to reward hacking is not claimed.

**Productive air time.** Retain the TOML-controlled `air_time` term, shorten its
window to 0.06–0.22 s, and require another foot's ground contact. Only the
non-contact foot inside the window is counted. Forward progress, low lateral
velocity, and upright posture gate its positive payout. Flight phases are
permitted but do not earn the stepping bonus.

**Forward swing drive.** Reward forward foot velocity relative to the trunk,
not translation of a frozen leg. Saturate at 2.0 m/s relative swing speed and
0.04 m clearance, require another foot's support, and gate on productive
forward travel. The term is bounded in [0,1], with weight 1.0. Stance sliding,
backward swings, stepping in place, and unsupported flailing do not earn it.
Selector order is explicitly left foot then right foot to match the sensor.

**Forward-facing head.** A nonnegative, capped yaw/roll misalignment cost uses
selected joint names, actor-view encoder positions, and the head command. A
0.15 rad deadband allows normal movement. Weight is -0.5, because the function
returns a positive cost. Pitch is deliberately not locked by this new term;
existing broad head tracking and slow head-bias correction remain. This means
head forward relative to the body, with body heading held separately—not a
new world-fixed camera-gaze controller.

### env.py

- Replace `track_linear_velocity`'s Gaussian with dense gain-2.5 progress while
  preserving its log name and TOML `forward_reward` connection.
- Add supported swing drive and head-forward cost; wire supported air time.
- Widen walking pose tolerances and reduce pose weight from 1.0 to 0.5, allowing
  larger hip/knee/ankle motion without changing physical joint limits.
- Raise clearance targets from 0.02 to 0.04 m.
- Reduce action-rate smoothing schedule from -0.1→-1.0 to -0.05→-0.3.
- Keep standing practice at 2–5% rather than increasing it to 25%.
- Disable random velocity pushes for this no-push race; remove any inherited
  push event, not merely the local toggle. Other randomisation remains.
- Reduce broad head-pose tracking weight from 2.0 to 1.0; retain its 0.5 rad
  tolerance and existing slow bias curriculum rather than tightening pitch.

Robot, 61D actor/14D actions, network, PPO, command slots/sweep, heading_hold,
critic-only heading error, collision/limit costs, and NaN guards remain intact.
No imported ONNX model is used as a checkpoint or silently converted to PPO.

## Verification and limits

- Full suite: 104 passed, including tests of speed gain, reverse/caps, frame and
  reset handling, support gating, relative foot drive, head selectors/commands,
  invalid state/scales, and TOML-to-environment wiring.
- Explicit CPU/CUDA full-output comparisons exercise the custom rewards at
  `rtol=1e-5, atol=1e-6`, checking device, shape, dtype, and finiteness. This is
  reward-function parity, not full-training parity or a throughput benchmark.
- Two-environment, two-update CPU smoke training at seed 2 completed and exported
  ONNX. ONNX Runtime CPU inference accepted [1,61] and returned finite [1,14].
- All 49 smoke TensorBoard scalar tags were finite; checked penalty terms had
  no positive sign violations. The existing actuator-selector warning remains.
- Ruff checks pass for the changed Python files; `git diff --check` passes.
- Final smoke evidence: `logs/sprint_aggressive_final_smoke.log` and
  `logs/20261023_sprint_aggressive_verification.json`.

This candidate may destabilise early learning or overfit simulation. The 2.5
speed gain, shorter air-time window, and reward weights are engineering choices,
not coefficients validated by the papers or the reference policies. The new
support bonuses may favour walking over a beneficial flight phase. Low drift
velocity does not directly recenter a duck already offset in the lane.

Acceptance remains Arena evaluation of every checkpoint, prioritising 5/5
finishes and then lower mean time. Recheck promising results with another
training seed. Compare this whole package against the previous recipe at the
same budget; component ablations are needed to identify which changes help.
No full training, submission, publication, commit, or push was launched by this
implementation work. Any training started independently in the workshop is
left untouched.

## Sources

[1] https://raw.githubusercontent.com/pollen-robotics/microduck_rl/develop/AGENTS.md
[2] https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
[3] https://arxiv.org/html/2109.11978v3
[4] https://pollen-robotics-microduck-arena.hf.space/api/events/sprint-2m/leaderboard
