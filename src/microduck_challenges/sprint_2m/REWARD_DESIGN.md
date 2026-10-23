# Sprint reward candidate: productive stepping

## Status and scope

This is an implemented, integration-tested reward hypothesis, **not a trained or
Arena-validated improvement**. It changes the training objective, not the policy
architecture. Full training, multi-seed evaluation, and checkpoint selection remain
necessary before claiming better racing.

Reviewed files:

- `challenge.toml`: keep the live workshop recipe and its three weight knobs.
- `env.py`: retain the existing training recipe; wire the two custom terms below.
- `rewards.py`: absent in the original Sprint folder; now contains custom terms.

No dependency, robot, task registration, actor observation, command range, PPO,
randomisation, head curriculum, or action-smoothing changes are part of this
candidate. All changes form one deliberately combined reward experiment.

## Review and research

The installed mjlab `track_linear_velocity` uses
`exp(-(||v_xy-command_xy||^2 + v_z^2) / std^2)`, with `std^2 = 0.1` here.
For a stationary upright robot, its unweighted values at commands 0.2, 0.6,
and 1.2 m/s are approximately 0.67032, 0.02732, and 0.00000055739. High-command
exploration therefore receives very little speed-tracking reward near rest.
This is a property of the formula, not a diagnosis established from a trained
rollout.

The installed `feet_air_time` counts feet currently inside the 0.125–0.300 s
air-time window. It gates on the command, not actual forward motion, so stepping
in place can earn it. This differs from legged_gym's touchdown-based air-time
term; copying weights or timings between implementations is not justified.[2]

Upstream MicroDuck guidance stresses penalty-sign checks, avoiding rewards for
bad resting states, keeping motion-blocking regularisers modest, and not adding
large early exploration taxes.[1] Rudin et al., section 3.2, use velocity tracking,
undesired-axis controls, regularisation, and a step reward without prescribing a
gait clock.[3] Those principles inform this candidate, but neither source
validates the new formula or its chosen coefficients on MicroDuck.

The challenge's own guide records a failed absolute `lane_hold` position reward.
This candidate does **not** reintroduce that term: it uses instantaneous lateral
velocity in the spawn-heading frame, not lateral displacement or wall distance.
It cannot directly pull a duck back toward lane centre once an offset exists.

## New objective

Let `v_cmd` be the positive commanded speed. Rotate world horizontal velocity
into the spawn-heading frame to obtain `v_forward` and `v_side`. Reuse the
reference owned by the retained `heading_hold` reward. For rows at their first
step after reset, use current yaw as that upstream reward does; do not mutate its
reference or depend on reward evaluation order.

Define:

- `p = clip(v_forward / v_cmd, -1, 1)`;
- `u = clip((-gravity_z - 0.5) / 0.5, 0, 1)`;
- `q = u * exp(-(v_side / 0.2)^2)`;
- `forward_progress = p*q` when `p > 0`, otherwise `p`;
- `productive_air_time = stock_air_time * max(forward_progress, 0)`.

Both terms are zero for idle/non-forward commands. Non-finite robot/command
state gets no new payout; existing NaN termination and manager guards remain.
The scale parameters must be finite and positive.

### Wiring in env.py

| Term | Configuration |
|---|---|
| Existing speed tracking | Unchanged; weight remains `forward_reward` from TOML |
| New `forward_progress` | Weight 0.5; independent of the TOML tracking knob |
| Existing `air_time` name | Function replaced with `productive_air_time`; weight remains `step_reward` |
| Existing upright | Unchanged; weight remains `upright_reward` |
| Heading, collision, limits, slip, head, smoothness | Unchanged |

Positive forward bonus is bounded at 0.5 before the manager's timestep scaling.
The bonus gives no extra payout for exceeding the command. Reverse motion stays
negative even when tilted or drifting, rather than allowing a bad-pose gate to
hide reverse travel. Air time still ranges from zero to two qualifying feet,
with productive motion attenuating rather than increasing that range.

No actor gets position, heading, or true linear velocity. These quantities are
training-only reward information; the actor remains 61D with 14 joint targets.

## Risks and acceptance criteria

- Multiplying the step reward by progress may delay initial stepping. This is
  why the standalone, low-weight progress term remains active from the start.
- The lateral scale 0.2 m/s and bonus weight 0.5 are engineering choices, not
  literature-derived optima. A stricter gate could suppress useful exploration.
- The candidate does not prevent hopping or sliding by itself. Existing foot,
  posture, collision, and limit terms remain responsible for gait quality.
- This is bounded rate shaping, **not** potential-based shaping; there is no
  policy-invariance or general anti-reward-hacking guarantee.
- Existing random pushes and sensor randomisation are retained for isolation,
  although the Arena does not apply all these disturbances.

Compare against the previous reward functions at the same environment count,
weights, seed, and iteration budget. To ablate this package, restore the stock
`mdp.feet_air_time` function and remove its `lateral_std` parameter, and omit the
`forward_progress` term. Do not change PPO or curriculum at the same time.
Evaluate every checkpoint using the Arena evaluator. Prefer 5/5 finishes, then
lower mean time. Recheck a promising result with a second training seed. A
higher total training reward is not acceptance evidence.

## Verification performed

- Baseline suite before code changes: 66 passed.
- Added tests using observed RED-to-GREEN cycles for progress, spawn-frame
  direction, gating, air-time integration, environment wiring, and scale checks.
- Full suite after implementation: 91 passed, including 25 new cases and explicit
  CPU/CUDA full-output comparison of the custom reward functions.
- CPU/CUDA comparison checks device, shape, dtype, finiteness, and all elements
  at `rtol=1e-5, atol=1e-6`. This is reward-function parity, not full-training
  parity or a GPU throughput benchmark.
- CPU smoke training: two environments, two updates, explicit seed 2; exit 0.
  Both updates ran, 61D actor/14D action contract was reported, an ONNX policy
  was exported, and `nan_state` termination metrics were zero.
- Exported smoke policy executed in ONNX Runtime on CPU: input shape [1, 61],
  output shape [1, 14], all outputs finite. All 47 logged TensorBoard scalar
  tags were checked for finiteness and passed.
- Smoke log: `logs/sprint_productive_smoke.log`.
- Smoke snapshot: tracking/upright/step weights 3/2/3. During this work, an
  external edit changed live TOML values to 5/1/6 and recipe seed 1. Those edits
  were preserved. The smoke result must not be attributed to that newer setup.
- CPU smoke deliberately hides CUDA; Warp reports unavailable CUDA at startup
  and then completes on CPU. The existing actuator-selector warning remains.

No full training, Arena submission, publication, commit, or push was performed.

## Sources

[1] https://raw.githubusercontent.com/pollen-robotics/microduck_rl/develop/AGENTS.md
[2] https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
[3] https://arxiv.org/html/2109.11978v3
