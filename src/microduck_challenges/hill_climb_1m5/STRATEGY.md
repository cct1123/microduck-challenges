# Hill Climb H1: bounded exploration, not promotion

## Ordinary Workshop / train defaults

The selected next recipe is **refine / explore / H1**, 500 new PPO updates,
2048 environments, checkpoint interval 50, seed 1, fixed learning rate 0.00005,
entropy 0.01, clip 0.2 and learned Gaussian std scale 1.0. There is no automatic
switch to exploitation, retry, promotion, publication or incumbent replacement.
The campaign wrapper has a 40-minute wall cap; ordinary Workshop/raw `train`
shares the recipe but does not inherit that wrapper watchdog or launch lock.
Nothing in this change authorizes a run. The user has forbidden smoke tests,
rollouts, training, real export and pushes; only the reviewed, scoped local
implementation and documentation commits are authorized.

`training.refine` pins exactly:

`logs/rsl_rl/hill_climb_1m5/2026-10-24_23-10-03_run-1024-2309/model_1000.pt`

SHA256 `799ed0b452a1a484b4f700362248779fe127a1ba821a2c545728cdf9c9e814b9`.
Saved environment, agent, provenance and Workshop result hashes are also pinned.
There is no latest-run glob and no cold-start fallback. Missing/tampered inputs
fail closed. Bootstrap remains explicitly available with
`MICRODUCK_HILL_STAGE=bootstrap`, using live Workshop knobs and the original
beginner curricula/optimizer. This is also the portable unit-collection override
on clean clones without the local source archive.

## Exact exploratory-source exception

The real checkpoint has **3/5 finishes**, failure-penalized official mean
**25.872 s at command 0.75 m/s**, not a reliable 5/5 result. Raw seed rows:
1 finishes 3.14 s; 2 falls; 3 falls; 4 finishes 3.12 s; 5 finishes 3.10 s.
Each fall contributes 60 seconds. All five available checkpoints from this
stopped bootstrap were evaluated: 0/250 score 60 s (0/5), 500 scores 49.164 s
(1/5), 750 scores 48.784 s (1/5), 1000 scores 25.872 s (3/5).

The user explicitly approved **model_1000 for exploration only**. That exception
is restricted to this exact path, checkpoint hash, four metadata/result hashes,
3/5 raw result and explore phase. It is not a global lowering of source or
promotion gates. Every other training source still needs consistent official
5/5 evidence. Exploitation still requires a 5/5 source and an approved exact
promotion report, preserving its reward profile. The partial-source marker
cannot be removed to bypass this restriction.

## Source recipe before H1

The source's provenance names clean commit `6ecf44903`. Its saved configuration
and that commit's TOML independently agree on these original construction inputs:

| Knob | Saved source | Current live bootstrap knob (not H1 baseline) |
|---|---:|---:|
| Linear velocity tracking weight | 3.95619419642857 | 5.0 |
| Upright weight | 2.0 | 1.00167410714286 |
| Air time weight | 6.0 | 6.0 |
| Maximum generated slope | 10 degrees | 10 degrees |

Refinement reconstructs and validates the entire source recipe with those pinned
inputs before target overrides; it does not treat today's edited knobs as old
training history or rewrite archived metadata. Only known runtime terrain
hydration, world count and seed differences are normalized. Descendants persist
original construction inputs, source pins, phase, H1 parameters and ramp origin
in saved agent configuration and checkpoint markers, and strictly revalidate
these identities. Missing or contradictory H1 markers/parameters are refused.

The real source environment clock is **24024**, with runner checkpoint iteration
1000 (1001 completed 24-step batches). Full actor/critic tensors and both running
normalizers are restored strictly, without architecture migration: 61D actor,
77D critic and 14 actions. Adam and runner numbering start fresh. Actual
environment age is retained; bootstrap curriculum stages freeze at the separate
source clock, not at mature endpoints. Here action-rate weight is -0.6,
head-bias weight 2.0 and standing fraction 0.15. `MICRODUCK_WARM_START` is refused
because upstream would reset the environment clock.

## H1 reward hypothesis (unvalidated performance)

A0, A1 (only linear tracking off) and B1 (only air time 1.5) keep their meanings.
H1 intentionally combines two changes, relative to the frozen actual source:

1. Add `horizontal_forward_speed`, weight **5.0**, returning the signed bounded
   value `alpha * tanh(v_horizontal_spawn_heading / 1.0 m/s)`. Reverse motion is
   negative, not a positive speed-magnitude reward. This uses true body linear
   velocity, not wheels/joints or commanded speed. The coefficient is a modest
   starting hypothesis alongside existing source tracking 3.956 and air time 6;
   it is not an experimentally selected optimum.
2. Keep upright weight **2.0**, upright optimum and baseline gravity-component
   variance **0.05**. Relax only positive-gravity-x (nose-down) pitch variance
   toward **0.30**, blended continuously with *actual positive forward motion*,
   reaching full configured allowance **by 0.1 m/s**. This is tolerance, not a
   forced lean angle or a changed joint limit. Roll, backward pitch, stationary
   and reverse motion retain baseline tolerance. No global upright weakening.

Both changes share `alpha`: zero through the first **50 new updates**, then a
linear **200-update ramp**, reaching full effect at new update 250 within the
500-update screen. Origin is the real source clock, not zero or an inherited
mature bootstrap schedule; H1 descendants inherit that origin rather than
restarting the ramp. Rewards are computed from the live clock, so no manager
weight mutation can silently jump the schedule.

The frame is explicitly **horizontal spawn-heading projection**. Generated Hill
training terrain contains pyramids and flat spawn platforms: the robot travels
down, up and across slopes. This is not along-slope race distance, nor a copied
fixed 8-degree course vector. Existing heading hold and critic heading error
remain intact. H1 positive payouts are suppressed on nonfinite root state or
more than 60-degree tilt. An independent Arena-style >60-degree held-0.3-second
termination is added; the original hard fall termination is not loosened.

Robot/BAM/joint mechanics, unclipped scale-1 actions, solver 30/50, terrain,
maximum slope, command sweep, actor/critic ordering, head/body slots, head EMA,
and all other source rewards are unchanged except the explicit curriculum freeze.

## Verification and limits

The current preparation is **unit-only**. It checks real Tyro parsing of plain
Workshop/train defaults and planned CLI arguments, then loads the real model on
CPU through the registered installed runner using an in-memory fake environment
(no physics). It checks exact actor/critic/normalizer tensors, both clocks, fresh
Adam, source/profile markers, ramp/frame/sign behavior, stationary/reverse/roll/
backpitch, nonfinite/deep-fall suppression, held-fall/reset and source/parameter
corruption. Portable tests explicitly request bootstrap instead of importing an
archive-dependent default. Shared/other-challenge tests are preserved.

Evidence: `logs/verification/hill-h1-explore/REPORT.md` and `verification.json`.
The guarded tracked suite excludes all smoke-named tests, two true Arena tests
and the true-child launcher test. It fails unexpected simulation/training/
installed-official-export boundaries. No smoke, rollout, optimization step,
installed real export, Arena race, training launch, push or performance
improvement is claimed. Scoped local commits are backed by guarded unit-only
snapshot verification, not runtime validation. The existing campaign launch
preflight still invokes a smoke and must not be used while the user's smoke ban
remains in force.

## Promotion gate (unchanged)

Reconcile every expected saved checkpoint/result; freeze one valid command
before fresh assessment. Require three adjacent entries in the unfiltered
inventory each to finish 5/5; select the fastest once, then compare with the
retained incumbent on paired held-out seeds 6–25. Require candidate 20/20,
strictly faster development mean and no worse failure-penalized held-out mean.
Every failure counts 60 s. Bind approval to exact raw rows/artifact hashes and
completed, unchanged campaign provenance where present. Approval is not a
leaderboard result or physical-robot validation. No assessment was run here.
