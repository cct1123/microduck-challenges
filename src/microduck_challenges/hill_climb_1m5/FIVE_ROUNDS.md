# Five additional Hill rounds

`five_rounds.Controller` subclasses `overnight.Controller`. Existing modules and
tests must remain byte-identical: the initial checkpoint's old campaign evidence
pins those files. Adding this module/test does not change that old frozen key set.

## Policy

- Read the **new** schema-1 authorization, additionally requiring
  `policy="five-rounds-v1"`, `requested_additional_rounds=5`,
  `training_seeds=[3,4,5,6,7]`, `allowed_reward_profiles=["S0","P1"]`, and
  `speed_not_before_epoch=1792923355`. Retain the original six-hour training and
  eight-hour hard deadlines relative to that authorization's epoch.
- `initial_source` pins path/hash/profile, official and fixed summaries,
  `prior_summary`/`prior_summary_sha256`, and
  `fixed_evaluation`/`fixed_evaluation_sha256`. Preparation verifies the prior
  authoritative journal, complete source inventory, actual official rows, worker
  request/raw rows/export hashes and the unchanged normal strict source loader.
  Only S0/P1 normal 5/5 sources are accepted, never E's partial exception.
- First round is S0 from the pinned initial best. Further S0 rounds use the global
  eligible best; retained descendant sources have their whole fixed-command
  sweep reverified. Isolated 5/5 does **not** create an anchor.
- Complete five work iterations even when the first two lack Gate S. Reservations
  precede launch, seeds are 3–7, and there is no sixth attempt or hidden retry.
  Existing uncertain journals fail closed rather than automatically resuming.
- P1 remains weight **1**, never P2 or a lean change. Its selected source needs its
  own exact-source Gate S: three unfiltered adjacent fixed-command 5/5 results,
  independent calibration 26–30, normal official 5/5 and strict source identity.
  The earliest speed epoch is the original campaign's 1792923355, not new T+4h.
  Repeated P1 is consolidation, not a weight increase. Best-candidate monotonicity
  is enforced before journal persistence. Original confirmed R is immutable.
- A P1 branch rollback permanently disables P1, removes speed sources from the
  active continuation pool and reverts to verified S0 fallback/R. No interrupted
  branch checkpoints are adopted.
- Each bounded launcher retains guarded units, `no_smoke=True`, 500 updates,
  2048 environments, save interval 50, 50-minute trainer wall bound, LR 5e-5,
  entropy .01, PPO clip .2, std scale 1, upload False, original trainer argv,
  source/manifest bindings, exclusive locks and independent owned-process timer.
- Software, missing/changed evidence, nonfinite state, competing trainers and
  authorization deadlines stop the campaign. Count completed work, rollback,
  reservations/uncertain and known failures separately.
- Final seeds 6–25 open only after all five reconciled iterations and one immutable
  candidate freeze, once, paired with original R. Both worker reports are read
  back strictly. No alternatives, retries, automatic promotion or publication.
  After round five the journal may tighten its effective training cutoff to start
  validation immediately. It never extends authorization. Without Gate S the
  outcome is `stability_unmet`, preserving the best normal source without heldouts.

## Commands

From `/home/ctcheung/Downloads/microduck-challenges`, inspect without launching:

```bash
MICRODUCK_SPRINT_STAGE=baseline MICRODUCK_HILL_STAGE=bootstrap .venv/bin/python -m microduck_challenges.hill_climb_1m5.five_rounds run --authorization /home/ctcheung/Downloads/microduck-challenges/logs/campaigns/hill-five-rounds-20261025-044013/authorization.json --output /home/ctcheung/Downloads/microduck-challenges/logs/campaigns/hill-five-rounds-20261025-044013/controller
```

Actual launch requires appending `--execute`. The parent owns launch, persistence,
verification and commit. No runtime training/export/Arena integration was executed
while implementing this module. A unit pass is not real-runtime validation.

Regenerate the journal-derived report:

```bash
MICRODUCK_SPRINT_STAGE=baseline MICRODUCK_HILL_STAGE=bootstrap .venv/bin/python -m microduck_challenges.hill_climb_1m5.five_rounds summary --output /home/ctcheung/Downloads/microduck-challenges/logs/campaigns/hill-five-rounds-20261025-044013/controller
```

Run the supported guarded suite from repository root, with logs outside project
discovery (choose a fresh output directory):

```bash
MICRODUCK_SPRINT_STAGE=baseline MICRODUCK_HILL_STAGE=bootstrap .venv/bin/python -m microduck_challenges.hill_climb_1m5.overnight units --output "$TMPDIR/five-rounds-guarded-units"
```
