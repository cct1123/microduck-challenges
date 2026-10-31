# AGENTS.md: Tug of War

You are helping a user train a Microduck policy for the Arena's **Tug of War** (event `tug-of-war`,
task `Mjlab-TugOfWar-MicroDuck`, this folder). The user may be new to reinforcement learning. Work so
that training stays:

- **simple**: the user decides the goal, the budget and what gets published; you do the mechanics;
- **transparent**: say what you change and why before each run, one commit per run, and report the
  Arena evaluator's numbers, never an impression from the curves;
- **efficient**: one change per run, a smoke test before any long run, every checkpoint judged,
  a failed run stopped instead of paid for.

Never launch a Hugging Face job, publish, enter the Arena or push without the user's explicit yes.

## The challenge

Two ducks stand neck to neck over a pit, a string between them, each pulling by walking
backwards. The rules, from `arena/events/tug-of-war.toml` in the installed Arena:

- **Course**: a wrestling ring 3.00 m from rope to rope, cut across the middle by a pit 0.30 m wide
  and 0.30 m deep. The ring's three ropes (6, 12 and 18 cm high) are the field's edges: a duck that
  backs into them bounces off. The two ducks start on either side of the pit. Start from the `STAND`
  pose, 0.5 s untimed settle, joints perturbed by ±0.01 rad, heading by ±2°.
- **The string**: 1.00 m, tied at the base of each neck, laid straight at the start. It is a real
  rope, 80 g and 8 mm thick in sixteen pieces: it sags, and a duck can step or lie on it. Its ends go
  taut (0.98 m or more apart) when the other duck stands and resists; when that duck lies down or
  gives way they stay short of it (at most 0.966 m in the bouts traced), its lowest point at the
  canvas or down over the pit.
- **Command**: the policy reads `[-0.30, 0, 0]`, constant for the whole bout: walk backwards at
  0.30 m/s. Head and body commands are zeros. It is never told its heading or its position.
- **A bout**: only the pit decides. The first duck whose trunk drops into the pit loses; a duck that
  falls, sits or lies down stays in play and can still be dragged in. If neither is in the pit by
  the time limit (**45 s**), or both go in on one step, the marker decides: the string's middle
  point, pulled towards one side, that side's duck wins.
- **A fault** loses the bout at once: a duck that gives no valid action, an action that is not a
  number (NaN), a policy that fails to answer, an answer longer than 1 s, or more thinking over the
  bout, settle included, than its 45 s. Both at once: nobody wins it. A step MuJoCo has to reset
  ends the bout for nobody.
- **The rating**: five bouts against the Arena's **reference duck** (`reference.onnx` here, t20-500,
  which walks backwards upright, as the robot walks, on this ring and string), the entry west on odd
  seeds. A bout scores the centimetres of ground the marker ends from the centre line towards the
  entry, plus **100 cm** when only the opponent is in the pit (minus 100 when only the entry is). The
  rating is the mean of the five, **in centimetres, higher wins**. A fault scores **-200 cm**, and in
  a rating a MuJoCo reset is the entry's fault too (the reference never causes one), as is a string
  end torn more than 5 cm off its neck: a physics explosion MuJoCo has not caught yet. `arena score`
  prints the rating, a faulted bout as `fault`. Against
  it, the ducks that pull bowed onto their heads rate about +135 cm, the recipe as shipped +138.1 cm
  after 250 iterations, and the reference itself +3.7 cm ("What is known").
- **The weekly cup**: each player's latest entry plays it, bout against bout; the rating ranks the
  board and breaks a match left level. Entries close on Friday at 16:00, Paris time, and the cup airs
  at 18:00; it needs at least 4 entrants.

## The policy

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish`
checks the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers: twist (3), head (4), body (6). Joint order: left leg (hip_yaw,
hip_roll, hip_pitch, knee, ankle), neck_pitch, head_pitch, head_yaw, head_roll, right leg (same
order). The Arena simulates the same robot with the same BAM actuator model as training (MuJoCo,
implicitfast integrator, 10/20 contact solver).

## The recipe

The task trains the learner (always the west duck) against `reference.onnx`, the frozen reference
duck, in the Arena's bout: the ring, the string, the pit, 45 s, only the pit decides. The opponent
runs as the Arena runs it, with no randomisation: its servos on the Arena's nominal battery (7.35 V,
sag gain 0.1), where the learner's are drawn per env.

- **The learner starts as the reference duck** (`ReferenceActor`, `RL_CFG`'s actor: the
  reference's layers and observation normalizer, action noise std `WARM_START_STD` 0.5, the
  reference's own, the critic fresh). Trained from scratch it plateaued or lay flat (t8r, t9r);
  started as the reference of the day it dragged it into the pit by 500 (t10 and t12, from t7-250)
  and by 250 (t22, t23 and t24, from t20-500). To train from scratch anyway, add
  `--agent.actor.class-name MLPModel`.
- **The string is the Arena's from the first iteration**: there is no slack phase. Every reset lays
  it straight between the two ducks' string ends.
- **1000 iterations, a checkpoint every 250**: every run so far peaked early, then decayed (t22 and
  t23 at 250, their gait sliding after; t20 at 500). Keep the best checkpoint, not the last: judge
  every one.
- **Stay upright** (`stay_upright`, off by default) means walking upright, as the robot walks. At 0
  a duck may crouch, sit, lie, crawl or plant its head, as the Arena allows, and the Arena's fall
  rule (`fell`: tilted past 60° for 0.3 s) ends a training episode only. Above 0, `down` replaces
  `fell`. The learner must hold:
  - its trunk 10 cm up or more (`UPRIGHT_Z_M`);
  - its trunk pitched 10° towards the opponent to 20° away from it (`UPRIGHT_PITCH_DEG`), rolled
    10° at most (`UPRIGHT_ROLL_DEG`);
  - its head's centre of mass 18 cm up or more (`UPRIGHT_HEAD_Z_M`);
  - its neck and head pitch within 30° of standing (`UPRIGHT_NECK_DEG`);
  - only its feet on the course: a contact sensor (`body_ground_contact`) watches every other body.

  Out of that for 0.3 s straight ends the training episode and costs that many pits (-50 each).
  The bands are the robot's own walk with margin: the Arena's golden walker (BEST_alpha_walking)
  holds its trunk at 11.0–12.4 cm, pitched -2.7 to +2.1°, its head at 22.6–24.1 cm. The 20° back is
  room to lean on the string.
- **Why the owner's ducks use it**: the owner asked that his ducks walk backwards upright, like the
  robot's own gait (2026-10-22). Under the earlier rule (trunk 7 cm or more, tilt 45° or less) they
  pulled bowed onto their heads, the head a third leg ("What is known"). They train with stay
  upright 1.0 and Win ground 4.0: set those two knobs and train as below (t22 and t23 trained
  that recipe, at commit `c2333dd`). Other players are free: the Arena judges the pit alone.
- **The reference was trained from the walker**: t20 started as the golden walker, a duck that
  walks backwards upright but never pulled, and trained against it (stay upright 1.0, Win ground
  4.0); its checkpoint 500 is the reference duck. Its recipe is commit `8391bf7` of this
  repository's `tug-of-war` branch and the command

  ```bash
  uv run train Mjlab-TugOfWar-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1000 \
      --agent.seed 1 --agent.logger tensorboard --agent.run-name t20 \
      --agent.actor.class-name microduck_challenges.tug_of_war.opponent:WalkerActor \
      --env.actions.opponent.onnx walker.onnx
  ```

  The walker (`walker.onnx`, `WalkerActor` and the opponent's `onnx` field) is in the history up to
  `c2333dd` and gone since `10cc7aa`; t20 reproduces only at `8391bf7`. Later commits start from the
  reference and play it.
- `win` pays (+50) when the opponent drops into the pit, `out` costs (-50) when the learner does (and
  when `fell` ends the episode), `marker_ground` pays the new ground the marker wins towards the
  learner, `pit_margin_reward` the ground kept between the learner's string end and the pit.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `challenge.toml` `[params]` | The four knobs below | Yes, first. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 1000, `seed` 1 | Yes, when agreed with the user |
| `env.py` | The whole environment: the ring and the string (copies of the Arena's), rewards, terminations, the opponent's wiring, domain randomization, `RL_CFG` (the actor, starting as the reference duck; PPO; checkpoints every 250 iterations) | Yes, once the knobs are not enough; not the copied constants (below) |
| `rewards.py` | The tug's own terms: the marker's new ground, the pit margin, the Arena's fall rule (`ArenaFall`), and walking upright: the rule's constants (`UPRIGHT_Z_M` 0.10 m, `UPRIGHT_PITCH_DEG` -10 to +20, `UPRIGHT_ROLL_DEG` 10, `UPRIGHT_HEAD_Z_M` 0.18 m, `UPRIGHT_NECK_DEG` 30; the bodies and joints they read, `HEAD`, `NECK`, `FEET`), `is_upright` and `Down` (with stay upright on) | Yes; `check.py` judges with the same `is_upright`, so a constant changes training and judging together |
| `opponent.py`, `reference.onnx` | The frozen opponent: the reference duck's network and the action term that drives it; `ReferenceActor`, the learner's actor, starting as that duck | No |
| `check.py` | Plays two policies against each other (B defaults to the reference), says how each bout was pulled and how A walked | No |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | The owner's ducks | Sets in `env.py` |
|---|---|---|---|---|---|
| `pull_reward` | Win ground | 2.0 | 0.5–5.0 | 4.0 | `marker_ground` weight |
| `back_reward` | Walk backwards | 3.0 | 0.0–4.0 | 3.0 | `track_linear_velocity` weight |
| `upright_reward` | Keep the trunk level | 1.0 | 0.0–4.0 | 1.0 | `upright` weight |
| `stay_upright` | Stay upright | 0.0 | 0.0–2.0 | 1.0 | above 0, `down` (walking upright) replaces `fell`, at that many pits (`down` weight -50 × value), and adds the `body_ground_contact` sensor |

Do not change, or the Arena cannot race the policy or the tests fail:

- the robot (`MICRODUCK_STANDUP_ROBOT_CFG`) and the 61-number observation layout: a command slot the
  task does not use keeps its tiny sampling range, never deleted;
- `experiment_name = "tug_of_war"` in `RL_CFG`: runs must land in `logs/rsl_rl/tug_of_war/`;
- the twist command, fixed at `-0.30` m/s, and the terminations: the pits (`in_pit`,
  `opponent_in_pit`) and a training-only end for a duck that goes down, `fell` or, with stay upright
  on, `down` (the Arena judges the pit alone; without one an untrained duck lies on its side from
  1 s and never learns to stand);
- `opponent.py`, `reference.onnx` and the opponent's servos (`OPPONENT_CFG`, the Arena's battery):
  the same bytes and servos the Arena rates against; any other opponent trains a duck the rating
  does not measure;
- the ring and the string copied from the Arena at the top of `env.py` (`RING_M`, `RING_ROPES_M`,
  the rope and post radii, `PIT_M`, `PIT_DEPTH_M`, `STRING_M` and every `STRING_*` number, the bout's
  length, the fall rule) and `STRING`/`MIDDLE` in `rewards.py`: the job installs no Arena, and the
  parity tests in `tests/test_tug_of_war.py` pin each against it (`test_the_constants_are_the_arenas`,
  `test_the_course_tiles_the_arenas`, `test_the_rings_ropes_and_posts_are_the_arenas_in_pieces`,
  `test_the_string_is_the_arenas`, `test_the_string_ends_are_the_arenas`,
  `test_the_string_end_is_where_the_arena_ties_it`, `test_the_marker_is_where_the_arena_reads_it`).
  The string's numbers are the Arena's measured ones: each step down from them was unstable, and
  without its armature every world turned NaN within a second. If the Arena's event changes, change
  the Arena first, then here, then the parity tests;
- the course's 2 m tiles (`TILE_M`) and the solver's buffers (`nconmax` 200, `njmax` 1500): see
  "What is known";
- every domain randomization range must contain the nominal value the Arena simulates;
- `pyproject.toml`, `uv.lock` and the other challenges' folders.

The robot model and the reward functions (`microduck_mdp`) live in the `mjlab-microduck` package
and are not edited here. Its [AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)
holds the reward-design rules learned on this robot: read its "Reward design" section before
writing a reward term.

## Running a training

### Once per session

```bash
uv sync --group workshop
nvidia-smi              # an NVIDIA GPU here trains locally; otherwise Hugging Face Jobs
uv run hf auth whoami   # the account a job runs and bills under, and publish publishes to
```

Agree with the user before the first run:

- the goal: a first entry, a given rating, a given place on the board, a duck that stays on its
  feet or not;
- the budget: how many runs; local GPU, or Hugging Face Jobs (default hardware `l4x1`, $0.80 per
  hour, billed to their account, which needs a payment method);
- what you may launch without asking each time (by default: local runs you announce and start;
  each Hugging Face job asked for).

### Each run

1. **State the hypothesis** in one sentence: what you change, why, and what you expect to see.
2. **Change one thing**, a knob first. Combine changes only on purpose, and say so.
3. **Test**: `uv run --group workshop pytest`.
4. **Smoke test** (under a minute, also on a machine without a GPU):

   ```bash
   uv run train Mjlab-TugOfWar-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
       --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
   rm -rf logs/rsl_rl/tug_of_war/*_smoke
   ```

5. **Commit** the change, its message the hypothesis:
   `git add src/microduck_challenges/tug_of_war && git commit -m "<hypothesis>"`. `publish` writes
   that commit into the policy's recipe, and refuses a run started from uncommitted changes.
6. **Launch** in the background, the output in a log under `logs/` (git-ignored). `<name>` is short,
   unique, and says the change (`pull4`, `upright1`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-TugOfWar-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-tugofwar-microduck-<name>
   ```

   A local run writes `logs/rsl_rl/tug_of_war/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt` every
   250 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the same
   tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-tugofwar-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint (every 250 iterations, several minutes apart), not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0; with stay upright
  on, ask for `Episode_Termination/down` in place of `Episode_Termination/fell`):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/tug_of_war/<run> Train/mean_episode_length Episode_Termination/in_pit Episode_Termination/opponent_in_pit Episode_Termination/fell Episode_Reward/marker_ground Episode_Reward/win
  ```

  Healthy, as measured against the walking-upright reference:
  - the owner's recipe (stay upright 1.0, Win ground 4.0; t23 at 250): `down` 1.5 at
    iteration 50 and 0.8 at 250 (worlds a step), `marker_ground` 0.19 at 50 and 0.24 from 100 on, `win` 0.02,
    `opponent_in_pit` 8.3, the learner's `in_pit` 0, episodes 432 control steps (8.6 s of training; in
    a Mac bout the reference is in the pit within about 2.5 s);
  - the recipe as shipped (t24): the learner's `in_pit` 5.7 at 100 falling to 0.7 by 250, while
    `opponent_in_pit` rose to 6.7 and episodes to 256 control steps.

  Then it decays, and the gait goes first: `upright` and `head_pose_tracking` rise while `air_time`
  and `opponent_in_pit` fall, the feet planted and sliding (t23 from 250 to 500: `upright` 0.18 to
  0.33, `opponent_in_pit` 8.3 to 5.3; t21 from 500 to 999, most of it after about 650: `upright`
  0.17 to 0.72, `head_pose_tracking` 0.33 to 1.43, `air_time` 0.008 to 0.002). That decay is the
  known shape, not a broken run: judge the checkpoints at 250, 500, 750 and 1000 and keep the best,
  not the last.
  Warning signs: `marker_ground` flat near 0.035 with the learner's `in_pit` near 1 (t16 never
  found the pull and was dragged in); `marker_ground` flat near 0.01 while `upright` rises (t13,
  t14 stood and never pulled); `fell`, `down` or the learner's `in_pit` climbing; `nan_state` in more
  than a few worlds per step (the term counts worlds). A run that has collapsed and not recovered
  within about 500 iterations: tell the user and propose stopping it.

### Judging every checkpoint

The curves are not the score. Judge each checkpoint with the Arena's own evaluator:

- If the user has the workshop open (`uv run --group workshop workshop tug-of-war`), it finds every
  run under `logs/rsl_rl/`, yours included, and evaluates each checkpoint as it arrives. Read its
  results instead of evaluating again: `logs/rsl_rl/tug_of_war/<run>/workshop/model_<N>.json`.
- Otherwise export each checkpoint as the real publish does (`--dry-run` publishes nothing, the
  observation normalizer baked in, into `publish-eval/`) and score it:

  ```bash
  run=logs/rsl_rl/tug_of_war/<run>
  for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
    uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
    echo "checkpoint $n:" && uv run --group workshop arena score publish-eval/policy.onnx --event tug-of-war
  done
  ```

  `arena score` prints one line per bout (five), then `rating: X.X cm`.

To see how a checkpoint pulls, and how two policies pull against each other, play them (bout k on
seed k, sides swapped after each, the first policy starting west; the second defaults to the
reference duck):

```bash
uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx [B.onnx] --bouts 10
```

It prints a line per bout, then the tally. The owner's candidate, t23-250, against the reference:

```
bout 1 · seed 1 · A west · B loses by pit at 2.56 s · taut at 0.08 s · A backed 22.9 cm by 2 s · marker 40.9 cm to A · upright 100 % · longest out 0.00 s · feet only 100 % · 5.5 steps/s · slide 9 %
...
A 10 · B 0 · draws 0 · A in the pit 0 · B in the pit 10 · by marker 0 · median decision 2.4 s
```

- **taut**: when the string's ends first stood 0.98 m apart (`TAUT_M`). Information only: against
  the reference, which stands and resists, it goes taut within 0.4 s; "never taut" is normal
  against a duck that lies down, gives way or bows, even dragged into the pit.
- **A backed**: how far A's end of the string moved away from the pit in the first 2 s, signed.
- **marker**: where the string's middle ended, in cm towards A.
- **upright**: the share of A's frames walking upright by stay upright's rule (`is_upright` and
  only its feet on the course), the rule training ends an episode on.
- **longest out**: A's longest straight run out of that rule; training ends an episode at 0.30 s.
- **feet only**: the share of A's frames with no body but its feet on the course.
- **steps/s**: both feet's clear steps a second: a foot back on the course after at least 2 frames
  (0.04 s) in the air, during which it lifted 5 mm or more.
- **slide**: the share of the feet's horizontal travel made while they touch the course. The
  walker slides 8–9 %, t17-250, bowed onto its head, 57–60 %.
- **standing still**, in place of both: the feet travelled less than 1 cm a second.

The frames with A's trunk within 8 cm of the pit's lip (`LIP_M`) are left out of the last five.

The gates the owner's ducks were judged by (10 `check.py` bouts, the rating from `arena score`, the
curves):

- **U2** (walking upright, only for a duck meant to walk upright): in at least 9 of the 10 bouts
  `upright` is 90 % or more and `longest out` under 0.30 s. The owner's ducks passed it twice:
  against their training opponent (the reference), and against a duck that pulls bowed onto its head
  (t15-250, the reference before t20-500; `git show
  8391bf7:src/microduck_challenges/tug_of_war/reference.onnx`). In a cup your duck meets other players' ducks, and must stay upright even when
  it loses.
- **S** (stepping): in at least 8 of the 10 bouts `standing still`, or `steps/s` 2.0 or more with
  `slide` 20 % or less. A pass that rests on `standing still` in more than 2 of the 10 bouts is not
  walking backwards: t21-250 passed with 8 bouts standing still, anchored, and was ruled out.
- **R1**: `nan_state` under 0.01 in the curves, and in at least 8 of the 10 bouts A backed +2.0 cm
  or more by 2 s.
- **R2**: a rating of +20.0 cm or more against the reference.
- **R3** (a target): the reference in the pit in at least 3 of the 10 bouts.

Pick the highest rating among the checkpoints that pass. Between ratings within 5 cm, a round robin
decides: `check.py A B --bouts 20` with each as A, more bouts won, then more pits; U2 and S must
hold over those bouts too, for a cup is duck against duck (t22-250 passed both against the
reference but stepped 1.0–2.1 times a second against t23-250).

Scores on this machine are indicative: only the Space rates an entry, and simulations on different
CPU architectures drift apart after a few seconds. GPU training varies a lot from run to run: before
building on a surprising result, good or bad, run the same commit with another `--agent.seed`.

### Reporting

After each run, tell the user in a few lines: the run and its commit (what changed, why), the best
checkpoint with its rating and its gates, the other checkpoints' ratings, the curves that matter
(`marker_ground`, `win`, `in_pit`, `fell` or `down`, episode length), and your proposal for the next
run and why.

### Publishing, on the user's yes

Publishing makes the policy public on their Hugging Face account and enters it on the Arena, which
rates it and prints its rating and the page of its result:

```bash
uv run publish --run logs/rsl_rl/tug_of_war/<run> --checkpoint <N> --repo <user>/microduck-<name> --no-private --enter
```

`--no-private` because the Arena reads public repos only; add `--livery <colour>` for its colour.
For the policy's code link to open, push the commit to the user's fork first (`origin` must be their
fork, never `pollen-robotics`; the workshop's home sets that up).

## What is known about this challenge

All numbers are from the Mac, indicative: the Space (x86) rates differently.

### Walking upright (2026-10-22)

- **The bow**: a probe of the contacts and forces, two bouts each, found that t15-250, t17-250 and
  t19-250, the ducks that pulled on their feet by the earlier rule (or, t19, with stay upright
  off), bowed onto their heads:
  - trunk 7–8 cm, pitched 33–44° towards the opponent, neck 50–85° down;
  - the head on the floor in 77–83 % of frames, carrying 39–43 % of the floor load and about half
    the backward friction (-0.9 to -1.3 N on the head, -0.9 to -1.1 N on both feet, against the
    ~2 N the tie pulls);
  - the feet sliding through 58–73 % of their travel;
  - with no string they bowed the same: it is how they move, not a reaction to the string.
- **The walker** (BEST_alpha_walking, the robot's own gait), in every bout `check.py` read: upright
  100 %, longest out 0.00 s, feet only 100 %, 3.5–3.8 steps/s, slide 8–9 %. On the open stage it
  stands still at the tug's -0.30 m/s and walks backwards at -0.40 (21 cm/s). It never pulled:
  t15-250 dragged it into the pit in 5.2–5.9 s, and t20-500 in 7.6 and 26.8 s, upright throughout.
- **The shin**: posed at random within every band, a shin touched the floor in 2.2 % of 200,000
  poses, the trunk at 10.0–11.1 cm with a foot down. The bands alone let a duck kneel, so stay
  upright also watches every body but the feet (`body_ground_contact`). A trunk sunk to 3 cm fires
  it on the trunk, both hips and both shins; its feet never fire it: no slot fired in 60
  steps of a duck standing, then tipping forward (the pitch band ended each episode before any body
  landed). It cost nothing measurable on the L4 (t20 7.3 s an iteration, t15 9.8 s without it). The
  walker as both ducks under the rule (64 worlds, 300 steps): upright 99 % of world-steps, no NaN,
  15 contacts and 94 constraint rows a world at most.
- **The reference search, t20 and t21** (commit `8391bf7`, seeds 1 and 2, stay upright 1.0, Win
  ground 4.0, started as the walker and against it; judged at 250, 500, 750 and 999; no lever was
  needed):
  - every checkpoint stayed upright against the walker and against t15-250 but one: t21-750
    (79–80 % upright in two bouts against t15-250, 0.40–0.42 s out);
  - t20-250 planted both feet after about 3 s and slid back (0.1–0.5 steps/s, slide 25–44 %);
    t21-250 passed S, but all 8 of its passes stood still: anchored, ruled out; t21-500 and
    t21-999 planted and slid (S 1 and 0 of 10);
  - t20-500 stepped back upright (3.7–6.6 steps/s, slide 11–14 %) and dragged the walker into the
    pit 10 of 10 (6.7–11.1 s, median 8.8 s; backed +10.3 to +15.3 cm by 2 s);
  - t20-750 and t20-999 still stepped but no longer pulled: the walker beat each 10-0 (in the pit,
    then on the marker at 45 s);
  - all eight rated -124.4 to -136.2 cm against t15-250: the head-planter dragged every
    walking-upright duck into the pit in about 7 s;
  - the pick, t20-500: the only checkpoint passing U2 both ways, S by walking and R1 that also beat
    the walker by pit; t20-750 and t20-999, within 5 cm of its rating, lost to it 10-0. It became
    the reference duck.
- **The planters against t20-500**: t15-250 +135.6, t17-250 +135.1, t19-250 +134.9 cm (15 pit wins
  of 15, at 5.1–9.2 s); t10-500 +8.3 (four marker losses, one pit win); t20-500 against itself
  +3.7 (five marker wins of 0.4–8.6 cm at 45 s).
- **The candidate search, t22 and t23** (commit `c2333dd`, seeds 1 and 2, stay upright 1.0, Win
  ground 4.0, started as t20-500 and against it):
  - at 250: t22 +142.6, t23 +143.3 cm, the reference in the pit 10 of 10 at about 2.5 s, taut
    within 0.4 s, upright 100 %, 4.2–6.9 steps/s, slide 7–10 %; against t15-250 both stayed
    upright (97–100 %) and were dragged in 9 of 10;
  - every later checkpoint still rated +117.0 to +141.7 cm but failed S: t22-500 (4 of
    10), t22-750 (0, slide 36–73 %), t23-500 (7, on slow
    steps of 0.9–1.9 a second, slide 16 % at most), t23-750 (4), t23-999 (2); t23-250 beat t23-500
    10-0 both ways;
  - the round robin t22-250 against t23-250, 20 bouts each as A: t23-250 won 36 (36 pits) and lost
    4, stepping 2.1–4.3 times a second; t22-250 stepped 1.0–2.1 times and failed S;
  - the pick, the owner's candidate: t23-250.
- **The recipe as shipped, verified** (t24, commit `10cc7aa`, stay upright 0, Win ground 2.0,
  started as t20-500 and against it, seed 1): +138.1 cm at 250, the reference in the pit 10 of 10
  (median 3.0 s), backed +15.6 to +30.9 cm by 2 s. Started upright, it stayed upright (100 % in 10
  of 10) and stepped (3.0–5.3 steps/s, slide 8–12 %) with no rule asking it to.
- **Every run peaks early**: t20 at 500, t22 and t23 at 250, then the gait slides. Judge every
  250 iterations and keep the best.

### On the ring and the string, the earlier upright rule (2026-10-22)

Stay upright then read only the trunk: 7 cm or more, tilted 45° or less. "On its feet" and "longest
down" below are that rule's, which `check.py` no longer prints.

- **The string runs**: a probe of 64 worlds on the Mac's CPU (stay upright on, t10-500 as the
  learner, 300 steps) had no NaN on either duck, at most 39 contacts a world (`nconmax` 200) and 190
  constraint rows (`njmax` 1500); `nan_state` read 0 at every judged checkpoint from t13 to t24 on
  the L4.
- **When it goes taut**: in every bout judged for t13, t14, t15, t17, t18 and t19 the ends never
  stood 0.98 m apart, even dragging the other duck into the pit (the other duck lay down or gave
  way). Only t16, whose reference stood and resisted, had it taut (28 of 30 traced bouts). Against
  t20-500, which stands and resists, t22-250, t23-250 and t24-250 had it taut within 0.4 s in every
  bout.
- **The reference search** (stay upright 1.0, warm-started from t7-250, the old reference, and
  against it; ratings against t7-250):
  - t13, t14 (Win ground 2.0, seeds 1 and 2): on their feet 99 to 100 % of every bout (trunk 7 to
    10 cm, tilt 24 to 38°) while t7-250 lay prone from 6 to 8 s, but they never dragged it: t13 -0.4,
    -2.6, -4.4 cm and t14 -3.5, -2.9, -4.4 cm at 250, 500, 750, every bout to the marker at 45 s.
    `marker_ground` stayed flat at 0.01 to 0.025 while `upright` rose from about 0.3 to 0.4 to 0.45:
    standing outpaid pulling. Cancelled at about 780.
  - t15, the one lever (Win ground 4.0, seed 1): `marker_ground` at 250 eleven times t13's (0.274
    against 0.025); +138.4 cm at 250 (the reference in the pit 10 of 10, median 18.4 s, on its feet
    81 to 84 %, longest down 0.10 to 0.14 s, backed +9.7 to +14.0 cm by 2 s), +134.4 at 500, +134.1
    at 750, then +16.9 at 999 with no pit: it peaks early, the pit coming later each time (18.4,
    24.3, 35.9 s).
  - t16 (t15's recipe, seed 2) never found the pull: -14.7, -130.8 (dragged into the pit in about
    7 s, down 0.32 to 0.58 s), -17.8 cm at 250, 500, 750. Cancelled. Seeds diverge: judge a second
    seed before building on one.
  - The pick, t15-250: it passed U and R1 with the highest rating, and beat t15-500 and t15-750
    10-0 each (17 pits). It was the reference duck until t20-500.
- **Postures** (trunk height and tilt over a bout, and how long the trunk touched the floor):
  standing 12.0 cm; t15-250 a low crouch, median 8.3 cm (lowest 5.9), 34°, its trunk never on the
  floor (its head was: "The bow", above); t13-500 9.3 cm, 26°; t10-500 4.5 cm, 57°, its trunk never
  on the floor either (its legs carry it; stay upright called it down); t7-250 lies with its trunk on
  the floor 73 to 88 % of a bout.
- **The candidate search** (stay upright 1.0, Win ground 4.0, warm-started from t15-250 and against
  it; ratings against t15-250):
  - t17 (seed 1): +132.3 cm at 250 (the reference in the pit 10 of 10, median 14.3 s, on its feet 54
    to 62 %, longest down 0.16 to 0.28 s, trunk median 7.2 cm, tilt 40°, its trunk never on the
    floor, its head on it),
    +27.6 at 500, +64.5 at 750.
  - t18 (seed 2): +134.0 at 250 (longest down under 0.30 s in 9 bouts of 10), +133.2 at 500, +63.0
    at 750.
  - Both peaked at 250. Round robin of the three within 5 cm (40 bouts each): t18-250 won 33 (9
    pits), t17-250 23 (23 pits), t18-500 4; but t18-250 and t18-500 sat down for up to 30 and 43 s
    against other ducks, where t17-250 was never down past 0.24 s. t18-750 lost 0-10 to t17-250, all
    by pit.
  - The pick, the owner's candidate then: t17-250, the only one that stays on its feet against
    other ducks, the most pits. It pulled bowed onto its head, and t23-250 replaced it.
- **The ducks trained before t15-250** rate against it: t7-250 -139.1 cm, t10-500 -138.6,
  t9r-1000 -132.1, each dragged into the pit in every bout, within 13 to 29 s. t15-250 against
  itself rates -1.7 cm.
- t19, the recipe as shipped (stay upright 0.0, Win ground 2.0, warm start and opponent t15-250),
  judged at 250 on the Mac: +131.5 cm against t15-250, the reference in the pit 10 of 10 (median
  29.7 s), backed 10 of 10. A low crouch: on its feet 5 to 11 %, tilt 46 to 47°, trunk median 7.2 cm,
  its trunk never on the floor, its head on it. t12, the same knobs before the ring, rated +140.3 cm
  at 500 against t7-250.

### Before the ring (2026-10-20 and 21: a 4 m lane with walls, a massless string, t7-250 the reference)

- t2, t3 (only the pit ends a bout): a duck that falls at 1 s lies on its side for the episode and is
  dragged in slowly; training never learns to stand. Fix: the fall ends a training episode (`fell`).
- t4, t5: each dove just under the fall line, propped, and held by friction: standing still earned
  0.41 of what walking back paid. Fix: `back_reward` 3.0 and a tighter std.
- t8r, t9r (from scratch, seeds 1 and 2): t8r crouched and held, +12.6 cm at 1000 and +10.5 at
  1500; t9r lay on its belly from 2 s into each bout and held, -18.3 cm at 1000.
- t10, t11, t12 (starting as the reference duck): each dragged it over the pit's edge in 10 bouts of
  10 (t10 +139.6 cm at 500, t11 +140.5 at 250, t12, the recipe as shipped then, +140.3 at 500); t10
  and t11 then decayed into lying prone by 1000 (-17.6 and +19.7 cm, no pit).
- The course comes in 2 m tiles (`TILE_M`): on the Arena's 40 m boxes mujoco_warp put a lying duck's
  contact 60 to 130 m away and whole worlds went NaN (t8, t9). Keep the tiles if you edit the course.
- `nconmax` 400 ran an L4 out of memory at 4096 envs; 200 holds the string's 31 to 40 contacts a
  world.

## The competition

- The Arena is the Hugging Face Space
  [pollen-robotics/microduck-arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena).
  The board ranks every entered policy by its rating in centimetres, higher first. Read it live:
  `curl -s https://pollen-robotics-microduck-arena.hf.space/api/events/tug-of-war/leaderboard`.
  Use it to tell the user what a place takes, never to promise one.
- Player points: +100 for a first publication in a challenge, +10 per publication (the first 3 per
  UTC day, all challenges together), +25 once per challenge for improving a personal best,
  +250/+150/+100 while holding first/second/third place. Podium points move with the rank; a
  withdrawn run loses its points. A weekly cup pays +250 to its champion, +150 to its finalist and
  +100 to each semi-finalist once its final has aired.
- The challenge's page on the Arena has the same rules in plain words, and the **Train my own robot**
  button that opens the workshop.
