# AGENTS.md: 6 m Sock Heist

You are helping a user train a Microduck policy for the Arena's **6 m Sock Heist** (event
`sock-heist-6m`, task `Mjlab-SockHeist6m-MicroDuck`, this folder). The user may be new to
reinforcement learning. Work so that training stays:

- **simple**: the user decides the goal, the budget and what gets published; you do the mechanics;
- **transparent**: say what you change and why before each run, one commit per run, and report the
  Arena evaluator's numbers, never an impression from the curves;
- **efficient**: one change per run, a smoke test before any long run, every checkpoint judged,
  a failed run stopped instead of paid for.

Never launch a Hugging Face job, publish, enter the Arena or push without the user's explicit yes.

## The challenge

Walk 6 m to the socks of a giant microduck's nest, and freeze whenever the giant turns its head.
The rules, from `arena/events/sock-heist-6m.toml` in the installed Arena:

- **Course**: flat lane, 6.0 m long, 0.8 m wide, walls 0.15 m high. Start from the `STAND` pose,
  0.5 s untimed settle standing on the line, joints perturbed by ±0.01 rad, heading by ±2°. The
  clock starts on a green.
- **The giant**: past the line, its back to the lanes. Each seed draws its own timetable from three
  ranges, and every entrant faces the same five: a first green of 0.6 to 1.0 s (short, so a duck
  that bolts off the line has to stop as hard as it started), then a red of 1.4 to 2.4 s, a green
  of 1.6 to 3.0 s, a red, and so on until the time limit. Every phase is a whole number of control
  steps (20 ms).
- **A red**: the giant takes **0.4 s** (`turn_s`) to turn its head to the lanes, and nothing counts
  while it turns. The **mark** is where the duck's trunk stands on the floor when the head has
  finished turning. Until the head turns away, a trunk more than **3 cm** across the floor from its
  mark (`tolerance_m`) is **caught**: its motors switch off and it collapses.
- **The lane's heading hold** (`[steering]`): on green the Arena turns the duck toward the lane's
  centre line 0.5 m ahead, through the yaw rate of its command:
  `clip(2 × wrap(atan2(−y, 0.5) − yaw), −1, 1)` rad/s, `y` the trunk's offset from the centre line
  (+ left), `yaw` its heading. Not in the settle, nor under a red.
- **Sweep**: every `vx` from 0.20 to 1.20 m/s in 0.05 steps is tried once (seed 1); the 3 fastest
  are re-run on seeds 1–5; the score is the best mean time over those 5 seeds.
- **Fall**: trunk tilted more than 60° (gravity z above −0.5) for 0.3 s. A fall, a catch, or no
  finish within the **40 s** time limit (`time_limit_s`) counts 40 s. **Finish 5/5 first, then get
  faster.**

## The policy

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish`
checks the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers: twist (3), head (4), body (6). Head and body commands are zeros.

The policy never sees the giant: **its only signal is its command**. On red, from the first step of
the turn until the red ends, the twist is zero. On green it is `[vx, 0, hold]`, `vx` the speed the
Arena times it at and `hold` the lane's yaw rate. So a policy that stands still on a zero twist and
walks on its command is all the challenge asks.

## The recipe

- **The learner starts as the Arena's reference walker** (`walker.onnx`, the Arena's
  `BEST_alpha_walking.onnx`; `WalkerActor`, `RL_CFG`'s actor: its layers and observation
  normalizer, action noise std `WARM_START_STD` 0.3, the critic fresh), as the Tug of War's
  reference was. It already walks and stands; it was never trained to stop on cue. To train from
  scratch anyway, add `--agent.actor.class-name MLPModel`.
- **The rules are the event's, from iteration 0** (`lights.py`): each training episode draws its
  reds from the event's three ranges, in whole steps, as the Arena draws a seed's timetable; the
  twist is zero through the settle and every red, from its first step; the mark and the watch fall
  on the Arena's steps; on green the yaw rate is the lane's hold, the lane being the spawn line
  along +x. Each episode holds one speed, drawn over the sweep. `lights.py` copies the event's
  numbers (a Hugging Face job installs no Arena); the tests pin every copy against the Arena.
- **`freeze`** pays `exp(−(drift / 3 cm)²)` on every watched step: nothing else sees a few
  centimetres of creep. **`caught`** ends the episode, as a fall does, with no extra penalty, so
  standing is never made cheaper than walking. The critic alone is told the light (standing,
  watched, seconds to the next change, the drift over 3 cm) and the lane (offset, heading).
- **No pushes**: a push while watched catches a duck that did nothing, and the Arena pushes nobody.
- **No walls in training**: the hold keeps the duck on the lane, and `off_lane` ends the episode of one
  that strays more than 0.3 m from it. heist3 learned to ignore its yaw-rate command, so the hold could
  not keep it straight and it veered right: off the lane on 83 % of timed steps in training, caught or
  fallen against the wall within ~1 s on the Arena (0/25), while `progress` paid it all the same.
- **No `heading_hold`, no standing episodes**: the hold commands the yaw rate, and the reds stand
  about 45 % of every episode.
- **1000 iterations, a checkpoint every 100**: warm-started Tug of War runs peaked early and fell
  between 250-iteration checkpoints. Judge every checkpoint; keep the best, not the last.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `challenge.toml` `[params]` | The five knobs below | Yes, first. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 1000, `seed` 1 | Yes, when agreed with the user |
| `env.py` | The whole environment: rewards, commands, domain randomization, curricula, terrain, `RL_CFG` (network, PPO, checkpoints every 100 iterations) | Yes, once the knobs are not enough |
| `rewards.py` | `freeze`, `progress`, `caught`, and the critic's `light_state` and `lane_state` | Yes, to add a reward term (then add it to `cfg.rewards` in `env.py`) |
| `lights.py` | The event's numbers and the Arena's arithmetic, and `LightsCommand`, the twist | Only the command term; never the copies |
| `walker.py`, `walker.onnx` | The learner's start | No |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | Sets in `env.py` |
|---|---|---|---|---|
| `forward_reward` | Walk on green | 2.0 | 0.5–5.0 | `track_linear_velocity` weight |
| `progress_reward` | Walk forward | 8.0 | 0.0–16.0 | `progress` weight |
| `freeze_reward` | Freeze while watched | 2.0 | 0.0–6.0 | `freeze` weight |
| `upright_reward` | Stay upright | 2.0 | 0.5–5.0 | `upright` weight |
| `step_reward` | Lift your feet | 3.0 | 0.0–6.0 | `air_time` weight |

The reward terms as they start:

| Term | Weight | Why |
|---|---|---|
| `track_linear_velocity` | `forward_reward` 2.0, std √0.1 | Progress on green at the commanded speed, and stillness on red (a zero command). At 0.45 m/s standing earns 13 % of walking; the Tug of War's t4/t5 never walked at 41 %. |
| `track_angular_velocity` | 2.0, std √0.5 | The lane's hold steers through the yaw rate, so tracking it is the steering. |
| `progress` | `progress_reward` 8.0 | Forward speed along the duck's own nose, up to the command, nothing on red. Without it a statue out-earned the walker under the lights (8.92 a step against 7.91): the walker reaches only ~0.45 of its command, which `track_linear_velocity` barely pays over standing, and heist1 and heist2 learned to creep. |
| `freeze` | `freeze_reward` 2.0, std 0.03 m | The rule is 3 cm, and nothing else sees a few centimetres of creep: tracking's std is 0.32 m/s. |
| `upright` | `upright_reward` 2.0, std √0.05 | The sprint's. A forward lean is what ends in a fall on a stop. |
| `air_time` | `step_reward` 3.0, threshold 0.01 | The sprint's. It pays steps only on a non-zero command, so stepping in place under a red earns nothing (on green it pays stepping in place too; `progress` pays walking over it). |
| `pose` | 1.0 | The sprint's: on a zero command its standing std pulls the legs to the standing pose. |
| gait and smoothness terms | the sprint's | Unchanged shaping of the walk the walker already has. |
| `caught` (termination) | none | It costs every reward still to come (γ 0.99), as a fall does. |
| `off_lane` (termination) | none | The trunk more than 0.3 m off its lane (the Arena's half-width less 0.1 m) ends the episode, as the Arena's wall would. |

Do not change, or the Arena cannot race the policy or the tests fail:

- the robot (`MICRODUCK_WALK_ROBOT_CFG`) and the 61-number observation layout: a command slot the
  task does not use keeps its tiny sampling range, never deleted;
- `experiment_name = "sock_heist_6m"` in `RL_CFG`: runs must land in `logs/rsl_rl/sock_heist_6m/`;
- the copies in `lights.py` (the event's numbers, the timetable's draw, the light steps, the hold)
  and the tests that pin them against the Arena;
- `walker.onnx`, the Arena's walker byte for byte;
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

Measure the walker on this machine: it is the bar to beat, and its number tells you how far this
machine's scores sit from the Space's.

```bash
uv run --group workshop arena score src/microduck_challenges/sock_heist_6m/walker.onnx --event sock-heist-6m
```

Agree with the user before the first run:

- the goal: a first entry, beating the walker, a given place on the board;
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
   uv run train Mjlab-SockHeist6m-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
       --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
   rm -rf logs/rsl_rl/sock_heist_6m/*_smoke
   ```

5. **Commit** the change, its message the hypothesis:
   `git add src/microduck_challenges/sock_heist_6m && git commit -m "<hypothesis>"`. `publish`
   writes that commit into the policy's recipe, and refuses a run started from uncommitted changes.
6. **Launch** in the background, the output in a log under `logs/` (git-ignored). `<name>` is short,
   unique, and says the change (`freeze4`, `upright3`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-SockHeist6m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-sockheist6m-microduck-<name>
   ```

   A local run writes `logs/rsl_rl/sock_heist_6m/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt`
   every 100 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the
   same tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-sockheist6m-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint, not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/sock_heist_6m/<run> Train/mean_episode_length Episode_Termination/caught Episode_Termination/fell_over Episode_Reward/freeze Episode_Reward/track_linear_velocity
  ```

  What to look for: `Train/mean_episode_length` toward 1000 (a full 20 s episode), `caught` and
  `fell_over` falling. A warning sign: `freeze` rising while `track_linear_velocity` falls, a duck
  learning to stand everywhere, which is never caught for it; so are `nan_state` terminations, and
  `Episode_Termination/off_lane` rising. The curves alone fooled the first four runs (episodes at the
  cap, nothing caught, yet a creeping or veering duck on the Arena): judge the checkpoints. A run
  that has collapsed and not recovered within a few checkpoints: tell the user and propose stopping it.

### Judging every checkpoint

The curves are not the score. Judge each checkpoint with the Arena's own evaluator:

- If the user has the workshop open (`uv run --group workshop workshop sock-heist-6m`), it finds
  every run under `logs/rsl_rl/`, yours included, and evaluates each checkpoint as it arrives. Read
  its results instead of evaluating again: `logs/rsl_rl/sock_heist_6m/<run>/workshop/model_<N>.json`
  (`score`, `seeds_finished`, `command_vx`).
- Otherwise export each checkpoint as the real publish does (`--dry-run` publishes nothing, the
  observation normalizer baked in, into `publish-eval/`) and score it:

  ```bash
  run=logs/rsl_rl/sock_heist_6m/<run>
  for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
    uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
    echo "checkpoint $n: $(uv run --group workshop arena score publish-eval/policy.onnx --event sock-heist-6m | grep -E '^(scored at|score )' | paste -sd' ' -)"
  done
  ```

Then see why each seed of a checkpoint ended, at the speed the Arena scores it at (or `--speed V`),
over seeds 1 to 5 (or `--seeds N`):

```bash
uv run --frozen --group workshop python -m microduck_challenges.sock_heist_6m.check publish-eval/policy.onnx
```

It prints `scored at V m/s` (the speed the Arena scores it at), a line per seed, then the tally:

```
seed 2 · finish 18.80 s · 5 reds met · worst drift 0.6 cm · 0.66 m/s on green
finished 25/25 · caught 0 (early 0) · fell 0 (on a stop 0, on a start 0) · out of time 0 (never left the line 0) · mean 17.17 s
```

- **`seed 2 · finish 18.80 s`**: the seed and how it ended, with its time. The other endings:
  `caught at 1.56 s, 0.46 s into the watch, 0.49 m/s as the red began` (the time, how long after the
  head finished turning, and the speed as the red began: `caught early` within 0.2 s of the watch's
  start, still moving, `caught late` after, creeping); `fell at 3.20 s on a stop, tipping 0.30 s after
  a red began` (or `on a start`, within 0.6 s of a red ending, or `on green`, or `under a red`);
  `out of time at 2.10 m`; `never left the line (0.12 m)` (ended less than 0.5 m from the line).
- **`5 reds met`**: the reds that began before the seed ended.
- **`worst drift 0.6 cm`**: the largest distance from a mark while watched, the tolerance being 3 cm.
- **`0.66 m/s on green`**: the mean speed along the lane over the green steps.
- **The tally**: `finished` seeds out of those replayed; `caught` (of which `early`); `fell` (of which
  on a stop and on a start); `out of time` (of which `never left the line`); `mean`, over all the
  seeds, an unfinished one counted at the time limit.

Pick by score: 5/5 finishes first, then the lowest time. The best checkpoint is often not the last.
Scores on this machine are indicative: only the Space times an entry, and simulations on different
CPU architectures drift apart after a few seconds, while a crossing lasts much longer. GPU training
varies a lot from run to run: before building on a surprising result, good or bad, run the same
commit with another `--agent.seed`.

### Reporting

After each run, tell the user in a few lines:

```
run freeze4 (commit 1a2b3c4: freeze_reward 2 → 4, to stop sooner once the head turns)
best: checkpoint 300, 41.20 s, 5/5 at 0.70 m/s   (walker <its score on this machine>)
others: 100 60.00 s 0/5 · 200 47.90 s 4/5 · 400 44.10 s 5/5
check at 300: finished 5/5 · caught 0 (early 0) · fell 0 (on a stop 0, on a start 0) · out of time 0 (never left the line 0) · mean 41.20 s
next: <your proposal and why>
```

### Publishing, on the user's yes

Publishing makes the policy public on their Hugging Face account and enters it on the Arena, which
races it and prints its time and the page of its result:

```bash
uv run publish --run logs/rsl_rl/sock_heist_6m/<run> --checkpoint <N> --repo <user>/microduck-<name> --no-private --enter
```

`--no-private` because the Arena reads public repos only; add `--livery <colour>` for its colour.
For the policy's code link to open, push the commit to the user's fork first (`origin` must be their
fork, never `pollen-robotics`; the workshop's home sets that up).

## What is known about this challenge

Every number below was measured on a Mac (indicative), except those marked x86, which an
Hugging Face Job timed on the Space's image: the Arena's x86 physics decides.

### The reference walker

It was never trained to stop and start again.

| | Mac | x86 |
|---|---|---|
| Score | 24.21 s at 1.00 m/s, 5/5 (seeds 25.14, 24.21, 24.39, 22.57, 24.75) | 24.16 s at 1.00 m/s, 5/5 |

- **The sweep**: it finishes from 0.40 to 1.00 m/s; from 1.05 m/s it falls or is caught.
- **Replayed at 1.00 m/s over 25 seeds, it finishes 24** with a mean finish of 24.01 s (21.42 s to
  26.57 s), **and is caught once** (seed 23, 0.46 s into the watch, at 1.56 s). None fell and none
  stayed at the start; the worst drift while watched is 3.0 cm (2.7 cm among the finishers).

### The six trainings

Each started as the walker, 4096 envs, 1000 iterations, seed 1 (heist2: seed 2), on an `l4x1` job;
one change each, all judged on every checkpoint (100 to 900, and 999). Best checkpoint, its Mac
score and finishes, and its CHECK tally over 25 seeds:

| Run | Commit | What it changed | Best | Mac score | CHECK 25 seeds |
|---|---|---|---|---|---|
| heist1, heist2 | 29dfda3 | The recipe as first committed | 100 each | heist1: none finishes (0/5, out of 60 s); heist2: 57.80 s 5/5 | heist1: 0/25 out of time (0.12 m/s on green); heist2: 20/25 out of time 5 (0.19 m/s) |
| heist3 | 69c387f | `progress` at weight 8 | 200 | 29.97 s 5/5 at 0.20 m/s | 22/25: caught 2, fell 1 |
| heist4 | 96960f7 | `progress` at weight 12 | 100 | 45.69 s 3/5 | 17/25: caught 2, fell 4, out of time 2 |
| **heist5** | **2efb9e1** | `progress` at weight 8 and the `off_lane` termination | **300** | **16.98 s 5/5 at 1.20 m/s** | **25/25** |
| heist6 | dff30d8 | heist5, `progress` along the lane | none | canceled once heist5-300 passed, unjudged | |

What each run showed:

- **heist1 and heist2 crept.** The training curves looked healthy (episodes at the 1000-step cap,
  nothing caught, nothing fallen), yet on the Arena the policy crept at 0.03 to 0.19 m/s on green and
  crept slower checkpoint after checkpoint. The reward made standing optimal: a statue earned 8.92 a
  step, the walker 7.91, a duck that walks on green and freezes on red 8.29, because the walker
  reaches only about 0.45 of its command, which `track_linear_velocity` barely pays over standing.
  A heavier forward reward would not have fixed it. `progress` did: measured on the training
  environment, walking now beats standing by 2.0 a step at 0.8 and 1.0 m/s.
- **heist3 and heist4 walked, then ignored the turn command.** Walking paid, but `progress` pays
  speed along the nose whatever the heading, and the training floor has no walls, so veering was
  free. From checkpoint 200 on the policy ignored its yaw-rate command and veered right (off a 0.3 m
  half-lane on 83 % of timed steps in training): the Arena's wall caught or felled it within about a
  second (heist3-999 at 0.75 m/s over the same 25 timetables: 0/25 on the Arena, 20 caught and 4 fell, all touching the right wall; 0 caught and 0 fell in training). Weight 12 was worse than 8:
  both collapsed near iteration 300 (caught peaked near 5 per step) and recovered by 600 in the
  curves, but not on the Arena.
- **heist5's `off_lane` termination worked.** It ends an episode whose trunk strays more than 0.3 m
  from its lane, as the Arena's wall would. Steering stopped being free: the policy follows its turn
  command again (at 0.75 m/s, commands -1 / -0.5 / 0 / +0.5 / +1 walked -0.76 / -0.26 / +0.18 / +0.55
  / +0.80 rad/s at checkpoint 300) and the first three checkpoints all beat the walker.
- **Checkpoints differ**; judge each. heist5: 100, 200 and 300 all finish 5/5 and pass CHECK 25/25
  (Mac 23.03 s at 1.20 m/s, 19.34 s at 0.95, 16.98 s at 1.20; 0.48, 0.58 and 0.66 m/s on green),
  each faster than the last. heist3 finished nothing after 200; heist4's best was 100, and later only 200, 400 and 999 finished anything (1/5, 1/5, 4/5).
- Scores before heist5 were taken with the placeholder 60 s time limit; the event's is 40 s, under which
  the finishes slower than 40 s no longer count: heist2-100 (57.80 s), and heist4-100, 200, 400 and 999
  (45.69 s, 53.80 s, 54.46 s, 45.98 s). heist3's and heist5's are unchanged.

### The reference

heist5, checkpoint 300 (training commit `2efb9e1`: the recipe as shipped). Trained by:

```bash
uv run --no-config --no-sync train Mjlab-SockHeist6m-MicroDuck --env.scene.num-envs 4096 \
    --agent.max-iterations 1000 --agent.seed 1 --agent.logger tensorboard --agent.run-name heist5 \
    --hf-jobs --detach --no-wandb --namespace Nico-robot --ckpt-repo Nico-robot/mjlab-sockheist6m-microduck-heist5
```

(canceled after checkpoint 300, once it had passed the walker on the Mac and on x86). Exported by
`publish --dry-run` and scored with `arena score publish-eval/policy.onnx --event sock-heist-6m`:

| | Mac | x86 |
|---|---|---|
| Score | 16.983 s at 1.20 m/s, 5/5 (seeds 16.37, 18.80, 17.11, 16.11, 16.53) | 16.982 s at 1.20 m/s, 5/5 (seeds 16.41, 18.80, 17.105, 16.055, 16.54) |
| The walker's | 24.210 s at 1.00 m/s | 24.159 s at 1.00 m/s |

Its CHECK over 25 seeds at 1.20 m/s: `finished 25/25 · caught 0 (early 0) · fell 0 (on a stop 0, on a
start 0) · out of time 0 (never left the line 0) · mean 17.17 s`; worst drift while watched 0.3 to
0.8 cm, and 0.66 m/s on green: it walks at about half its command and stops dead on every red.

Published by the release as `pollen-robotics/microduck-sock-heist-6m-reference`; fetch it with:

```bash
uv run hf download pollen-robotics/microduck-sock-heist-6m-reference policy.onnx --local-dir logs/reference/sock-heist-6m
```
