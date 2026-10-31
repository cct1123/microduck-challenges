# AGENTS.md: 2 m Roller Sprint

You are helping a user train a Microduck policy for the Arena's **2 m Roller Sprint** (event
`roller-sprint-2m`, task `Mjlab-RollerSprint2m-MicroDuck`, this folder). The user may be new to
reinforcement learning. Work so that training stays:

- **simple**: the user decides the goal, the budget and what gets published; you do the mechanics;
- **transparent**: say what you change and why before each run, one commit per run, and report the
  Arena evaluator's numbers, never an impression from the curves;
- **efficient**: one change per run, a smoke test before any long run, every checkpoint judged,
  a failed run stopped instead of paid for.

Never launch a Hugging Face job, publish, enter the Arena or push without the user's explicit yes.

## The challenge

Skate 2 m on roller blades on a flat lane as fast as possible. The rules, from
`arena/events/roller-sprint-2m.toml` in the installed Arena:

- **Course**: flat lane, 2.0 m long, 0.8 m wide, walls 0.15 m high. The duck wears roller blades:
  two passive, free-spinning wheels per foot. Start from the `STAND` pose, 0.5 s untimed settle,
  joints perturbed by ±0.01 rad, heading by ±2°.
- **Command**: the policy reads `[vx, 0, 0]`, constant for the whole run; head and body commands are
  zeros. It is never told its heading or its position in the lane.
- **Sweep**: every `vx` from 0.30 to 2.00 m/s in 0.10 steps is tried once (seed 1); the 3 fastest
  are re-run on seeds 1–5; the score is the best mean time over those 5 seeds.
- **Fall**: trunk tilted more than 60° (gravity z above −0.5) for 0.3 s. A fall, or no finish within
  **40 s**, counts 40 s: one fall among four 3 s finishes scores 10.4 s, worse than five 4 s
  finishes. **Finish 5/5 first, then get faster.**

## The policy

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish`
checks the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers: twist (3), head (4), body (6). Joint order: left leg (hip_yaw,
hip_roll, hip_pitch, knee, ankle), neck_pitch, head_pitch, head_yaw, head_roll, right leg (same
order). The wheels are not actuated and not observed: the policy pushes, glides and steers with its
14 joints. The Arena simulates the same robot with the same BAM actuator model as training (MuJoCo,
implicitfast integrator, 10/20 contact solver). Every randomization of training takes its nominal
value there, except the servos' command delay, drawn as in training; there are no pushes and no
observation noise.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `challenge.toml` `[params]` | The two knobs below | Yes, first. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 3000, `seed` 1 | Yes, when agreed with the user |
| `env.py` | The whole environment: rewards, commands, domain randomization, curricula, terrain, `RL_CFG` (network, PPO, checkpoints every 250 iterations) | Yes, once the knobs are not enough |
| `rewards.py` | Does not exist yet: create it for a custom reward term, then add the term to `cfg.rewards` in `env.py` | Yes |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | Sets in `env.py` |
|---|---|---|---|---|
| `speed_reward` | Wheel speed reward | 10.0 | 2.0–20.0 | `wheel_speed` weight |
| `upright_reward` | Stay upright | 2.0 | 0.5–5.0 | `upright` weight |

How the training reads the command: `vx` = 0 coasts, `vx` > 0 pushes, `vx` < 0 brakes. `wheel_speed`
is the only positive task reward (the wheels must actually spin); `braking`, `skating_air_time`,
`glide`, `single_support`, `gait_symmetry`, `forward_lean` and `heading_hold` shape the skating style.
Each term's comment in `env.py` says why it has its weight.

Do not change, or the Arena cannot race the policy or the tests fail:

- the robot (`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`: 14 actuated joints and 4 `passive_*` wheel joints
  interleaved after each ankle; select joints by name, never by index) and the 61-number
  observation layout: a command slot the task does not use stays zero-padded, never deleted;
- `experiment_name = "roller_sprint_2m"` in `RL_CFG`: runs must land in `logs/rsl_rl/roller_sprint_2m/`;
- the forward command range must start at or below the sweep's 0.30, with no sideways or turning
  command (its top, 0.6, is deliberate: see below);
- the wheels stay frictionless, as on the Arena: no wheel-bearing friction randomization or
  curriculum (a test checks it);
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

Measure Pollen's reference on this machine: it is the bar, and its number tells you how far this
machine's scores sit from the Space's.

```bash
uv run hf download pollen-robotics/microduck-roller-sprint-2m-reference policy.onnx --local-dir logs/reference/roller-sprint-2m
uv run --group workshop arena score logs/reference/roller-sprint-2m/policy.onnx --event roller-sprint-2m
```

It scores 3.121 s, 5/5, at 1.00 m/s (checkpoint 2999 of the recipe as it stands in this folder).

Agree with the user before the first run:

- the goal: a first entry, beating the reference, a given place on the board;
- the budget: how many runs; local GPU, or Hugging Face Jobs (default hardware `l4x1`, $0.80 per
  hour, billed to their account, which needs a payment method; the 3000-iteration recipe takes
  about 1 to 2.5 hours);
- what you may launch without asking each time (by default: local runs you announce and start;
  each Hugging Face job asked for).

### Each run

1. **State the hypothesis** in one sentence: what you change, why, and what you expect to see.
2. **Change one thing**, a knob first. Combine changes only on purpose, and say so.
3. **Test**: `uv run --group workshop pytest`.
4. **Smoke test** (under a minute, also on a machine without a GPU):

   ```bash
   uv run train Mjlab-RollerSprint2m-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
       --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
   rm -rf logs/rsl_rl/roller_sprint_2m/*_smoke
   ```

5. **Commit** the change, its message the hypothesis:
   `git add src/microduck_challenges/roller_sprint_2m && git commit -m "<hypothesis>"`. `publish` writes
   that commit into the policy's recipe, and refuses a run started from uncommitted changes.
6. **Launch** in the background, the output in a log under `logs/` (git-ignored). `<name>` is short,
   unique, and says the change (`speed12`, `upright3`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-RollerSprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-rollersprint2m-microduck-<name>
   ```

   A local run writes `logs/rsl_rl/roller_sprint_2m/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt` every
   250 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the same
   tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-rollersprint2m-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint (every 250 iterations, several minutes apart), not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/roller_sprint_2m/<run> Train/mean_episode_length Episode_Termination/fell_over Episode_Reward/wheel_speed
  ```

  Healthy: `Train/mean_episode_length` climbs toward 1000 (a full 20 s episode), `fell_over` drops
  toward 0, `wheel_speed` keeps growing. Warning signs: total reward rising while `wheel_speed` stays
  flat (the policy farms the other terms); a drop right at a curriculum stage (iterations 250, 500,
  1000); `nan_state` terminations. A run that has collapsed and not recovered within about 500
  iterations: tell the user and propose stopping it.

### Judging every checkpoint

The curves are not the score. Judge each checkpoint with the Arena's own evaluator:

- If the user has the workshop open (`uv run --group workshop workshop roller-sprint-2m`), it finds every
  run under `logs/rsl_rl/`, yours included, and evaluates each checkpoint as it arrives. Read its
  results instead of evaluating again: `logs/rsl_rl/roller_sprint_2m/<run>/workshop/model_<N>.json`
  (`score`, `seeds_finished`, `command_vx`). Its page also replays any checkpoint: the best way to
  show the user a run.
- Otherwise:

  ```bash
  run=logs/rsl_rl/roller_sprint_2m/<run>
  for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
    uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
    echo "checkpoint $n: $(uv run --group workshop arena score publish-eval/policy.onnx --event roller-sprint-2m | grep -E '^(scored at|score )' | paste -sd' ' -)"
  done
  ```

  `--dry-run` exports exactly as the real publish (the observation normalizer baked in) into
  `publish-eval/`, and publishes nothing.

Pick by score: 5/5 finishes first, then the lowest time. The best checkpoint is often not the last.
Scores on this machine are indicative: only the Space times an entry, and simulations on different
CPU architectures drift apart after a few seconds. GPU training varies a lot from run to run: before
building on a surprising result, good or bad, run the same commit with another `--agent.seed`.

### Reporting

After each run, tell the user in a few lines:

```
run speed12 (commit 1a2b3c4: wheel_speed weight 10 → 12, to push harder)
best: checkpoint 2250, 2.98 s, 5/5 at 1.10 m/s   (reference 3.13 s; its run's best 3.02 s)
others: 1000 3.12 s 5/5 · 1750 9.41 s 4/5 · 2999 3.05 s 5/5
curves: fell_over 0.02, episode length 990 of 1000
next: <your proposal and why>
```

### Publishing, on the user's yes

Publishing makes the policy public on their Hugging Face account and enters it on the Arena, which
races it and prints its time and the page of its result:

```bash
uv run publish --run logs/rsl_rl/roller_sprint_2m/<run> --checkpoint <N> --repo <user>/microduck-<name> --no-private --enter
```

`--no-private` because the Arena reads public repos only; add `--livery <colour>` for its colour.
`publish` reads the roller blades off the run's robot model and declares them, so the Arena races
the policy on its roller rig.
For the policy's code link to open, push the commit to the user's fork first (`origin` must be their
fork, never `pollen-robotics`; the workshop's home sets that up).

## What is known about this challenge

- **The training stops at 0.6 m/s on purpose.** Widening the forward command toward the sweep's
  2.00 collapsed training twice (straight to 2.00, and with a curriculum widening it in stages):
  `wheel_speed` grows with the command, and at high commands it out-weighed staying upright. Capping
  the command factor of every push reward at 0.6 did not save it either. The policy still races the
  whole sweep: commands above 0.6 are pushes at full effort, and the reference's best is commanded
  at 1.00 m/s here, 1.10 on the Space.
- **Wheels without friction.** Randomizing wheel-bearing friction (as the source task did) trained
  policies that broke on the Arena, whose wheels are frictionless. It is gone; keep it out.
- **Judge every checkpoint.** Every checkpoint of the reference run from 1000 on finishes 5/5, but
  between 3.02 s (checkpoint 1500) and 3.26 s (2750): the published 2999 (3.13 s) is not its best.
  Before the Arena drove the BAM actuator model, the same checkpoints swung between 3.1 s and 0/5:
  skating is a knife-edge balance, sensitive to any difference between training and the Arena.
- **Match the Arena's physics.** A domain randomization term that adds something the Arena does not
  simulate trains a policy that fails there.

## The competition

- The Arena is the Hugging Face Space
  [pollen-robotics/microduck-arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena).
  The board ranks every entered policy by score, then by finishes, then by who got there first.
  Read it live:
  `curl -s https://pollen-robotics-microduck-arena.hf.space/api/events/roller-sprint-2m/leaderboard`
  (each row: `user`, `policy_name`, `score`, `seeds_finished`, `command_vx`). Use it to tell the
  user what a place takes, never to promise one.
- Player points: +100 for a first publication in a challenge, +10 per publication (the first 3 per
  UTC day, all challenges together), +25 once per challenge for improving a personal best,
  +250/+150/+100 while holding first/second/third place. Podium points move with the rank; a
  withdrawn run loses its points.
- The challenge's page on the Arena has the same rules in plain words, and the **Train my own robot**
  button that opens the workshop.
