# Microduck challenges

One folder per challenge open on the Microduck Arena. Each is a complete,
working training environment: train it unchanged and the robot competes.

| Folder | Arena event | Task |
|---|---|---|
| sprint_2m | sprint-2m | Mjlab-Sprint2m-MicroDuck |
| got_a_talent | got-a-talent | Mjlab-GotATalent-MicroDuck |
| hill_climb_1m5 | hill-climb-1m5 | Mjlab-HillClimb1m5-MicroDuck |
| roller_sprint_2m | roller-sprint-2m | Mjlab-RollerSprint2m-MicroDuck |
| tug_of_war | tug-of-war | Mjlab-TugOfWar-MicroDuck |
| sock_heist_6m | sock-heist-6m | Mjlab-SockHeist6m-MicroDuck |

Training with a coding agent (Claude Code, Codex, Cursor…)? Each challenge folder has an
`AGENTS.md` that tells it which files to change, how to run and judge a training, and the
challenge's rules; point the agent at it.

MicroDuck Got a Talent is a performance, not a race: the Arena plays the entry's
`timeline.json` for up to 20 s on an open stage, and visitors vote.
`got_a_talent/timeline.json` is Pollen's: `twist[0]` is the sit flag (1 sits,
0 stands) and `head` is 4 deltas in radians from the default pose (neck_pitch,
head_pitch, head_yaw, head_roll). Check a policy against a timeline with
`uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.check policy.onnx [--timeline FILE]`,
and publish the timeline with the policy: `uv run publish … --timeline src/microduck_challenges/got_a_talent/timeline.json`.
A performance can also make the duck heard: `"triggers": [rt, lt]` in a keyframe (RT quacks, LT
wheees), a `song` of notes beside the keyframes, and a `voice` that tunes it (the Arena's
[Got a Talent guide](https://pollen-robotics-microduck-arena.hf.space/#/e/got-a-talent/guide)).
A performance can also be a troupe of up to 8 ducks in one physics. Each duck has its own policy
(yours, or any public one on the Hub pinned to a commit, on legs or rollers), place, keyframes,
voice and song: see the guide's Troupe and Placement sections.

The Tug of War is a duel: two ducks neck to neck in a 3 m ring, over a pit, each pulling a 1 m string by walking
backwards at `-0.30 m/s`; only the pit decides, and the marker if nobody falls in. The Arena rates a
policy by five 45 s bouts against its reference duck, in centimetres of ground won plus a pit bonus
(higher wins), and each player's latest entry plays the weekly cup. The recipe starts the learner as that reference
duck and trains it against it for 1000 iterations; it peaks early (250 to 500), so rate every
checkpoint and keep the best, not the last. Its `stay_upright` knob, off by default, makes a duck walk upright, as the robot walks.
Play two policies against each other (B defaults to the reference) with
`uv run --frozen --group workshop python -m microduck_challenges.tug_of_war.check A.onnx [B.onnx] [--bouts 10]`.

The Sock Heist is a race with a rule: 6 m to the socks of a giant microduck's nest, and freeze whenever the giant
turns its head to the lanes, or be caught. The policy sees only its command, zero on red, and each seed's timetable is
the same for every entrant. Say why each seed of a policy ended with
`uv run --frozen --group workshop python -m microduck_challenges.sock_heist_6m.check policy.onnx [--speed V] [--seeds N]`.
The recipe starts the learner as the reference walker.

## The workshop

Each challenge's page on the Arena has a **Train my own robot** button with one line to copy:

```bash
uv run https://pollen-robotics-microduck-arena.hf.space/workshop.py sprint-2m
```

It works in the clone you run it in (or clones this repo next to you) and opens the workshop in your
browser, on that challenge. Its home says what is set up (Hugging Face Jobs billing, your fork, an
NVIDIA GPU) and lists the challenges. A challenge's page says which files to change, commits them
before each training, trains on your NVIDIA GPU or on Hugging Face Jobs, and keeps one ticket per
training: every checkpoint is evaluated with the Arena's own rules as it arrives. A training's page
replays any checkpoint, and **Publish and compete** publishes the policy to your Hugging Face account
**and enters** it on the Arena, then opens its result. On MicroDuck Got a Talent, a training's page
also edits `timeline.json`, your performance, places its ducks on a plan of the stage, and plays it
with the checkpoint you pick. Publish and compete publishes every duck of a troupe in one click.
From a clone,
`uv run --group workshop workshop` opens the home and `uv run --group workshop workshop sprint-2m`
opens one challenge.

## Train from a shell

Needs Python 3.12 and `uv`. Training needs an NVIDIA GPU, or Hugging Face Jobs.

```bash
git clone https://github.com/pollen-robotics/microduck-challenges && cd microduck-challenges
uv sync --group workshop
uv run hf auth login   # once: Hugging Face Jobs and publish run under your account
uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
    --agent.seed 1 --agent.logger tensorboard --agent.run-name first
# no GPU: add --hf-jobs --no-wandb
```

A challenge's knobs are in its `challenge.toml` (`[params]`); its whole
environment is in `env.py`. Edit either, commit, train. To add a custom
reward, write the function in a `rewards.py` next to the challenge's `env.py`
and add it to `cfg.rewards` in `env.py`.

`--hf-jobs` trains on Hugging Face's infrastructure instead of a local GPU;
nothing lands in `logs/rsl_rl/` here. Its checkpoints go to the Hub repo the
submission prints (`[ckpt] checkpoints -> <repo>`), mirroring `logs/rsl_rl/` —
bring them back before publishing:

```bash
uv run hf download <that repo> --local-dir logs/rsl_rl
```

Then publish and enter it in one command:

```bash
uv run publish --run logs/rsl_rl/<folder>/<run> --repo <you>/microduck-<name> --no-private --enter
```

It uploads the policy to your Hugging Face account, enters it on the
challenge's event on the Arena, and prints its time and the page of its result.
`publish` creates a new Hub repo private by default, and the Arena reads public
repos only: hence `--no-private`.

## Tests

```bash
uv run --group workshop pytest
```

(the tests import the Arena, which lives in the `workshop` group).
