# AGENTS.md

This repository holds one training folder per challenge open on the Microduck Arena
(`src/microduck_challenges/<folder>/`). A user clones it to train a policy for one challenge,
publish it to their Hugging Face account and enter it on the Arena.

**Before any work, read the AGENTS.md of the challenge the user is working on.** It lists the
files to change, how to run and judge a training, the challenge's rules and what is known to work:

| Challenge | Arena event | Guide |
|---|---|---|
| 2 m Sprint | `sprint-2m` | [src/microduck_challenges/sprint_2m/AGENTS.md](src/microduck_challenges/sprint_2m/AGENTS.md) |
| 1.5 m Hill Climb | `hill-climb-1m5` | [src/microduck_challenges/hill_climb_1m5/AGENTS.md](src/microduck_challenges/hill_climb_1m5/AGENTS.md) |
| 2 m Roller Sprint | `roller-sprint-2m` | [src/microduck_challenges/roller_sprint_2m/AGENTS.md](src/microduck_challenges/roller_sprint_2m/AGENTS.md) |
| Tug of War | `tug-of-war` | [src/microduck_challenges/tug_of_war/AGENTS.md](src/microduck_challenges/tug_of_war/AGENTS.md) |
| 6 m Sock Heist | `sock-heist-6m` | [src/microduck_challenges/sock_heist_6m/AGENTS.md](src/microduck_challenges/sock_heist_6m/AGENTS.md) |
| MicroDuck Got a Talent | `got-a-talent` | [src/microduck_challenges/got_a_talent/AGENTS.md](src/microduck_challenges/got_a_talent/AGENTS.md) |

If the user has not said which challenge, ask.

Shared facts:

- The robot model and the reward functions come from the `mjlab-microduck` package
  ([microduck_rl](https://github.com/pollen-robotics/microduck_rl)); they are not edited here.
  Its [AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md) holds
  the reward-design rules learned on this robot.
- Tests: `uv run --group workshop pytest` (CPU, under a minute). Run them before every training.
