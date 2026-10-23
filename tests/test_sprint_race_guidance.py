"""Gait-free race guidance preserves the complete-episode time objective."""

from types import SimpleNamespace

import pytest
import torch

import microduck_challenges  # noqa: F401
from microduck_challenges.sprint_2m import rewards


def test_air_time_allows_flight_without_heading_posture_or_speed_gates():
    times = torch.tensor([[0.1, 0.0], [0.1, 0.1], [0.0, 0.0],
                          [0.06, 0.22], [float("nan"), 0.1]])
    env = SimpleNamespace(scene={"feet_ground_contact": SimpleNamespace(
        data=SimpleNamespace(current_air_time=times))})
    actual = rewards.race_air_time_potential(env)
    torch.testing.assert_close(actual, torch.tensor([0.5, 1.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("failed", [False, True])
def test_air_time_guidance_cannot_change_finish_or_failure_return(failed):
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02,
        scene={"feet_ground_contact": SimpleNamespace(
            data=SimpleNamespace(current_air_time=torch.zeros(1, 2)))},
        _sprint_race=SimpleNamespace(done=torch.zeros(1, dtype=torch.bool),
            steps=torch.tensor([26]), settle_steps=25,
            elapsed=torch.tensor([0.02]), failed=torch.tensor([False]),
            timed_out=torch.tensor([False])))
    term = rewards.PotentialDifference(None, env)
    total = torch.zeros(1)
    for step, airborne in enumerate([False, True, True, False], start=1):
        env.scene["feet_ground_contact"].data.current_air_time[:] = 0.1 if airborne else 0.0
        race = env._sprint_race
        race.elapsed[:] = step * env.step_dt
        race.done[:] = step == 4
        race.failed[:] = failed and step == 4
        total += (term(env, potential=rewards.race_air_time_potential)
                  - rewards.race_time_cost(env)) * env.step_dt
    torch.testing.assert_close(total, torch.tensor([-50.0 if failed else -0.08]))
    term.reset()
    assert term.previous.item() == 0.0
