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
@pytest.mark.parametrize("terminal_airborne", [False, True])
def test_air_time_guidance_cannot_change_finish_or_failure_return(failed, terminal_airborne):
    from microduck_challenges.sprint_2m.env import make_env_cfg

    time_weight = make_env_cfg().rewards["race_time"].weight
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02,
        scene={"feet_ground_contact": SimpleNamespace(
            data=SimpleNamespace(current_air_time=torch.zeros(1, 2)))},
        _sprint_race=SimpleNamespace(done=torch.zeros(1, dtype=torch.bool),
            steps=torch.tensor([26]), settle_steps=25,
            elapsed=torch.tensor([0.02]), failed=torch.tensor([False]),
            timed_out=torch.tensor([False])))
    term = rewards.PotentialDifference(None, env)
    total = torch.zeros(1)
    for step, airborne in enumerate([False, True, True, terminal_airborne], start=1):
        env.scene["feet_ground_contact"].data.current_air_time[:] = 0.1 if airborne else 0.0
        race = env._sprint_race
        race.elapsed[:] = step * env.step_dt
        race.done[:] = step == 4
        race.failed[:] = failed and step == 4
        total += (term(env, potential=rewards.race_air_time_potential)
                  + time_weight * rewards.race_time_cost(env)) * env.step_dt
    torch.testing.assert_close(total, torch.tensor([-5.0 if failed else -0.008]))
    term.reset()
    assert term.previous.item() == 0.0


def _race_env(count=2):
    data = SimpleNamespace(
        root_link_pos_w=torch.zeros(count, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
    )
    env = SimpleNamespace(num_envs=count, device="cpu", step_dt=0.02,
        scene={"robot": SimpleNamespace(data=data)},
        episode_length_buf=torch.full((count,), 2000))
    return env, data, rewards.RaceEpisode(None, env)


def test_race_settle_tracks_start_and_ignores_generic_episode_clock():
    env, data, race = _race_env()
    for step in range(race.settle_steps):
        data.root_link_pos_w[:, 0] = step * 0.1
        assert not race(env).any()
        assert not race.elapsed.any()
        assert not race.progress.any()
    start = data.root_link_pos_w[:, 0].clone()
    data.root_link_pos_w[:, 0] += 0.25
    assert not race(env).any()
    torch.testing.assert_close(race.start_x, start)
    torch.testing.assert_close(race.progress, torch.full((2,), 0.25))
    torch.testing.assert_close(race.elapsed, torch.full((2,), env.step_dt))


def test_sustained_fall_beats_finish_and_upright_step_resets_fall_clock():
    env, data, race = _race_env()
    race.steps[:] = race.settle_steps
    data.projected_gravity_b[:, 2] = 0.0
    for _ in range(race.fall_steps - 1):
        assert not race(env).any()
    data.projected_gravity_b[1, 2] = -1.0
    data.root_link_pos_w[:, 0] = rewards.DISTANCE_M
    assert race(env).all()
    assert race.failed.tolist() == [True, False]
    assert race.finished.tolist() == [False, True]
    assert race.tipped_steps[1] == 0


def test_finish_on_deadline_beats_timeout_and_partial_reset_is_isolated():
    env, data, race = _race_env()
    race.steps[:] = race.settle_steps + race.limit_steps - 1
    data.root_link_pos_w[0, 0] = rewards.DISTANCE_M
    assert race(env).all()
    assert race.finished.tolist() == [True, False]
    assert race.timed_out.tolist() == [False, True]
    fields = ("start_x", "progress", "elapsed", "steps", "tipped_steps",
              "finished", "failed", "timed_out", "done")
    before = {name: getattr(race, name).clone() for name in fields}
    race.reset(env_ids=torch.tensor([0]))
    for name in fields:
        value = getattr(race, name)
        assert value[0] == 0
        torch.testing.assert_close(value[1], before[name][1])
