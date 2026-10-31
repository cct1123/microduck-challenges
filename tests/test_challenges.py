"""Every challenge folder: it registers, it trains for an open event, its knobs are live."""

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import list_tasks

import microduck_challenges  # noqa: F401  (registers every task)
from arena.manifest import load_event
from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_STANDUP_ROBOT_CFG,
    MICRODUCK_WALK_ROBOT_CFG,
    MICRODUCK_WALK_ROLLERS_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp

# folder, and for each exposed knob: (reward term whose weight it sets, value to try).
CHALLENGES = {
    "sprint_2m": {
        "forward_reward": ("track_linear_velocity", 3.7),
        "upright_reward": ("upright", 1.3),
        "step_reward": ("air_time", 4.1),
    },
    "got_a_talent": {
        "head_reward": ("head_pose_tracking", 1.3),
        "stillness_reward": ("posture_stillness", 3.1),
        "upright_reward": ("upright_linear", 3.7),
    },
    "hill_climb_1m5": {
        "forward_reward": ("track_linear_velocity", 3.7),
        "upright_reward": ("upright", 1.3),
        "step_reward": ("air_time", 4.1),
        "max_slope_deg": (None, 12.0),
    },
    "roller_sprint_2m": {
        "speed_reward": ("wheel_speed", 13.5),
        "upright_reward": ("upright", 1.3),
    },
    "tug_of_war": {
        "pull_reward": ("marker_ground", 3.1),
        "back_reward": ("track_linear_velocity", 2.3),
        "upright_reward": ("upright", 1.7),
        "stay_upright": (None, 1.0),
    },
    "sock_heist_6m": {
        "forward_reward": ("track_linear_velocity", 3.7),
        "progress_reward": ("progress", 5.3),
        "freeze_reward": ("freeze", 1.3),
        "upright_reward": ("upright", 1.3),
        "step_reward": ("air_time", 4.1),
    },
}

ROOT = Path(microduck_challenges.__file__).parent


def _contract(folder):
    return challenge.load(ROOT / folder / "tasks.py")


# The challenges whose Arena event races a commanded speed; MicroDuck Got a Talent is judged on style.
RACES = ["sprint_2m", "hill_climb_1m5", "roller_sprint_2m", "sock_heist_6m"]


@pytest.mark.parametrize("folder", RACES)
def test_training_covers_the_speeds_the_arena_commands(folder):
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg(stage="baseline") if folder == "sprint_2m" else env.make_env_cfg()
    lin_vel_x = cfg.commands["twist"].ranges.lin_vel_x
    event = load_event(_contract(folder).event)
    assert lin_vel_x[0] <= event.sweep.lo
    ranges = cfg.commands["twist"].ranges
    assert ranges.lin_vel_y == (0.0, 0.0) and ranges.ang_vel_z == (0.0, 0.0)


@pytest.mark.parametrize("folder", ["sprint_2m", "hill_climb_1m5", "sock_heist_6m"])
def test_walking_training_reaches_the_top_of_the_sweep(folder):
    # roller_sprint_2m is excluded: its top command is the robot's measured push
    # limit, not the Arena's sweep top (see roller_sprint_2m/env.py).
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg(stage="baseline") if folder == "sprint_2m" else env.make_env_cfg()
    lin_vel_x = cfg.commands["twist"].ranges.lin_vel_x
    event = load_event(_contract(folder).event)
    assert lin_vel_x[1] >= event.sweep.hi


@pytest.mark.parametrize("folder", ["sprint_2m", "hill_climb_1m5"])
def test_walking_races_keep_heading_privileged_with_task_specific_guidance(folder):
    # Sprint's dense baseline restores heading guidance, but never exposes
    # privileged heading observations to its Arena-compatible actor.
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg(stage="baseline") if folder == "sprint_2m" else env.make_env_cfg()
    if folder == "sprint_2m":
        assert cfg.rewards["heading_hold"].weight == 1.0
        assert cfg.rewards["heading_hold"].func is microduck_mdp.heading_hold_reward
        assert cfg.rewards["heading_hold"].params["std"] == 0.4
    else:
        assert cfg.rewards["heading_hold"].weight > 0
    assert "heading_error" in cfg.observations["critic"].terms
    assert "heading_error" not in cfg.observations["actor"].terms


def _yaw_quat(yaws):
    half = torch.tensor(yaws) / 2
    zero = torch.zeros_like(half)
    return torch.stack([half.cos(), zero, zero, half.sin()], dim=-1)


@pytest.mark.parametrize("folder", ["sprint_2m", "hill_climb_1m5"])
def test_the_critic_sees_the_heading_error_heading_hold_prices(folder):
    env_module = importlib.import_module(f"microduck_challenges.{folder}.env")
    robot = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=_yaw_quat([0.3, 3.0, 1.0])))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=3, episode_length_buf=torch.tensor([1, 1, 1]))
    microduck_mdp.heading_hold_reward(env)  # captures each spawn heading

    robot.data.root_link_quat_w = _yaw_quat([0.5, -3.0, 2.0])
    env.episode_length_buf = torch.tensor([40, 40, 0])  # the third just reset
    error = env_module.heading_error(env).squeeze(-1)

    # +0.2; across ±π the short way round (2π - 6), not -6; 0 on a fresh episode.
    assert torch.allclose(error, torch.tensor([0.2, 2 * math.pi - 6.0, 0.0]), atol=1e-5)
    env.episode_length_buf = torch.tensor([40, 40, 40])
    reward = microduck_mdp.heading_hold_reward(env, std=0.4)
    assert torch.allclose(reward[:2], torch.exp(-error[:2] ** 2 / 0.4**2), atol=1e-5)


def test_the_roller_trains_with_the_arenas_free_wheels():
    # The Arena's roller model's wheel bearings are frictionless (frictionloss=0.0
    # in robot_allcollisions_rollers.xml's passive_joint default); a policy trained
    # against added bearing friction breaks once raced there (see roller_sprint_2m/env.py).
    env = importlib.import_module("microduck_challenges.roller_sprint_2m.env")
    cfg = env.make_env_cfg()
    assert "randomize_wheel_friction" not in cfg.events
    assert "wheel_friction" not in cfg.curriculum


@pytest.mark.parametrize("folder", CHALLENGES)
def test_the_task_is_registered(folder):
    assert _contract(folder).task in list_tasks()


@pytest.mark.parametrize("folder", CHALLENGES)
def test_the_event_is_on_the_installed_arena(folder):
    # Every event the Arena ships can be raced: a challenge only needs its own to be there.
    assert load_event(_contract(folder).event).id == _contract(folder).event


@pytest.mark.parametrize("folder", sorted(CHALLENGES))
def test_the_kind_is_the_events(folder):
    """The Space refuses a manifest whose kind is not its event's; publish writes the challenge's."""
    contract = _contract(folder)
    assert contract.kind == load_event(contract.event).kind


@pytest.mark.parametrize("folder", CHALLENGES)
def test_the_robot_matches_the_events_rig(folder):
    # One robot per rig. A legs challenge may use the standup variant, which is the
    # same robot with the collision set a seated trunk needs (next test).
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg(stage="baseline") if folder == "sprint_2m" else env.make_env_cfg()
    robot = cfg.scene.entities["robot"]
    expected = {
        "legs": (MICRODUCK_WALK_ROBOT_CFG, MICRODUCK_STANDUP_ROBOT_CFG),
        "rollers": (MICRODUCK_WALK_ROLLERS_ROBOT_CFG,),
    }
    assert any(robot is cfg for cfg in expected[load_event(_contract(folder).event).rig])


def test_the_standup_robot_is_the_walk_robot_with_other_collisions():
    """One robot in every challenge (MJCF, masses, actuators); only the collision set may differ."""
    walk, standup = MICRODUCK_WALK_ROBOT_CFG, MICRODUCK_STANDUP_ROBOT_CFG
    assert standup.articulation == walk.articulation
    assert standup.init_state == walk.init_state
    a, b = walk.spec_fn().compile(), standup.spec_fn().compile()
    assert [a.body(i).name for i in range(a.nbody)] == [b.body(i).name for i in range(b.nbody)]
    assert [a.joint(i).name for i in range(a.njnt)] == [b.joint(i).name for i in range(b.njnt)]
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                  "jnt_range", "dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), err_msg=field)


@pytest.mark.parametrize("folder", CHALLENGES)
def test_runs_log_under_the_folder_name(folder):
    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    assert env.RL_CFG.experiment_name == folder


def test_every_toml_knob_is_in_the_table():
    """A knob added to challenge.toml but not to CHALLENGES would otherwise go unchecked."""
    for folder in CHALLENGES:
        assert set(CHALLENGES[folder]) == set(challenge.params(ROOT / folder / "tasks.py")), folder


def test_every_challenge_folder_has_a_row():
    """A folder without a CHALLENGES row goes unchecked by every test above."""
    assert set(CHALLENGES) == {p.parent.name for p in ROOT.glob("*/challenge.toml")}


@pytest.mark.parametrize("folder", CHALLENGES)
def test_knobs_reach_the_config(folder, tmp_path, monkeypatch):
    """Rewrite the knobs to unusual values; the rebuilt config must carry them."""
    toml = ROOT / folder / "challenge.toml"
    text = toml.read_text()
    for knob, (_, value) in CHALLENGES[folder].items():
        text = _set_value(text, knob, value)
    copy = tmp_path / "challenge.toml"
    copy.write_text(text)
    real_params = challenge.params
    monkeypatch.setattr(challenge, "params", lambda f: real_params(copy))

    env = importlib.import_module(f"microduck_challenges.{folder}.env")
    cfg = env.make_env_cfg(stage="baseline") if folder == "sprint_2m" else env.make_env_cfg()
    for knob, (term, value) in CHALLENGES[folder].items():
        if term is None:
            continue
        assert math.isclose(cfg.rewards[term].weight, value), f"{knob} does not set {term}"


def _set_value(text: str, knob: str, value: float) -> str:
    head = f"[params.{knob}]"
    assert head in text, f"no {head}"
    before, after = text.split(head, 1)
    lines = after.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.split("=")[0].strip() == "value":
            lines[i] = f"value = {value}\n"
            return before + head + "".join(lines)
    raise AssertionError(f"{head} has no value")


def test_the_hill_trains_on_slopes_up_to_the_knob(tmp_path, monkeypatch):
    toml = ROOT / "hill_climb_1m5" / "challenge.toml"
    # Read the SHIPPED value before challenge.params gets monkeypatched below.
    shipped_max_slope_deg = challenge.params(ROOT / "hill_climb_1m5" / "tasks.py")["max_slope_deg"]
    copy = tmp_path / "challenge.toml"
    copy.write_text(_set_value(toml.read_text(), "max_slope_deg", 12.0))
    real_params = challenge.params
    monkeypatch.setattr(challenge, "params", lambda f: real_params(copy))
    from microduck_challenges.hill_climb_1m5.env import make_env_cfg

    terrain = make_env_cfg().scene.terrain
    assert terrain.terrain_type == "generator"
    slope = terrain.terrain_generator.sub_terrains["slope"]
    assert math.isclose(slope.slope_range[1], math.tan(math.radians(12.0)))
    assert shipped_max_slope_deg > 8.0, "must cover the course's 8°"
