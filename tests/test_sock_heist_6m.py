"""Sock Heist trains under the event's rules: every copy in lights.py is pinned against the Arena, and the
training's command plays them to the policy as the Arena does."""
import hashlib, importlib, math, subprocess, sys, tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np, onnxruntime, pytest, torch
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from arena import runner as arena_runner
from arena.assets import ASSET_ROOT
from arena.constants import ACTION_SCALE, CTRL_DT as ARENA_CTRL_DT, OBS_SIZE, TRACE_WIDTH
from arena.lights import Red, timetable
from arena.manifest import load_event
from arena.runner import RunOutcome
from microduck_challenges.sock_heist_6m import check, env, lights, rewards, walker
from microduck_challenges.tug_of_war.opponent import load_actor

EVENT = load_event("sock-heist-6m")
SETTLE = lights.steps(lights.SETTLE_S)
FOLDER = Path(lights.__file__).parent


def _quat(yaw, roll=0.0, pitch=0.0):
    """wxyz, z-y-x order."""
    cy, sy, cp, sp, cr, sr = (f(a / 2) for a in (yaw, pitch, roll) for f in (np.cos, np.sin))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float32)


def test_the_rules_are_the_events():
    l, s = EVENT.lights, EVENT.steering
    assert lights.CTRL_DT == ARENA_CTRL_DT and lights.SETTLE_S == EVENT.start.settle_s
    assert (lights.FIRST_GREEN_S, lights.GREEN_S, lights.RED_S) == (l.first_green_s, l.green_s, l.red_s)
    assert (lights.TURN_S, lights.TOLERANCE_M) == (l.turn_s, l.tolerance_m)
    assert (lights.LOOKAHEAD_M, lights.GAIN, lights.MAX_YAW_RATE) == (s.lookahead_m, s.gain, s.max_yaw_rate)


def test_every_phase_is_drawn_from_its_range_ends_included():
    torch.manual_seed(0)
    onset, length = lights.draw_reds(4096, 1000, "cpu")
    every = lambda r: set(range(lights.steps(r[0]), lights.steps(r[1]) + 1))
    assert set((onset[:, 0] - SETTLE).tolist()) == every(lights.FIRST_GREEN_S)     # 30..50
    assert set(length.flatten().tolist()) == every(lights.RED_S)                   # 70..120
    greens = onset[:, 1:] - onset[:, :-1] - length[:, :-1]
    assert set(greens.flatten().tolist()) == every(lights.GREEN_S)                 # 80..150
    assert (onset[:, -1] >= 1000).all()                                            # the table outlasts the episode


@pytest.mark.parametrize("turn", [lights.steps(lights.TURN_S), 0])
def test_the_light_steps_are_the_arenas(turn):
    steps = 3000
    for seed in EVENT.seeds:
        reds = timetable(EVENT.lights, EVENT.id, seed, steps)
        onset = torch.tensor([[r.onset + SETTLE for r in reds]]).expand(steps, -1)
        length = torch.tensor([[r.length for r in reds]]).expand(steps, -1)
        red, mark, watch = lights.light_steps(onset, length, turn, torch.arange(SETTLE, SETTLE + steps))
        expected = arena_runner._LightSteps.of(reds, turn, steps)
        assert (red.tolist(), mark.tolist(), watch.tolist()) == (
            expected.red.tolist(), expected.mark.tolist(), expected.watch.tolist())


def test_the_lane_is_the_arenas_less_a_duck_width():
    assert lights.LANE_HALF_WIDTH_M == pytest.approx(EVENT.course.width_m / 2 - 0.1)


def test_the_hold_is_the_arenas():
    command = np.asarray(EVENT.command, dtype=np.float64)
    for y in (-0.35, -0.05, 0.0, 0.12, 0.35):
        for yaw in (-3.1, -1.0, -0.1, 0.0, 0.2, 1.5, 3.1):
            q = _quat(yaw, roll=0.1, pitch=-0.08)                  # a tilted trunk: the yaw formula, not just qz
            data = SimpleNamespace(qpos=np.concatenate([[0.0, y, 0.12], q]))
            want = arena_runner._steered(EVENT.steering, data, SimpleNamespace(free=0), command)[2]
            got = lights.hold_yaw_rate(torch.tensor([y]), torch.from_numpy(q)[None]).item()
            assert got == pytest.approx(want, abs=1e-5)
    rate = lambda y, yaw: lights.hold_yaw_rate(torch.tensor([y]), torch.from_numpy(_quat(yaw))[None]).item()
    assert rate(0.2, 0.0) < 0 < rate(-0.2, 0.0)                    # left of the line it turns right
    assert (rate(0.0, 3.0), rate(0.0, -3.0)) == (-1.0, 1.0)        # facing back: the short way round, capped


@pytest.fixture(scope="module")
def cfg():
    return env.make_env_cfg()


SEED = 0  # the live env's randomisation, drawn the same on every run: TESTS gates every training


@pytest.fixture(scope="module")
def live():
    """The training env, 2 envs on the CPU, seeded."""
    cfg = env.make_env_cfg()
    cfg.scene.num_envs = 2
    cfg.seed = SEED
    training = ManagerBasedRlEnv(cfg, device="cpu")
    training.reset()
    yield training
    training.close()


@pytest.fixture(scope="module")
def flock():
    """The training env with 8 ducks, seeded: how far the walker gets on green varies by duck (0.17 m/s for
    two of them, 0.30 for eight), so a margin between walking and standing wants more than two."""
    cfg = env.make_env_cfg()
    cfg.scene.num_envs = 8
    cfg.seed = SEED
    training = ManagerBasedRlEnv(cfg, device="cpu")
    yield training
    training.close()


@pytest.fixture(scope="module")
def walker_step():
    """One step of `live` with the Arena's walker driving both ducks: on zero actions a duck tips over within
    about a second (fell_over at steps 48 and 59), where the walker asked to stand holds within a millimetre."""
    session = onnxruntime.InferenceSession(str(walker.WALKER), providers=["CPUExecutionProvider"])

    def step(live, stand=False):
        """`stand`: the walker is told no speed (its observed twist, obs[48:51], zeroed), so it stands."""
        rows = live.obs_buf["actor"].numpy().copy()
        if stand:
            rows[:, 48:51] = 0.0
        return live.step(torch.from_numpy(np.concatenate([session.run(None, {"obs": r[None]})[0] for r in rows])))

    return step


def _set_reds(live, onset, length):
    """A fresh episode in both envs, each with one red: `length` steps from step `onset`. Reseeded, so a
    test's randomisation is the same whichever tests ran before it."""
    live.reset(seed=SEED)
    term = live.command_manager.get_term("twist")
    term.onset[:, 0] = onset
    term.onset[:, 1:] = 10**6
    term.length[:] = length


def _qpos(live):
    """(2, 7): the robot's free-joint qpos."""
    robot = live.scene["robot"]
    return robot.data.data.qpos[:, robot.indexing.free_joint_q_adr]


def _nudge(live, env_id, dx=0.0, dy=0.0):
    """Shifts one env's trunk across the floor, as a duck that moved."""
    robot = live.scene["robot"]
    pose = robot.data.root_link_pose_w[env_id:env_id + 1].clone()
    pose[:, 0] += dx
    pose[:, 1] += dy
    robot.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([env_id]))
    live.sim.forward()


def _tracking_env(command_vx, vx):
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[vx, 0.0, 0.0]]))
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                           command_manager=SimpleNamespace(get_command={"twist": torch.tensor([[command_vx, 0.0, 0.0]])}.get))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_training_never_imports_the_arena():
    code = ("import sys; sys.modules['arena'] = None; "
            "import microduck_challenges.sock_heist_6m.env, microduck_challenges.sock_heist_6m.walker")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_command_and_the_start_are_the_events(cfg):
    twist = cfg.commands["twist"]
    assert isinstance(twist, lights.LightsCommandCfg)
    assert twist.ranges.lin_vel_x == (EVENT.sweep.lo, EVENT.sweep.hi) == env.SWEEP
    assert twist.ranges.lin_vel_y == twist.ranges.ang_vel_z == (0.0, 0.0) and not twist.heading_command
    assert (twist.rel_standing_envs, twist.rel_heading_envs, twist.rel_forward_envs, twist.rel_world_envs,
            twist.init_velocity_prob) == (0.0,) * 5
    assert twist.resampling_time_range[0] > cfg.episode_length_s
    assert "standing_envs" not in cfg.curriculum and "push_robot" not in cfg.events
    yaw = math.radians(EVENT.start.yaw_deg)
    assert env.YAW_PERTURBATION_DEG == EVENT.start.yaw_deg
    assert cfg.events["reset_base"].params["pose_range"]["yaw"] == pytest.approx((-yaw, yaw))
    assert cfg.actions["joint_pos"].scale == ACTION_SCALE


def test_the_actor_reads_the_arenas_61_numbers(cfg, live):
    sprint = importlib.import_module("microduck_challenges.sprint_2m.env").make_env_cfg()
    actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
    assert list(actor) == list(sprint.observations["actor"].terms)
    assert {"light", "lane", "base_lin_vel"} <= set(critic) and not {"light", "lane", "heading_error"} & set(actor)
    assert "heading_error" not in critic
    assert live.observation_manager.group_obs_dim["actor"] == (OBS_SIZE,)


def test_walking_on_green_pays_far_more_than_standing_still(cfg):
    # The tug's t4/t5 stood when standing kept 0.41 of tracking: here it keeps 0.13 at 0.45 m/s.
    term = cfg.rewards["track_linear_velocity"]
    assert term.weight == 2.0 and term.params["std"] == pytest.approx(math.sqrt(0.1))
    assert "heading_hold" not in cfg.rewards
    paid = lambda vx: term.weight * term.func(_tracking_env(0.45, vx), **term.params).item()
    assert paid(0.45) == pytest.approx(2.0) and paid(0.0) <= 2.0 / 6


@pytest.mark.parametrize("speed", [0.8, 1.0])
def test_walking_on_green_out_earns_standing_still(flock, walker_step, speed):
    # Under an endless green a statue once earned 6.9 a step to the walker's 6.0-6.5 (heist1 and heist2 crept):
    # walking must win by more than the 1.6 air_time pays a duck stepping in place.
    paid = {}
    for stand in (False, True):
        _set_reds(flock, onset=10**6, length=1)
        flock.command_manager.get_term("twist").speed[:] = speed
        total = []
        for k in range(300):
            reward = walker_step(flock, stand)[1]
            if k >= 100:
                total.append(reward.mean().item() / flock.step_dt)
        paid[stand] = sum(total) / len(total)
    print(f"speed {speed}: walking {paid[False]:.2f}, standing {paid[True]:.2f}, margin {paid[False] - paid[True]:.2f}")
    assert paid[False] - paid[True] > 1.6


def test_freeze_pays_only_while_watched_and_fades_at_the_tolerance(cfg):
    fake = lambda watch, drift: SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(
        stepped=lambda: (None, None, torch.tensor(watch)), drift=lambda: torch.tensor(drift))))
    params = cfg.rewards["freeze"].params
    got = rewards.freeze(fake([True, True, False], [0.0, 0.03, 0.0]), **params)
    assert got.tolist() == pytest.approx([1.0, math.exp(-1.0), 0.0])


def test_the_twist_is_zero_from_a_reds_first_step_to_its_last(live, walker_step):
    term = live.command_manager.get_term("twist")
    _set_reds(live, onset=40, length=30)
    term.speed[:] = 0.2                                  # the walker does not move at 0.20 (#3): none is caught or falls
    for _ in range(80):
        k = live.episode_length_buf.clone()
        twist, qpos = live.command_manager.get_command("twist"), _qpos(live)
        stand = (k < SETTLE) | ((40 <= k) & (k < 70))
        assert (twist[stand] == 0).all()
        hold = lights.hold_yaw_rate(qpos[:, 1] - term.lane_y, qpos[:, 3:7])
        walk = torch.stack([term.speed, torch.zeros_like(hold), hold], dim=-1)
        assert torch.allclose(twist[~stand], walk[~stand], atol=1e-6)
        walker_step(live)
    assert (live.episode_length_buf == 80).all()         # one episode each: the red's every step was seen


def test_the_hold_steers_from_the_spawn_line(live):
    term = live.command_manager.get_term("twist")
    _set_reds(live, onset=10**6, length=1)
    assert torch.allclose(term.lane_y, _qpos(live)[:, 1])
    for _ in range(SETTLE):
        live.step(torch.zeros(2, 14))
    _nudge(live, env_id=0, dy=0.2)                       # 20 cm left of its line
    live.step(torch.zeros(2, 14))
    assert live.command_manager.get_command("twist")[0, 2] < -0.5   # turns right, back to it


def test_moving_past_the_tolerance_while_watched_is_caught(live, walker_step):
    _set_reds(live, onset=40, length=40)                 # turning 40-59, marked at 59, watched 60-79
    live.command_manager.get_term("twist").speed[:] = 0.0    # the walker stands: only the nudges move a duck
    for k in range(66):
        if k == 50:
            _nudge(live, env_id=1, dx=0.04)              # while the head turns: nothing counts
        if k == 65:
            _nudge(live, env_id=0, dx=0.04)
            _nudge(live, env_id=1, dx=0.015)             # watched: 4 cm is caught, 1.5 cm is not
        walker_step(live)
        assert live.termination_manager.get_term("caught").tolist() == ([True, False] if k == 65 else [False, False])


def test_a_duck_entering_mid_watch_is_judged_from_where_it_entered(live):
    """mjlab's train starts rsl_rl's learn at random episode lengths (init_at_random_ep_len): every env is moved
    to a random step after the wrapper's reset, without another, so some start inside a watch, past its mark."""
    term = live.command_manager.get_term("twist")
    _set_reds(live, onset=40, length=40)                 # watched 60-79
    live.episode_length_buf = torch.full_like(live.episode_length_buf, 65)    # a new tensor, as learn() assigns it
    assert term.ahead()[1].all()                         # the step about to run is watched
    live.step(torch.zeros(2, 14))
    assert live.termination_manager.get_term("caught").tolist() == [False, False]


def test_straying_more_than_the_half_lane_off_the_lane_ends_the_episode(live):
    _set_reds(live, onset=10**6, length=1)
    _nudge(live, env_id=0, dy=0.31)
    _nudge(live, env_id=1, dy=0.29)
    live.step(torch.zeros(2, 14))
    assert live.termination_manager.get_term("off_lane").tolist() == [True, False]


def test_the_critic_is_told_the_light_and_the_lane(live, walker_step):
    term = live.command_manager.get_term("twist")
    _set_reds(live, onset=40, length=30)                 # turning 40-59, watched 60-69
    term.speed[:] = 0.0                                  # the walker stands, in one episode
    for k in range(80):
        standing, watched, until, drift = rewards.light_state(live, "twist").T
        assert standing.tolist() == [float(k < SETTLE or 40 <= k < 70)] * 2
        assert watched.tolist() == [float(60 <= k < 70)] * 2
        change = next(c for c in (SETTLE, 40, 70, 10**6) if c > k)      # the settle's end, the red's start, its end
        assert until.tolist() == pytest.approx([(change - k) * lights.CTRL_DT] * 2)
        assert drift.tolist() == pytest.approx((term.drift() / lights.TOLERANCE_M).tolist() if 60 <= k < 70 else [0.0] * 2)
        assert torch.allclose(rewards.lane_state(live, "twist")[:, 0], _qpos(live)[:, 1] - term.lane_y)
        walker_step(live)


def test_the_learner_starts_as_the_walker():
    assert sha256(walker.WALKER) == sha256(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx")
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=FOLDER.parents[2],
                            capture_output=True, text=True, check=True).stdout.split()
    assert "src/microduck_challenges/sock_heist_6m/walker.onnx" in listed   # hf_jobs.py tars exactly this list
    normalize = load_actor(walker.WALKER)[0]
    noise = torch.from_numpy(np.random.default_rng(0).normal(size=(32, OBS_SIZE)).astype(np.float32))
    rows = normalize.mean + normalize.std * noise
    obs = TensorDict({"actor": rows}, batch_size=[len(rows)])
    cfg = env.RL_CFG.actor
    assert resolve_callable(cfg.class_name) is walker.WalkerActor
    actor = resolve_callable(cfg.class_name)(
        obs, {"actor": ["actor"]}, "actor", 14, hidden_dims=cfg.hidden_dims, activation=cfg.activation,
        obs_normalization=cfg.obs_normalization, distribution_cfg=dict(cfg.distribution_cfg))
    session = onnxruntime.InferenceSession(str(walker.WALKER), providers=["CPUExecutionProvider"])
    theirs = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in rows.numpy()])
    with torch.no_grad():
        np.testing.assert_allclose(actor(obs).numpy(), theirs, atol=1e-5)
        assert actor.distribution.std_param.tolist() == pytest.approx([0.3] * 14)
        # PPO's update at the first rollout step, every env at one state: the walker's statistics hold.
        actor.update_normalization(TensorDict({"actor": rows[:1].expand(4096, -1)}, batch_size=[4096]))
        np.testing.assert_allclose(actor(obs).numpy(), theirs, atol=1e-2)


def test_the_walker_walks_in_the_training_env(live, walker_step):
    """The Arena's walker, fed the training's actor observation, walks: the layout is the Arena's."""
    term = live.command_manager.get_term("twist")
    _set_reds(live, onset=10**6, length=1)
    term.speed[:] = 0.6
    x0 = _qpos(live)[:, 0].clone()
    for _ in range(150):
        walker_step(live)
        assert not live.reset_terminated.any()
    # 2.5 s of green: #3 measured ~0.3 m/s at 0.6. Randomised, 64 envs made 0.37 to 0.74 m (median 0.65);
    # one standing on its line makes millimetres.
    assert (_qpos(live)[:, 0] - x0 > 0.3).all()


def test_the_recipe_judges_every_hundred_iterations():
    recipe = tomllib.loads((FOLDER / "challenge.toml").read_text())["recipe"]
    assert recipe == {"num_envs": 4096, "max_iterations": 1000, "seed": 1} and env.RL_CFG.save_interval == 100


def _outcome(status, x, caught_frame=None, time_s=None):
    """A run whose trace is zeros but its x column, after a one-frame settle at x 0."""
    trace = np.zeros((len(x), TRACE_WIDTH), dtype=np.float32)
    trace[:, 0] = x
    return RunOutcome(1, status, time_s, 0.0, 0.0, trace, np.zeros((1, TRACE_WIDTH), dtype=np.float32), caught_frame)


def test_a_catch_says_how_far_into_the_watch_and_how_fast_the_duck_came():
    x = np.minimum(np.arange(140) * 0.01, 0.72)            # 0.5 m/s, then still from step 72
    kind, line = check.judge(_outcome("caught", x, caught_frame=72), (Red(50, 70),), 20, 15)
    assert kind == "caught early"
    assert line.startswith("caught at 1.46 s, 0.04 s into the watch, 0.50 m/s as the red began · 1 red met")


def test_a_fall_tipping_in_a_reds_first_moments_is_on_a_stop():
    kind, line = check.judge(_outcome("fall", np.zeros(75)), (Red(50, 70),), 20, 15)
    assert (kind, line.split(" · ")[0]) == ("fell on a stop", "fell at 1.50 s on a stop, tipping 0.20 s after a red began")


def test_a_fall_far_from_any_red_is_on_green():
    assert check.judge(_outcome("fall", np.zeros(75)), (Red(200, 70),), 20, 15)[0] == "fell on green"


def test_a_duck_that_never_left_the_line_is_named():
    kind, line = check.judge(_outcome("timeout", np.full(500, 0.1)), (Red(50, 70),), 20, 15)
    assert (kind, line.split(" · ")[0]) == ("never left the line", "never left the line (0.10 m)")


def test_the_worst_drift_is_read_over_the_watched_steps():
    x = np.zeros(200); x[69:] = 0.05; x[75] = 0.07          # marked at 69 (0.05 m), 2 cm off at 75
    assert "worst drift 2.0 cm" in check.judge(_outcome("finish", x, time_s=4.0), (Red(50, 70),), 20, 15)[1]


def test_the_tally_counts_each_kind():
    kinds = ["finish", "caught early", "fell on a stop", "finish", "never left the line"]
    assert check.tally(kinds, [10.0, 30.0, 30.0, 12.0, 30.0]) == (
        "finished 2/5 · caught 1 (early 1) · fell 1 (on a stop 1, on a start 0) · out of time 1 (never left the line 1) · mean 22.40 s")


def test_check_judges_the_walker(capsys):
    assert check.main([str(walker.WALKER), "--speed", "0.95", "--seeds", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("seed 1 · ") and lines[-1].startswith("finished ")   # no verdict: platforms differ


RED = (Red(50, 70),)   # watched from step 70, ends at step 120


@pytest.mark.parametrize("caught_frame, kind", [(80, "caught early"), (81, "caught late")])
def test_a_catch_is_early_up_to_ten_steps_into_the_watch(caught_frame, kind):
    assert check.judge(_outcome("caught", np.zeros(140), caught_frame=caught_frame), RED, 20, 15)[0] == kind


@pytest.mark.parametrize("tip, kind", [
    (80, "fell on a stop"), (81, "fell under a red"), (150, "fell on a start"), (151, "fell on green")])
def test_a_fall_is_on_a_stop_or_a_start_up_to_thirty_steps_after_it(tip, kind):
    assert check.judge(_outcome("fall", np.zeros(tip + 15)), RED, 20, 15)[0] == kind   # tipping begins at k - 14


@pytest.mark.parametrize("progress, kind", [(0.6, "out of time"), (0.4, "never left the line")])
def test_out_of_time_is_never_left_the_line_below_half_a_metre(progress, kind):
    assert check.judge(_outcome("timeout", np.full(500, progress)), RED, 20, 15)[0] == kind
