"""The Tug of War trains in the Arena's tug: the same course, string end, start and command."""
import itertools, math, subprocess, sys, tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import mujoco, numpy as np, onnx, onnxruntime, pytest, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab_microduck import challenge as ch
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

import microduck_challenges
from arena import actuator as arena_actuator, course as arena_course, runner as arena_runner, tug as arena_tug
from arena.assets import ASSET_ROOT
from arena.constants import ACTION_SCALE, DEFAULT_POSE, JOINT_NAMES
from arena.manifest import load_event
from microduck_challenges.tug_of_war import env as tug, opponent, rewards

EVENT = load_event("tug-of-war")
CONTACT = ("geom_type", "geom_size", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
           "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap")
STRING_FIELDS = {"body": ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos"),
                 "jnt": ("jnt_type", "jnt_axis", "jnt_pos"), "geom": CONTACT + ("geom_pos", "geom_quat"),
                 "site": ("site_pos",)}


@pytest.fixture(scope="module")
def arena_model():
    return arena_course.compile_tug(EVENT)


@pytest.fixture(scope="module")
def training_model():
    scene = SceneCfg(num_envs=1, entities={"robot": MICRODUCK_STANDUP_ROBOT_CFG, "opponent": tug.OPPONENT_CFG},
                     spec_fn=tug.tug_scene)
    return Scene(scene, device="cpu").compile()


def test_the_training_never_imports_the_arena():
    """HF Jobs install the default groups only: no Arena there. The package's __init__ imports every tasks.py."""
    code = ("import sys; sys.modules['arena'] = None; "
            "import microduck_challenges.tug_of_war.env, microduck_challenges.tug_of_war.opponent")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_constants_are_the_arenas():
    c, s = EVENT.course, EVENT.start
    assert (tug.RING_M, tug.PIT_M, tug.PIT_DEPTH_M, tug.STRING_M) == (c.ring_m, c.pit_m, c.pit_depth_m, c.string_m)
    assert (tug.RING_ROPES_M, tug.RING_ROPE_RADIUS_M, tug.RING_POST_RADIUS_M, tug.RING_POST_TOP_M) == (
        arena_course.RING_ROPES_M, arena_course.RING_ROPE_RADIUS_M, arena_course.RING_POST_RADIUS_M,
        arena_course.RING_POST_TOP_M)
    assert tuple(EVENT.command) == tug.COMMAND == (tug.COMMAND_VX,) + (0.0,) * 12
    assert (tug.BOUT_S, tug.SETTLE_S, tug.JOINT_PERTURBATION_RAD, tug.YAW_PERTURBATION_DEG) == (
        EVENT.bout.time_limit_s, s.settle_s, s.joint_rad, s.yaw_deg)
    # The Arena's fall rule, summaries only there: the training ends an episode on it.
    assert (tug.FALL_GRAVITY_Z, tug.FALL_HOLD_S) == (EVENT.termination.fall_gravity_z, EVENT.termination.fall_hold_s)
    assert tug.STRING_OFFSET == pytest.approx(arena_course.string_offset("legs"), abs=1e-12)
    assert (tug.WALL_THICKNESS_M, tug.PLATFORM_OVERRUN_M, tug.PIT_DROP_M) == (
        arena_course.WALL_THICKNESS_M, arena_course.PLATFORM_OVERRUN_M, arena_tug.PIT_DROP_M)
    assert tug.STRING_SITE == arena_course.STRING_SITE


def _static_colliders(model, kind):
    return {model.geom(g).name for g in range(model.ngeom)
            if model.body_weldid[model.geom_bodyid[g]] == 0 and (model.geom_contype[g] or model.geom_conaffinity[g])
            and model.geom_type[g] == kind}


def _capsule_ends(model, name):
    """A static capsule's (from, to), off its position, orientation and half length (MuJoCo's z axis points at `from`)."""
    g = model.geom(name).id
    axis = np.zeros(9)
    mujoco.mju_quat2Mat(axis, model.geom_quat[g])
    axis = axis.reshape(3, 3)[:, 2] * model.geom_size[g][1]
    return model.geom_pos[g] + axis, model.geom_pos[g] - axis


def test_the_course_tiles_the_arenas(arena_model, training_model):
    """Each of the Arena's boxes comes as tiles at most TILE_M along x and y (D18): together they fill it, each
    with its height, its top and its contacts."""
    boxes = _static_colliders(arena_model, mujoco.mjtGeom.mjGEOM_BOX)
    tiles = _static_colliders(training_model, mujoco.mjtGeom.mjGEOM_BOX)
    assert boxes == {n for n, _, _ in tug.course_boxes()} == {n.rsplit("_", 2)[0] for n in tiles}  # no floor plane left
    datas = [mujoco.MjData(m) for m in (arena_model, training_model)]
    for m, d in zip((arena_model, training_model), datas):
        mujoco.mj_forward(m, d)
    for name in boxes:
        a = arena_model.geom(name).id
        centre, half = datas[0].geom_xpos[a], arena_model.geom_size[a]
        area, rects = 0.0, []
        for t in (training_model.geom(n).id for n in tiles if n.rsplit("_", 2)[0] == name):
            pos, size = datas[1].geom_xpos[t], training_model.geom_size[t]
            assert (2 * size[:2] <= tug.TILE_M + 1e-12).all(), name
            assert (pos - size >= centre - half - 1e-12).all() and (pos + size <= centre + half + 1e-12).all(), name
            assert (pos[2], size[2]) == pytest.approx((centre[2], half[2]), abs=1e-12)
            area += 4 * size[0] * size[1]
            rects.append((pos[:2] - size[:2], pos[:2] + size[:2]))
            for field in set(CONTACT) - {"geom_size"}:
                np.testing.assert_allclose(getattr(training_model, field)[t], getattr(arena_model, field)[a], err_msg=f"{name} {field}")
        for i, (low, high) in enumerate(rects):   # no two tiles overlap: with the area, they fill the box
            for other_low, other_high in rects[i + 1:]:
                overlap = np.clip(np.minimum(high, other_high) - np.maximum(low, other_low), 0, None)
                assert overlap[0] * overlap[1] <= 1e-12, name
        assert area == pytest.approx(4 * half[0] * half[1], rel=1e-12), name


def test_the_rings_ropes_and_posts_are_the_arenas_in_pieces(arena_model, training_model):
    """D18 for capsules: every Arena rope or post comes as pieces at most TILE_M long, end to end from its `from`
    to its `to`, each with its radius and contacts. A capsule cut at a point is the same shape."""
    capsules = _static_colliders(arena_model, mujoco.mjtGeom.mjGEOM_CAPSULE)
    pieces = _static_colliders(training_model, mujoco.mjtGeom.mjGEOM_CAPSULE)
    assert capsules == {n for n, *_ in tug.course_capsules()} == {n.rsplit("_", 1)[0] for n in pieces}
    assert len(capsules) == 16
    for name in capsules:
        ends = _capsule_ends(arena_model, name)                       # (from, to) off geom_pos, geom_quat, size[1]
        chain = [_capsule_ends(training_model, p) for p in sorted(n for n in pieces if n.rsplit("_", 1)[0] == name)]
        np.testing.assert_allclose([chain[0][0], chain[-1][1]], ends, atol=1e-12)
        for (_, a), (b, _) in zip(chain, chain[1:]):
            np.testing.assert_allclose(a, b, atol=1e-12)
        for p in (n for n in pieces if n.rsplit("_", 1)[0] == name):
            t, g = training_model.geom(p).id, arena_model.geom(name).id
            assert 2 * training_model.geom_size[t][1] <= tug.TILE_M + 1e-12
            for field in set(CONTACT) - {"geom_size"}:
                np.testing.assert_allclose(getattr(training_model, field)[t], getattr(arena_model, field)[g], err_msg=f"{p} {field}")
            assert training_model.geom_size[t][0] == arena_model.geom_size[g][0]


def _cfg_with_knobs(**knobs):
    """The recipe built with these knobs, whatever challenge.toml says (as the hill's test does)."""
    with pytest.MonkeyPatch.context() as patch:
        real_params = ch.params
        patch.setattr(ch, "params", lambda f: {**real_params(f), **knobs})
        return tug.make_env_cfg()


@pytest.fixture(scope="module")
def cfg():
    return _cfg_with_knobs(stay_upright=0.0)


def test_the_scene_and_solver_are_the_tugs(cfg, arena_model):
    assert cfg.scene.terrain is None and cfg.scene.spec_fn is tug.tug_scene
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG  # collisions pinned by test_got_a_talent
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (arena_model.opt.iterations, arena_model.opt.ls_iterations)
    assert cfg.sim.mujoco.timestep == pytest.approx(arena_model.opt.timestep) and cfg.decimation == 4
    assert cfg.sim.mujoco.integrator == "implicitfast" and arena_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_the_actor_reads_the_events_command(cfg):
    ranges = cfg.commands["twist"].ranges
    assert ranges.lin_vel_x == (EVENT.command[0],) * 2 and ranges.lin_vel_y == (0.0, 0.0) and ranges.ang_vel_z == (0.0, 0.0)
    assert cfg.commands["twist"].rel_standing_envs == 0.0 and "standing_envs" not in cfg.curriculum
    for name in ("head_pose", "body_pose"):
        assert all(lo <= 0.0 <= hi for lo, hi in cfg.commands[name].ranges)  # the Arena sends zeros
    actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    privileged = {"opponent", "string_length", "marker_x", "pit_margin", "heading_error", "base_lin_vel"}
    assert privileged <= set(critic) and not privileged & set(actor)


def test_no_env_is_sent_forward(cfg):
    """mjlab's template sends a fifth of the envs forward (vx made |vx| ≥ 0.3): the Arena never does."""
    assert cfg.commands["twist"].rel_forward_envs == 0.0


def test_the_start_is_the_arenas(cfg):
    pose, west = cfg.events["reset_base"].params["pose_range"], arena_course.tug_layout(EVENT.course)[0]
    assert pose["x"][0] == pytest.approx(west.x) and sum(pose["y"]) / 2 == pytest.approx(west.y)
    assert pose["z"][0] <= arena_course.SPAWN_HEIGHT_M <= pose["z"][1]
    yaw = math.radians(EVENT.start.yaw_deg)
    assert pose["yaw"] == pytest.approx((-yaw, yaw))   # the Arena's, not the template's ±π
    east = cfg.events["reset_opponent_base"].params["pose_range"]["yaw"]
    assert east == pytest.approx((math.pi - yaw, math.pi + yaw))
    joints = cfg.events["reset_robot_joints"].params["position_range"]
    assert joints[0] <= -EVENT.start.joint_rad and EVENT.start.joint_rad <= joints[1]
    assert cfg.episode_length_s == pytest.approx(tug.SETTLE_S + EVENT.bout.time_limit_s)


def test_backing_away_pays_far_more_than_standing_still(cfg):
    """t1–t5 never discovered the walk back: standing still kept 0.41 of track_linear_velocity's 1.0.
    Walk backwards defaults to 3.0 at std √0.05: at the event's -0.30 m/s the duck earns 3.0, standing
    still 3·e^-1.8 = 0.50."""
    knob = tomllib.loads((Path(tug.__file__).parent / "challenge.toml").read_text())["params"]["back_reward"]
    assert knob["min"] <= knob["value"] <= knob["max"]
    term = cfg.rewards["track_linear_velocity"]
    assert term.weight == knob["value"] and term.params["std"] == pytest.approx(math.sqrt(0.05))

    def paid(vx):
        data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[vx, 0.0, 0.0]]))
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                              command_manager=SimpleNamespace(get_command={"twist": torch.tensor([[tug.COMMAND_VX, 0.0, 0.0]])}.get))
        return term.weight * term.func(env, **term.params).item()

    assert paid(tug.COMMAND_VX) == pytest.approx(3.0)
    assert paid(0.0) == pytest.approx(3.0 * math.exp(-1.8), abs=1e-6)   # 0.50, a sixth of walking back


def test_either_pit_ends_the_episode_and_pays_its_side(cfg):
    """Either pit, as on the Arena; and the learner's fall, training only (the Arena judges only the pit),
    so the duck keeps learning to stand and pull instead of lying on its side for the rest of the episode."""
    t, r = cfg.terminations, cfg.rewards
    assert set(t) == {"time_out", "nan_state", "opponent_nan_state", "in_pit", "opponent_in_pit", "fell"}
    # The critic reads the opponent's state: a NaN there ends the episode too, paying nothing (not `out` or `win`).
    assert (t["opponent_nan_state"].func, t["opponent_nan_state"].time_out) == (t["nan_state"].func, False)
    assert t["opponent_nan_state"].params == {"asset_cfg": SceneEntityCfg("opponent")}
    assert t["in_pit"].params["minimum_height"] == t["opponent_in_pit"].params["minimum_height"] == -arena_tug.PIT_DROP_M
    assert t["opponent_in_pit"].params["asset_cfg"].name == "opponent"
    assert t["fell"].func is rewards.ArenaFall
    assert t["fell"].params == {"fall_gravity_z": EVENT.termination.fall_gravity_z,
                                "hold_steps": round(EVENT.termination.fall_hold_s / 0.02)}
    assert (r["out"].func, r["out"].weight, r["out"].params) == (rewards.ended_by, -50.0, {"terms": ("in_pit", "fell")})
    assert (r["win"].func, r["win"].weight, r["win"].params) == (rewards.ended_by, 50.0, {"terms": ("opponent_in_pit",)})
    fired = {"in_pit": torch.tensor([True, False, False, True]), "fell": torch.tensor([False, True, False, False]),
             "opponent_in_pit": torch.tensor([False, False, True, True])}
    env = SimpleNamespace(termination_manager=SimpleNamespace(get_term=fired.get))
    paid = -50 * rewards.ended_by(env, ("in_pit", "fell")) + 50 * rewards.ended_by(env, ("opponent_in_pit",))
    assert paid.tolist() == [-50.0, -50.0, 50.0, 0.0]   # both in on one step: neither


def test_the_learners_randomisation_leaves_the_opponent_alone(cfg):
    for name, term in cfg.events.items():
        named = {v.name for v in term.params.values() if isinstance(v, SceneEntityCfg)}
        if name in ("reset_opponent_base", "reset_opponent_joints"):
            assert named == {"opponent"}, name
        else:
            assert named <= {"robot"}, name


def test_a_fall_is_the_arenas_rule(cfg):
    """Gravity z above fall_gravity_z for the hold's straight control steps, as arena.stage.fell_at counts it."""
    params = cfg.terminations["fell"].params
    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])   # env 0 upright, env 1 on its side
    env = SimpleNamespace(num_envs=2, device="cpu",
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))})
    term = rewards.ArenaFall(cfg.terminations["fell"], env)
    seen = [term(env, **params).tolist() for _ in range(params["hold_steps"])]
    assert seen[-2:] == [[False, False], [False, True]]   # out on the step that completes the hold
    term.reset(torch.tensor([1]))
    assert term(env, **params).tolist() == [False, False]


# Postures the posture probe measured (2026-10-22, pull-phase medians): trunk z m, pitch ° (+ the trunk's top leaning away
# from the opponent), roll °, head COM z m, neck_pitch and head_pitch off STAND °.
WALKER_BACK = (0.120, -1.0, 2.0, 0.236, 8.0, -1.0)     # BEST_alpha_walking backwards at -0.40
WALKER_STILL = (0.116, 2.0, 2.0, 0.234, 1.0, -1.0)     # it, at the tug's -0.30: it stands
LEANING = (0.101, 19.5, -9.5, 0.181, 29.5, -29.5)      # a puller leaning back, inside every edge
T17_250 = (0.072, -38.0, -11.0, 0.063, -69.0, -6.0)    # bowed, the head on the floor
T15_250 = (0.084, -33.0, 9.0, 0.064, -85.0, -10.0)
T19_250 = (0.071, -44.0, 12.0, 0.063, -50.0, 12.0)
OUT_BY_ONE = [(0.095, -1.0, 2.0, 0.236, 8.0, -1.0), (0.120, -11.0, 2.0, 0.236, 8.0, -1.0), (0.120, 21.0, 2.0, 0.236, 8.0, -1.0),
              (0.120, -1.0, 11.0, 0.236, 8.0, -1.0), (0.120, -1.0, -11.0, 0.236, 8.0, -1.0), (0.120, -1.0, 2.0, 0.175, 8.0, -1.0),
              (0.120, -1.0, 2.0, 0.236, -31.0, -1.0), (0.120, -1.0, 2.0, 0.236, 8.0, 31.0)]
ENTITY = SimpleNamespace(name="robot", body_ids=[0], joint_ids=[0, 1])
STAND = 0.3491   # neck_pitch and head_pitch at STAND, the training's default_joint_pos


def _rule_inputs(poses):
    """is_upright's (z, gravity_b, head_z, neck_dev) for rows of postures: projected gravity (-sin pitch, -sin roll,
    -√(1 - both²)), neck_pitch and head_pitch off STAND in rad."""
    poses = np.asarray(poses, dtype=np.float64)
    x, y = -np.sin(np.radians(poses[:, 1])), -np.sin(np.radians(poses[:, 2]))
    return poses[:, 0], np.stack([x, y, -np.sqrt(1 - x ** 2 - y ** 2)], axis=-1), poses[:, 3], np.radians(poses[:, 4:])


def _posed_env(z, pitch_deg, roll_deg, head_z, dneck_deg, dhead_deg, touching=False):
    """One env as the probe measured it: projected gravity (-sin pitch, -sin roll, -√(1 - both²)), the head's COM at
    head_z (body 0), neck_pitch and head_pitch off STAND (0.3491) by these, the body sensor's 13 slots touching or not."""
    z, gravity, head_z, neck = (torch.tensor(v, dtype=torch.float32)
                                for v in _rule_inputs([(z, pitch_deg, roll_deg, head_z, dneck_deg, dhead_deg)]))
    data = SimpleNamespace(root_link_pos_w=torch.cat([torch.zeros(1, 2), z[:, None]], dim=-1), projected_gravity_b=gravity,
                           body_com_pos_w=torch.cat([torch.zeros(1, 1, 2), head_z[:, None, None]], dim=-1),
                           joint_pos=STAND + neck, default_joint_pos=torch.full((1, 2), STAND))
    sensor = SimpleNamespace(data=SimpleNamespace(found=torch.full((1, 13), int(touching))))
    return SimpleNamespace(num_envs=1, device="cpu", scene={"robot": SimpleNamespace(data=data), tug.BODY_GROUND: sensor})


@pytest.mark.parametrize("pose, touching, fires_at", [(WALKER_BACK, False, None), (WALKER_STILL, False, None), (LEANING, False, None),
    (T17_250, False, 15), (T15_250, False, 15), (T19_250, False, 15), (WALKER_BACK, True, 15),
    *[(pose, False, 15) for pose in OUT_BY_ONE]])
def test_the_walking_upright_rule_keeps_the_walker_and_ends_the_bow(pose, touching, fires_at):
    """The walker walks or stands inside every band, a puller may lean back to 20°; the bowed pullers, a shin on the floor
    and each band crossed alone go down on the 15th straight step."""
    env = _posed_env(*pose, touching=touching)
    term = rewards.Down(None, env)
    fired = [bool(term(env, tug.BODY_GROUND, 15, ENTITY)[0]) for _ in range(20)]
    assert (fired.index(True) + 1 if any(fired) else None) == fires_at


def test_one_rule_trains_and_judges():
    """check.py and Down call one predicate: numpy and torch give the same mask."""
    poses = np.array([WALKER_BACK, WALKER_STILL, LEANING, T17_250, T15_250, T19_250, *OUT_BY_ONE])
    inputs = _rule_inputs(poses)
    numpy_mask = rewards.is_upright(*inputs)
    torch_mask = rewards.is_upright(*(torch.from_numpy(v) for v in inputs))
    assert numpy_mask.tolist() == torch_mask.tolist() == [True] * 3 + [False] * (3 + len(OUT_BY_ONE))


def test_a_down_count_restarts_on_a_reset():
    env = _posed_env(*T17_250)
    term = rewards.Down(None, env)
    for _ in range(14):
        term(env, tug.BODY_GROUND, 15, ENTITY)
    term.reset(torch.tensor([0]))
    assert not term(env, tug.BODY_GROUND, 15, ENTITY)[0]


def test_stay_upright_ends_a_down_duck_and_costs_its_pits():
    cfg = _cfg_with_knobs(stay_upright=1.0)
    down = cfg.terminations["down"]
    assert down.func is rewards.Down and (down.params["sensor_name"], down.params["hold_steps"]) == ("body_ground_contact", 15)
    entity = down.params["asset_cfg"]
    assert (entity.name, tuple(entity.body_names), tuple(entity.joint_names)) == ("robot", ("jaw_soft",), ("neck_pitch", "head_pitch"))
    sensor = {s.name: s for s in cfg.scene.sensors}["body_ground_contact"]
    assert (sensor.primary.mode, sensor.primary.pattern, sensor.primary.entity, tuple(sensor.primary.exclude)) == ("body", ".*", "robot", ("ankle_left", "ankle_right"))
    assert (sensor.secondary.mode, sensor.secondary.pattern, tuple(sensor.fields)) == ("body", "terrain", ("found",))
    assert (cfg.rewards["down"].weight, cfg.rewards["down"].params) == (-50.0, {"terms": ("down",)})
    assert cfg.rewards["out"].params == {"terms": ("in_pit",)} and "fell" not in cfg.terminations
    off = _cfg_with_knobs(stay_upright=0.0)
    assert "body_ground_contact" not in {s.name for s in off.scene.sensors} and "down" not in off.terminations
    assert "down" not in off.rewards and off.rewards["out"].params == {"terms": ("in_pit", "fell")}
    assert _cfg_with_knobs(stay_upright=2.0).rewards["down"].weight == -100.0


@pytest.fixture(scope="module")
def upright_live():
    """The owner's training env (stay upright 1.0), 2 envs, against the reference."""
    cfg = _cfg_with_knobs(stay_upright=1.0)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    yield env
    env.close()


def test_down_reads_the_head_the_neck_and_a_body_on_the_course(upright_live):
    entity = upright_live.termination_manager.get_term_cfg("down").params["asset_cfg"]
    robot = upright_live.scene["robot"]
    assert [robot.body_names[i] for i in entity.body_ids] == ["jaw_soft"]
    assert [robot.joint_names[i] for i in entity.joint_ids] == ["neck_pitch", "head_pitch"]
    upright_live.reset()
    upright_live.step(torch.zeros(2, 14))
    assert ((robot.data.body_com_pos_w[:, entity.body_ids[0], 2] - 0.237).abs() <= 0.015).all()   # STAND: 23.7 cm (probe)
    sensor = upright_live.scene["body_ground_contact"]
    assert not sensor.data.found.any()                                  # on its feet: the feet are not watched
    pose = robot.data.root_link_pose_w.clone()
    pose[1, 2] = 0.03                                                   # env 1's trunk sunk into the platform
    robot.write_root_link_pose_to_sim(pose)
    upright_live.sim.forward()
    upright_live.step(torch.zeros(2, 14))
    assert sensor.data.found.any(dim=-1).tolist() == [False, True]


def test_the_knobs_are_labelled_for_what_they_do():
    knobs = tomllib.loads((Path(tug.__file__).parent / "challenge.toml").read_text())["params"]
    assert knobs["upright_reward"]["label"] == "Keep the trunk level"
    stay = knobs["stay_upright"]
    assert (stay["label"], stay["min"], stay["max"]) == ("Stay upright", 0.0, 2.0)


def test_the_recipe_users_get_lets_the_duck_go_down():
    """Off by default (owner, 2026-10-21): other players may crawl; the owner's ducks trained with stay_upright 1.0."""
    params = tomllib.loads((Path(tug.__file__).with_name("challenge.toml")).read_text())["params"]
    knob = params["stay_upright"]
    shipped = tug.make_env_cfg()   # the committed toml, not the fixture's
    assert (knob["label"], knob["value"]) == ("Stay upright", 0.0)
    assert {k: params[k]["value"] for k in ("pull_reward", "back_reward", "upright_reward")} == {
        "pull_reward": 2.0, "back_reward": 3.0, "upright_reward": 1.0}   # t24 verified this recipe against t20-500 (+138.1 cm at 250, Mac)
    assert "down" not in shipped.terminations and "fell" in shipped.terminations


def _duck(pos, quat, **fields):
    return SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor(pos), root_link_quat_w=torch.tensor(quat), **fields))


def _straight(west, east):
    """(frames, STRING_POINTS, 3), west to east: a string straight between its ends' x per frame, y 0, z 0.1."""
    west, east = np.asarray(west, dtype=float)[:, None], np.asarray(east, dtype=float)[:, None]
    x = west + (east - west) * np.linspace(0.0, 1.0, arena_tug.STRING_POINTS)
    return np.stack([x, np.zeros_like(x), np.full_like(x, 0.1)], axis=-1)


def _strung(model, string):
    """An env of len(string) worlds whose string's points, site_xpos, are `string`, where the Arena records them."""
    site_xpos = torch.zeros(len(string), model.nsite, 3)
    site_xpos[:, arena_course.string_sites(model)] = torch.tensor(string, dtype=torch.float32)
    return SimpleNamespace(num_envs=len(string), device="cpu", step_dt=0.02,
                           sim=SimpleNamespace(mj_model=model, data=SimpleNamespace(site_xpos=site_xpos)))


def _marker_run(cfg, model, path, reset_at=None):
    """What marker_ground pays each step for learner string ends at `path[k]` (x per env, m), the
    opponent's end held at +0.5 m and the string straight between them, so its middle point, the
    marker, moves at half the learner's end; the env reset just before step `reset_at`."""
    params, n = cfg.rewards["marker_ground"].params, len(path[0])
    envs = [_strung(model, _straight(xs, [0.5] * n)) for xs in path]
    term = rewards.MarkerGround(cfg.rewards["marker_ground"], envs[0])
    paid = []
    for k, env in enumerate(envs):
        if k == reset_at:
            term.reset(torch.tensor([0]))
        paid.append(term(env, **params).tolist())
    return paid


def test_marker_ground_pays_new_ground_towards_the_learner(cfg, training_model):
    """Each step, the ground the marker gains west of the westmost point it reached this episode, at
    the scale's pace (0.05 m/s) tanh(1): a steady pull pays it every step, being dragged east nothing,
    a sway with no net gain only its first new ground. The knob a novice tunes is "Win ground"."""
    knob = tomllib.loads((Path(tug.__file__).parent / "challenge.toml").read_text())["params"]["pull_reward"]
    assert (knob["label"], knob["min"], knob["max"]) == ("Win ground", 0.5, 5.0)
    term = cfg.rewards["marker_ground"]
    assert term.func is rewards.MarkerGround and term.weight == knob["value"]
    assert term.params == {"scale": 0.05}
    step = 2 * 0.05 * 0.02   # the learner's end, so the marker moves 0.05 m/s over a control step
    sway = [-0.5, -0.5 - step, -0.5 + step] * 4
    paid = _marker_run(cfg, training_model, [(-0.5 - k * step, -0.5 + k * step, sway[k]) for k in range(12)])
    walk, dragged, swayed = zip(*paid)
    assert walk[0] == 0.0 and walk[1:] == pytest.approx([math.tanh(1.0)] * 11, abs=1e-4)
    assert dragged == (0.0,) * 12
    assert swayed[1] == pytest.approx(math.tanh(1.0), abs=1e-4) and swayed[:1] + swayed[2:] == (0.0,) * 11


def test_a_reset_clears_the_marker_record(cfg, training_model):
    """Back at the start after a reset, the ground the last episode won is new again."""
    step = 2 * 0.05 * 0.02
    path = [(-0.5 - k * step,) for k in range(10)] + [(-0.5 - k * step,) for k in range(3)]
    paid = _marker_run(cfg, training_model, path, reset_at=10)
    assert [p[0] for p in paid[10:]] == pytest.approx([0.0, math.tanh(1.0), math.tanh(1.0)], abs=1e-4)


def test_the_marker_is_where_the_arena_reads_it(training_model):
    """The string's points where the Arena records them: the marker is the middle one's x, the string's
    length its two ends' distance."""
    string = np.random.default_rng(2).uniform(-0.6, 0.6, (1, arena_tug.STRING_POINTS, 3))
    env = _strung(training_model, string)
    assert rewards.marker_x(env)[:, 0].tolist() == pytest.approx(arena_tug.marker_x(string).tolist(), abs=1e-6)
    length = np.linalg.norm(string[0, -1] - string[0, 0])
    assert rewards.string_length(env)[:, 0].tolist() == pytest.approx([length], abs=1e-6)


def test_the_critic_reads_the_opponent_from_the_learner(cfg):
    """The opponent's trunk from the learner's, its projected gravity, its world linear and own-frame
    angular velocities; the string's length and the marker, where the Arena reads them."""
    terms = cfg.observations["critic"].terms
    assert terms["opponent"].func is rewards.opponent_state
    for name in ("string_length", "marker_x"):
        assert (terms[name].func, terms[name].params) == (getattr(rewards, name), {})
    fields = ("projected_gravity_b", "root_link_lin_vel_w", "root_link_ang_vel_b")
    robot = _duck([[-0.53, 0.01, 0.11]], [[1.0, 0.0, 0.0, 0.0]])
    opponent = _duck([[0.55, -0.02, 0.12]], [[0.0, 0.0, 0.0, 1.0]], **{f: torch.rand(1, 3) for f in fields})
    state = rewards.opponent_state(SimpleNamespace(scene={"robot": robot, "opponent": opponent}))
    assert state.shape == (1, 12) and state[0, :3].tolist() == pytest.approx([1.08, -0.03, 0.01])
    assert torch.equal(state[0, 3:], torch.cat([getattr(opponent.data, f)[0] for f in fields]))


def _ends_apart(west_x, west_y, west_yaw, east_x, east_y, east_yaw):
    """The string ends' distance for trunks at these (x, y, yaw): STRING_OFFSET turned by each yaw."""
    ends = [(x + tug.STRING_OFFSET[0] * math.cos(yaw) - tug.STRING_OFFSET[1] * math.sin(yaw),
             y + tug.STRING_OFFSET[0] * math.sin(yaw) + tug.STRING_OFFSET[1] * math.cos(yaw))
            for x, y, yaw in ((west_x, west_y, west_yaw), (east_x, east_y, east_yaw))]
    return math.dist(*ends)


def test_the_string_ends_are_the_arenas(arena_model, training_model):
    for ours, theirs in (("robot/", ""), ("opponent/", "duck1/")):
        site, arena_site = training_model.site(ours + tug.STRING_SITE), arena_model.site(theirs + tug.STRING_SITE)
        assert training_model.body(site.bodyid[0]).name == ours + "trunk_base"
        np.testing.assert_allclose(site.pos, arena_site.pos, atol=1e-12)
    assert training_model.nu == arena_model.nu == 28   # two ducks' servos, nothing else


def _names(model, kind):
    """The model's names of a `kind` (body, jnt, geom or site), in id order."""
    return [getattr(model, kind)(i).name for i in range(getattr(model, f"n{kind}"))]


def _id(model, kind, name):
    return getattr(model, kind)(name).id


def _string_dofs(model):
    return [d for d in range(model.nv) if model.body(model.dof_bodyid[d]).name.startswith(tug.STRING)]


def test_the_string_is_the_arenas(arena_model, training_model):
    """Every string body, joint, dof, geom and site, its tree, both ties and both excludes, as compile_tug builds them."""
    for kind, fields in STRING_FIELDS.items():
        names = [n for n in _names(arena_model, kind) if n.startswith(tug.STRING)]
        assert names == [n for n in _names(training_model, kind) if n.startswith(tug.STRING)] and names
        for name in names:
            a, t = _id(arena_model, kind, name), _id(training_model, kind, name)
            for field in fields:
                np.testing.assert_allclose(getattr(training_model, field)[t], getattr(arena_model, field)[a], err_msg=f"{name} {field}")
    for field in ("dof_damping", "dof_armature"):
        np.testing.assert_array_equal(getattr(training_model, field)[_string_dofs(training_model)],
                                      getattr(arena_model, field)[_string_dofs(arena_model)])
    named = lambda name, prefixes: name.replace(prefixes[0], "").replace(prefixes[1], "#")   # "#" marks the east duck
    tree = lambda m, prefixes: [(m.body(b).name, named(m.body(m.body_parentid[b]).name, prefixes))
                                for b in range(m.nbody) if m.body(b).name.startswith(tug.STRING)]
    ties = lambda m, prefixes: [(named(m.site(m.eq_obj1id[e]).name, prefixes),
                                 m.site(m.eq_obj2id[e]).name, tuple(m.eq_solref[e]), tuple(m.eq_solimp[e]), int(m.eq_type[e])) for e in range(m.neq)]
    excludes = lambda m, prefixes: sorted((named(m.body(s >> 16).name, prefixes), named(m.body(s & 0xFFFF).name, prefixes))
                                          for s in m.exclude_signature)   # MuJoCo's (body1 << 16) + body2
    for wiring in (tree, ties, excludes):
        assert wiring(training_model, ("robot/", "opponent/")) == wiring(arena_model, ("", "duck1/"))
    assert training_model.nexclude == arena_model.nexclude == 2 and training_model.ntendon == 0


def test_the_bout_is_the_same_from_either_side():
    boxes = {(tuple(round(v, 9) for v in pos), tuple(size)) for _, pos, size in tug.course_boxes()}
    assert {((round(-x, 9), round(-y, 9), z), size) for (x, y, z), size in boxes} == boxes
    west, east = arena_course.tug_layout(EVENT.course)
    assert (east.x, east.y, east.yaw_deg) == pytest.approx((-west.x, -west.y, 180.0))
    assert tug.east_start() == pytest.approx((east.x, east.y)) and tug.west_start() == pytest.approx((west.x, west.y))


def test_no_start_stretches_the_string(cfg):
    """No corner of the start ranges puts the ends more than 5 mm past the string (the Arena's ±2° alone gives ~1 mm)."""
    west, east = cfg.events["reset_base"].params["pose_range"], cfg.events["reset_opponent_base"].params["pose_range"]
    corners = itertools.product(west["x"], west["y"], west["yaw"], east["x"], east["y"], east["yaw"])
    assert max(_ends_apart(*corner) for corner in corners) <= tug.STRING_M + 0.005   # STRING_OFFSET turned by each yaw
    assert west["x"][0] == pytest.approx(tug.west_start()[0]) and east["x"][1] == pytest.approx(tug.east_start()[0])


def test_the_string_end_is_where_the_arena_ties_it(arena_model):
    """pit_margin's string end, off the learner's trunk, is the west duck's string_end site in the Arena's model."""
    pos, quat = torch.tensor([[-0.53, -0.01, 0.11]]), torch.tensor([[0.97, 0.05, -0.2, 0.1]])
    quat = quat / quat.norm()
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos, root_link_quat_w=quat))})
    margin = rewards.pit_margin(env, tug.STRING_OFFSET, -tug.PIT_M / 2)
    data = mujoco.MjData(arena_model)
    free = arena_model.jnt_qposadr[arena_model.body("trunk_base").jntadr[0]]
    data.qpos[free:free + 7] = torch.cat([pos[0], quat[0]]).numpy()
    mujoco.mj_kinematics(arena_model, data)
    end = data.site_xpos[arena_model.site(tug.STRING_SITE).id]
    assert margin.item() == pytest.approx(-tug.PIT_M / 2 - end[0], abs=1e-6)


def test_the_manifest_says_what_the_policy_reads():
    assert "-0.30" in ch.for_task("Mjlab-TugOfWar-MicroDuck").command["twist"]


@pytest.fixture(scope="module")
def live():
    cfg = _cfg_with_knobs(stay_upright=0.0)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    yield env
    env.close()


def test_the_opponent_starts_where_the_arenas_east_duck_does(live):
    live.reset()
    east, data = arena_course.tug_layout(EVENT.course)[1], live.scene["opponent"].data
    pos, quat = data.root_link_pos_w, data.root_link_quat_w
    assert ((pos[:, 0] - east.x).abs() <= 0.02).all() and ((pos[:, 1] - east.y).abs() <= 0.01).all()
    yaw = torch.rad2deg(2 * torch.atan2(quat[:, 3], quat[:, 0])) % 360
    assert ((yaw - 180.0).abs() <= EVENT.start.yaw_deg + 1e-3).all()


def test_a_reset_lays_the_string_between_the_ducks(live):
    data, model = live.sim.data, live.sim.mj_model
    dofs = _string_dofs(model)
    site = lambda name: data.site_xpos[:, model.site(name).id]
    points = lambda: torch.stack([site(f"{tug.STRING}p{k}") for k in range(17)], dim=1)   # (2, 17, 3)

    def assert_laid_straight_and_centred(rows):
        pts, west, east = points()[rows], site("robot/string_end")[rows], site("opponent/string_end")[rows]
        gap = (east - west).norm(dim=-1)                                                     # 0.96–1.00 m: the starts' jitter
        assert (pts[:, 8] - (west + east) / 2).norm(dim=-1).max() <= 1e-5                    # centred between the ties
        over = (1.0 - gap).abs() / 2 + 1e-5                                                  # a straight 1 m string misses ties off 1 m apart
        assert ((pts[:, 0] - west).norm(dim=-1) <= over).all() and ((pts[:, 16] - east).norm(dim=-1) <= over).all()
        line = pts[:, 16] - pts[:, 0]
        off = torch.linalg.cross(pts - pts[:, :1], line[:, None].expand_as(pts)).norm(dim=-1) / line.norm(dim=-1)[:, None]
        assert off.max() <= 1e-5                                                             # straight
        assert not data.qvel[rows][:, dofs].any()

    for _ in range(5):   # the string has swung and is moving when the reset comes
        live.step(torch.full((2, 14), 0.3))
    assert data.qvel[:, dofs].any()
    live.reset()
    assert_laid_straight_and_centred(slice(None))

    for _ in range(5):   # a reset of env 1 alone leaves env 0's string where it swung
        live.step(torch.full((2, 14), 0.3))
    kept_points, kept_qvel = points()[0].clone(), data.qvel[0, dofs].clone()
    assert kept_qvel.any()
    live.reset(env_ids=torch.tensor([1], device=live.device))
    assert torch.equal(points()[0], kept_points) and torch.equal(data.qvel[0, dofs], kept_qvel)
    assert_laid_straight_and_centred(slice(1, 2))


def test_the_marker_is_the_strings_middle_point(live):
    live.reset()
    live.step(torch.zeros(2, 14))
    p8 = live.sim.data.site_xpos[:, live.sim.mj_model.site(f"{tug.STRING}p{rewards.MIDDLE}").id]
    assert rewards.MIDDLE == arena_tug.MIDDLE and torch.equal(rewards.marker_x(live)[:, 0], p8[:, 0])
    assert (tug.make_env_cfg().sim.nconmax, tug.make_env_cfg().sim.njmax) == (200, 1500)


def test_a_pit_on_either_side_ends_the_episode(live):
    live.reset()
    for env, name in ((0, "robot"), (1, "opponent")):
        duck = live.scene[name]
        pose = duck.data.root_link_pose_w.clone()
        pose[env, 2] = -0.2
        duck.write_root_link_pose_to_sim(pose)
    live.sim.forward()
    _, _, terminated, _, _ = live.step(torch.zeros(2, 14))
    assert terminated.tolist() == [True, True]
    assert live.termination_manager.get_term("in_pit").tolist() == [True, False]
    assert live.termination_manager.get_term("opponent_in_pit").tolist() == [False, True]


def test_a_bout_steps(live):
    live.reset()
    for _ in range(25):
        obs, *_ = live.step(torch.zeros(2, 14))
    assert obs["actor"].shape == (2, 61) and torch.isfinite(obs["critic"]).all()


# The east duck's qpos (trunk, then its joints) in a NaN world t9-250's learner met on CPU (taut, seed 6, step 107): fallen
# forward, its head shell flat on the platform. Under one 21.85 x 40.9 m platform box, mujoco_warp put that contact 95 m
# away and the step went NaN.
LYING = (0.628722, 0.022052, 0.047935, 0.040967, -0.411265, 0.012048, 0.910515, 0.079293, -0.131604, -0.048572, -0.478089,
         0.528811, 0.077669, 1.153934, -0.110056, 0.134809, 0.034916, 0.092652, -0.059213, 0.280138, -0.401073)


def test_a_duck_lying_on_the_platform_keeps_the_world_finite(live):
    live.reset()
    model = live.sim.mj_model
    adr = model.jnt_qposadr[model.body("opponent/trunk_base").jntadr[0]]
    live.sim.data.qpos[0, adr:adr + len(LYING)] = torch.tensor(LYING)
    live.sim.data.qvel[0] = 0.0
    live.sim.forward()
    live.step(torch.zeros(2, 14))
    assert not live.termination_manager.get_term("opponent_nan_state")[0]


def _relu_actor(tmp_path):
    """An actor the loader does not know: Sub, Div, Gemm, Relu, Gemm, 61 → 14."""
    rng, helper = np.random.default_rng(0), onnx.helper
    shapes = {"mean": (1, 61), "std": (1, 61), "w0": (32, 61), "b0": (32,), "w1": (14, 32), "b1": (14,)}
    weights = [onnx.numpy_helper.from_array(rng.normal(size=shape).astype(np.float32), name) for name, shape in shapes.items()]
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["centred"]), helper.make_node("Div", ["centred", "std"], ["scaled"]),
             helper.make_node("Gemm", ["scaled", "w0", "b0"], ["hidden"], transB=1), helper.make_node("Relu", ["hidden"], ["relu"]),
             helper.make_node("Gemm", ["relu", "w1", "b1"], ["actions"], transB=1)]
    graph = helper.make_graph(nodes, "actor", [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [1, 61])],
                              [helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, 14])], weights)
    path = tmp_path / "relu.onnx"
    onnx.save(helper.make_model(graph), path)
    return path


def test_the_opponent_net_is_its_onnx():
    """On observations as the reference's normalizer saw them: its mean plus its std times N(0, 1)."""
    actor = opponent.load_actor(opponent.REFERENCE)
    noise = torch.from_numpy(np.random.default_rng(0).normal(size=(1000, 61)).astype(np.float32))
    obs = actor[0].mean + actor[0].std * noise
    session = onnxruntime.InferenceSession(str(opponent.REFERENCE), providers=["CPUExecutionProvider"])
    theirs = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs.numpy()])
    with torch.no_grad():
        np.testing.assert_allclose(actor(obs).numpy(), theirs, atol=1e-5)


def test_the_learner_starts_as_the_reference_duck():
    """RL_CFG names its actor; PPO builds it from RL_CFG as below. The rows are observations as the
    reference's normalizer saw them (its constant command slots constant)."""
    normalize = opponent.load_actor(opponent.REFERENCE)[0]
    noise = torch.from_numpy(np.random.default_rng(0).normal(size=(4096, 61)).astype(np.float32))
    rows = normalize.mean + normalize.std * noise
    obs = TensorDict({"actor": rows}, batch_size=[len(rows)])
    cfg = tug.RL_CFG.actor
    assert resolve_callable(cfg.class_name) is opponent.ReferenceActor
    actor = resolve_callable(cfg.class_name)(
        obs, {"actor": ["actor"]}, "actor", 14, hidden_dims=cfg.hidden_dims, activation=cfg.activation,
        obs_normalization=cfg.obs_normalization, distribution_cfg=dict(cfg.distribution_cfg))
    session = onnxruntime.InferenceSession(str(opponent.REFERENCE), providers=["CPUExecutionProvider"])
    theirs = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in rows.numpy()])
    with torch.no_grad():
        np.testing.assert_allclose(actor(obs).numpy(), theirs, atol=1e-5)
        assert actor.distribution.std_param.tolist() == pytest.approx([0.5] * 14)
        # PPO's update at the first rollout step, every env at one state: the reference's statistics hold.
        actor.update_normalization(TensorDict({"actor": rows[:1].expand(4096, -1)}, batch_size=[4096]))
        np.testing.assert_allclose(actor(obs).numpy(), theirs, atol=1e-2)


def test_the_recipe_stops_after_the_warm_starts_peak():
    """The workshop trains challenge.toml's [recipe] with RL_CFG as it is. Started as the reference duck, t10
    dragged it into the pit by 500 (+139.6 cm) and lay prone by 1000 (-17.6 cm): the run stops there, and the
    best of its checkpoints at 250, 500, 750 and 1000 is kept, not the last."""
    recipe = tomllib.loads((Path(tug.__file__).parent / "challenge.toml").read_text())["recipe"]
    assert recipe["max_iterations"] == 1000 and tug.RL_CFG.save_interval == 250


def test_a_graph_the_loader_does_not_know_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Relu"):
        opponent.load_actor(_relu_actor(tmp_path))


def test_the_reference_ships_with_the_challenge():
    assert opponent.REFERENCE.read_bytes() == (ASSET_ROOT / "policies" / EVENT.rating.reference).read_bytes()
    root = Path(microduck_challenges.__file__).parents[2]
    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.split()
    assert "src/microduck_challenges/tug_of_war/reference.onnx" in listed   # hf_jobs.py tars exactly this list


def test_the_opponent_acts_on_the_arenas_joints(live):
    term, duck = live.action_manager.get_term("opponent"), live.scene["opponent"]
    assert term.target_names == list(JOINT_NAMES) and live.action_manager.total_action_dim == 14
    home = duck.data.default_joint_pos[:, term.target_ids]
    # mjlab_microduck's HOME_FRAME is rounded to 4 decimals: the learner carries the same gap.
    assert home[0].tolist() == pytest.approx(list(DEFAULT_POSE), abs=5e-5)
    act = torch.linspace(-0.2, 0.2, 14).expand(2, 14)
    policy, term.policy = term.policy, lambda obs: act
    try:
        term.process_actions(torch.zeros(2, 0))
        term.apply_actions()
    finally:
        term.policy = policy
    expected = home + act * ACTION_SCALE   # runner._Duck.act, on the robot's own default
    assert torch.allclose(duck.data.joint_pos_target[:, term.target_ids], expected, atol=1e-6)
    assert torch.equal(term.last_action, act)   # the policy's raw output, as runner._Duck keeps it


def _duck_addresses(model, prefix):
    """The duck's qpos and qvel addresses: its trunk's free joint, then its joints in JOINT_NAMES order."""
    trunk, joints = model.body(prefix + "trunk_base").jntadr[0], [model.joint(prefix + name) for name in JOINT_NAMES]
    qpos = [*range(model.jnt_qposadr[trunk], model.jnt_qposadr[trunk] + 7), *(j.qposadr[0] for j in joints)]
    qvel = [*range(model.jnt_dofadr[trunk], model.jnt_dofadr[trunk] + 6), *(j.dofadr[0] for j in joints)]
    return qpos, qvel


def test_the_opponent_observes_as_the_arena_does(live, arena_model, monkeypatch):
    """Env 0's opponent pitched 20°, yaw 170°, joints ± U(0.2) off home, velocities U(±2), forward; known memory.
    MjData(arena_model) gets env 0's raw qpos/qvel: trunk free joint by body, joints "opponent/x" → "duck1/x".
    `_observe` is fed the robot's own default pose: the same formula (the HOME_FRAME gap is pinned above)."""
    rng, term = np.random.default_rng(1), live.action_manager.get_term("opponent")
    home = live.scene["opponent"].data.default_joint_pos[0, term.target_ids].cpu().numpy().astype(np.float64)
    monkeypatch.setattr(arena_runner, "DEFAULT_POSE", home)
    (qpos, qvel), (their_qpos, their_qvel) = _duck_addresses(live.sim.mj_model, "opponent/"), _duck_addresses(arena_model, "duck1/")
    state = live.sim.data.qpos[0, qpos].cpu().numpy().astype(np.float64)
    mujoco.mju_mulQuat(state[3:7], [math.cos(math.radians(85.0)), 0.0, 0.0, math.sin(math.radians(85.0))],
                       [math.cos(math.radians(10.0)), 0.0, math.sin(math.radians(10.0)), 0.0])
    state[7:] = DEFAULT_POSE + rng.uniform(-0.2, 0.2, 14)
    live.sim.data.qpos[0, qpos] = torch.tensor(state, dtype=torch.float32)
    live.sim.data.qvel[0, qvel] = torch.tensor(rng.uniform(-2.0, 2.0, 20), dtype=torch.float32)
    live.sim.forward()
    last_action, prev_joint_vel = (rng.uniform(-1.0, 1.0, 14).astype(np.float32) for _ in range(2))
    term.last_action[0], term.prev_joint_vel[0] = torch.from_numpy(last_action), torch.from_numpy(prev_joint_vel)

    data = mujoco.MjData(arena_model)
    data.qpos[their_qpos], data.qvel[their_qvel] = live.sim.data.qpos[0, qpos].cpu().numpy(), live.sim.data.qvel[0, qvel].cpu().numpy()
    mujoco.mj_forward(arena_model, data)
    adr = arena_runner._Addresses.of(arena_model, EVENT.start.keyframe, "duck1/")
    theirs = arena_runner._observe(arena_model, data, adr, last_action, np.asarray(EVENT.command, dtype=np.float64), prev_joint_vel)
    np.testing.assert_allclose(term.observe()[0].numpy(), theirs, atol=1e-6)


def test_the_opponent_remembers_as_the_arena_does(live):
    term, duck = live.action_manager.get_term("opponent"), live.scene["opponent"]
    live.reset()
    live.step(torch.zeros(2, 14))   # moving: a reset's velocities are zeros
    before = duck.data.joint_vel[:, term.target_ids].clone()
    live.step(torch.zeros(2, 14))
    obs, now = term.observe(), duck.data.joint_vel[:, term.target_ids]
    assert torch.equal(obs[:, 20:34], before) and not torch.equal(obs[:, 20:34], now)   # a step old
    assert torch.equal(obs[:, 34:48], term.last_action) and term.last_action.any()


def test_the_opponents_servos_are_the_arenas(live):
    """The Arena runs its reference duck on arena.actuator's nominal battery: every opponent world at VIN_NOMINAL
    with VIN_DROP_GAIN's sag, where the learner's are drawn per world. Nothing else of its servos differs."""
    (servos,), (learners,) = live.scene["opponent"].actuators, live.scene["robot"].actuators
    assert torch.equal(servos.vin_tensor, torch.full_like(servos.vin_tensor, arena_actuator.VIN_NOMINAL))
    assert torch.equal(servos.vin_drop_gain, torch.full_like(servos.vin_drop_gain, arena_actuator.VIN_DROP_GAIN))
    assert replace(servos.cfg, vin_range=learners.cfg.vin_range,
                   vin_drop_gain_range=learners.cfg.vin_drop_gain_range) == learners.cfg
    # Its forcerange follows its battery (the Arena's is VIN_MAX's): the current window caps a servo first.
    model = live.sim.mj_model
    limits = [model.actuator_forcerange[a, 1] for a in range(model.nu) if model.actuator(a).name.startswith(tug.OPPONENT)]
    assert len(limits) == 14 and min(limits) > arena_actuator.KT * arena_actuator.MAX_CURRENT


def test_a_reset_clears_the_opponents_memory(live):
    term = live.action_manager.get_term("opponent")
    live.step(torch.zeros(2, 14))
    live.reset(env_ids=torch.tensor([1]))
    assert not term.last_action[1].any() and not term.prev_joint_vel[1].any() and term.last_action[0].any()


def _standing(frames, x=0.6):
    """(2, frames, 21): both ducks at STAND, trunks 12 cm up at -x and +x, facing each other: west unturned, east
    turned 180° about z."""
    trace = np.zeros((2, frames, 21), dtype=np.float32)
    trace[:, :, 2], trace[:, :, 7:] = 0.12, DEFAULT_POSE
    trace[0, :, 0], trace[0, :, 3] = -x, 1.0
    trace[1, :, 0], trace[1, :, 6] = x, 1.0
    return trace


def _bout(seed, loser, how, time_s, west_end=-0.5, east_end=0.4):
    """A bout whose string's ends stand still at these x, the settle's last frame too, the string straight
    between them, both ducks standing."""
    frames = round(time_s * 50)
    string = _straight(np.full(frames, west_end), np.full(frames, east_end))
    return arena_tug.BoutOutcome(seed, loser, how, time_s, _standing(frames), np.zeros((2, 1, 21)),
                                 string, _straight([west_end], [east_end]))


def _pulled_by_a_step_back():
    """2250 frames: the west end from -0.50 (the settle's last frame) back by 8 cm over 1.00 s, then still;
    the east end still at +0.42: the ends 0.92 m apart, 1.00 m from 1 s on, the string straight between them."""
    west = -0.50 - 0.08 * np.minimum(np.arange(1, 2251) / 50, 1.0)
    return arena_tug.BoutOutcome(1, None, "draw", 45.0, _standing(2250), np.zeros((2, 1, 21)),
                                 _straight(west, np.full(2250, 0.42)), _straight([-0.50], [0.42]))


def test_a_bout_line_says_how_the_pull_went():
    from microduck_challenges.tug_of_war import check
    outcome = _pulled_by_a_step_back()
    assert check.TAUT_M == 0.98   # the ends 0.98 m apart from the 38th frame
    assert check.pull(outcome, "west") == "taut at 0.76 s · A backed 8.0 cm by 2 s · marker 8.0 cm to A"
    assert check.pull(outcome, "east") == "taut at 0.76 s · A backed 0.0 cm by 2 s · marker -8.0 cm to A"
    held = _bout(1, None, "draw", 45.0, -0.485, 0.485)   # 0.97 m apart
    assert check.pull(held, "west") == "never taut · A backed 0.0 cm by 2 s · marker 0.0 cm to A"


def _turn(axis, deg):
    """The quaternion turning `deg` about the unit `axis`."""
    half = math.radians(deg) / 2
    return np.array([math.cos(half), *(math.sin(half) * np.asarray(axis, dtype=np.float64))])


def _posed_bout(frames, side, x=0.6):
    """Both ducks standing at -x and +x, facing each other (_standing); A, on `side`, posed frame by frame from
    dict(z, pitch_deg, roll_deg, dneck_deg, dhead_deg): its quaternion the yaw (0 west, 180° east), then the pitch (a
    nose-up rotation is +), then the roll, in the trunk's frame; its joints STAND but neck_pitch and head_pitch."""
    a, n = ("west", "east").index(side), len(frames)
    trace, neck = _standing(n, x), [JOINT_NAMES.index(name) for name in ("neck_pitch", "head_pitch")]
    for k, pose in enumerate(frames):
        tilt, quat = np.empty(4), np.empty(4)
        mujoco.mju_mulQuat(tilt, _turn((0, 1, 0), -pose["pitch_deg"]), _turn((1, 0, 0), pose["roll_deg"]))
        mujoco.mju_mulQuat(quat, _turn((0, 0, 1), 180.0 * a), tilt)
        trace[a, k, 2], trace[a, k, 3:7] = pose["z"], quat
        trace[a, k, 7:][neck] += np.radians([pose["dneck_deg"], pose["dhead_deg"]])
    return arena_tug.BoutOutcome(1, None, "draw", n * 0.02, trace, np.zeros((2, 1, 21)), np.zeros((n, arena_tug.STRING_POINTS, 3)),
                                 np.zeros((1, arena_tug.STRING_POINTS, 3)))


WALK = dict(z=0.12, pitch_deg=-1.0, roll_deg=2.0, dneck_deg=8.0, dhead_deg=-1.0)       # the walker backwards
BOW = dict(z=0.072, pitch_deg=-38.0, roll_deg=-11.0, dneck_deg=-69.0, dhead_deg=-6.0)  # t17-250


def _feet(n=100, period=20, lag=10):
    """(contact, feet): each foot on the floor period/2 frames then in the air as long, the right one `lag` frames behind;
    in the air a foot is 1 cm up and moves 3 mm back a frame, on the floor it stays still."""
    frame = np.arange(n)
    contact = np.stack([frame % period < period // 2, (frame - lag) % period < period // 2], axis=1)
    feet = np.zeros((n, 2, 3))
    feet[..., 2] = np.where(contact, 0.0, 0.01)
    feet[..., 0] = -0.003 * np.cumsum(~contact, axis=0)
    return contact, feet


def test_gait_counts_clear_steps_and_the_slide():
    from microduck_challenges.tug_of_war import check
    contact, feet = _feet()
    every = np.ones(100, dtype=bool)
    assert check.gait(contact, feet, every) == "4.0 steps/s · slide 0 %"   # left lands at 20, 40, 60, 80; right at 30, 50, 70, 90
    assert check.gait(contact, feet, np.arange(100) < 50) == "3.0 steps/s · slide 0 %"   # 20, 40 and 30 in the counted second
    dragged = np.zeros((100, 2, 3))
    dragged[:, :, 0] = -0.001 * np.arange(100)[:, None]                   # both feet down, slid 1 mm a frame
    assert check.gait(np.ones((100, 2), dtype=bool), dragged, every) == "0.0 steps/s · slide 100 %"
    chatter = np.ones((100, 2), dtype=bool)
    chatter[10::10, 0] = False                                            # the left foot off one frame in ten
    lifted = dragged.copy()
    lifted[10::10, 0, 2] = 0.001                                          # 1 mm up: no clear step
    assert check.gait(chatter, lifted, every) == "0.0 steps/s · slide 91 %"   # 180 of 198 moves in contact
    assert check.gait(np.ones((100, 2), dtype=bool), np.zeros((100, 2, 3)), every) == "standing still"
    assert check.gait(contact, feet, np.zeros(100, dtype=bool)) == "standing still"


def test_a_bout_line_says_how_a_walked(arena_model):
    from microduck_challenges.tug_of_war import check
    for side in ("west", "east"):
        assert check.posture(_posed_bout([WALK] * 50, side), side, arena_model) == "upright 100 % · longest out 0.00 s · feet only 100 % · standing still"
    for side in ("west", "east"):
        assert check.posture(_posed_bout([dict(WALK, pitch_deg=15.0)] * 50, side), side, arena_model).startswith("upright 100 % · ")
        assert check.posture(_posed_bout([dict(WALK, pitch_deg=-15.0)] * 50, side), side, arena_model).startswith("upright 0 % · ")
    assert check.posture(_posed_bout([BOW] * 50, "west"), "west", arena_model).startswith("upright 0 % · longest out 1.00 s · ")
    dipped = _posed_bout([WALK] * 20 + [BOW] * 15 + [WALK] * 15, "west")
    assert check.posture(dipped, "west", arena_model).startswith("upright 70 % · longest out 0.30 s · ")
    assert check.posture(_posed_bout([dict(WALK, z=0.02)] * 50, "west"), "west", arena_model).endswith(" · feet only 0 % · standing still")
    assert check.posture(_posed_bout([BOW] * 50, "west", x=0.20), "west", arena_model) == "upright 100 % · longest out 0.00 s · feet only 100 % · standing still"   # at the lip: left out


def test_the_summary_counts_who_lost_and_how(arena_model):
    from microduck_challenges.tug_of_war import check
    results = [("west", _bout(1, 0, "pit", 6.2)), ("east", _bout(2, 0, "pit", 3.0)),
               ("west", _bout(3, 1, "marker", 9.0)), ("east", _bout(4, None, "draw", 1.0))]
    lines = check.summary(results, arena_model)
    assert lines[0].startswith("bout 1 · seed 1 · A west · A loses by pit at 6.20 s · ")
    assert lines[1].startswith("bout 2 · seed 2 · A east · B loses by pit at 3.00 s · ")
    assert lines[2].startswith("bout 3 · seed 3 · A west · B loses by marker at 9.00 s · ")
    assert lines[3].startswith("bout 4 · seed 4 · A east · drawn at 1.00 s · ")
    assert lines[4] == ("A 2 · B 1 · draws 1 · A in the pit 1 · B in the pit 1 · by marker 1"
                        " · median decision 4.6 s")


def test_the_check_plays_the_reference_by_default(capsys):
    from microduck_challenges.tug_of_war import check
    walker = str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx")
    assert check.main([walker, "--bouts", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("bout 1 · seed 1 · A west") and lines[1].startswith("bout 2 · seed 2 · A east")
    assert lines[2].startswith("A ") and "median decision" in lines[2]
    assert all("upright 100 % · longest out 0.00 s · feet only 100 % · " in line and "steps/s" in line for line in lines[:2])


def test_an_unreadable_policy_is_reported(tmp_path, capsys):
    from microduck_challenges.tug_of_war import check
    missing = str(tmp_path / "gone.onnx")
    assert check.main([missing, missing]) == 1
    assert missing in capsys.readouterr().err
