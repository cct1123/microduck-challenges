"""Hill-local software contracts; no archived models, network, or simulator rollouts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from microduck_challenges.hill_climb_1m5 import env


def test_default_is_real_from_scratch_bootstrap_and_preserves_hill_contract():
    cfg = env.make_env_cfg(stage="bootstrap")
    assert env.TRAINING_STAGE == "bootstrap"
    assert env.RL_CFG.max_iterations == 3000 and not env.RL_CFG.resume
    assert env.RL_CFG.algorithm.learning_rate == 1e-3
    assert env.RL_CFG.algorithm.schedule == "adaptive"
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
        "command", "head_command", "body_command"]
    assert "heading_error" not in cfg.observations["actor"].terms
    assert cfg.observations["critic"].terms["heading_error"].func is env.heading_error
    assert cfg.rewards["heading_hold"].weight == 1
    assert cfg.commands["twist"].ranges.lin_vel_x == (.15, 1.)
    assert cfg.commands["twist"].ranges.lin_vel_y == (0., 0.)
    assert cfg.actions["joint_pos"].scale == 1 and cfg.actions["joint_pos"].clip is None
    assert cfg.scene.terrain.terrain_type == "generator"
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (30, 50)
    assert cfg.rewards["action_rate_l2"].weight == -.1
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["step"] == 1500 * 24
    with pytest.raises(ValueError, match="bootstrap.*A0"):
        env.make_env_cfg(stage="bootstrap", reward_profile="A1")


def test_refinement_is_explicit_hill_only_and_freezes_curricula(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="explicit.*source"):
        env.make_env_cfg(stage="refine")
    with pytest.raises(ValueError, match="exact.*Hill"):
        env.make_env_cfg(stage="refine", source=tmp_path / "model_01.pt")


def test_independent_profiles_only_change_named_rewards_and_freeze_endpoints():
    from microduck_challenges.hill_climb_1m5 import learning

    control = learning.refinement_recipe(env.make_env_cfg(stage="bootstrap"), 36001, "A0")
    assert control.rewards["action_rate_l2"].weight == -1
    assert control.rewards["head_pose_bias"].weight == 3
    assert control.commands["twist"].rel_standing_envs == .2
    assert control.events["randomize_com"].params["ranges"] == (-.015, .015)
    for profile, overrides in learning.REWARD_PROFILES.items():
        cfg = learning.refinement_recipe(env.make_env_cfg(stage="bootstrap"), 36001, profile)
        before, after = learning.config_data(control), learning.config_data(cfg)
        for name, weight in overrides.items():
            assert after["rewards"][name]["weight"] == weight
            after["rewards"][name]["weight"] = before["rewards"][name]["weight"]
        assert before == after
    assert learning.REWARD_PROFILES["A1"] == {"track_linear_velocity": 0.}
    assert learning.REWARD_PROFILES["B1"] == {"air_time": 1.5}
    with pytest.raises(ValueError, match="Unknown Hill"):
        learning.refinement_recipe(control, 36000, "C1")
    assert learning.refinement_recipe(env.make_env_cfg(stage="bootstrap"), 12000, "A0").rewards["action_rate_l2"].weight == -.1
    assert learning.refinement_recipe(env.make_env_cfg(stage="bootstrap"), 12001, "A0").rewards["action_rate_l2"].weight == -.2


def test_strict_restore_keeps_normalizers_and_environment_age_but_not_adam_or_runner():
    from microduck_challenges.hill_climb_1m5 import learning
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    algorithm = object.__new__(learning.HillPPO)
    obs = TensorDict({"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 77)}, batch_size=[1])
    for name, width in (("actor", 14), ("critic", 1)):
        model = MLPModel(obs, {name: [name]}, name, width, hidden_dims=(8,), obs_normalization=True,
                         distribution_cfg=({"class_name": "GaussianDistribution", "init_std": 1.,
                                            "std_type": "scalar"} if name == "actor" else None))
        model.obs_normalizer._mean.fill_(2)
        setattr(algorithm, name, model)
    algorithm.rnd = None
    algorithm.stage, algorithm.reward_profile = "refine", "A1"
    algorithm.source_clock, algorithm.exploration_std_scale = 36000, .5
    algorithm.optimizer = torch.optim.Adam([*algorithm.actor.parameters(), *algorithm.critic.parameters()], lr=1e-5)
    payload = {"actor_state_dict": deepcopy(algorithm.actor.state_dict()),
               "critic_state_dict": deepcopy(algorithm.critic.state_dict()),
               "infos": {"env_state": {"common_step_counter": 36000}}, "iter": 1499}
    assert algorithm.load(payload) is False
    for name in ("actor", "critic"):
        for key, value in getattr(algorithm, name).state_dict().items():
            expected = payload[f"{name}_state_dict"][key]
            if name == "actor" and key == "distribution.std_param":
                expected = expected * .5
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not algorithm.optimizer.state and algorithm.optimizer.param_groups[0]["lr"] == 1e-5
    assert payload["infos"]["env_state"]["common_step_counter"] == 36000
    assert algorithm.save()["hill_reward_profile"] == "A1"
    with pytest.raises(ValueError, match="strict full"):
        algorithm.load(payload, strict=False)
    broken = deepcopy(payload)
    broken["critic_state_dict"] = {"bad": torch.zeros(1)}
    with pytest.raises(RuntimeError):
        algorithm.load(broken)


def _hill_runner(*, play=False, stage="bootstrap"):
    """Real registered runner/algorithm; only the simulator boundary is replaced."""
    from dataclasses import asdict
    from mjlab.tasks.registry import load_env_cfg, load_runner_cls
    from tensordict import TensorDict
    from microduck_challenges.hill_climb_1m5.learning import TASK

    cfg = load_env_cfg(TASK, play=play)
    raw = SimpleNamespace(common_step_counter=0)
    marker = cfg.events.get("actor_only_inference")
    if marker is not None:
        assert play and marker.mode == "startup"
        marker.func(raw, None, **marker.params)
    obs = TensorDict({"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 77)}, batch_size=[1])
    wrapped = SimpleNamespace(unwrapped=raw, cfg=cfg, num_envs=1, num_actions=14,
                              get_observations=lambda: obs)
    agent = asdict(env.BASE_RL_CFG)
    for name in ("actor", "critic"):
        agent[name]["hidden_dims"] = (8,)
    if stage == "refine":
        agent["algorithm"].update(stage="refine", source_clock=36000,
                                  curriculum_clock=12000, exploration_std_scale=.5)
    runner_cls = load_runner_cls(TASK)
    assert runner_cls is not None
    return runner_cls(wrapped, agent, device="cpu")


@pytest.mark.parametrize("stage", ["bootstrap", "refine"])
def test_registered_play_runner_loads_only_exact_actor_without_training_resume(tmp_path, stage):
    source = _hill_runner()
    source.alg.actor.obs_normalizer._mean.fill_(.2)
    source.alg.actor.obs_normalizer._std.fill_(1.7)
    payload = {"actor_state_dict": deepcopy(source.alg.actor.state_dict()),
               "infos": {"env_state": {"common_step_counter": 24}}, "iter": 123,
               "hill_exploration_std_scale": .8}
    path = tmp_path / "model_123.pt"
    torch.save(payload, path)
    before = path.read_bytes()
    runner = _hill_runner(play=True, stage=stage)
    critic = deepcopy(runner.alg.critic.state_dict())
    # The real installed runner calls construct_algorithm and then alg.load.
    runner.load(str(path), map_location="cpu")
    assert runner.alg._hill_inference_only
    assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
    for key, value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(value, payload["actor_state_dict"][key], rtol=0, atol=0)
    for key, value in runner.alg.critic.state_dict().items():
        torch.testing.assert_close(value, critic[key], rtol=0, atol=0)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="inference-only"):
        runner.alg.train_mode()
    training = _hill_runner()
    assert not getattr(training.alg, "_hill_inference_only", False)
    with pytest.raises(ValueError, match="Bootstrap is from scratch"):
        training.load(str(path), load_cfg={"actor": True, "critic": False,
                      "optimizer": False, "iteration": False, "rnd": False}, map_location="cpu")


@pytest.mark.parametrize("corruption", ["nonfinite", "negative_std", "missing_normalizer", "wrong_shape", "non_strict"])
def test_play_restore_rejects_invalid_or_partial_actor(corruption):
    runner = _hill_runner(play=True)
    state = deepcopy(runner.alg.actor.state_dict())
    if corruption == "nonfinite":
        state["mlp.0.weight"][0, 0] = float("nan")
    elif corruption == "negative_std":
        state["obs_normalizer._std"][0, 0] = -1
    elif corruption == "missing_normalizer":
        state.pop("obs_normalizer._mean")
    elif corruption == "wrong_shape":
        state["mlp.0.weight"] = state["mlp.0.weight"][:, :-1]
    with pytest.raises((ValueError, RuntimeError), match="Nonfinite|standard deviation|strict|Missing key|size mismatch"):
        runner.alg.load({"actor_state_dict": state}, strict=corruption != "non_strict")


def test_training_refinement_still_requires_full_strict_source_and_retains_both_clocks(tmp_path, monkeypatch):
    monkeypatch.delenv("MICRODUCK_WARM_START", raising=False)
    source = _hill_runner()
    payload = {**source.alg.save(), "infos": {"env_state": {"common_step_counter": 36000}},
               "iter": 1499, "hill_exploration_std_scale": .8}
    # Invalid old optimizer proves that training warm starts never load it.
    payload["optimizer_state_dict"] = {"must_not_load": True}
    path = tmp_path / "model_1499.pt"
    torch.save(payload, path)
    runner = _hill_runner(stage="refine")
    with pytest.raises(RuntimeError, match="strict source restore"):
        runner.alg.train_mode()
    with pytest.raises(ValueError, match="strict full"):
        runner.alg.load(payload, {"actor": True, "critic": False})
    broken = deepcopy(payload)
    broken["infos"]["env_state"]["common_step_counter"] += 24
    with pytest.raises(ValueError, match="clock changed"):
        runner.alg.load(broken)
    broken = deepcopy(payload)
    broken["critic_state_dict"].pop("obs_normalizer._mean")
    with pytest.raises(RuntimeError, match="Missing key"):
        runner.alg.load(broken)
    with pytest.raises(RuntimeError, match="strict source restore"):
        runner.alg.train_mode()
    monkeypatch.setenv("MICRODUCK_WARM_START", "1")
    with pytest.raises(ValueError, match="restart Hill"):
        runner.alg.load(payload)
    monkeypatch.delenv("MICRODUCK_WARM_START")
    runner.load(str(path), map_location="cpu")
    runner.alg.train_mode()
    assert runner.env.unwrapped.common_step_counter == 36000
    assert runner.alg.curriculum_clock == 12000 and runner.current_learning_iteration == 0
    assert not runner.alg.optimizer.state
    for name in ("actor", "critic"):
        for key, value in getattr(runner.alg, name).state_dict().items():
            expected = payload[f"{name}_state_dict"][key]
            if name == "actor" and key == "distribution.std_param":
                expected = expected * (.5 / .8)
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_smoke_api_is_available_without_a_checkpoint_or_launch_permission():
    from microduck_challenges.hill_climb_1m5 import campaign
    import inspect
    assert callable(campaign.smoke)
    assert inspect.signature(campaign.smoke).parameters["seed"].default is None


def test_plan_is_nonlaunching_bootstrap_has_its_own_budget_and_no_archive(monkeypatch):
    from microduck_challenges.hill_climb_1m5 import campaign
    plan = campaign.make_plan()
    assert plan["launching"] is False and plan["source"] is None
    assert plan["max_iterations"] == 3000
    assert plan["controls"]["num_envs"] == env.RECIPE["num_envs"]
    assert plan["command"][1:3] == ["-m", "mjlab.scripts.train"]
    import tyro
    import mjlab
    from mjlab.scripts.train import TrainConfig
    parsed = tyro.cli(TrainConfig, args=plan["command"][4:],
                      default=TrainConfig.from_task(campaign.TASK), config=mjlab.TYRO_FLAGS)
    assert parsed.agent.max_iterations == 3000 and parsed.env.scene.num_envs == env.RECIPE["num_envs"]
    assert parsed.gpu_ids == [0] and not parsed.agent.resume
    assert parsed.agent.algorithm.stage == "bootstrap" and parsed.agent.algorithm.reward_profile == "A0"
    with pytest.raises(ValueError, match="explicit.*source"):
        campaign.make_plan(stage="refine")
    with pytest.raises(ValueError, match="stage budget"):
        campaign.make_plan(iterations=3001)
    with pytest.raises(ValueError, match="Bootstrap"):
        campaign.make_plan(phase="exploit")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [123])
    with pytest.raises(ValueError, match="trainer"):
        campaign.run_bounded({}, Path("must-not-create"))


def test_hill_penalties_inventory_frozen_command_and_unfiltered_neighbors(tmp_path):
    from microduck_challenges.hill_climb_1m5 import evidence
    rows = [{"seed": 1, "status": "finish", "time_s": 2., "command_vx": .55},
            {"seed": 2, "status": "fall", "time_s": None, "command_vx": .55}]
    assert evidence.summarize(rows, [1, 2], .55)["mean_s"] == 31
    with pytest.raises(ValueError, match="sweep"):
        evidence.summarize(rows, [1, 2], .551)
    with pytest.raises(ValueError, match="exactly once"):
        evidence.summarize(rows * 2, [1, 2], .55)
    (tmp_path / "params").mkdir()
    (tmp_path / "params/agent.yaml").write_text(yaml.safe_dump({"max_iterations": 5, "save_interval": 2}))
    for n in (0, 2, 4):
        (tmp_path / f"model_{n}.pt").touch()
    assert evidence.saved_checkpoints(tmp_path) == [0, 2, 4]
    (tmp_path / "model_2.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        evidence.saved_checkpoints(tmp_path)


def test_promotion_recomputes_raw_rows_and_does_not_skip_failed_neighbors():
    from microduck_challenges.hill_climb_1m5 import evidence
    gate = {**env.TRAINING["gate"], "command_vx": .55}
    def rows(seeds, time=2., fail=False):
        return [{"seed": s, "status": "fall" if fail and s == 1 else "finish",
                 "time_s": time, "command_vx": .55} for s in seeds]
    report = {"schema": 1, "status": "complete", "gate": gate,
              "checkpoints": [0, 50, 100, 150, 199], "candidate_checkpoint": 0,
              "screen": {str(n): rows(range(1, 6), fail=n in (50, 150)) for n in (0, 50, 100, 150, 199)},
              "candidate_heldout": rows(range(6, 26)), "incumbent_rows": rows(range(1, 26), time=3.)}
    assert evidence.promotion_decision(report, gate)["approved"] is False
    report["screen"] = {str(n): rows(range(1, 6), time=2 + n / 1000) for n in report["checkpoints"]}
    assert evidence.promotion_decision(report, gate)["approved"] is True
    report["candidate_heldout"][0]["status"] = "fall"
    assert evidence.promotion_decision(report, gate)["approved"] is False


@pytest.fixture
def hill_source(tmp_path, monkeypatch):
    import json
    from microduck_challenges.hill_climb_1m5 import learning
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    monkeypatch.setattr(learning, "LOG_ROOT", tmp_path)
    folder = tmp_path / "hill-source"
    (folder / "params").mkdir(parents=True)
    (folder / "workshop").mkdir()
    cfg, agent = env.make_env_cfg(stage="bootstrap"), learning.config_data(env.BASE_RL_CFG)
    (folder / "params/env.yaml").write_text(yaml.safe_dump(learning.config_data(cfg), sort_keys=False))
    (folder / "params/agent.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
    (folder / "provenance.json").write_text(json.dumps({"task": learning.TASK}))
    obs = TensorDict({"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 77)}, batch_size=[1])
    payload = {"infos": {"env_state": {"common_step_counter": 24}}, "iter": 0,
               "hill_reward_profile": "A0", "hill_exploration_std_scale": 1.}
    for name, width in (("actor", 14), ("critic", 1)):
        params = agent[name]
        model = MLPModel(obs, {name: [name]}, name, width, hidden_dims=params["hidden_dims"],
                         activation=params["activation"], obs_normalization=True,
                         distribution_cfg=deepcopy(params["distribution_cfg"]))
        model.obs_normalizer._mean.fill_(.2)
        model.obs_normalizer._std.fill_(1.7)
        payload[f"{name}_state_dict"] = model.state_dict()
    path = folder / "model_0.pt"
    torch.save(payload, path)
    result = {"checkpoint": 0, "score": 5., "command_vx": .55, "seeds_total": 5, "seeds_finished": 5,
              "per_seed": [{"seed": s, "status": "finish", "time_s": 5.} for s in range(1, 6)]}
    (folder / "workshop/model_0.json").write_text(json.dumps(result))
    return path, payload


def test_source_validation_binds_hill_task_order_recipe_profile_and_both_clocks(hill_source):
    import json
    from microduck_challenges.hill_climb_1m5 import learning
    path, payload = hill_source
    select = lambda: learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)
    assert select()["clock"] == 24
    payload["infos"]["env_state"]["common_step_counter"] = 25
    torch.save(payload, path)
    with pytest.raises(ValueError, match="clock.*iteration"):
        select()
    payload["infos"]["env_state"]["common_step_counter"] = 24
    payload["hill_reward_profile"] = "A1"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="profile"):
        select()
    payload["hill_reward_profile"] = "A0"
    torch.save(payload, path)
    config_path = path.parent / "params/env.yaml"
    original = config_path.read_text()
    config = yaml.safe_load(original)
    config["observations"]["actor"]["terms"] = dict(reversed(list(config["observations"]["actor"]["terms"].items())))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="ordered actor"):
        select()
    config_path.write_text(original)
    (path.parent / "provenance.json").write_text(json.dumps({"task": "Mjlab-Sprint2m-MicroDuck"}))
    with pytest.raises(ValueError, match="not Hill"):
        select()


def test_export_includes_normalizer_and_checks_finite_61_to_14_parity(hill_source, tmp_path):
    from microduck_challenges.hill_climb_1m5 import campaign, evidence
    path, payload = hill_source
    payload["actor_state_dict"]["obs_normalizer._std"][0, 49] = 0.
    torch.save(payload, path)
    before = evidence.digest(path)
    exported = campaign.export_actor(path, tmp_path / "hill.onnx")
    assert exported["input_shape"] == [1, 61] and exported["output_shape"] == [1, 14]
    assert exported["parity_cases"] == 8 and evidence.digest(path) == before
    with pytest.raises(ValueError, match="previous exports"):
        campaign.export_actor(path, tmp_path / "hill.onnx")
    payload["actor_state_dict"]["obs_normalizer._mean"][0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Nonfinite"):
        campaign.export_actor(path, tmp_path / "bad.onnx")


def test_campaign_export_rejects_negative_normalizer_before_serialization(hill_source, tmp_path):
    from microduck_challenges.hill_climb_1m5 import campaign
    path, payload = hill_source
    payload["actor_state_dict"]["obs_normalizer._std"][0, 49] = -1.
    torch.save(payload, path)
    output = tmp_path / "negative.onnx"
    with pytest.raises(ValueError, match="standard deviation"):
        campaign.export_actor(path, output)
    assert not output.exists()


def test_source_accepts_only_known_runtime_terrain_hydration(hill_source):
    from microduck_challenges.hill_climb_1m5 import learning
    path, _ = hill_source
    config_path = path.parent / "params/env.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["scene"]["num_envs"] = cfg["scene"]["terrain"]["num_envs"] = 2
    generator = cfg["scene"]["terrain"]["terrain_generator"]
    for sub in generator["sub_terrains"].values():
        sub["size"] = generator["size"]
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)["clock"] == 24
    generator["sub_terrains"]["slope"]["size"] = [9., 9.]
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="terrain"):
        learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)


def test_refinement_descendant_preserves_frozen_curriculum_not_just_age(hill_source):
    from microduck_challenges.hill_climb_1m5 import learning
    path, payload = hill_source
    agent_path = path.parent / "params/agent.yaml"
    agent = yaml.safe_load(agent_path.read_text())
    agent["algorithm"].update(stage="refine", source_clock=12000, curriculum_clock=12000)
    agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
    cfg = learning.refinement_recipe(env.make_env_cfg(stage="bootstrap"), 12000, "A0")
    (path.parent / "params/env.yaml").write_text(yaml.safe_dump(learning.config_data(cfg), sort_keys=False))
    payload["infos"]["env_state"]["common_step_counter"] = 12024
    torch.save(payload, path)
    descendant = env.make_env_cfg(stage="refine", source=path)
    assert descendant.rewards["action_rate_l2"].weight == -.1


def test_source_requires_full_model_and_normalizer_state_before_launch(hill_source):
    from microduck_challenges.hill_climb_1m5 import learning
    path, payload = hill_source
    payload["actor_state_dict"].pop("obs_normalizer._mean")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="full.*state"):
        learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)


def test_zero_variance_commands_are_valid_normalizer_state(hill_source):
    from microduck_challenges.hill_climb_1m5 import learning
    path, payload = hill_source
    payload["actor_state_dict"]["obs_normalizer._std"][0, 49] = 0.
    torch.save(payload, path)
    assert learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)["clock"] == 24
    payload["actor_state_dict"]["obs_normalizer._std"][0, 49] = -1.
    torch.save(payload, path)
    with pytest.raises(ValueError, match="standard deviation"):
        learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)


def test_run_without_execute_only_prints_plan(monkeypatch, capsys):
    import sys
    from microduck_challenges.hill_climb_1m5 import campaign
    monkeypatch.setattr(sys, "argv", ["campaign", "run"])
    monkeypatch.setattr(campaign, "run_bounded", lambda *a: pytest.fail("unauthorized launch"))
    campaign.main()
    assert '"launching": false' in capsys.readouterr().out


@pytest.mark.parametrize("journal_error", [False, True])
def test_bounded_launcher_reaps_only_its_own_nontraining_child(tmp_path, monkeypatch, journal_error):
    import sys
    import subprocess
    from microduck_challenges.hill_climb_1m5 import campaign
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "smoke", lambda *a: None)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    spawned = []
    real_spawn = subprocess.Popen
    def spawn(*a, **kw):
        process = real_spawn(*a, **kw)
        spawned.append(process)
        return process
    monkeypatch.setattr(campaign.subprocess, "Popen", spawn)
    write_json = campaign.write_json
    def write(path, data):
        if journal_error and data.get("state") == "running":
            raise OSError("injected journal error")
        write_json(path, data)
    monkeypatch.setattr(campaign, "write_json", write)
    plan = {"requires_promotion": False, "max_wall_minutes": .001, "source": None,
            "source_files": {}, "source_config_files": {}, "environment": {},
            "stage": "bootstrap", "phase": "explore", "reward_profile": "A0", "seed": 4,
            "command": [sys.executable, "-c", "import time; time.sleep(60)"]}
    if journal_error:
        with pytest.raises(OSError, match="injected"):
            campaign.run_bounded(plan, tmp_path / "result")
    else:
        result = campaign.run_bounded(plan, tmp_path / "result")
        assert result["state"] == "wall_budget_exhausted" and result["exit_code"] != 0
    assert len(spawned) == 1 and spawned[0].poll() is not None


def test_launcher_lock_is_exclusive_and_isolated_from_real_logs(tmp_path, monkeypatch):
    import fcntl
    from microduck_challenges.hill_climb_1m5 import campaign
    root = tmp_path / "logs/rsl_rl/hill_climb_1m5"
    monkeypatch.setattr(campaign, "LOG_ROOT", root)
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    root.parent.parent.mkdir(parents=True)
    with (root.parent.parent / ".hill-strategy.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="launch lock"):
            campaign.run_bounded({}, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def _bind_campaign(path, payload, tmp_path, iterations=1):
    import hashlib
    import json
    from microduck_challenges.hill_climb_1m5 import campaign, evidence
    plan = campaign.make_plan(iterations=iterations, seed=4)
    plan["controls"]["save_interval"] = 1
    plan["command"][plan["command"].index("--agent.save-interval") + 1] = "1"
    frozen = tmp_path / "frozen-source.py"
    frozen.write_text("# immutable launch source\n")
    plan["source_files"] = {str(frozen): evidence.digest(frozen)}
    journal_path = tmp_path / "launch.json"
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    agent_path = path.parent / "params/agent.yaml"
    agent = yaml.safe_load(agent_path.read_text())
    agent.update(max_iterations=iterations, save_interval=1, seed=4, run_name=plan["command"][-1])
    agent["algorithm"].update(campaign_journal=str(journal_path), campaign_plan_digest=identity)
    agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
    payload.update(hill_campaign_journal=str(journal_path), hill_campaign_plan_digest=identity)
    torch.save(payload, path)
    campaign.write_json(path.parent / "provenance.json", {"task": campaign.TASK, "seed": 4})
    journal = {"plan": plan, "plan_digest": identity, "state": "completed", "exit_code": 0,
               "source_files_unchanged": True, "source_artifacts_unchanged": True}
    campaign.write_json(journal_path, journal)
    return journal_path, journal


def test_campaign_source_cannot_bypass_launchers_source_change_veto(hill_source, tmp_path):
    from microduck_challenges.hill_climb_1m5 import campaign, learning
    path, payload = hill_source
    journal_path, journal = _bind_campaign(path, payload, tmp_path)
    journal.update(state="source_changed_not_promotable", source_files_unchanged=False)
    campaign.write_json(journal_path, journal)
    with pytest.raises(ValueError, match="campaign.*completed|Campaign.*completed"):
        learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)


def test_bounded_launch_binds_child_config_and_checkpoint_to_exact_plan(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import campaign, evidence
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "smoke", lambda *a: None)
    captured = {}
    class Child:
        pid, returncode = 999999, 0
        def wait(self, **kw): return 0
        def poll(self): return 0
    def spawn(*a, **kw):
        captured.update(kw["env"])
        return Child()
    plan = campaign.make_plan(iterations=1, seed=4)
    output = tmp_path / "launch"
    with monkeypatch.context() as boundary:
        boundary.setattr(campaign.subprocess, "run", lambda *a, **kw: None)
        boundary.setattr(campaign.subprocess, "Popen", spawn)
        result = campaign._run_bounded(plan, output)
    assert captured["MICRODUCK_HILL_CAMPAIGN_JOURNAL"] == str(output / "run.json")
    assert captured["MICRODUCK_HILL_CAMPAIGN_PLAN_DIGEST"] == evidence.plan_digest(plan)
    assert result["plan_digest"] == evidence.plan_digest(plan)
    # Registry import must consume binding before the trainer saves its agent.
    import os
    import subprocess
    import sys
    check = ("import json; from dataclasses import asdict; "
             "from microduck_challenges.hill_climb_1m5.env import RL_CFG; "
             "print('BINDING=' + json.dumps(asdict(RL_CFG.algorithm)))")
    actual = subprocess.check_output([sys.executable, "-c", check],
                                    env={**os.environ, **captured}, text=True)
    import json
    cfg = json.loads(actual.split("BINDING=")[1])
    assert cfg["campaign_journal"] == str(output / "run.json")
    assert cfg["campaign_plan_digest"] == result["plan_digest"]
    runner = _hill_runner()
    runner.alg.campaign_journal = cfg["campaign_journal"]
    runner.alg.campaign_plan_digest = cfg["campaign_plan_digest"]
    assert runner.alg.save()["hill_campaign_journal"] == cfg["campaign_journal"]


def _assessment_fixture(hill_source, tmp_path, monkeypatch):
    import shutil
    from microduck_challenges.hill_climb_1m5 import campaign
    from arena import course, manifest, policy, runner
    path, payload = hill_source
    incumbent = tmp_path / "incumbent" / path.name
    shutil.copytree(path.parent, incumbent.parent)
    journal_path, journal = _bind_campaign(path, payload, tmp_path, iterations=3)
    for number in (1, 2):
        other = deepcopy(payload)
        other["iter"] = number
        other["infos"]["env_state"]["common_step_counter"] = (number + 1) * 24
        torch.save(other, path.parent / f"model_{number}.pt")
        official = campaign.read_report(path.parent / "workshop/model_0.json")
        official["checkpoint"] = number
        campaign.write_json(path.parent / "workshop" / f"model_{number}.json", official)
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(course, "compile_course", lambda *a: None)
    monkeypatch.setattr(manifest, "load_event", lambda *a: SimpleNamespace(command_at=lambda vx: vx))
    monkeypatch.setattr(policy, "load_policy", lambda path: path)
    calls = []
    def race(manifest, policy, seed, **kw):
        calls.append(seed)
        return SimpleNamespace(status="finish", time_s=3. if policy.stem == "incumbent" else 2., progress_m=1.5)
    monkeypatch.setattr(runner, "run_once", race)
    def export(checkpoint, output):
        output.write_bytes(b"unit-test export boundary")
        return {"checkpoint_sha256": campaign.digest(checkpoint), "onnx_sha256": campaign.digest(output)}
    monkeypatch.setattr(campaign, "export_actor", export)
    return path, incumbent, journal_path, journal, calls


def test_assessment_and_exploitation_preserve_launch_veto_and_journal_hashes(hill_source, tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import campaign, evidence
    path, incumbent, journal_path, journal, calls = _assessment_fixture(hill_source, tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    report = campaign.assess(path.parent, incumbent, output, .55)
    assert report["decision"]["approved"]
    assert report["campaign_evidence"][str(path)]["journal_sha256"] == evidence.digest(journal_path)
    assert report["artifact_hashes"][str(journal_path)] == evidence.digest(journal_path)
    assert evidence.require_promotion(output / "assessment.json", path, report["gate"])
    # Omission must not turn a bound campaign into an ordinary run.
    missing = deepcopy(report)
    missing.pop("campaign_evidence")
    campaign.write_json(output / "assessment.json", missing)
    with pytest.raises(ValueError, match="campaign"):
        evidence.require_promotion(output / "assessment.json", path, report["gate"])
    campaign.write_json(output / "assessment.json", report)
    journal["state"] = "source_changed_not_promotable"
    campaign.write_json(journal_path, journal)
    with pytest.raises(ValueError, match="campaign"):
        evidence.require_promotion(output / "assessment.json", path, report["gate"])
    calls.clear()
    rejected = tmp_path / "vetoed-assessment"
    with pytest.raises(ValueError, match="campaign"):
        campaign.assess(path.parent, incumbent, rejected, .55)
    assert not calls and not rejected.exists()


def test_campaign_binding_rejects_partial_mismatched_or_changed_launch_inputs(hill_source, tmp_path):
    from microduck_challenges.hill_climb_1m5 import campaign, evidence, learning
    path, payload = hill_source
    journal_path, journal = _bind_campaign(path, payload, tmp_path)
    agent_path = path.parent / "params/agent.yaml"
    original_agent = agent_path.read_text()
    original_payload = deepcopy(payload)
    assert evidence.campaign_evidence(path)["plan_digest"] == evidence.plan_digest(journal["plan"])
    # Each independent corruption must fail without optional caller arguments.
    for corruption in ("half_agent", "half_checkpoint", "mismatch", "missing", "running", "failed",
                       "veto", "plan", "frozen", "source_artifact", "budget", "run_name", "task"):
        agent_path.write_text(original_agent)
        payload = deepcopy(original_payload)
        changed = deepcopy(journal)
        frozen = Path(next(iter(journal["plan"]["source_files"])))
        frozen.write_text("# immutable launch source\n")
        if corruption == "half_agent":
            agent = yaml.safe_load(original_agent)
            agent["algorithm"]["campaign_plan_digest"] = None
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
        elif corruption == "half_checkpoint": payload.pop("hill_campaign_journal")
        elif corruption == "mismatch": payload["hill_campaign_plan_digest"] = "0" * 64
        elif corruption in ("running", "failed", "veto"):
            changed["state"] = "source_changed_not_promotable" if corruption == "veto" else corruption
        elif corruption == "plan": changed["plan"]["max_iterations"] += 1
        elif corruption == "frozen": frozen.write_text("# changed\n")
        elif corruption == "source_artifact":
            artifact = tmp_path / "old-source.pt"
            artifact.write_bytes(b"original")
            changed["plan"]["source_config_files"] = {str(artifact): evidence.digest(artifact)}
            changed["plan_digest"] = evidence.plan_digest(changed["plan"])
            payload["hill_campaign_plan_digest"] = changed["plan_digest"]
            agent = yaml.safe_load(original_agent)
            agent["algorithm"]["campaign_plan_digest"] = changed["plan_digest"]
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
            artifact.write_bytes(b"changed")
        elif corruption in ("budget", "run_name"):
            agent = yaml.safe_load(original_agent)
            agent["max_iterations" if corruption == "budget" else "run_name"] = 99 if corruption == "budget" else "other"
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
        elif corruption == "task":
            campaign.write_json(path.parent / "provenance.json", {"task": "wrong", "seed": 4})
        torch.save(payload, path)
        campaign.write_json(journal_path, changed)
        if corruption == "missing": journal_path.unlink()
        with pytest.raises(ValueError, match="campaign|Campaign|not Hill"):
            learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)
    # Metadata-absent ordinary Workshop runs remain supported.
    agent_path.write_text(original_agent)
    agent = yaml.safe_load(original_agent)
    for key in ("campaign_journal", "campaign_plan_digest"): agent["algorithm"].pop(key)
    agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
    for key in ("hill_campaign_journal", "hill_campaign_plan_digest"): payload.pop(key, None)
    torch.save(payload, path)
    campaign.write_json(path.parent / "provenance.json", {"task": campaign.TASK, "seed": 4})
    assert learning.validate_source(path, env.make_env_cfg(stage="bootstrap"), env.BASE_RL_CFG)["clock"] == 24


def test_assessment_rechecks_journal_after_last_heldout_race(hill_source, tmp_path, monkeypatch):
    from arena import runner
    from microduck_challenges.hill_climb_1m5 import campaign
    path, incumbent, journal_path, journal, calls = _assessment_fixture(hill_source, tmp_path, monkeypatch)
    race = runner.run_once
    def change_on_last_race(manifest, policy, seed, **kw):
        result = race(manifest, policy, seed, **kw)
        if policy.stem == "incumbent" and seed == 25:
            journal["state"] = "source_changed_not_promotable"
            campaign.write_json(journal_path, journal)
        return result
    monkeypatch.setattr(runner, "run_once", change_on_last_race)
    output = tmp_path / "assessment"
    with pytest.raises(ValueError, match="campaign"):
        campaign.assess(path.parent, incumbent, output, .55)
    assert campaign.read_report(output / "assessment.json").get("decision", {}).get("approved") is not True


@pytest.mark.parametrize("stage", ["bootstrap", "refine"])
def test_standard_smoke_exercises_official_cpu_export_of_new_checkpoint(hill_source, tmp_path, monkeypatch, stage):
    from dataclasses import asdict
    import shutil
    import mjlab.scripts.train as train
    import mjlab_microduck.export as official
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    from microduck_challenges.hill_climb_1m5 import campaign
    source, _ = hill_source
    source_bytes = source.read_bytes()
    calls = []
    def train_without_simulator(task, cfg, log_dir):
        log_dir.mkdir()
        obs = TensorDict({"actor": torch.zeros(2, 61), "critic": torch.zeros(2, 77)}, batch_size=[2])
        raw = SimpleNamespace(device="cpu", common_step_counter=0,
                              reward_manager=SimpleNamespace(get_term_cfg=lambda name: cfg.env.rewards[name]))
        def step(*a): raw.common_step_counter += 1
        wrapped = SimpleNamespace(unwrapped=raw, cfg=cfg.env, num_envs=2, num_actions=14,
                                  get_observations=lambda: obs, step=step)
        runner = MicroduckOnPolicyRunner(wrapped, asdict(cfg.agent), device="cpu")
        if stage == "refine":
            runner.load(str(source), map_location="cpu")
        runner.alg.train_mode()  # Training must never reuse actor-only inference.
        for _ in range(48): wrapped.step()
        payload = {**runner.alg.save(), "iter": 1,
                   "infos": {"env_state": {"common_step_counter": raw.common_step_counter}}}
        torch.save(payload, log_dir / "model_1.pt")
        (log_dir / "params").mkdir()
        (log_dir / "params/agent.yaml").write_text(yaml.dump(asdict(cfg.agent)))
        campaign.write_json(log_dir / "provenance.json", {"task": task, "seed": cfg.agent.seed})
        with SummaryWriter(str(log_dir)) as writer: writer.add_scalar("unit/finite", 1., 1)
    def export_without_simulator(task, cfg):
        calls.append((task, cfg))
        shutil.copyfile(Path(cfg.onnx_file).parent / "verified.onnx", cfg.onnx_file)
        return SimpleNamespace(onnx_path=Path(cfg.onnx_file), checkpoint_path=Path(cfg.checkpoint_file))
    monkeypatch.setattr(train, "run_train", train_without_simulator)
    monkeypatch.setattr(official, "run_export", export_without_simulator)
    output = tmp_path / "smoke"
    report = campaign.smoke(stage=stage, source=source if stage == "refine" else None, output=output, seed=4)
    assert len(calls) == 1
    task, config = calls[0]
    assert task == campaign.TASK and config.device == "cpu" and config.num_envs == 1
    assert Path(config.checkpoint_file) == output / "cpu/model_1.pt"
    assert Path(config.onnx_file) != output / "verified.onnx"
    assert report["official_export"]["parity_cases"] == 8
    assert report["official_export"]["input_shape"] == [1, 61]
    assert report["official_export"]["output_shape"] == [1, 14]
    assert not (output / "cpu/workshop").exists()  # No forged 5/5 on the new checkpoint.
    assert source.read_bytes() == source_bytes and report["racing_performance_verified"] is False


def test_approval_must_match_saved_raw_rows_not_only_an_approved_flag(tmp_path):
    import json
    from microduck_challenges.hill_climb_1m5 import evidence
    rows = [{"seed": 1, "status": "finish", "time_s": 2., "command_vx": .55}]
    raw = tmp_path / "model_0-screen.jsonl"
    raw.write_text(json.dumps(rows[0]) + "\n")
    report = {"screen": {"0": rows}, "checkpoints": [0], "candidate_checkpoint": 0,
              "artifact_hashes": {str(raw): evidence.digest(raw)}}
    # A fabricated summary must fail against preserved raw rows, before approval.
    report["screen"]["0"][0]["time_s"] = 1.
    with pytest.raises(ValueError, match="raw.*rows"):
        evidence.verify_raw_rows(report, tmp_path)
