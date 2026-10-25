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
    assert plan["controls"]["num_envs"] == 4096
    assert plan["command"][1:3] == ["-m", "mjlab.scripts.train"]
    import tyro
    import mjlab
    from mjlab.scripts.train import TrainConfig
    parsed = tyro.cli(TrainConfig, args=plan["command"][4:],
                      default=TrainConfig.from_task(campaign.TASK), config=mjlab.TYRO_FLAGS)
    assert parsed.agent.max_iterations == 3000 and parsed.env.scene.num_envs == 4096
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
