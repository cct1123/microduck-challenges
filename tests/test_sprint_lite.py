"""Slim CPU Sprint contracts, written after implementation; no archived models or rollouts."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from microduck_challenges.sprint_2m import campaign, diagnostics, env, experiments, promotion, rewards


@pytest.fixture
def source(tmp_path):
    """A tiny local clock payload plus an independently reconstructed known recipe."""
    baseline = env.make_env_cfg(stage="baseline")
    settings = env.REFINE_SETTINGS
    selected = {"checkpoint": settings["source_baseline_checkpoint"],
                "source_agent": {"num_steps_per_env": 24},
                "source_env": rewards._config_data(baseline)}
    mature = rewards.configure_drive_locomotion(deepcopy(baseline), settings, selected)
    mature = rewards.configure_leg_freedom(rewards._mature_refine_recipe(mature, settings))
    path = tmp_path / "model_1.pt"
    torch.save({"infos": {"env_state": {"common_step_counter": 222648}},
                "sprint_reward_profile": "A0"}, path)
    selected.update(path=path, source_env=rewards._config_data(mature),
                    source_agent={"num_steps_per_env": 24, "algorithm": {
                        "class_name": "microduck_challenges.sprint_2m.rewards:SprintRefinePPO",
                        "reward_profile": "A0", "exploration_std_scale": 1.0}})
    return baseline, selected


@pytest.mark.parametrize("profile", experiments.REWARD_PROFILES)
def test_profile_changes_only_named_weight_and_cannot_reramp(source, profile):
    baseline, selected = source
    control = rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected)
    candidate = rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected,
                                                   reward_profile=profile)
    overrides = experiments.reward_overrides(profile)
    before, after = rewards._config_data(control), rewards._config_data(candidate)
    assert list(before["observations"]["actor"]["terms"]) == list(after["observations"]["actor"]["terms"])
    for group in ("observations", "actions", "scene", "sim", "commands", "events", "terminations"):
        assert before[group] == after[group]
    assert candidate.actions["joint_pos"].scale == 1 and candidate.actions["joint_pos"].clip is None
    for name in candidate.rewards:
        expected = {**before["rewards"][name], "weight": overrides.get(name, control.rewards[name].weight)}
        assert after["rewards"][name] == expected
    terms = {name: SimpleNamespace(weight=term.weight) for name, term in candidate.rewards.items()}
    probe = SimpleNamespace(device="cpu", common_step_counter=10**9,
                            reward_manager=SimpleNamespace(get_term_cfg=terms.__getitem__))
    rewards.speed_reward_curriculum(probe, None, **candidate.curriculum["speed_rewards"].params)
    assert all(terms[name].weight == weight for name, weight in overrides.items())
    assert control.rewards["speed_forward"].weight == 12 and control.rewards["air_time"].weight == 6
    assert control.rewards["pose"].weight == 0


def test_profile_source_is_independently_validated_and_marker_bound(source):
    baseline, selected = source
    candidate = rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected,
                                                   reward_profile="A1")
    selected["source_env"] = rewards._config_data(candidate)
    selected["source_agent"]["algorithm"]["reward_profile"] = "A1"
    with pytest.raises(ValueError, match="saved reward profile"):
        rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected)
    torch.save({"infos": {"env_state": {"common_step_counter": 222648}},
                "sprint_reward_profile": "A1"}, selected["path"])
    before = promotion.digest(selected["path"])
    result = rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected,
                                                reward_profile="A1")
    assert result.rewards["track_linear_velocity"].weight == 0
    selected["source_env"]["rewards"]["heading_hold"]["weight"] = 0
    with pytest.raises(ValueError, match="Invalid source reward weight"):
        rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected,
                                            reward_profile="A1")
    assert promotion.digest(selected["path"]) == before


def test_reward_finiteness_signs_and_held_fall_guard():
    data = SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(4, 1),
                           projected_gravity_b=torch.tensor([[0., 0, -1.]]).repeat(4, 1),
                           root_link_lin_vel_w=torch.tensor([[2., 0, 0], [-2., 0, 0],
                                                            [float("nan"), 0, 0], [0., 0, 0]]))
    probe = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, num_envs=4,
                            device="cpu", step_dt=.02, episode_length_buf=torch.ones(4))
    actual = rewards.course_forward_speed(probe)
    assert torch.isfinite(actual).all() and actual[0] > 0 and actual[1] < 0
    assert actual[2] == actual[3] == 0
    guard = rewards.ArenaTiltGuard(None, probe)
    data.projected_gravity_b[:, 2] = 0
    for _ in range(14):
        assert not guard(probe).any()
    assert guard(probe).all()
    assert not rewards.course_forward_speed(probe, max_tilt_deg=70).any()
    guard.reset(torch.tensor([0]))
    assert not guard.failed[0] and guard.failed[1]


def test_strict_warm_start_preserves_weights_normalizers_clock_and_fresh_adam(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    algorithm = cast(Any, object.__new__(rewards.SprintRefinePPO))
    obs = TensorDict({"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 81)}, batch_size=[1])
    for name, width in (("actor", 14), ("critic", 1)):
        model = MLPModel(obs, {name: [name]}, name, width, hidden_dims=(8,), obs_normalization=True)
        cast(Any, model.obs_normalizer)._mean.fill_(2)
        setattr(algorithm, name, model)
    algorithm.rnd = None
    algorithm.exploration_std_scale, algorithm.reward_profile = 1.0, "A1"
    algorithm.optimizer = torch.optim.Adam([*algorithm.actor.parameters(), *algorithm.critic.parameters()], lr=5e-5)
    payload = {"actor_state_dict": deepcopy(algorithm.actor.state_dict()),
               "critic_state_dict": deepcopy(algorithm.critic.state_dict()),
               "infos": {"env_state": {"common_step_counter": 222648}}, "iter": 100}
    path = tmp_path / "source.pt"
    torch.save(payload, path)
    before = promotion.digest(path)
    for model in (algorithm.actor, algorithm.critic):
        model.obs_normalizer._mean.zero_()
    assert algorithm.load(torch.load(path, map_location="cpu", weights_only=False)) is False
    for name in ("actor", "critic"):
        for key, value in getattr(algorithm, name).state_dict().items():
            torch.testing.assert_close(value, payload[f"{name}_state_dict"][key], rtol=0, atol=0)
    assert not algorithm.optimizer.state and algorithm.optimizer.param_groups[0]["lr"] == 5e-5
    assert algorithm.save()["sprint_reward_profile"] == "A1"
    assert payload["infos"]["env_state"]["common_step_counter"] == 222648
    with pytest.raises(ValueError, match="strict full"):
        algorithm.load(payload, strict=False)
    bad = deepcopy(payload)
    bad["critic_state_dict"] = {"bad": torch.zeros(1)}
    with pytest.raises(RuntimeError):
        algorithm.load(bad)
    assert promotion.digest(path) == before


def test_export_preserves_normalization_and_61_to_14_contract(tmp_path):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    config = rewards._config_data(env.RL_CFG)["actor"]
    actor = MLPModel(TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1]),
                     {"actor": ["actor"]}, "actor", 14, hidden_dims=config["hidden_dims"],
                     activation=config["activation"], obs_normalization=True,
                     distribution_cfg=deepcopy(config["distribution_cfg"]))
    cast(Any, actor.obs_normalizer)._mean.fill_(.2)
    cast(Any, actor.obs_normalizer)._std.fill_(1.7)
    (tmp_path / "params").mkdir()
    (tmp_path / "params/agent.yaml").write_text(yaml.safe_dump({"actor": config}))
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint)
    result = campaign.export_actor(checkpoint, tmp_path / "policy.onnx")
    assert result["input_shape"] == [1, 61] and result["output_shape"] == [1, 14]
    assert result["parity_cases"] == 8


def test_audit_uses_exact_iteration_complete_terms_and_honest_accounting(tmp_path):
    (tmp_path / "params").mkdir()
    cfg = {"rewards": {"speed_forward": {"weight": 12}, "air_time": {"weight": 6}},
           "curriculum": {}, "episode_length_s": 20, "scale_rewards_by_dt": True}
    (tmp_path / "params/env.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "params/agent.yaml").write_text(yaml.safe_dump({"experiment_name": "sprint_2m", "algorithm": {}}))
    with SummaryWriter(str(tmp_path)) as writer:
        for step in range(263, 266):
            writer.add_scalar("Train/mean_reward", 245.5, step)
            writer.add_scalar("Episode_Reward/speed_forward", 6, step)
            writer.add_scalar("Episode_Reward/air_time", 4, step)
    result = diagnostics.audit_run(tmp_path, 264, 1)
    assert result["mean_reward"]["at_iteration"] == 245.5
    assert result["positive_logged_sum"] == 10 and not result["missing_anchor_terms"]
    assert result["fixed_logged_data_weight_estimates"]["B1"]["air_time"]["estimated_logged_contribution"] == 2
    assert "last 100" in result["accounting"] and "not retrained" in result["accounting"]
    assert "speed_forward" in diagnostics.markdown_report(result)
    with pytest.raises(ValueError, match="no Train/mean_reward"):
        diagnostics.audit_run(tmp_path, 266)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        diagnostics.scalar_series([SimpleNamespace(step=1, value=1), SimpleNamespace(step=1, value=2)], "x")


def test_scoring_and_launch_defaults_fail_closed(tmp_path, monkeypatch):
    rows = [{"seed": 1, "status": "finish", "time_s": 1., "command_vx": .3},
            {"seed": 2, "status": "fall", "time_s": None, "command_vx": .3}]
    assert promotion.summarize(rows, [1, 2], .3)["mean_s"] == 25.5
    with pytest.raises(ValueError, match="every expected seed"):
        promotion.summarize(rows + rows, [1, 2], .3)
    with pytest.raises(ValueError, match="Unknown Sprint reward"):
        experiments.reward_overrides("A1+B1")
    with pytest.raises(ValueError, match="approved promotion report"):
        promotion.require_promotion(None, tmp_path / "model_0.pt", {})
    monkeypatch.setattr(campaign, "active_trainers", lambda: [123])
    with pytest.raises(ValueError, match="trainer"):
        campaign.run_bounded({}, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_recipe_default_and_explicit_control_reach_the_factory(source, monkeypatch):
    _, selected = source
    assert env.STAGE_SETTINGS["reward_profile"] == "C1"
    default = experiments.resolve_reward_profile(env.STAGE_SETTINGS, "refine")
    assert default == "C1"
    assert experiments.resolve_reward_profile(env.STAGE_SETTINGS, "refine", "A0") == "A0"
    assert experiments.resolve_reward_profile(env.STAGE_SETTINGS, "baseline") == "A0"
    with pytest.raises(ValueError, match="strict refinement"):
        experiments.resolve_reward_profile(env.STAGE_SETTINGS, "baseline", "A1")
    monkeypatch.setattr(env, "REFINE_CHECKPOINT", selected)
    monkeypatch.setattr(env, "TRAINING_REWARD_PROFILE", default)
    cfg = env.make_env_cfg(stage="refine")
    control = env.make_env_cfg(stage="refine", reward_profile="A0")
    assert cfg.rewards["track_linear_velocity"].weight == 0
    assert control.rewards["track_linear_velocity"].weight > 0
    assert cfg.rewards["speed_forward"].weight == control.rewards["speed_forward"].weight == 12
    assert cfg.rewards["air_time"].weight == 3 and control.rewards["air_time"].weight == 6


@pytest.mark.parametrize("before_profile,after_profile,name,initial,target", (
    ("A1", "C1", "air_time", 6, 3),
))
def test_successive_profiles_change_only_named_reward_and_endpoint(
    source, before_profile, after_profile, name, initial, target,
):
    baseline, selected = source
    control = rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected,
                                                 reward_profile=before_profile)
    candidate = rewards.configure_refine_locomotion(deepcopy(baseline), env.REFINE_SETTINGS, selected,
                                                   reward_profile=after_profile)
    before, after = rewards._config_data(control), rewards._config_data(candidate)
    assert before["rewards"]["track_linear_velocity"]["weight"] == after["rewards"]["track_linear_velocity"]["weight"] == 0
    assert before["rewards"][name]["weight"] == initial
    assert after["rewards"][name]["weight"] == target
    after["rewards"][name]["weight"] = initial
    targets = after["curriculum"]["speed_rewards"]["params"]["targets"]
    if name in targets:
        assert tuple(targets[name]) == (target, target)
        targets[name] = before["curriculum"]["speed_rewards"]["params"]["targets"][name]
    assert before == after
    assert experiments.reward_overrides("B1") == {"air_time": 3.0}
    assert experiments.reward_overrides("B2") == {"speed_forward": 18.0}
