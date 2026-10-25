"""Hill-local bootstrap/refinement contracts, not Sprint checkpoint migration."""

import math
import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg
from rsl_rl.algorithms import PPO

TASK = "Mjlab-HillClimb1m5-MicroDuck"
LOG_ROOT = Path("logs/rsl_rl/hill_climb_1m5").resolve()
REWARD_PROFILES = {"A0": {}, "A1": {"track_linear_velocity": 0.0}, "B1": {"air_time": 1.5}}


class ConfigLoader(yaml.SafeLoader):
    """Inspect saved configs without constructing arbitrary Python objects."""


def _tag(loader, suffix, node):
    if node.tag.startswith("tag:yaml.org,2002:python/name:"):
        return suffix.removeprefix("name:")
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _tag)


def config_data(cfg):
    return yaml.load(yaml.dump(asdict(cfg), sort_keys=False), Loader=ConfigLoader)


def read_config(path):
    return yaml.load(Path(path).read_text(), Loader=ConfigLoader)


def verify_model_state(payload, agent):
    """Validate exact native 61D/77D hill MLPs, including both normalizers."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    obs = TensorDict({"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 77)}, batch_size=[1])
    # Do not perturb the training seed merely by inspecting a source checkpoint.
    with torch.random.fork_rng(devices=[]):
        for name, output in (("actor", 14), ("critic", 1)):
            cfg = agent[name]
            model = MLPModel(obs, {name: [name]}, name, output, hidden_dims=cfg["hidden_dims"],
                             activation=cfg["activation"], obs_normalization=cfg["obs_normalization"],
                             distribution_cfg=deepcopy(cfg["distribution_cfg"]))
            state = dict(payload[f"{name}_state_dict"])
            if name == "actor" and "std" in state:
                state["distribution.std_param"] = state.pop("std")
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise ValueError(f"Hill requires full compatible {name} state and normalizer") from error
            # Constant Arena commands legitimately have zero variance; the
            # installed normalizer divides by std + eps, not std alone.
            if (state["obs_normalizer._std"] < 0).any():
                raise ValueError("Invalid Hill normalizer standard deviation")


def reward_overrides(profile):
    if not isinstance(profile, str) or profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown Hill reward profile: {profile!r}")
    return dict(REWARD_PROFILES[profile])


def phase_settings(training, phase):
    if phase not in ("explore", "exploit"):
        raise ValueError("Unknown Hill learning phase")
    settings = dict(training["phases"][phase])
    for name in ("max_iterations", "num_envs", "save_interval"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"Invalid {phase}.{name}")
    for name in ("learning_rate", "entropy_coef", "clip_param", "exploration_std_scale", "max_wall_minutes"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {phase}.{name}")
    if settings["clip_param"] >= 1 or settings["exploration_std_scale"] > 1:
        raise ValueError("Unsafe clipping/noise control")
    if settings["save_interval"] > settings["max_iterations"]:
        raise ValueError("Checkpoint interval exceeds phase budget")
    return settings


def refinement_recipe(cfg, clock, profile):
    """Freeze each known bootstrap curriculum at the *source's* age, not its end."""
    overrides = reward_overrides(profile)
    if type(clock) is not int or clock <= 0:
        raise ValueError("Refinement requires a positive source environment clock")
    for name, term in cfg.curriculum.items():
        params = term.params
        for key in ("weight_stages", "standing_stages", "range_stages"):
            if key not in params:
                continue
            stages = params[key]
            # Installed microduck curricula use strict >, not >=, at boundaries.
            stage = deepcopy(max((s for s in stages if s["step"] < clock), key=lambda s: s["step"]))
            stage["step"] = 0
            params[key] = [stage]
            if key == "weight_stages":
                cfg.rewards[params["reward_name"]].weight = stage["weight"]
            elif key == "standing_stages":
                cfg.commands[params["command_name"]].rel_standing_envs = stage["rel_standing_envs"]
            elif name in ("com_range", "head_com_range"):
                cfg.events[params["event_name"]].params["ranges"] = (-stage["range"], stage["range"])
            elif name == "body_pose_range":
                cfg.commands[params["command_name"]].ranges = stage["ranges"]
            else:
                raise ValueError(f"Unsupported Hill curriculum: {name}")
    for name, weight in overrides.items():
        cfg.rewards[name].weight = weight
        for term in cfg.curriculum.values():
            if term.params.get("reward_name") == name:
                for stage in term.params.get("weight_stages", []):
                    stage["weight"] = weight
    return cfg


@dataclass
class HillPpoCfg(PpoWithSymmetryCfg):
    stage: str = "bootstrap"
    reward_profile: str = "A0"
    source_clock: int = 0
    curriculum_clock: int = 0
    exploration_std_scale: float = 1.0


class HillPPO(PPO):
    """Strict full hill warm start; fresh Adam/runner clock, retained env clock."""

    def __init__(self, *args, stage="bootstrap", reward_profile="A0", source_clock=0, curriculum_clock=0,
                 exploration_std_scale=1.0, **kwargs):
        reward_overrides(reward_profile)
        if stage not in ("bootstrap", "refine") or (stage == "bootstrap" and reward_profile != "A0"):
            raise ValueError("Hill bootstrap requires A0; unknown stages are refused")
        self.stage, self.reward_profile, self.source_clock = stage, reward_profile, source_clock
        self.curriculum_clock = curriculum_clock
        self.exploration_std_scale = self._noise(exploration_std_scale)
        self._restored = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _noise(value):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Invalid Hill exploration noise scale")
        return float(value)

    def save(self):
        return {**super().save(), "hill_reward_profile": self.reward_profile,
                "hill_exploration_std_scale": self.exploration_std_scale}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if self.stage == "bootstrap":
            raise ValueError("Bootstrap is from scratch; use explicit Hill refinement, not resume")
        required = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
        if not strict or (load_cfg is not None and load_cfg != required):
            raise ValueError("Hill refinement requires a strict full actor/critic warm start")
        self._restored = False
        if os.environ.get("MICRODUCK_WARM_START", "") not in ("", "0"):
            raise ValueError("MICRODUCK_WARM_START would restart Hill environment curricula")
        clock = (loaded_dict.get("infos") or {}).get("env_state", {}).get("common_step_counter")
        if clock != self.source_clock or type(clock) is not int or clock <= 0:
            raise ValueError("Hill source environment clock changed")
        for name in ("actor_state_dict", "critic_state_dict"):
            if not all(torch.isfinite(v).all() for v in loaded_dict[name].values()):
                raise ValueError("Nonfinite Hill state")
        result = super().load(loaded_dict, required, strict=True)
        factor = self._noise(self.exploration_std_scale) / self._noise(loaded_dict.get("hill_exploration_std_scale", 1.0))
        if factor != 1:
            with torch.no_grad():
                self.actor.distribution.std_param.mul_(factor)
        if self.optimizer.state:
            raise ValueError("Hill refinement requires fresh Adam")
        self._restored = True
        return result

    def train_mode(self):
        if self.stage == "refine" and not self._restored:
            raise RuntimeError("Hill refinement cannot train without its strict source restore")
        super().train_mode()


def validate_source(path, baseline, agent, *, require_result=True):
    """Pin an exact Hill source, independently validate its declared recipe first."""
    from .evidence import read_report, summarize

    if path is None:
        raise ValueError("Hill refinement requires an explicit hill source")
    path = Path(path)
    # Validate the literal filename before resolving/looking anything up.
    if not re.fullmatch(r"model_(0|[1-9][0-9]*)\.pt", path.name) or path.resolve().parent.parent != LOG_ROOT:
        raise ValueError("Source must be an exact compatible Hill model_N.pt in the Hill experiment")
    path = path.resolve()
    saved_env = read_config(path.parent / "params/env.yaml")
    saved_agent = read_config(path.parent / "params/agent.yaml")
    provenance = read_report(path.parent / "provenance.json")
    if provenance.get("task") != TASK or saved_agent.get("experiment_name") != "hill_climb_1m5":
        raise ValueError("Source task is not Hill Climb")
    algorithm = saved_agent["algorithm"]
    source_stage, profile = algorithm.get("stage", "bootstrap"), algorithm.get("reward_profile", "A0")
    reward_overrides(profile)
    if source_stage not in ("bootstrap", "refine") or (source_stage == "bootstrap" and profile != "A0"):
        raise ValueError("Invalid saved Hill source stage/profile")
    known_classes = {"PPO", "microduck_challenges.hill_climb_1m5.learning:HillPPO"}
    if algorithm["class_name"] not in known_classes:
        raise ValueError("Unrecognized Hill source algorithm")
    canonical = deepcopy(baseline)
    if source_stage == "refine":
        curriculum_clock = algorithm["curriculum_clock"]
        if type(algorithm["source_clock"]) is not int or algorithm["source_clock"] <= 0:
            raise ValueError("Invalid saved Hill source clock")
        if type(curriculum_clock) is not int or not 0 < curriculum_clock <= algorithm["source_clock"]:
            raise ValueError("Invalid saved Hill curriculum clock")
        canonical = refinement_recipe(canonical, curriculum_clock, profile)
    expected = config_data(canonical)
    for group in ("actor", "critic"):
        if list(saved_env["observations"][group]["terms"]) != list(expected["observations"][group]["terms"]):
            raise ValueError(f"Incompatible ordered {group} observations")
    if list(saved_env["actions"]) != list(expected["actions"]):
        raise ValueError("Incompatible action order")
    # Permit only runtime seed/count changes; physics, terrain, curricula and rewards must match.
    comparable = deepcopy(saved_env)
    terrain = comparable["scene"]["terrain"]
    expected_terrain = expected["scene"]["terrain"]
    if terrain["num_envs"] not in (expected_terrain["num_envs"], comparable["scene"]["num_envs"]):
        raise ValueError("Incompatible runtime terrain world count")
    terrain["num_envs"] = expected_terrain["num_envs"]
    generator, canonical_generator = terrain["terrain_generator"], expected_terrain["terrain_generator"]
    for name, sub in generator["sub_terrains"].items():
        canonical_sub = canonical_generator["sub_terrains"].get(name)
        if canonical_sub is None or sub["size"] not in (canonical_sub["size"], canonical_generator["size"]):
            raise ValueError("Incompatible generated terrain sub-patch size")
        # TerrainGenerator fills each sub-config size from its enclosing tile.
        sub["size"] = canonical_sub["size"]
    comparable["scene"]["num_envs"] = expected["scene"]["num_envs"]
    comparable["seed"] = expected["seed"]
    if comparable != expected:
        raise ValueError("Saved Hill environment differs from independently reconstructed source recipe")
    wanted_agent = config_data(agent)
    for name in ("actor", "critic", "obs_groups", "clip_actions", "num_steps_per_env"):
        if saved_agent[name] != wanted_agent[name]:
            raise ValueError(f"Incompatible Hill {name}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("hill_reward_profile", "A0") != profile:
        raise ValueError("Checkpoint and saved reward profile disagree")
    clock = (payload.get("infos") or {}).get("env_state", {}).get("common_step_counter")
    if type(clock) is not int or clock <= 0:
        raise ValueError("Missing Hill source environment clock")
    if type(payload.get("iter")) is not int or payload["iter"] != int(path.stem.removeprefix("model_")):
        raise ValueError("Checkpoint iteration and filename disagree")
    initial_clock = algorithm.get("source_clock", 0) if source_stage == "refine" else 0
    if clock != initial_clock + (payload["iter"] + 1) * saved_agent["num_steps_per_env"]:
        raise ValueError("Hill source environment clock and runner iteration disagree")
    if source_stage == "refine" and clock <= algorithm["source_clock"]:
        raise ValueError("Refinement source age did not advance")
    if HillPPO._noise(payload.get("hill_exploration_std_scale", 1.)) != HillPPO._noise(algorithm.get("exploration_std_scale", 1.)):
        raise ValueError("Checkpoint and agent exploration scales disagree")
    for name in ("actor_state_dict", "critic_state_dict"):
        if not payload.get(name) or not all(torch.isfinite(t).all() for t in payload[name].values()):
            raise ValueError("Missing/nonfinite full Hill state")
    verify_model_state(payload, saved_agent)
    result_path = path.parent / "workshop" / f"{path.stem}.json"
    if require_result:
        result = read_report(result_path)
        rows = [{**row, "command_vx": result["command_vx"]} for row in result["per_seed"]]
        score = summarize(rows, range(1, 6), result["command_vx"])
        if (result.get("error") or result.get("checkpoint") != payload["iter"]
                or result.get("seeds_total") != 5 or result.get("seeds_finished") != 5
                or score["finishes"] != 5 or not math.isclose(score["mean_s"], result["score"], abs_tol=1e-9)):
            raise ValueError("Hill refinement requires a consistent complete 5/5 official result")
    return {"path": path, "clock": clock, "profile": profile, "source_stage": source_stage,
            "curriculum_clock": algorithm["curriculum_clock"] if source_stage == "refine" else clock,
            "source_env": saved_env, "source_agent": saved_agent, "result_path": result_path}
