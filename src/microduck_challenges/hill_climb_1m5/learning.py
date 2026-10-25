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
REWARD_PROFILES = {"A0": {}, "A1": {"track_linear_velocity": 0.0}, "B1": {"air_time": 1.5},
                   "H1": {}, "S0": {}, "P1": {}}
STABILITY_PROFILES = frozenset(("S0", "P1"))

# User-authorized exception for ONE genuine artifact, never a relaxed global gate.
APPROVED_PARTIAL = {
    "source": "logs/rsl_rl/hill_climb_1m5/2026-10-24_23-10-03_run-1024-2309/model_1000.pt",
    "sha256": "799ed0b452a1a484b4f700362248779fe127a1ba821a2c545728cdf9c9e814b9",
    "env_sha256": "9d3c78e38075c64d0e00f48a1c7489e65bbda3b999fb0315e1dd3dcee0453b67",
    "agent_sha256": "cd9b6ef25b4155d86f1d7c35023bc2aba5164bdd31c12280b537f2a261910155",
    "provenance_sha256": "8bc46a919424626d502f1f17c9f55ae1a1508d45cbf10ebc1d613f88eee06670",
    "result_sha256": "bca6ac805265d2a91e13207cf58f6223305eaa5d1ee2f40c83ae7c0727a4ab03",
    "partial_exception": "model1000-3of5-exploration-only",
    "base_knobs": {"forward_reward": 3.95619419642857, "upright_reward": 2.,
                   "step_reward": 6., "max_slope_deg": 10.},
}


def partial_source_pins(config, path, phase):
    from .evidence import digest
    if config != APPROVED_PARTIAL:
        raise ValueError("Missing/tampered exploratory source exception pins")
    if phase != "explore":
        raise ValueError("The 3/5 source exception is exploration only; exploitation requires 5/5")
    path = Path(path)
    if path.resolve() != Path(APPROVED_PARTIAL["source"]).resolve():
        raise ValueError("Exploratory exception is only for the pinned model_1000")
    artifacts = {path: config["sha256"], path.parent / "params/env.yaml": config["env_sha256"],
                 path.parent / "params/agent.yaml": config["agent_sha256"],
                 path.parent / "provenance.json": config["provenance_sha256"],
                 path.parent / "workshop/model_1000.json": config["result_sha256"]}
    if any(digest(p) != value for p, value in artifacts.items()):
        raise ValueError("Exploratory source artifact pin changed")
    return {str(p.resolve()): value for p, value in artifacts.items()}


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


def validate_normalized_state(state):
    """Shared finite-state contract; constant commands may have zero variance."""
    if not state or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite Hill model state")
    std = state.get("obs_normalizer._std")
    if std is not None and (std < 0).any():
        raise ValueError("Invalid Hill normalizer standard deviation")


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
            validate_normalized_state(state)
            if name == "actor" and "std" in state:
                state["distribution.std_param"] = state.pop("std")
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise ValueError(f"Hill requires full compatible {name} state and normalizer") from error



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


def refinement_recipe(cfg, clock, profile, *, reward_origin=None):
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
    if profile == "H1":
        from .rewards import configure_h1
        cfg = configure_h1(cfg, clock if reward_origin is None else reward_origin)
    elif profile in STABILITY_PROFILES:
        from .rewards import configure_stability
        cfg = configure_stability(cfg, profile)
    return cfg


@dataclass
class HillPpoCfg(PpoWithSymmetryCfg):
    stage: str = "bootstrap"
    reward_profile: str = "A0"
    source_clock: int = 0
    curriculum_clock: int = 0
    exploration_std_scale: float = 1.0
    campaign_journal: str | None = None
    campaign_plan_digest: str | None = None
    learning_phase: str = "explore"
    source_identity: dict | None = None
    base_knobs: dict | None = None
    reward_origin: int = 0
    reward_parameters: dict | None = None
    stability_gate: str | None = None
    stability_gate_sha256: str | None = None


CONTRACT_FIELDS = ("learning_phase", "source_identity", "base_knobs", "reward_origin", "reward_parameters")


def training_contract(algorithm):
    # Historical H1 markers predate Gate S. Do not add keys to their contract.
    fields = CONTRACT_FIELDS
    profile = algorithm.get("reward_profile") if isinstance(algorithm, dict) else getattr(algorithm, "reward_profile", None)
    if profile in STABILITY_PROFILES:
        fields += ("stability_gate", "stability_gate_sha256")
    if isinstance(algorithm, dict):
        return {k: algorithm.get(k) for k in fields}
    return {k: getattr(algorithm, k, None) for k in fields}


def stability_parameters(cfg):
    """Persist EVERY canonical reward's function, weight and parameters + guards."""
    data = config_data(cfg)
    return {"rewards": data["rewards"], "terminations": data["terminations"]}


def stability_gate_binding(selected, gate_path, gate_sha256=None):
    """Require raw-evidence Gate S for THIS exact normal-5/5 S0/P1 source.

    Configuration creation may pin a newly verified report digest. Restoration
    must pass the already saved digest, never repair it from ambient variables.
    The verifier is lazy so ordinary S0 imports and actor-only inference do not
    depend on overnight supervision or evaluation code.
    """
    from .evidence import digest
    if selected["profile"] not in STABILITY_PROFILES or selected["partial_exception"] is not None:
        raise ValueError("P1 requires a normal official 5/5 S0/P1 source, never a partial exception")
    if not isinstance(gate_path, (str, Path)) or not str(gate_path).strip():
        raise ValueError("P1 requires a verified stability gate for the exact source")
    if gate_sha256 is not None and (not isinstance(gate_sha256, str)
                                    or not re.fullmatch(r"[0-9a-f]{64}", gate_sha256)):
        raise ValueError("Invalid stability gate sha256 digest")
    path = Path(gate_path).resolve()
    try:
        actual = digest(path)
    except OSError as error:
        raise ValueError("Missing stability gate report") from error
    if gate_sha256 is not None and actual != gate_sha256:
        raise ValueError("Stability gate digest changed")
    from .overnight import require_stability_gate
    source = str(selected["path"])
    report = require_stability_gate(path, source, command_vx=.75)
    if not isinstance(report, dict) or report.get("candidate_path") != source:
        raise ValueError("Stability gate candidate must equal the exact source")
    if digest(path) != actual:
        raise ValueError("Stability gate report changed during verification")
    return str(path), actual


def validate_stability_contract(algorithm):
    """Reconstruct S0/P1 independently from their pinned ancestor, never metadata alone."""
    from .env import make_env_cfg, selected_training_source
    profile = algorithm["reward_profile"]
    required = ("source_identity", "base_knobs", "source_clock", "curriculum_clock",
                "reward_origin", "reward_parameters", "learning_phase", "stability_gate", "stability_gate_sha256")
    if (profile not in STABILITY_PROFILES or any(k not in algorithm for k in required)
            or not isinstance(algorithm["source_identity"], dict)
            or algorithm["learning_phase"] not in ("explore", "exploit")):
        raise ValueError(f"Invalid {profile} stability source/parameters contract")
    identity = algorithm["source_identity"]
    verify_source_identity(identity)
    selected = selected_training_source(identity["path"], phase=algorithm["learning_phase"])
    if (identity != selected["source_identity"] or algorithm["base_knobs"] != selected["base_knobs"]
            or any(type(algorithm[k]) is not int or algorithm[k] != selected[k] for k in
                   ("curriculum_clock", "reward_origin"))
            or type(algorithm["source_clock"]) is not int or algorithm["source_clock"] != selected["clock"]):
        raise ValueError(f"Invalid {profile} stability source recipe/clocks")
    canonical = refinement_recipe(make_env_cfg(stage="bootstrap", knobs=selected["base_knobs"]),
                                   selected["curriculum_clock"], profile)
    if algorithm["reward_parameters"] != stability_parameters(canonical):
        raise ValueError(f"Invalid {profile} stability reward parameters")
    if profile == "S0" and any(algorithm[k] is not None for k in ("stability_gate", "stability_gate_sha256")):
        raise ValueError("S0 stability does not carry an acceleration gate")
    if profile == "P1":
        if not isinstance(algorithm["stability_gate_sha256"], str):
            raise ValueError("P1 requires its saved stability gate digest")
        binding = stability_gate_binding(selected, algorithm["stability_gate"], algorithm["stability_gate_sha256"])
        if binding != (algorithm["stability_gate"], algorithm["stability_gate_sha256"]):
            raise ValueError("Invalid saved P1 stability gate binding")
    return canonical


def verify_source_identity(identity):
    from .evidence import digest, read_report, summarize
    if (not isinstance(identity, dict) or set(identity) != {"path", "sha256", "artifacts", "partial_exception"}
            or not identity["artifacts"] or identity["artifacts"].get(identity["path"]) != identity["sha256"]
            or any(digest(p) != value for p, value in identity["artifacts"].items())):
        raise ValueError("Missing/tampered Hill source identity pins")
    if (Path(identity["path"]).resolve() == Path(APPROVED_PARTIAL["source"]).resolve()
            and identity["partial_exception"] != APPROVED_PARTIAL["partial_exception"]):
        raise ValueError("Missing approved partial-source exception identity")
    if identity["partial_exception"] is not None:
        expected = partial_source_pins(APPROVED_PARTIAL, identity["path"], "explore")
        if (identity["partial_exception"] != APPROVED_PARTIAL["partial_exception"]
                or identity["artifacts"] != expected):
            raise ValueError("Invalid partial-source exception identity")
    else:
        path = Path(identity["path"])
        required = {str(p) for p in (path, path.parent / "params/env.yaml", path.parent / "params/agent.yaml",
                                    path.parent / "provenance.json", path.parent / "workshop" / f"{path.stem}.json")}
        if set(identity["artifacts"]) != required:
            raise ValueError("Incomplete Hill source identity artifacts")
        result = read_report(path.parent / "workshop" / f"{path.stem}.json")
        summary = summarize([{**r, "command_vx": result["command_vx"]} for r in result["per_seed"]],
                            range(1, 6), result["command_vx"])
        if (result.get("error") or result.get("seeds_total") != 5 or result.get("seeds_finished") != 5
                or summary["finishes"] != 5 or not math.isclose(summary["mean_s"], result["score"], abs_tol=1e-9)):
            raise ValueError("Non-exception training source requires consistent 5/5")


def mark_actor_only_inference(env, env_ids=None):
    """Explicit play/export marker, never inferred from load flags or shapes."""
    del env_ids
    env._hill_inference_only = True


class HillPPO(PPO):
    """Strict full hill warm start; fresh Adam/runner clock, retained env clock."""

    def __init__(self, *args, stage="bootstrap", reward_profile="A0", source_clock=0, curriculum_clock=0,
                 exploration_std_scale=1.0, campaign_journal=None, campaign_plan_digest=None,
                 learning_phase="explore", source_identity=None, base_knobs=None,
                 reward_origin=0, reward_parameters=None, stability_gate=None, stability_gate_sha256=None, **kwargs):
        reward_overrides(reward_profile)
        if stage not in ("bootstrap", "refine") or (stage == "bootstrap" and reward_profile != "A0"):
            raise ValueError("Hill bootstrap requires A0; unknown stages are refused")
        self.stage, self.reward_profile, self.source_clock = stage, reward_profile, source_clock
        self.curriculum_clock = curriculum_clock
        self.exploration_std_scale = self._noise(exploration_std_scale)
        self._restored = False
        self.campaign_journal, self.campaign_plan_digest = campaign_journal, campaign_plan_digest
        self.learning_phase, self.source_identity, self.base_knobs = learning_phase, source_identity, base_knobs
        self.reward_origin, self.reward_parameters = reward_origin, reward_parameters
        self.stability_gate, self.stability_gate_sha256 = stability_gate, stability_gate_sha256
        super().__init__(*args, **kwargs)

    @staticmethod
    def _noise(value):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Invalid Hill exploration noise scale")
        return float(value)

    def save(self):
        return {**super().save(), "hill_reward_profile": self.reward_profile,
                "hill_exploration_std_scale": self.exploration_std_scale,
                "hill_campaign_journal": getattr(self, "campaign_journal", None),
                "hill_campaign_plan_digest": getattr(self, "campaign_plan_digest", None),
                "hill_training_contract": training_contract(self)}

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        setattr(algorithm, "_hill_inference_only", bool(
            getattr(getattr(env, "unwrapped", env), "_hill_inference_only", False)))
        return algorithm

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if getattr(self, "_hill_inference_only", False):
            if not strict:
                raise ValueError("Hill inference requires a strict full actor restore")
            state = loaded_dict["actor_state_dict"]
            validate_normalized_state(state)
            # No refinement clocks, noise rescaling, critic or optimizer in play.
            return super().load(loaded_dict, {"actor": True, "critic": False,
                                "optimizer": False, "iteration": False, "rnd": False}, strict=True)
        if self.stage == "bootstrap":
            raise ValueError("Bootstrap is from scratch; use explicit Hill refinement, not resume")
        required = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
        if not strict or (load_cfg is not None and load_cfg != required):
            raise ValueError("Hill refinement requires a strict full actor/critic warm start")
        self._restored = False
        if os.environ.get("MICRODUCK_WARM_START", "") not in ("", "0"):
            raise ValueError("MICRODUCK_WARM_START would restart Hill environment curricula")
        identity = getattr(self, "source_identity", None)
        if self.reward_profile in STABILITY_PROFILES:
            validate_stability_contract({**training_contract(self), "reward_profile": self.reward_profile,
                "source_clock": self.source_clock, "curriculum_clock": self.curriculum_clock})
        if self.reward_profile == "H1" and identity is None:
            raise ValueError("H1 training requires its exact source identity pins")
        if self.reward_profile == "H1":
            from .rewards import H1
            assert identity is not None
            if (self.reward_parameters != H1 or not self.base_knobs
                    or type(self.reward_origin) is not int or not 0 < self.reward_origin <= self.source_clock
                    or self.learning_phase not in ("explore", "exploit")):
                raise ValueError("Invalid H1 target training contract")
            if identity["partial_exception"] and self.base_knobs != APPROVED_PARTIAL["base_knobs"]:
                raise ValueError("H1 target differs from the approved source recipe")
        if identity is not None:
            verify_source_identity(identity)
            if identity["partial_exception"] and self.learning_phase != "explore":
                raise ValueError("The partial-source exception is exploration only")
            original = torch.load(identity["path"], map_location="cpu", weights_only=False)
            if self.reward_profile in STABILITY_PROFILES:
                markers = ("hill_reward_profile", "hill_exploration_std_scale", "iter",
                           "hill_campaign_journal", "hill_campaign_plan_digest")
                if any(loaded_dict.get(k) != original.get(k) for k in markers):
                    raise ValueError("Restored source markers differ from pinned Hill source")
            for name in ("actor_state_dict", "critic_state_dict"):
                if set(loaded_dict[name]) != set(original[name]) or any(
                        not torch.equal(loaded_dict[name][k].cpu(), v) for k, v in original[name].items()):
                    raise ValueError("Restored tensors differ from pinned Hill source")
            if loaded_dict.get("hill_training_contract") != original.get("hill_training_contract"):
                raise ValueError("Restored source training contract differs from pinned source")
        clock = (loaded_dict.get("infos") or {}).get("env_state", {}).get("common_step_counter")
        if clock != self.source_clock or type(clock) is not int or clock <= 0:
            raise ValueError("Hill source environment clock changed")
        for name in ("actor_state_dict", "critic_state_dict"):
            validate_normalized_state(loaded_dict[name])
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
        if getattr(self, "_hill_inference_only", False):
            raise RuntimeError("The actor-only Hill play/export model is inference-only.")
        if self.stage == "refine" and not self._restored:
            raise RuntimeError("Hill refinement cannot train without its strict source restore")
        super().train_mode()


def validate_source(path, baseline, agent, *, require_result=True, exploratory_exception=None, phase="explore"):
    """Pin an exact Hill source, independently validate its declared recipe first."""
    from .evidence import campaign_evidence, read_report, summarize

    if path is None:
        raise ValueError("Hill refinement requires an explicit hill source")
    path = Path(path)
    # Validate the literal filename before resolving/looking anything up.
    if not re.fullmatch(r"model_(0|[1-9][0-9]*)\.pt", path.name) or path.resolve().parent.parent != LOG_ROOT:
        raise ValueError("Source must be an exact compatible Hill model_N.pt in the Hill experiment")
    path = path.resolve()
    if exploratory_exception is not None:
        partial_source_pins(exploratory_exception, path, phase)
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
        if profile in STABILITY_PROFILES:
            validate_stability_contract(algorithm)
        if algorithm.get("source_identity") is not None:
            from .env import make_env_cfg
            verify_source_identity(algorithm["source_identity"])
            if not algorithm.get("base_knobs"):
                raise ValueError("Missing Hill source construction recipe")
            canonical = make_env_cfg(stage="bootstrap", knobs=algorithm["base_knobs"])
        curriculum_clock = algorithm["curriculum_clock"]
        if type(algorithm["source_clock"]) is not int or algorithm["source_clock"] <= 0:
            raise ValueError("Invalid saved Hill source clock")
        if type(curriculum_clock) is not int or not 0 < curriculum_clock <= algorithm["source_clock"]:
            raise ValueError("Invalid saved Hill curriculum clock")
        if profile == "H1":
            from .env import make_env_cfg
            from .rewards import H1
            if (algorithm.get("reward_parameters") != H1 or not algorithm.get("base_knobs")
                    or type(algorithm.get("reward_origin")) is not int
                    or not 0 < algorithm["reward_origin"] <= algorithm["source_clock"]):
                raise ValueError("Invalid saved H1 reward parameters/origin/recipe")
            verify_source_identity(algorithm.get("source_identity"))
            canonical = make_env_cfg(stage="bootstrap", knobs=algorithm["base_knobs"])
        canonical = refinement_recipe(canonical, curriculum_clock, profile,
                                      reward_origin=algorithm.get("reward_origin") if profile == "H1" else None)
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
    if (profile in STABILITY_PROFILES or profile == "H1" or algorithm.get("source_identity") is not None) and payload.get("hill_training_contract") != training_contract(algorithm):
        raise ValueError("Checkpoint and saved Hill training contract disagree")
    campaign_evidence(path, saved_agent, payload)
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
        finishes = 3 if exploratory_exception is not None else 5
        if (result.get("error") or result.get("checkpoint") != payload["iter"]
                or result.get("seeds_total") != 5 or result.get("seeds_finished") != finishes
                or score["finishes"] != finishes or not math.isclose(score["mean_s"], result["score"], abs_tol=1e-9)
                or (exploratory_exception is not None and (result["command_vx"] != .75
                    or not math.isclose(score["mean_s"], 25.872, abs_tol=1e-9)))):
            raise ValueError("Hill refinement requires a consistent complete 5/5 official result")
    return {"path": path, "clock": clock, "profile": profile, "source_stage": source_stage,
            "curriculum_clock": algorithm["curriculum_clock"] if source_stage == "refine" else clock,
            "source_env": saved_env, "source_agent": saved_agent, "result_path": result_path,
            "partial_exception": exploratory_exception["partial_exception"] if exploratory_exception else None}
