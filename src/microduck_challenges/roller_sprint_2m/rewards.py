"""Roller-specific warm-start validation and conservative propulsion shaping."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import tomllib
from typing import Any

import torch
import yaml

from mjlab.managers import EventTermCfg, RewardTermCfg
from rsl_rl.algorithms import PPO


def course_progress(env, source_step, hold_steps, ramp_steps, speed_cap=1.5, command_name="twist"):
    """Bounded world-+x velocity during push, on a separate refinement clock.

    This is dense signed velocity guidance, not an exact time-to-finish objective.
    No position occupancy, wheel-spin, forced lean or flight bonus is introduced.
    """
    if (not math.isfinite(speed_cap) or speed_cap <= 0 or ramp_steps <= 0
            or hold_steps < 0 or source_step < 0):
        raise ValueError("positive speed/ramp scales and nonnegative clock settings are required")
    fraction = min(1.0, max(0.0, (env.common_step_counter - source_step - hold_steps) / ramp_steps))
    data = env.scene["robot"].data
    velocity = data.root_link_lin_vel_w
    gravity = data.projected_gravity_b
    command = env.command_manager.get_command(command_name)[:, 0]
    valid = (torch.isfinite(velocity).all(dim=-1) & torch.isfinite(gravity).all(dim=-1)
             & torch.isfinite(command) & (command > 0.0) & (gravity[:, 2] <= -0.5))
    speed = torch.nan_to_num(velocity[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    env.extras["log"]["Curriculum/roller_progress_fraction"] = torch.tensor(fraction, device=env.device)
    return torch.where(valid, (speed / speed_cap).clamp(-1.0, 1.0) * fraction, 0.0)


def training_settings(stage=None):
    document = tomllib.loads(Path(__file__).with_name("challenge.toml").read_text())
    settings = {
        "stage": "baseline",
        "source_run": "2026-10-23_03-57-00_run-1023-0356",
        "source_checkpoint": 2750,
        "source_sha256": "aa08986e4f1d5afcd48a733bb7c6e1a272f85b8e26c216164932596859298a8a",
        "hold_iterations": 25, "ramp_iterations": 100,
        "progress_weight": 1.0, "speed_cap": 1.5,
    }
    extra = document.get("training", {})
    if extra.keys() - settings.keys():
        raise ValueError(f"Unknown Roller training settings: {extra.keys() - settings.keys()}")
    settings.update(extra)
    if stage is not None:
        settings["stage"] = stage
    if settings["stage"] not in ("baseline", "warm_start"):
        raise ValueError("Roller stage must be baseline or warm_start")
    settings.update(document["recipe"])
    for key in ("source_checkpoint", "hold_iterations", "seed"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("ramp_iterations", "max_iterations", "num_envs"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("progress_weight", "speed_cap"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if settings["speed_cap"] == 0:
        raise ValueError("speed_cap must be positive")
    run = settings["source_run"]
    if not isinstance(run, str) or not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("source_run must be one exact directory name")
    if not isinstance(settings["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", settings["source_sha256"]):
        raise ValueError("source_sha256 must be exactly 64 lowercase hexadecimal characters")
    return settings


class _ConfigLoader(yaml.SafeLoader):
    """Read tagged configuration structure without constructing Python objects."""


def _config_tag(loader, suffix, node):
    del suffix
    if node.tag.startswith("tag:yaml.org,2002:python/name:"):
        return node.tag.removeprefix("tag:yaml.org,2002:python/name:")
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _config_tag)


def _config_data(config):
    return yaml.load(yaml.dump(asdict(config), sort_keys=False), Loader=_ConfigLoader)


def model_signature(payload):
    digest = hashlib.sha256()
    for group in ("actor_state_dict", "critic_state_dict"):
        state = payload[group]
        for name, tensor in sorted(state.items()):
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid model tensor: {group}.{name}")
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{group}.{name}:{value.dtype}:{tuple(value.shape)}\0".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_source(expected_env, expected_agent, settings, *, log_root=None):
    """Validate the exact pinned Roller source; never select a legged or latest run."""
    root = (Path(log_root) if log_root is not None else
            Path(__file__).resolve().parents[3] / "logs/rsl_rl/roller_sprint_2m")
    run = root / settings["source_run"]
    number = settings["source_checkpoint"]
    path = run / f"model_{number}.pt"
    source_env = yaml.load((run / "params/env.yaml").read_text(), Loader=_ConfigLoader)
    source_agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=_ConfigLoader)
    provenance = json.loads((run / "provenance.json").read_text())
    if provenance.get("task") != "Mjlab-RollerSprint2m-MicroDuck" or source_agent["experiment_name"] != "roller_sprint_2m":
        raise ValueError("Warm start requires a Roller task checkpoint")
    result = json.loads((run / f"workshop/model_{number}.json").read_text())
    rows = result.get("per_seed", [])
    if (result.get("error") or result.get("checkpoint") != number
            or result.get("seeds_finished") != 5 or result.get("seeds_total") != 5
            or len(rows) != 5 or any(type(row.get("seed")) is not int for row in rows)
            or {row["seed"] for row in rows} != {1, 2, 3, 4, 5}
            or any(row.get("status") != "finish" for row in rows)):
        raise ValueError("Roller source requires complete five-seed finish evidence")
    times = [row.get("time_s") for row in rows]
    score, command = result.get("score"), result.get("command_vx")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 40
           for value in [score, *times]):
        raise ValueError("Invalid Roller finish times")
    if not math.isclose(score, sum(times) / 5, abs_tol=1e-6, rel_tol=1e-8):
        raise ValueError("Roller score disagrees with its five seed times")
    if type(command) not in (int, float) or not math.isfinite(command) or not 0.3 <= command <= 2.0:
        raise ValueError("Invalid Roller evaluation command")
    current = _config_data(expected_env)
    for group in ("actor", "critic"):
        if list(source_env["observations"][group]["terms"]) != list(current["observations"][group]["terms"]):
            raise ValueError(f"Incompatible Roller {group} observation order")
    if list(source_env["actions"]) != list(current["actions"]):
        raise ValueError("Incompatible Roller action order")
    for key in ("observations", "actions", "sim", "decimation", "events", "commands", "terminations",
                "episode_length_s", "is_finite_horizon", "scale_rewards_by_dt", "curriculum"):
        if source_env[key] != current[key]:
            raise ValueError(f"Incompatible Roller source {key}")
    scenes = [deepcopy(source_env["scene"]), deepcopy(current["scene"])]
    for scene in scenes:
        scene.pop("num_envs", None)
        if scene.get("terrain") is not None:
            scene["terrain"].pop("num_envs", None)
    if scenes[0] != scenes[1]:
        raise ValueError("Incompatible Roller source scene/robot")
    if source_env["rewards"].keys() != current["rewards"].keys():
        raise ValueError("Incompatible Roller source reward set")
    for name, term in current["rewards"].items():
        saved = source_env["rewards"][name]
        if term["func"] != saved["func"] or term["params"] != saved["params"]:
            raise ValueError(f"Incompatible Roller reward semantics: {name}")
        if type(saved["weight"]) not in (int, float) or not math.isfinite(saved["weight"]):
            raise ValueError(f"Invalid source reward weight: {name}")
        if name not in ("wheel_speed", "upright") and saved["weight"] != term["weight"]:
            raise ValueError(f"Unexpected Roller source reward weight: {name}")
    if expected_agent is not None:
        agent = _config_data(expected_agent)
        for key in ("actor", "critic", "obs_groups", "clip_actions", "num_steps_per_env"):
            if source_agent[key] != agent[key]:
                raise ValueError(f"Incompatible Roller source agent {key}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Missing/empty pinned Roller checkpoint; refusing a cold start")
    if hashlib.sha256(path.read_bytes()).hexdigest() != settings["source_sha256"]:
        raise ValueError("Pinned Roller checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    clock = (payload.get("infos") or {}).get("env_state", {}).get("common_step_counter")
    if payload.get("iter") != number or type(clock) is not int or clock <= 0:
        raise ValueError("Roller checkpoint lacks the correct iteration/environment age")
    actor, critic = payload["actor_state_dict"], payload["critic_state_dict"]
    if (actor["mlp.0.weight"].shape[1] != 61 or actor["mlp.6.weight"].shape[0] != 14
            or critic["mlp.0.weight"].shape[1] != 78):
        raise ValueError("Expected the Roller 61D actor / 78D critic / 14D action layout")
    return {"path": path.resolve(), "run": run.resolve(), "source_step": clock,
            "signature": model_signature(payload), "sha256": settings["source_sha256"],
            "source_env": source_env, "source_agent": source_agent, "score": score}


def initialize_warm_start(env, env_ids, contract, inference_only=False):
    del env_ids
    env._roller_warm_start = contract
    env._roller_inference_only = inference_only


class RollerWarmStartPPO(PPO):
    """Restore the pinned Roller actor/critic, but start a fresh Adam/runner stage."""

    _roller_env: Any
    _roller_contract: dict[str, Any]
    _roller_inference_only: bool = False
    _roller_warm_started: bool = False

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        if not isinstance(algorithm, RollerWarmStartPPO):
            raise TypeError("Roller warm start requires RollerWarmStartPPO")
        algorithm._roller_env = getattr(env, "unwrapped", env)
        algorithm._roller_inference_only = getattr(algorithm._roller_env, "_roller_inference_only", False)
        algorithm._roller_contract = getattr(algorithm._roller_env, "_roller_warm_start", {})
        algorithm._roller_warm_started = False
        return algorithm

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if self._roller_inference_only:
            return super().load(loaded_dict, {"actor": True, "critic": False, "optimizer": False,
                                             "iteration": False, "rnd": False}, strict=True)
        required = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
        if not strict or (load_cfg is not None and load_cfg != required):
            raise ValueError("Roller warm start requires strict actor/critic loading without optimizer restore")
        contract = self._roller_contract
        clock = (loaded_dict.get("infos") or {}).get("env_state", {}).get("common_step_counter")
        if (type(clock) is not int or clock != contract.get("source_step")
                or model_signature(loaded_dict) != contract.get("signature")):
            raise ValueError("Loaded checkpoint does not match the pinned Roller source signature/age")
        if hashlib.sha256(Path(contract["path"]).read_bytes()).hexdigest() != contract["sha256"]:
            raise ValueError("Pinned Roller source checkpoint changed after configuration")
        result = super().load(loaded_dict, required, strict=True)
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self._roller_warm_started = True
        print(f"[roller warm start] loaded {contract['path']}; fresh Adam, fixed LR={self.learning_rate}; "
              f"runner iteration restarts, saved environment age={clock} preserved")
        return result

    def train_mode(self):
        if self._roller_inference_only:
            raise RuntimeError("Roller play/export loading is inference-only")
        if not self._roller_warm_started:
            raise RuntimeError("Roller warm-start training requires the pinned checkpoint to be loaded")
        if self._roller_env.common_step_counter < self._roller_contract["source_step"]:
            raise RuntimeError("Roller environment clock was rewound; unset MICRODUCK_WARM_START for refinement")
        super().train_mode()


def configure_warm_env(cfg, settings, *, play=False):
    if play:
        cfg.events["roller_warm_start"] = EventTermCfg(
            func=initialize_warm_start, mode="startup", params={"contract": {}, "inference_only": True})
        return cfg
    selected = select_source(cfg, None, settings)
    clock = selected["source_step"]
    # Freeze existing schedules at their saved source age, including construction-time resets.
    for term in cfg.curriculum.values():
        if "weight_stages" in term.params:
            stages = term.params["weight_stages"]
            value = stages[0]["weight"]
            for entry in stages:
                if clock > entry["step"]:
                    value = entry["weight"]
            cfg.rewards[term.params["reward_name"]].weight = value
            term.params["weight_stages"] = [{"step": -1, "weight": value}]
        elif "range_stages" in term.params:
            stages = term.params["range_stages"]
            value = stages[0]["range"]
            for entry in stages:
                if clock > entry["step"]:
                    value = entry["range"]
            cfg.events[term.params["event_name"]].params["ranges"] = (-value, value)
            term.params["range_stages"] = [{"step": -1, "range": value}]
    rollout_steps = selected["source_agent"]["num_steps_per_env"]
    cfg.rewards["course_progress"] = RewardTermCfg(
        func=course_progress, weight=settings["progress_weight"],
        params={"source_step": clock, "hold_steps": settings["hold_iterations"] * rollout_steps,
                "ramp_steps": settings["ramp_iterations"] * rollout_steps, "speed_cap": settings["speed_cap"]},
    )
    contract = {key: selected[key] for key in ("source_step", "signature", "sha256")}
    contract["path"] = str(selected["path"])
    cfg.events["roller_warm_start"] = EventTermCfg(
        func=initialize_warm_start, mode="startup", params={"contract": contract})
    return cfg
