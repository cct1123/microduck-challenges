"""Bounded Hill training and frozen-command assessment; never auto-publish."""

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, cast

from . import env as hill_env
from .learning import (TASK, LOG_ROOT, HillPpoCfg, REWARD_PROFILES, config_data, read_config,
                       phase_settings, refinement_recipe, reward_overrides, validate_source)
from .evidence import (digest, promotion_decision, read_report, require_promotion,
                       runtime_signature, saved_checkpoints, summarize, validate_command)


def smoke(stage="bootstrap", phase="explore", source=None, output=None, reward_profile="A0", seed=None):
    """Exactly two CPU envs/two updates, live-clock checks and finite ONNX parity.

    A refinement smoke still needs a genuine compatible 5/5 hill source; an
    untrained smoke checkpoint is not forged into race evidence.
    """
    import torch
    import mjlab.scripts.train as train
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if type(seed) is not int or seed < 0:
        raise ValueError("Smoke requires an explicit nonnegative seed")
    if output is None:
        raise ValueError("Smoke requires a new output directory")
    if stage not in ("bootstrap", "refine") or phase not in ("explore", "exploit"):
        raise ValueError("Unknown Hill smoke stage/phase")
    if stage == "bootstrap" and (source or phase != "explore" or reward_profile != "A0"):
        raise ValueError("Bootstrap smoke is from scratch, explore/A0 only")
    selected = selected_source(source) if stage == "refine" else None
    source_clock = selected["clock"] if selected else 0
    source = selected["path"] if selected else None
    before = digest(source) if source else None
    saved = torch.load(source, map_location="cpu", weights_only=False) if source else None
    output = Path(output).resolve()
    if source and (output == source.parent or source.parent in output.parents):
        raise ValueError("Smoke output must be separate from the source run")
    output.mkdir(parents=True, exist_ok=False)
    log_dir = output / "cpu"
    cfg = replace(train.TrainConfig.from_task(TASK), gpu_ids=None,
                  env=hill_env.make_env_cfg(stage="bootstrap"), agent=deepcopy(hill_env.BASE_RL_CFG))
    if selected:
        controls = phase_settings(hill_env.TRAINING, phase)
        cfg = replace(cfg, env=refinement_recipe(cfg.env, selected["curriculum_clock"], reward_profile))
        cfg.agent.algorithm = HillPpoCfg(**{**asdict(cfg.agent.algorithm),
            "stage": "refine", "reward_profile": reward_profile, "source_clock": source_clock,
            "curriculum_clock": selected["curriculum_clock"],
            "schedule": "fixed", **{k: controls[k] for k in
                ("learning_rate", "entropy_coef", "clip_param", "exploration_std_scale")}})
        (output / source.parent.name).symlink_to(source.parent, target_is_directory=True)
        cfg.agent.resume = True
        cfg.agent.load_run = "^" + re.escape(source.parent.name) + "$"
        cfg.agent.load_checkpoint = "^" + re.escape(source.name) + "$"
    cfg.env.scene.num_envs = 2
    cfg.agent.max_iterations, cfg.agent.save_interval = 2, 1
    cfg.agent.seed, cfg.agent.logger, cfg.agent.upload_model = seed, "tensorboard", False
    expected_weights = {n: t.weight for n, t in cfg.env.rewards.items()}
    observations, live_checks, restored_tensors = {}, 0, 0
    original_init, original_load = MicroduckOnPolicyRunner.__init__, MicroduckOnPolicyRunner.load

    def check_weights(runner):
        nonlocal live_checks
        manager = runner.env.unwrapped.reward_manager
        if {n: manager.get_term_cfg(n).weight for n in expected_weights} != expected_weights:
            raise ValueError("Live Hill reward weights changed from requested profile")
        live_checks += 1

    def checked_init(runner, *args, **kwargs):
        original_init(runner, *args, **kwargs)
        obs = runner.env.get_observations()
        observations.update({k: list(v.shape) for k, v in obs.items()})
        if str(runner.env.unwrapped.device) != "cpu" or observations["actor"] != [2, 61]:
            raise ValueError("Smoke must resolve to two 61D CPU environments")
        if source is None:
            check_weights(runner)
        step = runner.env.step
        def checked_step(*a, **kw):
            result = step(*a, **kw)
            check_weights(runner)
            return result
        runner.env.step = checked_step

    def checked_load(runner, path, *args, **kwargs):
        nonlocal restored_tensors
        if Path(path).resolve() != source:
            raise ValueError("Smoke restored the wrong Hill source")
        result = original_load(runner, path, *args, **{**kwargs, "map_location": "cpu"})
        factor = cfg.agent.algorithm.exploration_std_scale / saved.get("hill_exploration_std_scale", 1.)
        for name in ("actor", "critic"):
            for key, value in getattr(runner.alg, name).state_dict().items():
                expected = saved[f"{name}_state_dict"][key]
                if name == "actor" and key == "distribution.std_param":
                    expected = expected * factor
                torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
                restored_tensors += 1
        if (runner.current_learning_iteration != 0 or runner.alg.optimizer.state
                or runner.env.unwrapped.common_step_counter != source_clock):
            raise ValueError("Hill warm-start clocks/Adam are incorrect")
        check_weights(runner)
        return result

    previous_cuda, previous_warm = os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("MICRODUCK_WARM_START")
    previous_threads, previous_argv = torch.get_num_threads(), sys.argv
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("MICRODUCK_WARM_START", None)
    torch.set_num_threads(2)
    # Exercise the installed, recognized provenance hook with the actual config overrides.
    sys.argv = [train.__file__, TASK, "--gpu-ids", "None", "--env.scene.num-envs", "2",
                "--agent.max-iterations", "2", "--agent.save-interval", "1", "--agent.seed", str(seed),
                "--agent.logger", "tensorboard", "--agent.upload-model", "False"]
    MicroduckOnPolicyRunner.__init__, MicroduckOnPolicyRunner.load = checked_init, checked_load
    try:
        train.run_train(TASK, cfg, log_dir)
    finally:
        MicroduckOnPolicyRunner.__init__, MicroduckOnPolicyRunner.load = original_init, original_load
        sys.argv = previous_argv
        torch.set_num_threads(previous_threads)
        for key, previous in (("CUDA_VISIBLE_DEVICES", previous_cuda), ("MICRODUCK_WARM_START", previous_warm)):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    final = torch.load(log_dir / "model_1.pt", map_location="cpu", weights_only=False)
    if final["infos"]["env_state"]["common_step_counter"] != source_clock + 48 or final["iter"] != 1:
        raise ValueError("Smoke did not complete exactly two 24-step updates")
    if final.get("hill_reward_profile") != reward_profile or live_checks != 49:
        raise ValueError("Missing live reward/profile checks")
    provenance = read_report(log_dir / "provenance.json")
    if provenance.get("task") != TASK or provenance.get("seed") != seed:
        raise ValueError("Smoke provenance did not record the Hill task and seed")
    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
    scalars = [e.value for tag in acc.Tags()["scalars"] for e in acc.Scalars(tag)]
    if not scalars or not all(math.isfinite(v) for v in scalars) or (source and digest(source) != before):
        raise ValueError("Nonfinite smoke scalars or changed source")
    export = export_actor(log_dir / "model_1.pt", output / "verified.onnx")
    report = {"stage": stage, "phase": phase, "reward_profile": reward_profile,
              "device": "cpu", "seed": seed, "environments": 2, "updates": 2,
              "observations": observations, "source": str(source) if source else None,
              "source_clock": source_clock, "final_environment_clock": source_clock + 48,
              "runner_iteration": final["iter"], "restored_state_tensors": restored_tensors,
              "live_reward_checks": live_checks, "reward_weights": expected_weights,
              "finite_scalar_values": len(scalars), "export": export,
              "provenance": str(log_dir / "provenance.json"), "racing_performance_verified": False}
    write_json(output / "smoke.json", report)
    return report


def selected_source(source=None):
    return validate_source(source, hill_env.make_env_cfg(stage="bootstrap"), hill_env.BASE_RL_CFG)


def make_plan(stage="bootstrap", phase="explore", source=None, promotion_path=None,
              iterations=None, seed=None, reward_profile="A0"):
    overrides = reward_overrides(reward_profile)
    selected = None
    environment = {"MICRODUCK_HILL_STAGE": stage, "MICRODUCK_HILL_PHASE": phase,
                   "MICRODUCK_HILL_REWARD_PROFILE": reward_profile,
                   "MICRODUCK_SPRINT_STAGE": "baseline"}
    if stage == "bootstrap":
        if phase != "explore" or source or promotion_path or reward_profile != "A0":
            raise ValueError("Bootstrap is from scratch, explore/A0 only; no source or promotion")
        controls = {"max_iterations": hill_env.RECIPE["max_iterations"],
                    "num_envs": hill_env.RECIPE["num_envs"], "save_interval": 250,
                    "max_wall_minutes": hill_env.TRAINING["bootstrap_wall_minutes"],
                    "learning_rate": .001, "schedule": "adaptive", "entropy_coef": .01,
                    "clip_param": .2, "exploration_std_scale": 1.}
        target = hill_env.make_env_cfg(stage="bootstrap")
    elif stage == "refine":
        controls = phase_settings(hill_env.TRAINING, phase)
        if promotion_path and source is None:
            source = read_report(promotion_path)["candidate_path"]
        selected = selected_source(source)
        source = selected["path"]
        environment["MICRODUCK_HILL_SOURCE"] = str(source)
        if phase == "exploit":
            if not promotion_path:
                raise ValueError("Exploitation requires an approved promotion report")
            gate = {**hill_env.TRAINING["gate"], "command_vx": read_report(promotion_path)["gate"]["command_vx"]}
            require_promotion(promotion_path, source, gate)
            if reward_profile != selected["profile"]:
                raise ValueError("Exploitation must preserve the promoted reward profile")
            environment["MICRODUCK_HILL_PROMOTION"] = str(Path(promotion_path).resolve())
        elif promotion_path:
            raise ValueError("Promotion is only meaningful for exploitation")
        target = refinement_recipe(hill_env.make_env_cfg(stage="bootstrap"), selected["curriculum_clock"], reward_profile)
    else:
        raise ValueError("Unknown Hill stage")
    budget = controls["max_iterations"]
    iterations = budget if iterations is None else iterations
    if type(iterations) is not int or not 1 <= iterations <= budget:
        raise ValueError(f"Requested updates exceed the {budget}-update stage budget")
    seed = hill_env.RECIPE["seed"] if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    minutes = controls["max_wall_minutes"]
    if type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Wall-time budget must be finite and positive")
    name = f"{stage}-{phase}-{reward_profile}-s{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    command = [sys.executable, "-m", "mjlab.scripts.train", TASK,
               "--gpu-ids", "[0]", "--env.scene.num-envs", str(controls["num_envs"]),
               "--agent.max-iterations", str(iterations), "--agent.save-interval", str(controls["save_interval"]),
               "--agent.seed", str(seed), "--agent.logger", "tensorboard", "--agent.upload-model", "False",
               "--agent.run-name", name]
    artifacts = {} if selected is None else {str(p): digest(p) for p in
        (source, source.parent / "params/env.yaml", source.parent / "params/agent.yaml",
         source.parent / "provenance.json", selected["result_path"])}
    if promotion_path:
        artifacts[str(Path(promotion_path).resolve())] = digest(promotion_path)
    return {"stage": stage, "phase": phase, "reward_profile": reward_profile,
            "reward_overrides": overrides, "source": str(source) if source else None,
            "source_clock": selected["clock"] if selected else 0,
            "source_config_files": artifacts, "max_iterations": iterations, "seed": seed,
            "controls": controls, "max_wall_minutes": minutes,
            "target_reward_weights": {n: t.weight for n, t in target.rewards.items()},
            "source_files": {str(p): digest(p) for p in sorted(Path(__file__).parent.iterdir())
                             if p.suffix in (".py", ".toml")},
            "environment": environment, "command": command, "requires_promotion": phase == "exploit",
            "launching": False}



def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def export_actor(checkpoint, output):
    """Strictly export the saved actor, never the current critic or Workshop cache."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve previous exports: choose a new assessment directory")
    before = digest(checkpoint)
    agent = read_config(checkpoint.parent / "params/agent.yaml")
    if read_report(checkpoint.parent / "provenance.json").get("task") != TASK:
        raise ValueError("Export only a saved Hill actor")
    actor_cfg = agent["actor"]
    if actor_cfg != config_data(hill_env.BASE_RL_CFG)["actor"]:
        raise ValueError("Unsupported saved actor architecture/normalization configuration")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reward_profile = agent.get("algorithm", {}).get("reward_profile", "A0")
    reward_overrides(reward_profile)
    if payload.get("hill_reward_profile", "A0") != reward_profile:
        raise ValueError("Export reward profile disagrees with saved agent configuration")
    state = dict(payload["actor_state_dict"])
    for old, new in (("std", "distribution.std_param"), ("log_std", "distribution.log_std_param")):
        if old in state:
            state[new] = state.pop(old)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite saved actor state")
    actor = MLPModel(TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1]),
                     {"actor": ["actor"]}, "actor", 14,
                     hidden_dims=actor_cfg["hidden_dims"], activation=actor_cfg["activation"],
                     obs_normalization=actor_cfg["obs_normalization"],
                     distribution_cfg=deepcopy(actor_cfg["distribution_cfg"]))
    actor.load_state_dict(state, strict=True)
    wrapper = cast(Any, actor.eval().as_onnx(verbose=False).cpu().eval())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, wrapper.get_dummy_inputs(), str(output), export_params=True,
                      opset_version=18, input_names=wrapper.input_names,
                      output_names=wrapper.output_names, dynamic_axes={}, dynamo=False)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    if output.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Hill export exceeds the Arena 64 MiB limit")
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
        raise ValueError("Export violated the Arena 61/14 contract")
    for seed in range(8):
        obs = np.random.default_rng(seed).normal(0, 0.05, (1, 61)).astype(np.float32)
        obs[0, 5], obs[0, 48] = -1, 0.3
        actual = np.asarray(cast(Any, session.run(None, {"obs": obs})[0]))
        with torch.no_grad():
            expected = wrapper(torch.from_numpy(obs)).numpy()
        if not np.isfinite(actual).all():
            raise ValueError("Nonfinite exported action")
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
    if digest(checkpoint) != before:
        raise ValueError("Source checkpoint changed during export")
    return {"checkpoint_sha256": before, "onnx_sha256": digest(output), "parity_cases": 8,
            "reward_profile": reward_profile,
            "source_agent_sha256": digest(checkpoint.parent / "params/agent.yaml"),
            "input_shape": [1, 61], "output_shape": [1, 14]}


def inventory(run):
    """Require a complete saved run, not just whichever files happen to exist."""
    run = Path(run).resolve()
    actual = saved_checkpoints(run)
    official = sorted(int(p.stem.removeprefix("model_")) for p in (run / "workshop").glob("model_*.json"))
    if official != actual:
        raise ValueError("Wait for Workshop to judge every saved checkpoint before assessment")
    for number in actual:
        result = read_report(run / "workshop" / f"model_{number}.json")
        if result.get("error") or result.get("checkpoint") != number or result.get("seeds_total") != 5:
            raise ValueError("Workshop result is incomplete or failed; explicitly retry it first")
        rows = [{**row, "command_vx": result["command_vx"]} for row in result["per_seed"]]
        summary = summarize(rows, range(1, 6), result["command_vx"])
        if (summary["finishes"] != result["seeds_finished"]
                or not math.isclose(summary["mean_s"], result["score"], rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Workshop result totals disagree with its seed rows")
    return actual


def assess(run, incumbent, output, command_vx):
    """Select once on development seeds, then open exactly one held-out set."""
    from arena.course import compile_course
    from arena.manifest import load_event
    from arena.policy import load_policy
    from arena.runner import run_once

    run, output = Path(run).resolve(), Path(output).resolve()
    if run.parent != LOG_ROOT:
        raise ValueError("Assess only a saved Hill experiment")
    validate_command(command_vx)
    ids = inventory(run)
    incumbent = selected_source(incumbent)["path"]
    if output == run or run in output.parents or incumbent.parent == output or incumbent.parent in output.parents:
        raise ValueError("Assessment output must be separate from source runs")
    output.mkdir(parents=True, exist_ok=False)
    gate = {**hill_env.TRAINING["gate"], "command_vx": command_vx}
    manifest, model = load_event("hill-climb-1m5"), None
    model = compile_course(manifest)
    command = manifest.command_at(gate["command_vx"])
    artifacts = {}
    for number in ids:
        official = run / "workshop" / f"model_{number}.json"
        artifacts[str(official)] = digest(official)
    for folder in {run, incumbent.parent}:
        for name in ("params/env.yaml", "params/agent.yaml", "provenance.json"):
            path = folder / name
            artifacts[str(path)] = digest(path)
    for path in [*(run / f"model_{n}.pt" for n in ids), incumbent]:
        artifacts[str(path)] = digest(path)
    report = {"schema": 1, "status": "screening", "gate": gate, "checkpoints": ids,
              "incumbent_path": str(incumbent), "screen": {}, "exports": {},
              "artifact_hashes": artifacts, "runtime": runtime_signature()}

    def races(checkpoint, label, seeds, partition="screen"):
        validate_source(checkpoint, hill_env.make_env_cfg(stage="bootstrap"), hill_env.BASE_RL_CFG, require_result=False)
        export = output / f"{label}.onnx"
        if not export.exists():
            report["exports"][label] = export_actor(checkpoint, export)
            artifacts[str(export)] = digest(export)
        policy = load_policy(export)
        rows = []
        for seed in seeds:
            result = run_once(manifest, policy, seed, model=model, command=command)
            row = {"seed": seed, "status": result.status, "time_s": result.time_s,
                   "progress_m": result.progress_m, "command_vx": gate["command_vx"]}
            rows.append(row)
            with (output / f"{label}-{partition}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        rows_path = output / f"{label}-{partition}.jsonl"
        if [json.loads(line) for line in rows_path.read_text().splitlines()] != rows:
            raise ValueError("Persisted seed rows disagree with collected races")
        artifacts[str(rows_path)] = digest(rows_path)
        return rows

    for number in ids:
        report["screen"][str(number)] = races(run / f"model_{number}.pt", f"model_{number}", range(1, 6))
        write_json(output / "assessment.json", report)
    count = gate["neighbor_checkpoints"]
    stable = set()
    for i in range(len(ids) - count + 1):
        window = ids[i:i + count]
        if all(summarize(report["screen"][str(n)], range(1, 6), gate["command_vx"])["finishes"] == 5 for n in window):
            stable.update(window)
    if not stable:
        report.update(status="screen_rejected", decision={"approved": False,
                      "reasons": ["no three adjacent saved checkpoints finish all five seeds"]})
    else:
        candidate = min(stable, key=lambda n: (summarize(report["screen"][str(n)], range(1, 6),
                                                         gate["command_vx"])["mean_s"], n))
        report.update(candidate_checkpoint=candidate, candidate_path=str(run / f"model_{candidate}.pt"),
                      selection="fastest stable development mean; selected before opening held-outs")
        write_json(output / "assessment.json", report)
        first, size = gate["heldout_first_seed"], gate["heldout_seed_count"]
        report["candidate_heldout"] = races(run / f"model_{candidate}.pt", f"model_{candidate}", range(first, first + size), "heldout")
        report["incumbent_rows"] = (races(incumbent, "incumbent", range(1, 6))
                                    + races(incumbent, "incumbent", range(first, first + size), "heldout"))
        report["status"] = "complete"
        report["decision"] = promotion_decision(report, gate)
    if (inventory(run) != ids or any(digest(path) != value for path, value in artifacts.items())
            or report["runtime"] != runtime_signature()):
        raise ValueError("Assessment inputs changed; promotion is forbidden")
    report["screen_rows"] = sum(len(rows) for rows in report["screen"].values())
    if report["screen_rows"] != 5 * len(ids):
        raise ValueError("Incomplete checkpoint seed coverage")
    write_json(output / "assessment.json", report)
    return report


def active_trainers():
    result = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    markers = ("/.venv/bin/train Mjlab-", " -m mjlab.scripts.train Mjlab-",
               " -m mjlab_microduck.train_cli Mjlab-")
    return [int(line.split(None, 1)[0]) for line in result.splitlines()
            if any(marker in line for marker in markers)]


@contextmanager
def _termination_requests():
    """Defer ordinary signals until the spawned child's handle is available."""
    requested: list[int | None] = [None]
    previous = {}

    def request_stop(signum, _frame):
        requested[0] = signum

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        yield requested
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _stop_owned_process(process):
    """Terminate and reap this launch's own process group, including error paths."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_bounded(plan, output):
    if active_trainers():
        raise ValueError("Another local trainer is active; refusing a concurrent run")
    lock_path = LOG_ROOT.parent.parent / ".hill-strategy.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another Hill strategy trainer owns the launch lock") from error
        return _run_bounded(plan, output)


def _run_bounded(plan, output):
    if plan["requires_promotion"]:
        require_promotion(plan["environment"].get("MICRODUCK_HILL_PROMOTION"),
                          plan["source"], {**hill_env.TRAINING["gate"], "command_vx": read_report(plan["environment"]["MICRODUCK_HILL_PROMOTION"])["gate"]["command_vx"]})
    minutes = plan["max_wall_minutes"]
    if type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Wall-time budget must be finite and positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "tests.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pytest", "-q"],
                        env={**os.environ, "MICRODUCK_SPRINT_STAGE": "baseline", "MICRODUCK_HILL_STAGE": "bootstrap"}, stdout=stream,
                        stderr=subprocess.STDOUT, check=True)
    smoke(plan["stage"], plan["phase"], plan["source"], output / "smoke", plan["reward_profile"], plan["seed"])
    if active_trainers():
        raise ValueError("Another trainer started during preflight; refusing to launch")
    if (any(digest(p) != value for p, value in plan.get("source_config_files", {}).items())
            or any(digest(p) != value for p, value in plan["source_files"].items())):
        raise ValueError("Source checkpoint or scripts changed during preflight")
    journal = {"plan": plan, "state": "reserved", "pid": None}
    write_json(output / "run.json", journal)
    with (output / "training.log").open("w") as stream, _termination_requests() as stop:
        environment = {**os.environ, **plan["environment"]}
        environment.pop("MICRODUCK_WARM_START", None)
        process = None
        try:
            process = subprocess.Popen(plan["command"], env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            journal.update(state="running", pid=process.pid)
            write_json(output / "run.json", journal)
            deadline = monotonic() + minutes * 60
            while True:
                if stop[0] is not None:
                    journal.update(state="interrupted", signal=stop[0])
                    break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    journal["state"] = "wall_budget_exhausted"
                    break
                try:
                    code = process.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    continue
                journal.update(state="completed" if code == 0 else "failed")
                break
        except KeyboardInterrupt:
            journal["state"] = "interrupted"
        finally:
            if process is not None:
                _stop_owned_process(process)
                journal["exit_code"] = process.returncode
    journal["source_files_unchanged"] = all(digest(p) == value for p, value in plan["source_files"].items())
    journal["source_artifacts_unchanged"] = all(digest(p) == value for p, value in plan.get("source_config_files", {}).items())
    if not journal["source_files_unchanged"] or not journal["source_artifacts_unchanged"]:
        journal["state"] = "source_changed_not_promotable"
    write_json(output / "run.json", journal)
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "smoke", "assess"))
    parser.add_argument("--stage", choices=("bootstrap", "refine"), default="bootstrap")
    parser.add_argument("--phase", choices=("explore", "exploit"), default="explore")
    parser.add_argument("--reward-profile", choices=tuple(REWARD_PROFILES), default="A0")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run", type=Path, dest="run_dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--command", type=float, dest="command_vx",
                        help="Freeze a Hill sweep command BEFORE assessment; never choose on held-outs")
    parser.add_argument("--execute", action="store_true", help="Launch ONE bounded local training; no paid jobs/publication")
    args = parser.parse_args()
    try:
        if args.action == "assess":
            if not args.run_dir or not args.source or not args.output or args.command_vx is None:
                parser.error("assess requires --run, explicit incumbent --source, --command and new --output")
            report = assess(args.run_dir, args.source, args.output, args.command_vx)
            print(json.dumps({"assessment": str(args.output / "assessment.json"), "decision": report["decision"]}, indent=2))
        elif args.action == "smoke":
            if not args.output or args.seed is None:
                parser.error("smoke requires a new --output directory and explicit --seed")
            print(json.dumps(smoke(args.stage, args.phase, args.source, args.output, args.reward_profile, args.seed), indent=2))
        else:
            plan = make_plan(args.stage, args.phase, args.source, args.promotion,
                             args.iterations, args.seed, args.reward_profile)
            if args.action == "run" and args.execute:
                if not args.output:
                    parser.error("run --execute requires a new --output directory")
                result = run_bounded(plan, args.output)
                print(json.dumps(result, indent=2))
                if result["state"] != "completed":
                    parser.exit(1)
            else:
                print(json.dumps(plan, indent=2))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Hill strategy refused: {error}\n")


if __name__ == "__main__":
    main()
