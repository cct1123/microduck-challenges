"""Bounded local Sprint learning and frozen-command assessment CLI."""

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
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, cast

import yaml

from . import env as sprint_env
from .learning import phase_settings
from .promotion import digest, promotion_decision, read_report, require_promotion, runtime_signature, saved_checkpoints, summarize
from .rewards import _ConfigLoader, _config_data, configure_refine_locomotion, select_sprint_checkpoint

TASK = "Mjlab-Sprint2m-MicroDuck"
LOG_ROOT = Path("logs/rsl_rl/sprint_2m").resolve()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selected_source(source=None):
    path = Path(source or sprint_env.REFINE_CHECKPOINT["path"]).resolve()
    match = re.fullmatch(r"model_(0|[1-9][0-9]*)\.pt", path.name)
    if not match or path.parent.parent != LOG_ROOT:
        raise ValueError("Source must be an exact Sprint model_N.pt")
    selected = select_sprint_checkpoint(
        LOG_ROOT, run_pattern=re.escape(path.parent.name), checkpoint=int(match[1]), stage="speed",
        expected_env=sprint_env.make_env_cfg(stage="baseline"), expected_agent=sprint_env.RL_CFG,
    )
    configure_refine_locomotion(sprint_env.make_env_cfg(stage="baseline"),
                                sprint_env.REFINE_SETTINGS, selected)
    return selected


def make_plan(phase, source=None, promotion_path=None, iterations=None, seed=None):
    controls = phase_settings(sprint_env.STAGE_SETTINGS, phase)
    if promotion_path and source is None:
        source = read_report(promotion_path)["candidate_path"]
    source = selected_source(source)["path"]
    budget = controls["max_iterations"]
    iterations = budget if iterations is None else iterations
    if type(iterations) is not int or not 1 <= iterations <= budget:
        raise ValueError(f"Requested updates exceed the {budget}-update phase budget")
    seed = sprint_env.RL_CFG.seed if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    environment = {"MICRODUCK_SPRINT_STAGE": "refine", "MICRODUCK_SPRINT_PHASE": phase,
                   "MICRODUCK_SPRINT_SOURCE": str(source)}
    if promotion_path:
        require_promotion(promotion_path, source, sprint_env.STAGE_SETTINGS["gate"])
        environment["MICRODUCK_SPRINT_PROMOTION"] = str(Path(promotion_path).resolve())
    name = f"{phase}-s{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    command = [sys.executable, "-m", "mjlab.scripts.train", TASK,
               "--gpu-ids", "all", "--env.scene.num-envs", str(sprint_env.TRAINING_RECIPE["num_envs"]),
               "--agent.max-iterations", str(iterations), "--agent.save-interval", str(controls["save_interval"]),
               "--agent.seed", str(seed), "--agent.logger", "tensorboard", "--agent.upload-model", "False",
               "--agent.run-name", name]
    return {"phase": phase, "source": str(source), "source_sha256": digest(source),
            "max_iterations": iterations, "seed": seed, "controls": controls,
            "source_files": {str(p): digest(p) for p in sorted(Path(__file__).parent.iterdir())
                             if p.suffix in (".py", ".toml")},
            "environment": environment, "command": command,
            "requires_promotion": phase == "exploit",
            "max_wall_minutes": sprint_env.STAGE_SETTINGS["gate"]["max_wall_minutes"]}


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
    agent = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(), Loader=_ConfigLoader)
    actor_cfg = agent["actor"]
    if actor_cfg != _config_data(sprint_env.RL_CFG)["actor"]:
        raise ValueError("Unsupported saved actor architecture/normalization configuration")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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


def assess(run, incumbent, output):
    """Select once on development seeds, then open exactly one held-out set."""
    from arena.course import compile_course
    from arena.manifest import load_event
    from arena.policy import load_policy
    from arena.runner import run_once

    run, output = Path(run).resolve(), Path(output).resolve()
    if run.parent != LOG_ROOT:
        raise ValueError("Assess only a saved Sprint experiment")
    ids = inventory(run)
    incumbent = selected_source(incumbent)["path"]
    output.mkdir(parents=True, exist_ok=False)
    gate = dict(sprint_env.STAGE_SETTINGS["gate"])
    manifest, model = load_event("sprint-2m"), None
    model = compile_course(manifest)
    command = manifest.command_at(gate["command_vx"])
    artifacts = {}
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


def smoke(phase, source, output):
    """Two real CPU updates of phase controls; this does not approve racing fitness."""
    import torch
    from mjlab.scripts.train import TrainConfig, run_train
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    selected = selected_source(source)
    source = selected["path"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / source.parent.name).symlink_to(source.parent, target_is_directory=True)
    log_dir = output / "cpu"
    controls = phase_settings(sprint_env.STAGE_SETTINGS, phase)
    cfg = replace(TrainConfig.from_task(TASK), gpu_ids=None,
                  env=configure_refine_locomotion(sprint_env.make_env_cfg(stage="baseline"),
                                                  sprint_env.REFINE_SETTINGS, selected))
    cfg.env.scene.num_envs = 2
    cfg.agent.max_iterations, cfg.agent.save_interval = 2, 1
    cfg.agent.seed, cfg.agent.logger, cfg.agent.upload_model = 4, "tensorboard", False
    cfg.agent.load_run = "^" + re.escape(source.parent.name) + "$"
    cfg.agent.load_checkpoint = "^" + re.escape(source.name) + "$"
    for key in ("learning_rate", "entropy_coef", "clip_param", "exploration_std_scale"):
        setattr(getattr(cfg.agent, "algorithm"), key, controls[key])
    original = MicroduckOnPolicyRunner.load
    saved = torch.load(source, map_location="cpu", weights_only=False)
    before = digest(source)
    source_clock = saved["infos"]["env_state"]["common_step_counter"]
    restored_tensors = 0

    def checked_load(runner, path, *args, **kwargs):
        nonlocal restored_tensors
        if Path(path).resolve() != source:
            raise ValueError("Smoke loaded the wrong source")
        result = original(runner, path, *args, **{**kwargs, "map_location": "cpu"})
        factor = controls["exploration_std_scale"] / saved.get("sprint_exploration_std_scale", 1.0)
        for name, model in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
            for key, value in model.state_dict().items():
                expected = saved[f"{name}_state_dict"][key]
                if name == "actor" and key == "distribution.std_param":
                    expected = expected * factor
                torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
                restored_tensors += 1
        if (runner.current_learning_iteration != 0 or runner.alg.optimizer.state
                or runner.env.unwrapped.common_step_counter != source_clock):
            raise ValueError("Smoke warm-start clocks or fresh optimizer are incorrect")
        return result

    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    setattr(MicroduckOnPolicyRunner, "load", checked_load)
    try:
        run_train(TASK, cfg, log_dir)
    finally:
        setattr(MicroduckOnPolicyRunner, "load", original)
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
    final = torch.load(log_dir / "model_1.pt", map_location="cpu", weights_only=False)
    if final["infos"]["env_state"]["common_step_counter"] != source_clock + 48:
        raise ValueError("Smoke did not complete both 24-step updates")
    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
    scalars = [e.value for tag in acc.Tags()["scalars"] for e in acc.Scalars(tag)]
    if not scalars or not all(math.isfinite(value) for value in scalars) or digest(source) != before:
        raise ValueError("Nonfinite smoke metrics or changed source")
    export = export_actor(log_dir / "model_1.pt", output / "verified.onnx")
    report = {"phase": phase, "controls": controls, "device": "cpu", "updates": 2,
              "restored_state_tensors": restored_tensors, "source": str(source),
              "source_unchanged": True, "finite_scalar_values": len(scalars), "export": export}
    write_json(output / "smoke.json", report)
    return report


def run_bounded(plan, output):
    if active_trainers():
        raise ValueError("Another local trainer is active; refusing a concurrent run")
    lock_path = LOG_ROOT.parent.parent / ".sprint-strategy.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another Sprint strategy trainer owns the launch lock") from error
        return _run_bounded(plan, output)


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


def _run_bounded(plan, output):
    if plan["requires_promotion"]:
        require_promotion(plan["environment"].get("MICRODUCK_SPRINT_PROMOTION"),
                          plan["source"], sprint_env.STAGE_SETTINGS["gate"])
    minutes = plan["max_wall_minutes"]
    if type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("Wall-time budget must be finite and positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "tests.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], stdout=stream,
                        stderr=subprocess.STDOUT, check=True)
    smoke(plan["phase"], plan["source"], output / "smoke")
    if active_trainers():
        raise ValueError("Another trainer started during preflight; refusing to launch")
    if (digest(plan["source"]) != plan["source_sha256"]
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
    if not journal["source_files_unchanged"]:
        journal["state"] = "source_changed_not_promotable"
    write_json(output / "run.json", journal)
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "smoke", "assess"))
    parser.add_argument("--phase", choices=("explore", "exploit"), default="explore")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run", type=Path, dest="run_dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="explicitly launch one bounded local GPU run")
    args = parser.parse_args()
    try:
        if args.action == "assess":
            if not args.run_dir or not args.output:
                parser.error("assess needs --run and a new --output directory")
            report = assess(args.run_dir, args.source, args.output)
            print(json.dumps({"assessment": str(args.output / "assessment.json"),
                              "checkpoints": len(report["checkpoints"]), "decision": report["decision"]}, indent=2))
        elif args.action == "smoke":
            if not args.output:
                parser.error("smoke needs a new --output directory")
            print(json.dumps(smoke(args.phase, args.source, args.output), indent=2))
        else:
            plan = make_plan(args.phase, args.source, args.promotion, args.iterations, args.seed)
            if args.action == "run" and args.execute:
                if not args.output:
                    parser.error("run --execute needs a new --output directory")
                result = run_bounded(plan, args.output)
                print(json.dumps(result, indent=2))
                if result["state"] != "completed":
                    parser.exit(1)
            else:
                print(json.dumps(plan, indent=2))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Sprint strategy refused: {error}\n")


if __name__ == "__main__":
    main()
