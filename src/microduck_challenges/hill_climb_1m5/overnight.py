"""Durable, local-only Hill stability controller and guarded unit preflight.

Importing this module never constructs a simulator, exports an actor or launches a
process. Execution is opt-in through ``run --execute``; no publication is offered.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .evidence import (campaign_snapshots, digest, plan_digest, read_report,
                       runtime_signature, saved_checkpoints, summarize)

MODULE = "microduck_challenges.hill_climb_1m5.overnight"
DEV = list(range(1, 6))
CAL = list(range(26, 31))
HELD = list(range(6, 26))
VX = .75


def durable_json(path, data):
    """Atomic JSON replacement with both file and directory fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def repository():
    return Path(__file__).resolve().parents[3]


# Real-child parser checks and smoke/actual exporter/rollout cases are deliberately
# excluded, not relabelled as units. Mocked assessment and launcher cases remain.
UNIT_EXCLUSIONS = (
    "test_standard_smoke_exercises_official_cpu_export_of_new_checkpoint", "test_check_plays_the_stage_and_fails_a_walker",
    "test_check_reads_a_troupes_star_alone",
    "test_plain_workshop_tyro_defaults",
    "test_bounded_launcher_reaps_only_its_own_nontraining_child",
    "test_bounded_launch_binds_child_config_and_checkpoint_to_exact_plan",
    "test_export_includes_normalizer_and_checks_finite_61_to_14_parity",
    "test_export_preserves_normalization_and_61_to_14_contract",
)


def unit_command():
    return [sys.executable, "-m", "pytest", "tests", "-q", "-p", MODULE,
            "-k", " and ".join(f"not {name}" for name in UNIT_EXCLUSIONS)]


def run_permitted_units(output, *, global_deadline):
    remaining = global_deadline - time.time() - 5
    if remaining <= 0:
        raise ValueError("Unit preflight deadline expired")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = unit_command()
    before = source_hashes()
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("MICRODUCK_HILL_") and k != "MICRODUCK_WARM_START"}
    environment.update(PYTHONPATH=str(repository() / "src"), CUDA_VISIBLE_DEVICES="",
                       MICRODUCK_SPRINT_STAGE="baseline", MICRODUCK_HILL_STAGE="bootstrap",
                       MICRODUCK_OVERNIGHT_UNIT_GUARD="1")
    with (output / "tests.log").open("w") as stream:
        result = subprocess.run(command, cwd=repository(), env=environment,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=min(600, remaining), check=True)
    if source_hashes() != before:
        raise ValueError("Source/tests changed during guarded unit preflight")
    report = {"schema": 1, "exit_code": result.returncode, "command": command,
              "runtime_guard": True, "smoke_executed": False,
              "tests_log_sha256": digest(output / "tests.log"),
              "source_hashes": before, "exclusions": list(UNIT_EXCLUSIONS)}
    durable_json(output / "units.json", report)
    return report


def source_hashes():
    root = repository()
    paths = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.toml"),
             *(root / "tests").glob("*.py"), root / "pyproject.toml", root / "uv.lock"]
    return {str(p.resolve()): digest(p) for p in sorted(paths) if p.is_file()}


def _install_pytest_guard():
    # Imported only by pytest's explicit plugin loader, never by the controller.
    import pytest

    @pytest.fixture(autouse=True)
    def overnight_runtime_guard(monkeypatch):
        import mujoco
        import mjlab.envs
        import mjlab.scripts.train
        import mjlab_microduck.export
        import arena.runner
        import arena.course
        import torch
        def denied(*args, **kwargs):
            raise AssertionError("forbidden simulator/trainer/export/Arena/child in unit check")
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
            monkeypatch.setattr(mujoco, name, denied)
        monkeypatch.setattr(mjlab.envs.ManagerBasedRlEnv, "__init__", denied)
        monkeypatch.setattr(mjlab.scripts.train, "run_train", denied)
        monkeypatch.setattr(mjlab_microduck.export, "run_export", denied)
        monkeypatch.setattr(arena.runner, "run_once", denied)
        monkeypatch.setattr(arena.course, "compile_course", denied)
        monkeypatch.setattr(torch.onnx, "export", denied)
        monkeypatch.setattr(subprocess, "Popen", denied)
        monkeypatch.setattr(os, "execvpe", denied)
        monkeypatch.setattr(sys.modules[MODULE], "arm_owned_process", denied)
    globals()["overnight_runtime_guard"] = overnight_runtime_guard


if os.environ.get("MICRODUCK_OVERNIGHT_UNIT_GUARD") == "1":
    _install_pytest_guard()


def validate_authorization(path):
    """Use the original authorization's epoch, never controller launch time."""
    from datetime import datetime
    auth = read_report(path)
    fixed = {"training_hours": 6, "total_hours": 8, "maximum_new_screens": 5,
             "maximum_total_screens_including_preexisting": 6,
             "maximum_updates_per_screen": 500, "maximum_wall_minutes_per_screen": 50,
             "num_envs": 2048, "save_interval": 50, "command_vx": VX,
             "development_seeds": DEV, "calibration_seeds": CAL, "final_heldout_seeds": HELD,
             "no_smoke": True, "gpu_trainers_at_once": 1,
             "speed_profile_maximum_uses_new_reward_weight": 1.0,
             "larger_lean_allowed": False, "paid_remote_jobs_allowed": False,
             "publication_allowed": False, "push_allowed": False, "uploads_allowed": False}
    if auth.get("schema") != 1 or any(auth.get(k) != v for k, v in fixed.items()):
        raise ValueError("Authorization does not match the bounded eight-hour policy")
    start = auth.get("start_epoch")
    if type(start) not in (int, float) or not math.isfinite(start):
        raise ValueError("Missing finite authorization start epoch")
    training, hard = start + 21600, start + 28800
    if (datetime.fromisoformat(auth["training_deadline_utc"]).timestamp() != training
            or datetime.fromisoformat(auth["hard_deadline_utc"]).timestamp() != hard):
        raise ValueError("Authorization deadline mismatch")
    return {**auth, "training_deadline_epoch": training, "hard_deadline_epoch": hard}


@contextmanager
def reservation(path):
    """One controller across output folders; stale PID text is never a lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another Hill overnight controller owns the reservation") from error
        try:
            yield lock
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def stable_selection(ids, screen, official_eligible=None):
    if set(screen) != {str(n) for n in ids}:
        raise ValueError("Missing unfiltered checkpoint seed results")
    scores = {n: summarize(screen[str(n)], DEV, VX) for n in ids}
    windows = [ids[i:i+3] for i in range(len(ids)-2)
               if all(scores[n]["finishes"] == 5 for n in ids[i:i+3])]
    eligible = {n for window in windows for n in window}
    if official_eligible is not None:
        eligible &= set(official_eligible)
    if not eligible:
        raise ValueError("No three adjacent unfiltered 5/5 checkpoints with a normal official 5/5 candidate")
    candidate = min(eligible, key=lambda n: (scores[n]["mean_s"], n))
    return candidate, next(w for w in windows if candidate in w), scores


def consecutive_failures(ids, screen):
    # Never filter out an unevaluated middle checkpoint to create adjacency.
    return any(str(a) in screen and str(b) in screen
               and summarize(screen[str(a)], DEV, VX)["finishes"] < 5
               and summarize(screen[str(b)], DEV, VX)["finishes"] < 5
               for a, b in zip(ids, ids[1:]))


def _raw_and_export(report, label, checkpoint, rows, seeds):
    artifacts = report["artifact_hashes"]
    raw = Path(report["raw_rows"][label])
    exported = report["exports"][label]
    onnx = Path(exported["onnx_path"])
    for p in (raw, onnx):
        if not p.is_absolute() or str(p) not in artifacts or digest(p) != artifacts[str(p)]:
            raise ValueError("Missing/tampered raw rows or export hashes")
    actual = [read_report_line(line) for line in raw.read_text().splitlines()]
    if actual != rows:
        raise ValueError("Report differs from preserved raw rows")
    score = summarize(actual, seeds, VX)
    if (exported.get("checkpoint_sha256") != digest(checkpoint)
            or exported.get("onnx_sha256") != digest(onnx)
            or exported.get("source_agent_sha256") != digest(checkpoint.parent / "params/agent.yaml")
            or exported.get("input_shape") != [1, 61] or exported.get("output_shape") != [1, 14]
            or exported.get("parity_cases") != 8):
        raise ValueError("Export identity/normalizer/parity changed")
    return score


def read_report_line(line):
    # Same duplicate-key/nonfinite rejection as evidence.read_report, without temp files.
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out: raise ValueError("Duplicate raw row key")
            out[k] = v
        return out
    def invalid(value): raise ValueError(f"Nonfinite raw row: {value}")
    return json.loads(line, object_pairs_hook=unique, parse_constant=invalid)


def verify_worker_report(path, *, checkpoint=None, seeds=None, request=None, freeze=None):
    """Read durable worker evidence and require its complete nonempty identity set."""
    path = Path(path).resolve()
    before = digest(path)
    try:
        report = read_report(path)
        actual_request = read_report(path.parent / "request.json")
        source = Path(actual_request["checkpoint"])
        partition = actual_request["seeds"]
        if (report.get("schema") != 1 or report["request"] != actual_request
                or request is not None and actual_request != request
                or checkpoint is not None and source != Path(checkpoint).resolve()
                or seeds is not None and partition != list(seeds)
                or partition not in (DEV, CAL, HELD) or actual_request["command_vx"] != VX
                or report.get("runtime") != runtime_signature()):
            raise ValueError("Worker request/source/runtime identity changed")
        raw, onnx = Path(report["raw_path"]), Path(report["export"]["onnx_path"])
        required = {source, source.parent / "params/agent.yaml", source.parent / "params/env.yaml",
                    source.parent / "provenance.json", path.parent / "request.json", raw, onnx}
        if partition == HELD:
            frozen_path = Path(actual_request["freeze"])
            if (freeze is not None and frozen_path != Path(freeze).resolve()
                    or digest(frozen_path) != actual_request["freeze_sha256"]
                    or actual_request["partition"] != "heldout"):
                raise ValueError("Worker immutable final freeze changed")
            frozen = read_report(frozen_path)
            identities = {r["path"]: r["sha256"] for r in (frozen["candidate"], frozen["anchor"])}
            if (identities.get(str(source)) != digest(source)
                    or frozen["seeds"] != HELD or frozen["command_vx"] != VX):
                raise ValueError("Worker source differs from immutable final freeze")
        artifacts = report["artifact_hashes"]
        if (not isinstance(artifacts, dict) or not required
                or not {str(p) for p in required} <= artifacts.keys()
                or any(not p.is_absolute() or not p.is_file() or p.stat().st_size == 0 for p in required)
                or any(not Path(p).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", h)
                       or digest(p) != h for p, h in artifacts.items())):
            raise ValueError("Missing/empty/tampered required worker artifacts")
        actual = [read_report_line(line) for line in raw.read_text().splitlines()]
        if actual != report["rows"] or summarize(actual, partition, VX) != report["summary"]:
            raise ValueError("Worker report differs from durable raw rows/summary")
        _raw_and_export({"artifact_hashes": artifacts, "raw_rows": {"worker": str(raw)},
                         "exports": {"worker": report["export"]}}, "worker", source, actual, partition)
        if digest(path) != before or any(digest(p) != h for p, h in artifacts.items()):
            raise ValueError("Worker evidence changed during verification")
        return report
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Missing/malformed required worker evidence") from error


def require_stability_gate(path, source, command_vx=.75):
    """RECOMPUTE Gate S: raw rows, inventory, official source, runtime and hashes.

    No approved/stable flag, timestamp or elapsed timer conveys authority. All
    evidence is tied to the literal frozen candidate and complete finished run.
    This function is deliberately lazy-imported by env/learning to avoid cycles.
    """
    from .campaign import inventory, selected_source
    if command_vx != VX or type(command_vx) not in (int, float):
        raise ValueError("Stability command must remain frozen at vx=.75")
    try:
        path, source = Path(path).resolve(), Path(source).resolve()
        report_digest = digest(path)
        report = read_report(path)
        if (report.get("schema") != 1 or report.get("kind") != "hill-stability"
                or report.get("command_vx") != VX
                or Path(report["candidate_path"]).resolve() != source):
            raise ValueError("Stability gate does not bind this exact frozen candidate/command")
        ids = saved_checkpoints(source.parent)
        if inventory(source.parent) != ids or report["checkpoints"] != ids:
            raise ValueError("Stability gate has a changed/incomplete saved inventory")
        official_eligible = [n for n in ids if official_summary(source.parent / "workshop" / f"model_{n}.json", n)["finishes"] == 5]
        chosen, neighbors, scores = stable_selection(ids, report["screen"], official_eligible)
        if report["candidate_checkpoint"] != chosen or source.name != f"model_{chosen}.pt":
            raise ValueError("Candidate is not the frozen best stable neighborhood selection")
        checkpoints = [source.parent / f"model_{n}.pt" for n in ids]
        if report.get("campaign_evidence") != campaign_snapshots(checkpoints):
            raise ValueError("Stability campaign_evidence changed or was omitted")
        required = {str(p) for p in checkpoints}
        required.update(str(source.parent / n) for n in ("params/env.yaml", "params/agent.yaml", "provenance.json"))
        required.update(str(source.parent / "workshop" / f"model_{n}.json") for n in ids)
        for snapshot in report["campaign_evidence"].values():
            required.add(snapshot["journal"])
            required.update(snapshot["frozen_hashes"])
        artifacts = report["artifact_hashes"]
        if not required <= artifacts.keys() or any(digest(p) != h for p, h in artifacts.items()):
            raise ValueError("Stability source/raw evidence hashes changed or are missing")
        labels = {*(f"screen:{n}" for n in ids), "calibration"}
        if set(report["raw_rows"]) != labels or set(report["exports"]) != labels:
            raise ValueError("Missing complete checkpoint raw rows and exports")
        for n in ids:
            _raw_and_export(report, f"screen:{n}", source.parent / f"model_{n}.pt", report["screen"][str(n)], DEV)
        calibration = _raw_and_export(report, "calibration", source, report["candidate_calibration"], CAL)
        if calibration["finishes"] != 5:
            raise ValueError("Independent calibration is not 5/5")
        # This is independent of custom fixed-command results: normal official
        # 5/5 and the existing complete finite model/recipe loader are required.
        official = read_report(source.parent / "workshop" / f"model_{chosen}.json")
        if official.get("seeds_finished") != 5:
            raise ValueError("Stability source requires normal official 5/5")
        selected = selected_source(source, phase="explore")
        if selected.get("partial_exception") is not None:
            raise ValueError("A partial-source exception is not a reliable anchor")
        if report.get("runtime") != runtime_signature():
            raise ValueError("Stability evaluator runtime changed")
        if (report["campaign_evidence"] != campaign_snapshots(checkpoints)
                or digest(path) != report_digest or any(digest(p) != h for p, h in artifacts.items())):
            raise ValueError("Campaign/source/raw evidence changed during stability validation")
        return {**report, "verified": {"neighbors": neighbors, "development": scores[chosen],
                                      "calibration": calibration, "official_5of5": True}}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Missing/malformed stability gate evidence") from error


def freeze_final(state, candidate):
    """Irreversible candidate identity, committed BEFORE opening held-outs."""
    if not state.get("anchor"):
        raise ValueError("Final heldouts require a verified reliable anchor")
    candidate = Path(candidate).resolve()
    frozen = {"path": str(candidate), "sha256": digest(candidate), "command_vx": VX,
              "heldouts_opened": False, "candidate_rows": None, "anchor_rows": None}
    if state.get("final"):
        if any(state["final"].get(k) != frozen[k] for k in ("path", "sha256", "command_vx")):
            raise ValueError("Final candidate is already frozen; alternatives are forbidden")
        return state["final"]
    state["final"] = frozen
    return frozen


class Journal:
    """Authoritative crash-safe state; reservations precede every side effect."""
    def __init__(self, path, data):
        self.path, self.data = Path(path), data

    @classmethod
    def create(cls, path, initial):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError as error:
            raise ValueError("Authoritative campaign journal already exists") from error
        instance = cls(path, {**deepcopy(initial), "schema": 1, "revision": 0, "events": []})
        instance.save()
        return instance

    @classmethod
    def load(cls, path):
        data = read_report(path)
        checksum = data.pop("journal_sha256")
        if checksum != plan_digest(data) or data["revision"] != len(data["events"]):
            raise ValueError("Authoritative journal checksum/history mismatch")
        return cls(path, data)

    def save(self):
        durable_json(self.path, {**self.data, "journal_sha256": plan_digest(self.data)})

    def record(self, event, **changes):
        self.data.update(deepcopy(changes))
        previous = self.data["events"][-1]["sha256"] if self.data["events"] else None
        entry = {"event": event, "epoch": time.time(), "changes": deepcopy(changes), "previous": previous}
        entry["sha256"] = plan_digest(entry)
        self.data["events"].append(entry)
        self.data["revision"] += 1
        self.save()


def next_action(state, now):
    """Conservative policy: two stabilization seeds, at most one gated P1 screen."""
    screens = state["screens"]
    if now >= state["training_deadline_epoch"] or len(screens) >= 5:
        return "finalize", None
    s0 = sum(s["profile"] == "S0" for s in screens)
    if s0 < 2:
        return "S0", s0 + 1
    if not state.get("anchor"):
        return "stop_no_stability", None
    if any(s["profile"] == "P1" for s in screens):
        return "finalize", None
    if now < state["start_epoch"] + 14400:
        return "wait_speed_window", state["start_epoch"] + 14400
    return "P1", 1


class OwnedEvaluation:
    """One isolated CPU scorer process group, with a non-extendable wall deadline."""
    def __init__(self, checkpoint, output, seeds, *, deadline, authorization=None,
                 freeze=None, partition="development"):
        if deadline <= time.time() + 5:
            raise ValueError("Scoring worker deadline has expired")
        if list(seeds) == HELD and not freeze:
            raise ValueError("Heldout scoring requires the immutable final freeze")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.deadline = min(deadline - 3, time.time() + (1200 if list(seeds) == HELD else 600))
        self.request = {"checkpoint": str(Path(checkpoint).resolve()), "seeds": list(seeds),
                        "command_vx": VX, "deadline_epoch": self.deadline,
                        "authorization": str(Path(authorization).resolve()) if authorization else None,
                        "freeze": str(Path(freeze).resolve()) if freeze else None,
                        "freeze_sha256": digest(freeze) if freeze else None,
                        "partition": partition, "owner_pid": os.getpid()}
        durable_json(self.output / "request.json", self.request)
        self.stream = (self.output / "worker.log").open("w")
        environment = {k: v for k, v in os.environ.items()
                       if not k.startswith("MICRODUCK_HILL_") and k != "MICRODUCK_WARM_START"}
        environment.update(PYTHONPATH=str(repository() / "src"), CUDA_VISIBLE_DEVICES="",
                           MICRODUCK_SPRINT_STAGE="baseline", MICRODUCK_HILL_STAGE="bootstrap",
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", MODULE, "worker", "--request", str(self.output / "request.json"),
                 "--request-sha256", digest(self.output / "request.json"), "--execute"],
                cwd=repository(), env=environment, stdout=self.stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        from .campaign import _stop_owned_process
        try:
            _stop_owned_process(self.process, self.deadline)
        finally:
            self.stream.close()

    def poll(self):
        if time.time() >= self.deadline:
            self.close()
            raise ValueError("Owned scoring worker deadline exhausted")
        code = self.process.poll()
        if code is None:
            return None
        self.stream.close()
        if code != 0:
            raise ValueError(f"Owned scoring worker failed (exit {code}); preserve {self.output}")
        return verify_worker_report(self.output / "evaluation.json", request=self.request)


def arm_owned_process(owner_pid, deadline, *, kill_group=False):
    """Linux parent-death protection and an independent absolute wall timer."""
    import ctypes
    if os.getppid() != owner_pid:
        raise ValueError("Owned process controller is no longer alive")
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
        raise OSError("Could not install owned-process parent-death signal")
    if os.getppid() != owner_pid or deadline <= time.time():
        raise ValueError("Owned process owner/deadline changed before execution")
    if kill_group:
        def expired(*_): os.killpg(os.getpgrp(), signal.SIGKILL)
        signal.signal(signal.SIGALRM, expired)
    else:
        # ITIMER_REAL survives exec; SIGALRM's default disposition kills the
        # single-GPU trainer even if its controller is stuck in an I/O call.
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())


def execute_trainer_worker(journal_path, identity, owner_pid, deadline):
    journal = read_report(journal_path)
    plan = journal["plan"]
    if (plan_digest(plan) != identity or journal.get("plan_digest") != identity
            or journal.get("preflight", {}).get("no_smoke") is not True
            or journal.get("trainer_deadline_epoch") != deadline
            or deadline > journal["global_deadline_epoch"]
            or plan["command"][1:4] != ["-m", "mjlab.scripts.train", "Mjlab-HillClimb1m5-MicroDuck"]
            or plan["command"][0] != sys.executable):
        raise ValueError("Trainer wrapper does not match the immutable bounded launch")
    if any(digest(p) != h for p, h in {**plan["source_files"], **plan["source_config_files"]}.items()):
        raise ValueError("Trainer source changed before exec")
    arm_owned_process(owner_pid, deadline)
    os.execvpe(plan["command"][0], plan["command"], os.environ.copy())


def execute_worker(request_path, request_sha256):
    """Actual authorized race assessment; never called by unit preflight."""
    if digest(request_path) != request_sha256:
        raise ValueError("Changed scoring request")
    request = read_report(request_path)
    if os.getppid() != request["owner_pid"]:
        raise ValueError("Scoring worker owner is no longer alive")
    # Kernel parent-death signal plus a process-group alarm bound orphaned races.
    import ctypes
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
        raise OSError("Could not install worker parent-death signal")
    if os.getppid() != request["owner_pid"]:
        raise ValueError("Scoring worker owner exited during startup")
    auth = validate_authorization(request["authorization"])
    deadline = min(request["deadline_epoch"], auth["hard_deadline_epoch"] - 2)
    if time.time() >= deadline:
        raise ValueError("Scoring deadline expired before initialization")
    def expired(*_):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    seeds = request["seeds"]
    if request["command_vx"] != VX or seeds not in (DEV, CAL, HELD):
        raise ValueError("Undeclared seed partition or mixed command")
    checkpoint = Path(request["checkpoint"])
    if seeds == HELD:
        freeze = request["freeze"]
        if not freeze or digest(freeze) != request["freeze_sha256"]:
            raise ValueError("Final heldouts cannot open without immutable candidate freeze")
        frozen = read_report(freeze)
        identities = {r["path"]: r["sha256"] for r in (frozen["candidate"], frozen["anchor"])}
        if (str(checkpoint) not in identities or digest(checkpoint) != identities[str(checkpoint)]
                or frozen["seeds"] != HELD or frozen["command_vx"] != VX):
            raise ValueError("Heldout candidate is not the one frozen candidate/paired R")
    elif request["partition"] == "heldout":
        raise ValueError("Wrong final seed partition")
    import torch
    from .campaign import export_actor
    from arena.course import compile_course
    from arena.manifest import load_event
    from arena.policy import load_policy
    from arena.runner import run_once
    torch.set_num_threads(2)
    output = Path(request_path).parent
    before = {str(p): digest(p) for p in (checkpoint, checkpoint.parent / "params/agent.yaml",
              checkpoint.parent / "params/env.yaml", checkpoint.parent / "provenance.json")}
    runtime = runtime_signature()
    onnx = output / "policy.onnx"
    exported = {**export_actor(checkpoint, onnx), "onnx_path": str(onnx)}
    manifest = load_event("hill-climb-1m5")
    model, policy = compile_course(manifest), load_policy(onnx)
    rows, raw = [], output / "rows.jsonl"
    with raw.open("x") as stream:
        for seed in seeds:
            if time.time() >= deadline:
                raise ValueError("Scoring deadline expired before next seed")
            result = run_once(manifest, policy, seed, model=model, command=manifest.command_at(VX))
            row = {"seed": seed, "status": result.status, "time_s": result.time_s,
                   "progress_m": result.progress_m, "command_vx": VX}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    summary = summarize(rows, seeds, VX)
    if runtime_signature() != runtime or any(digest(p) != h for p, h in before.items()):
        raise ValueError("Race inputs/runtime changed")
    artifacts = {**before, str(onnx): digest(onnx), str(raw): digest(raw),
                 str(Path(request_path)): request_sha256}
    report = {"schema": 1, "request": request, "rows": rows, "summary": summary,
              "raw_path": str(raw), "export": exported, "artifact_hashes": artifacts,
              "runtime": runtime}
    durable_json(output / "evaluation.json", report)
    signal.setitimer(signal.ITIMER_REAL, 0)
    return report


class BranchRollback(ValueError):
    """Late race evidence vetoes the finished speed branch as a source, too."""


class ScreenMonitor:
    """Score live saved checkpoints without adopting an unfinished run as source."""
    def __init__(self, controller, plan, output):
        self.controller, self.plan, self.output = controller, plan, Path(output)
        self.run = None
        self.worker = None
        self.worker_number = None
        self.records = {}
        self.next_poll = 0.

    def locate_run(self):
        from .campaign import LOG_ROOT
        name = self.plan["command"][self.plan["command"].index("--agent.run-name")+1]
        matches = list(LOG_ROOT.glob(f"*_{name}"))
        if len(matches) > 1:
            raise ValueError("Ambiguous training run identity")
        if matches:
            found = matches[0].resolve()
            if self.run != found:
                current = {**self.controller.state["current_screen"], "run": str(found)}
                self.controller.journal.record("trainer_run_discovered", active_run=str(found), current_screen=current)
            self.run = found
        return self.run

    def tick(self, *, finished=False):
        self.controller.check_deadline()
        if not self.run and not self.locate_run():
            return
        if self.worker:
            report = self.worker.poll()
            if report is not None:
                self.records[str(self.worker_number)] = report
                self.worker = None
                self.controller.journal.record("checkpoint_assessed", active_run=str(self.run),
                    active_checkpoint=self.worker_number, active_assessments=len(self.records))
                durable_json(self.output / "live.json", {"run": str(self.run), "records": self.records})
        if self.worker is None:
            paths = sorted(self.run.glob("model_*.pt"), key=lambda p: int(p.stem.removeprefix("model_")))
            for path in paths:
                number = int(path.stem.removeprefix("model_"))
                if str(number) in self.records:
                    continue
                # The trainer saves directly; give a newly written file time to close.
                if not finished and time.time() - path.stat().st_mtime < 3:
                    continue
                self.worker_number = number
                self.worker = self.controller.start_evaluation(path, self.output / f"model-{number}", DEV)
                break

    def __call__(self, process, launch_journal):
        if time.monotonic() < self.next_poll:
            return None
        self.next_poll = time.monotonic() + 10
        from .campaign import active_trainers
        if self.controller.state.get("active_trainer_pid") != process.pid:
            self.controller.journal.record("trainer_process_owned", active_trainer_pid=process.pid)
        others = set(active_trainers()) - {process.pid}
        if others:
            raise ValueError(f"Competing trainers appeared: {sorted(others)}")
        self.tick()
        return self.rollback_reason()

    def rollback_reason(self):
        if self.plan["reward_profile"] == "P1" and self.run:
            ids = sorted(int(p.stem.removeprefix("model_")) for p in self.run.glob("model_*.pt"))
            rows = {key: r["rows"] for key, r in self.records.items()}
            if consecutive_failures(ids, rows):
                return "two consecutive sub-5/5 P1 fixed-command checkpoints; preserve R"
            # Ordinary Workshop may finish first; failures there also veto P1.
            official_finishes = {}
            for number in ids:
                path = self.run / "workshop" / f"model_{number}.json"
                if path.exists():
                    official_finishes[number] = official_summary(path, number)["finishes"]
            if any(official_finishes.get(a, 5) < 5 and official_finishes.get(b, 5) < 5
                   for a, b in zip(ids, ids[1:])):
                return "two consecutive sub-5/5 P1 official checkpoints; preserve R"
        return None

    def drain(self):
        from .campaign import inventory
        if not self.run and not self.locate_run():
            raise ValueError("Trainer produced no uniquely identified saved run")
        ids = saved_checkpoints(self.run)
        deadline = min(time.time()+1800, self.controller.state["hard_deadline_epoch"]-1800)
        self.controller.journal.record("draining_checkpoint_backlog", active_run=str(self.run),
                                      expected_checkpoints=ids, backlog_deadline_epoch=deadline)
        while True:
            self.controller.check_deadline(deadline)
            self.tick(finished=True)
            reason = self.rollback_reason()
            if reason:
                raise BranchRollback(reason)
            official = [n for n in ids if (self.run / "workshop" / f"model_{n}.json").exists()]
            for number in official:
                official_summary(self.run / "workshop" / f"model_{number}.json", number)
            if set(self.records) == {str(n) for n in ids} and official == ids:
                if inventory(self.run) != ids:
                    raise ValueError("Saved/official inventory changed while draining")
                return ids
            self.controller.pause(2)

    def close(self):
        if self.worker:
            self.worker.close()
            self.worker = None


def official_summary(path, number):
    result = read_report(path)
    if result.get("error") or result.get("checkpoint") != number or result.get("seeds_total") != 5:
        raise ValueError(f"Incomplete/error official Workshop assessment: {path}")
    score = summarize([{**row, "command_vx": result["command_vx"]} for row in result["per_seed"]], DEV, result["command_vx"])
    if score["finishes"] != result["seeds_finished"] or not math.isclose(score["mean_s"], result["score"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("Contradictory official Workshop raw rows")
    return score


def process_identity(pid):
    """Read a process generation and literal argv, detecting PID reuse mid-read."""
    proc = Path("/proc") / str(pid)
    before = proc.joinpath("stat").read_text().rsplit(") ", 1)[1].split()[19]
    argv = proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
    after = proc.joinpath("stat").read_text().rsplit(") ", 1)[1].split()[19]
    if before != after:
        raise ValueError("Preexisting process identity changed during discovery")
    return {"start_ticks": before, "argv": argv}


class Controller:
    """Bounded unattended supervisor. Resume after an uncertain launch fails closed."""
    TERMINAL = {"stability_unmet", "complete", "rejected", "incomplete", "blocked", "interrupted", "deadline_exhausted"}

    def __init__(self, authorization, output, source=None):
        self.authorization, self.output = Path(authorization).resolve(), Path(output).resolve()
        self.source = source
        self.journal = None
        self.workers = []
        self.stop_signal = None

    @property
    def state(self):
        return self.journal.data

    def prepare(self):
        from .campaign import selected_source
        from .learning import APPROVED_PARTIAL
        auth = validate_authorization(self.authorization)
        if time.time() >= auth["hard_deadline_epoch"] - 10:
            raise ValueError("Original authorization hard deadline has expired")
        source = Path(self.source or APPROVED_PARTIAL["source"]).resolve()
        if source != Path(APPROVED_PARTIAL["source"]).resolve():
            raise ValueError("Only the exact original E has an initial partial-source exception")
        selected = selected_source(source, phase="explore")
        if selected["partial_exception"] != APPROVED_PARTIAL["partial_exception"]:
            raise ValueError("Original E identity/exception was not verified")
        self.output.mkdir(parents=True, exist_ok=False)
        durable_json(self.output / "authorization.json", read_report(self.authorization))
        frozen = {**source_hashes(), **selected["source_identity"]["artifacts"],
                  str(self.authorization): digest(self.authorization),
                  str(self.output / "authorization.json"): digest(self.output / "authorization.json")}
        inputs = {"schema": 1, "files": frozen, "runtime": runtime_signature(),
                  "source_identity": selected["source_identity"], "authorization": auth,
                  "repository": str(repository())}
        durable_json(self.output / "inputs.json", inputs)
        os.chmod(self.output / "authorization.json", 0o444)
        os.chmod(self.output / "inputs.json", 0o444)
        return Journal.create(self.output / "campaign.json", {
            "state": "prepared", "start_epoch": auth["start_epoch"],
            "training_deadline_epoch": auth["training_deadline_epoch"],
            "hard_deadline_epoch": auth["hard_deadline_epoch"],
            "maximum_new_screens": auth["maximum_new_screens"],
            "preexisting": auth["preexisting_hill_job"],
            "superseded_preexisting_jobs": auth.get("superseded_preexisting_jobs", []),
            "controller_pid": os.getpid(), "controller_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "controller_start_ticks": Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19],
            "inputs_path": str(self.output / "inputs.json"), "inputs_sha256": digest(self.output / "inputs.json"),
            "original_source": selected["source_identity"], "screens": [], "current_screen": None,
            "reliable_sources": [], "best_reliable": None, "anchor": None, "best_candidate": None,
            "final": None, "phase": "explore", "automatic_publication": False})

    def verify_inputs(self):
        path = self.state["inputs_path"]
        if digest(path) != self.state["inputs_sha256"]:
            raise ValueError("Immutable campaign inputs manifest changed")
        inputs = read_report(path)
        if (any(digest(p) != h for p, h in inputs["files"].items())
                or inputs["runtime"] != runtime_signature()
                or validate_authorization(self.authorization) != inputs["authorization"]):
            raise ValueError("Frozen authorization/source/evaluator inputs changed")
        for record in (self.state.get("anchor"), self.state.get("best_candidate")):
            if record:
                if digest(record["gate_path"]) != record["gate_sha256"]:
                    raise ValueError("Retained reliable gate changed")
                require_stability_gate(record["gate_path"], record["path"])

    def check_deadline(self, deadline=None):
        if self.stop_signal is not None:
            raise InterruptedError(f"Controller interrupted by signal {self.stop_signal}")
        limit = self.state["hard_deadline_epoch"] - 5
        if deadline is not None:
            limit = min(limit, deadline)
        if time.time() >= limit:
            raise TimeoutError("Campaign/backlog hard deadline exhausted")

    def pause(self, seconds):
        self.check_deadline()
        time.sleep(max(0, min(seconds, self.state["hard_deadline_epoch"] - time.time() - 5)))

    def wait_until(self, epoch, reason):
        self.journal.record(reason, state="waiting", wait_until_epoch=epoch)
        while time.time() < epoch:
            self.check_deadline()
            self.pause(min(30, epoch-time.time()))

    def reconcile_preexisting(self):
        from .campaign import LOG_ROOT, active_trainers, inventory
        diagnostics = []
        for record in self.state.get("superseded_preexisting_jobs", []):
            run = LOG_ROOT / record["run_id"]
            paths = sorted((p for p in run.glob("model_*.pt")
                            if re.fullmatch(r"model_(0|[1-9][0-9]*)", p.stem)),
                           key=lambda p: int(p.stem.removeprefix("model_")))
            diagnostics.append({**record, "run": str(run),
                "saved": [int(p.stem.removeprefix("model_")) for p in paths],
                "hashes": {str(p): digest(p) for p in paths},
                "diagnostic_only": True, "valid_continuation_source": False})
        if diagnostics:
            self.journal.record("superseded_preexisting_diagnostic_only",
                                superseded_preexisting_inventories=diagnostics)
        old = self.state.get("preexisting")
        if not old:
            return
        deadline = min(time.time()+2700, self.state["training_deadline_epoch"]-300)
        run = LOG_ROOT / old["run_id"]
        while True:
            self.check_deadline(deadline)
            trainers = active_trainers()
            if trainers and trainers != [old["pid_at_discovery"]]:
                raise ValueError(f"Unowned/competing trainer detected: {trainers}")
            if trainers:
                identity = process_identity(old["pid_at_discovery"])
                argv = identity["argv"]
                names = [argv[i+1] for i, token in enumerate(argv[:-1]) if token == "--agent.run-name"]
                names += [token.split("=", 1)[1] for token in argv if token.startswith("--agent.run-name=")]
                if (not old.get("start_ticks_at_discovery")
                        or identity["start_ticks"] != str(old["start_ticks_at_discovery"])
                        or names != [old["id"]]
                        or "Mjlab-HillClimb1m5-MicroDuck" not in argv):
                    raise ValueError("Preexisting trainer identity does not match recorded start ticks/run argv")
            if not trainers:
                ids = saved_checkpoints(run)
                pending = [n for n in ids if not (run / "workshop" / f"model_{n}.json").exists()]
                if not pending:
                    inventory(run)
                    if active_trainers():
                        raise ValueError("Unowned trainer appeared during preexisting inventory reconciliation")
                    self.journal.record("preexisting_reconciled", preexisting_inventory={
                        "run": str(run), "expected": ids, "evaluated": ids, "pending": [],
                        "hashes": {str(run / f"model_{n}.pt"): digest(run / f"model_{n}.pt") for n in ids}})
                    return
            self.pause(15)

    def start_evaluation(self, checkpoint, output, seeds, *, freeze=None, partition="development"):
        self.check_deadline()
        worker = OwnedEvaluation(checkpoint, output, seeds,
            deadline=self.state["hard_deadline_epoch"]-15,
            authorization=self.output / "authorization.json", freeze=freeze, partition=partition)
        self.workers.append(worker)
        return worker

    def evaluate(self, checkpoint, output, seeds, *, freeze=None, partition="development"):
        worker = self.start_evaluation(checkpoint, output, seeds, freeze=freeze, partition=partition)
        try:
            while True:
                self.check_deadline()
                result = worker.poll()
                if result is not None:
                    return result
                self.pause(1)
        finally:
            worker.close()

    def launch_plan(self, profile, seed, source):
        from .campaign import make_plan
        if profile not in ("S0", "P1"):
            raise ValueError("Overnight reward profile must remain S0 or the single P1 step")
        if profile == "P1":
            if time.time() < self.state["start_epoch"] + 14400:
                raise ValueError("P1 is forbidden during the first four hours (T+4h)")
            if not self.state.get("anchor") or any(s["profile"] == "P1" for s in self.state["screens"]):
                raise ValueError("P1 requires Gate S and permits only one speed experiment")
        if len(self.state["screens"]) >= 5:
            raise ValueError("Maximum five new screen reservations exhausted")
        record = self.state.get("best_candidate") or self.state.get("anchor")
        gate = record["gate_path"] if profile == "P1" else None
        if profile == "P1" and Path(record["path"]).resolve() != Path(source).resolve():
            raise ValueError("P1 must bind the strongest confirmed source's own stability gate")
        previous = {key: os.environ.get(key) for key in
                    ("MICRODUCK_HILL_STABILITY_GATE", "MICRODUCK_HILL_STABILITY_GATE_SHA256")}
        try:
            if gate:
                require_stability_gate(gate, source)
                os.environ["MICRODUCK_HILL_STABILITY_GATE"] = gate
                os.environ["MICRODUCK_HILL_STABILITY_GATE_SHA256"] = digest(gate)
            else:
                for key in previous: os.environ.pop(key, None)
            plan = make_plan(stage="refine", phase="explore", source=source,
                             iterations=500, seed=seed, reward_profile=profile)
        finally:
            for key, value in previous.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
        plan["controls"].update(max_iterations=500, num_envs=2048, save_interval=50, max_wall_minutes=50)
        plan.update(max_iterations=500, max_wall_minutes=50)
        for flag, value in {"--env.scene.num-envs": "2048", "--agent.max-iterations": "500",
                            "--agent.save-interval": "50", "--agent.upload-model": "False"}.items():
            if plan["command"].count(flag) != 1:
                raise ValueError("Malformed bounded launch command")
            plan["command"][plan["command"].index(flag)+1] = value
        for p in (self.output / "inputs.json", self.output / "authorization.json", self.authorization):
            plan["source_config_files"][str(p)] = digest(p)
        if gate:
            plan["environment"]["MICRODUCK_HILL_STABILITY_GATE"] = gate
            plan["environment"]["MICRODUCK_HILL_STABILITY_GATE_SHA256"] = digest(gate)
            plan["source_config_files"][gate] = digest(gate)
        if (plan["phase"] != "explore" or plan["requires_promotion"]
                or plan["target_algorithm"]["learning_rate"] != .00005
                or plan["target_algorithm"]["entropy_coef"] != .01
                or plan["target_algorithm"]["clip_param"] != .2
                or plan["target_algorithm"]["exploration_std_scale"] != 1.):
            raise ValueError("Overnight exploration controls changed")
        return plan

    def choose_source(self, profile):
        from .campaign import selected_source
        if profile == "P1":
            record = self.state.get("best_candidate") or self.state.get("anchor")
        else:
            record = self.state.get("best_reliable")
        if record:
            if digest(record["path"]) != record["sha256"]:
                raise ValueError("Selected best reliable source changed")
            selected = selected_source(record["path"], phase="explore")
            if selected.get("partial_exception") is not None:
                raise ValueError("A descendant cannot inherit E's exception")
            return selected["path"]
        return Path(self.state["original_source"]["path"])

    def run_screen(self, profile, seed):
        from .campaign import run_bounded, selected_source
        self.verify_inputs()
        if time.time() + 300 >= self.state["training_deadline_epoch"]:
            raise TimeoutError("Insufficient launch time before training cutoff")
        source = self.choose_source(profile)
        plan = self.launch_plan(profile, seed, source)
        ordinal = len(self.state["screens"]) + 1
        launch = self.output / "launches" / f"screen-{ordinal:02d}"
        assessment = self.output / "assessments" / f"screen-{ordinal:02d}"
        durable_json(self.output / "plans" / f"screen-{ordinal:02d}.json", plan)
        current = {"ordinal": ordinal, "profile": profile, "seed": seed, "source": str(source),
                   "source_sha256": digest(source), "state": "reserved", "launch_journal": str(launch / "run.json"),
                   "plan_digest": plan_digest(plan), "assessment": str(assessment)}
        self.journal.record("screen_reserved", state="training", current_screen=current,
                            active_trainer_pid=None, active_run=None)
        monitor = ScreenMonitor(self, plan, assessment)
        try:
            launched = run_bounded(plan, launch, no_smoke=True,
                                   global_deadline=self.state["training_deadline_epoch"], monitor=monitor)
            current.update(state=launched["state"], exit_code=launched["exit_code"],
                           run=str(monitor.run) if monitor.run else None)
            if launched["state"] == "branch_rollback":
                self.journal.record("speed_branch_rolled_back", rollback_anchor=self.state["anchor"], current_screen=current)
                return current
            if launched["state"] != "completed":
                raise ValueError(f"Screen did not complete: {launched['state']}")
            try:
                ids = monitor.drain()
            except BranchRollback as error:
                launched.update(state="branch_rollback", stop_reason=str(error))
                durable_json(launch / "run.json", launched)
                current.update(state="branch_rollback", stop_reason=str(error))
                self.journal.record("late_speed_evidence_rolled_back", current_screen=current,
                                    rollback_anchor=self.state["anchor"])
                return current
            run = monitor.run
            current.update(run=str(run), expected_checkpoints=ids, evaluated_checkpoints=ids, pending_checkpoints=[])
            screen = {str(n): monitor.records[str(n)]["rows"] for n in ids}
            reliable = []
            for n in ids:
                fixed = summarize(screen[str(n)], DEV, VX)
                official = official_summary(run / "workshop" / f"model_{n}.json", n)
                if fixed["finishes"] == official["finishes"] == 5:
                    checkpoint = run / f"model_{n}.pt"
                    selected_source(checkpoint, phase="explore")
                    reliable.append({"path": str(checkpoint), "sha256": digest(checkpoint),
                                     "profile": profile, "development": fixed, "official": official})
            current["reliable"] = reliable
            if reliable:
                candidates = [*self.state["reliable_sources"], *reliable]
                best = min(candidates, key=lambda r: (r["development"]["mean_s"], r["path"]))
                self.journal.record("global_reliable_sources_updated", reliable_sources=candidates, best_reliable=best)
            try:
                eligible = [int(Path(r["path"]).stem.removeprefix("model_")) for r in reliable]
                number, neighbors, scores = stable_selection(ids, screen, eligible)
            except ValueError as error:
                current["gate_rejection"] = str(error)
                return current
            candidate = run / f"model_{number}.pt"
            # Freeze ONCE before independent calibration; no alternate after failure.
            durable_json(assessment / "selection.json", {"candidate_path": str(candidate),
                "candidate_sha256": digest(candidate), "neighbors": neighbors, "command_vx": VX})
            if official_summary(run / "workshop" / f"model_{number}.json", number)["finishes"] != 5:
                current["gate_rejection"] = "Selected stability candidate lacks normal official 5/5"
                return current
            calibration = self.evaluate(candidate, assessment / "calibration", CAL, partition="calibration")
            if summarize(calibration["rows"], CAL, VX)["finishes"] != 5:
                current["gate_rejection"] = "Selected stability candidate failed independent calibration"
                return current
            checkpoints = [run / f"model_{n}.pt" for n in ids]
            campaigns = campaign_snapshots(checkpoints)
            artifacts = {str(p): digest(p) for p in [*checkpoints, run / "params/env.yaml", run / "params/agent.yaml",
                run / "provenance.json", *(run / "workshop" / f"model_{n}.json" for n in ids), assessment / "selection.json"]}
            for c in campaigns.values():
                artifacts[c["journal"]] = c["journal_sha256"]
                artifacts.update(c["frozen_hashes"])
            records = {**{f"screen:{n}": monitor.records[str(n)] for n in ids}, "calibration": calibration}
            for record in records.values(): artifacts.update(record["artifact_hashes"])
            report = {"schema": 1, "kind": "hill-stability", "command_vx": VX,
                "checkpoints": ids, "candidate_checkpoint": number, "candidate_path": str(candidate),
                "screen": screen, "candidate_calibration": calibration["rows"],
                "raw_rows": {label: r["raw_path"] for label, r in records.items()},
                "exports": {label: r["export"] for label, r in records.items()},
                "runtime": runtime_signature(), "campaign_evidence": campaigns, "artifact_hashes": artifacts}
            gate_path = assessment / "stability.json"
            durable_json(gate_path, report)
            require_stability_gate(gate_path, candidate)
            retained = {"path": str(candidate), "sha256": digest(candidate), "profile": profile,
                "development": scores[number], "gate_path": str(gate_path), "gate_sha256": digest(gate_path),
                "training_seed": seed, "neighbors": neighbors}
            current["stability_gate"] = retained
            if profile == "S0" and not self.state.get("anchor"):
                self.journal.record("reliable_anchor_frozen", anchor=retained, best_candidate=retained)
            elif profile == "S0" and scores[number]["mean_s"] < self.state["best_candidate"]["development"]["mean_s"]:
                # Preserve the ORIGINAL R for all speed and paired comparisons.
                self.journal.record("stable_candidate_improved", best_candidate=retained)
            elif profile == "P1" and scores[number]["mean_s"] < self.state["anchor"]["development"]["mean_s"]:
                self.journal.record("speed_candidate_frozen_for_final", best_candidate=retained)
            return current
        finally:
            monitor.close()

    def finalize(self):
        if self.state.get("final", {}) and self.state["final"].get("heldouts_opened"):
            raise ValueError("Heldouts already opened; automatic retry is forbidden")
        if not self.state.get("anchor"):
            s0_count = sum(s["profile"] == "S0" for s in self.state["screens"])
            self.journal.record("stability_gate_unmet_stop_blind_compute", state="stability_unmet",
                                current_screen=None,
                                stop_reason=f"No confirmed stability anchor after {s0_count} completed S0 screens; no speed/heldouts opened")
            return
        self.verify_inputs()
        candidate = self.state["best_candidate"] or self.state["anchor"]
        freeze_final(self.state, candidate["path"])
        self.journal.record("final_candidate_frozen", state="final_validation", final=self.state["final"], current_screen=None)
        freeze_path = self.output / "final-freeze.json"
        frozen = {"schema": 1, "candidate": candidate, "anchor": self.state["anchor"],
                  "command_vx": VX, "seeds": HELD, "automatic_promotion": False}
        if freeze_path.exists():
            if read_report(freeze_path) != frozen:
                raise ValueError("Final freeze changed; heldout alternatives forbidden")
        else:
            durable_json(freeze_path, frozen)
            os.chmod(freeze_path, 0o444)
        if time.time() < self.state["training_deadline_epoch"]:
            self.wait_until(self.state["training_deadline_epoch"], "final_candidate_frozen_wait_for_validation_reserve")
        self.verify_inputs()
        if self.state["final"]["heldouts_opened"]:
            raise ValueError("Heldouts already opened; automatic retry is forbidden")
        self.state["final"]["heldouts_opened"] = True
        self.journal.record("heldouts_opened_irreversibly", final=self.state["final"], final_freeze_sha256=digest(freeze_path))
        result = self.evaluate(candidate["path"], self.output / "final-candidate", HELD,
                               freeze=freeze_path, partition="heldout")
        self.state["final"].update(candidate_rows=result["rows"],
            candidate_score=summarize(result["rows"], HELD, VX), artifact_hashes=result["artifact_hashes"],
            candidate_evaluation=str(self.output / "final-candidate/evaluation.json"))
        self.journal.record("final_candidate_assessment_persisted", final=self.state["final"])
        paired = self.evaluate(self.state["anchor"]["path"], self.output / "final-anchor", HELD,
                               freeze=freeze_path, partition="heldout")
        # The candidate may have changed while the paired anchor was racing.
        # Read BOTH complete worker reports again before any terminal success.
        for label, expected, identity in (("candidate", result, candidate), ("anchor", paired, self.state["anchor"])):
            persisted = verify_worker_report(self.output / f"final-{label}/evaluation.json",
                checkpoint=identity["path"], seeds=HELD, freeze=freeze_path)
            if persisted != expected:
                raise ValueError("Final worker report changed after paired assessment")
        score, anchor = summarize(result["rows"], HELD, VX), summarize(paired["rows"], HELD, VX)
        speed = candidate["profile"] == "P1"
        reasons = []
        if score["finishes"] != 20: reasons.append("candidate did not finish 20/20 final heldouts")
        if speed and score["finishes"] < anchor["finishes"]: reasons.append("speed candidate reliability regressed")
        if speed and candidate["development"]["mean_s"] >= self.state["anchor"]["development"]["mean_s"]:
            reasons.append("speed candidate is not faster on development")
        if speed and score["mean_s"] > anchor["mean_s"] + 1e-9:
            reasons.append("speed candidate is worse on paired heldout penalized mean")
        self.state["final"].update(candidate_rows=result["rows"], anchor_rows=paired["rows"],
            candidate_score=score, anchor_score=anchor, reasons=reasons,
            outcome="speed_comparison_passed_no_automatic_promotion" if speed and not reasons else
                    "stability_milestone" if not reasons else "rejected",
            anchor_evaluation=str(self.output / "final-anchor/evaluation.json"),
            artifact_hashes={**result["artifact_hashes"], **paired["artifact_hashes"],
                **{str(self.output / f"final-{label}/evaluation.json"): digest(self.output / f"final-{label}/evaluation.json")
                   for label in ("candidate", "anchor")}})
        self.verify_inputs()
        self.journal.record("final_validation_complete", state="rejected" if reasons else "complete", final=self.state["final"])

    def run(self):
        from .campaign import LOG_ROOT
        lock = LOG_ROOT.parent.parent / ".hill-overnight.lock"
        previous = {}
        def stop(signum, frame): self.stop_signal = signum
        with reservation(lock):
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous[sig] = signal.signal(sig, stop)
            try:
                if (self.output / "campaign.json").exists():
                    self.journal = Journal.load(self.output / "campaign.json")
                    self.verify_inputs()
                    if self.state["state"] not in self.TERMINAL:
                        self.journal.record("uncertain_restart_fail_closed", state="interrupted",
                            stop_reason="Existing nonterminal reservation is never automatically relaunched/retested")
                    return self.write_summary()
                self.journal = self.prepare()
                self.verify_inputs()
                self.reconcile_preexisting()
                first = self.state["start_epoch"] + 3600
                if time.time() < first:
                    self.wait_until(first, "reserve_first_hour_for_implementation_and_confirmation")
                while True:
                    self.check_deadline()
                    self.verify_inputs()
                    action, value = next_action(self.state, time.time())
                    if action == "wait_speed_window":
                        self.wait_until(value, "first_four_hours_reserved_for_stability")
                        continue
                    if action in ("stop_no_stability", "finalize"):
                        self.finalize()
                        break
                    record = self.run_screen(action, value)
                    self.journal.record("screen_reconciled", screens=[*self.state["screens"], record],
                                        current_screen=None, active_trainer_pid=None, state="assessing")
                return self.write_summary()
            except BaseException as error:
                if self.journal is None:
                    raise
                state = "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else \
                        "deadline_exhausted" if isinstance(error, TimeoutError) else "blocked"
                self.journal.record("terminal_stop", state=state, stop_reason=f"{type(error).__name__}: {error}")
                return self.write_summary()
            finally:
                for worker in self.workers:
                    worker.close()
                for sig, handler in previous.items(): signal.signal(sig, handler)

    def write_summary(self):
        return write_summary(self.output / "campaign.json")


def write_summary(journal_path):
    """Reproduce a report entirely from the durable journal and preserved rows."""
    journal = Journal.load(journal_path)
    state = journal.data
    inventories = []
    screens = deepcopy(state["screens"])
    current = state.get("current_screen")
    if current and not any(s.get("ordinal") == current.get("ordinal") for s in screens):
        screens.append(deepcopy(current))
    for screen in screens:
        run = screen.get("run")
        if not run and current and screen.get("ordinal") == current.get("ordinal"):
            run = state.get("active_run")
        expected = screen.get("expected_checkpoints", [*range(0, 500, 50), 499])
        saved = sorted(int(p.stem.removeprefix("model_")) for p in Path(run).glob("model_*.pt")
                       if re.fullmatch(r"model_(0|[1-9][0-9]*)", p.stem)) if run else []
        evaluated, fixed_evaluated, errors, checkpoint_rows = [], [], {}, {}
        for n in saved:
            row = {"checkpoint": n, "checkpoint_sha256": digest(Path(run) / f"model_{n}.pt"),
                   "official": None, "fixed_command": None, "raw_path": None}
            official = Path(run) / "workshop" / f"model_{n}.json"
            if official.exists():
                try:
                    row["official"] = official_summary(official, n)
                    evaluated.append(n)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    errors[str(n)] = str(error)
            assessment = screen.get("assessment")
            scored = Path(assessment) / f"model-{n}" / "evaluation.json" if assessment else None
            if scored and scored.exists():
                try:
                    actual = verify_worker_report(scored, checkpoint=Path(run) / f"model_{n}.pt", seeds=DEV)
                    row.update(fixed_command=actual["summary"], raw_path=actual["raw_path"])
                    fixed_evaluated.append(n)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    errors[f"fixed:{n}"] = str(error)
            checkpoint_rows[str(n)] = row
        launch_state = screen["state"]
        launch_journal = screen.get("launch_journal")
        evidence_unchanged = True
        if launch_journal and Path(launch_journal).exists():
            launch = read_report(launch_journal)
            launch_state = launch["state"]
            evidence_unchanged = (launch.get("source_files_unchanged") is True
                                  and launch.get("source_artifacts_unchanged") is True)
            plan = launch.get("plan", {})
            try:
                evidence_unchanged &= all(digest(p) == h for p, h in
                    {**plan.get("source_files", {}), **plan.get("source_config_files", {})}.items())
            except OSError:
                evidence_unchanged = False
        if saved == expected and evaluated == expected and fixed_evaluated == expected:
            try:
                if saved_checkpoints(run) != expected:
                    raise ValueError("Changed saved checkpoint inventory")
                campaign_snapshots([Path(run) / f"model_{n}.pt" for n in saved])
            except (ValueError, OSError, KeyError, TypeError) as error:
                errors["source_inventory"] = str(error)
        evidence_unchanged &= not errors
        eligible = [n for n in saved if checkpoint_rows[str(n)]["official"]
                    and checkpoint_rows[str(n)]["fixed_command"]
                    and checkpoint_rows[str(n)]["official"]["finishes"] == 5
                    and checkpoint_rows[str(n)]["fixed_command"]["finishes"] == 5]
        inventories.append({"run": run, "profile": screen["profile"], "state": launch_state,
            "expected": expected, "saved": saved, "evaluated": evaluated,
            "expected_count": len(expected), "saved_count": len(saved), "evaluated_count": len(evaluated),
            "pending": sorted(set(expected)-set(evaluated)), "evaluation_errors": errors,
            "official_inventory": {"expected": expected, "evaluated": evaluated,
                "pending": sorted(set(expected)-set(evaluated)), "expected_count": len(expected), "evaluated_count": len(evaluated)},
            "fixed_command_inventory": {"expected": expected, "evaluated": fixed_evaluated,
                "pending": sorted(set(expected)-set(fixed_evaluated)), "expected_count": len(expected), "evaluated_count": len(fixed_evaluated)},
            "evidence_unchanged": evidence_unchanged,
            "checkpoints": checkpoint_rows, "last_saved_checkpoint": saved[-1] if saved else None,
            "final_checkpoint": checkpoint_rows.get(str(expected[-1])),
            "valid_continuation_source": launch_state == "completed" and bool(expected) and saved == expected
                and evaluated == expected and fixed_evaluated == expected and bool(eligible) and evidence_unchanged})
    final = state.get("final")
    summary = {"schema": 1, "state": state["state"], "journal": str(Path(journal_path).resolve()),
        "journal_sha256": digest(journal_path), "stop_reason": state.get("stop_reason"),
        "start_epoch": state["start_epoch"], "training_deadline_epoch": state["training_deadline_epoch"],
        "hard_deadline_epoch": state["hard_deadline_epoch"], "new_screens": len(screens),
        "maximum_new_screens": state["maximum_new_screens"], "preexisting": state.get("preexisting_inventory"),
        "superseded_preexisting": state.get("superseded_preexisting_inventories", []),
        "E": state["original_source"], "R": state.get("anchor"), "B": state.get("best_candidate"),
        "best_complete_reliable_source": state.get("best_reliable"), "inventories": inventories,
        "heldouts_opened": bool(final and final.get("heldouts_opened")), "final": final,
        "phase": "explore", "automatic_promotion": False, "publication": False,
        "smoke_executed": False, "physical_robot_validation": False}
    if (final and final.get("heldouts_opened")) or state["state"] in ("complete", "rejected"):
        try:
            if not isinstance(final, dict):
                raise ValueError("Missing required final validation evidence")
            scores = {}
            labels = ("candidate", "anchor") if final.get("anchor_rows") or state["state"] in ("complete", "rejected") else ("candidate",)
            for label in labels:
                source = final["path"] if label == "candidate" else state["anchor"]["path"]
                actual = verify_worker_report(final[f"{label}_evaluation"], checkpoint=source, seeds=HELD,
                                              freeze=Path(journal_path).parent / "final-freeze.json")
                if actual["rows"] != final[f"{label}_rows"]:
                    raise ValueError("Final journal differs from preserved worker rows")
                scores[label] = actual["summary"]
            if not final["artifact_hashes"] or any(digest(p) != h for p, h in final["artifact_hashes"].items()):
                raise ValueError("Final artifacts changed after validation")
            summary["final_recomputed"] = {**scores, "artifacts_unchanged": True}
        except (ValueError, OSError, KeyError, TypeError) as error:
            journal.record("final_evidence_invalid", state="incomplete", stop_reason=f"Final evidence invalid: {error}")
            summary.update(state="incomplete", stop_reason=state["stop_reason"], journal_sha256=digest(journal_path),
                           final_recomputed={"error": str(error), "artifacts_unchanged": False})
    durable_json(Path(journal_path).parent / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summary", "units", "worker", "trainer-worker"))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "trainer-worker":
            if not args.execute or not args.request or not args.request_sha256 or not args.owner_pid or args.deadline is None:
                parser.error("trainer-worker requires the explicit bounded launcher request")
            execute_trainer_worker(args.request, args.request_sha256, args.owner_pid, args.deadline)
            return
        if args.action == "worker":
            if not args.execute or not args.request or not args.request_sha256:
                parser.error("worker requires explicit --execute and immutable --request/--request-sha256")
            execute_worker(args.request, args.request_sha256)
            return
        if not args.output:
            parser.error("--output is required")
        if args.action == "summary":
            result = write_summary(args.output / "campaign.json")
        elif args.action == "units":
            result = run_permitted_units(args.output, global_deadline=time.time()+600)
        elif not args.execute:
            if not args.authorization: parser.error("run requires --authorization")
            result = {"launching": False, "authorization": validate_authorization(args.authorization),
                      "output": str(args.output.resolve()), "no_smoke": True,
                      "profiles": ["S0", "S0", "P1 only after Gate S and T+4h"],
                      "hard_limit_new_screens": 5, "automatic_publication": False}
        else:
            if not args.authorization: parser.error("run --execute requires --authorization")
            result = Controller(args.authorization, args.output, args.source).run()
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("state") in ("blocked", "interrupted", "deadline_exhausted", "incomplete"):
            parser.exit(1)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Hill overnight refused: {error}\n")


if __name__ == "__main__":
    main()
