"""Frozen-command race evidence and fail-closed promotion, separate from Workshop."""

import hashlib
import json
import math
import statistics
import re
from pathlib import Path


def read_report(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=unique_keys,
                      parse_constant=invalid_constant)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def saved_checkpoints(run):
    """Detect missing checkpoints using the saved budget and checkpoint cadence."""
    import yaml
    from .learning import ConfigLoader as _ConfigLoader

    run = Path(run).resolve()
    agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=_ConfigLoader)
    budget, interval = agent["max_iterations"], agent["save_interval"]
    if type(budget) is not int or type(interval) is not int or min(budget, interval) <= 0:
        raise ValueError("Invalid saved checkpoint cadence")
    expected = sorted({*range(0, budget, interval), budget - 1})
    paths = list(run.glob("model_*.pt"))
    if any(not re.fullmatch(r"model_(0|[1-9][0-9]*)\.pt", p.name) for p in paths):
        raise ValueError("Malformed checkpoint identifier")
    actual = sorted(int(p.stem.removeprefix("model_")) for p in paths)
    if actual != expected:
        raise ValueError("Incomplete checkpoint inventory for the saved recipe")
    return actual


def runtime_signature():
    """Invalidate evidence when the event, evaluator or key dependencies change."""
    from importlib.metadata import version
    import arena

    root = Path(arena.__file__).parent
    files = ("runner.py", "actuator.py", "course.py", "manifest.py", "constants.py", "events/hill-climb-1m5.toml")
    return {"files": {str(root / name): digest(root / name) for name in files},
            "versions": {name: version(name) for name in
                         ("torch", "mujoco", "mjlab", "mjlab-microduck", "microduck-arena", "onnxruntime")}}


def summarize(rows, seeds, command):
    """Validate complete seed coverage and compute the actual 60 s failure score."""
    if (not isinstance(rows, list) or len(rows) != len(seeds)
            or any(type(row.get("seed")) is not int for row in rows)
            or {row["seed"] for row in rows} != set(seeds)):
        raise ValueError("Evidence must contain every expected seed exactly once")
    validate_command(command)
    times = []
    finishes = 0
    for row in rows:
        vx = row.get("command_vx")
        if (type(vx) not in (int, float) or not math.isfinite(vx)
                or not math.isclose(vx, command, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Every race must use the frozen command")
        if row.get("status") not in ("finish", "fall", "timeout"):
            raise ValueError("Invalid race status in evidence")
        value = row.get("time_s")
        if row["status"] == "finish":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 60:
                raise ValueError("Invalid finish time in evidence")
            finishes += 1
            times.append(value)
        else:
            times.append(60.0)
    return {"finishes": finishes, "total": len(rows), "mean_s": statistics.mean(times)}


def promotion_decision(report, gate):
    """Recompute a decision from seed rows; never trust a saved approved flag."""
    if report.get("schema") != 1 or report.get("gate") != gate or report.get("status") != "complete":
        raise ValueError("Promotion criteria differ from the frozen assessment")
    command = gate["command_vx"]
    validate_command(command)
    count = gate["neighbor_checkpoints"]
    first, heldout_count = gate["heldout_first_seed"], gate["heldout_seed_count"]
    if (type(count) is not int or count < 3 or type(first) is not int or first <= 5
            or type(heldout_count) is not int or heldout_count < 20
            or type(command) not in (int, float) or not math.isfinite(command)
            or not 0.15 <= command <= 1.0):
        raise ValueError("Unsafe promotion criteria")
    ids = report["checkpoints"]
    candidate = report["candidate_checkpoint"]
    if (not isinstance(ids, list) or not ids
            or any(type(number) is not int or number < 0 for number in ids)
            or ids != sorted(set(ids)) or type(candidate) is not int or candidate not in ids
            or set(report["screen"]) != {str(number) for number in ids}):
        raise ValueError("Incomplete checkpoint inventory or screen evidence")
    screens = {number: summarize(report["screen"][str(number)], range(1, 6), command)
               for number in ids}
    windows = [ids[i:i + count] for i in range(len(ids) - count + 1)]
    stable = [window for window in windows if candidate in window
              and all(screens[number]["finishes"] == 5 for number in window)]
    heldout_seeds = range(first, first + heldout_count)
    candidate_extra = summarize(report["candidate_heldout"], heldout_seeds, command)
    incumbent = report["incumbent_rows"]
    # A single combined list makes missing/duplicated incumbent coverage visible.
    summarize(incumbent, [*range(1, 6), *heldout_seeds], command)
    incumbent_screen = summarize([r for r in incumbent if r["seed"] <= 5], range(1, 6), command)
    incumbent_extra = summarize([r for r in incumbent if r["seed"] >= first], heldout_seeds, command)
    eligible = {n for window in windows if all(screens[k]["finishes"] == 5 for k in window) for n in window}
    if eligible and candidate != min(eligible, key=lambda n: (screens[n]["mean_s"], n)):
        raise ValueError("Candidate was not the frozen fastest stable development selection")
    reasons = []
    if not stable:
        reasons.append("candidate has no complete three-checkpoint stable neighborhood")
    if candidate_extra["finishes"] != heldout_count:
        reasons.append("candidate failed a held-out start")
    if screens[candidate]["mean_s"] >= incumbent_screen["mean_s"] - 1e-9:
        reasons.append("candidate is not faster on the frozen original seeds")
    if candidate_extra["mean_s"] > incumbent_extra["mean_s"] + 1e-9:
        reasons.append("candidate is worse on held-out failure-penalized mean")
    return {"approved": not reasons, "reasons": reasons,
            "stable_neighbors": stable[0] if stable else [],
            "candidate_screen": screens[candidate], "candidate_heldout": candidate_extra,
            "incumbent_screen": incumbent_screen, "incumbent_heldout": incumbent_extra,
            "automatic_publication": False}


def verify_raw_rows(report, folder):
    """Bind recomputed decisions to persisted races rather than mutable summaries."""
    folder = Path(folder).resolve()
    def check(label, partition, expected):
        path = folder / f"{label}-{partition}.jsonl"
        if str(path) not in report["artifact_hashes"]:
            raise ValueError("Missing preserved raw race rows")
        actual = [json.loads(line) for line in path.read_text().splitlines()]
        if actual != expected or digest(path) != report["artifact_hashes"][str(path)]:
            raise ValueError("Assessment disagrees with preserved raw race rows")
    for number in report["checkpoints"]:
        check(f"model_{number}", "screen", report["screen"][str(number)])
    check(f"model_{report['candidate_checkpoint']}", "heldout", report["candidate_heldout"])
    first = report["gate"]["heldout_first_seed"]
    check("incumbent", "screen", [r for r in report["incumbent_rows"] if r["seed"] <= 5])
    check("incumbent", "heldout", [r for r in report["incumbent_rows"] if r["seed"] >= first])


def require_promotion(report_path, source, gate):
    """Bind a passing assessment to the exact source and unchanged artifacts."""
    if not report_path:
        raise ValueError("Exploitation requires an approved promotion report")
    report = read_report(report_path)
    if report.get("status") != "complete":
        raise ValueError("Exploitation requires a complete approved promotion report")
    if Path(report["candidate_path"]).resolve() != Path(source).resolve():
        raise ValueError("Promotion report does not approve this exact source checkpoint")
    if Path(source).name != f"model_{report['candidate_checkpoint']}.pt":
        raise ValueError("Promotion checkpoint identifier does not match the source artifact")
    decision = promotion_decision(report, gate)
    verify_raw_rows(report, Path(report_path).resolve().parent)
    if not decision["approved"]:
        raise ValueError("Exploitation requires an approved promotion report: "
                         + "; ".join(decision["reasons"]))
    if report.get("runtime") != runtime_signature():
        raise ValueError("Assessment runtime changed or was not recorded")
    run = Path(source).resolve().parent
    actual_ids = saved_checkpoints(run)
    if actual_ids != report["checkpoints"]:
        raise ValueError("Checkpoint inventory changed since assessment")
    from .campaign import inventory
    inventory(run)
    artifacts = report["artifact_hashes"]
    required = {str(Path(report["incumbent_path"]).resolve()),
                *(str((run / f"model_{n}.pt").resolve()) for n in actual_ids),
                str(run / "params/env.yaml"), str(run / "params/agent.yaml"),
                str(run / "provenance.json")}
    incumbent_run = Path(report["incumbent_path"]).resolve().parent
    required.update(str(run / "workshop" / f"model_{n}.json") for n in actual_ids)
    required.update(str(incumbent_run / name) for name in ("params/env.yaml", "params/agent.yaml", "provenance.json"))
    for label, exported in report["exports"].items():
        onnx = Path(report_path).resolve().parent / f"{label}.onnx"
        required.add(str(onnx))
        checkpoint = Path(report["incumbent_path"]) if label == "incumbent" else run / f"{label}.pt"
        if exported["checkpoint_sha256"] != digest(checkpoint) or exported["onnx_sha256"] != digest(onnx):
            raise ValueError("Assessment exports changed since selection")
    if not required <= artifacts.keys():
        raise ValueError("Missing source artifact hashes in promotion report")
    for path, expected in artifacts.items():
        if digest(path) != expected:
            raise ValueError(f"Assessment artifact changed: {path}")
    return report


def validate_command(command):
    if (type(command) not in (int, float) or not math.isfinite(command)
            or not .15 <= command <= 1.0 or not math.isclose(command * 20, round(command * 20), abs_tol=1e-8)):
        raise ValueError("Frozen Hill command must be on the 0.15–1.00 m/s sweep in 0.05 steps")
    return command
