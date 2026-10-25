"""Read-only TensorBoard reward audit; never launches or changes a training run."""

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .experiments import REWARD_PROFILES, reward_overrides
from .promotion import digest
from .rewards import _ConfigLoader


ACCOUNTING = (
    "Episode_Reward values are weighted episode sums divided by configured maximum episode seconds, "
    "then averaged over reset reports. Train/mean_reward averages the last 100 completed returns. "
    "Different samples/windows: do not reconcile their sums or divide terms by the headline return. "
    "Unweighted estimates below retain the episode-duration normalization; they are not instantaneous values. "
    "Reweighted logged terms estimate fixed-data incentives, not retrained behaviour or whole-episode rescoring."
)


def scalar_series(events, tag):
    """Deduplicate identical samples, but reject conflicting/nonfinite evidence."""
    result = {}
    for event in events:
        if not math.isfinite(event.value):
            raise ValueError(f"Nonfinite scalar: {tag} at {event.step}")
        if event.step in result and result[event.step] != event.value:
            raise ValueError(f"Conflicting duplicate scalar: {tag} at {event.step}")
        result[event.step] = float(event.value)
    return result


def _summary(series, iteration, radius):
    values = [value for step, value in sorted(series.items()) if abs(step - iteration) <= radius]
    if not values:
        return None
    split = max(1, len(values) // 2)
    return {"at_iteration": series.get(iteration), "samples": len(values),
            "mean": statistics.mean(values), "min": min(values), "max": max(values),
            "first_half_mean": statistics.mean(values[:split]),
            "last_half_mean": statistics.mean(values[split:] or values[-1:])}


def audit_run(run, iteration, radius=10):
    """Snapshot existing logs/config; require the requested iteration, not nearest."""
    if type(iteration) is not int or iteration < 0 or type(radius) is not int or radius < 0:
        raise ValueError("Iteration and radius must be nonnegative integers")
    run = Path(run).resolve()
    config_paths = [run / "params/env.yaml", run / "params/agent.yaml"]
    before = {str(p): digest(p) for p in config_paths}
    cfg = yaml.load(config_paths[0].read_text(), Loader=_ConfigLoader)
    agent = yaml.load(config_paths[1].read_text(), Loader=_ConfigLoader)
    if agent.get("experiment_name") != "sprint_2m":
        raise ValueError("Audit only a saved Sprint run")
    if not list(run.glob("events.out.tfevents*")):
        raise ValueError("No TensorBoard events in the requested run")
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    tags = acc.Tags()["scalars"]
    series = {tag: scalar_series(acc.Scalars(tag), tag) for tag in tags
              if tag.startswith(("Episode_Reward/", "Train/", "Loss/", "Policy/", "Metrics/"))
              and not tag.endswith("/time")}
    total = series.get("Train/mean_reward", {})
    if iteration not in total:
        raise ValueError(f"Requested iteration {iteration} has no Train/mean_reward sample")
    terms = {}
    for name, term in cfg["rewards"].items():
        weight = term["weight"]
        if type(weight) not in (int, float) or not math.isfinite(weight):
            raise ValueError(f"Invalid saved weight: {name}")
        values = series.get(f"Episode_Reward/{name}", {})
        summary = _summary(values, iteration, radius)
        terms[name] = {"saved_weight": weight, "logged": summary,
                       "unweighted_normalized_mean_estimate": (
                           summary["mean"] / weight if summary and weight else None)}
    # A changing curriculum invalidates estimates based only on the startup weight.
    varying = set()
    for curriculum in cfg.get("curriculum", {}).values():
        params = curriculum.get("params", {})
        for name, endpoints in params.get("targets", {}).items():
            if endpoints[0] != endpoints[1] and params.get("initial_fraction", 0) < 1:
                varying.add(name)
        if params.get("weight_stages"):
            varying.add(params["reward_name"])
    for name in varying & terms.keys():
        terms[name]["unweighted_normalized_mean_estimate"] = None
    estimates = {}
    for profile in REWARD_PROFILES:
        changed = {}
        for name, target in reward_overrides(profile).items():
            term = terms.get(name)
            raw = term["unweighted_normalized_mean_estimate"] if term else None
            changed[name] = {"target_weight": target,
                             "estimated_logged_contribution": raw * target if raw is not None else None}
        estimates[profile] = changed
    positive = sum(max(0, term["logged"]["mean"]) for term in terms.values() if term["logged"])
    negative = sum(min(0, term["logged"]["mean"]) for term in terms.values() if term["logged"])
    for term in terms.values():
        value = term["logged"]["mean"] if term["logged"] else 0
        term["share_of_positive_logged_terms"] = max(0, value) / positive if positive else None
    if before != {str(p): digest(p) for p in config_paths}:
        raise ValueError("Saved recipe changed during the audit; retry the snapshot")
    return {"schema": 1, "run": str(run), "iteration": iteration, "radius": radius,
            "created_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": before,
            "reward_profile": agent["algorithm"].get("reward_profile", "A0"),
            "max_episode_seconds": cfg["episode_length_s"],
            "scale_rewards_by_dt": cfg.get("scale_rewards_by_dt"), "accounting": ACCOUNTING,
            "events_are_live_snapshot": True, "last_observed_iteration": max(total),
            "mean_reward": _summary(total, iteration, radius), "terms": terms,
            "missing_anchor_terms": [name for name, term in terms.items()
                                     if not term["logged"] or term["logged"]["at_iteration"] is None],
            "varying_weight_terms_not_rescored": sorted(varying),
            "positive_logged_sum": positive, "negative_logged_sum": negative,
            "fixed_logged_data_weight_estimates": estimates,
            "metrics": {tag: summary for tag, values in series.items()
                        if not tag.startswith("Episode_Reward/")
                        and (summary := _summary(values, iteration, radius)) is not None},
            "verdict": "Diagnostic only: a single reward value does not establish a plateau or racing gain."}


def markdown_report(report):
    lines = ["# Sprint reward audit", "", f"Run: `{report['run']}`",
             f"Iteration: **{report['iteration']}**, window radius: {report['radius']}", "",
             report["accounting"], "", "| Term | Saved weight | Anchor contribution | Window mean |",
             "|---|---:|---:|---:|"]
    for name, term in report["terms"].items():
        logged = term["logged"]
        anchor = logged["at_iteration"] if logged else None
        mean = logged["mean"] if logged else None
        lines.append(f"| {name} | {term['saved_weight']:.6g} | "
                     f"{format(anchor, '.6g') if anchor is not None else 'missing'} | "
                     f"{format(mean, '.6g') if mean is not None else 'missing'} |")
    lines.extend(["", f"Mean episode return at anchor: {report['mean_reward']['at_iteration']:.6g}",
                  "", report["verdict"], "", "## Explicit reward profiles", "",
                  "A0: unchanged control. A1: linear tracking -> 0. "
                  "B1: air time -> 3. B2: course speed -> 18. "
                  "C1: linear tracking -> 0 AND air time -> 3; compared with A1 only air time changes. "
                  "C2: C1 plus course speed -> 18; compared with C1 only speed weight changes.",
                  "These are unvalidated experiments; select by legal finish time and completion, not larger reward."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--output", type=Path, help="new report directory; never an existing run")
    args = parser.parse_args()
    try:
        report = audit_run(args.run, args.iteration, args.radius)
        if args.output:
            output = args.output.resolve()
            run = args.run.resolve()
            if output == run or run in output.parents:
                raise ValueError("Write reports separately from the source training run")
            output.mkdir(parents=True, exist_ok=False)
            (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            (output / "audit.md").write_text(markdown_report(report))
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError) as error:
        parser.exit(1, f"Sprint audit refused: {error}\n")


if __name__ == "__main__":
    main()
