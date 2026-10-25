"""Five additional bounded Hill rounds; no runtime work without --execute.

Extends overnight's strict evidence, scoring, launch watchdog and final-validation
implementation. Never resumes an uncertain reservation or opens alternate heldouts.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from copy import deepcopy
from pathlib import Path

from . import overnight as base
from .evidence import digest, read_report, runtime_signature

MODULE = "microduck_challenges.hill_climb_1m5.five_rounds"
POLICY = "five-rounds-v1"
SEEDS = list(range(3, 8))
EARLIEST_SPEED = 1792923355.


def next_action(state, now):
    if state.get("current_screen"):
        raise ValueError("Unreconciled reservation; never duplicate a trainer launch")
    count = len(state["screens"])
    if count >= 5:
        return "finalize", None
    if now >= state["training_deadline_epoch"]:
        raise TimeoutError("Five-round training deadline exhausted")
    speed = count > 0 and state.get("anchor") and not state.get("p1_disabled", False)
    return ("P1" if speed and now >= EARLIEST_SPEED else "S0"), SEEDS[count]


def validate_authorization(path):
    """Retain schema 1 so existing scorers and strict input checks agree."""
    auth = base.validate_authorization(path)
    required = {"policy": POLICY, "requested_additional_rounds": 5, "training_seeds": SEEDS,
                "speed_not_before_epoch": EARLIEST_SPEED, "allowed_reward_profiles": ["S0", "P1"],
                "preexisting_hill_job": None, "superseded_preexisting_jobs": []}
    if any(auth.get(k) != v for k, v in required.items()):
        raise ValueError("Authorization does not match five-rounds-v1 policy")
    if not isinstance(auth.get("initial_source"), dict):
        raise ValueError("Missing pinned normal initial_source")
    return auth


def verified_initial(auth, source=None):
    """Verify the pinned normal source, prior journal and preserved raw evidence."""
    from .campaign import selected_source
    pin = auth["initial_source"]
    try:
        checkpoint = Path(pin["path"]).resolve()
        if (source is not None and Path(source).resolve() != checkpoint
                or digest(checkpoint) != pin["sha256"] or pin["profile"] not in ("S0", "P1")):
            raise ValueError("Initial normal source identity/profile differs from authorization")
        frozen = {}
        for key in ("prior_summary", "fixed_evaluation"):
            path = Path(pin[key])
            if not path.is_absolute() or digest(path) != pin[f"{key}_sha256"]:
                raise ValueError("Initial source prior/fixed evidence pin changed")
            frozen[str(path)] = digest(path)
        prior = read_report(pin["prior_summary"])
        prior_journal = base.Journal.load(prior["journal"])
        if (digest(prior["journal"]) != prior["journal_sha256"]
                or prior["state"] not in base.Controller.TERMINAL
                or prior_journal.data["state"] != prior["state"]):
            raise ValueError("Initial source prior authoritative journal changed")
        record = {k: deepcopy(pin[k]) for k in ("path", "sha256", "profile", "development", "official")}
        if (prior["best_complete_reliable_source"] != record
                or prior_journal.data["best_reliable"] != record):
            raise ValueError("Initial source is not the pinned prior best normal source")
        number = int(checkpoint.stem.removeprefix("model_"))
        matching = [i for i in prior["inventories"] if i["run"] == str(checkpoint.parent)]
        if len(matching) != 1:
            raise ValueError("Missing initial source complete prior inventory")
        inventory = matching[0]
        expected = inventory["expected"]
        if (not expected or number not in expected or not inventory["valid_continuation_source"]
                or not inventory["evidence_unchanged"] or inventory["saved"] != expected
                or inventory["official_inventory"]["evaluated"] != expected
                or inventory["fixed_command_inventory"]["evaluated"] != expected):
            raise ValueError("Initial source prior run was not completely assessed")
        fixed = base.verify_worker_report(pin["fixed_evaluation"], checkpoint=checkpoint, seeds=base.DEV)
        official = base.official_summary(checkpoint.parent / "workshop" / f"model_{number}.json", number)
        row = inventory["checkpoints"][str(number)]
        if (fixed["summary"]["finishes"] != 5 or official["finishes"] != 5
                or fixed["summary"] != pin["development"] or official != pin["official"]
                or row["checkpoint_sha256"] != pin["sha256"] or row["fixed_command"] != fixed["summary"]
                or row["official"] != official or row["raw_path"] != fixed["raw_path"]):
            raise ValueError("Initial source normal official/fixed raw 5/5 evidence differs")
        selected = selected_source(checkpoint, phase="explore")
        if (selected.get("partial_exception") is not None or selected["profile"] != pin["profile"]
                or selected["source_identity"]["sha256"] != pin["sha256"]):
            raise ValueError("Initial source must pass the normal strict source loader, without E's exception")
        frozen.update(selected["source_identity"]["artifacts"])
        frozen.update(fixed["artifact_hashes"])
        frozen[str(Path(prior["journal"]))] = prior["journal_sha256"]
        if any(digest(p) != h for p, h in frozen.items()):
            raise ValueError("Initial evidence changed during strict validation")
        return selected, record, frozen
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Missing/malformed pinned initial normal evidence") from error


class Journal(base.Journal):
    """Enforce retention BEFORE persistence, including a crash between screens."""
    def record(self, event, **changes):
        if event == "global_reliable_sources_updated":
            hashes = dict(self.data.get("official_report_hashes", {}))
            for record in changes["reliable_sources"]:
                checkpoint = Path(record["path"])
                official = checkpoint.parent / "workshop" / f"{checkpoint.stem}.json"
                actual = digest(official)
                if str(checkpoint) in hashes and hashes[str(checkpoint)] != actual:
                    raise ValueError("Retained official evidence changed before source adoption")
                hashes[str(checkpoint)] = actual
            changes["official_report_hashes"] = hashes
        original = self.data.get("anchor")
        if original and "anchor" in changes and changes["anchor"] != original:
            raise ValueError("The original confirmed R anchor is immutable")
        old, new = self.data.get("best_candidate"), changes.get("best_candidate")
        if new and event != "five_round_p1_disabled":
            if old and old["development"]["mean_s"] <= new["development"]["mean_s"]:
                changes["best_candidate"] = old
            retained = self.data.get("confirmed_candidates", [])
            if not any(r["path"] == new["path"] for r in retained):
                changes["confirmed_candidates"] = [*retained, new]
        if event == "screen_reserved":
            current = changes["current_screen"]
            count = len(self.data["screens"])
            if (self.data.get("current_screen") or count >= 5 or current["ordinal"] != count+1
                    or current["seed"] != SEEDS[count]):
                raise ValueError("Invalid or duplicate five-round reservation")
        super().record(event, **changes)


class Controller(base.Controller):
    """Five new attempts, with inherited launch/evaluation safety boundaries."""
    def prepare(self):
        auth = validate_authorization(self.authorization)
        if time.time()+300 >= auth["training_deadline_epoch"]:
            raise TimeoutError("Five-round authorization training deadline expired")
        selected, initial, evidence = verified_initial(auth, self.source)
        self.output.mkdir(parents=True, exist_ok=False)
        base.durable_json(self.output / "authorization.json", read_report(self.authorization))
        frozen = {**base.source_hashes(), **evidence, str(self.authorization): digest(self.authorization),
                  str(self.output / "authorization.json"): digest(self.output / "authorization.json")}
        inputs = {"schema": 1, "files": frozen, "runtime": runtime_signature(),
                  "source_identity": selected["source_identity"], "authorization": auth,
                  "repository": str(base.repository())}
        base.durable_json(self.output / "inputs.json", inputs)
        for name in ("authorization.json", "inputs.json"):
            os.chmod(self.output / name, 0o444)
        return Journal.create(self.output / "campaign.json", {
            "state": "prepared", "policy": POLICY, "start_epoch": auth["start_epoch"],
            "training_deadline_epoch": auth["training_deadline_epoch"],
            "authorized_training_deadline_epoch": auth["training_deadline_epoch"],
            "hard_deadline_epoch": auth["hard_deadline_epoch"], "maximum_new_screens": 5,
            "earliest_speed_epoch": EARLIEST_SPEED, "preexisting": None,
            "controller_pid": os.getpid(),
            "controller_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "controller_start_ticks": Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19],
            "inputs_path": str(self.output / "inputs.json"), "inputs_sha256": digest(self.output / "inputs.json"),
            "original_source": selected["source_identity"], "screens": [], "current_screen": None,
            "reliable_sources": [initial], "best_reliable": initial, "anchor": None,
            "official_report_hashes": {str(selected["path"]): digest(selected["result_path"])},
            "best_candidate": None, "confirmed_candidates": [], "p1_disabled": False,
            "final": None, "phase": "explore", "automatic_publication": False})

    def verify_inputs(self):
        super().verify_inputs()
        auth = validate_authorization(self.authorization)
        if (self.state.get("policy") != POLICY
                or self.state["start_epoch"] != auth["start_epoch"]
                or self.state["hard_deadline_epoch"] != auth["hard_deadline_epoch"]
                or self.state["training_deadline_epoch"] > auth["training_deadline_epoch"]):
            raise ValueError("Five-round clock/policy changed or deadline extended")

    def choose_source(self, profile):
        """Never reuse a prior 5/5 claim after its complete raw sweep changes."""
        from .campaign import inventory
        if profile not in ("S0", "P1"):
            raise ValueError("Five-round source profile must remain S0/P1")
        source = Path(super().choose_source(profile)).resolve()
        if profile == "P1":
            record = self.state.get("best_candidate") or self.state.get("anchor")
            if (not record or self.state.get("p1_disabled")
                    or digest(record["gate_path"]) != record["gate_sha256"]):
                raise ValueError("P1 requires this chosen source's unchanged own gate")
            base.require_stability_gate(record["gate_path"], source)
            return source
        record = self.state["best_reliable"]
        if record["profile"] not in ("S0", "P1"):
            raise ValueError("No E or unknown-profile continuation source")
        if str(source) == self.state["original_source"]["path"]:
            pin = validate_authorization(self.authorization)["initial_source"]
            fixed = base.verify_worker_report(pin["fixed_evaluation"], checkpoint=source, seeds=base.DEV)
        else:
            screens = [s for s in self.state["screens"] if s.get("run") == str(source.parent)]
            if len(screens) != 1 or screens[0]["state"] != "completed":
                raise ValueError("Source is not from a completed assessed screen; partial branches forbidden")
            screen = screens[0]
            ids = inventory(source.parent)
            if (ids != screen["expected_checkpoints"] or ids != screen["evaluated_checkpoints"]
                    or screen["pending_checkpoints"]):
                raise ValueError("Retained source has an incomplete checkpoint inventory")
            reports = {n: base.verify_worker_report(Path(screen["assessment"]) / f"model-{n}/evaluation.json",
                        checkpoint=source.parent / f"model_{n}.pt", seeds=base.DEV) for n in ids}
            fixed = reports[int(source.stem.removeprefix("model_"))]
        number = int(source.stem.removeprefix("model_"))
        official_path = source.parent / f"workshop/model_{number}.json"
        expected = self.state.get("official_report_hashes", {}).get(str(source))
        if not expected or digest(official_path) != expected:
            raise ValueError("Retained official evidence hash changed or was not pinned")
        official = base.official_summary(official_path, number)
        if (fixed["summary"] != record["development"] or fixed["summary"]["finishes"] != 5
                or official["finishes"] != 5 or official != record["official"]):
            raise ValueError("Retained source no longer has verified fixed and official normal 5/5")
        return source

    def launch_plan(self, profile, seed, source):
        from .campaign import make_plan
        action, expected_seed = next_action(self.state, time.time())
        if profile not in ("S0", "P1") or seed != expected_seed:
            raise ValueError("Invalid five-round profile, seed or exhausted reservations")
        if profile == "S0":
            return super().launch_plan(profile, seed, source)
        if action != "P1" or not self.state.get("anchor") or self.state.get("p1_disabled"):
            raise ValueError("P1 requires Gate S, the original speed window and a non-vetoed branch")
        record = self.state.get("best_candidate") or self.state["anchor"]
        if Path(record["path"]).resolve() != Path(source).resolve() or digest(source) != record["sha256"]:
            raise ValueError("P1 requires the chosen exact source's own Gate S")
        gate = record["gate_path"]
        if digest(gate) != record["gate_sha256"]:
            raise ValueError("P1 chosen source gate digest changed")
        base.require_stability_gate(gate, source)
        keys = ("MICRODUCK_HILL_STABILITY_GATE", "MICRODUCK_HILL_STABILITY_GATE_SHA256")
        previous = {k: os.environ.get(k) for k in keys}
        try:
            os.environ.update({keys[0]: gate, keys[1]: digest(gate)})
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
        for path in (self.output / "inputs.json", self.output / "authorization.json", self.authorization):
            plan["source_config_files"][str(path)] = digest(path)
        plan["environment"].update({keys[0]: gate, keys[1]: digest(gate)})
        plan["source_config_files"][gate] = digest(gate)
        algorithm = plan["target_algorithm"]
        if (plan["phase"] != "explore" or plan["requires_promotion"]
                or any(algorithm[k] != v for k, v in {"learning_rate": .00005, "entropy_coef": .01,
                       "clip_param": .2, "exploration_std_scale": 1.}.items())
                or plan["target_reward_weights"].get("horizontal_forward_speed") != 1.):
            raise ValueError("Five-round P1 exploration/reward controls changed")
        return plan

    def run_screen(self, profile, seed):
        if self.state.get("current_screen") or len(self.state["screens"]) >= 5:
            raise ValueError("Five-round reservation exhausted or uncertain")
        current = super().run_screen(profile, seed)
        if current["state"] == "branch_rollback":
            # The vetoed branch never reached base's reliable-source adoption.
            fallback = [r for r in self.state["reliable_sources"] if r["profile"] == "S0"]
            if not fallback:
                fallback = self.state["reliable_sources"][:1]
            confirmed = [r for r in self.state.get("confirmed_candidates", []) if r["profile"] == "S0"]
            best = min(fallback, key=lambda r: (r["development"]["mean_s"], r["path"]))
            stable = min(confirmed, key=lambda r: (r["development"]["mean_s"], r["path"])) if confirmed else self.state["anchor"]
            self.journal.record("five_round_p1_disabled", p1_disabled=True, reliable_sources=fallback,
                                best_reliable=best, best_candidate=stable, rollback_anchor=self.state["anchor"])
        return current

    def finalize(self):
        if (len(self.state["screens"]) != 5 or self.state.get("current_screen")
                or any(s["state"] not in ("completed", "branch_rollback") for s in self.state["screens"])):
            raise ValueError("Final validation requires all five reconciled work iterations")
        self.check_deadline()
        if self.state.get("anchor") and time.time() < self.state["training_deadline_epoch"]:
            self.journal.record("five_round_training_complete_validation_early",
                authorized_training_deadline_epoch=self.state["training_deadline_epoch"],
                training_deadline_epoch=time.time())
        return super().finalize()

    def run(self):
        from .campaign import LOG_ROOT, active_trainers
        previous = {}
        def stop(signum, frame): self.stop_signal = signum
        with base.reservation(LOG_ROOT.parent.parent / ".hill-overnight.lock"):
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous[sig] = signal.signal(sig, stop)
            try:
                if (self.output / "campaign.json").exists():
                    self.journal = Journal.load(self.output / "campaign.json")
                    self.verify_inputs()
                    if self.state["state"] not in self.TERMINAL:
                        self.journal.record("uncertain_restart_fail_closed", state="interrupted",
                            stop_reason="Existing reservation is never automatically relaunched/retested")
                    return self.write_summary()
                self.journal = self.prepare()
                self.verify_inputs()
                self.reconcile_preexisting()
                while True:
                    self.check_deadline()
                    self.verify_inputs()
                    if active_trainers():
                        raise ValueError("Competing trainer detected before five-round launch")
                    action, seed = next_action(self.state, time.time())
                    if action == "finalize":
                        self.finalize()
                        break
                    record = self.run_screen(action, seed)
                    self.journal.record("screen_reconciled", screens=[*self.state["screens"], record],
                                        current_screen=None, active_trainer_pid=None, state="assessing")
                return self.write_summary()
            except BaseException as error:
                if self.journal is None:
                    raise
                outcome = "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else \
                          "deadline_exhausted" if isinstance(error, TimeoutError) else "blocked"
                self.journal.record("terminal_stop", state=outcome, stop_reason=f"{type(error).__name__}: {error}")
                return self.write_summary()
            finally:
                for worker in self.workers:
                    worker.close()
                for sig, handler in previous.items(): signal.signal(sig, handler)

    def write_summary(self):
        return write_summary(self.output / "campaign.json")


def write_summary(path):
    summary = base.write_summary(path)
    state = Journal.load(path).data
    screens = list(state["screens"])
    current = state.get("current_screen")
    if current and not any(s.get("ordinal") == current.get("ordinal") for s in screens):
        screens.append(current)
    # A reservation survives uncertain restarts, but a durable failed launcher is
    # known failed work, not an unevaluated reservation or a completed iteration.
    for index, (screen, inventory) in enumerate(zip(screens, summary["inventories"], strict=True)):
        if screen["state"] == "reserved" and inventory["state"] in {
                "failed", "interrupted", "wall_budget_exhausted", "source_changed_not_promotable"}:
            screens[index] = {**screen, "state": "failed_or_incomplete"}
    completed = sum(s["state"] == "completed" for s in screens)
    rollback = sum(s["state"] == "branch_rollback" for s in screens)
    reserved = sum(s["state"] == "reserved" for s in screens)
    counts = {"attempts": len(screens), "completed": completed, "branch_rollback": rollback,
              "reserved_or_uncertain": reserved, "failed_or_incomplete": len(screens)-completed-rollback-reserved,
              "remaining_authorized": max(0, 5-len(screens))}
    summary.update(policy=POLICY, initial_source=state["original_source"], round_counts=counts,
                   five_work_iterations_complete=completed+rollback == 5,
                   requested_additional_rounds=5, training_seeds=SEEDS,
                   speed_not_before_epoch=EARLIEST_SPEED, p1_disabled=state.get("p1_disabled", False))
    base.durable_json(Path(path).parent / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summary"))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "summary":
            result = write_summary(args.output / "campaign.json")
        elif not args.authorization:
            parser.error("run requires --authorization")
        elif not args.execute:
            result = {"launching": False, "authorization": validate_authorization(args.authorization),
                      "output": str(args.output.resolve()), "no_smoke": True,
                      "hard_limit_new_screens": 5, "training_seeds": SEEDS,
                      "automatic_publication": False}
        else:
            result = Controller(args.authorization, args.output, args.source).run()
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("state") in ("blocked", "interrupted", "deadline_exhausted", "incomplete"):
            parser.exit(1)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Hill five-rounds refused: {error}\n")


if __name__ == "__main__":
    main()
