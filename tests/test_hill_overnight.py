"""Durable Hill supervision contracts; all process/simulator boundaries are mocked."""
import inspect
from types import SimpleNamespace

import pytest

from microduck_challenges.hill_climb_1m5 import campaign


def test_supported_no_smoke_launch_keeps_journal_and_deadline(tmp_path, monkeypatch):
    assert "no_smoke" in inspect.signature(campaign.run_bounded).parameters
    plan = campaign.make_plan(stage="bootstrap", iterations=1, seed=2)
    plan["max_wall_minutes"] = 50
    output = tmp_path / "screen"
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "smoke", lambda *a, **kw: pytest.fail("smoke forbidden"))
    checks, children = [], []
    monkeypatch.setattr(campaign, "run_permitted_units", lambda *a, **kw: checks.append(kw) or {"exit_code": 0})
    class Child:
        pid, returncode = 1234567, 0
        def wait(self, **kw): return 0
        def poll(self): return 0
    def spawn(*a, **kw):
        assert a[0][1:4] == ["-m", "microduck_challenges.hill_climb_1m5.overnight", "trainer-worker"]
        children.append(kw)
        return Child()
    monkeypatch.setattr(campaign.subprocess, "Popen", spawn)
    import time
    result = campaign.run_bounded(plan, output, no_smoke=True, global_deadline=time.time() + 600)
    assert result["state"] == "completed" and len(checks) == len(children) == 1
    assert result["preflight"]["no_smoke"] is True
    assert children[0]["env"]["MICRODUCK_HILL_CAMPAIGN_PLAN_DIGEST"] == campaign.plan_digest(plan)
    assert result["source_files_unchanged"] and result["source_artifacts_unchanged"]


def test_no_smoke_refuses_expired_deadline_without_preflight_or_child(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **kw: pytest.fail("child forbidden"))
    with pytest.raises(ValueError, match="deadline"):
        campaign.run_bounded({}, tmp_path / "screen", no_smoke=True, global_deadline=1.)
    assert not (tmp_path / "screen").exists()


@pytest.fixture
def stable_evidence(tmp_path, monkeypatch):
    """Synthetic CPU evidence; only strict model reconstruction is mocked."""
    import json
    import torch
    import yaml
    from microduck_challenges.hill_climb_1m5 import evidence, overnight
    run, out = tmp_path / "saved", tmp_path / "gate"
    (run / "params").mkdir(parents=True)
    (run / "workshop").mkdir()
    out.mkdir()
    (run / "params/agent.yaml").write_text(yaml.safe_dump({"max_iterations": 200, "save_interval": 50, "algorithm": {}}))
    (run / "params/env.yaml").write_text("scene: {}\n")
    (run / "provenance.json").write_text("{}")
    ids = [0, 50, 100, 150, 199]
    rows = lambda seeds, n=0: [{"seed": s, "status": "finish", "time_s": 5 + n / 1000, "command_vx": .75} for s in seeds]
    report = {"schema": 1, "kind": "hill-stability", "command_vx": .75,
              "checkpoints": ids, "candidate_checkpoint": 0, "candidate_path": str(run / "model_0.pt"),
              "screen": {str(n): rows(range(1, 6), n) for n in ids},
              "candidate_calibration": rows(range(26, 31)), "raw_rows": {}, "exports": {},
              "artifact_hashes": {}, "runtime": evidence.runtime_signature(), "campaign_evidence": {}}
    for n in ids:
        checkpoint = run / f"model_{n}.pt"
        torch.save({}, checkpoint)
        official = {"checkpoint": n, "command_vx": .75, "seeds_total": 5, "seeds_finished": 5,
                    "score": 5., "per_seed": rows(range(1, 6))}
        (run / "workshop" / f"model_{n}.json").write_text(json.dumps(official))
    for label, actual, number in [(f"screen:{n}", report["screen"][str(n)], n) for n in ids] + [("calibration", report["candidate_calibration"], 0)]:
        raw, onnx = out / f"{label.replace(':', '-')}.jsonl", out / f"{label.replace(':', '-')}.onnx"
        raw.write_text("".join(json.dumps(r) + "\n" for r in actual))
        onnx.write_bytes(b"synthetic mock export")
        report["raw_rows"][label] = str(raw)
        report["exports"][label] = {"onnx_path": str(onnx), "onnx_sha256": evidence.digest(onnx),
                                   "checkpoint_sha256": evidence.digest(run / f"model_{number}.pt"),
                                   "source_agent_sha256": evidence.digest(run / "params/agent.yaml"),
                                   "input_shape": [1, 61], "output_shape": [1, 14], "parity_cases": 8}
    report["artifact_hashes"] = {str(p): evidence.digest(p) for p in [*run.rglob("*"), *out.glob("*")] if p.is_file()}
    campaign.write_json(out / "stability.json", report)
    monkeypatch.setattr(campaign, "selected_source", lambda p, **kw: {"path": p, "partial_exception": None})
    return out / "stability.json", report


def test_stability_gate_recomputes_unfiltered_neighbors_and_calibration(stable_evidence):
    from microduck_challenges.hill_climb_1m5 import overnight
    path, report = stable_evidence
    assert callable(getattr(overnight, "require_stability_gate", None))
    actual = overnight.require_stability_gate(path, report["candidate_path"])
    assert actual["candidate_path"] == report["candidate_path"]
    assert actual["verified"]["neighbors"] == [0, 50, 100]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "command", "tamper", "runtime", "inventory", "flags", "filtered", "official", "campaign", "export"])
def test_stability_gate_fails_closed_on_incomplete_or_changed_evidence(stable_evidence, corruption):
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    path, report = stable_evidence
    assert callable(getattr(overnight, "require_stability_gate", None))
    if corruption == "missing": report["screen"]["50"].pop()
    elif corruption == "duplicate": report["candidate_calibration"][1]["seed"] = 26
    elif corruption == "command": report["screen"]["50"][0]["command_vx"] = .8
    elif corruption == "tamper":
        from pathlib import Path
        Path(report["raw_rows"]["calibration"]).write_text("tampered")
    elif corruption == "runtime": report["runtime"] = {}
    elif corruption == "inventory": report["checkpoints"].remove(50)
    elif corruption in ("flags", "filtered"):
        report["approved"] = report["stable"] = True
        for n in (50, 150): report["screen"][str(n)][0]["status"] = "fall"
    elif corruption == "official":
        from pathlib import Path
        official = Path(report["candidate_path"]).parent / "workshop/model_0.json"
        data = evidence.read_report(official)
        data["per_seed"][0]["status"] = "fall"
        data.update(seeds_finished=4, score=16.)
        campaign.write_json(official, data)
        report["artifact_hashes"][str(official)] = evidence.digest(official)
    elif corruption == "campaign": report["campaign_evidence"] = {"invented": {}}
    elif corruption == "export": report["exports"]["screen:0"]["checkpoint_sha256"] = "0" * 64
    campaign.write_json(path, report)
    with pytest.raises((ValueError, OSError)):
        overnight.require_stability_gate(path, report["candidate_path"])


def test_authorization_deadlines_and_duplicate_controller_reservation(tmp_path):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "validate_authorization", None))
    from pathlib import Path
    source = Path("/home/ctcheung/Downloads/microduck-challenges/logs/campaigns/hill-stability-20261025-011555/authorization.json")
    if not source.exists(): pytest.skip("local authorization absent")
    auth = overnight.validate_authorization(source)
    assert auth["training_deadline_epoch"] == 1792930555.
    assert auth["hard_deadline_epoch"] == 1792937755.
    lock = tmp_path / "reservation.lock"
    with overnight.reservation(lock):
        with pytest.raises(ValueError, match="controller"):
            with overnight.reservation(lock): pass


def test_final_freeze_forbids_alternative_and_early_heldouts(tmp_path):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "freeze_final", None))
    one, two = tmp_path / "model_1.pt", tmp_path / "model_2.pt"
    one.write_bytes(b"one"); two.write_bytes(b"two")
    state = {"anchor": None, "final": None}
    with pytest.raises(ValueError, match="anchor"):
        overnight.freeze_final(state, one)
    state["anchor"] = {"path": str(one)}
    overnight.freeze_final(state, one)
    state["final"]["heldouts_opened"] = True
    with pytest.raises(ValueError, match="frozen"):
        overnight.freeze_final(state, two)


def test_subfive_speed_branch_requests_actual_owned_trainer_stop(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "consecutive_failures", None))
    rows = lambda fail=False: [{"seed": s, "status": "fall" if fail and s == 1 else "finish", "time_s": 5., "command_vx": .75} for s in range(1, 6)]
    assert not overnight.consecutive_failures([0, 50, 100], {"0": rows(True), "100": rows(True)})
    assert overnight.consecutive_failures([0, 50, 100], {"0": rows(True), "50": rows(True)})
    plan = campaign.make_plan(stage="bootstrap", iterations=1, seed=2)
    plan["max_wall_minutes"] = 50
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "run_permitted_units", lambda *a, **kw: {})
    child = SimpleNamespace(pid=99999, returncode=-15, poll=lambda: None)
    stopped = []
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(campaign, "_stop_owned_process", lambda p, deadline=None: stopped.append(p))
    import time
    result = campaign._run_bounded(plan, tmp_path / "screen", no_smoke=True,
                                   global_deadline=time.time()+600,
                                   monitor=lambda p, j: "two consecutive sub-5/5 checkpoints")
    assert result["state"] == "branch_rollback" and stopped == [child]


def test_policy_never_accelerates_on_clock_and_stops_two_failed_s0s():
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "next_action", None))
    state = {"start_epoch": 100., "training_deadline_epoch": 21700.,
             "screens": [], "anchor": None, "best_reliable": None}
    assert overnight.next_action(state, 14500.) == ("S0", 1)
    state["screens"] = [{"profile": "S0"}, {"profile": "S0"}]
    assert overnight.next_action(state, 14500.)[0] == "stop_no_stability"
    state["anchor"] = {"path": "R"}
    assert overnight.next_action(state, 14499.)[0] == "wait_speed_window"
    assert overnight.next_action(state, 14500.) == ("P1", 1)
    state["screens"].append({"profile": "P1", "state": "branch_rollback"})
    assert overnight.next_action(state, 14501.)[0] == "finalize"
    assert overnight.next_action(state, 21700.)[0] == "finalize"


def test_authoritative_journal_is_durable_and_reservations_never_relaunch(tmp_path):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "Journal", None))
    path = tmp_path / "campaign.json"
    journal = overnight.Journal.create(path, {"state": "prepared", "screens": []})
    journal.record("screen_reserved", screen={"ordinal": 1, "state": "reserved"})
    again = overnight.Journal.load(path)
    assert again.data["revision"] == 1 and again.data["screen"]["state"] == "reserved"
    assert again.data["events"][0]["event"] == "screen_reserved"
    with pytest.raises(ValueError, match="exists"):
        overnight.Journal.create(path, {})


def test_scoring_worker_deadline_kills_only_its_owned_group(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "OwnedEvaluation", None))
    import time
    monkeypatch.setattr(overnight.subprocess, "Popen", lambda *a, **kw: pytest.fail("expired worker launched"))
    with pytest.raises(ValueError, match="deadline"):
        overnight.OwnedEvaluation(tmp_path / "model_0.pt", tmp_path / "eval", DEV_SEEDS := [1,2,3,4,5], deadline=1.)
    child = SimpleNamespace(pid=987654, returncode=None, poll=lambda: None)
    captured, stopped = [], []
    monkeypatch.setattr(overnight.subprocess, "Popen", lambda *a, **kw: captured.append((a,kw)) or child)
    monkeypatch.setattr(campaign, "_stop_owned_process", lambda p, deadline=None: stopped.append(p))
    worker = overnight.OwnedEvaluation(tmp_path / "model_0.pt", tmp_path / "eval", DEV_SEEDS, deadline=time.time()+60)
    assert captured[0][1]["start_new_session"] is True
    worker.deadline = 0
    with pytest.raises(ValueError, match="deadline"):
        worker.poll()
    assert stopped == [child]


def test_controller_two_mocked_failed_s0_screens_end_honestly_and_persist_summary(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight, "Controller", None))
    import time
    root = tmp_path / "controller"
    root.mkdir()
    state = {"state": "prepared", "start_epoch": time.time()-3600,
             "training_deadline_epoch": time.time()+18000, "hard_deadline_epoch": time.time()+25000,
             "screens": [], "anchor": None, "best_reliable": None, "best_candidate": None,
             "final": None, "original_source": {"path": "E", "sha256": "pinned"}, "maximum_new_screens": 5}
    controller = overnight.Controller(tmp_path / "auth.json", root)
    monkeypatch.setattr(controller, "prepare", lambda: overnight.Journal.create(root / "campaign.json", state))
    monkeypatch.setattr(controller, "verify_inputs", lambda: None)
    monkeypatch.setattr(controller, "reconcile_preexisting", lambda: None)
    monkeypatch.setattr(campaign, "LOG_ROOT", tmp_path / "logs/rsl_rl/hill_climb_1m5")
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    calls = []
    def run_screen(profile, seed):
        calls.append((profile, seed))
        return {"profile": profile, "seed": seed, "state": "completed", "reliable": []}
    monkeypatch.setattr(controller, "run_screen", run_screen)
    result = controller.run()
    assert calls == [("S0", 1), ("S0", 2)]
    assert result["state"] == "stability_unmet" and result["heldouts_opened"] is False
    assert overnight.Journal.load(root / "campaign.json").data["state"] == "stability_unmet"
    assert (root / "summary.json").exists()


def test_unit_preflight_rejects_source_change_during_mocked_pytest(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight
    snapshots = iter([{"unit.py": "before"}, {"unit.py": "after"}])
    monkeypatch.setattr(overnight, "source_hashes", lambda: next(snapshots))
    monkeypatch.setattr(overnight.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    import time
    with pytest.raises(ValueError, match="changed"):
        overnight.run_permitted_units(tmp_path, global_deadline=time.time()+600)
    assert not (tmp_path / "units.json").exists()


def test_best_confirmed_source_not_first_anchor_is_used_for_speed(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    import time
    root = tmp_path / "controller"
    first, best = tmp_path / "R.pt", tmp_path / "B.pt"
    first.write_bytes(b"R"); best.write_bytes(b"B")
    controller = overnight.Controller(tmp_path / "auth.json", root)
    controller.journal = overnight.Journal.create(root / "campaign.json", {
        "anchor": {"path": str(first), "sha256": evidence.digest(first), "profile": "S0"},
        "best_candidate": {"path": str(best), "sha256": evidence.digest(best), "profile": "S0"}})
    monkeypatch.setattr(campaign, "selected_source", lambda p, **kw: {"path": p, "partial_exception": None})
    assert str(controller.choose_source("P1")) == str(best)
    assert controller.state["anchor"]["path"] == str(first)


def test_monitor_rejects_p1_even_when_backlog_reports_failures_after_trainer_exit(tmp_path):
    from microduck_challenges.hill_climb_1m5 import overnight
    assert callable(getattr(overnight.ScreenMonitor, "rollback_reason", None))
    monitor = overnight.ScreenMonitor(None, {"reward_profile": "P1"}, tmp_path / "assessment")
    monitor.run = tmp_path / "run"
    monitor.run.mkdir()
    for n in (0, 50): (monitor.run / f"model_{n}.pt").touch()
    rows = [{"seed": s, "status": "fall" if s == 1 else "finish", "time_s": 5., "command_vx": .75} for s in range(1,6)]
    monitor.records = {"0": {"rows": rows}, "50": {"rows": rows}}
    assert "two consecutive" in monitor.rollback_reason()


def test_child_boundary_error_is_recorded_failed_and_owned_child_is_reaped(tmp_path, monkeypatch):
    plan = campaign.make_plan(stage="bootstrap", iterations=1, seed=2)
    plan["max_wall_minutes"] = 50
    monkeypatch.setattr(campaign, "active_trainers", lambda: [])
    monkeypatch.setattr(campaign, "run_permitted_units", lambda *a, **kw: {})
    child = SimpleNamespace(pid=99999, returncode=-15, poll=lambda: None)
    stopped = []
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(campaign, "_stop_owned_process", lambda p, deadline=None: stopped.append(p))
    def failed_monitor(*a): raise ValueError("owned monitor evidence error")
    import time
    with pytest.raises(ValueError, match="evidence error"):
        campaign._run_bounded(plan, tmp_path / "screen", no_smoke=True,
                              global_deadline=time.time()+600, monitor=failed_monitor)
    assert stopped == [child]
    assert campaign.read_report(tmp_path / "screen/run.json")["state"] == "failed"


@pytest.fixture
def worker_report_factory(tmp_path):
    """Persist synthetic worker boundary output; the real verifier reads it."""
    import json
    import os
    import time
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    def create(checkpoint, output, seeds, *, freeze=None, partition="development", rows=None):
        checkpoint, output = Path(checkpoint), Path(output)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.joinpath("params").mkdir(exist_ok=True)
        for name in ("params/agent.yaml", "params/env.yaml", "provenance.json"):
            path = checkpoint.parent / name
            if not path.exists(): path.write_text("{}")
        auth = tmp_path / "worker-authorization.json"
        if not auth.exists(): auth.write_text("{}")
        request = {"checkpoint": str(checkpoint), "seeds": list(seeds), "command_vx": .75,
                   "deadline_epoch": time.time()+600, "authorization": str(auth),
                   "freeze": str(freeze) if freeze else None,
                   "freeze_sha256": evidence.digest(freeze) if freeze else None,
                   "partition": partition, "owner_pid": os.getpid()}
        overnight.durable_json(output / "request.json", request)
        rows = rows or [{"seed": s, "status": "finish", "time_s": 3., "command_vx": .75} for s in seeds]
        raw, onnx = output / "rows.jsonl", output / "policy.onnx"
        raw.write_text("".join(json.dumps(row)+"\n" for row in rows))
        onnx.write_bytes(b"unit-only mock export boundary")
        artifacts = [checkpoint, checkpoint.parent / "params/agent.yaml", checkpoint.parent / "params/env.yaml",
                     checkpoint.parent / "provenance.json", raw, onnx, output / "request.json", auth]
        if freeze: artifacts.append(Path(freeze))
        report = {"schema": 1, "request": request, "rows": rows, "summary": evidence.summarize(rows, seeds, .75),
                  "raw_path": str(raw), "runtime": evidence.runtime_signature(),
                  "export": {"onnx_path": str(onnx), "onnx_sha256": evidence.digest(onnx),
                    "checkpoint_sha256": evidence.digest(checkpoint), "source_agent_sha256": evidence.digest(checkpoint.parent / "params/agent.yaml"),
                    "input_shape": [1,61], "output_shape": [1,14], "parity_cases": 8},
                  "artifact_hashes": {str(p): evidence.digest(p) for p in artifacts}}
        overnight.durable_json(output / "evaluation.json", report)
        return report
    return create


@pytest.mark.parametrize("corruption", [None, "candidate_raw", "anchor_raw", "missing_artifacts"])
def test_final_validation_waits_for_reserve_and_scores_one_frozen_candidate_against_r(tmp_path, monkeypatch, worker_report_factory, corruption):
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    import time
    r, b = tmp_path / "R.pt", tmp_path / "B.pt"
    r.write_bytes(b"R"); b.write_bytes(b"B")
    anchor = {"path": str(r), "sha256": evidence.digest(r), "profile": "S0", "development": {"mean_s": 3.}}
    candidate = {"path": str(b), "sha256": evidence.digest(b), "profile": "P1", "development": {"mean_s": 2.}}
    controller = overnight.Controller(tmp_path / "authorization.json", tmp_path / "controller")
    cutoff = time.time()+300
    controller.journal = overnight.Journal.create(controller.output / "campaign.json", {
        "state": "assessing", "anchor": anchor, "best_candidate": candidate, "final": None,
        "training_deadline_epoch": cutoff, "hard_deadline_epoch": cutoff+7200})
    monkeypatch.setattr(controller, "verify_inputs", lambda: None)
    waited, races = [], []
    monkeypatch.setattr(controller, "wait_until", lambda epoch, reason: waited.append(epoch))
    def evaluate(path, out, seeds, **kwargs):
        assert controller.state["final"]["heldouts_opened"]
        frozen = evidence.read_report(kwargs["freeze"])
        assert frozen["candidate"]["path"] == str(b) and list(seeds) == list(range(6,26))
        races.append(str(path))
        if str(path) == str(r):
            assert controller.state["final"]["candidate_rows"] is not None
        rows = [{"seed": s, "status": "fall" if str(path) == str(b) and s == 6 else "finish",
                 "time_s": 2. if str(path) == str(b) else 3., "command_vx": .75} for s in seeds]
        report = worker_report_factory(path, out, seeds, rows=rows, **kwargs)
        if str(path) == str(r) and corruption:
            from pathlib import Path
            target = controller.output / ("final-candidate" if corruption == "candidate_raw" else "final-anchor")
            if corruption == "missing_artifacts":
                report["artifact_hashes"] = {}
                overnight.durable_json(target / "evaluation.json", report)
            else:
                Path(target / "rows.jsonl").write_text("tampered after paired race")
        return report
    monkeypatch.setattr(controller, "evaluate", evaluate)
    if corruption:
        with pytest.raises((ValueError, OSError)):
            controller.finalize()
        assert controller.state["state"] not in ("complete", "rejected")
        return
    controller.finalize()
    assert waited == [cutoff]
    assert races == [str(b), str(r)]
    assert controller.state["state"] == "rejected"
    assert controller.state["final"]["candidate_score"]["finishes"] == 19
    assert controller.state["anchor"] == anchor
    with pytest.raises(ValueError, match="opened"):
        controller.finalize()
    assert races == [str(b), str(r)]


def test_heldout_worker_cannot_launch_without_an_immutable_freeze(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight
    import time
    monkeypatch.setattr(overnight.subprocess, "Popen", lambda *a, **kw: pytest.fail("unfrozen heldout launch"))
    with pytest.raises(ValueError, match="freeze"):
        overnight.OwnedEvaluation(tmp_path / "model_0.pt", tmp_path / "eval", list(range(6,26)), deadline=time.time()+600)
    assert not (tmp_path / "eval").exists()


def test_gate_selects_best_official_five_of_five_inside_stable_neighborhood(stable_evidence):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    path, report = stable_evidence
    run = Path(report["candidate_path"]).parent
    official = run / "workshop/model_0.json"
    result = evidence.read_report(official)
    result["per_seed"][0]["status"] = "fall"
    result.update(seeds_finished=4, score=16.)
    campaign.write_json(official, result)
    report["artifact_hashes"][str(official)] = evidence.digest(official)
    report.update(candidate_checkpoint=50, candidate_path=str(run / "model_50.pt"))
    report["exports"]["calibration"]["checkpoint_sha256"] = evidence.digest(run / "model_50.pt")
    campaign.write_json(path, report)
    assert overnight.require_stability_gate(path, report["candidate_path"])["verified"]["neighbors"] == [0,50,100]


def test_trainer_wrapper_checks_bound_plan_and_arms_before_exec(tmp_path, monkeypatch):
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    import time
    plan = campaign.make_plan(stage="bootstrap", iterations=1, seed=4)
    identity, deadline = evidence.plan_digest(plan), time.time()+60
    path = tmp_path / "run.json"
    campaign.write_json(path, {"plan": plan, "plan_digest": identity, "global_deadline_epoch": deadline+10,
        "trainer_deadline_epoch": deadline, "preflight": {"no_smoke": True}})
    calls = []
    monkeypatch.setattr(overnight, "arm_owned_process", lambda owner, cutoff: calls.append(("armed", owner, cutoff)))
    monkeypatch.setattr(overnight.os, "execvpe", lambda executable, args, environment: calls.append(("exec", args)))
    overnight.execute_trainer_worker(path, identity, 1234, deadline)
    assert calls == [("armed", 1234, deadline), ("exec", plan["command"])]
    with pytest.raises(ValueError, match="bounded launch"):
        overnight.execute_trainer_worker(path, identity, 1234, deadline+1)
    assert len(calls) == 2


def test_summary_counts_reserved_screen_and_does_not_count_an_error_cache_as_complete(tmp_path):
    from microduck_challenges.hill_climb_1m5 import overnight
    import time
    run = tmp_path / "saved"
    (run / "workshop").mkdir(parents=True)
    (run / "model_0.pt").write_bytes(b"unit-only checkpoint")
    campaign.write_json(run / "workshop/model_0.json", {"error": "unit exporter boundary failed"})
    journal = overnight.Journal.create(tmp_path / "controller/campaign.json", {
        "state": "blocked", "start_epoch": 100., "training_deadline_epoch": 21700., "hard_deadline_epoch": 28900.,
        "original_source": {"path": "E"}, "maximum_new_screens": 5, "screens": [], "final": None,
        "current_screen": {"ordinal": 1, "profile": "S0", "state": "reserved", "run": str(run),
                           "expected_checkpoints": [0,50,499]}})
    summary = overnight.write_summary(journal.path)
    assert summary["new_screens"] == 1
    inventory = summary["inventories"][0]
    assert inventory["evaluated"] == [] and inventory["pending"] == [0,50,499]
    assert inventory["evaluation_errors"]
    assert inventory["final_checkpoint"] is None


def test_p1_launch_guard_is_enforced_at_actual_plan_boundary_not_only_policy(tmp_path):
    from microduck_challenges.hill_climb_1m5 import overnight
    import time
    controller = overnight.Controller(tmp_path / "auth.json", tmp_path / "controller")
    controller.journal = overnight.Journal.create(controller.output / "campaign.json", {
        "start_epoch": time.time(), "screens": [], "anchor": {"path": "R", "gate_path": "missing"},
        "best_candidate": None})
    with pytest.raises(ValueError, match="four hours|T\\+4"):
        controller.launch_plan("P1", 1, "R")
    with pytest.raises(ValueError, match="profile"):
        controller.launch_plan("H1", 1, "R")


def test_p1_plan_freezes_verified_gate_for_child_without_ambient_environment(stable_evidence, monkeypatch):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import evidence, learning
    gate, report = stable_evidence
    source = Path(report["candidate_path"])
    selected = {"path": source, "profile": "S0", "partial_exception": None,
                "clock": 24024, "curriculum_clock": 24024, "reward_origin": 24024,
                "base_knobs": learning.APPROVED_PARTIAL["base_knobs"],
                "source_identity": {"path": str(source)},
                "result_path": source.parent / "workshop/model_0.json"}
    monkeypatch.setattr(campaign, "selected_source", lambda *a, **kw: selected)
    monkeypatch.setenv("MICRODUCK_HILL_STABILITY_GATE", str(gate.parent / "." / gate.name))
    monkeypatch.setenv("MICRODUCK_HILL_STABILITY_GATE_SHA256", evidence.digest(gate))
    plan = campaign.make_plan(stage="refine", source=source, reward_profile="P1", iterations=1)
    monkeypatch.delenv("MICRODUCK_HILL_STABILITY_GATE")
    monkeypatch.delenv("MICRODUCK_HILL_STABILITY_GATE_SHA256")
    assert plan["environment"]["MICRODUCK_HILL_STABILITY_GATE"] == str(gate.resolve())
    assert plan["environment"]["MICRODUCK_HILL_STABILITY_GATE_SHA256"] == evidence.digest(gate)
    assert plan["source_config_files"][str(gate.resolve())] == evidence.digest(gate)
    child = campaign.hill_env.refinement_algorithm(selected, "explore", "P1",
        stability_gate=plan["environment"]["MICRODUCK_HILL_STABILITY_GATE"],
        stability_gate_sha256=plan["environment"]["MICRODUCK_HILL_STABILITY_GATE_SHA256"])
    assert child.stability_gate == plan["target_algorithm"]["stability_gate"]


@pytest.mark.parametrize("identity", [
    {"start_ticks": "reused", "argv": ["python", "-m", "mjlab.scripts.train", "Mjlab-HillClimb1m5-MicroDuck", "--agent.run-name", "user-replacement"]},
    {"start_ticks": "recorded", "argv": ["python", "-m", "mjlab.scripts.train", "Mjlab-HillClimb1m5-MicroDuck", "--agent.run-name", "unrecognized"]},
])
def test_preexisting_rejects_reused_pid_or_unrecognized_run_without_wait_or_kill(tmp_path, monkeypatch, identity):
    from microduck_challenges.hill_climb_1m5 import overnight
    controller = overnight.Controller(tmp_path / "auth.json", tmp_path / "controller")
    controller.journal = overnight.Journal.create(controller.output / "campaign.json", {
        "training_deadline_epoch": 1e20, "hard_deadline_epoch": 1e20,
        "preexisting": {"pid_at_discovery": 111, "start_ticks_at_discovery": "recorded",
                        "id": "user-replacement", "run_id": "timestamp_user-replacement"}})
    monkeypatch.setattr(campaign, "active_trainers", lambda: [111])
    monkeypatch.setattr(overnight, "process_identity", lambda pid: identity, raising=False)
    monkeypatch.setattr(controller, "pause", lambda *_: pytest.fail("unrecognized process was waited for"))
    monkeypatch.setattr(overnight.os, "killpg", lambda *_: pytest.fail("preexisting process killed"))
    with pytest.raises(ValueError, match="identity"):
        controller.reconcile_preexisting()


def test_preexisting_transition_preserves_stopped_partial_as_diagnostic_and_waits_for_full_official_inventory(stable_evidence, tmp_path, monkeypatch):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import overnight
    gate, report = stable_evidence
    run = Path(report["candidate_path"]).parent
    stopped = run.parent / "stopped"
    stopped.mkdir()
    for n in (0, 50): (stopped / f"model_{n}.pt").write_bytes(b"diagnostic only")
    partial = {"run_id": "stopped", "saved_checkpoints": [0,50], "observed_state": "stopped"}
    controller = overnight.Controller(tmp_path / "auth.json", tmp_path / "controller")
    controller.journal = overnight.Journal.create(controller.output / "campaign.json", {
        "training_deadline_epoch": 1e20, "hard_deadline_epoch": 1e20,
        "superseded_preexisting_jobs": [partial],
        "preexisting": {"pid_at_discovery": 111, "start_ticks_at_discovery": "recorded",
                        "id": "user-replacement", "run_id": run.name}})
    monkeypatch.setattr(campaign, "LOG_ROOT", run.parent)
    trainers = iter([[111], [], [], []])
    monkeypatch.setattr(campaign, "active_trainers", lambda: next(trainers))
    monkeypatch.setattr(overnight, "process_identity", lambda pid: {"start_ticks": "recorded",
        "argv": ["python", "-m", "mjlab.scripts.train", "Mjlab-HillClimb1m5-MicroDuck", "--agent.run-name", "user-replacement"]}, raising=False)
    missing = run / "workshop/model_199.json"
    official = campaign.read_report(missing)
    missing.unlink()
    pauses = []
    def pause(seconds):
        pauses.append(seconds)
        if len(pauses) == 2: campaign.write_json(missing, official)
        assert len(pauses) <= 2
    monkeypatch.setattr(controller, "pause", pause)
    controller.reconcile_preexisting()
    assert len(pauses) == 2
    assert controller.state["preexisting_inventory"]["evaluated"] == report["checkpoints"]
    diagnostic = controller.state["superseded_preexisting_inventories"][0]
    assert diagnostic["saved"] == [0,50]
    assert diagnostic["diagnostic_only"] and not diagnostic["valid_continuation_source"]


def test_summary_requires_both_complete_inventories_not_one_fixed_five_of_five(stable_evidence, tmp_path, worker_report_factory):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import overnight
    gate, report = stable_evidence
    run = Path(report["candidate_path"]).parent
    assessment = tmp_path / "assessment"
    worker_report_factory(run / "model_0.pt", assessment / "model-0", overnight.DEV)
    journal = overnight.Journal.create(tmp_path / "controller/campaign.json", {
        "state": "stability_unmet", "start_epoch": 100., "training_deadline_epoch": 21700., "hard_deadline_epoch": 28900.,
        "original_source": {"path": "E"}, "maximum_new_screens": 5, "final": None,
        "screens": [{"ordinal": 1, "profile": "S0", "state": "completed", "run": str(run),
                     "assessment": str(assessment), "expected_checkpoints": report["checkpoints"]}]})
    summary = overnight.write_summary(journal.path)
    inventory = summary["inventories"][0]
    assert not inventory["valid_continuation_source"]
    assert inventory["official_inventory"]["evaluated"] == report["checkpoints"]
    assert inventory["official_inventory"]["pending"] == []
    assert inventory["fixed_command_inventory"]["evaluated"] == [0]
    assert inventory["fixed_command_inventory"]["pending"] == [50,100,150,199]
    for n in report["checkpoints"][1:]:
        worker_report_factory(run / f"model_{n}.pt", assessment / f"model-{n}", overnight.DEV)
    complete = overnight.write_summary(journal.path)["inventories"][0]
    assert complete["fixed_command_inventory"]["pending"] == []
    assert complete["valid_continuation_source"]


@pytest.mark.parametrize("corruption", ["raw", "missing_candidate_rows", "missing_anchor_report", "missing_artifacts"])
def test_summary_tampered_final_raw_is_incomplete_not_complete_with_false_flag(tmp_path, worker_report_factory, corruption):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import overnight, evidence
    source = tmp_path / "R.pt"
    source.write_bytes(b"unit-only checkpoint")
    identity = {"path": str(source), "sha256": evidence.digest(source), "profile": "S0"}
    root = tmp_path / "controller"
    freeze = root / "final-freeze.json"
    overnight.durable_json(freeze, {"candidate": identity, "anchor": identity, "seeds": overnight.HELD, "command_vx": .75})
    candidate = worker_report_factory(source, root / "final-candidate", overnight.HELD, freeze=freeze, partition="heldout")
    anchor = worker_report_factory(source, root / "final-anchor", overnight.HELD, freeze=freeze, partition="heldout")
    final = {"path": str(source), "sha256": evidence.digest(source), "heldouts_opened": True,
             "candidate_rows": candidate["rows"], "anchor_rows": anchor["rows"],
             "candidate_evaluation": str(root / "final-candidate/evaluation.json"),
             "anchor_evaluation": str(root / "final-anchor/evaluation.json"),
             "artifact_hashes": {**candidate["artifact_hashes"], **anchor["artifact_hashes"]}}
    journal = overnight.Journal.create(root / "campaign.json", {
        "state": "complete", "start_epoch": 100., "training_deadline_epoch": 21700., "hard_deadline_epoch": 28900.,
        "original_source": {"path": "E"}, "maximum_new_screens": 5, "screens": [],
        "anchor": identity, "best_candidate": identity, "final": final})
    assert overnight.write_summary(journal.path)["state"] == "complete"
    if corruption == "raw": Path(candidate["raw_path"]).write_text("tampered after completion")
    elif corruption == "missing_anchor_report": Path(final["anchor_evaluation"]).unlink()
    else:
        if corruption == "missing_candidate_rows": del journal.data["final"]["candidate_rows"]
        else: journal.data["final"]["artifact_hashes"] = {}
        journal.record("unit_boundary_corruption", final=journal.data["final"])
    summary = overnight.write_summary(journal.path)
    assert summary["state"] == "incomplete"
    assert overnight.Journal.load(journal.path).data["state"] == "incomplete"
    assert summary["final_recomputed"]["error"]


@pytest.mark.parametrize("corruption", ["request_hash", "raw_hash", "export_hash", "checkpoint_hash",
    "agent_hash", "environment_hash", "provenance_hash", "empty_export", "export_identity", "request_identity", "runtime"])
def test_worker_verifier_requires_all_nonempty_hashed_artifacts_and_exact_identities(tmp_path, worker_report_factory, corruption):
    from pathlib import Path
    from microduck_challenges.hill_climb_1m5 import overnight
    source = tmp_path / "model_0.pt"
    source.write_bytes(b"unit checkpoint")
    out = tmp_path / "worker"
    report = worker_report_factory(source, out, overnight.DEV)
    assert overnight.verify_worker_report(out / "evaluation.json", checkpoint=source, seeds=overnight.DEV) == report
    paths = {"request_hash": out / "request.json", "raw_hash": Path(report["raw_path"]),
             "export_hash": Path(report["export"]["onnx_path"]), "checkpoint_hash": source,
             "agent_hash": source.parent / "params/agent.yaml", "environment_hash": source.parent / "params/env.yaml",
             "provenance_hash": source.parent / "provenance.json"}
    if corruption in paths: del report["artifact_hashes"][str(paths[corruption])]
    elif corruption == "empty_export": Path(report["export"]["onnx_path"]).write_bytes(b"")
    elif corruption == "export_identity": report["export"]["checkpoint_sha256"] = "0"*64
    elif corruption == "request_identity": report["request"]["checkpoint"] = str(tmp_path / "wrong.pt")
    elif corruption == "runtime": report["runtime"] = {}
    overnight.durable_json(out / "evaluation.json", report)
    with pytest.raises((ValueError, OSError)):
        overnight.verify_worker_report(out / "evaluation.json", checkpoint=source, seeds=overnight.DEV)
