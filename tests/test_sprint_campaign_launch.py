"""Launch-path regressions: real hooks/locks/signals, never GPU training."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from microduck_challenges.sprint_2m import campaign


def test_planned_entrypoint_records_real_training_provenance(tmp_path, monkeypatch):
    from mjlab_microduck.tasks import _record_if_training

    plan = campaign.make_plan("explore")
    spec = importlib.util.find_spec(plan["command"][2])
    assert spec is not None and spec.origin is not None
    monkeypatch.setattr(sys, "argv", [spec.origin, *plan["command"][3:]])
    _record_if_training(tmp_path, {"seed": plan["seed"]})
    path = tmp_path / "provenance.json"
    assert path.is_file(), "the actual runner hook must recognize the planned training entrypoint"
    record = json.loads(path.read_text())
    assert record["task"] == campaign.TASK and record["seed"] == plan["seed"]


def test_preflight_lifecycle_tests_do_not_contend_for_the_parent_launch_lock(tmp_path):
    """A sandbox operational lock models the lock held by campaign run --execute."""
    code = textwrap.dedent("""
        import fcntl
        from pathlib import Path
        import sys
        import pytest
        from microduck_challenges.sprint_2m import campaign
        plan = campaign.make_plan('explore')
        campaign.make_plan = lambda *a, **k: plan
        campaign.LOG_ROOT = Path(sys.argv[1]) / 'operational/logs/rsl_rl/sprint_2m'
        lock = campaign.LOG_ROOT.parent.parent / '.sprint-strategy.lock'
        lock.parent.mkdir(parents=True)
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(pytest.main(['-q', sys.argv[2]]))
    """)
    test = Path(__file__).with_name("test_sprint_strategy.py")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path),
                             str(test) + "::test_bounded_process_lifecycle_is_journaled"],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def _alive(pid):
    try:
        # Test-owned processes only; a reparented zombie cannot keep computing.
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("cause", ["sigterm", "journal_error"])
def test_owned_child_is_reaped_on_wrapper_termination_or_post_spawn_error(tmp_path, cause):
    """The child is a sleeping Python probe; test adapters skip tests/smoke only."""
    code = textwrap.dedent("""
        from pathlib import Path
        import subprocess
        import sys
        from microduck_challenges.sprint_2m import campaign
        root, cause = Path(sys.argv[1]), sys.argv[2]
        plan = campaign.make_plan('explore')
        plan['command'] = [sys.executable, '-c', 'import time; time.sleep(60)']
        plan['max_wall_minutes'] = 1
        campaign.LOG_ROOT = root / 'isolated/logs/rsl_rl/sprint_2m'
        campaign.active_trainers = lambda: []
        campaign.smoke = lambda *a: None
        campaign.subprocess.run = lambda *a, **k: None
        original_spawn = campaign.subprocess.Popen
        def spawn(*a, **k):
            process = original_spawn(*a, **k)
            (root / 'child.pid').write_text(str(process.pid))
            return process
        campaign.subprocess.Popen = spawn
        original_write = campaign.write_json
        def write(path, data):
            if cause == 'journal_error' and data.get('state') == 'running':
                raise OSError('injected test-only post-spawn journal error')
            return original_write(path, data)
        campaign.write_json = write
        result = campaign.run_bounded(plan, root / 'run')
        (root / 'result.json').write_text(__import__('json').dumps(result))
    """)
    child_pid = None
    with (tmp_path / "probe.log").open("w") as log:
        wrapper = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), cause],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 40
            pid_file = tmp_path / "child.pid"
            while not pid_file.exists() and wrapper.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert pid_file.exists(), (tmp_path / "probe.log").read_text()
            child_pid = int(pid_file.read_text())
            if cause == "sigterm":
                journal = tmp_path / "run/run.json"
                while time.monotonic() < deadline:
                    if journal.exists() and json.loads(journal.read_text())["state"] == "running":
                        break
                    assert wrapper.poll() is None, (tmp_path / "probe.log").read_text()
                    time.sleep(0.02)
                else:
                    pytest.fail("wrapper did not reach the supervised running state")
                wrapper.terminate()
            wrapper.wait(timeout=15)
            assert not _alive(child_pid), "wrapper left its owned child running without a watchdog"
            if cause == "sigterm":
                assert json.loads((tmp_path / "run/run.json").read_text())["state"] == "interrupted"
        finally:
            if wrapper.poll() is None:
                os.killpg(wrapper.pid, signal.SIGKILL)
                wrapper.wait(timeout=10)
            if child_pid is not None and _alive(child_pid):
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
