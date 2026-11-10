"""lmpt on llmt-runtime (2026.11.09.1, phase 1): the shared pieces are the runtime's, configured for lmpt; a LAMMPS
run killed for memory is out of memory, not timed out.
Version: 2026.11.09.1
"""
import json
import os
from pathlib import Path

import llmt_runtime
from llmt_runtime import memory as M
from llmt_runtime import workers as W

from lammps_translator import __version__, api, lammps, workers
from lammps_translator import executor as X
from lammps_translator.runtime import RT


def test_runtime_configured_for_lmpt(monkeypatch):
    assert (RT.prefix, RT.cli, RT.package, RT.kind) == ("LMPT", "lmpt", "lammps_translator", "mpi")
    monkeypatch.setenv("LMPT_OUT", "/tmp/lmpt-x")
    assert RT.out_root() == Path("/tmp/lmpt-x") == api.out_root()


def test_runtime_version_reported(monkeypatch):
    monkeypatch.setenv("LMPT_EXECUTOR", "local")
    assert X.status()["runtime"] == llmt_runtime.__version__


def test_workers_are_the_runtimes_with_lammps_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(lammps, "kill_all", lambda: calls.append(1))
    monkeypatch.setattr(W, "_HOOKS", [])
    workers._kill_lammps()
    assert calls == [1]
    monkeypatch.setattr(W, "watch_parent", lambda ppid, name, **kw: calls.append(name))
    workers.watch_parent(os.getpid())
    assert calls[-1] == "lmpt" and workers._kill_lammps in W._HOOKS and workers._STARTED is W._STARTED


class _Watch:
    """OomWatch's stand-in: the verdict a test needs."""
    oom = True

    def sample(self):
        pass

    def classify(self, code, rss, pid=None, threads=None):
        return {"out_of_memory": self.oom, "oom_scope": "job" if self.oom else None, "peak_rss_mb": 0.0,
                "limit_mb": 8192.0, "threads": threads}


def test_a_kill_for_memory_is_not_a_time_out(tmp_path, monkeypatch):
    """Before 2026.11.09.1 every SIGKILL read 'timed out after ... (LMPT_TIMEOUT)', also the out-of-memory killer's."""
    monkeypatch.setattr(lammps, "find_lmp", lambda: ["sh", "-c", "kill -9 $$"])
    monkeypatch.setattr(M, "OomWatch", _Watch)
    r = lammps.run("in.x", "log.x", tmp_path, 1, 30)
    assert r["rc"] == -9 and r["out_of_memory"] and r["oom_scope"] == "job"
    assert r["stderr"].startswith("LAMMPS ran out of memory with 1 rank") and "timed out" not in r["stderr"]
    _Watch.oom = False                                            # a crash: neither out of memory nor a time-out
    try:
        r = lammps.run("in.x", "log.x", tmp_path, 1, 30)
        assert r["rc"] == -9 and "out_of_memory" not in r and "timed out" not in r["stderr"]
    finally:
        _Watch.oom = True


def test_a_real_time_out_stays_one(tmp_path, monkeypatch):
    monkeypatch.setattr(lammps, "find_lmp", lambda: ["sh", "-c", "sleep 30"])
    monkeypatch.setattr(M, "OomWatch", _Watch)
    r = lammps.run("in.x", "log.x", tmp_path, 1, 1.5)
    assert r["stderr"] == "timed out after 1.5 s (LMPT_TIMEOUT)" and "out_of_memory" not in r


def test_ended_state_oom_kill_wording_with_accounting(tmp_path, monkeypatch):
    d = tmp_path / "s"
    d.mkdir()
    (d / "job.json").write_text(json.dumps({"procs": 2, "mem": "4G", "natoms": 32000}))
    (d / "job.log").write_text("slurmstepd: error: Detected 1 oom_kill event in StepId=9.batch.\n")
    monkeypatch.setattr(X, "_sacct", lambda jid: "FAILED")
    st = X.ended_state(d, "9")
    assert st["state"] == "out_of_memory" and st["why"].startswith("SLURM killed the job for memory (4G with 2 ranks)")
    assert '"run": {"mem"' in st["why"] and "32,000 atoms" in st["why"]
    monkeypatch.setattr(X, "_sacct", lambda jid: "CANCELLED by 1000")
    (d / "job.log").write_text("")
    assert X.ended_state(d, "9") == {"state": "cancelled", "why": "cancelled by 1000 (scancel or cancel())"}


def test_request_records_versions_and_status_says_what_ran(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    monkeypatch.setenv("LMPT_EXECUTOR", "local")
    from test_slurm import LJ
    r = api.run({"x": {**LJ, "stages": [{"kind": "nope"}]}}, wait=1, request_name="run_versions")   # invalid
    meta = json.loads((tmp_path / "run_versions" / "request.json").read_text())
    assert meta["translator"] == __version__ and meta["runtime"] == llmt_runtime.__version__
    assert r["translator"] == __version__ and "translator_now" not in r
    meta["translator"] = "2026.11.7.1"                               # submitted by the lmpt before
    (tmp_path / "run_versions" / "request.json").write_text(json.dumps(meta))
    again = api.status(str(tmp_path / "run_versions"))
    assert again["translator"] == "2026.11.7.1" and again["translator_now"] == __version__
