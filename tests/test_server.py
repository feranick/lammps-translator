"""MCP server behaviour: stale servers after a deploy (llm-translators docs/SHARED.md) and tools that return
within wait_s. Version: 2026.11.7.1"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lammps_translator import api, stale
from lammps_translator.lammps import find_lmp

ROOT = Path(__file__).parent.parent
OLD, NEW = "2001.01.01.1", "2001.01.01.2"          # fake release names (not this package's version)
LJ = {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [3, 3, 3]},
      "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 8.5, "shift": True},
      "stages": [{"kind": "energy"}]}


def _releases(tmp_path):
    prefix = tmp_path / "opt"
    for r in (OLD, NEW):
        d = prefix / "releases" / r / "lammps-translator" / "lammps_translator"
        d.mkdir(parents=True)
        (prefix / "releases" / r / "RELEASE.json").write_text(json.dumps({"release": r}))
    (prefix / "current").symlink_to(f"releases/{OLD}")
    return prefix


def _switch(prefix, r):
    tmp = prefix / "current.new"
    tmp.symlink_to(f"releases/{r}")
    os.replace(tmp, prefix / "current")


PKG = f"releases/{OLD}/lammps-translator/lammps_translator"


@pytest.fixture
def clean_stale(monkeypatch):
    saved = dict(stale.STATE)
    exits = []
    monkeypatch.setattr(stale, "_exit_soon", lambda why, delay: exits.append(why))
    yield exits
    stale.STATE.clear()
    stale.STATE.update(saved)


def test_stale_reason_shared_and_checkout(tmp_path):
    prefix = _releases(tmp_path)
    pkg = prefix / PKG
    assert stale.stale_reason(pkg, OLD) == (None, True)
    _switch(prefix, NEW)
    why, shared = stale.stale_reason(pkg, OLD)
    assert shared and f"{NEW} is installed" in why and OLD in why
    co = tmp_path / "checkout" / "lammps_translator"
    co.mkdir(parents=True)
    (co / "__init__.py").write_text('__version__ = "2026.10.25.9"\n')
    assert stale.stale_reason(co, "2026.10.25.9") == (None, False)
    why, shared = stale.stale_reason(co, OLD)
    assert not shared and "2026.10.25.9 is installed" in why


def test_only_a_serving_stale_server_refuses(tmp_path, clean_stale, monkeypatch):
    prefix = _releases(tmp_path)
    stale.STATE.update(pkg=prefix / PKG, checked=0.0, serving=False, inflight=0, exiting=False)
    _switch(prefix, NEW)
    assert stale.enter_call() is None                  # tests / run-study: not serving, served normally
    stale.leave_call()
    stale.STATE.update(serving=True, checked=0.0)
    monkeypatch.setattr(stale, "pool_busy", lambda: True)            # a local study runs: answer, with a note
    assert stale.enter_call() is None and "old code" in stale.note()
    stale.leave_call()
    monkeypatch.setattr(stale, "pool_busy", lambda: False)
    msg = stale.enter_call()
    assert msg and msg.startswith("STALE_SERVER") and clean_stale
    monkeypatch.setenv("LMPT_EXIT_WHEN_STALE", "0")                   # never
    stale.STATE.update(exiting=False)
    assert stale.enter_call() is None
    stale.leave_call()


def test_stale_refusal_reaches_the_caller_as_tool_error(tmp_path, clean_stale):
    from lammps_translator import mcp_server as m
    prefix = _releases(tmp_path)
    stale.STATE.update(pkg=prefix / PKG, checked=0.0, serving=True, inflight=0, exiting=False)
    _switch(prefix, NEW)
    with pytest.raises(m.ToolError, match="STALE_SERVER"):
        asyncio.run(m.jobs())


def test_idle_stale_server_exits_by_itself(tmp_path):
    prefix = _releases(tmp_path)
    code = (f"import time; from lammps_translator import stale; stale.serve({str(prefix / PKG)!r}, period=0.3); "
            "print('watching', flush=True); time.sleep(60)")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env)
    assert p.stdout.readline().strip() == "watching"
    time.sleep(1.0)
    assert p.poll() is None                                          # current: keeps running
    _switch(prefix, NEW)
    try:
        assert p.wait(timeout=10) == 0                               # stale and idle: gone within seconds
        assert "STALE_SERVER" in p.stderr.read()
    finally:
        p.kill()


def test_guarded_tools_keep_their_parameters():
    from lammps_translator import mcp_server as m
    tools = {t.name: t for t in asyncio.run(m.server.list_tools())}
    assert {"run_studies", "study_status", "jobs", "cancel", "run_benchmarks", "bench_status"} <= set(tools)
    def props(n):
        t = tools[n]
        sch = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or getattr(t, "parameters", {})
        if not isinstance(sch, dict):
            sch = sch.model_dump() if hasattr(sch, "model_dump") else dict(sch)
        return set(sch.get("properties", {}))
    assert {"studies", "wait_s"} <= props("run_studies") and {"bench", "wait_s"} <= props("bench_status")


@pytest.mark.skipif(find_lmp() is None, reason="needs LAMMPS")
def test_tools_return_within_wait_s(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    long = {**LJ, "stages": [{"kind": "nvt", "T": 30, "steps": 3_000_000}]}
    t0 = time.time()
    r = api.run({"long": long}, wait=2)
    assert time.time() - t0 < 6 and r["pending"] == ["long"]
    t0 = time.time()
    b = api.bench(["madelung_nacl", "spce_water"], wait=3)          # benchmarks run in workers, not here
    assert time.time() - t0 < 8 and b["pending"] == ["spce_water"], b
    assert b["cases"]["madelung_nacl"]["passed"] and "bench_status" in b["next"]
    st = api.bench_status(b["request"], wait=0)
    assert st["pending"] == ["spce_water"] and st["cases"]["spce_water"]["state"] in ("running", "queued")
    api.cancel(b["request"])
    api.cancel(r["request"])


def test_orphaned_worker_ends_with_its_lammps_runs(tmp_path):
    """A pool worker whose server is killed (kill -9) ends, and so does the LAMMPS process group it started."""
    out = tmp_path / "pids"
    child = ("import os, subprocess, sys, time; from lammps_translator import lammps, workers; "
             "p = subprocess.Popen(['sleep', '120'], start_new_session=True); lammps._RUNNING.add(p); "
             "workers.PARENT_POLL_S = 0.2; workers.watch_parent(); "
             f"open({str(out)!r}, 'w').write(f'{{os.getpid()}} {{p.pid}}'); time.sleep(120)")
    parent = (f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(120)")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    srv = subprocess.Popen([sys.executable, "-c", parent], env=env)
    t0 = time.time()
    while not (out.is_file() and out.read_text().strip()):
        assert time.time() - t0 < 20
        time.sleep(0.1)
    worker, sleeper = (int(x) for x in out.read_text().split())
    srv.kill()                                                       # the server dies without stopping its pool
    srv.wait()

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
        except OSError:
            return True
    t0 = time.time()
    while (alive(worker) or alive(sleeper)) and time.time() - t0 < 15:
        time.sleep(0.2)
    try:
        assert not alive(worker) and not alive(sleeper)
    finally:
        for pid in (worker, sleeper):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


def test_pool_workers_watch_their_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    pool = api._Pool()
    pool.submit(tmp_path, [])
    try:
        r = pool.pool.apply_async(_watcher_started).get(timeout=60)
        assert r is True
    finally:
        pool.pool.terminate()


def _watcher_started():
    from lammps_translator import workers
    return workers._STARTED["pid"] == os.getpid()
