"""SLURM executor tests with fake sbatch/squeue/scancel/sacct/sinfo (tests/fakeslurm.py). Version: 2026.10.18.1
They run anywhere; with LAMMPS (set LMPT_LMP, or lmp on the PATH) the jobs also compute real results."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lammps_translator import api
from lammps_translator import executor as X
from lammps_translator.lammps import find_lmp

HERE = Path(__file__).parent
ROOT = HERE.parent
HAVE_LAMMPS = find_lmp() is not None
LJ = {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [3, 3, 3]},
      "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 8.5, "shift": True},
      "stages": [{"kind": "energy"}]}


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    fake = tmp_path / "fake"
    fake.mkdir()
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for c in ("sbatch", "squeue", "scancel", "sacct", "sinfo"):
        w = bin_ / c
        w.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fakeslurm.py'} {c} \"$@\"\n")
        w.chmod(0o755)
    lmp = find_lmp()
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    if lmp:
        monkeypatch.setenv("LMPT_LMP", " ".join(lmp))
    monkeypatch.setenv("FAKE_SLURM", str(fake))
    monkeypatch.setenv("LMPT_EXECUTOR", "slurm")
    monkeypatch.setenv("LMPT_SLURM_MIN_MINUTES", "0")
    monkeypatch.setenv("LMPT_OUT", str(tmp_path / "runs"))
    monkeypatch.setenv("PYTHONPATH", f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.delenv("LMPT_TIMEOUT", raising=False)
    return fake


def _state(fake):
    return json.loads((fake / "state.json").read_text())


def _new_process_status(request, wait=90):
    code = f"import json; from lammps_translator import api; print(json.dumps(api.status({request!r}, {wait})))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=wait + 60, env=os.environ)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_status_key_and_job_file(slurm, monkeypatch):
    s = api.tool_status()["executor"]
    assert s["kind"] == "slurm" and s["sbatch"].endswith("sbatch") and s["time_limit_min"] == 370   # 6 h + 10 min
    (slurm / "hold").write_text("")
    monkeypatch.setenv("LMPT_PROCS", "3")
    r = api.run({"lj": LJ}, wait=0)
    req = Path(r["request"])
    job = (req / "lj" / "job.sh").read_text()
    assert f"#SBATCH --job-name=lmpt:{req.name}:lj" in job and "#SBATCH --ntasks=3" in job
    assert "#SBATCH --time=370" in job and "#SBATCH --export=ALL" in job and "run-study" in job
    info = json.loads((req / "lj" / "job.json").read_text())
    assert info["executor"] == "slurm" and info["procs"] == 3 and info["time_min"] == 370
    assert r["pending"] == ["lj"] and r["progress"]["lj"]["state"] == "queued"
    assert r["progress"]["lj"]["reason"] == "Resources" and r["progress"]["lj"]["ahead"] == 0
    assert "SLURM jobs" in r["next"] and info["job_id"] in r["next"]


def test_submit_run_and_collect_from_a_new_process(slurm):
    r = api.run({"lj": LJ}, wait=0)
    assert r["pending"] == ["lj"] or r["done"]
    out = _new_process_status(r["request"])
    assert out["done"], out
    res = out["studies"]["lj"]
    if HAVE_LAMMPS:
        assert res["ok"] and res["checks"]["reference"]["pass"], res.get("why")
    else:
        assert res["ok"] is False and "LAMMPS not found" in res["why"]
    d = Path(r["request"]) / "lj"
    assert not list(d.glob("*.tmp*"))     # written atomically
    assert (d / "job.log").is_file()


def test_short_studies_stay_local(slurm, monkeypatch):
    monkeypatch.setenv("LMPT_SLURM_MIN_MINUTES", "1")
    folder = api.start({"quick": LJ, "forced": {**LJ, "queue": True}})
    meta = json.loads((folder / "request.json").read_text())
    assert meta["executor"] == {"quick": "local", "forced": "slurm"}
    assert meta["estimate_min"]["quick"] < 1
    assert not (folder / "quick" / "job.json").exists() and (folder / "forced" / "job.json").exists()
    monkeypatch.setenv("LMPT_SLURM_MIN_MINUTES", "0")
    folder = api.start({"never": {**LJ, "queue": False}})
    assert json.loads((folder / "request.json").read_text())["executor"] == {"never": "local"}
    api.status(str(folder), 60)


def test_submit_failure_is_a_result(slurm):
    (slurm / "fail_submit").write_text("")
    r = api.run({"lj": LJ}, wait=0)
    res = r["studies"]["lj"]
    assert r["done"] and res["ok"] is False and res["why"].startswith("could not submit:")
    assert "slurm controller" in res["why"]


def test_cancel_queued(slurm):
    (slurm / "hold").write_text("")
    r = api.run({"a": LJ, "b": LJ}, wait=0)
    c = api.cancel(r["request"], ["a"])
    assert c["cancel"]["a"]["cancelled"]
    st = c["status"]
    assert st["studies"]["a"]["state"] == "cancelled" and st["studies"]["a"]["ok"] is False
    assert st["pending"] == ["b"]


@pytest.mark.parametrize("sacct, log, state", [
    ("TIMEOUT", "", "timed_out"), ("OUT_OF_MEMORY", "", "out_of_memory"), ("CANCELLED by 1000", "", "cancelled"),
    ("FAILED", "Traceback ... boom", "failed"), ("NODE_FAIL", "", "failed"),
    (None, "slurmstepd: error: *** JOB 1001 ON x CANCELLED AT 2026 DUE TO TIME LIMIT ***", "timed_out"),
    (None, "slurmstepd: error: *** JOB 1001 ON x CANCELLED AT 2026 ***", "cancelled"),
    (None, "Segmentation fault", "failed"),
])
def test_failure_states(slurm, sacct, log, state):
    (slurm / "hold").write_text("")
    r = api.run({"lj": LJ}, wait=0)
    jid = json.loads((Path(r["request"]) / "lj" / "job.json").read_text())["job_id"]
    s = _state(slurm)
    s["jobs"][jid]["vanished"] = True          # gone from the queue, no result
    (slurm / "state.json").write_text(json.dumps(s))
    if sacct:
        (slurm / f"sacct_{jid}").write_text(sacct)
    else:
        (slurm / "sacct_fail").write_text("")
    (Path(r["request"]) / "lj" / "job.log").write_text(log + "\n")
    out = api.status(r["request"])
    res = out["studies"]["lj"]
    assert out["done"] and res["state"] == state and res["ok"] is False and res["why"], res
    if state == "failed":
        assert "log_tail" in res or "SLURM state" in res["why"]
    if state == "timed_out":
        assert "LMPT_TIMEOUT" in res["why"]
    if state == "out_of_memory":
        assert "--mem" in res["why"]


def test_jobs_lists_requests_queue_and_machine(slurm):
    (slurm / "hold").write_text("")
    r1 = api.run({"x": LJ}, wait=0)
    j = api.jobs()
    assert j["requests"][0]["request"] == r1["request"] and j["requests"][0]["studies"]["x"]["state"] == "queued"
    assert j["queue"] == {"running": 0, "queued": 1, "cpus_in_use": 0}
    assert j["machine"]["cpus"] == 8
    api.cancel(r1["request"])
    assert api.jobs(active=True)["requests"] == []
    assert api.jobs()["requests"][0]["studies"]["x"]["state"] == "cancelled"


def test_run_study_cli_by_hand(slurm, tmp_path):
    (slurm / "hold").write_text("")
    r = api.run({"lj": LJ}, wait=0)
    p = subprocess.run([sys.executable, "-m", "lammps_translator.cli", "run-study", r["request"], "lj"],
                       capture_output=True, text=True, timeout=120, env=os.environ)
    assert p.returncode == 0, p.stderr
    res = json.loads((Path(r["request"]) / "lj" / "lj.json").read_text())
    assert "ok" in res


@pytest.mark.skipif(not HAVE_LAMMPS, reason="needs LAMMPS")
def test_running_state_and_cancel_running_job(slurm):
    st = {**LJ, "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "nvt", "T": 30, "steps": 2000000}]}
    r = api.run({"md": st}, wait=0)
    t0 = time.time()
    while True:
        s = api.status(r["request"])
        p = s["progress"]["md"]
        if p.get("stage", "").endswith("nvt") and p.get("eta_min_max") is not None:
            break
        assert time.time() - t0 < 60, p
        time.sleep(1)
    assert p["state"] == "running" and "job_elapsed_min" in p and p["md_steps_total"] == 2000000
    c = api.cancel(r["request"])
    assert c["status"]["studies"]["md"]["state"] == "cancelled"


@pytest.mark.skipif(not HAVE_LAMMPS, reason="needs LAMMPS")
def test_bench_through_slurm(slurm):
    out = api.bench(["madelung_nacl", "nve_conservation"], wait=120)
    assert out["passed"] == ["madelung_nacl", "nve_conservation"], json.dumps(out["cases"])[:2000]
    meta = json.loads((Path(out["request"]) / "request.json").read_text())
    assert set(meta["executor"].values()) == {"slurm"}


@pytest.mark.skipif(not HAVE_LAMMPS, reason="needs LAMMPS")
def test_local_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    monkeypatch.setenv("LMPT_EXECUTOR", "local")
    st = {**LJ, "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "nvt", "T": 30, "steps": 3000000}]}
    r = api.run({"md": st}, wait=3)
    assert r["pending"] == ["md"] and r["progress"]["md"]["state"] in ("running", "queued")
    c = api.cancel(r["request"])
    assert c["cancel"]["md"]["cancelled"]
    out = api.status(r["request"], 60)
    assert out["done"] and out["studies"]["md"]["state"] == "cancelled"
