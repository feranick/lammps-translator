"""lmpt on llmt-runtime: the shared pieces are the runtime's, configured for lmpt (2026.11.09.1, phase 1: workers, out
of memory, ended_state, versions; a LAMMPS run killed for memory is out of memory, not timed out. 2026.11.09.3, phase
2: the SLURM layer, with lmpt's sizing as hooks).
Version: 2026.11.09.3
"""
import json
import os
import time
from pathlib import Path

import pytest

import llmt_runtime
from llmt_runtime import jobs as J
from llmt_runtime import memory as M
from llmt_runtime import slurm as S
from llmt_runtime import workers as W
from llmt_runtime.testing import fake_slurm

from lammps_translator import __version__, api, lammps, workers
from lammps_translator import executor as X
from lammps_translator.runtime import RT

REPO = Path(__file__).resolve().parents[1]


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


# ---- 2026.11.09.3, phase 2: the SLURM layer (settings, the machine, time limits, job.sh and job.json, submission,
# states, the queue position, scancel, the machine's CPUs) is llmt-runtime's; lmpt's sizing comes in as hooks

FAKE_KNOBS = ("FAKE_SBATCH_FAIL", "FAKE_SBATCH_PENDING", "FAKE_PENDING", "FAKE_SBATCH_DELAY", "FAKE_SBATCH_SLEEP",
              "FAKE_SBATCH_SLOW_AFTER", "FAKE_MIN_JOB_AGE", "FAKE_PRIORITY", "FAKE_RUN_NODE", "FAKE_START",
              "FAKE_SACCT_FAIL", "FAKE_SQUEUE_FAIL", "FAKE_SCONTROL_NODES", "FAKE_SCONTROL_FAIL", "FAKE_SCONTROL_DENY",
              "FAKE_SINFO_FAIL", "FAKE_SINFO_PARTITIONS", "FAKE_SINFO_CPUS", "FAKE_TIME_LEFT")
LJ = {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [3, 3, 3]},          # 108 atoms
      "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 8.5, "shift": True},
      "stages": [{"kind": "energy"}]}


@pytest.fixture()
def slurm(tmp_path, monkeypatch):
    """llmt-runtime's fake SLURM (one node fake1: 40 CPUs, 20 cores, 64000 MB); returns its state folder."""
    for k in FAKE_KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LMPT_EXECUTOR", "slurm")
    monkeypatch.setenv("LMPT_SLURM_MIN_MINUTES", "0")
    monkeypatch.setenv("LMPT_OUT", str(tmp_path / "runs"))
    monkeypatch.setenv("LMPT_ATOMS_PER_RANK", "1")
    monkeypatch.setenv("PYTHONPATH", f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    state_dir = fake_slurm(tmp_path, monkeypatch)
    X.forget_nodes()
    yield state_dir
    X.forget_nodes()


def _fake_job(state_dir: Path, jid: str, **rec) -> None:
    """A job in the fake's queue, as its sbatch writes them (pid 0: no process), this user's."""
    d = state_dir / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    base = {"name": "lmpt:r:s", "cpus": 1, "t0": time.time(), "log": str(state_dir / f"{jid}.log"), "pid": 0,
            "node": "fake1", "prio": 1000, "user": X._user()}
    (d / f"{jid}.json").write_text(json.dumps({**base, **rec}))


def _request(state_dir: Path, jid: str, log: str | None = None, job: dict | None = None) -> Path:
    """A request with one SLURM study 's' whose job.json names the job `jid`."""
    req = state_dir.parent / "runs" / f"run_{jid}"
    (req / "s").mkdir(parents=True)
    (req / "request.json").write_text(json.dumps({"names": ["s"], "executor": {"s": "slurm"}, "started": time.time(),
                                                  "studies": {"s": LJ}}))
    (req / "s" / "job.json").write_text(json.dumps({"executor": "slurm", "job_id": jid, "procs": 1, **(job or {})}))
    if log is not None:
        (req / "s" / "job.log").write_text(log)
    return req


def test_slurm_layer_is_the_runtimes_with_lmpts_hooks(slurm):
    h = RT.hooks
    assert RT.cancel_file == "cancelled.json" and RT.one_node is False                  # no --nodes=1, as lmpt wrote
    assert h.time_need_min is None and h.resource_lines is None and h.solves is None    # the runtime's defaults
    assert S.time_need_min(RT) == 370 == X.time_limit_min()                             # ceil(21600 / 60) + 10
    six = {**LJ, "procs": 6}
    assert h.study_procs(six) == 6 == S.procs_of(RT, six) == X.procs_of(six)         # the top-level procs
    assert X.atoms(LJ) == 108 and h.procs(six) == X.ranks_for(six, 108) == 6
    assert h.mem_auto(LJ) == "2048M" == X.mem_auto(LJ)                 # 1 GB + 256 MB + 4 kB x 108, at least 2 GB
    assert h.job_extra(LJ) == {"natoms": 108, "estimate_min": X.estimate_minutes(LJ, 108), "procs_requested": 1}
    assert h.running_progress(Path("/nonexistent"), LJ) == {"stage": "queued"}       # api.progress
    assert X.FINAL == J.FINAL and X.slurm_args_tokens() == S.slurm_args(RT) == []
    ex = X.status()
    assert {"kind", "sbatch", "min_minutes", "procs", "time_limit_min", "time", "mem", "slurm_args", "workers",
            "timeout_s", "atoms_per_rank", "caps", "run_keys", "node", "runtime"} <= set(ex)
    assert set(ex["node"]) == {"setting", "nodelist", "cluster_nodes", "this_node", "cores"}
    assert ex["node"]["cores"] == 20 and ex["runtime"] == llmt_runtime.__version__


def test_job_sh_and_job_json(slurm, monkeypatch):
    """job.sh as lmpt wrote it (now llmt-runtime's job_script), --mem inside; job.json with lmpt's keys and the
    runtime's."""
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("LMPT_PROCS", "3")
    req = slurm.parent / "runs" / "run_j"
    info = X.submit(req, "a", LJ, 0.05)
    sh = (req / "a" / "job.sh").read_text().splitlines()
    assert [ln for ln in sh if ln.startswith("#SBATCH")] == [
        "#SBATCH --job-name=lmpt:run_j:a", f"#SBATCH --chdir={req / 'a'}", f"#SBATCH --output={req / 'a' / 'job.log'}",
        "#SBATCH --ntasks=3", "#SBATCH --ntasks-per-core=1", "#SBATCH --time=370", "#SBATCH --mem=2048M",
        "#SBATCH --export=ALL"]
    assert sh[0] == "#!/bin/bash" and sh[-1].startswith("exec ")
    assert sh[-1].endswith(f"-m lammps_translator.cli run-study {req} a")
    j = json.loads((req / "a" / "job.json").read_text())
    old = {"executor", "job_id", "submitted", "procs", "procs_requested", "time_min", "time_from", "timeout_s", "mem",
           "estimate_min", "natoms", "run", "nodelist", "command"}
    assert old <= set(j) and j == info and j["job_id"] == "1001"
    assert (j["procs"], j["procs_requested"], j["natoms"], j["mem"]) == (3, 3, 108, "2048M")
    assert (j["time_min"], j["time_from"], j["timeout_s"], j["estimate_min"]) == (370, "timeout", 21600, 0.05)
    assert j["translator"] == __version__ and j["runtime"] == llmt_runtime.__version__
    # llmt-runtime's fixes: -c in LMPT_SLURM_ARGS places the ranks (no --ntasks-per-core=1); a run.mem sbatch would
    # refuse ('1.5G') is written in MB
    monkeypatch.setenv("LMPT_SLURM_ARGS", "-c 2")
    X.submit(req, "b", {**LJ, "run": {"mem": "1.5G"}}, None)
    sh = (req / "b" / "job.sh").read_text()
    assert "--ntasks-per-core" not in sh and "#SBATCH --mem=1536M" in sh and not X._whole_cores()


def test_estimate_time_gives_the_study_its_own_timeout(slurm, monkeypatch):
    """LMPT_SLURM_TIME=estimate (lmpt's, now the runtime's): the job's limit from the estimate and the study's own
    timeout 10 min less (job.json timeout_s, which run-study reads); run.timeout_s wins; max and minutes are new."""
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("LMPT_SLURM_TIME", "estimate")
    req = slurm.parent / "runs" / "run_e"
    info = X.submit(req, "a", LJ, 2.0)                                # max(20, 3 x 2 + 10) min
    assert "#SBATCH --time=20" in (req / "a" / "job.sh").read_text()
    assert (info["time_min"], info["time_from"], info["timeout_s"]) == (20, "estimate", 600)
    assert J.job_timeout_s(req / "a", X.timeout_s(LJ)) == 600
    info = X.submit(req, "b", {**LJ, "run": {"timeout_s": 1200}}, 2.0)
    assert (info["time_min"], info["time_from"], info["timeout_s"]) == (30, "study", 1200)
    assert X.status()["time"] == "estimate" and X.status()["time_factor"] == 3.0
    monkeypatch.setenv("LMPT_SLURM_TIME", "max")
    monkeypatch.setenv("FAKE_SINFO_PARTITIONS", "main*|2:00:00")
    assert X.job_time(None, LJ) == (120, "max") and X.status()["time"] == "max"
    monkeypatch.setenv("LMPT_SLURM_TIME", "400")
    assert X.job_time(None, LJ) == (400, "minutes")


def test_an_ended_job_squeue_still_lists_is_ended(slurm, monkeypatch):
    """squeue lists an ended job for MinJobAge (300 s) in its final state: lmpt 2026.11.09.2 read every state but the
    queued ones as running. It is now final, from job.log (no accounting here)."""
    monkeypatch.setenv("FAKE_MIN_JOB_AGE", "300")
    monkeypatch.setenv("FAKE_SACCT_FAIL", "1")
    _fake_job(slurm, "1001", ended=time.time(), final="FAILED")
    req = _request(slurm, "1001", log="Traceback (most recent call last):\nValueError: boom\n")
    q = X.queue()
    assert q["1001"]["state"] == "FAILED"
    st = X.study_state(req, "s", q)
    assert st["state"] == "failed" and "ValueError: boom" in st["why"] and st["job_id"] == "1001"
    assert X.queue_summary(q) == {"running": 0, "queued": 0, "cpus_in_use": 0}
    res = api.status(str(req))
    assert res["done"] and res["studies"]["s"]["state"] == "failed"


def test_queue_position_by_priority(slurm):
    """ahead: pending jobs SLURM considers first (higher priority, or equal and older), anyone's (llmt-runtime's
    ahead_of, from lmpt's)."""
    _fake_job(slurm, "1001", state="PENDING", prio=10)
    _fake_job(slurm, "1002", state="PENDING", prio=2000, name="feat:r:y")
    _fake_job(slurm, "1003", state="PENDING", prio=10)
    req = _request(slurm, "1001")
    st = X.study_state(req, "s", X.queue())
    assert st["state"] == "queued" and st["reason"] == "Priority" and st["ahead"] == 1
    assert J.ahead_of("1003") == 2


def test_cancelled_json_is_still_lmpts(slurm):
    """cancel() writes lmpt's cancelled.json ({by, at}) and scancel (llmt-runtime's); the flag makes the study
    cancelled, also when it holds the time as llmt-runtime's cancel_request writes it."""
    _fake_job(slurm, "1001", state="PENDING")
    req = _request(slurm, "1001")
    assert X.cancel(req, ["s"]) == {"s": {"cancelled": True, "job_id": "1001"}}
    c = json.loads((req / "s" / "cancelled.json").read_text())
    assert set(c) == {"by", "at"} and c["by"] == "cancel()"
    assert json.loads((slurm / "jobs" / "1001.json").read_text())["state"] == "CANCELLED"
    st = X.study_state(req, "s", X.queue())
    assert st["state"] == "cancelled" and st["why"].startswith("cancelled by cancel() at ")
    (req / "s" / "cancelled.json").write_text(str(time.time()))
    assert X.ended_state(req / "s", "1001")["state"] == "cancelled"


def test_budgeted_submission_still_says_submitting(slurm, monkeypatch):
    """run_studies submits within its budget and leaves the rest to a background thread: those studies are queued,
    'submitting', until their job.json is written."""
    monkeypatch.setenv("FAKE_SBATCH_PENDING", "1")
    monkeypatch.setenv("FAKE_SBATCH_SLEEP", "2")                       # each sbatch takes 2 s
    folder = api.start({"a": {**LJ, "queue": True}}, budget_s=0.0)
    assert X.study_state(folder, "a", X.queue()) == {"state": "queued", "reason": "submitting"}
    t0 = time.time()
    while not (folder / "a" / "job.json").is_file():
        assert time.time() - t0 < 60
        time.sleep(0.2)
    st = X.study_state(folder, "a", X.queue())
    assert st["state"] == "queued" and st["job_id"] == "1001"
