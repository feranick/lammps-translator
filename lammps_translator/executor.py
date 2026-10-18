"""executor.py — where studies run: the local worker pool or SLURM jobs (llm-translators docs/EXECUTION.md).
Version: 2026.10.18.2

Settings (environment): LMPT_EXECUTOR local | slurm (local); LMPT_SLURM_MIN_MINUTES (1): studies estimated
shorter run locally; LMPT_SLURM_ARGS: extra sbatch arguments; LMPT_PROCS (MPI ranks, --ntasks); LMPT_TIMEOUT
(seconds per study; the job's --time is ceil(timeout/60) + 10 minutes). A study's "queue": true / false forces
SLURM / local (not with LMPT_EXECUTOR=local).

One SLURM job per study: <request>/<name>/job.sh (complete, rerunnable with sbatch) and job.json. States come from
the files and the queue (squeue, then sacct, then job.log), never from in-process handles, so any server process,
after restarts or from another conversation, can collect them.
"""
from __future__ import annotations

import getpass
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLI = "lmpt"
FINAL = ("timed_out", "out_of_memory", "cancelled", "failed")


# ------------------------------------------------------------------------------------------- settings
def kind() -> str:
    k = os.environ.get("LMPT_EXECUTOR", "local").strip().lower()
    return "slurm" if k == "slurm" else "local"


def timeout_s() -> float:
    return float(os.environ.get("LMPT_TIMEOUT", "21600"))


def time_limit_min() -> int:
    return int(math.ceil(timeout_s() / 60.0)) + 10


def min_minutes() -> float:
    return float(os.environ.get("LMPT_SLURM_MIN_MINUTES", "1"))


def default_procs() -> int:
    return int(os.environ.get("LMPT_PROCS", "1"))


def study_procs(study: dict) -> int:
    return int(study.get("procs", default_procs()))


def sbatch_path() -> str | None:
    return shutil.which("sbatch")


def status() -> dict:
    out = {"kind": kind(), "sbatch": sbatch_path(), "min_minutes": min_minutes(), "procs": default_procs(),
           "time_limit_min": time_limit_min(), "slurm_args": os.environ.get("LMPT_SLURM_ARGS", ""),
           "workers": int(os.environ.get("LMPT_WORKERS", "2"))}
    if out["kind"] == "slurm" and not out["sbatch"]:
        out["error"] = "LMPT_EXECUTOR=slurm but sbatch is not on the PATH: studies will fail to submit"
    return out


# ------------------------------------------------------------------------------------------- estimate
_COST = {"lj": 1.5e-6, "morse": 1.5e-6, "buck": 1.5e-6, "coul": 1.5e-6, "eam": 2e-6, "eam/alloy": 2e-6,
         "eam/fs": 2e-6, "tersoff": 5e-6, "tersoff/mod": 5e-6, "tersoff/zbl": 5e-6, "sw": 5e-6, "edip": 5e-6,
         "vashishta": 5e-6, "adp": 1e-5, "meam": 2e-5, "airebo": 3e-5, "rebo": 2e-5, "snap": 2e-4}


def estimate_minutes(study: dict) -> float | None:
    """A rough upper estimate of the study's run time (one core per rank, ~1 GHz-µs per atom-step costs)."""
    try:
        from .build import build_system
        from .format import molecular

        s = build_system(study)
        n = s.n
        if molecular(study):
            ff = study.get("forcefield") or {}
            c = 2e-6 + (4e-6 if ff.get("coulomb", "long") == "long" else 0.0)
        else:
            p = study["potential"]
            c = _COST.get(p["style"], 2e-5)
            if p.get("charges") and p.get("coulomb", "long") == "long":
                c += 5e-6
        evals = 1.0
        for st in study["stages"]:
            k = st["kind"]
            if k in ("nve", "nvt", "npt"):
                evals += int(st.get("steps", 0))
            elif k == "thermal_expansion":
                evals += int(st.get("steps", 20000)) * len(st.get("T", []))
            elif k == "minimize":
                evals += 3000 * (6 if st.get("box", "fixed") != "fixed" else 1)
            elif k == "elastic":
                evals += 13 * (500 if st.get("relax", True) else 1)
            elif k == "eos":
                evals += int(st.get("n", 9)) * (500 if st.get("relax", True) else 1)
            elif k == "vacancy":
                evals += 2 * 2000
            elif k == "surface":
                evals += 3 * 2000 * 3      # slabs are larger than the cell
            else:
                evals += 1
        procs = max(1, study_procs(study))
        sec = evals * n * c / (procs ** 0.85) + 2.0 * (len(study["stages"]) + 1)
        return round(sec / 60.0, 3)
    except Exception:  # noqa: BLE001
        return None


def goes_to_slurm(study: dict, estimate: float | None) -> bool:
    if kind() != "slurm":
        return False
    q = study.get("queue")
    if q is True:
        return True
    if q is False:
        return False
    return estimate is None or estimate >= min_minutes()


# ------------------------------------------------------------------------------------------- submit
def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def _whole_cores() -> bool:
    """One physical core per rank (--hint=nomultithread), unless LMPT_SLURM_ARGS says otherwise. LMPT_PROCS is
    sized in physical cores (llmt tune), and Open MPI binds ranks to cores."""
    a = os.environ.get("LMPT_SLURM_ARGS", "")
    return not any(k in a for k in ("--hint", "--threads-per-core", "--ntasks-per-core", "--cpus-per-task"))


def write_job(request: Path, name: str, procs: int) -> Path:
    d = request / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={CLI}:{request.name}:{name}",
             f"#SBATCH --chdir={d}",
             f"#SBATCH --output={d / 'job.log'}",
             f"#SBATCH --ntasks={procs}",
             *(["#SBATCH --hint=nomultithread"] if _whole_cores() else []),
             f"#SBATCH --time={time_limit_min()}",
             "#SBATCH --export=ALL",
             f"exec {shlex.quote(sys.executable)} -m lammps_translator.cli run-study {shlex.quote(str(request))} "
             f"{shlex.quote(name)}"]
    p = d / "job.sh"
    p.write_text("\n".join(lines) + "\n")
    p.chmod(0o755)
    return p


def submit(request: Path, name: str, study: dict, estimate: float | None) -> dict:
    """Writes job.sh, runs sbatch; job.json on success, else the study's result says why (no local fallback)."""
    procs = study_procs(study)
    job = write_job(request, name, procs)
    cmd = ["sbatch", "--parsable", *shlex.split(os.environ.get("LMPT_SLURM_ARGS", "")), str(job)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=job.parent)
        ok = r.returncode == 0 and r.stdout.strip()
        msg = (r.stderr or r.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, msg, r = False, str(exc), None
    if not ok:
        res = {"ok": False, "name": name, "state": "failed", "why": f"could not submit: {msg[:500] or 'sbatch failed'}"}
        _atomic_json(request / name / f"{name}.json", res)
        return res
    jid = r.stdout.strip().split(";")[0].split()[0]
    info = {"executor": "slurm", "job_id": jid, "submitted": time.time(), "procs": procs,
            "time_min": time_limit_min(), "estimate_min": estimate, "command": cmd}
    _atomic_json(request / name / "job.json", info)
    return info


# ------------------------------------------------------------------------------------------- queue
def _run(cmd: list[str], timeout: float = 30) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def _minutes(t: str) -> float | None:
    """squeue %M (elapsed): [D-]HH:MM:SS or MM:SS."""
    m = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", t.strip())
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return round(d * 1440 + h * 60 + mi + s / 60.0, 1)


def queue() -> dict | None:
    """This user's jobs: {id: {state, reason, start, elapsed_min, name, cpus}}; None if squeue cannot be used."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = os.environ.get("USER", "")
    rc, out, err = _run(["squeue", "-h", "-u", user, "-o", "%i|%T|%r|%S|%M|%j|%C"])
    if rc != 0:
        return None
    q = {}
    for ln in out.splitlines():
        t = ln.strip().split("|")
        if len(t) < 7:
            continue
        q[t[0]] = {"state": t[1], "reason": t[2], "start": t[3], "elapsed_min": _minutes(t[4]), "name": t[5],
                   "cpus": int(t[6]) if t[6].isdigit() else 0}
    return q


def _ahead(jid: str) -> int | None:
    rc, out, _ = _run(["squeue", "-h", "-t", "PENDING", "-o", "%i"])
    if rc != 0:
        return None
    ids = [x.strip() for x in out.splitlines() if x.strip()]
    return ids.index(jid) if jid in ids else None


def _sacct(jid: str) -> str | None:
    rc, out, _ = _run(["sacct", "-n", "-X", "-P", "-j", jid, "-o", "State"])
    if rc != 0:
        return None
    for ln in out.splitlines():
        s = ln.strip()
        if s:
            return s.split()[0].upper()
    return None


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def ended_state(d: Path, jid: str) -> dict:
    """A job that left the queue without a result: what happened."""
    log = d / "job.log"
    tail = _tail(log)
    if (d / "cancelled.json").is_file():
        c = json.loads((d / "cancelled.json").read_text())
        return {"state": "cancelled", "why": f"cancelled by {c.get('by', 'cancel()')} at {c.get('at', '?')}"}
    s = _sacct(jid)
    if s:
        if s.startswith("TIMEOUT") or s.startswith("DEADLINE"):
            return {"state": "timed_out", "why": f"the job hit its SLURM time limit ({s}): raise LMPT_TIMEOUT "
                                                 "(the job asks for it + 10 min) or shorten the study (fewer MD steps, "
                                                 "a smaller cell), or give it more procs"}
        if s.startswith("OUT_OF_ME"):
            return {"state": "out_of_memory", "why": "out of memory: add --mem=... to LMPT_SLURM_ARGS, or use a "
                                                     "smaller cell"}
        if s.startswith("CANCELLED"):
            by = s.split("BY", 1)[1].strip() if "BY" in s else None
            return {"state": "cancelled", "why": "cancelled" + (f" by uid {by}" if by else "")}
        if s in ("PENDING", "RUNNING", "COMPLETING", "REQUEUED", "SUSPENDED", "CONFIGURING"):
            return {"state": "running" if s != "PENDING" else "queued", "why": None}
        if s.startswith("COMPLETED"):
            return {"state": "failed", "why": "the job ended without writing a result", "log_tail": tail}
        return {"state": "failed", "why": f"SLURM state {s}", "log_tail": tail}
    # no accounting: read job.log
    if "DUE TO TIME LIMIT" in tail:
        return {"state": "timed_out", "why": "the job hit its SLURM time limit (job.log): raise LMPT_TIMEOUT or "
                                             "shorten the study"}
    if "CANCELLED" in tail:
        return {"state": "cancelled", "why": "cancelled (job.log)"}
    if "oom" in tail.lower() and "kill" in tail.lower():
        return {"state": "out_of_memory", "why": "out of memory (job.log): add --mem=... to LMPT_SLURM_ARGS"}
    return {"state": "failed", "why": "the job is gone without a result" + ("" if log.is_file() else " (no job.log)"),
            "log_tail": tail}


def study_state(request: Path, name: str, q: dict | None) -> dict:
    """State of a SLURM study: done (with 'result'), queued, running, or one of FINAL (a result)."""
    d = request / name
    f = d / f"{name}.json"
    job = {}
    try:
        job = json.loads((d / "job.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    jid = str(job.get("job_id", ""))
    qi = q.get(jid) if (q is not None and jid) else None
    if f.is_file():
        try:
            return {"state": "done", "job_id": jid, "result": json.loads(f.read_text())}
        except json.JSONDecodeError:
            pass
    if not jid:
        return {"state": "failed", "why": "no job.json and no result (submission did not finish?)"}
    if q is None:
        return {"state": "unknown", "job_id": jid, "why": "squeue failed: cannot tell the job's state now"}
    if qi is not None:
        st = qi["state"]
        if st in ("PENDING", "REQUEUED", "SUSPENDED", "REQUEUE_HOLD", "RESV_DEL_HOLD"):
            out = {"state": "queued", "job_id": jid, "reason": qi["reason"]}
            if qi["start"] and qi["start"] not in ("N/A", "Unknown", "None"):
                out["expected_start"] = qi["start"]
            a = _ahead(jid)
            if a is not None:
                out["ahead"] = a
            return out
        from .api import progress
        out = {"state": "running", "job_id": jid, **progress(d)}
        if qi["elapsed_min"] is not None:
            out["job_elapsed_min"] = qi["elapsed_min"]
        return out
    # not in the queue: the result may have appeared just now
    if f.is_file():
        try:
            return {"state": "done", "job_id": jid, "result": json.loads(f.read_text())}
        except json.JSONDecodeError:
            pass
    e = ended_state(d, jid)
    if e["state"] in ("queued", "running"):
        return {"job_id": jid, **e}
    return {"job_id": jid, **e}


def final_result(name: str, st: dict) -> dict:
    """A finished-without-result state as the study's result (and written as <name>.json is not: the job may
    still be found later by sacct)."""
    out = {"ok": False, "name": name, "state": st["state"], "why": st.get("why"), "job_id": st.get("job_id")}
    if st.get("log_tail"):
        out["log_tail"] = st["log_tail"]
    return out


def cancel(request: Path, names: list[str]) -> dict:
    out = {}
    ids = {}
    for n in names:
        try:
            ids[n] = str(json.loads((request / n / "job.json").read_text())["job_id"])
        except (OSError, json.JSONDecodeError, KeyError):
            out[n] = {"cancelled": False, "why": "no SLURM job for this study"}
    if ids:
        for n in ids:
            if not (request / n / f"{n}.json").is_file():
                _atomic_json(request / n / "cancelled.json", {"by": "cancel()", "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        rc, o, e = _run(["scancel", *ids.values()])
        for n, j in ids.items():
            out[n] = {"cancelled": rc == 0, "job_id": j, **({} if rc == 0 else {"why": (e or o).strip()[:300]})}
    return out


def machine() -> dict | None:
    rc, out, _ = _run(["sinfo", "-h", "-o", "%C"])
    if rc != 0 or not out.strip():
        return None
    t = out.strip().splitlines()[0].split("/")
    if len(t) != 4:
        return None
    a, i, o, tot = (int(x) for x in t)
    return {"cpus": tot, "cpus_allocated": a, "cpus_idle": i}


def queue_summary(q: dict | None) -> dict | None:
    if q is None:
        return None
    mine = [v for v in q.values() if v["name"].startswith(f"{CLI}:")]
    run = [v for v in mine if v["state"] in ("RUNNING", "COMPLETING", "CONFIGURING")]
    return {"running": len(run), "queued": len([v for v in mine if v["state"] == "PENDING"]),
            "cpus_in_use": sum(v["cpus"] for v in run)}
