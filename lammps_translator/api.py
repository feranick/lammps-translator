"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.10.16.1

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks (builds the structures, finds potential files; no LAMMPS run)
    run(studies, wait)               runs and checks each study in a request folder; with wait, returns after that
                                     many seconds with the rest pending (progress and ETA)
    start(studies)                   the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending
    inputs(study, name)              the LAMMPS inputs of one study, without running
    bench(names)                     the benchmark studies with known answers
    read_file(path)                  a written file as base64 (only under the output folder)
    tool_status()                    versions, LAMMPS and its packages, potentials folders

Studies run in worker processes with a time limit each. Environment: LMPT_OUT (default ~/lmpt_runs),
LMPT_WORKERS (2), LMPT_TIMEOUT (seconds per study, 21600), LMPT_LMP (LAMMPS command), LMPT_PROCS (MPI ranks, 1),
LMPT_MPIRUN (mpirun), LMPT_POTENTIALS (extra potential folders, ':'-separated).
"""
from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

from . import __version__
from .format import study_format as _format
from .format import validate as _validate
from .format import validate_study


def out_root() -> Path:
    return Path(os.environ.get("LMPT_OUT", "~/lmpt_runs")).expanduser()


def tool_status() -> dict:
    from .lammps import lmp_info
    from .potential import potential_dirs

    info = lmp_info()
    out = {"translator": __version__, "lammps": {k: v for k, v in info.items() if k != "styles"},
           "potentials": [str(p) for p in potential_dirs()], "procs": int(os.environ.get("LMPT_PROCS", "1"))}
    try:
        import fea_translator  # type: ignore
        out["fea_translator"] = getattr(fea_translator, "__version__", "yes")
    except ImportError:
        out["fea_translator"] = None
    return out


def study_format() -> dict:
    return {**_format(), "translator": __version__, "status": tool_status()}


def validate(studies: dict) -> dict:
    return _validate(studies)


def _run_one(job: dict) -> dict:
    from .runner import run_study

    return run_study(job["name"], job["study"], Path(job["folder"]) / job["name"], job.get("timeout", 21600))


class _Pool:
    """Worker processes that live across calls; each study writes <request>/<name>/<name>.json when done."""

    def __init__(self):
        self.pool = None
        self.jobs: dict[str, dict] = {}

    def submit(self, folder: Path, jobs: list[dict]) -> None:
        workers = int(os.environ.get("LMPT_WORKERS", "2"))
        timeout = float(os.environ.get("LMPT_TIMEOUT", "21600"))
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=10)
        self.jobs[str(folder)] = {j["name"]: self.pool.apply_async(_run_one, (dict(j, timeout=timeout),)) for j in jobs}


_POOL = _Pool()


def _request_dir(prefix: str) -> Path:
    d = out_root() / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def progress(d: Path) -> dict:
    """Where a running study is: stage i of n, MD step of the stage, an ETA from the speed so far."""
    from .results import read_log

    pf = d / "progress.json"
    if not pf.is_file():
        return {"stage": "queued" if not d.exists() or not any(d.iterdir()) else "building"}
    try:
        p = json.loads(pf.read_text())
    except (OSError, json.JSONDecodeError):
        return {"stage": "running"}
    i, plan = p["current"], p["stages"]
    cur = plan[i]
    out = {"stage": cur["id"], "stage_index": i, "stages": len(plan), "kind": cur["kind"],
           "elapsed_min": round((time.time() - p["started"]) / 60, 1)}
    total = sum(s["steps"] for s in plan)
    if total == 0:
        out["note"] = "static stages only (minimizations, single points): no step count"
        return out
    log = d / cur["id"] / "log.lammps"
    done_cur = 0
    if log.is_file() and cur["steps"]:
        lg = read_log(log)
        done_cur = sum(l["steps"] for l in lg["loops"])
        if lg["blocks"]:
            b = lg["blocks"][-1]["rows"]
            if len(b) and len(lg["blocks"][-1]["columns"]) and b.shape[0]:
                last = int(b[-1, 0]) - int(b[0, 0])
                if not lg["loops"] or len(lg["blocks"]) > len(lg["loops"]):
                    done_cur += last
    done = sum(s["steps"] for s in plan[:i]) + min(done_cur, cur["steps"])
    out.update({"md_steps_done": done, "md_steps_total": total, "stage_steps_done": done_cur,
                "stage_steps": cur["steps"]})
    el_stage = time.time() - p.get("stage_started", p["started"])
    if done_cur > 0:
        per = el_stage / done_cur
        out["s_per_step"] = round(per, 5)
        out["eta_min_max"] = round(per * (total - done) / 60, 1)
        out["note"] = "eta from this stage's speed, for the MD steps left (static stages add a little)"
    return out


def _collect(folder: Path) -> dict:
    meta = json.loads((folder / "request.json").read_text())
    handles = _POOL.jobs.get(str(folder), {})
    out, pending = dict(meta.get("invalid", {})), []
    for name in meta["names"]:
        f = folder / name / f"{name}.json"
        h = handles.get(name)
        if f.is_file() and (h is None or h.ready()):
            try:
                out[name] = json.loads(f.read_text())
                continue
            except json.JSONDecodeError:
                pass
        if h is not None and h.ready():
            try:
                out[name] = h.get()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"ok": False, "why": "worker failed", "error": str(exc)[:500]}
        elif h is None:
            out[name] = {"ok": False, "why": "not finished and not running here (the server restarted?); run it again"}
        else:
            pending.append(name)
    for v in out.values():
        if isinstance(v, dict):
            v.pop("trace", None)
    res = {"request": str(folder), "translator": __version__, "done": not pending, "pending": pending,
           "elapsed_s": round(time.time() - meta.get("started", time.time()), 1), "studies": out}
    if pending:
        prog = {n: progress(folder / n) for n in pending}
        res["progress"] = prog
        eta = max((p.get("eta_min_max") or 0) for p in prog.values())
        res["next"] = (f"call study_status(request='{folder}') to collect the rest. Polling costs tokens: if eta_min_max "
                       f"(now {eta:.0f} min) is more than about 5 min, tell the user when to come back and stop polling")
    else:
        (folder / "results.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def start(studies: dict) -> Path:
    folder = _request_dir("run")
    (folder / "studies.json").write_text(json.dumps(studies, indent=1))
    invalid, jobs = {}, []
    if not isinstance(studies, dict):
        studies = {}
    for n, st in studies.items():
        v = validate_study(st)
        if not v["ok"]:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": v["errors"], "warnings": v["warnings"]}
        else:
            jobs.append({"name": n, "study": st, "folder": str(folder)})
    (folder / "request.json").write_text(json.dumps({"names": [j["name"] for j in jobs], "invalid": invalid,
                                                     "started": time.time()}, default=float))
    if jobs:
        _POOL.submit(folder, jobs)
    return folder


def status(request: str, wait: float = 0.0) -> dict:
    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    t_end = time.time() + max(0.0, float(wait))
    while True:
        res = _collect(folder)
        if res["done"] or time.time() >= t_end:
            return res
        time.sleep(1.0)


def run(studies: dict, wait: float | None = None) -> dict:
    """Runs the studies into a new request folder under LMPT_OUT: per study <name>/ (initial.data, one folder per
    stage with in.lammps and log.lammps, run_all.sh, <name>.json), <name>.tgz; studies.json, results.json."""
    folder = start(studies)
    return status(str(folder), 1e9 if wait is None else wait)


def inputs(study: dict, name: str = "study") -> dict:
    from .runner import inputs as _inputs

    v = validate_study(study)
    if not v["ok"]:
        return {"error": "invalid study", "errors": v["errors"]}
    folder = _request_dir("inputs") / name
    try:
        r = _inputs(name, study, folder)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    r["warnings"] = v["warnings"]
    return r


def bench(names=None) -> dict:
    from . import bench as B

    folder = _request_dir("bench")
    r = B.run(names, folder)
    passed = [k for k, v in r.items() if v.get("passed")]
    failed = [k for k, v in r.items() if v.get("passed") is False or v.get("ok") is False]
    out = {"request": str(folder), "translator": __version__, "passed": passed, "failed": failed,
           "skipped": [k for k, v in r.items() if "skipped" in v], "cases": r}
    (folder / "bench.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def read_file(path: str, max_mb: float = 20) -> dict:
    p = Path(path).expanduser().resolve()
    root = out_root().resolve()
    if root not in p.parents:
        return {"error": f"only files under {root} can be read"}
    if not p.is_file():
        return {"error": f"no file {p}"}
    if p.stat().st_size > max_mb * 1e6:
        return {"error": f"{p.stat().st_size / 1e6:.1f} MB is more than max_mb"}
    return {"name": p.name, "bytes": p.stat().st_size, "base64": base64.b64encode(p.read_bytes()).decode()}
