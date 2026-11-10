"""api.py — what other programs and the MCP server call. JSON in, JSON out.
Version: 2026.11.09.3

    study_format()                   the study format, for the caller to read
    validate(studies)                static checks (builds the structures, finds potential files; no LAMMPS run)
    run(studies, wait)               runs and checks each study in a request folder; with wait, returns after that
                                     many seconds with the rest pending (progress and ETA)
    start(studies)                   the same in the background: returns the request folder
    status(request, wait)            finished studies with results, the others pending
    inputs(study, name)              the LAMMPS inputs of one study, without running
    bench(names, wait)               the benchmark studies with known answers (submitted; returns within wait)
    bench_status(bench, wait)        evaluates the finished benchmarks of a bench request
    read_file(path)                  a written file as base64 (only under the output folder)
    tool_status()                    versions, LAMMPS and its packages, potentials folders, executor
    jobs(limit, active)              recent requests with SLURM jobs and their states; the queue; the machine
    cancel(request, names)           stops a request's studies (scancel; locally only those this process runs)

Studies run in worker processes with a time limit each, or (LMPT_EXECUTOR=slurm) as one SLURM job per study
(executor.py: llmt-runtime's SLURM layer from 2026.11.09.3; jobs and cancel keep their shapes, built from its states,
queue position, sacct, scancel and machine CPUs; run_studies keeps its budgeted submission). Environment: LMPT_OUT
(default ~/lmpt_runs), LMPT_WORKERS (2), LMPT_TIMEOUT (seconds per study, 21600), LMPT_LMP (LAMMPS command), LMPT_PROCS (MPI ranks, 1), LMPT_MPIRUN (mpirun), LMPT_POTENTIALS (extra
potential folders, ':'-separated), LMPT_EXECUTOR, LMPT_SLURM_MIN_MINUTES (1), LMPT_SLURM_ARGS.
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
from . import executor as X
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
           "potentials": [str(p) for p in potential_dirs()], "procs": int(os.environ.get("LMPT_PROCS", "1")),
           "executor": X.status()}
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

    from . import executor as X_
    return run_study(job["name"], job["study"], Path(job["folder"]) / job["name"], X_.timeout_s(job["study"]))


class _Pool:
    """Worker processes that live across calls; each study writes <request>/<name>/<name>.json when done."""

    def __init__(self):
        self.pool = None
        self.jobs: dict[str, dict] = {}

    def submit(self, folder: Path, jobs: list[dict]) -> None:
        workers = int(os.environ.get("LMPT_WORKERS", "2"))
        timeout = float(os.environ.get("LMPT_TIMEOUT", "21600"))
        if self.pool is None:
            from .workers import watch_parent       # every worker (and its LAMMPS runs) ends with this server
            self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=10, initializer=watch_parent,
                                                     initargs=(os.getpid(),))
        self.jobs[str(folder)] = {j["name"]: self.pool.apply_async(_run_one, (dict(j, timeout=timeout),)) for j in jobs}


_POOL = _Pool()


from llmt_runtime.requests import REQUEST_NAME  # noqa: E402,F401 (run_<id>: the router names requests)


def _request_dir(prefix: str, name: str | None = None) -> Path:
    """A new request folder: <LMPT_OUT>/<prefix>_<time>_<hex>, or <LMPT_OUT>/<name> when the caller chose it
    (request_name; FileExistsError if it is taken)."""
    if name:
        d = out_root() / name
        out_root().mkdir(parents=True, exist_ok=True)
        d.mkdir()
        return d
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


def _local_state(folder: Path, name: str) -> dict:
    p = progress(folder / name)
    running = (folder / name / "progress.json").is_file()
    return {"state": "running" if running else "queued", **({} if running else {"reason": "waiting for a free worker"}),
            **p}


def _collect(folder: Path) -> dict:
    meta = json.loads((folder / "request.json").read_text())
    handles = _POOL.jobs.get(str(folder), {})
    where = meta.get("executor", {})
    out, pending, prog = dict(meta.get("invalid", {})), [], {}
    slurm = [n for n in meta["names"] if where.get(n) == "slurm"]
    q = None
    if any(not (folder / n / f"{n}.json").is_file() for n in slurm):
        q = X.queue()
    for name in meta["names"]:
        if where.get(name) == "slurm":
            st = X.study_state(folder, name, q)
            if st["state"] == "done":
                out[name] = st["result"]
            elif st["state"] in X.FINAL:
                out[name] = X.final_result(name, st)
            else:
                pending.append(name)
                prog[name] = st
            continue
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
            prog[name] = _local_state(folder, name)
    for v in out.values():
        if isinstance(v, dict):
            v.pop("trace", None)
    # the version the studies ran with (their results', else request.json's), and translator_now when this server
    # runs another (llmt-runtime's ran_with; before 2026.11.09.1 the installed version, whatever the run used)
    from llmt_runtime import requests as R

    from .runtime import RT
    res = {"request": str(folder), **R.ran_with(RT, meta, out), "done": not pending, "pending": pending,
           "elapsed_s": round(time.time() - meta.get("started", time.time()), 1), "studies": out}
    if pending:
        res["progress"] = prog
        eta = max((p.get("eta_min_max") or 0) for p in prog.values())
        jobs_ = [f"{n}: {prog[n]['job_id']}" for n in pending if prog[n].get("job_id")]
        if jobs_:
            res["next"] = (f"The studies are SLURM jobs on this machine (ids {', '.join(jobs_)}). They keep running "
                           f"without this conversation. Tell the user when they should finish (eta_min_max now "
                           f"{eta:.0f} min for the running ones; queued ones wait for cores) and stop; collect them "
                           f"later with study_status(request='{folder}') or find them with jobs().")
        else:
            res["next"] = (f"call study_status(request='{folder}') to collect the rest. Polling costs tokens: if "
                           f"eta_min_max (now {eta:.0f} min) is more than about 5 min, tell the user when to come back "
                           "and stop polling")
    else:
        X._atomic_json(folder / "results.json", res)
    return res


def start(studies: dict, prefix: str = "run", extra: dict | None = None, budget_s: float = 30.0,
          request_name: str | None = None) -> Path:
    """Validates, then submits each valid study: a SLURM job (LMPT_EXECUTOR=slurm and long enough, or
    "queue": true) or the local worker pool. Submissions that would run past budget_s go on in a background
    thread (their studies show as queued, 'submitting'), so a tool call returns in time."""
    t0 = time.time()
    folder = _request_dir(prefix, request_name)
    (folder / "studies.json").write_text(json.dumps(studies, indent=1))
    invalid, jobs, where, est, natoms = {}, [], {}, {}, {}
    if not isinstance(studies, dict):
        studies = {}
    for n, st in studies.items():
        v = validate_study(st)
        if not v["ok"]:
            invalid[n] = {"ok": False, "why": "invalid study", "errors": v["errors"], "warnings": v["warnings"]}
            continue
        e = X.estimate(st) if X.kind() == "slurm" else {"minutes": None, "natoms": None}
        est[n] = e["minutes"]
        natoms[n] = e["natoms"]
        where[n] = "slurm" if X.goes_to_slurm(st, est[n]) else "local"
        if where[n] == "local":
            jobs.append({"name": n, "study": st, "folder": str(folder)})
    names = [n for n in studies if n in where]
    from llmt_runtime import requests as R

    from .runtime import RT
    # with the versions it is submitted with ("translator", llmt-runtime's "runtime"): status says what it ran with
    X._atomic_json(folder / "request.json", R.request_meta(
        RT, names=names, invalid=invalid, started=time.time(), executor=where, estimate_min=est,
        studies={n: studies[n] for n in names}, **(extra or {})))
    if jobs:
        _POOL.submit(folder, jobs)
    todo = [n for n in names if where[n] == "slurm"]
    while todo and time.time() - t0 < budget_s:
        n = todo.pop(0)
        X.submit(folder, n, studies[n], est[n], natoms[n])
    if todo:
        import threading

        def rest():
            for n in todo:
                X.submit(folder, n, studies[n], est[n], natoms[n])
        threading.Thread(target=rest, name="lmpt-submit", daemon=False).start()
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
        time.sleep(2.0 if res.get("pending") else 1.0)


def run(studies: dict, wait: float | None = None, request_name: str | None = None) -> dict:
    """Runs the studies into a new request folder under LMPT_OUT: per study <name>/ (initial.data, one folder per
    stage with in.lammps and log.lammps, run_all.sh, <name>.json; with SLURM job.sh, job.json, job.log),
    <name>.tgz; studies.json, request.json, results.json.

    request_name ('run_<id>', optional): the folder's name, chosen by the caller (llm-translators' router) so it
    knows the request path even when the answer does not arrive in time. Repeating a call with the same name
    returns that request's status and submits nothing again."""
    from llmt_runtime import requests as R

    from .runtime import RT
    kind_, found = R.check_request_name(RT, request_name)     # llmt-runtime: new | repeat (its status) | error
    if kind_ == "error":
        return found
    if kind_ == "repeat":
        res = status(str(found), 0.0 if wait is None else min(float(wait), 15.0))
        return dict(res, note=R.REPEAT_NOTE)
    t0 = time.time()
    try:
        folder = start(studies, budget_s=30.0 if wait is None else max(1.0, 0.6 * float(wait)),
                       request_name=request_name)
    except FileExistsError:
        return {"error": f"request_name {request_name!r} is being submitted by another call; call "
                         "study_status(request) in a moment", "request": str(out_root() / request_name)}
    if wait is None:
        return status(str(folder), 1e9)
    return status(str(folder), max(0.0, float(wait) - (time.time() - t0)))


def run_one(request: str, name: str) -> int:
    """The SLURM job's command (lmpt run-study): runs one study of a request as a pool worker would."""
    from llmt_runtime import jobs as J

    from .runner import run_study

    folder = Path(request).expanduser()
    meta = json.loads((folder / "request.json").read_text())
    study = meta["studies"][name]
    procs = int(os.environ.get("SLURM_NTASKS") or 0) or X.procs_of(study)
    job = J.read_job(folder / name)
    # the study's own timeout as its job was sized (job.json timeout_s: shorter with LMPT_SLURM_TIME=estimate)
    timeout = J.job_timeout_s(folder / name, X.timeout_s(study))
    res = run_study(name, study, folder / name, timeout, procs,
                    executor="slurm" if os.environ.get("SLURM_JOB_ID") else "by hand")
    if not res.get("ok") and "timed out" in str(res.get("why", "")):
        res["state"] = "timed_out"
        if job.get("time_from") == "estimate":
            res["why"] += (f" — the job's limit came from the estimate ({job.get('estimate_min')} min x "
                           f"LMPT_SLURM_TIME_FACTOR): rerun with LMPT_SLURM_TIME=timeout or a larger factor")
        X._atomic_json(folder / name / f"{name}.json", res)
    return 0


def jobs(limit: int = 20, active: bool = False) -> dict:
    """The newest requests under LMPT_OUT with SLURM jobs, their studies' states; the queue and the machine."""
    root = out_root()
    reqs = []
    if root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and (d / "request.json").is_file() and any(d.glob("*/job.json")):
                reqs.append(d)
    reqs.sort(key=lambda d: (d / "request.json").stat().st_mtime, reverse=True)
    from llmt_runtime import slurm as S

    q = X.queue()
    out = []
    for d in reqs:
        try:
            meta = json.loads((d / "request.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        left = S.remaining()            # within an MCP call's deadline (llmt-runtime's), a request not checked says so
        if left is not None and left < 5:
            out.append({"request": str(d), "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                   time.localtime(meta.get("started", 0))),
                        "note": "not checked: SLURM was slow and the call ran out of time; ask again"})
            if len(out) >= limit:
                break
            continue
        sts = {}
        for n in meta.get("names", []):
            if meta.get("executor", {}).get(n) != "slurm":
                continue
            st = X.study_state(d, n, q)
            row = {"state": st["state"], "job_id": st.get("job_id")}
            if st["state"] == "done":
                r = st["result"]
                row.update({"ok": r.get("ok"), "reliable": r.get("reliable")})
                if not r.get("ok"):
                    row["why"] = r.get("why")
            for k in ("reason", "expected_start", "ahead", "eta_min_max", "stage", "job_elapsed_min", "why"):
                if k in st and k not in row:
                    row[k] = st[k]
            sts[n] = row
        is_active = any(v["state"] in ("queued", "running", "unknown") for v in sts.values())
        if active and not is_active:
            continue
        out.append({"request": str(d), "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                                               time.localtime(meta.get("started", 0))),
                    "active": is_active, "studies": sts})
        if len(out) >= limit:
            break
    return {"translator": __version__, "executor": X.status(), "queue": X.queue_summary(q), "machine": X.machine(),
            "requests": out}


def cancel(request: str, names: list[str] | None = None) -> dict:
    folder = Path(request).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no request {folder}"}
    meta = json.loads((folder / "request.json").read_text())
    names = names or meta["names"]
    where = meta.get("executor", {})
    out = {}
    unknown = [n for n in names if n not in meta["names"]]
    for n in unknown:
        out[n] = {"cancelled": False, "why": f"no study '{n}' in this request ({meta['names']})"}
    sl = [n for n in names if where.get(n) == "slurm" and n not in unknown]
    out.update(X.cancel(folder, sl))
    handles = _POOL.jobs.get(str(folder), {})
    for n in names:
        if n in unknown or where.get(n) == "slurm":
            continue
        h = handles.get(n)
        if (folder / n / f"{n}.json").is_file() and (h is None or h.ready()):
            out[n] = {"cancelled": False, "why": "already finished"}
        elif h is not None and not h.ready():
            (folder / n).mkdir(parents=True, exist_ok=True)
            (folder / n / "CANCEL").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            out[n] = {"cancelled": True, "how": "LAMMPS is stopped and the study ends as cancelled"}
        else:
            out[n] = {"cancelled": False, "why": "it does not run in this server process (local executor): it "
                                                "cannot be stopped from here"}
    time.sleep(1.0)
    return {"request": str(folder), "cancel": out, "status": _collect(folder)}


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


def bench(names=None, request: str | None = None, wait: float | None = None, queue: bool | None = None) -> dict:
    """Benchmarks, submitted like studies: to the worker pool, or with LMPT_EXECUTOR=slurm as SLURM jobs (long ones;
    queue=True: all). Waits at most wait seconds (None: until done, for the command line), then reports the
    finished cases; the others are pending: evaluate them later with bench_status(request)."""
    from . import bench as B

    if request is not None:
        return bench_status(request, wait)
    t0 = time.time()
    studies, pre = B.prepare(names)
    if queue is not None:
        studies = {n: {**st, "queue": bool(queue)} for n, st in studies.items()}
    folder = start(studies, prefix="bench", extra={"bench": list(studies), "bench_pre": pre},
                   budget_s=30.0 if wait is None else max(1.0, 0.6 * float(wait)))
    return bench_status(str(folder), None if wait is None else max(0.0, float(wait) - (time.time() - t0)))


def bench_status(bench: str, wait: float | None = None) -> dict:
    """The benchmarks of a bench request: rows for the finished cases (evaluated here, the fea-translator chain in a
    background thread), the others pending. Waits up to wait seconds (None: until all are done)."""
    from . import bench as B

    folder = Path(bench).expanduser()
    if not (folder / "request.json").is_file():
        return {"error": f"no benchmark request {folder}"}
    meta = json.loads((folder / "request.json").read_text())
    t_end = time.time() + (1e9 if wait is None else max(0.0, float(wait)))
    while True:
        res = _collect(folder)
        r = dict(meta.get("bench_pre", {}))
        pending = []
        for n in meta.get("bench", []):
            if n in res["studies"]:
                r[n] = B.evaluate(n, res["studies"][n], folder)
                if r[n].get("pending"):
                    pending.append(n)
            else:
                pending.append(n)
                r[n] = {"pending": True, **{k: v for k, v in res.get("progress", {}).get(n, {}).items()
                                           if k in ("state", "job_id", "reason", "eta_min_max", "stage")}}
        if not pending or time.time() >= t_end:
            break
        time.sleep(2.0)
    out = _bench_out(folder, r, {k: res[k] for k in ("translator", "translator_now", "translator_note") if k in res})
    if pending:
        out["pending"] = pending
        out["next"] = (f"call bench_status(bench='{folder}') later to evaluate the rest. " + res.get("next", "")).strip()
    return out


def _bench_out(folder: Path, r: dict, ver: dict | None = None) -> dict:
    """ver: the versions the benchmarks ran with (_collect's ran_with), else this server's."""
    passed = [k for k, v in r.items() if v.get("passed")]
    failed = [k for k, v in r.items() if v.get("passed") is False or v.get("ok") is False]
    out = {"request": str(folder), **(ver or {"translator": __version__}), "passed": passed, "failed": failed,
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
