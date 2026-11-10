"""executor.py — where studies run: the local worker pool or SLURM jobs (llm-translators docs/EXECUTION.md).
Version: 2026.11.09.4

2026.11.09.4: queued_note / print_queued_note, what `lmpt run` and `lmpt bench` say when interrupted while waiting for
SLURM jobs (the job ids still queued and how to cancel them; nothing is cancelled).

From 2026.11.09.3 the SLURM layer is llmt-runtime's (llmt_runtime.slurm and llmt_runtime.jobs, configured by
runtime.RT): the settings (LMPT_EXECUTOR, LMPT_SLURM_ARGS split by RT.slurm.slurm_args, LMPT_SLURM_MEM, LMPT_SLURM_NODE,
LMPT_SLURM_TIME, caps), the machine (`scontrol -o show node`: a success kept 60 s, a failure 5 s; this node by name or
address; node lists; a node's cores and memory), job.sh (--ntasks=N, --ntasks-per-core=1 unless LMPT_SLURM_ARGS places
the ranks, --time, --mem, --nodelist) and its submission with job.json, the queue position by priority, sacct, how a job
that left the queue ended (also one squeue still lists in a final state), scancel and the machine's CPUs. The names
below stay as thin wrappers with lmpt's signatures and result shapes (api, runner, format and the tests call them).
2026.11.09.1 (phase 1): ended_state's rules (jobs.classify_end) and the runtime's version.

What stays lmpt's, reaching the runtime as hooks on runtime.RT (procs, mem_auto, job_extra, running_progress,
oom_advice, timeout_hint):
  sizing    atoms() (the built structure's atom count, kept 60 s per study), ranks_for (procs capped at
            ceil(atoms / LMPT_ATOMS_PER_RANK) and the node's physical cores), the auto memory (1 GB + 256 MB per rank +
            4 kB per atom, at least 2 GB, rounded up to 512 MB, within the caps), estimate_minutes and the per-study
            validation (check_run, sizing)
  job.json  natoms, estimate_min (unrounded: validation's 'minutes'), procs_requested, and time_min / time_from as the
            job's own limit (the study's run.timeout_s reads 'study')
  states    study_state (one squeue of this user's jobs per call, queue()), LAMMPS's progress from its logs
            (api.progress), cancel()'s cancelled.json ({by, at}; RT.cancel_file), final_result, queue_summary(q)

Settings (environment): LMPT_EXECUTOR local | slurm (local); LMPT_SLURM_MIN_MINUTES (1): studies estimated
shorter run locally; LMPT_SLURM_ARGS: extra sbatch arguments; LMPT_PROCS (MPI ranks, --ntasks); LMPT_TIMEOUT
(seconds per study). A study's "queue": true / false forces SLURM / local (not with LMPT_EXECUTOR=local).

Job limits:
  --time   LMPT_SLURM_TIME=timeout (default): ceil(LMPT_TIMEOUT/60) + 10 min. =estimate: max(LMPT_SLURM_TIME_MIN (20),
           ceil(LMPT_SLURM_TIME_FACTOR (3) x the estimate) + 10) min, at most the former; the study's own timeout is
           10 min less (job.json timeout_s, which run-study reads). =max: the partition's limit (UNLIMITED where it has
           none). =minutes: that many, at least 'timeout'. Short limits let SLURM backfill short studies around long
           jobs; a study that outruns its estimate ends timed_out (rerun with LMPT_SLURM_TIME=timeout).
  --mem    LMPT_SLURM_MEM=auto (default): 1 GB + 256 MB per rank + 4 kB per atom, rounded up to 512 MB, at least
           2 GB; a size (8G, 8000M) as given; none: SLURM's default (on nodes with CR_Core_Memory and no DefMemPerCPU
           that is the whole node, and jobs run one at a time). A --mem / --mem-per-cpu in LMPT_SLURM_ARGS wins.
  --ntasks-per-core=1 (one physical core per rank) unless LMPT_SLURM_ARGS sets the core/thread layout
           (--hint, --threads-per-core, --ntasks-per-core, --cpus-per-task / -c).
  --ntasks min(procs, ceil(atoms / LMPT_ATOMS_PER_RANK (2000))): no more ranks than the study can use.

Per study (docs/EXECUTION.md, "Per-study run settings"): "run": {timeout_s, mem, procs, queue} overrides
LMPT_TIMEOUT (and so --time), --mem (replacing a --mem / --mem-per-cpu in LMPT_SLURM_ARGS), LMPT_PROCS and the queue
decision for that study, within LMPT_MAX_TIMEOUT, LMPT_MAX_MEM, LMPT_MAX_PROCS (unset: no cap).

Clusters of several nodes: LMPT_SLURM_NODE = local (default: --nodelist=<this node> when scontrol lists more than
one node, because run folders are in this machine's home) | any (SLURM chooses: shared homes) | a node list. A
--nodelist / -w in LMPT_SLURM_ARGS wins. Ranks are capped at that node's physical cores.

One SLURM job per study: <request>/<name>/job.sh (complete, rerunnable with sbatch) and job.json. States come from
the files and the queue (squeue, then sacct, then job.log), never from in-process handles, so any server process,
after restarts or from another conversation, can collect them.
"""
from __future__ import annotations

import getpass
import json
import math
import os
import shutil
import time
from pathlib import Path

from llmt_runtime import jobs as J
from llmt_runtime import slurm as S

from .runtime import RT

CLI = "lmpt"
FINAL = J.FINAL            # ("timed_out", "out_of_memory", "cancelled", "failed")


# ------------------------------------------------------------------------------------------- settings
def kind() -> str:
    return S.kind(RT)


RUN_KEYS = ("timeout_s", "mem", "procs", "queue")
run_of = S.run_of          # the study's "run" block, {} when absent
parse_mem_mb = S.mem_mb    # SLURM sizes: 500M, 8G, 1T, 2048 (MB), 512K; None if malformed


def timeout_s(study: dict | None = None) -> float:
    """The study's run.timeout_s, else LMPT_TIMEOUT (21600 s)."""
    return S.timeout_s(RT, study)


def time_limit_min(study: dict | None = None) -> int:
    """ceil(timeout / 60) + 10: the job's limit with LMPT_SLURM_TIME=timeout (llmt-runtime's time_need_min)."""
    return S.time_need_min(RT, study)


def caps() -> dict:
    """What a study's run may ask (None: no cap): LMPT_MAX_TIMEOUT (s) and LMPT_MAX_PROCS; mem is the smaller of
    LMPT_MAX_MEM and, with SLURM, the memory of the node(s) jobs go to (RealMemory, exactly as SLURM gives it), with
    mem_mb in MB for llm-translators' router and mem_from saying which. A job asking for more than a node has fails
    only at sbatch ('Memory specification can not be satisfied'), so validation refuses it first. llmt-runtime's caps,
    in lmpt's shape (every key present)."""
    c = S.caps(RT)
    out = {"timeout_s": c.get("timeout_s"), "mem": c.get("mem"), "procs": c.get("procs")}
    if c.get("mem_mb"):
        out["mem_mb"] = c["mem_mb"]
        out["mem_from"] = c.get("mem_from")
    return out


def check_run(study: dict) -> list[str]:
    """Errors for a study's run block and top-level procs/queue against the caps (each names the fix)."""
    E = []
    r = study.get("run")
    c = caps()
    if r is not None:
        if not isinstance(r, dict):
            return ['run: an object {timeout_s, mem, procs, queue} (all optional)']
        unknown = sorted(set(r) - set(RUN_KEYS))
        if unknown:
            E.append(f"run: unknown keys {unknown}; allowed {list(RUN_KEYS)} (no sbatch arguments from a study)")
        if "timeout_s" in r:
            t = r["timeout_s"]
            if not (isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0):
                E.append("run.timeout_s: seconds, a positive number")
            elif c["timeout_s"] is not None and t > c["timeout_s"]:
                E.append(f"run.timeout_s: {t:g} s is above LMPT_MAX_TIMEOUT ({c['timeout_s']:g} s)")
        if "mem" in r:
            mb = parse_mem_mb(r["mem"]) if isinstance(r["mem"], str) else None
            if mb is None or mb <= 0:
                E.append("run.mem: a SLURM size such as '500M', '8G' or '1T'")
            elif c["mem"] and parse_mem_mb(c["mem"]) and mb > parse_mem_mb(c["mem"]):
                E.append(f"run.mem: {r['mem']} is above {c['mem']}, the most on this machine "
                         f"({c.get('mem_from') or 'LMPT_MAX_MEM'}); ask for less, or a machine with more "
                         "(machines(): executor.caps.mem)")
        if "procs" in r:
            if not (isinstance(r["procs"], int) and not isinstance(r["procs"], bool) and r["procs"] >= 1):
                E.append("run.procs: MPI ranks, an integer >= 1")
            elif "procs" in study and study["procs"] != r["procs"]:
                E.append(f"run.procs ({r['procs']}) disagrees with procs ({study['procs']}): give one of them")
        if "queue" in r:
            if not isinstance(r["queue"], bool):
                E.append("run.queue: true (a SLURM job) or false (local)")
            elif "queue" in study and study["queue"] != r["queue"]:
                E.append(f"run.queue ({r['queue']}) disagrees with queue ({study['queue']}): give one of them")
    p = r.get("procs") if isinstance(r, dict) and isinstance(r.get("procs"), int) else study.get("procs")
    if isinstance(p, int) and c["procs"] is not None and p > c["procs"]:
        E.append(f"{'run.procs' if isinstance(r, dict) and 'procs' in r else 'procs'}: {p} is above LMPT_MAX_PROCS "
                 f"({c['procs']})")
    return E


def from_study(study: dict | None) -> list[str]:
    return [k for k in RUN_KEYS if k in run_of(study)]


def min_minutes() -> float:
    return S.min_minutes(RT)


def default_procs() -> int:
    return S.default_procs(RT)


def procs_of(study: dict | None) -> int:
    """Ranks asked for: run.procs, then the study's procs, then LMPT_PROCS (llmt-runtime's procs_of)."""
    return S.procs_of(RT, study)


def study_procs(study: dict) -> int:
    return procs_of(study)


def atoms_per_rank() -> int:
    try:
        return max(1, int(float(RT.env("ATOMS_PER_RANK", "2000") or 2000)))
    except ValueError:
        return 2000


def ranks_for(study: dict, natoms: int | None) -> int:
    """The size cap: min(procs, ceil(atoms / LMPT_ATOMS_PER_RANK), this node's physical cores); procs (capped at
    the cores) when the atoms are not known. Jobs ask for one rank per core (--ntasks-per-core=1). The runtime's
    hooks.procs."""
    p = procs_of(study)
    cores = node_cores()
    if cores:
        p = min(p, cores)
    if not natoms:
        return max(1, p)
    return max(1, min(p, int(math.ceil(natoms / float(atoms_per_rank())))))


def queue_of(study: dict | None) -> bool | None:
    r = run_of(study)
    if "queue" in r:
        return bool(r["queue"])
    q = (study or {}).get("queue")
    return q if isinstance(q, bool) else None


def slurm_args_tokens() -> list[str]:
    """LMPT_SLURM_ARGS split as a shell would (it may hold several options, e.g. '--mem=8G --nodelist=x');
    llmt-runtime's slurm_args."""
    return RT.slurm.slurm_args()


def sbatch_args(study: dict | None = None) -> list[str]:
    """LMPT_SLURM_ARGS, without --mem / --mem-per-cpu when the study sets run.mem (llmt-runtime's sbatch_args)."""
    return S.sbatch_args(RT, study)


def sbatch_path() -> str | None:
    return shutil.which("sbatch")


def time_mode() -> str:
    """LMPT_SLURM_TIME as read: timeout (default) | estimate | max | minutes."""
    return S.time_setting(RT)


def job_time(estimate: float | None, study: dict | None = None) -> tuple:
    """(the job's --time: minutes, or 'UNLIMITED' with LMPT_SLURM_TIME=max on a partition without a limit; how it was
    chosen: study | timeout | estimate | max | minutes). llmt-runtime's job_time; a study's run.timeout_s reads
    'study', as lmpt had it."""
    t, info = S.job_time(RT, study, estimate)
    how = info.get("time_from") or "timeout"
    if run_of(study).get("timeout_s") and how in ("timeout", "study"):
        how = "study"
    return (int(t) if str(t).isdigit() else t), how


def mem_setting() -> str:
    return RT.env("SLURM_MEM", "auto").strip() or "auto"


def _auto_mem_mb(procs: int, natoms: int | None, cap: bool = True) -> int:
    """LMPT_SLURM_MEM=auto: 1 GB + 256 MB per rank + 4 kB per atom (100000 atoms when unknown), rounded up to 512 MB,
    at least 2 GB; with cap, never above caps().mem_mb (LMPT_MAX_MEM or the node's memory)."""
    mb = 1024 + 256 * procs + 0.004 * (natoms or 100000)
    mb = max(2048, int(math.ceil(mb / 512.0)) * 512)
    c = caps().get("mem_mb") if cap else None
    if c and mb > c:
        mb = int(c)            # what a node has: asking for more fails at sbatch
    return mb


def mem_auto(study: dict) -> str:
    """The runtime's hooks.mem_auto: lmpt's auto memory for this study's ranks and atoms."""
    n = atoms(study)
    return f"{_auto_mem_mb(ranks_for(study, n), n)}M"


def job_mem(procs: int, natoms: int | None, study: dict | None = None) -> str | None:
    """The job's --mem value (llmt-runtime's mem_plan rules, lmpt's auto): the study's run.mem, else None when
    LMPT_SLURM_ARGS has a --mem / --mem-per-cpu or LMPT_SLURM_MEM=none, else LMPT_SLURM_MEM (auto: from ranks and
    atoms). Sizes are written as sbatch takes them ('1.5G' becomes '1536M')."""
    r = run_of(study).get("mem")
    if r:
        return S.mem_arg(r) or str(r).strip()
    if S.args_mem(RT) is not None:
        return None
    m = mem_setting()
    if m.lower() == "none":
        return None
    if m.lower() != "auto":
        return S.mem_arg(m) or m
    return f"{_auto_mem_mb(procs, natoms)}M"


def mem_mb_for(procs: int, natoms: int | None, study: dict | None = None) -> int:
    """The memory a study's job reserves, in MB: run.mem, else a --mem / --mem-per-cpu in LMPT_SLURM_ARGS, else
    LMPT_SLURM_MEM (a size, or auto: from ranks and atoms; with none, the auto value as what it needs)."""
    m = job_mem(procs, natoms, study)
    if m is None:
        for t in slurm_args_tokens():
            if t.startswith("--mem-per-cpu="):
                v = parse_mem_mb(t.split("=", 1)[1])
                if v:
                    return int(math.ceil(v * procs))
            elif t.startswith("--mem="):
                v = parse_mem_mb(t.split("=", 1)[1])
                if v:
                    return int(math.ceil(v))
        return _auto_mem_mb(procs, natoms, cap=False)
    v = parse_mem_mb(m)
    return int(math.ceil(v)) if v else 2048


def sizing(study: dict) -> dict:
    """What a study's run is sized as, before running: {'minutes' (upper estimate with these ranks), 'procs'
    (ranks after the size cap), 'mem_mb'} — the same numbers submit() puts in job.sh. For validate_studies
    (llm-translators' router places runs with them)."""
    e = estimate(study)
    procs = ranks_for(study, e["natoms"])
    return {"minutes": e["minutes"], "procs": procs, "mem_mb": mem_mb_for(procs, e["natoms"], study)}


def _args_mem_text() -> str | None:
    """The --mem / --mem-per-cpu option of LMPT_SLURM_ARGS as written ('--mem=8G'), else None."""
    toks = slurm_args_tokens()
    for i, t in enumerate(toks):
        k = t.split("=", 1)[0]
        if k in ("--mem", "--mem-per-cpu"):
            return t if "=" in t else f"{t}={toks[i + 1] if i + 1 < len(toks) else ''}"
    return None


def _mem_status() -> str:
    a = _args_mem_text()
    return f"{a} (from LMPT_SLURM_ARGS)" if a else mem_setting()


def _env_num(name: str, default: float) -> float:
    try:
        return float(RT.env(name, "") or default)
    except ValueError:
        return float(default)


def status() -> dict:
    """tool_status().executor: lmpt's keys (kind, sbatch, min_minutes, procs, time_limit_min, time, mem, slurm_args,
    workers, timeout_s, atoms_per_rank, caps, run_keys, node {setting, nodelist, cluster_nodes, this_node, cores,
    warning?}, time_factor / time_min_min with time 'estimate', error when sbatch is missing) and llmt-runtime's
    version (runtime). 'time' is LMPT_SLURM_TIME as read (timeout, estimate, max or minutes)."""
    ts = time_mode()
    out = {"kind": kind(), "sbatch": sbatch_path(), "min_minutes": min_minutes(), "procs": default_procs(),
           "time_limit_min": time_limit_min(), "time": ts, "mem": _mem_status(),
           "slurm_args": RT.env("SLURM_ARGS", ""), "workers": int(os.environ.get("LMPT_WORKERS", "2")),
           "timeout_s": timeout_s(), "atoms_per_rank": atoms_per_rank(), "caps": caps(),
           "run_keys": "a study's run {timeout_s, mem, procs, queue} overrides these for itself, within caps"}
    out["runtime"] = RT.runtime_version          # llmt-runtime's version (machines() and llmt status show it)
    plan = node_plan()
    out["node"] = {k: plan.get(k) for k in ("setting", "nodelist", "cluster_nodes", "this_node")}
    out["node"]["cores"] = node_cores()
    if plan.get("warning"):
        out["node"]["warning"] = plan["warning"]
    if ts == "estimate":
        out["time_factor"] = _env_num("SLURM_TIME_FACTOR", 3.0)
        out["time_min_min"] = int(_env_num("SLURM_TIME_MIN", 20))
    if out["kind"] == "slurm" and not out["sbatch"]:
        out["error"] = "LMPT_EXECUTOR=slurm but sbatch is not on the PATH: studies will fail to submit"
    return out


# ------------------------------------------------------------------------------------------- estimate
_COST = {"lj": 1.5e-6, "morse": 1.5e-6, "buck": 1.5e-6, "coul": 1.5e-6, "eam": 2e-6, "eam/alloy": 2e-6,
         "eam/fs": 2e-6, "tersoff": 5e-6, "tersoff/mod": 5e-6, "tersoff/zbl": 5e-6, "sw": 5e-6, "edip": 5e-6,
         "vashishta": 5e-6, "adp": 1e-5, "meam": 2e-5, "airebo": 3e-5, "rebo": 2e-5, "snap": 2e-4}

_NATOMS: dict = {}
NATOMS_TTL = 60.0


def _key(study: dict) -> str | None:
    try:
        return json.dumps(study, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None


def _remember(key: str | None, n) -> None:
    if key is None:
        return
    _NATOMS[key] = (time.time(), n)
    while len(_NATOMS) > 64:
        _NATOMS.pop(next(iter(_NATOMS)))


def atoms(study: dict) -> int | None:
    """The study's atom count (its structure built), None when it cannot be built; kept NATOMS_TTL s per study: one
    submission asks for it in several hooks (ranks, memory, job.json)."""
    key = _key(study)
    hit = _NATOMS.get(key) if key else None
    if hit is not None and time.time() - hit[0] < NATOMS_TTL:
        return hit[1]
    try:
        from .build import build_system
        n = build_system(study).n
    except Exception:  # noqa: BLE001
        n = None
    _remember(key, n)
    return n


def estimate(study: dict) -> dict:
    """{'minutes': rough run-time estimate or None, 'natoms': atoms or None}."""
    n = atoms(study)
    if n is None:
        return {"minutes": None, "natoms": None}
    return {"minutes": estimate_minutes(study, n), "natoms": n}


def estimate_minutes(study: dict, natoms: int | None = None) -> float | None:
    """A rough upper estimate of the study's run time (one core per rank, ~1 GHz-µs per atom-step costs)."""
    try:
        from .format import molecular

        n = natoms if natoms is not None else atoms(study)
        if n is None:
            return None
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
        procs = max(1, ranks_for(study, n))
        sec = evals * n * c / (procs ** 0.85) + 2.0 * (len(study["stages"]) + 1)
        return round(sec / 60.0, 3)
    except Exception:  # noqa: BLE001
        return None


def goes_to_slurm(study: dict, estimate: float | None) -> bool:
    """Only with LMPT_EXECUTOR=slurm; then "queue" decides if given, else the estimate (none: SLURM).
    llmt-runtime's use_slurm."""
    return S.use_slurm(RT, study, estimate)


# ------------------------------------------------------------------------------------------- cluster nodes
def forget_nodes() -> None:
    """Drops the kept SLURM answers (llmt-runtime's node list and partition limit, the pending order, a sbatch that
    did not answer)."""
    S.forget()
    J._PENDING.update(t=0.0, v=None)
    J._SBATCH_DOWN.update(until=0.0, why="")
    _JOB_NODE.clear()


def _norm(n: dict) -> dict:
    """A node of `scontrol -o show node` in lmpt's shape."""
    def num(k, d=0):
        try:
            return int(float(n.get(k) or d))
        except ValueError:
            return d
    name = n.get("NodeName", "")
    return {"name": name, "hostname": n.get("NodeHostName") or name, "addr": n.get("NodeAddr", ""),
            "cpus": S.node_cpus(n), "threads_per_core": max(1, num("ThreadsPerCore", 1)), "mem_mb": num("RealMemory"),
            "cores": S.node_core_count(n), "state": n.get("State", "")}


def cluster_nodes() -> list[dict] | None:
    """The SLURM cluster's nodes (name, hostname, addr, cpus, threads_per_core, mem_mb, cores, state); None when
    scontrol fails. llmt-runtime's: a success is kept 60 s; a failure only 5 s (within one tool call, so a controller
    that does not answer costs one timeout per call), then it is asked again."""
    nodes = S.cluster_nodes()
    return None if nodes is None else [_norm(n) for n in nodes]


def this_node(nodes: list[dict] | None = None) -> dict | None:
    """The cluster node that is this machine (llmt-runtime's: host name = NodeName or NodeHostName, else an address =
    NodeAddr), in lmpt's shape."""
    me = S.this_node(S.cluster_nodes())
    if me is None:
        return None
    pool = cluster_nodes() if nodes is None else nodes
    return next((n for n in pool or [] if n["name"] == me.get("NodeName")), None)


def node_setting() -> str:
    return S.node_setting(RT)


def node_plan() -> dict:
    """{'setting', 'nodelist' (what the job asks for; None: SLURM chooses), 'cluster_nodes', 'this_node',
    'from' (LMPT_SLURM_ARGS | local | list | any), 'warning'?}. A --nodelist / -w in LMPT_SLURM_ARGS wins.
    llmt-runtime's node_pin."""
    _, nodelist, info = S.node_pin(RT)
    setting = node_setting()
    out = {"setting": setting, "nodelist": nodelist, "cluster_nodes": info.get("cluster_nodes"),
           "this_node": info.get("this_node")}
    if S.args_nodelist(RT):
        out["from"] = "LMPT_SLURM_ARGS"
    elif setting.lower() == "any":
        out["from"] = "any"
    elif setting.lower() != "local":
        out["from"] = "list"
    else:
        out["from"] = "local"
    if info.get("warning"):
        out["warning"] = info["warning"]
    return out


def node_mem_mb() -> float | None:
    """The most memory one job can ask for (MB): the largest RealMemory among the nodes jobs go to; None without
    SLURM, when scontrol fails (asked again next time), or when SLURM does not know it. llmt-runtime's."""
    return S.node_mem_mb(RT)


def node_cores() -> int | None:
    """Physical cores of the node the jobs run on (this node; with a node list or 'any', the smallest of them),
    from SLURM; without SLURM, this machine's cores. None if unknown."""
    if kind() == "slurm" and S.cluster_nodes():
        ns = S.job_nodes(RT)
        if ns:
            return min(S.node_core_count(n) for n in ns)
    try:
        from .lammps import cpu_topology
        return cpu_topology()["cores"] or None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------- submit
def _atomic_json(path: Path, obj) -> None:
    """A JSON file written whole or not at all (llmt-runtime's write_atomic)."""
    J.write_atomic(Path(path), json.dumps(obj, indent=1, default=float))


def _whole_cores() -> bool:
    """One physical core per rank (--ntasks-per-core=1), unless LMPT_SLURM_ARGS says otherwise (--hint,
    --threads-per-core, --ntasks-per-core, --cpus-per-task / -c). LMPT_PROCS is sized in physical cores (llmt tune),
    and Open MPI binds ranks to cores. The same rule as llmt-runtime's job.sh."""
    return not S.has_opt(slurm_args_tokens(), "--hint", "--threads-per-core", "--ntasks-per-core", "--cpus-per-task",
                         shorts=("-c",))


def write_job(request: Path, name: str, procs: int, time_min=None, mem: str | None = None,
              nodelist: str | None = None) -> Path:
    """<request>/<name>/job.sh (llmt-runtime's job_script)."""
    request = Path(request)
    d = request / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "job.sh"
    p.write_text(J.job_script(RT, request, name, {}, int(procs), str(time_min or time_limit_min()), mem, nodelist))
    p.chmod(0o755)
    return p


def job_extra(study: dict) -> dict:
    """lmpt's keys in job.json (the runtime's hooks.job_extra): natoms, estimate_min, procs_requested."""
    n = atoms(study)
    return {"natoms": n, "estimate_min": estimate_minutes(study, n) if n else None,
            "procs_requested": procs_of(study)}


def submit(request: Path, name: str, study: dict, estimate: float | None, natoms: int | None = None) -> dict:
    """Writes job.sh, runs sbatch, writes job.json (llmt-runtime's submit, with lmpt's hooks); returns job.json's
    content, else the study's result saying why (no local fallback). job.json: the runtime's keys and lmpt's (natoms,
    procs_requested, estimate_min as given, time_min / time_from as the job's own limit, timeout_s: the study's own
    timeout, shorter with LMPT_SLURM_TIME=estimate)."""
    request = Path(request)
    if natoms is not None:
        _remember(_key(study), natoms)
    tmin, how = job_time(estimate, study)
    extra = {"estimate_min": estimate, "time_min": tmin if isinstance(tmin, int) else time_limit_min(study),
             "time_from": how}
    info = J.submit(RT, request, name, study, estimate_min=estimate, extra=extra)
    if info.get("job_id"):
        return info
    n = atoms(study)
    return J.failed_submission(RT, request, name, info, execution={
        "executor": "slurm", "slurm_job": None, "procs_requested": procs_of(study),
        "limits": {"timeout_s": timeout_s(study), "mem": job_mem(ranks_for(study, n), n, study),
                   "from_study": from_study(study)}})


# ------------------------------------------------------------------------------------------- queue
def _run(cmd: list[str], timeout: float = 10) -> tuple[int, str, str]:
    """(returncode, stdout, stderr) through llmt-runtime's slurm.run (within an MCP call's deadline when one is set);
    (-1, '', why) when the command is missing or does not answer."""
    p = S.run(cmd, timeout)
    if p is None:
        return -1, "", f"{cmd[0]}: not found or no answer"
    return p.returncode, p.stdout or "", p.stderr or ""


_minutes = S.elapsed_minutes   # squeue %M (elapsed): [D-]HH:MM:SS or MM:SS, in minutes


def _user() -> str:
    try:
        return os.environ.get("USER") or getpass.getuser()
    except Exception:  # noqa: BLE001
        return ""


def queue() -> dict | None:
    """This user's jobs: {id: {state, reason, start, elapsed_min, name, cpus}}; None if squeue cannot be used. Ended
    jobs squeue still lists (MinJobAge) are among them, in their final state: study_state reads them as ended."""
    rc, out, _ = _run(["squeue", "-h", "-u", _user(), "-o", "%i|%T|%r|%S|%M|%j|%C"])
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
    """Pending jobs (anyone's) that SLURM will consider first: higher priority, then earlier job id (llmt-runtime's
    ahead_of; one squeue call serves a few seconds of states)."""
    return J.ahead_of(str(jid))


_JOB_NODE: dict = {}


def _sacct(jid: str) -> str | None:
    """sacct's state of the job, whole ('CANCELLED by 1000' keeps who); None without accounting. The node it ran on is
    kept in _JOB_NODE (llmt-runtime's sacct_rows), for the lost-node diagnosis."""
    r = J.sacct_rows([str(jid)]).get(str(jid))
    _JOB_NODE.pop(str(jid), None)
    if not r:
        return None
    if r.get("node"):
        _JOB_NODE[str(jid)] = r["node"]
    return r.get("state") or None


def _cancelled(d: Path) -> str | None:
    """cancel()'s flag (RT.cancel_file, cancelled.json: {by, at}) as its why; None when there is none. A number (the
    time, as llmt-runtime's cancel_request writes the flag) reads as cancel() at that time."""
    f = d / RT.cancel_file
    if not f.is_file():
        return None
    try:
        c = json.loads(f.read_text())
    except (OSError, ValueError):
        c = None
    if not isinstance(c, dict):
        at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c)) if isinstance(c, (int, float)) else "?"
        c = {"by": "cancel()", "at": at}
    return f"cancelled by {c.get('by', 'cancel()')} at {c.get('at', '?')}"


def ended_state(d: Path, jid: str, listed: str | None = None) -> dict:
    """A job that left the queue without a result (or that squeue lists in a final state: `listed`): what happened.
    cancel()'s cancelled.json first; a job sacct still sees as pending or running is that; else llmt-runtime's rules
    (jobs.classify_end: sacct's state, else squeue's final state, else job.log; SLURM's out-of-memory wording
    'Detected N oom_kill events' whatever sacct says; a job that went to a node that cannot see the run folder), with
    lmpt's out-of-memory advice (runtime.oom_advice: run.mem)."""
    d = Path(d)
    log = d / "job.log"
    tail = J.tail(log)
    why = _cancelled(d)
    if why:
        return {"state": "cancelled", "why": why}
    s = _sacct(jid)
    first = s.split()[0].upper() if s else ""
    if first in J.QUEUED or first in J.ACTIVE:
        return {"state": "queued" if first in J.QUEUED else "running", "why": None}
    job = J.read_job(d)
    state, why = J.classify_end(RT, s or listed, tail, has_log=log.is_file(), folder=d, job=job,
                                node=_JOB_NODE.get(str(jid)) or job.get("node"))
    out = {"state": state, "why": why}
    if state == "failed":
        out["log_tail"] = tail
    return out


def study_state(request: Path, name: str, q: dict | None) -> dict:
    """State of a SLURM study: done (with 'result'), queued (reason, expected_start, ahead; 'submitting' while the
    budgeted submission is under way), running (LAMMPS's progress, job_elapsed_min), unknown (squeue failed), or one of
    FINAL. squeue's states are llmt-runtime's: queued (jobs.QUEUED), running (jobs.ACTIVE), anything else an ended job
    squeue still lists (MinJobAge: it read as running before 2026.11.09.3)."""
    request = Path(request)
    d = request / name
    f = d / f"{name}.json"
    job = J.read_job(d)
    jid = str(job.get("job_id") or "")
    if f.is_file():
        try:
            return {"state": "done", "job_id": jid, "result": json.loads(f.read_text())}
        except json.JSONDecodeError:
            pass
    if not jid:
        try:
            age = time.time() - json.loads((request / "request.json").read_text()).get("started", 0)
        except (OSError, json.JSONDecodeError):
            age = 1e9
        if age < 600:
            return {"state": "queued", "reason": "submitting"}
        return {"state": "failed", "why": "no job.json and no result (submission did not finish?)"}
    if q is None:
        return {"state": "unknown", "job_id": jid, "why": "squeue failed: cannot tell the job's state now"}
    qi = q.get(jid)
    if qi is not None and qi["state"] in J.QUEUED:
        out = {"state": "queued", "job_id": jid, "reason": qi["reason"]}
        if qi["start"] and qi["start"] not in ("N/A", "Unknown", "None"):
            out["expected_start"] = qi["start"]
        a = _ahead(jid)
        if a is not None:
            out["ahead"] = a
        return out
    if qi is not None and qi["state"] in J.ACTIVE:
        from .api import progress
        out = {"state": "running", "job_id": jid, **progress(d)}
        if qi["elapsed_min"] is not None:
            out["job_elapsed_min"] = qi["elapsed_min"]
        return out
    # not in the queue, or listed in a final state: the result may have appeared just now
    if f.is_file():
        try:
            return {"state": "done", "job_id": jid, "result": json.loads(f.read_text())}
        except json.JSONDecodeError:
            pass
    return {"job_id": jid, **ended_state(d, jid, listed=qi["state"] if qi is not None else None)}


def final_result(name: str, st: dict) -> dict:
    """A finished-without-result state as the study's result (and written as <name>.json is not: the job may
    still be found later by sacct)."""
    out = {"ok": False, "name": name, "state": st["state"], "why": st.get("why"), "job_id": st.get("job_id"),
           "execution": {"executor": "slurm", "slurm_job": st.get("job_id")}}
    if st.get("log_tail"):
        out["log_tail"] = st["log_tail"]
    return out


def cancel(request: Path, names: list[str]) -> dict:
    """{name: {cancelled, job_id, why?}}: cancelled.json ({by, at}) in each unfinished study's folder, then one
    scancel (llmt-runtime's) for their jobs."""
    request = Path(request)
    out = {}
    ids = {}
    for n in names:
        jid = J.read_job(request / n).get("job_id")
        if jid:
            ids[n] = str(jid)
        else:
            out[n] = {"cancelled": False, "why": "no SLURM job for this study"}
    if ids:
        for n in ids:
            if not (request / n / f"{n}.json").is_file():
                _atomic_json(request / n / RT.cancel_file, {"by": "cancel()", "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        err = J.scancel(list(ids.values()))
        for n, j in ids.items():
            out[n] = {"cancelled": err is None, "job_id": j, **({} if err is None else {"why": err})}
    return out


def queued_note(request: Path) -> str | None:
    """What to tell someone who interrupted a command that was waiting for a request's SLURM jobs (`lmpt run`,
    `lmpt bench`): the jobs still in the queue and how to stop them. They are not cancelled: they run without the
    command, and their results are collected later. None when no job of the request is left in the queue. (As
    cfd-translator's executor.queued_note, 2026.11.09.4.)"""
    request = Path(request)
    try:
        names = json.loads((request / "request.json").read_text()).get("names", [])
    except (OSError, ValueError):
        names = [p.name for p in request.iterdir() if (p / "job.json").is_file()] if request.is_dir() else []
    ids = []
    for n in names:
        if not (request / n / f"{n}.json").is_file():
            jid = J.read_job(request / n).get("job_id")
            if jid:
                ids.append(str(jid))
    if not ids:
        return None
    q = J.squeue(ids)
    if q is not None:                       # SLURM did not answer: name them all
        ids = [i for i in ids if i in q and (q[i]["state"] in J.QUEUED or q[i]["state"] in J.ACTIVE)]
    if not ids:
        return None
    many = len(ids) > 1
    return (f"interrupted: SLURM job{'s' if many else ''} {', '.join(ids)} of {request} {'are' if many else 'is'} "
            f"still in the queue and will run without this command (nothing was cancelled). `{CLI} jobs` shows "
            f"them; collect the results later with study_status(request) (bench_status for benchmarks) from the MCP "
            f"tools; to stop {'them' if many else 'it'}: "
            f"`{CLI} cancel {request}` (cancel(request) from the MCP tools) or `scancel {' '.join(ids)}`")


def print_queued_note(request: Path) -> None:
    """queued_note on stderr (stdout keeps the command's output); never raises."""
    import sys
    try:
        note = queued_note(request)
    except Exception:  # noqa: BLE001
        note = None
    if note:
        print(f"[{CLI}] {note}", file=sys.stderr, flush=True)


def machine() -> dict | None:
    """{cpus, cpus_allocated, cpus_idle} of the nodes this translator's jobs go to, each node counted once
    (llmt-runtime's queue_summary: sinfo -N); None when sinfo does not answer."""
    return J.queue_summary(RT).get("machine") or None


def queue_summary(q: dict | None) -> dict | None:
    """{running, queued, cpus_in_use} of this user's lmpt jobs in q (queue()); ended jobs squeue still lists count as
    neither."""
    if q is None:
        return None
    mine = [v for v in q.values() if v["name"].startswith(f"{CLI}:")]
    run = [v for v in mine if v["state"] in J.ACTIVE]
    return {"running": len(run), "queued": len([v for v in mine if v["state"] in J.QUEUED]),
            "cpus_in_use": sum(v["cpus"] for v in run)}
