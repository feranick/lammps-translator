"""executor.py — where studies run: the local worker pool or SLURM jobs (llm-translators docs/EXECUTION.md).
Version: 2026.10.25.2

Settings (environment): LMPT_EXECUTOR local | slurm (local); LMPT_SLURM_MIN_MINUTES (1): studies estimated
shorter run locally; LMPT_SLURM_ARGS: extra sbatch arguments; LMPT_PROCS (MPI ranks, --ntasks); LMPT_TIMEOUT
(seconds per study). A study's "queue": true / false forces SLURM / local (not with LMPT_EXECUTOR=local).

Job limits:
  --time   LMPT_SLURM_TIME=timeout (default): ceil(LMPT_TIMEOUT/60) + 10 min. =estimate: max(LMPT_SLURM_TIME_MIN (20),
           ceil(LMPT_SLURM_TIME_FACTOR (3) x the estimate) + 10) min, at most the former; the study's own timeout is
           10 min less. Short limits let SLURM backfill short studies around long jobs; a study that outruns its
           estimate ends timed_out (rerun with LMPT_SLURM_TIME=timeout).
  --mem    LMPT_SLURM_MEM=auto (default): 1 GB + 256 MB per rank + 4 kB per atom, rounded up to 512 MB, at least
           2 GB; a size (8G, 8000M) as given; none: SLURM's default (on nodes with CR_Core_Memory and no DefMemPerCPU
           that is the whole node, and jobs run one at a time). A --mem* in LMPT_SLURM_ARGS wins.
  --ntasks-per-core=1 (one physical core per rank) unless LMPT_SLURM_ARGS sets the core/thread layout.
  --ntasks min(procs, ceil(atoms / LMPT_ATOMS_PER_RANK (2000))): no more ranks than the study can use.

Per study (docs/EXECUTION.md, "Per-study run settings"): "run": {timeout_s, mem, procs, queue} overrides
LMPT_TIMEOUT (and so --time), --mem (replacing any --mem* in LMPT_SLURM_ARGS), LMPT_PROCS and the queue decision
for that study, within LMPT_MAX_TIMEOUT, LMPT_MAX_MEM, LMPT_MAX_PROCS (unset: no cap).

Clusters of several nodes: LMPT_SLURM_NODE = local (default: --nodelist=<this node> when scontrol lists more than
one node, because run folders are in this machine's home) | any (SLURM chooses: shared homes) | a node list. A
--nodelist / -w in LMPT_SLURM_ARGS wins. Ranks are capped at that node's physical cores. scontrol, sinfo and squeue
results that failed are asked again next time (a controller that was down is not remembered as "no SLURM").

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


RUN_KEYS = ("timeout_s", "mem", "procs", "queue")


def run_of(study: dict | None) -> dict:
    r = (study or {}).get("run")
    return r if isinstance(r, dict) else {}


def timeout_s(study: dict | None = None) -> float:
    """The study's run.timeout_s, else LMPT_TIMEOUT (21600 s)."""
    t = run_of(study).get("timeout_s")
    return float(t) if t else float(os.environ.get("LMPT_TIMEOUT", "21600"))


def time_limit_min(study: dict | None = None) -> int:
    return int(math.ceil(timeout_s(study) / 60.0)) + 10


def parse_mem_mb(v) -> float | None:
    """SLURM sizes: 500M, 8G, 1T, 2048 (MB), 512K; None if malformed."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KkMmGgTt]?)[Bb]?\s*", str(v))
    if not m:
        return None
    x = float(m.group(1))
    return x * {"": 1, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}[m.group(2).lower()]


def caps() -> dict:
    """LMPT_MAX_TIMEOUT (s), LMPT_MAX_MEM (size), LMPT_MAX_PROCS: what a study's run may ask; None: no cap."""
    def num(k):
        v = os.environ.get(k, "").strip()
        try:
            return float(v) if v else None
        except ValueError:
            return None
    mem = os.environ.get("LMPT_MAX_MEM", "").strip() or None
    p = num("LMPT_MAX_PROCS")
    return {"timeout_s": num("LMPT_MAX_TIMEOUT"), "mem": mem, "procs": int(p) if p else None}


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
                E.append(f"run.mem: {r['mem']} is above LMPT_MAX_MEM ({c['mem']})")
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
    return float(os.environ.get("LMPT_SLURM_MIN_MINUTES", "1"))


def default_procs() -> int:
    return int(os.environ.get("LMPT_PROCS", "1"))


def procs_of(study: dict | None) -> int:
    """Ranks asked for: run.procs, then the study's procs, then LMPT_PROCS."""
    r = run_of(study)
    if r.get("procs"):
        return int(r["procs"])
    return int((study or {}).get("procs") or default_procs())


def study_procs(study: dict) -> int:
    return procs_of(study)


def atoms_per_rank() -> int:
    return int(os.environ.get("LMPT_ATOMS_PER_RANK", "2000"))


def ranks_for(study: dict, natoms: int | None) -> int:
    """The size cap: min(procs, ceil(atoms / LMPT_ATOMS_PER_RANK), this node's physical cores); procs (capped at
    the cores) when the atoms are not known. Jobs ask for one rank per core (--ntasks-per-core=1)."""
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


def sbatch_args(study: dict | None = None) -> list[str]:
    """LMPT_SLURM_ARGS, without --mem / --mem-per-cpu / --mem-per-gpu when the study sets run.mem."""
    a = slurm_args_tokens()
    if not run_of(study).get("mem"):
        return a
    out, skip = [], False
    for t in a:
        if skip:
            skip = False
            continue
        if t.startswith("--mem"):
            skip = "=" not in t
            continue
        out.append(t)
    return out


def sbatch_path() -> str | None:
    return shutil.which("sbatch")


def time_mode() -> str:
    return "estimate" if os.environ.get("LMPT_SLURM_TIME", "timeout").strip().lower() == "estimate" else "timeout"


def job_time(estimate: float | None, study: dict | None = None) -> tuple[int, str]:
    """(the job's --time in minutes, how it was chosen). A study's run.timeout_s always wins."""
    full = time_limit_min(study)
    if run_of(study).get("timeout_s"):
        return full, "study"
    if time_mode() == "estimate" and estimate is not None:
        f = float(os.environ.get("LMPT_SLURM_TIME_FACTOR", "3"))
        floor = int(os.environ.get("LMPT_SLURM_TIME_MIN", "20"))
        return min(full, max(floor, int(math.ceil(f * estimate)) + 10)), "estimate"
    return full, "timeout"


def mem_setting() -> str:
    return os.environ.get("LMPT_SLURM_MEM", "auto").strip() or "auto"


def job_mem(procs: int, natoms: int | None, study: dict | None = None) -> str | None:
    """The job's --mem value: the study's run.mem, else None when LMPT_SLURM_ARGS has one or LMPT_SLURM_MEM=none,
    else LMPT_SLURM_MEM (auto: from ranks and atoms)."""
    if run_of(study).get("mem"):
        return str(run_of(study)["mem"])
    if _has_opt(slurm_args_tokens(), "--mem", "--mem-per-cpu", "--mem-per-gpu"):
        return None
    m = mem_setting()
    if m.lower() == "none":
        return None
    if m.lower() != "auto":
        return m
    mb = 1024 + 256 * procs + 0.004 * (natoms or 100000)
    mb = max(2048, int(math.ceil(mb / 512.0)) * 512)
    return f"{mb}M"


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
        mb = 1024 + 256 * procs + 0.004 * (natoms or 100000)
        return max(2048, int(math.ceil(mb / 512.0)) * 512)
    v = parse_mem_mb(m)
    return int(math.ceil(v)) if v else 2048


def sizing(study: dict) -> dict:
    """What a study's run is sized as, before running: {'minutes' (upper estimate with these ranks), 'procs'
    (ranks after the size cap), 'mem_mb'} — the same numbers submit() puts in job.sh. For validate_studies
    (llm-translators' router places runs with them)."""
    e = estimate(study)
    procs = ranks_for(study, e["natoms"])
    return {"minutes": e["minutes"], "procs": procs, "mem_mb": mem_mb_for(procs, e["natoms"], study)}


def _mem_status() -> str:
    a = [t for t in slurm_args_tokens() if t.startswith("--mem")]
    return f"{a[0]} (from LMPT_SLURM_ARGS)" if a else mem_setting()


def status() -> dict:
    out = {"kind": kind(), "sbatch": sbatch_path(), "min_minutes": min_minutes(), "procs": default_procs(),
           "time_limit_min": time_limit_min(), "time": time_mode(), "mem": _mem_status(),
           "slurm_args": os.environ.get("LMPT_SLURM_ARGS", ""), "workers": int(os.environ.get("LMPT_WORKERS", "2")),
           "timeout_s": timeout_s(), "atoms_per_rank": atoms_per_rank(), "caps": caps(),
           "run_keys": "a study's run {timeout_s, mem, procs, queue} overrides these for itself, within caps"}
    plan = node_plan()
    out["node"] = {k: plan.get(k) for k in ("setting", "nodelist", "cluster_nodes", "this_node")}
    out["node"]["cores"] = node_cores()
    if plan.get("warning"):
        out["node"]["warning"] = plan["warning"]
    if time_mode() == "estimate":
        out["time_factor"] = float(os.environ.get("LMPT_SLURM_TIME_FACTOR", "3"))
        out["time_min_min"] = int(os.environ.get("LMPT_SLURM_TIME_MIN", "20"))
    if out["kind"] == "slurm" and not out["sbatch"]:
        out["error"] = "LMPT_EXECUTOR=slurm but sbatch is not on the PATH: studies will fail to submit"
    return out


# ------------------------------------------------------------------------------------------- estimate
_COST = {"lj": 1.5e-6, "morse": 1.5e-6, "buck": 1.5e-6, "coul": 1.5e-6, "eam": 2e-6, "eam/alloy": 2e-6,
         "eam/fs": 2e-6, "tersoff": 5e-6, "tersoff/mod": 5e-6, "tersoff/zbl": 5e-6, "sw": 5e-6, "edip": 5e-6,
         "vashishta": 5e-6, "adp": 1e-5, "meam": 2e-5, "airebo": 3e-5, "rebo": 2e-5, "snap": 2e-4}


def estimate(study: dict) -> dict:
    """{'minutes': rough run-time estimate or None, 'natoms': atoms or None}."""
    try:
        from .build import build_system
        n = build_system(study).n
    except Exception:  # noqa: BLE001
        return {"minutes": None, "natoms": None}
    return {"minutes": estimate_minutes(study, n), "natoms": n}


def estimate_minutes(study: dict, natoms: int | None = None) -> float | None:
    """A rough upper estimate of the study's run time (one core per rank, ~1 GHz-µs per atom-step costs)."""
    try:
        from .build import build_system
        from .format import molecular

        n = natoms if natoms is not None else build_system(study).n
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
    if kind() != "slurm":
        return False
    q = queue_of(study)
    if q is True:
        return True
    if q is False:
        return False
    return estimate is None or estimate >= min_minutes()


# ------------------------------------------------------------------------------------------- cluster nodes
_NODES_CACHE: dict = {}


def forget_nodes() -> None:
    """Drops the kept SLURM answers (node list, pending order)."""
    _NODES_CACHE.clear()
    _PENDING.clear()


def cluster_nodes() -> list[dict] | None:
    """The SLURM cluster's nodes from 'scontrol -o show node' (name, hostname, addr, cpus, threads per core,
    cores); None when scontrol fails. A success is kept 60 s; a failure only 5 s (within one tool call, so a
    controller that does not answer costs one timeout per call), then it is asked again."""
    c = _NODES_CACHE.get("v")
    if c and time.time() - c[0] < 60:
        return c[1]
    f = _NODES_CACHE.get("failed")
    if f and time.time() - f < 5:          # one tool call asks once; the next call (seconds later) asks again
        return None
    rc, out, _ = _run(["scontrol", "-o", "show", "node"], timeout=10)
    if rc != 0 or not out.strip():
        _NODES_CACHE["failed"] = time.time()
        return None
    _NODES_CACHE.pop("failed", None)
    nodes = []
    for ln in out.splitlines():
        kv = dict(re.findall(r"(\w+)=(\S*)", ln))
        if "NodeName" not in kv:
            continue
        def num(k, d=0):
            try:
                return int(kv.get(k, d))
            except ValueError:
                return d
        cpus = num("CPUTot")
        tpc = max(1, num("ThreadsPerCore", 1))
        cores = num("CoresPerSocket") * max(1, num("Sockets", 1)) * max(1, num("Boards", 1)) or (cpus // tpc)
        nodes.append({"name": kv["NodeName"], "hostname": kv.get("NodeHostName", kv["NodeName"]),
                      "addr": kv.get("NodeAddr", ""), "cpus": cpus, "threads_per_core": tpc,
                      "cores": min(cores, cpus // tpc) if cpus else cores, "state": kv.get("State", "")})
    _NODES_CACHE["v"] = (time.time(), nodes)
    return nodes


def _my_addresses() -> set[str]:
    addrs = set()
    rc, out, _ = _run(["hostname", "-I"], timeout=5)
    if rc == 0:
        addrs.update(out.split())
    try:
        import socket
        addrs.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return addrs


def _hostname() -> str:
    import socket
    return socket.gethostname()


def this_node(nodes: list[dict] | None = None) -> dict | None:
    """The cluster node that is this machine: host name = NodeName or NodeHostName, else an address = NodeAddr."""
    nodes = cluster_nodes() if nodes is None else nodes
    if not nodes:
        return None
    h = _hostname()
    names = {h, h.split(".")[0]}
    for n in nodes:
        if n["name"] in names or n["hostname"] in names or n["hostname"].split(".")[0] in names:
            return n
    addrs = _my_addresses()
    for n in nodes:
        if n["addr"] and n["addr"] in addrs:
            return n
    return None


def node_setting() -> str:
    return os.environ.get("LMPT_SLURM_NODE", "local").strip() or "local"


def _args_nodelist(tokens: list[str]) -> str | None:
    for i, t in enumerate(tokens):
        if t.startswith("--nodelist="):
            return t.split("=", 1)[1]
        if t in ("--nodelist", "-w") and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith("-w") and len(t) > 2:
            return t[2:].lstrip("=")
    return None


def node_plan() -> dict:
    """{'setting', 'nodelist' (what the job asks for; None: SLURM chooses), 'cluster_nodes', 'this_node',
    'from' (args | local | list | any), 'warning'?}. A --nodelist / -w in LMPT_SLURM_ARGS wins."""
    setting = node_setting()
    nodes = cluster_nodes()
    me = this_node(nodes) if nodes else None
    out = {"setting": setting, "nodelist": None, "cluster_nodes": len(nodes) if nodes is not None else None,
           "this_node": me["name"] if me else None}
    given = _args_nodelist(slurm_args_tokens())
    if given:
        out.update({"nodelist": given, "from": "LMPT_SLURM_ARGS"})
        return out
    if setting == "any":
        out["from"] = "any"
        return out
    if setting != "local":
        out.update({"nodelist": setting, "from": "list"})
        return out
    out["from"] = "local"
    if nodes is None:
        return out
    if len(nodes) > 1:
        if me:
            out["nodelist"] = me["name"]
        else:
            out["warning"] = ("LMPT_SLURM_NODE=local, but this machine is none of the cluster's nodes (by host name "
                              "or address): jobs are not pinned and may run on another node; set LMPT_SLURM_NODE")
    return out


def node_cores() -> int | None:
    """Physical cores of the node the jobs run on (this node; with a node list or 'any', the smallest of them),
    from SLURM; without SLURM, this machine's cores. None if unknown."""
    if kind() == "slurm":
        nodes = cluster_nodes()
        if nodes:
            plan = node_plan()
            if plan["nodelist"]:
                names = set(_expand_nodelist(plan["nodelist"]))
                sel = [n for n in nodes if n["name"] in names] or nodes
            else:
                me = this_node(nodes)
                sel = [me] if (me and plan["setting"] == "local") else nodes
            cores = [n["cores"] for n in sel if n.get("cores")]
            if cores:
                return min(cores)
    try:
        from .lammps import cpu_topology
        return cpu_topology()["cores"] or None
    except Exception:  # noqa: BLE001
        return None


def _expand_nodelist(s: str) -> list[str]:
    """'a,b' and simple ranges 'node[1-3,5]' (enough for the names we compare)."""
    out = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?", s):
        m = re.fullmatch(r"(.*)\[(.*)\]", part)
        if not m:
            out.append(part)
            continue
        pre, body = m.groups()
        for r in body.split(","):
            if "-" in r:
                a, b = r.split("-", 1)
                w = len(a)
                out += [f"{pre}{i:0{w}d}" for i in range(int(a), int(b) + 1)]
            else:
                out.append(pre + r)
    return out


# ------------------------------------------------------------------------------------------- submit
def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def slurm_args_tokens() -> list[str]:
    """LMPT_SLURM_ARGS split as a shell would (it may hold several options, e.g. '--mem=8G --nodelist=x')."""
    try:
        return shlex.split(os.environ.get("LMPT_SLURM_ARGS", ""))
    except ValueError:
        return os.environ.get("LMPT_SLURM_ARGS", "").split()


def _has_opt(tokens: list[str], *names: str) -> bool:
    """Whether an option is among the tokens: --name, --name=value, or a short -x / -xvalue."""
    for t in tokens:
        for n in names:
            if t == n or (n.startswith("--") and t.startswith(n + "=")) or (len(n) == 2 and t.startswith(n)):
                return True
    return False


def _whole_cores() -> bool:
    """One physical core per rank (--ntasks-per-core=1), unless LMPT_SLURM_ARGS says otherwise. LMPT_PROCS is
    sized in physical cores (llmt tune), and Open MPI binds ranks to cores."""
    return not _has_opt(slurm_args_tokens(), "--hint", "--threads-per-core", "--ntasks-per-core", "--cpus-per-task")


def write_job(request: Path, name: str, procs: int, time_min: int | None = None, mem: str | None = None,
              nodelist: str | None = None) -> Path:
    d = request / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={CLI}:{request.name}:{name}",
             f"#SBATCH --chdir={d}",
             f"#SBATCH --output={d / 'job.log'}",
             f"#SBATCH --ntasks={procs}",
             *(["#SBATCH --ntasks-per-core=1"] if _whole_cores() else []),
             f"#SBATCH --time={time_min or time_limit_min()}",
             *([f"#SBATCH --mem={mem}"] if mem else []),
             *([f"#SBATCH --nodelist={nodelist}"] if nodelist else []),
             "#SBATCH --export=ALL",
             f"exec {shlex.quote(sys.executable)} -m lammps_translator.cli run-study {shlex.quote(str(request))} "
             f"{shlex.quote(name)}"]
    p = d / "job.sh"
    p.write_text("\n".join(lines) + "\n")
    p.chmod(0o755)
    return p


def submit(request: Path, name: str, study: dict, estimate: float | None, natoms: int | None = None) -> dict:
    """Writes job.sh, runs sbatch; job.json on success, else the study's result says why (no local fallback)."""
    procs = ranks_for(study, natoms)
    tmin, how = job_time(estimate, study)
    mem = job_mem(procs, natoms, study)
    plan = node_plan()
    nodelist = plan["nodelist"] if plan.get("from") in ("local", "list") else None   # args' own -w stays in args
    job = write_job(request, name, procs, tmin, mem, nodelist)
    cmd = ["sbatch", "--parsable", *sbatch_args(study), str(job)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=job.parent)
        ok = r.returncode == 0 and r.stdout.strip()
        msg = (r.stderr or r.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, msg, r = False, str(exc), None
    if not ok:
        res = {"ok": False, "name": name, "state": "failed", "why": f"could not submit: {msg[:500] or 'sbatch failed'}",
               "execution": {"executor": "slurm", "slurm_job": None, "procs_requested": procs_of(study),
                             "limits": {"timeout_s": timeout_s(study), "mem": mem, "from_study": from_study(study)}}}
        _atomic_json(request / name / f"{name}.json", res)
        return res
    jid = r.stdout.strip().split(";")[0].split()[0]
    tout = timeout_s(study) if how != "estimate" else (tmin - 10) * 60
    info = {"executor": "slurm", "job_id": jid, "submitted": time.time(), "procs": procs,
            "procs_requested": procs_of(study), "time_min": tmin, "time_from": how, "timeout_s": tout, "mem": mem,
            "estimate_min": estimate, "natoms": natoms, "run": run_of(study), "nodelist": plan["nodelist"],
            "command": cmd}
    _atomic_json(request / name / "job.json", info)
    return info


# ------------------------------------------------------------------------------------------- queue
def _run(cmd: list[str], timeout: float = 10) -> tuple[int, str, str]:
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


_PENDING: dict = {}


def _ahead(jid: str) -> int | None:
    """Pending jobs (anyone's) that SLURM will consider first: higher priority, then earlier job id.
    (squeue's own listing order is not the scheduling order.) One squeue call serves a few seconds of states."""
    c = _PENDING.get("v")
    if c and time.time() - c[0] < 3:
        ids = c[1]
        return ids.index(jid) if jid in ids else None
    rc, out, _ = _run(["squeue", "-h", "-t", "PENDING", "-o", "%i|%Q"])
    if rc != 0:
        return None
    jobs = []
    for ln in out.splitlines():
        t = ln.strip().split("|")
        if len(t) == 2 and t[0]:
            try:
                jobs.append((-float(t[1]), int(t[0].split("_")[0]), t[0]))
            except ValueError:
                continue
    jobs.sort()
    ids = [j[2] for j in jobs]
    _PENDING["v"] = (time.time(), ids)
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


def _time_hint(d: Path) -> str:
    try:
        j = json.loads((d / "job.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if j.get("time_from") == "estimate":
        return (f" (a {j.get('time_min')} min limit from the estimate, {j.get('estimate_min')} min: rerun with "
                "LMPT_SLURM_TIME=timeout or a larger LMPT_SLURM_TIME_FACTOR)")
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
            return {"state": "timed_out", "why": f"the job hit its SLURM time limit ({s}){_time_hint(d)}: raise LMPT_TIMEOUT "
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
        return {"state": "timed_out", "why": "the job hit its SLURM time limit (job.log)" + _time_hint(d) +
                                             ": raise LMPT_TIMEOUT or shorten the study"}
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
        try:
            age = time.time() - json.loads((request / "request.json").read_text()).get("started", 0)
        except (OSError, json.JSONDecodeError):
            age = 1e9
        if age < 600:
            return {"state": "queued", "reason": "submitting"}
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
    out = {"ok": False, "name": name, "state": st["state"], "why": st.get("why"), "job_id": st.get("job_id"),
           "execution": {"executor": "slurm", "slurm_job": st.get("job_id")}}
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
