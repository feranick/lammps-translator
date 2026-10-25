"""lammps.py — the only module that runs LAMMPS: finds the executable, reads its capabilities, runs inputs.
Version: 2026.10.25.1

Executable: LMPT_LMP (a command, e.g. 'lmp', '/opt/lammps/bin/lmp_mpi' or 'mpirun -np 4 lmp'), else the first of
lmp, lmp_mpi, lmp_serial, lmp_stable, lmp_openmpi, lammps on the PATH, else the same names in /usr/bin,
/usr/local/bin, /opt/lammps/bin, ~/.local/bin. MPI: LMPT_PROCS ranks (or the study's 'procs') through LMPT_MPIRUN.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

NAMES = ("lmp", "lmp_mpi", "lmp_serial", "lmp_stable", "lmp_openmpi", "lammps")
DIRS = ("/usr/bin", "/usr/local/bin", "/opt/lammps/bin", "~/.local/bin")


def _exe(name: str) -> str | None:
    p = shutil.which(os.path.expanduser(name))
    return p if p else None


def find_lmp() -> list[str] | None:
    env = os.environ.get("LMPT_LMP")
    if env:
        cmd = shlex.split(env)
        cmd[0] = os.path.expanduser(cmd[0])
        return cmd if _exe(cmd[0]) else None
    for name in NAMES:
        p = _exe(name)
        if p:
            return [p]
    for d in DIRS:
        for name in NAMES:
            p = _exe(str(Path(d).expanduser() / name))
            if p:
                return [p]
    return None


def not_found() -> str:
    env = os.environ.get("LMPT_LMP")
    if env:
        return (f"LAMMPS not found: LMPT_LMP='{env}' is not an executable on this machine (check the path, or "
                "unset it to search the PATH)")
    return (f"LAMMPS not found: none of {list(NAMES)} on the PATH or in {list(DIRS)}; set LMPT_LMP to the "
            "executable (e.g. export LMPT_LMP=/usr/bin/lmp; for the MCP server put it in the ssh command: "
            "env LMPT_LMP=... lmpt-mcp)")


_INFO: dict = {}


def _info(cmd: tuple) -> dict:
    """'lmp -h': version, packages, styles. Kept only when it worked: a failure is asked again next time."""
    if cmd in _INFO:
        return _INFO[cmd]
    try:
        r = subprocess.run([*cmd, "-h"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{cmd[0]} -h failed: {exc}"}
    text = r.stdout + r.stderr
    out = {"exe": " ".join(cmd)}
    m = re.search(r"Large-scale Atomic/Molecular Massively Parallel Simulator - (.+)", text)
    out["version"] = m.group(1).strip() if m else None
    m = re.search(r"Installed packages:\s*\n\s*\n(.*?)\n\s*\n", text, re.S)
    out["packages"] = m.group(1).split() if m else []
    styles = {}
    for sec in ("Pair", "Bond", "Angle", "Dihedral", "KSpace", "Fix", "Compute"):
        m = re.search(rf"\* {sec} styles:\s*\n\s*\n(.*?)\n\s*\n", text, re.S)
        styles[sec.lower()] = m.group(1).split() if m else []
    out["styles"] = styles
    out["mpi"] = "MPI v" in text or "MPICH" in text or "Open MPI" in text
    if out["version"] is None:
        out["error"] = f"{cmd[0]} -h did not print a LAMMPS version (exit {r.returncode}): {text.strip()[-300:]}"
        return out
    _INFO[cmd] = out
    return out


def lmp_info() -> dict:
    cmd = find_lmp()
    if cmd is None:
        return {"error": not_found()}
    return _info(tuple(cmd))


def missing_styles(needed: dict) -> list[str]:
    """Styles the study needs that this LAMMPS lacks ({'pair': [...], 'fix': [...]})."""
    info = lmp_info()
    if "error" in info:
        return []
    out = []
    for sec, names in needed.items():
        have = set(info["styles"].get(sec, []))
        if not have:
            continue
        for n in names:
            if n and n not in have:
                out.append(f"{sec} style {n}")
    return out


def cpu_topology() -> dict:
    """CPUs this process may use (its affinity: the SLURM allocation inside a job) and the physical cores they
    belong to."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:
        cpus = list(range(os.cpu_count() or 1))
    cores = set()
    for c in cpus:
        base = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        try:
            cores.add((base.joinpath("physical_package_id").read_text().strip(), base.joinpath("core_id").read_text().strip()))
        except OSError:
            cores.add(("?", str(c)))
    return {"hwthreads": len(cpus), "cores": len(cores)}


_FLAVOR: dict = {}


def mpi_flavor(launcher: str) -> str:
    """openmpi | mpich | unknown; 'unknown' (mpirun missing or silent) is asked again next time."""
    if launcher in _FLAVOR:
        return _FLAVOR[launcher]
    f = _flavor(launcher)
    if f != "unknown":
        _FLAVOR[launcher] = f
    return f


def _flavor(launcher: str) -> str:
    r = _info_text([launcher, "--version"])
    t = r.lower()
    if "open mpi" in t or "open-mpi" in t or "prrte" in t or "prterun" in t:
        return "openmpi"
    if "hydra" in t or "mpich" in t or "intel(r) mpi" in t:
        return "mpich"
    return "unknown"


def _info_text(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired):
        return ""


def mpi_plan(procs: int) -> dict:
    """The MPI launch for procs ranks: LMPT_MPIRUN (mpirun) -np procs, with a binding that fits the CPUs we have.
    Open MPI binds each rank to a core by default and refuses ('allocation-overload') when ranks outnumber the
    cores, as in a SLURM allocation of hardware threads: then bind to hardware threads, or not at all."""
    env = os.environ.get("LMPT_MPIRUN", "mpirun")
    launcher = shlex.split(env)
    topo = cpu_topology()
    out = {"launcher": [*launcher, "-np", str(procs)], "binding": "default", **topo}
    user_set = any(a.startswith(("--bind-to", "-bind-to", "--map-by", "-map-by", "--oversubscribe")) for a in launcher)
    if procs <= topo["cores"] or user_set:
        return out
    flavor = mpi_flavor(launcher[0])
    out["mpi"] = flavor
    if flavor == "openmpi":
        extra = ["--bind-to", "hwthread"] if procs <= topo["hwthreads"] else ["--bind-to", "none", "--oversubscribe"]
    elif flavor == "mpich":
        extra = ["-bind-to", "hwthread"] if procs <= topo["hwthreads"] else []
    else:
        extra = []
    out["launcher"] = [*launcher, *extra, "-np", str(procs)]
    out["binding"] = " ".join(extra) or "default"
    out["note"] = (f"{procs} ranks on {topo['cores']} physical cores ({topo['hwthreads']} hardware threads): ranks "
                   "share cores, which is slower; with SLURM, ask for whole cores (the job's --ntasks-per-core=1)")
    return out


def run(input_name: str, log_name: str, cwd: Path, procs: int = 1, timeout: float = 3600.0) -> dict:
    """Runs LAMMPS on cwd/input_name, writing cwd/log_name. Returns rc, seconds, stderr tail."""
    cmd = find_lmp()
    if cmd is None:
        return {"rc": -1, "error": not_found(), "seconds": 0.0}
    full = [*cmd, "-in", input_name, "-log", log_name, "-screen", "none"]
    if procs > 1:
        full = [*mpi_plan(procs)["launcher"], *full]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    cancel = Path(cwd).parent / "CANCEL"      # written by api.cancel for a local study
    cancelled = False
    try:
        p = subprocess.Popen(full, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                             start_new_session=True)
    except OSError as exc:
        return {"rc": -1, "seconds": 0.0, "stderr": str(exc), "command": " ".join(full)}
    rc = None
    while rc is None:
        try:
            out, err = p.communicate(timeout=1.0)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            el = time.perf_counter() - t0
            if cancel.is_file() or el > timeout:
                cancelled = cancel.is_file()
                _kill(p)
                out, err = p.communicate()
                rc = -9
                err = (err or "") + ("\ncancelled" if cancelled else f"\ntimed out after {timeout:g} s")
    msg = ((err or "")[-2000:] + (out or "")[-1000:]).replace("\x00", "")
    if rc == -9 and not cancelled:
        msg = f"timed out after {timeout:g} s (LMPT_TIMEOUT)"
    res = {"rc": rc, "seconds": round(time.perf_counter() - t0, 3), "stderr": msg.strip(), "command": " ".join(full)}
    if cancelled:
        res["cancelled"] = True
    return res


def _kill(p: subprocess.Popen) -> None:
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
