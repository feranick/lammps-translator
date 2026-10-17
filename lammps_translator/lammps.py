"""lammps.py — the only module that runs LAMMPS: finds the executable, reads its capabilities, runs inputs.
Version: 2026.10.16.2

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
from functools import lru_cache
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


@lru_cache(maxsize=4)
def _info(cmd: tuple) -> dict:
    try:
        r = subprocess.run([*cmd, "-h"], capture_output=True, text=True, timeout=60)
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


def run(input_name: str, log_name: str, cwd: Path, procs: int = 1, timeout: float = 3600.0) -> dict:
    """Runs LAMMPS on cwd/input_name, writing cwd/log_name. Returns rc, seconds, stderr tail."""
    cmd = find_lmp()
    if cmd is None:
        return {"rc": -1, "error": not_found(), "seconds": 0.0}
    full = [*cmd, "-in", input_name, "-log", log_name, "-screen", "none"]
    if procs > 1:
        mpirun = shlex.split(os.environ.get("LMPT_MPIRUN", "mpirun"))
        full = [*mpirun, "-np", str(procs), *full]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(full, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
        rc, err = r.returncode, (r.stderr or "")[-2000:] + (r.stdout or "")[-1000:]
    except subprocess.TimeoutExpired:
        rc, err = -9, f"timed out after {timeout:g} s"
    except OSError as exc:
        rc, err = -1, str(exc)
    return {"rc": rc, "seconds": round(time.perf_counter() - t0, 3), "stderr": err.strip(), "command": " ".join(full)}
