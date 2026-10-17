"""lammps.py — the only module that runs LAMMPS: finds the executable, reads its capabilities, runs inputs.
Version: 2026.10.16.1

Executable: LMPT_LMP (a command, e.g. 'lmp' or '/opt/lammps/bin/lmp_mpi'), else the first of lmp, lmp_mpi,
lmp_serial, lammps on the PATH. MPI: LMPT_PROCS ranks (or the study's 'procs') through LMPT_MPIRUN ('mpirun').
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


def find_lmp() -> list[str] | None:
    env = os.environ.get("LMPT_LMP")
    if env:
        cmd = shlex.split(env)
        return cmd if shutil.which(cmd[0]) or Path(cmd[0]).is_file() else None
    for name in ("lmp", "lmp_mpi", "lmp_serial", "lammps"):
        p = shutil.which(name)
        if p:
            return [p]
    return None


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
        return {"error": "LAMMPS not found: install it (apt install lammps) or set LMPT_LMP"}
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
        return {"rc": -1, "error": "LAMMPS not found (set LMPT_LMP)", "seconds": 0.0}
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
