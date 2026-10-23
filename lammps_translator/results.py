"""results.py — tolerant readers of LAMMPS output: log files (thermo blocks, LMPT tagged lines, errors, warnings,
minimizer stops), dump files, fix ave/time vector files.
Version: 2026.10.22.1
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


THERMO_NAMES = {"Step": "step", "Temp": "temp", "PotEng": "pe", "KinEng": "ke", "TotEng": "etotal",
                "Press": "press", "Volume": "vol", "Density": "density"}


def _float(t: str):
    try:
        return float(t)
    except ValueError:
        return None


def read_log(path: Path) -> dict:
    """Thermo blocks (columns + rows), LMPT tags in order, ERROR/WARNING lines, minimizer stops, loop times."""
    out = {"blocks": [], "tags": [], "errors": [], "warnings": [], "stops": [], "loops": [], "version": None}
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        out["errors"].append(f"no log {path}")
        return out
    lines = text.splitlines()
    i = 0
    cols = None
    rows: list = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("LAMMPS (") and out["version"] is None:
            out["version"] = s[len("LAMMPS ("):-1] if s.endswith(")") else s
        if cols is not None:
            t = s.split()
            vals = [_float(x) for x in t]
            if len(t) == len(cols) and all(v is not None for v in vals):
                rows.append(vals)
                i += 1
                continue
            if s.startswith("WARNING"):
                out["warnings"].append(s)
                i += 1
                continue
            out["blocks"].append({"columns": cols, "rows": np.array(rows, float).reshape(-1, len(cols))})
            cols, rows = None, []
        t = s.split()
        if t and t[0] == "Step" and len(t) > 1 and t[1] != "size":
            cols = [THERMO_NAMES.get(c, c.lower()) for c in t]
        elif s.startswith("LMPT "):
            parts = s.split()
            d = {}
            for k in range(2, len(parts) - 1, 2):
                v = _float(parts[k + 1])
                d[parts[k]] = v if v is not None else parts[k + 1]
            out["tags"].append((parts[1], d))
        elif s.startswith("ERROR"):
            out["errors"].append(" ".join(lines[i:i + 2]).strip())
        elif s.startswith("WARNING"):
            out["warnings"].append(s)
        elif s.startswith("Stopping criterion ="):
            out["stops"].append(s.split("=", 1)[1].strip())
        elif s.startswith("Loop time of"):
            m = re.match(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps", s)
            if m:
                out["loops"].append({"seconds": float(m.group(1)), "procs": int(m.group(2)), "steps": int(m.group(3))})
        i += 1
    if cols is not None:
        out["blocks"].append({"columns": cols, "rows": np.array(rows, float).reshape(-1, len(cols))})
    return out


def tags(log: dict) -> dict:
    """Last value of each tag."""
    return {k: v for k, v in log["tags"]}


def read_dump(path: Path) -> dict:
    """One-frame text dump: columns from the 'ITEM: ATOMS' line; box."""
    lines = Path(path).read_text().splitlines()
    out = {}
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: ATOMS"):
            cols = ln.split()[2:]
            data = np.array([[float(x) for x in l.split()] for l in lines[i + 1:] if l.strip() and not l.startswith("ITEM")])
            for k, c in enumerate(cols):
                out[c] = data[:, k]
            break
    return out


def read_ave_vector(path: Path) -> list[np.ndarray]:
    """fix ave/time mode vector files: list of (rows x cols) blocks, one per output time."""
    blocks, cur, n = [], [], 0
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        t = ln.split()
        if len(t) == 2 and n == 0:
            n = int(t[1])
            cur = []
            continue
        cur.append([float(x) for x in t])
        if len(cur) == n:
            blocks.append(np.array(cur))
            n = 0
    return blocks


def last_step(path: Path) -> int | None:
    """Last thermo step printed in a log (for progress), reading only the tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20000))
            tail = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for ln in reversed(tail):
        t = ln.split()
        if len(t) > 5 and t[0].isdigit() and all(_float(x) is not None for x in t[:4]):
            return int(t[0])
    return None
