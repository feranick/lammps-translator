"""potential.py — interactions: potential files, force-field typing, and the LAMMPS commands that set them.
Version: 2026.10.18.2

Every pair coefficient is written explicitly (mixing done here, not by LAMMPS), so the reference implementation
uses the same numbers. 'model' (from model()) is the normalised description refcalc.py evaluates.
"""
from __future__ import annotations

import importlib.util
import math
import os
import shutil
from pathlib import Path

import numpy as np

from .data import MOLECULES
from .format import FILE_STYLES, molecular


# ------------------------------------------------------------------------------------------ potential files
def potential_dirs() -> list[Path]:
    out = []
    for var in ("LMPT_POTENTIALS", "LAMMPS_POTENTIALS"):
        for p in os.environ.get(var, "").split(os.pathsep):
            if p:
                out.append(Path(p).expanduser())
    out += [Path("/usr/share/lammps/potentials"), Path("/usr/local/share/lammps/potentials"),
            Path("/opt/lammps/potentials")]
    try:
        spec = importlib.util.find_spec("lammps")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                out.append(Path(loc) / "share" / "lammps" / "potentials")
    except (ImportError, ValueError):
        pass
    exe = shutil.which(os.environ.get("LMPT_LMP", "lmp").split()[-1]) if os.environ.get("LMPT_LMP", "lmp") else None
    if exe:
        out.append(Path(exe).resolve().parent.parent / "share" / "lammps" / "potentials")
    seen, res = set(), []
    for p in out:
        if p.is_dir() and str(p) not in seen:
            seen.add(str(p)); res.append(p)
    return res


def find_file(name: str) -> Path | None:
    p = Path(name).expanduser()
    if p.is_absolute() or "/" in name:
        return p if p.is_file() else None
    for d in potential_dirs():
        if (d / name).is_file():
            return d / name
    return None


def _floats(path: Path, skip: int, n: int):
    """n numbers after 'skip' lines (eam files wrap values freely)."""
    vals = []
    with open(path) as f:
        for _ in range(skip):
            f.readline()
        for ln in f:
            for t in ln.split():
                vals.append(float(t))
            if len(vals) >= n:
                break
    return vals[:n]


def file_info(style: str, path: Path) -> dict:
    """Elements and cutoff from a potential file (what validation and auto-size need)."""
    info = {"elements": None, "cutoff": None}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return info
    if style == "eam":
        t = lines[2].split()
        info["cutoff"] = float(t[4])
        info["mass"] = float(lines[1].split()[1])
    elif style in ("eam/alloy", "eam/fs", "adp"):
        els = lines[3].split()[1:]
        info["elements"] = els
        h = lines[4].split()
        info["cutoff"] = float(h[4])
        nrho, nr = int(h[0]), int(h[2])
        # element blocks: 'Z mass a lattice', F(nrho), rho(nr) (fs: rho(nr) per element), read line by line
        from .refcalc import _Reader
        rd = _Reader(lines)
        rd.k = 5
        masses = {}
        try:
            for e in els:
                masses[e] = float(rd.line()[1])
                rd.values(nrho)
                for _ in range(len(els) if style == "eam/fs" else 1):
                    rd.values(nr)
        except (IndexError, ValueError):
            masses = {}
        info["masses"] = masses
    elif style.startswith("tersoff") or style in ("sw", "vashishta"):
        els, cut = set(), 0.0
        toks = []
        for ln in lines:
            ln = ln.split("#")[0]
            toks += ln.split()
        nper = {"sw": 14, "tersoff": 17, "tersoff/mod": 20, "tersoff/zbl": 21, "vashishta": 17}.get(style, 17)
        for k in range(0, len(toks) - nper + 1, nper):
            e = toks[k:k + 3]
            els.update(e)
            try:
                p = [float(x) for x in toks[k + 3:k + nper]]
            except ValueError:
                break
            if style == "sw":
                cut = max(cut, p[1] * p[2])
            elif style == "tersoff":
                cut = max(cut, p[10] + p[11])
        info["elements"] = sorted(els)
        info["cutoff"] = cut or None
    return info


# ------------------------------------------------------------------------------------------ force fields
def forcefield_of(study: dict) -> dict:
    """Built-in molecules' force fields merged with the study's; defaults filled in."""
    ff = {"types": {}, "pairs": {}, "bonds": {}, "angles": {}, "dihedrals": {}, "rigid": []}
    for m in study["system"].get("molecules") or []:
        if isinstance(m.get("molecule"), str) and m["molecule"] in MOLECULES:
            b = MOLECULES[m["molecule"]]["forcefield"]
            for k in ("types", "bonds", "angles", "dihedrals"):
                ff[k].update(b.get(k, {}))
            ff["rigid"] += [r for r in b.get("rigid", []) if r not in ff["rigid"]]
    user = study.get("forcefield") or {}
    for k in ("types", "pairs", "bonds", "angles", "dihedrals"):
        ff[k].update(user.get(k) or {})
    if "rigid" in user:
        ff["rigid"] = list(user["rigid"])
    ff["mixing"] = user.get("mixing", "geometric")
    ff["cutoff"] = float(user.get("cutoff", 10.0))
    ff["coulomb"] = user.get("coulomb", "long")
    ff["kspace_accuracy"] = float(user.get("kspace_accuracy", 1e-5))
    ff["tail"] = bool(user.get("tail", True))
    ff["special"] = list(user.get("special", [0.0, 0.0, 0.0]))
    return ff


def _canon(labels):
    t = tuple(labels)
    r = tuple(reversed(t))
    return "-".join(min(t, r))


def assign_forcefield(s, ff: dict, keep_charges: bool = False) -> None:
    """Charges, bond/angle/dihedral types from atom labels (order-insensitive keys)."""
    names = s.type_names
    if not keep_charges:
        s.charges = np.array([float(ff["types"][names[t - 1]].get("q", 0.0)) for t in s.types])
    raw = getattr(s, "_raw_topology", {"bonds": [], "angles": [], "dihedrals": []})
    for key, coeffs in (("bonds", ff["bonds"]), ("angles", ff["angles"]), ("dihedrals", ff["dihedrals"])):
        table = {_canon(k.split("-")): k for k in coeffs}
        labels, items, missing = [], [], []
        for labs, atoms in raw[key]:
            c = _canon(labs)
            if c not in table:
                if c not in missing:
                    missing.append(c)
                continue
            if c not in labels:
                labels.append(c)
            items.append((labels.index(c) + 1, *atoms))
        if missing and key != "dihedrals":
            raise ValueError(f"forcefield.{key}: no coefficients for {missing} (keys like '{missing[0]}')")
        if missing:
            s.warnings.append(f"forcefield.dihedrals: none for {missing}; those dihedrals are left out")
        setattr(s, key, items)
        setattr(s, {"bonds": "bond_types", "angles": "angle_types", "dihedrals": "dihedral_types"}[key], labels)
    s._ff = ff


# ------------------------------------------------------------------------------------------ model
def _pairs_table(p: dict, els: list[str], need: tuple) -> dict:
    out = {}
    for k, v in (p.get("pairs") or {}).items():
        a, b = k.split("-")
        out[tuple(sorted((a, b)))] = {n: float(v[n]) for n in need}
    if p["style"] == "lj" and not p.get("pairs") and "eps" in p:
        out[(els[0], els[0])] = {"eps": float(p["eps"]), "sigma": float(p["sigma"])}
    return out


def potential_cutoff(study: dict) -> float | None:
    if molecular(study):
        return forcefield_of(study)["cutoff"]
    p = study.get("potential") or {}
    st = p.get("style")
    if "cutoff" in p:
        return float(p["cutoff"])
    if st == "lj":
        els = list(study["system"].get("elements") or study["system"].get("type_names") or [])
        tab = _pairs_table(p, els, ("eps", "sigma"))
        return 2.5 * max(v["sigma"] for v in tab.values()) if tab else None
    if st in FILE_STYLES:
        name = p.get("file") or (list(p.get("files", {}).values())[0] if isinstance(p.get("files"), dict) else None)
        if st in ("meam", "snap"):
            return 6.0
        f = find_file(name) if name else None
        return file_info(st, f)["cutoff"] if f else None
    return None


def model(study: dict, s) -> dict:
    """Normalised interactions over atom types (what LAMMPS gets and refcalc evaluates)."""
    nt = len(s.type_names)
    names = s.type_names
    if molecular(study):
        ff = getattr(s, "_ff", None) or forcefield_of(study)
        T = ff["types"]
        pair = {}
        for i in range(nt):
            for j in range(i, nt):
                a, b = names[i], names[j]
                key = next((k for k in ff["pairs"] if sorted(k.split("-")) == sorted([a, b])), None)
                if key:
                    eps, sig = float(ff["pairs"][key]["eps"]), float(ff["pairs"][key]["sigma"])
                else:
                    ei, ej, si, sj = float(T[a]["eps"]), float(T[b]["eps"]), float(T[a]["sigma"]), float(T[b]["sigma"])
                    eps = math.sqrt(ei * ej)
                    sig = math.sqrt(si * sj) if ff["mixing"] == "geometric" else 0.5 * (si + sj)
                pair[(i + 1, j + 1)] = {"eps": eps, "sigma": sig}
        bonds = [dict(ff["bonds"][next(k for k in ff["bonds"] if _canon(k.split("-")) == lab)]) for lab in s.bond_types]
        angles = [dict(ff["angles"][next(k for k in ff["angles"] if _canon(k.split("-")) == lab)]) for lab in s.angle_types]
        dihs = [dict(ff["dihedrals"][next(k for k in ff["dihedrals"] if _canon(k.split("-")) == lab)])
                for lab in s.dihedral_types]
        rigid_b = [k + 1 for k, lab in enumerate(s.bond_types) if any(_canon(r.split("-")) == lab for r in ff["rigid"])]
        rigid_a = [k + 1 for k, lab in enumerate(s.angle_types) if any(_canon(r.split("-")) == lab for r in ff["rigid"])]
        return {"units": "real", "kind": "lj", "pair": pair, "cutoff": ff["cutoff"], "shift": False,
                "tail": ff["tail"], "coulomb": ff["coulomb"] if np.any(s.charges != 0) else "none",
                "kspace_accuracy": ff["kspace_accuracy"], "special": ff["special"], "bonds": bonds,
                "angles": angles, "dihedrals": dihs, "rigid_bonds": rigid_b, "rigid_angles": rigid_a,
                "molecular": True}
    p = study["potential"]
    st = p["style"]
    m = {"units": "metal", "kind": st, "molecular": False, "special": [0.0, 0.0, 0.0], "bonds": [], "angles": [],
         "dihedrals": []}
    if st in ("lj", "morse", "buck", "coul"):
        need = {"lj": ("eps", "sigma"), "morse": ("D0", "alpha", "r0"), "buck": ("A", "rho", "C"), "coul": ()}[st]
        tab = _pairs_table(p, names, need) if st != "coul" else {}
        pair = {}
        for i in range(nt):
            for j in range(i, nt):
                a, b = names[i], names[j]
                v = tab.get(tuple(sorted((a, b))))
                if v is None and st == "lj":
                    va, vb = tab.get((a, a)), tab.get((b, b))
                    if va is None or vb is None:
                        raise ValueError(f"potential.pairs: no '{a}-{b}' and no '{a}-{a}' and '{b}-{b}' to mix from")
                    v = {"eps": math.sqrt(va["eps"] * vb["eps"]), "sigma": math.sqrt(va["sigma"] * vb["sigma"])}
                if v is None:
                    v = {"morse": {"D0": 0.0, "alpha": 1.0, "r0": 1.0}, "buck": {"A": 0.0, "rho": 1.0, "C": 0.0},
                         "coul": {}}[st]
                pair[(i + 1, j + 1)] = v
        m["pair"] = pair
        m["cutoff"] = float(p["cutoff"]) if "cutoff" in p else potential_cutoff(study)
        m["shift"] = bool(p.get("shift", False))
        m["tail"] = bool(p.get("tail", False))
        q = p.get("charges")
        m["coulomb"] = p.get("coulomb", "long" if (q or st == "coul") else "none")
        m["kspace_accuracy"] = float(p.get("kspace_accuracy", 1e-8))
        return m
    m["files"] = p.get("files") if st in ("eam", "meam", "snap") and p.get("files") else None
    m["file"] = p.get("file")
    m["args"] = p.get("args", "3.0 1 1" if st == "airebo" else "")
    m["coulomb"] = "none"
    return m


def resolve(study: dict, s) -> dict:
    """Finds potential files and checks they fit the system (static, on the machine that runs LAMMPS)."""
    E, W, files = [], [], {}
    if molecular(study):
        return {"errors": E, "warnings": W, "files": files}
    p = study["potential"]
    st = p["style"]
    q = p.get("charges")
    if q and s.charges is not None:
        tot = float(np.sum(s.charges))
        if p.get("coulomb", "long") == "long" and abs(tot) > 1e-6:
            E.append(f"potential.charges: the cell has net charge {tot:g} e; long-range Coulomb needs a neutral cell")
    if st not in FILE_STYLES:
        return {"errors": E, "warnings": W, "files": files}
    names = []
    if st == "eam":
        f = p.get("files") or {s.type_names[0]: p["file"]}
        names = list(f.values())
    elif st in ("meam", "snap"):
        names = list(p["files"])
    else:
        names = [p["file"]]
    dirs = potential_dirs()
    for nm in names:
        path = find_file(nm)
        if path is None:
            msg = (f"potential: file '{nm}' not found in {[str(d) for d in dirs] or 'any potentials folder'} "
                   "(set LMPT_POTENTIALS, or give a path on the server)")
            (E if dirs or "/" in nm else W).append(msg)
            continue
        files[nm] = str(path)
        info = file_info(st, path)
        # LAMMPS sets the masses from eam files (pair_coeff overrides the data file): use the same
        if st == "eam" and "mass" in info:
            for k, e in enumerate(s.type_names):
                if (p.get("files") or {e: p.get("file")}).get(e) == nm:
                    s.masses[k] = info["mass"]
        for e, mval in (info.get("masses") or {}).items():
            if e in s.type_names:
                s.masses[s.type_names.index(e)] = mval
        if info.get("elements") is not None:
            miss = [e for e in s.type_names if e not in info["elements"]]
            if miss:
                E.append(f"potential: {nm} has elements {info['elements']}, not {miss}")
    return {"errors": E, "warnings": W, "files": files}


# ------------------------------------------------------------------------------------------ LAMMPS commands
def _g(x: float) -> str:
    return f"{x:.12g}"


def commands(study: dict, s, m: dict, pot_names: dict | None = None) -> tuple[list[str], list[str]]:
    """(lines before read_data, lines after read_data) that set units, styles and coefficients.
    pot_names maps potential file names to the names written in the input (copies next to it)."""
    pot_names = pot_names or {}
    nt = len(s.type_names)
    pre = [f"units {m['units']}", f"atom_style {s.atom_style}", "boundary p p p"]
    post = []
    if m["molecular"]:
        if s.bond_types:
            pre.append("bond_style harmonic")
        if s.angle_types:
            pre.append("angle_style harmonic")
        if s.dihedral_types:
            pre.append("dihedral_style opls")
    rc = m.get("cutoff")
    st = m["kind"]
    coul = m.get("coulomb", "none")
    if st in ("lj", "morse", "buck", "coul"):
        base = {"lj": "lj/cut", "morse": "morse", "buck": "buck", "coul": ""}[st]
        if coul == "none":
            style = base
        else:
            style = (base + "/" if base else "") + ("coul/long" if coul == "long" else "coul/cut")
        post.append(f"pair_style {style} {_g(rc)}")
        if st == "coul":
            post.append("pair_coeff * *")
        for (i, j), v in sorted(m["pair"].items()):
            if st == "lj":
                post.append(f"pair_coeff {i} {j} {_g(v['eps'])} {_g(v['sigma'])}")
            elif st == "morse":
                post.append(f"pair_coeff {i} {j} {_g(v['D0'])} {_g(v['alpha'])} {_g(v['r0'])}")
            elif st == "buck":
                post.append(f"pair_coeff {i} {j} {_g(v['A'])} {_g(v['rho'])} {_g(v['C'])}")
        mods = []
        if m.get("shift"):
            mods.append("shift yes")
        if m.get("tail"):
            mods.append("tail yes")
        if coul == "long" and not m["molecular"]:
            mods.append("table 0")
        if mods:
            post.append("pair_modify " + " ".join(mods))
        if coul == "long":
            kind = "ewald" if s.n <= 2000 and not m["molecular"] else "pppm"
            post.append(f"kspace_style {kind} {m['kspace_accuracy']:g}")
    else:
        els = " ".join(s.type_names)
        fn = lambda n: pot_names.get(n, n)
        args = (" " + m["args"]) if m.get("args") else ""
        post.append(f"pair_style {st}{args}")
        if st == "eam":
            files = m["files"] or {s.type_names[0]: m["file"]}
            for k, e in enumerate(s.type_names):
                post.append(f"pair_coeff {k + 1} {k + 1} {fn(files[e])}")
        elif st == "meam":
            lib, par = m["files"]
            uniq = " ".join(dict.fromkeys(s.type_names))
            post.append(f"pair_coeff * * {fn(lib)} {uniq} {fn(par)} {els}")
        elif st == "snap":
            c, par = m["files"]
            post.append(f"pair_coeff * * {fn(c)} {fn(par)} {els}")
        else:
            post.append(f"pair_coeff * * {fn(m['file'])} {els}")
    if m["molecular"]:
        for k, b in enumerate(m["bonds"]):
            post.append(f"bond_coeff {k + 1} {_g(b['k'])} {_g(b['r0'])}")
        for k, a in enumerate(m["angles"]):
            post.append(f"angle_coeff {k + 1} {_g(a['k'])} {_g(a['theta0'])}")
        for k, d in enumerate(m["dihedrals"]):
            post.append(f"dihedral_coeff {k + 1} " + " ".join(_g(x) for x in d["k"]))
        sp = " ".join(_g(x) for x in m["special"])
        post.append(f"special_bonds lj {sp} coul {sp}")
    post.append(f"neighbor 2.0 bin")
    post.append("neigh_modify every 1 delay 0 check yes")
    return pre, post
