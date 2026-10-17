"""deck.py — LAMMPS input scripts, one per stage, written from the current state just before the stage runs.
Version: 2026.10.16.1

Every number the post-processing needs is printed by LAMMPS as a tagged line
    LMPT <tag> pe <v> pxx <v> ... (thermo keywords, %.15g)
after 'run 0' (or a minimization), so results never depend on LAMMPS's thermo layout. Strains, volumes and
removed atoms are computed here and kept in the stage's 'meta' for the analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .build import System, oriented_cell, write_data
from .data import UNITS, lattice_cell

EMIT_KEYS = ("pe", "ke", "etotal", "temp", "press", "pxx", "pyy", "pzz", "pyz", "pxz", "pxy", "vol",
             "lx", "ly", "lz", "xy", "xz", "yz", "fmax", "fnorm", "atoms", "evdwl", "ecoul", "elong", "ebond",
             "eangle", "edihed", "density")
THERMO = "step temp pe ke etotal press pxx pyy pzz pyz pxz pxy vol lx ly lz xy xz yz density"


@dataclass
class Ctx:
    study: dict
    model: dict
    pre: list
    post: list
    units: dict = field(default_factory=dict)
    seed: int = 12345
    initial: System | None = None   # the study's initial structure (lattice reference)
    iso: dict = field(default_factory=dict)


def emit(tag: str, run0: bool = True) -> list[str]:
    vals = " ".join(f"{k} $({k}:%.15g)" if k != "atoms" else "atoms $(atoms)" for k in EMIT_KEYS)
    return (["run 0"] if run0 else []) + [f'print "LMPT {tag} {vals}"']


def header(ctx: Ctx, data_file: str, title: str) -> list[str]:
    return [f"# lammps-translator: {title}", *ctx.pre, f"read_data {data_file}", *ctx.post,
            f"thermo_style custom {THERMO}", "thermo_modify format float %.15g flush yes norm no"]


def _p_native(ctx: Ctx, p_bar: float) -> float:
    return p_bar if ctx.model["units"] == "metal" else p_bar / 1.01325


def _minimize(ftol: float = 1e-8, maxiter: int = 5000) -> list[str]:
    return ["min_style cg", f"minimize 0.0 {ftol:g} {maxiter} {10 * maxiter}"]


def _zero_v() -> list[str]:
    return ["velocity all set 0.0 0.0 0.0"]


def _box_final(box, tri: bool = True) -> str:
    lx, ly, lz, xy, xz, yz = box
    tilt = f"xy final {xy:.16g} xz final {xz:.16g} yz final {yz:.16g} " if tri else ""
    return (f"change_box all x final 0.0 {lx:.16g} y final 0.0 {ly:.16g} z final 0.0 {lz:.16g} "
            f"{tilt}remap units box")


def strained_box(box, voigt: int, eps: float):
    """Box after the deformation gradient F = I + eps E_voigt (upper triangular: keeps LAMMPS's restricted
    triclinic form; engineering shear strain = eps)."""
    lx, ly, lz, xy, xz, yz = box
    if voigt == 1:
        return [lx * (1 + eps), ly, lz, xy * (1 + eps), xz * (1 + eps), yz]
    if voigt == 2:
        return [lx, ly * (1 + eps), lz, xy, xz, yz * (1 + eps)]
    if voigt == 3:
        return [lx, ly, lz * (1 + eps), xy, xz, yz]
    if voigt == 4:
        return [lx, ly, lz, xy, xz, yz + eps * lz]
    if voigt == 5:
        return [lx, ly, lz, xy, xz + eps * lz, yz]
    return [lx, ly, lz, xy + eps * ly, xz + eps * yz, yz]


# ------------------------------------------------------------------------------------------- stages
def setup(ctx: Ctx, state_file: str) -> dict:
    L = header(ctx, state_file, "setup: the initial structure as built") + _zero_v() + emit("initial")
    L.append("write_dump all custom forces0.dump id type x y z fx fy fz modify sort id format float %.15g")
    iso = []
    m = ctx.model
    if not m["molecular"] and m.get("coulomb", "none") == "none":
        rc = m.get("cutoff") or 10.0
        box = 2.0 * rc + 10.0
        nt = len(ctx.initial.type_names)
        for k in range(nt):
            L += ["", "clear", *ctx.pre[:3], f"region b block 0 {box:g} 0 {box:g} 0 {box:g}", f"create_box {nt} b",
                  f"create_atoms {k + 1} single {box / 2:g} {box / 2:g} {box / 2:g}", "mass * 1.0", *ctx.post,
                  "thermo_modify format float %.15g norm no", *emit(f"iso{k + 1}")]
            iso.append(k + 1)
    return {"input": L, "meta": {"iso_types": iso}, "steps": 0}


def energy(ctx: Ctx, st: dict, state: System, state_file: str) -> dict:
    return {"input": header(ctx, state_file, "energy") + _zero_v() + emit("E"), "meta": {}, "steps": 0}


def minimize(ctx: Ctx, st: dict, state: System, state_file: str) -> dict:
    box = st.get("box", "fixed")
    ftol, maxiter = float(st.get("ftol", 1e-8)), int(st.get("maxiter", 5000))
    L = header(ctx, state_file, f"minimize (box {box})") + _zero_v()
    cycles = 0
    if box != "fixed":
        if box == "tri" and not state.triclinic():
            L.append("change_box all triclinic")
        P = _p_native(ctx, float(st.get("P_bar", 0.0)))
        cycles = 5
        for c in range(cycles):
            L += [f"fix br all box/relax {box} {P:.10g} vmax 0.001", "min_modify line quadratic",
                  *_minimize(ftol, maxiter), "unfix br", *emit(f"cycle{c + 1}")]
    L += ["min_modify line quadratic", *_minimize(ftol, maxiter), *emit("final"),
          "write_data state.data nocoeff"]
    return {"input": L, "meta": {"cycles": cycles, "maxiter": maxiter}, "steps": 0, "writes_state": True}


def _sample(steps: int, af: float):
    prod = max(1, int(steps * (1 - af)))
    n = max(1, prod // 100)
    while steps % n:
        n -= 1
    return n, max(1, prod // n)


def md(ctx: Ctx, st: dict, state: System, state_file: str, index: int) -> dict:
    kind = st["kind"]
    m = ctx.model
    steps = int(st["steps"])
    dt_fs = float(st.get("dt_fs", 2.0 if m.get("rigid_bonds") else 1.0))
    dt = dt_fs * UNITS[m["units"]]["time_per_fs"]
    T = float(st.get("T", 0.0))
    T2 = float(st.get("T_end", T))
    seed = int(st.get("seed", ctx.seed + 1000 * index))
    tdamp = float(st.get("tdamp_fs", 100 * dt_fs)) * UNITS[m["units"]]["time_per_fs"]
    pdamp = float(st.get("pdamp_fs", 1000 * dt_fs)) * UNITS[m["units"]]["time_per_fs"]
    every = int(st.get("thermo_every", max(1, steps // 200)))
    af = float(st.get("average_from", 0.5))
    L = header(ctx, state_file, f"{kind} {steps} steps") + [f"timestep {dt:.10g}"]
    if m.get("rigid_bonds") or m.get("rigid_angles"):
        L.append("fix rig all shake 1e-6 200 0 " + " ".join(
            [f"b {b}" for b in m["rigid_bonds"]] + [f"a {a}" for a in m["rigid_angles"]]))
    create = st.get("velocities", "create" if state.velocities is None or not np.any(state.velocities) else "keep")
    if create == "create" or state.velocities is None:
        if T > 0:
            L.append(f"velocity all create {T:g} {seed} mom yes rot yes dist gaussian")
    therm = st.get("thermostat", "nose_hoover")
    if kind == "nve":
        L.append("fix int all nve")
    elif kind == "nvt":
        if therm == "langevin":
            L += ["fix int all nve", f"fix th all langevin {T:g} {T2:g} {tdamp:.10g} {seed + 1} zero yes"]
        else:
            L.append(f"fix int all nvt temp {T:g} {T2:g} {tdamp:.10g}")
    else:
        P = _p_native(ctx, float(st.get("P_bar", 0.0)))
        cpl = st.get("coupling", "iso")
        if cpl == "tri" and not state.triclinic():
            L.insert(L.index(f"timestep {dt:.10g}"), "change_box all triclinic")
        if therm == "langevin":
            L += [f"fix int all nph {cpl} {P:.10g} {P:.10g} {pdamp:.10g}",
                  f"fix th all langevin {T:g} {T2:g} {tdamp:.10g} {seed + 1} zero yes"]
        else:
            L.append(f"fix int all npt temp {T:g} {T2:g} {tdamp:.10g} {cpl} {P:.10g} {P:.10g} {pdamp:.10g}")
    thermo = THERMO
    meta = {"dt_ps": dt_fs * 1e-3, "steps": steps, "every": every, "average_from": af, "T": T, "T_end": T2}
    if "msd" in st:
        of = (st["msd"] or {}).get("of")
        if of:
            ids = [state.type_names.index(t) + 1 for t in of]
            L.append("group msdg type " + " ".join(map(str, ids)))
            L.append("compute msd msdg msd com yes")
        else:
            L.append("compute msd all msd com yes")
        thermo += " c_msd[4]"
        meta["msd"] = True
    if "rdf" in st:
        r = st["rdf"] or {}
        bins = int(r.get("bins", 200))
        pairs = r.get("pairs") or []
        args = " ".join(f"{state.type_names.index(a) + 1} {state.type_names.index(b) + 1}" for a, b in pairs)
        rc = float(m.get("cutoff") or 10.0)
        rmax = float(r.get("r_max", rc))
        cut = ""
        if rmax > rc:
            L.append(f"comm_modify cutoff {rmax + 2.0:g}")
            cut = f" cutoff {rmax:g}"
        elif rmax < rc:
            cut = f" cutoff {rmax:g}"
        L.append(f"compute rdf all rdf {bins} {args}{cut}".rstrip())
        ne, nr = _sample(steps, af)
        L.append(f"fix rdfav all ave/time {ne} {nr} {steps} c_rdf[*] file rdf.txt mode vector")
        meta["rdf"] = {"pairs": pairs or [["all", "all"]], "bins": bins, "r_max": rmax}
    if st.get("dump_every"):
        L += [f"dump trj all custom {int(st['dump_every'])} traj.lammpstrj id type x y z",
              "dump_modify trj sort id"]
    L += [f"thermo_style custom {thermo}", "thermo_modify format float %.15g flush yes norm no", f"thermo {every}",
          f"run {steps}"]
    if st.get("dump_every"):
        L.append("undump trj")
    L += emit("final", run0=False) + ["write_data state.data nocoeff"]
    return {"input": L, "meta": meta, "steps": steps, "writes_state": True}


def elastic(ctx: Ctx, st: dict, state: System, state_file: str) -> dict:
    eps = float(st.get("strain", 1e-3))
    relax = bool(st.get("relax", True))
    L = header(ctx, state_file, "elastic constants: reference") + _zero_v() + ["change_box all triclinic"] + emit("ref")
    for v in range(1, 7):
        for sgn, tag in ((1, "p"), (-1, "m")):
            b = strained_box(state.box, v, sgn * eps)
            L += ["", "clear"] + header(ctx, state_file, f"strain {v} {sgn * eps:g}")[1:] + _zero_v()
            L += ["change_box all triclinic", _box_final(b)]
            if relax:
                L += _minimize(1e-10, 5000)
            L += emit(f"s{v}{tag}")
    return {"input": L, "meta": {"strain": eps, "relax": relax}, "steps": 0}


def eos(ctx: Ctx, st: dict, state: System, state_file: str) -> dict:
    smax = float(st.get("strain", 0.04))
    n = int(st.get("n", 9))
    relax = bool(st.get("relax", True))
    scales = np.linspace(1 - smax, 1 + smax, n)
    L = header(ctx, state_file, "equation of state")[0:1]
    for k, f in enumerate(scales):
        b = [state.box[i] * f for i in range(6)]
        L += ([] if k == 0 else ["", "clear"]) + header(ctx, state_file, f"scale {f:.6f}")[1:] + _zero_v()
        L.append(_box_final(b, state.triclinic()))
        if relax:
            L += _minimize(1e-10, 5000)
        L += emit(f"v{k}")
    return {"input": L, "meta": {"scales": scales.tolist(), "relax": relax}, "steps": 0}


def vacancy(ctx: Ctx, st: dict, state: System, state_file: str, folder: Path) -> dict:
    el = st.get("element", state.type_names[0])
    if el not in state.type_names:
        raise ValueError(f"vacancy.element: '{el}' is not in the system {state.type_names}")
    relax = bool(st.get("relax", True))
    t = state.type_names.index(el) + 1
    H = state.cell()
    centre = H.sum(axis=0) / 2.0
    cand = np.where(state.types == t)[0]
    d = np.linalg.norm(state.pos[cand] - centre, axis=1)
    rm = int(cand[np.argmin(d)])
    keep = np.ones(state.n, bool)
    keep[rm] = False
    v = state.copy()
    v.pos, v.types = v.pos[keep], v.types[keep]
    if v.charges is not None:
        v.charges = v.charges[keep]
    if v.velocities is not None:
        v.velocities = None
    write_data(v, folder / "vacancy.data")
    L = header(ctx, state_file, "vacancy: perfect crystal") + _zero_v()
    if relax:
        L += _minimize(1e-10, 10000)
    L += emit("perfect") + ["", "clear"] + header(ctx, "vacancy.data", "vacancy")[1:]
    if relax:
        L += _minimize(1e-10, 10000)
    L += emit("vacancy")
    return {"input": L, "meta": {"removed_atom": rm + 1, "element": el, "relax": relax, "n": state.n}, "steps": 0}


def surface_orient(miller):
    n = np.array(miller, int)
    best = None
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                v = np.array([i, j, k])
                if not v.any() or v @ n:
                    continue
                key = (v @ v, -v[0], -v[1], -v[2])
                if best is None or key < best[0]:
                    best = (key, v)
    x = best[1]
    y = np.cross(n, x)
    g = math.gcd(math.gcd(abs(int(y[0])), abs(int(y[1]))), abs(int(y[2])))
    y = y // g
    return {"x": x.tolist(), "y": y.tolist(), "z": n.tolist()}


def surface(ctx: Ctx, st: dict, state: System, state_file: str, folder: Path) -> dict:
    sysd = dict(ctx.study["system"])
    lat0 = ctx.initial.lattice
    vol_scale = (state.volume() / ctx.initial.volume()) ** (1 / 3)
    kind = sysd["lattice"]
    mil = [int(t) for t in st["miller"]]
    if kind == "hcp":
        sysd["a"] = float(sysd["a"]) * state.box[0] / ctx.initial.box[0]
        sysd["c"] = (lat0["c"] or lattice_cell("hcp", float(lat0["a"]))[2]) * state.box[2] / ctx.initial.box[2]
        orient = None
    elif kind == "custom":
        sysd["cell"] = [c * state.box[i] / ctx.initial.box[i] for i, c in enumerate(sysd["cell"])]
        orient = None
        if mil != [0, 0, 1] and [abs(t) for t in mil] != [0, 0, 1]:
            ax = [abs(t) for t in mil].index(1)
            perm = {0: [1, 2, 0], 1: [2, 0, 1]}[ax]
            sysd["cell"] = [sysd["cell"][p] for p in perm]
            sysd["basis"] = [[b[0], b[1 + perm[0]], b[1 + perm[1]], b[1 + perm[2]]] for b in sysd["basis"]]
    else:
        sysd["a"] = float(sysd["a"]) * vol_scale
        orient = surface_orient(mil)
    Lc, pos, sites = oriented_cell(sysd, orient)
    rc = float(ctx.model.get("cutoff") or 6.0)
    nx = max(1, int(math.ceil(max(10.0, 2 * rc) / Lc[0] - 1e-9)))
    ny = max(1, int(math.ceil(max(10.0, 2 * rc) / Lc[1] - 1e-9)))
    nz = int(st.get("layers") or max(2, int(math.ceil(max(20.0, 3 * rc) / Lc[2] - 1e-9))))
    shifts = np.array([[i, j, k] for i in range(nx) for j in range(ny) for k in range(nz)], float) * np.array(Lc)
    P = (shifts[:, None, :] + pos[None, :, :]).reshape(-1, 3)
    S = np.tile(sites, len(shifts))
    els = list(sysd["elements"])
    names = [els[s] for s in S]
    tn = state.type_names
    for nm in set(names):
        if nm not in tn:
            raise ValueError(f"surface: element {nm} is not a type of the current state {tn}")
    bulk = System(box=[Lc[0] * nx, Lc[1] * ny, Lc[2] * nz, 0, 0, 0], pos=P, types=np.array([tn.index(nm) + 1 for nm in names]),
                  type_names=list(tn), masses=list(state.masses), atom_style=state.atom_style)
    if state.charges is not None:
        q = {state.type_names[t - 1]: c for t, c in zip(state.types, state.charges)}
        bulk.charges = np.array([q.get(nm, 0.0) for nm in names])
    if any(s.get("to") for s in (ctx.study["system"].get("substitute") or [])):
        raise ValueError("surface: not available for substituted (random) lattices")
    vac = float(st.get("vacuum", 15.0))
    slab = bulk.copy()
    slab.box = [bulk.box[0], bulk.box[1], bulk.box[2] + vac, 0, 0, 0]
    slab.pos = bulk.pos + np.array([0, 0, vac / 2.0])
    # shift atoms so no plane sits on the periodic boundary of the bulk cell
    zs = np.unique(np.round(bulk.pos[:, 2] % bulk.box[2], 6))
    gaps = np.diff(np.concatenate([zs, [zs[0] + bulk.box[2]]]))
    k = int(np.argmax(gaps))
    cut = (zs[k] + gaps[k] / 2.0) % bulk.box[2]
    for sysx, extra in ((bulk, 0.0), (slab, vac / 2.0)):
        z = (bulk.pos[:, 2] - cut) % bulk.box[2]
        sysx.pos = np.column_stack([bulk.pos[:, 0], bulk.pos[:, 1], z + extra])
    write_data(bulk, folder / "surface_bulk.data")
    write_data(slab, folder / "surface_slab.data")
    relax = bool(st.get("relax", True))
    L = header(ctx, "surface_bulk.data", f"surface {mil}: bulk cell") + _zero_v()
    if relax:
        L += _minimize(1e-10, 10000)
    L += emit("bulk") + ["", "clear"] + header(ctx, "surface_slab.data", "slab")[1:] + emit("slab_unrelaxed")
    if relax:
        L += _minimize(1e-10, 10000) + emit("slab")
    return {"input": L, "meta": {"miller": mil, "orient": orient, "area_A2": bulk.box[0] * bulk.box[1],
                                 "n": bulk.n, "repeats": [nx, ny, nz], "vacuum": vac, "relax": relax,
                                 "thickness_A": bulk.box[2]}, "steps": 0}


def thermal_expansion(ctx: Ctx, st: dict, state: System, state_file: str, index: int) -> dict:
    Ts = [float(t) for t in st["T"]]
    steps = int(st.get("steps", 20000))
    m = ctx.model
    dt_fs = float(st.get("dt_fs", 1.0))
    dt = dt_fs * UNITS[m["units"]]["time_per_fs"]
    P = _p_native(ctx, float(st.get("P_bar", 0.0)))
    every = max(1, steps // 200)
    cpl = "aniso" if (ctx.initial.lattice and ctx.initial.lattice["kind"] == "hcp") else "iso"
    L = header(ctx, state_file, "thermal expansion")[0:1]
    for k, T in enumerate(Ts):
        L += ([] if k == 0 else ["", "clear"]) + header(ctx, state_file, f"NPT at {T:g} K")[1:]
        L += [f"timestep {dt:.10g}", f"velocity all create {T:g} {ctx.seed + 77 * (k + 1) + index} mom yes rot yes dist gaussian",
              f"fix int all npt temp {T:g} {T:g} {100 * dt:.10g} {cpl} {P:.10g} {P:.10g} {1000 * dt:.10g}",
              f"thermo {every}", f"run {steps}"] + emit(f"T{k}", run0=False)
    return {"input": L, "meta": {"T": Ts, "steps": steps, "every": every, "average_from": 0.5, "dt_ps": dt_fs * 1e-3},
            "steps": steps * len(Ts)}
