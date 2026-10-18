"""format.py — the study format (what an LLM reads first) and static validation with fixable messages.
Version: 2026.10.18.2
"""
from __future__ import annotations

import math
from pathlib import Path

from .data import LATTICE_DOC, LATTICES, MASSES, MOLECULES

ANALYTIC_STYLES = ("lj", "morse", "buck", "coul")
FILE_STYLES = ("eam", "eam/alloy", "eam/fs", "adp", "tersoff", "tersoff/mod", "tersoff/zbl", "sw", "meam",
               "airebo", "rebo", "edip", "vashishta", "snap")
STAGE_KINDS = ("energy", "minimize", "nve", "nvt", "npt", "elastic", "eos", "vacancy", "surface",
               "thermal_expansion")
MD_KINDS = ("nve", "nvt", "npt")

STAGES = {
    "energy": {"what": "single point: energy, cohesive energy, stress, largest force", "params": {}},
    "minimize": {"what": "relax atoms (and the box); changes the state",
                 "params": {"box": "fixed (default) | iso | aniso | tri", "P_bar": "target pressure with box (0)",
                            "ftol": "force tolerance, eV/A or kcal/mol/A (1e-8)", "maxiter": "per cycle (5000)"}},
    "nve": {"what": "MD at constant energy; changes the state", "params": {}},
    "nvt": {"what": "MD at constant temperature; changes the state", "params": {}},
    "npt": {"what": "MD at constant temperature and pressure; changes the state",
            "params": {"P_bar": "target pressure (0 for solids; 1.01325 = 1 atm)",
                       "coupling": "iso (default; liquids, cubic crystals) | aniso | tri",
                       "pdamp_fs": "barostat time (1000 dt)"}},
    "elastic": {"what": "elastic constants Cij (GPa) by +/- strain of the current cell, atoms relaxed at each "
                        "strain; VRH moduli, Born stability, feat_material. Does not change the state",
                "params": {"strain": "1e-3", "relax": "true: relax atoms at each strain (needed for "
                                                         "diamond, hcp, compounds)"}},
    "eos": {"what": "energy-volume curve by isotropic scaling, Birch-Murnaghan fit (V0, E0, B0, B0'). "
                    "Does not change the state",
            "params": {"strain": "largest linear strain (0.04)", "n": "points (9)", "relax": "true"}},
    "vacancy": {"what": "vacancy formation energy at constant volume (atoms relaxed). Does not change the state",
                "params": {"element": "which element to remove (the first)", "relax": "true"}},
    "surface": {"what": "surface energy (J/m2) of a lattice system: oriented slab of the current lattice "
                        "constant, minus the same cell periodic. Does not change the state",
                "params": {"miller": "[h, k, l] (cubic lattices; hcp: [0, 0, 1] only)",
                           "layers": "repeats of the oriented cell along the normal (auto: >= 20 A)",
                           "vacuum": "A (15)", "relax": "true"}},
    "thermal_expansion": {"what": "NPT at each temperature from the current state: lattice constant a(T), "
                                  "linear and volume expansion coefficients. Does not change the state",
                          "params": {"T": "[K, ...] (>= 2)", "steps": "per temperature (20000)",
                                     "P_bar": "0"}},
}
MD_PARAMS = {
    "T": "K (nvt, npt; for nve: initial velocities)", "T_end": "K: linear ramp from T",
    "steps": "number of steps", "dt_fs": "time step, fs (1; 2 with rigid water)",
    "thermostat": "nose_hoover (default) | langevin", "tdamp_fs": "thermostat time (100 dt)",
    "velocities": "create (default when the state has none) | keep", "seed": "velocity/thermostat seed",
    "average_from": "fraction of the run discarded before averaging (0.5)",
    "thermo_every": "steps between thermo samples (steps/200)",
    "rdf": "{pairs: [[A, B], ...] (all), r_max: A (cutoff), bins: 200}",
    "msd": "{of: [types or elements] (all)}: mean-squared displacement, diffusion coefficient",
    "dump_every": "steps between trajectory frames (none)",
}


def study_format() -> dict:
    return {
        "format": "LAMMPS study (lammps-translator)",
        "conventions": [
            "One study = one system + its interactions + an ordered list of stages. Stages that change the state "
            "(minimize, nve, nvt, npt) pass their final structure and velocities to the next stage; property stages "
            "(energy, elastic, eos, vacancy, surface, thermal_expansion) measure the current state and leave it as is.",
            "Atomic systems (lattice or atomic data) use 'potential' with LAMMPS units metal: eV, A, ps; charges in e. "
            "Molecular systems use 'forcefield' with units real: kcal/mol, A, e (as OPLS, TraPPE and water models are "
            "published). Every study gives T in K, P_bar in bar, dt_fs in fs.",
            "Results are in eV, A, GPa, K, ps, g/cm3, J/m2 (molecular systems also kJ/mol per molecule) whatever the "
            "units; key names carry the unit (pe_per_atom_eV, C11_GPa, D_m2_s).",
            "LAMMPS harmonic bonds and angles have no 1/2: E = K (r - r0)^2, E = K (theta - theta0)^2.",
            "A file potential is a name in the server's potentials folders (see status.potentials) or a path on the "
            "server. Elements in the system map to the file's elements by name.",
            "Results: per stage under its name (default: its kind; repeated kinds get _2, _3); 'structure' (initial), "
            "'final', 'feat_material' (fea-translator material from elastic, eos/thermal_expansion and density), "
            "'checks' (reference energy/forces/pressure from an independent implementation, energy drift, Born "
            "stability, ...), 'reliable' (false: do not report the numbers; see unreliable_because), warnings, files.",
            "expect: [{quantity, value, tol | abs}] with quantity a dotted path into the results (elastic.C11_GPa, "
            "minimize.a_A, nvt.mean.T_K); tol relative (0.02).",
            "MD averages are over the last (1 - average_from) of the run with block-average errors: <stage>.mean.<key>, "
            "<stage>.err.<key>. Statistical results need long enough runs: check err.",
        ],
        "study_keys": {
            "system": "one of {lattice: kind, ...}, {molecules: [...], ...}, {data: ...} (below)",
            "potential": "atomic systems (below)",
            "forcefield": "molecular systems (optional when every molecule is built in)",
            "stages": "[{kind, name?, ...}] in order (below)",
            "procs": "MPI ranks (default LMPT_PROCS or 1; SLURM --ntasks)", "seed": "random seed (12345)",
            "expect": "checks", "queue": "with LMPT_EXECUTOR=slurm: true always a SLURM job, false always local "
                                         "(default: by the time estimate)",
        },
        "system": {
            "lattice": {"kind": {**LATTICE_DOC},
                        "keys": {"a": "lattice constant, A", "c": "hcp only, A", "elements": "[A, B] by lattice site",
                                 "size": "[nx, ny, nz] repeats (auto: each side >= 2 cutoffs and >= 3 cells)",
                                 "orient": "{x: [h,k,l], y: [...], z: [...]} orthogonal integer directions (cubic)",
                                 "substitute": "[{from: El, to: El2, fraction: 0.25}]: random solid solution (seed)",
                                 "perturb": "random displacement amplitude, A (0)"}},
            "molecules": {"keys": {"molecules": "[{molecule: name | {atoms: [[label, x, y, z], ...], bonds: [[i, j], ...] "
                                                "(0-based), angles?: [[i,j,k]], dihedrals?: [[i,j,k,l]]}, count: n}]",
                                   "density_g_cm3": "packing density (or box)", "box": "[lx, ly, lz] A"},
                          "built_in": {k: v["doc"] for k, v in MOLECULES.items()},
                          "notes": "Molecules go on a cubic grid with random orientations (seeded). Angles and "
                                   "dihedrals default to all those of the bond graph. Minimize first."},
            "data": {"keys": {"data": "LAMMPS data file text (or a server path)",
                              "atom_style": "atomic | charge | full", "type_names": "[element or ff label per type]"},
                     "notes": "full: bond/angle/dihedral types are re-derived from the atom labels and forcefield"},
        },
        "potential": {
            "style": list(ANALYTIC_STYLES + FILE_STYLES),
            "analytic": {"lj": "pairs {'A-B': {eps, sigma}} (eV, A); missing cross pairs mixed geometrically; single "
                               "element: {eps, sigma} directly",
                         "morse": "pairs {'A-B': {D0, alpha, r0}} (eV, 1/A, A)",
                         "buck": "pairs {'A-B': {A, rho, C}} (eV, A, eV A^6); missing pairs: no short-range term",
                         "coul": "charges only (Madelung)",
                         "common": "cutoff A (lj 2.5 sigma); shift: true shifts the pair energy to 0 at the cutoff; "
                                   "tail: true adds long-range LJ/C6 corrections; charges {El: q}; coulomb: none | cut | "
                                   "long (Ewald/PPPM, kspace_accuracy 1e-8; needs a neutral cell)"},
            "file": {"file": "potential file (eam: one per element: files {El: file})",
                     "files": "meam: [library, parameters]; snap: [coeff, param]",
                     "args": "extra pair_style numbers (airebo: '3.0 1 1')"},
        },
        "forcefield": {
            "types": "{label: {mass, q, eps, sigma}} kcal/mol, A",
            "pairs": "{'A-B': {eps, sigma}} explicit cross terms (else mixing)",
            "mixing": "geometric (OPLS, default) | arithmetic (TraPPE, CHARMM, Amber)",
            "bonds": "{'A-B': {k, r0}}", "angles": "{'A-B-C': {k, theta0 (deg)}}",
            "dihedrals": "{'A-B-C-D': {k: [K1, K2, K3, K4]}} (OPLS form)",
            "rigid": "bond/angle labels held by SHAKE (built-in waters: theirs)",
            "cutoff": "A (10)", "coulomb": "long (default) | cut | none", "kspace_accuracy": "1e-5",
            "tail": "true (default)", "special": "[lj/coul 1-2, 1-3, 1-4] ([0, 0, 0.5] OPLS; default [0, 0, 0])",
        },
        "stages": {k: v for k, v in STAGES.items()},
        "md_params": MD_PARAMS,
        "example": {
            "cu_elastic": {
                "system": {"lattice": "fcc", "a": 3.615, "elements": ["Cu"], "size": [4, 4, 4]},
                "potential": {"style": "eam", "file": "Cu_u3.eam"},
                "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic"}, {"kind": "vacancy"},
                           {"kind": "surface", "miller": [1, 1, 1]}],
                "expect": [{"quantity": "minimize.a_A", "value": 3.615, "tol": 0.002}]},
            "water_300K": {
                "system": {"molecules": [{"molecule": "spce", "count": 216}], "density_g_cm3": 0.997},
                "stages": [{"kind": "minimize"}, {"kind": "npt", "T": 300, "P_bar": 1.01325, "steps": 20000,
                                                   "dt_fs": 2},
                           {"kind": "nvt", "T": 300, "steps": 20000, "dt_fs": 2, "rdf": {"pairs": [["OW", "OW"]]},
                            "msd": {"of": ["OW"]}}]},
        },
    }


# ----------------------------------------------------------------------------------------------- validation
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pos(x) -> bool:
    return _num(x) and x > 0


def _vec3i(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(t, int) and not isinstance(t, bool) for t in v)


def molecular(study: dict) -> bool:
    s = study.get("system", {}) if isinstance(study.get("system"), dict) else {}
    return "molecules" in s or ("data" in s and s.get("atom_style") == "full")


def system_elements(system: dict) -> list[str]:
    """Element / label names used by the system (as far as they can be known statically)."""
    if "lattice" in system:
        els = list(system.get("elements") or [])
        for sub in system.get("substitute") or []:
            if isinstance(sub, dict) and sub.get("to") not in els:
                els.append(sub.get("to"))
        return els
    if "data" in system:
        return list(system.get("type_names") or [])
    out = []
    for m in system.get("molecules") or []:
        mol = m.get("molecule") if isinstance(m, dict) else None
        atoms = MOLECULES[mol]["atoms"] if isinstance(mol, str) and mol in MOLECULES else (
            mol.get("atoms", []) if isinstance(mol, dict) else [])
        for a in atoms:
            if isinstance(a, (list, tuple)) and a and a[0] not in out:
                out.append(a[0])
    return out


def _check_system(s, E, W):
    if not isinstance(s, dict):
        E.append("system: an object with lattice, molecules or data"); return
    kinds = [k for k in ("lattice", "molecules", "data") if k in s]
    if len(kinds) != 1:
        E.append(f"system: give exactly one of lattice, molecules, data (got {kinds or 'none'})"); return
    if "lattice" in s:
        kind = s["lattice"]
        if kind != "custom" and kind not in LATTICES:
            E.append(f"system.lattice: '{kind}' is not one of {sorted(LATTICES) + ['custom']}"); return
        els = s.get("elements")
        if kind == "custom":
            cell, basis = s.get("cell"), s.get("basis")
            if not (isinstance(cell, list) and len(cell) == 3 and all(_pos(c) for c in cell)):
                E.append("system.cell: [ax, ay, az] in A (orthorhombic) for a custom lattice")
            if not (isinstance(basis, list) and basis and all(isinstance(b, list) and len(b) == 4 for b in basis)):
                E.append("system.basis: [[site, fx, fy, fz], ...] with site an index into elements")
            nsites = 1 + max((int(b[0]) for b in basis or [] if isinstance(b, list) and b and isinstance(b[0], int)),
                             default=0)
        else:
            nsites = LATTICES[kind]["sites"]
            if not _pos(s.get("a")):
                E.append("system.a: the lattice constant in A (e.g. 3.615 for Cu)")
            if kind == "hcp" and "c" in s and not _pos(s["c"]):
                E.append("system.c: hcp c in A")
        if not (isinstance(els, list) and len(els) == nsites and all(isinstance(e, str) for e in els)):
            E.append(f"system.elements: {nsites} name(s) for lattice '{kind}' ({LATTICE_DOC.get(kind, '')})")
        size = s.get("size")
        if size is not None and not (_vec3i(size) and all(n >= 1 for n in size)):
            E.append("system.size: [nx, ny, nz] positive integers (or omit for auto)")
        if "orient" in s:
            o = s["orient"]
            if kind not in LATTICES or not LATTICES[kind]["cubic"]:
                E.append("system.orient: only for cubic lattices")
            elif not (isinstance(o, dict) and all(_vec3i(o.get(k)) for k in "xyz")):
                E.append("system.orient: {x: [h,k,l], y: [h,k,l], z: [h,k,l]} integer directions")
            else:
                x, y, z = o["x"], o["y"], o["z"]
                d = lambda u, v: sum(a * b for a, b in zip(u, v))
                if d(x, y) or d(y, z) or d(x, z):
                    E.append("system.orient: x, y, z must be mutually orthogonal")
                cr = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
                if d(cr, z) <= 0:
                    E.append("system.orient: x, y, z must be right-handed (x cross y along z)")
        for i, sub in enumerate(s.get("substitute") or []):
            if not (isinstance(sub, dict) and isinstance(sub.get("from"), str) and isinstance(sub.get("to"), str)
                    and _num(sub.get("fraction")) and 0 < sub["fraction"] <= 1):
                E.append(f"system.substitute[{i}]: {{from: El, to: El2, fraction: 0..1}}")
            elif isinstance(els, list) and sub["from"] not in els:
                E.append(f"system.substitute[{i}].from: '{sub['from']}' is not in elements {els}")
        if "perturb" in s and not (_num(s["perturb"]) and s["perturb"] >= 0):
            E.append("system.perturb: displacement amplitude in A (>= 0)")
    elif "molecules" in s:
        ms = s["molecules"]
        if not (isinstance(ms, list) and ms):
            E.append("system.molecules: [{molecule: name or {...}, count: n}, ...]"); return
        for i, m in enumerate(ms):
            if not isinstance(m, dict):
                E.append(f"system.molecules[{i}]: {{molecule, count}}"); continue
            if not (isinstance(m.get("count"), int) and m["count"] >= 1):
                E.append(f"system.molecules[{i}].count: a positive integer")
            mol = m.get("molecule")
            if isinstance(mol, str):
                if mol not in MOLECULES:
                    E.append(f"system.molecules[{i}].molecule: '{mol}' is not built in ({sorted(MOLECULES)}); "
                             "or give {atoms, bonds}")
            elif isinstance(mol, dict):
                atoms = mol.get("atoms")
                if not (isinstance(atoms, list) and atoms and all(
                        isinstance(a, list) and len(a) == 4 and isinstance(a[0], str) and all(_num(t) for t in a[1:])
                        for a in atoms)):
                    E.append(f"system.molecules[{i}].molecule.atoms: [[label, x, y, z], ...] in A")
                    continue
                n = len(atoms)
                for key, k in (("bonds", 2), ("angles", 3), ("dihedrals", 4)):
                    for j, b in enumerate(mol.get(key) or []):
                        if not (isinstance(b, list) and len(b) == k and all(isinstance(t, int) and 0 <= t < n for t in b)):
                            E.append(f"system.molecules[{i}].molecule.{key}[{j}]: {k} atom indices 0..{n - 1}")
            else:
                E.append(f"system.molecules[{i}].molecule: a built-in name or {{atoms, bonds}}")
        if ("density_g_cm3" in s) == ("box" in s):
            E.append("system: give density_g_cm3 or box (one of them) for molecules")
        if "density_g_cm3" in s and not _pos(s["density_g_cm3"]):
            E.append("system.density_g_cm3: positive")
        if "box" in s and not (isinstance(s["box"], list) and len(s["box"]) == 3 and all(_pos(t) for t in s["box"])):
            E.append("system.box: [lx, ly, lz] in A")
    else:
        if not isinstance(s["data"], str) or not s["data"].strip():
            E.append("system.data: the data file text or a path")
        if s.get("atom_style", "atomic") not in ("atomic", "charge", "full"):
            E.append("system.atom_style: atomic | charge | full")
        tn = s.get("type_names")
        if not (isinstance(tn, list) and tn and all(isinstance(t, str) for t in tn)):
            E.append("system.type_names: [name per atom type, in type order]")


def _pair_key(k, names):
    parts = k.split("-") if isinstance(k, str) else []
    return len(parts) == 2 and all(p in names for p in parts)


def _check_potential(p, els, E, W):
    if not isinstance(p, dict) or "style" not in p:
        E.append("potential: {style, ...}; styles " + ", ".join(ANALYTIC_STYLES + FILE_STYLES)); return
    st = p["style"]
    if st not in ANALYTIC_STYLES + FILE_STYLES:
        E.append(f"potential.style: '{st}' is not one of {list(ANALYTIC_STYLES + FILE_STYLES)}"); return
    for e in els:
        if st not in ANALYTIC_STYLES and e not in MASSES:
            E.append(f"system: '{e}' is not an element symbol (file potentials need element names)")
    if st in ANALYTIC_STYLES:
        pairs = p.get("pairs")
        if st == "lj" and pairs is None and len(els) == 1 and _pos(p.get("eps", -1) if "eps" in p else -1):
            pairs = {f"{els[0]}-{els[0]}": {"eps": p["eps"], "sigma": p.get("sigma")}}
        need = {"lj": ("eps", "sigma"), "morse": ("D0", "alpha", "r0"), "buck": ("A", "rho", "C")}.get(st, ())
        if st != "coul":
            if not isinstance(pairs, dict) or not pairs:
                E.append(f"potential.pairs: {{'A-B': {{{', '.join(need)}}}}} for style {st}")
            else:
                for k, v in pairs.items():
                    if not _pair_key(k, els):
                        E.append(f"potential.pairs: key '{k}' must be 'A-B' with A, B in {els}")
                    elif not (isinstance(v, dict) and all(_num(v.get(n)) for n in need)):
                        E.append(f"potential.pairs['{k}']: needs {list(need)} (numbers)")
                if st == "morse":
                    have = {tuple(sorted(k.split("-"))) for k in pairs if _pair_key(k, els)}
                    miss = [f"{a}-{b}" for i, a in enumerate(els) for b in els[i:] if tuple(sorted((a, b))) not in have]
                    if miss:
                        W.append(f"potential.pairs: no morse term for {miss} (they do not interact short-range)")
        if st in ("buck", "coul") and "cutoff" not in p:
            E.append(f"potential.cutoff: needed for {st} (A, e.g. 10)")
        if "cutoff" in p and not _pos(p["cutoff"]):
            E.append("potential.cutoff: positive, A")
        q = p.get("charges")
        coul = p.get("coulomb", "long" if st == "coul" else ("none" if not q else "long"))
        if coul not in ("none", "cut", "long"):
            E.append("potential.coulomb: none | cut | long")
        if st == "coul" and coul == "none":
            E.append("potential.coulomb: style coul needs cut or long")
        if q is not None or coul != "none":
            if not (isinstance(q, dict) and all(k in els and _num(v) for k, v in q.items())):
                E.append(f"potential.charges: {{element: charge in e}} for {els}")
            if st == "morse" and coul != "none":
                E.append("potential: morse with charges is not supported (use buck or lj)")
        if "kspace_accuracy" in p and not (_pos(p["kspace_accuracy"]) and p["kspace_accuracy"] < 1e-2):
            E.append("potential.kspace_accuracy: e.g. 1e-6")
    else:
        if st == "eam":
            f = p.get("files") or ({els[0]: p["file"]} if "file" in p and len(els) == 1 else None)
            if not (isinstance(f, dict) and set(f) == set(els)):
                E.append(f"potential: eam (funcfl) needs file (one element) or files {{El: file}} for {els}")
        elif st in ("meam", "snap"):
            f = p.get("files")
            if not (isinstance(f, list) and len(f) == 2 and all(isinstance(t, str) for t in f)):
                E.append(f"potential.files: [{'library, parameters' if st == 'meam' else 'coeff, param'}] for {st}")
        elif not isinstance(p.get("file"), str):
            E.append(f"potential.file: the {st} file name (e.g. Si.tersoff) or a server path")
        if "args" in p and not (isinstance(p["args"], str) and all(_isnum(t) for t in p["args"].split())):
            E.append("potential.args: numbers separated by spaces")


def _isnum(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def _check_forcefield(ff, study, E, W):
    s = study["system"]
    labels = system_elements(s)
    built = {}
    for m in s.get("molecules") or []:
        if isinstance(m, dict) and isinstance(m.get("molecule"), str) and m["molecule"] in MOLECULES:
            built.update(MOLECULES[m["molecule"]]["forcefield"]["types"])
    if ff is None:
        ff = {}
    if not isinstance(ff, dict):
        E.append("forcefield: an object"); return
    types = {**built, **(ff.get("types") or {})}
    for lab in labels:
        t = types.get(lab)
        if not (isinstance(t, dict) and all(_num(t.get(k)) for k in ("mass", "eps", "sigma"))):
            E.append(f"forcefield.types['{lab}']: {{mass, q, eps, sigma}} (kcal/mol, A)")
        elif t["mass"] <= 0:
            E.append(f"forcefield.types['{lab}'].mass: positive")
    if ff.get("mixing", "geometric") not in ("geometric", "arithmetic"):
        E.append("forcefield.mixing: geometric | arithmetic")
    if ff.get("coulomb", "long") not in ("none", "cut", "long"):
        E.append("forcefield.coulomb: none | cut | long")
    for k, need in (("bonds", ("k", "r0")), ("angles", ("k", "theta0"))):
        for lab, v in (ff.get(k) or {}).items():
            if not (isinstance(v, dict) and all(_num(v.get(n)) for n in need)):
                E.append(f"forcefield.{k}['{lab}']: needs {list(need)}")
    for lab, v in (ff.get("dihedrals") or {}).items():
        if not (isinstance(v, dict) and isinstance(v.get("k"), list) and len(v["k"]) == 4 and all(_num(t) for t in v["k"])):
            E.append(f"forcefield.dihedrals['{lab}']: {{k: [K1, K2, K3, K4]}} kcal/mol")
    sp = ff.get("special")
    if sp is not None and not (isinstance(sp, list) and len(sp) == 3 and all(_num(t) and 0 <= t <= 1 for t in sp)):
        E.append("forcefield.special: [f12, f13, f14] between 0 and 1")
    if "cutoff" in ff and not _pos(ff["cutoff"]):
        E.append("forcefield.cutoff: positive, A")


def _check_stage(i, st, study, E, W, names):
    where = f"stages[{i}]"
    if not isinstance(st, dict) or st.get("kind") not in STAGE_KINDS:
        E.append(f"{where}.kind: one of {list(STAGE_KINDS)}"); return
    k = st["kind"]
    allowed = set(STAGES[k]["params"]) | {"kind", "name"}
    if k in MD_KINDS:
        allowed |= set(MD_PARAMS) | {"P_bar", "coupling", "pdamp_fs"} if k == "npt" else set(MD_PARAMS)
    unknown = sorted(set(st) - allowed)
    if unknown:
        E.append(f"{where} ({k}): unknown keys {unknown}; allowed {sorted(allowed - {'kind'})}")
    name = st.get("name")
    if name is not None and not (isinstance(name, str) and name.replace("_", "").isalnum()):
        E.append(f"{where}.name: letters, digits and _ only")
    is_lattice = "lattice" in study.get("system", {})
    mol = molecular(study)
    if k == "minimize":
        if st.get("box", "fixed") not in ("fixed", "iso", "aniso", "tri"):
            E.append(f"{where}.box: fixed | iso | aniso | tri")
        for key in ("ftol",):
            if key in st and not _pos(st[key]):
                E.append(f"{where}.{key}: positive")
        if "maxiter" in st and not (isinstance(st["maxiter"], int) and st["maxiter"] > 0):
            E.append(f"{where}.maxiter: positive integer")
    if k in MD_KINDS:
        if not (isinstance(st.get("steps"), int) and st["steps"] > 0):
            E.append(f"{where}.steps: a positive integer number of MD steps")
        if k in ("nvt", "npt") and not _pos(st.get("T")):
            E.append(f"{where}.T: temperature in K (> 0)")
        if "T" in st and not (_num(st["T"]) and st["T"] >= 0):
            E.append(f"{where}.T: K >= 0")
        if "T_end" in st and not _pos(st["T_end"]):
            E.append(f"{where}.T_end: K > 0")
        if "dt_fs" in st:
            if not _pos(st["dt_fs"]):
                E.append(f"{where}.dt_fs: positive, fs")
            elif st["dt_fs"] > 5:
                E.append(f"{where}.dt_fs: {st['dt_fs']} fs is too large (<= 5; 1-2 typical)")
            else:
                rigid = bool((study.get("forcefield") or {}).get("rigid")) or any(
                    isinstance(m, dict) and m.get("molecule") in ("spce", "tip3p")
                    for m in study.get("system", {}).get("molecules") or [])
                lim = 3.0 if not mol else (2.0 if rigid else 1.0)
                if st["dt_fs"] > lim + 1e-9:
                    W.append(f"{where}.dt_fs: {st['dt_fs']} fs is large for this system (<= {lim:g} fs usual); "
                             "check the energy drift with an nve stage")
        if st.get("thermostat", "nose_hoover") not in ("nose_hoover", "langevin"):
            E.append(f"{where}.thermostat: nose_hoover | langevin")
        if k == "npt":
            if st.get("coupling", "iso") not in ("iso", "aniso", "tri"):
                E.append(f"{where}.coupling: iso | aniso | tri")
            if "P_bar" in st and not _num(st["P_bar"]):
                E.append(f"{where}.P_bar: a number (bar)")
        if st.get("velocities", "create") not in ("create", "keep"):
            E.append(f"{where}.velocities: create | keep")
        af = st.get("average_from", 0.5)
        if not (_num(af) and 0 <= af < 1):
            E.append(f"{where}.average_from: 0 <= fraction < 1")
        for key in ("thermo_every", "dump_every", "seed"):
            if key in st and not (isinstance(st[key], int) and st[key] > 0):
                E.append(f"{where}.{key}: positive integer")
        if "rdf" in st and not isinstance(st["rdf"], dict):
            E.append(f"{where}.rdf: {{pairs?, r_max?, bins?}}")
        if "msd" in st and not isinstance(st["msd"], dict):
            E.append(f"{where}.msd: {{of?: [types]}}")
        labels = system_elements(study.get("system", {}))
        for pr in (st.get("rdf") or {}).get("pairs") or []:
            if not (isinstance(pr, list) and len(pr) == 2 and all(p in labels for p in pr)):
                E.append(f"{where}.rdf.pairs: [[A, B], ...] with names in {labels}")
        for t in (st.get("msd") or {}).get("of") or []:
            if t not in labels:
                E.append(f"{where}.msd.of: '{t}' is not in {labels}")
    if k in ("elastic", "eos"):
        if "strain" in st and not (_pos(st["strain"]) and st["strain"] <= 0.2):
            E.append(f"{where}.strain: 0 < strain <= 0.2")
        if k == "eos" and "n" in st and not (isinstance(st["n"], int) and st["n"] >= 5):
            E.append(f"{where}.n: >= 5 points")
        if mol:
            W.append(f"{where}: {k} of a molecular system is a static (0 K) property of the current structure")
    if k == "vacancy" and mol:
        E.append(f"{where}: vacancy needs an atomic system")
    if k == "surface":
        if not is_lattice:
            E.append(f"{where}: surface needs a lattice system")
        mil = st.get("miller")
        if not (_vec3i(mil) and any(mil)):
            E.append(f"{where}.miller: [h, k, l] integers, not all zero")
        else:
            kind = study["system"].get("lattice")
            if kind == "hcp" and [abs(t) for t in mil] != [0, 0, 1]:
                E.append(f"{where}.miller: hcp supports [0, 0, 1] only")
            if kind == "custom" and [abs(t) for t in mil] not in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
                E.append(f"{where}.miller: custom lattices support [1,0,0], [0,1,0], [0,0,1] only")
        if "vacuum" in st and not (_num(st["vacuum"]) and st["vacuum"] >= 5):
            E.append(f"{where}.vacuum: >= 5 A")
        if "layers" in st and not (isinstance(st["layers"], int) and st["layers"] >= 2):
            E.append(f"{where}.layers: integer >= 2")
    if k == "thermal_expansion":
        T = st.get("T")
        if not (isinstance(T, list) and len(T) >= 2 and all(_pos(t) for t in T)):
            E.append(f"{where}.T: [K, K, ...] at least two temperatures")
        if "steps" in st and not (isinstance(st["steps"], int) and st["steps"] >= 1000):
            E.append(f"{where}.steps: >= 1000")
        if not is_lattice:
            W.append(f"{where}: no lattice: a(T) is reported as V^(1/3) scaling of the initial cell")
    for key in ("relax",):
        if key in st and not isinstance(st[key], bool):
            E.append(f"{where}.{key}: true | false")


def validate_study(study) -> dict:
    E: list[str] = []
    W: list[str] = []
    if not isinstance(study, dict):
        return {"ok": False, "errors": ["a study is an object {system, potential|forcefield, stages}"], "warnings": []}
    unknown = sorted(set(study) - {"system", "potential", "forcefield", "stages", "procs", "seed", "expect",
                                   "description", "queue"})
    if unknown:
        E.append(f"study: unknown keys {unknown}")
    _check_system(study.get("system"), E, W)
    if E:
        return {"ok": False, "errors": E, "warnings": W}
    s = study["system"]
    mol = molecular(study)
    if mol:
        if "potential" in study:
            E.append("potential: molecular systems use 'forcefield' (units real), not 'potential'")
        _check_forcefield(study.get("forcefield"), study, E, W)
    else:
        if "forcefield" in study:
            E.append("forcefield: atomic systems use 'potential' (units metal)")
        _check_potential(study.get("potential"), system_elements(s), E, W)
    stages = study.get("stages")
    if not (isinstance(stages, list) and stages):
        E.append("stages: a non-empty list, e.g. [{kind: minimize, box: iso}, {kind: elastic}]")
    else:
        names = set()
        for i, st in enumerate(stages):
            _check_stage(i, st, study, E, W, names)
        if mol and isinstance(stages[0], dict) and stages[0].get("kind") in MD_KINDS and "molecules" in s:
            W.append("stages[0]: packed molecules may overlap; start with {kind: minimize}")
    tail = (study.get("forcefield") or {}).get("tail", True) if mol else (study.get("potential") or {}).get("tail", False)
    if tail and isinstance(stages, list):
        for i, st in enumerate(stages):
            if isinstance(st, dict) and st.get("kind") == "minimize" and st.get("box", "fixed") != "fixed":
                E.append(f"stages[{i}]: box relaxation with tail corrections stalls (LAMMPS's tail pressure is not "
                         "-dE/dV of its tail energy, so the minimizer's line search fails): set tail false, or use npt")
        if "lattice" in s:
            W.append("tail: corrections assume a uniform fluid beyond the cutoff; for crystals use a longer cutoff instead")
    if "queue" in study and not isinstance(study["queue"], bool):
        E.append("queue: true (always a SLURM job) or false (always local); omit to let the time estimate decide")
    if "procs" in study and not (isinstance(study["procs"], int) and study["procs"] >= 1):
        E.append("procs: MPI ranks, integer >= 1")
    if "seed" in study and not (isinstance(study["seed"], int) and 0 < study["seed"] < 900000000):
        E.append("seed: integer 1..9e8")
    for i, e in enumerate(study.get("expect") or []):
        if not (isinstance(e, dict) and isinstance(e.get("quantity"), str) and _num(e.get("value"))):
            E.append(f"expect[{i}]: {{quantity: dotted path, value, tol | abs}}")
    if not E:
        # deeper checks that need the builder and the potential files (on this machine)
        try:
            from .build import build_system
            from .potential import resolve
            sysobj = build_system(study)
            r = resolve(study, sysobj)
            W.extend(r.get("warnings", []))
            E.extend(r.get("errors", []))
            W.extend(sysobj.warnings)
        except ValueError as exc:
            E.append(str(exc))
    return {"ok": not E, "errors": E, "warnings": W}


def validate(studies) -> dict:
    if not isinstance(studies, dict) or not studies:
        return {"ok": False, "studies": {}, "errors": ["give {name: study, ...}"]}
    out = {}
    for n, st in studies.items():
        if not (isinstance(n, str) and n.replace("_", "").replace("-", "").isalnum()):
            out[str(n)] = {"ok": False, "errors": ["study name: letters, digits, _ and - only"], "warnings": []}
            continue
        out[n] = validate_study(st)
    return {"ok": all(v["ok"] for v in out.values()), "studies": out}


def is_path(text: str) -> bool:
    return "\n" not in text and len(text) < 1024 and Path(text).expanduser().is_file()
