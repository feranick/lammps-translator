"""build.py — builds the atoms of a study (lattices, packed molecules, data files), writes and reads LAMMPS data files.
Version: 2026.10.20.2

Atom positions, types, charges and topology are computed here, not by LAMMPS (lattice, create_atoms), so the
translator knows every atom and the reference implementation can check LAMMPS's numbers.
Boxes are LAMMPS restricted triclinic: a = (lx, 0, 0), b = (xy, ly, 0), c = (xz, yz, lz), origin 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

import numpy as np

from .data import AMU_A3_TO_G_CM3, AVOGADRO, LATTICES, MASSES, MOLECULES, lattice_cell


@dataclass
class System:
    box: list            # [lx, ly, lz, xy, xz, yz]
    pos: np.ndarray      # (N, 3)
    types: np.ndarray    # (N,) 1-based
    type_names: list     # name per type
    masses: list         # per type
    atom_style: str = "atomic"   # atomic | charge | full
    charges: np.ndarray | None = None
    mol: np.ndarray | None = None
    bonds: list = field(default_factory=list)       # [(type, i, j)] 0-based atoms, 1-based types
    angles: list = field(default_factory=list)
    dihedrals: list = field(default_factory=list)
    bond_types: list = field(default_factory=list)  # labels per type
    angle_types: list = field(default_factory=list)
    dihedral_types: list = field(default_factory=list)
    velocities: np.ndarray | None = None
    lattice: dict | None = None   # kind, a, c, cells, orient, elements, cell (A)
    warnings: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.pos)

    def cell(self) -> np.ndarray:
        lx, ly, lz, xy, xz, yz = self.box
        return np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]], float)   # rows a, b, c

    def volume(self) -> float:
        return self.box[0] * self.box[1] * self.box[2]

    def triclinic(self) -> bool:
        return any(abs(t) > 0 for t in self.box[3:])

    def names(self) -> list:
        return [self.type_names[t - 1] for t in self.types]

    def total_mass(self) -> float:
        return float(sum(self.masses[t - 1] for t in self.types))

    def density(self) -> float:
        return self.total_mass() / self.volume() * AMU_A3_TO_G_CM3

    def composition(self) -> dict:
        out = {}
        for nm in self.names():
            out[nm] = out.get(nm, 0) + 1
        return out

    def nmolecules(self) -> int:
        return int(len(set(self.mol.tolist()))) if self.mol is not None else 0

    def copy(self) -> "System":
        import copy
        return copy.deepcopy(self)


# ------------------------------------------------------------------------------------------------ lattices
def _basis(system: dict):
    kind = system["lattice"]
    if kind == "custom":
        cell = [float(c) for c in system["cell"]]
        basis = [(int(b[0]), float(b[1]) % 1.0, float(b[2]) % 1.0, float(b[3]) % 1.0) for b in system["basis"]]
        return cell, basis
    a = float(system["a"])
    return lattice_cell(kind, a, system.get("c")), list(LATTICES[kind]["basis"])


def _is_translation(t_frac: np.ndarray, basis_frac: np.ndarray, sites: np.ndarray) -> bool:
    """t (in conventional-cell fractions) maps every basis atom onto an atom of the same site."""
    for p, s in zip(basis_frac, sites):
        q = (p + t_frac) % 1.0
        d = np.abs(((basis_frac - q) + 0.5) % 1.0 - 0.5).max(axis=1)
        if not np.any((d < 1e-8) & (sites == s)):
            return False
    return True


def oriented_cell(system: dict, orient: dict | None):
    """The smallest orthogonal periodic cell of a cubic lattice with x, y, z along the orient directions.
    Returns (lengths [Lx, Ly, Lz], positions (n, 3) in that frame, sites (n,))."""
    cell, basis = _basis(system)
    sites = np.array([b[0] for b in basis])
    bfrac = np.array([b[1:] for b in basis], float)
    if not orient:
        pos = bfrac * np.array(cell)
        return list(cell), pos, sites
    a = cell[0]
    vecs = []
    for k in "xyz":
        v = np.array(orient[k], float)
        g = reduce(math.gcd, [abs(int(t)) for t in orient[k]])
        v = v / g
        vecs.append(v / 2.0 if _is_translation((v / 2.0) % 1.0, bfrac, sites) else v)
    T = np.array(vecs)                      # rows: periodic vectors in units of a
    L = np.linalg.norm(T, axis=1) * a
    R = T / np.linalg.norm(T, axis=1)[:, None]   # rotation: crystal -> box frame
    nexp = abs(np.linalg.det(T)) * len(basis)
    nexp_i = int(round(nexp))
    if abs(nexp - nexp_i) > 1e-6:
        raise ValueError("system.orient: these directions do not give a periodic cell")
    # enumerate conventional cells covering the box
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ T
    lo = np.floor(corners.min(axis=0)) - 1
    hi = np.ceil(corners.max(axis=0)) + 1
    grid = np.array(np.meshgrid(*[np.arange(lo[d], hi[d] + 1) for d in range(3)], indexing="ij")).reshape(3, -1).T
    allp = (grid[:, None, :] + bfrac[None, :, :]).reshape(-1, 3)
    alls = np.tile(sites, len(grid))
    frac = allp @ np.linalg.inv(T)          # fractional in the new cell
    frac = np.round(frac, 10)
    keep = np.all((frac >= 0) & (frac < 1 - 1e-9), axis=1)
    frac, alls = frac[keep], alls[keep]
    if len(frac) != nexp_i:
        raise ValueError(f"system.orient: built {len(frac)} atoms per cell, expected {nexp_i}")
    pos = frac * L
    return list(L), pos, alls


def build_lattice(system: dict, seed: int, cutoff: float | None = None) -> System:
    L, pos, sites = oriented_cell(system, system.get("orient"))
    size = system.get("size")
    if size is None:
        rc = cutoff or 6.0
        size = [max(3, int(math.ceil(2.0 * rc / l - 1e-9))) for l in L]
        if system["lattice"] == "custom" or len(pos) * size[0] * size[1] * size[2] > 20000:
            size = [max(1, int(math.ceil(2.0 * rc / l - 1e-9))) for l in L]
    nx, ny, nz = size
    shifts = np.array([[i, j, k] for i in range(nx) for j in range(ny) for k in range(nz)], float) * np.array(L)
    P = (shifts[:, None, :] + pos[None, :, :]).reshape(-1, 3)
    S = np.tile(sites, len(shifts))
    els = list(system["elements"])
    names = [els[s] for s in S]
    rng = np.random.RandomState(seed)
    for sub in system.get("substitute") or []:
        idx = [i for i, nm in enumerate(names) if nm == sub["from"]]
        k = int(round(sub["fraction"] * len(idx)))
        for i in rng.permutation(idx)[:k]:
            names[i] = sub["to"]
        if sub["to"] not in els:
            els.append(sub["to"])
    tnames = [e for e in els if e in set(names)]
    types = np.array([tnames.index(nm) + 1 for nm in names])
    box = [L[0] * nx, L[1] * ny, L[2] * nz, 0.0, 0.0, 0.0]
    if system.get("perturb"):
        P = P + rng.uniform(-1, 1, P.shape) * float(system["perturb"])
        P = P % np.array(box[:3])
    masses = [MASSES.get(e, 1.0) for e in tnames]
    s = System(box=box, pos=P, types=types, type_names=tnames, masses=masses)
    for e in tnames:
        if e not in MASSES:
            s.warnings.append(f"'{e}' is not an element: mass 1")
    s.lattice = {"kind": system["lattice"], "a": float(system.get("a", 1.0)), "c": L[2] if system["lattice"] == "hcp" else None,
                 "size": list(size), "orient": system.get("orient"), "cell": L, "atoms_per_cell": len(pos)}
    return s


# ----------------------------------------------------------------------------------------------- molecules
def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def molecule_def(m) -> dict:
    if isinstance(m, str):
        d = MOLECULES[m]
        return {"name": m, "atoms": [list(a) for a in d["atoms"]], "bonds": [list(b) for b in d["bonds"]],
                "angles": None, "dihedrals": None}
    return {"name": m.get("name", "mol"), "atoms": m["atoms"], "bonds": m.get("bonds") or [],
            "angles": m.get("angles"), "dihedrals": m.get("dihedrals")}


def graph_angles(nat, bonds):
    nb = [set() for _ in range(nat)]
    for i, j in bonds:
        nb[i].add(j); nb[j].add(i)
    ang = []
    for j in range(nat):
        lst = sorted(nb[j])
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                ang.append([lst[a], j, lst[b]])
    dih = []
    for j, k in bonds:
        for i in sorted(nb[j] - {k}):
            for l in sorted(nb[k] - {j}):
                if i != l:
                    dih.append([i, j, k, l])
    return ang, dih


def build_molecules(system: dict, seed: int, masses_of) -> System:
    rng = np.random.RandomState(seed)
    species = []
    for m in system["molecules"]:
        d = molecule_def(m["molecule"])
        xyz = np.array([a[1:] for a in d["atoms"]], float)
        xyz -= xyz.mean(axis=0)
        ang, dih = graph_angles(len(xyz), d["bonds"])
        d["angles"] = d["angles"] if d["angles"] is not None else ang
        d["dihedrals"] = d["dihedrals"] if d["dihedrals"] is not None else dih
        species += [(d, xyz)] * int(m["count"])
    nmol = len(species)
    if "box" in system:
        box3 = [float(t) for t in system["box"]]
    else:
        mass = sum(sum(masses_of(a[0]) for a in d["atoms"]) for d, _ in species)
        vol = mass / AVOGADRO / float(system["density_g_cm3"]) * 1e24     # A^3
        box3 = [vol ** (1 / 3)] * 3
    ng = [max(1, int(math.ceil((nmol * box3[i] ** 3 / (box3[0] * box3[1] * box3[2])) ** (1 / 3) - 1e-9)))
          for i in range(3)]
    while ng[0] * ng[1] * ng[2] < nmol:
        ng[int(np.argmin(np.array(ng) / np.array(box3)))] += 1
    sites = np.array([[i + 0.5, j + 0.5, k + 0.5] for i in range(ng[0]) for j in range(ng[1]) for k in range(ng[2])])
    sites = sites / np.array(ng) * np.array(box3)
    order = rng.permutation(len(sites))[:nmol]
    perm = rng.permutation(nmol)
    pos, labels, molid, bonds, angles, dihs = [], [], [], [], [], []
    for mi, si in enumerate(order):
        d, xyz = species[perm[mi]]
        R = _random_rotation(rng)
        base = len(pos)
        for a, x in zip(d["atoms"], xyz @ R.T + sites[si]):
            pos.append(x % np.array(box3)); labels.append(a[0]); molid.append(mi + 1)
        lab = lambda i: d["atoms"][i][0]
        for b in d["bonds"]:
            bonds.append((tuple(lab(i) for i in b), [base + i for i in b]))
        for b in d["angles"]:
            angles.append((tuple(lab(i) for i in b), [base + i for i in b]))
        for b in d["dihedrals"]:
            dihs.append((tuple(lab(i) for i in b), [base + i for i in b]))
    tnames = []
    for l in labels:
        if l not in tnames:
            tnames.append(l)
    s = System(box=[*box3, 0.0, 0.0, 0.0], pos=np.array(pos), types=np.array([tnames.index(l) + 1 for l in labels]),
               type_names=tnames, masses=[masses_of(t) for t in tnames], atom_style="full",
               charges=np.zeros(len(pos)), mol=np.array(molid))
    s._raw_topology = {"bonds": bonds, "angles": angles, "dihedrals": dihs}
    if min(box3) / max(ng) < 2.0:
        s.warnings.append("system: molecules are packed closer than 2 A apart; minimize first")
    return s


# ------------------------------------------------------------------------------------------- data files
def read_data(text: str, atom_style: str = "atomic") -> System:
    """Tolerant reader of LAMMPS data files (atomic, charge, full; orthogonal or triclinic; Velocities)."""
    lines = text.splitlines()
    head = {}
    box = [None] * 6
    lo = [0.0, 0.0, 0.0]
    i = 1
    sections = {}
    names = ("Atoms", "Velocities", "Masses", "Bonds", "Angles", "Dihedrals", "Impropers", "Pair Coeffs",
             "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs")
    cur = None
    while i < len(lines):
        raw = lines[i]
        ln = raw.split("#")[0].strip()
        i += 1
        if not ln:
            continue
        sec = ln if ln in names else None
        if sec is not None:
            cur = sec
            sections[cur] = []
            continue
        t = ln.split()
        if cur is None:
            if len(t) >= 4 and t[2] == "xlo":
                box[0] = float(t[1]) - float(t[0]); lo[0] = float(t[0])
            elif len(t) >= 4 and t[2] == "ylo":
                box[1] = float(t[1]) - float(t[0]); lo[1] = float(t[0])
            elif len(t) >= 4 and t[2] == "zlo":
                box[2] = float(t[1]) - float(t[0]); lo[2] = float(t[0])
            elif len(t) >= 6 and t[3] == "xy":
                box[3], box[4], box[5] = float(t[0]), float(t[1]), float(t[2])
            elif len(t) >= 2:
                head[" ".join(t[1:])] = t[0]
        else:
            sections[cur].append(t)
    for k in range(3, 6):
        box[k] = box[k] or 0.0
    if None in box[:3]:
        raise ValueError("system.data: no xlo xhi / ylo yhi / zlo zhi lines")
    rows = sections.get("Atoms") or []
    if not rows:
        raise ValueError("system.data: no Atoms section")
    rows = sorted(rows, key=lambda r: int(r[0]))
    ids = [int(r[0]) for r in rows]
    idx = {a: k for k, a in enumerate(ids)}
    if atom_style == "atomic":
        types = [int(r[1]) for r in rows]; xyz = [[float(v) for v in r[2:5]] for r in rows]; q = None; mol = None
    elif atom_style == "charge":
        types = [int(r[1]) for r in rows]; q = [float(r[2]) for r in rows]; xyz = [[float(v) for v in r[3:6]] for r in rows]
        mol = None
    else:
        mol = [int(r[1]) for r in rows]; types = [int(r[2]) for r in rows]; q = [float(r[3]) for r in rows]
        xyz = [[float(v) for v in r[4:7]] for r in rows]
    ntypes = int(head.get("atom types", max(types)))
    masses = [1.0] * ntypes
    for r in sections.get("Masses") or []:
        masses[int(r[0]) - 1] = float(r[1])
    s = System(box=box, pos=np.array(xyz) - np.array(lo), types=np.array(types), type_names=[str(k + 1) for k in range(ntypes)],
               masses=masses, atom_style=atom_style, charges=None if q is None else np.array(q),
               mol=None if mol is None else np.array(mol))
    if "Velocities" in sections:
        v = np.zeros((len(ids), 3))
        for r in sections["Velocities"]:
            v[idx[int(r[0])]] = [float(t) for t in r[1:4]]
        s.velocities = v
    for key, n in (("Bonds", 2), ("Angles", 3), ("Dihedrals", 4)):
        out = [(int(r[1]), *[idx[int(t)] for t in r[2:2 + n]]) for r in sections.get(key) or []]
        setattr(s, key.lower(), out)
    s.ids = np.array(ids)
    return s


def write_data(s: System, path: Path, title: str = "lammps-translator") -> None:
    """LAMMPS data file. Atom ids are 1..N in our order; positions wrapped into the box."""
    lx, ly, lz, xy, xz, yz = s.box
    out = [f"{title}", "", f"{s.n} atoms", f"{len(s.type_names)} atom types"]
    if s.atom_style == "full":
        out += [f"{len(s.bonds)} bonds", f"{len(s.bond_types)} bond types",
                f"{len(s.angles)} angles", f"{len(s.angle_types)} angle types",
                f"{len(s.dihedrals)} dihedrals", f"{len(s.dihedral_types)} dihedral types"]
    out += ["", f"0.0 {lx:.16g} xlo xhi", f"0.0 {ly:.16g} ylo yhi", f"0.0 {lz:.16g} zlo zhi"]
    if s.triclinic():
        out.append(f"{xy:.16g} {xz:.16g} {yz:.16g} xy xz yz")
    out += ["", "Masses", ""]
    out += [f"{k + 1} {m:.10g}  # {s.type_names[k]}" for k, m in enumerate(s.masses)]
    out += ["", f"Atoms  # {s.atom_style}", ""]
    P = wrap(s.pos, s.box)
    for k in range(s.n):
        x, y, z = P[k]
        if s.atom_style == "atomic":
            out.append(f"{k + 1} {s.types[k]} {x:.16g} {y:.16g} {z:.16g}")
        elif s.atom_style == "charge":
            out.append(f"{k + 1} {s.types[k]} {s.charges[k]:.12g} {x:.16g} {y:.16g} {z:.16g}")
        else:
            out.append(f"{k + 1} {s.mol[k]} {s.types[k]} {s.charges[k]:.12g} {x:.16g} {y:.16g} {z:.16g}")
    if s.velocities is not None:
        out += ["", "Velocities", ""]
        out += [f"{k + 1} {v[0]:.16g} {v[1]:.16g} {v[2]:.16g}" for k, v in enumerate(s.velocities)]
    for key, n in (("Bonds", 2), ("Angles", 3), ("Dihedrals", 4)):
        items = getattr(s, key.lower())
        if items:
            out += ["", key, ""]
            out += [f"{k + 1} {it[0]} " + " ".join(str(a + 1) for a in it[1:1 + n]) for k, it in enumerate(items)]
    path.write_text("\n".join(out) + "\n")


def wrap(pos: np.ndarray, box) -> np.ndarray:
    lx, ly, lz, xy, xz, yz = box
    H = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])   # columns a, b, c
    f = np.linalg.solve(H, pos.T).T
    f = f - np.floor(f)
    f[f >= 1.0] -= 1.0
    return (H @ f.T).T


# --------------------------------------------------------------------------------------------- entry point
def build_system(study: dict, cutoff: float | None = None) -> System:
    """The initial System of a (valid) study, with force-field typing applied for molecular systems."""
    from .potential import assign_forcefield, forcefield_of, potential_cutoff

    system = study["system"]
    seed = int(study.get("seed", 12345))
    if "lattice" in system:
        if cutoff is None:
            try:
                cutoff = potential_cutoff(study)
            except Exception:  # noqa: BLE001
                cutoff = None
        s = build_lattice(system, seed, cutoff)
        q = (study.get("potential") or {}).get("charges")
        if q:
            s.atom_style = "charge"
            s.charges = np.array([float(q.get(nm, 0.0)) for nm in s.names()])
        return s
    if "molecules" in system:
        ff = forcefield_of(study)
        def masses_of(lab):
            t = ff["types"].get(lab)
            if t is None:
                raise ValueError(f"forcefield.types: no type '{lab}'")
            return float(t["mass"])
        s = build_molecules(system, seed, masses_of)
        assign_forcefield(s, ff)
        return s
    text = system["data"]
    if "\n" not in text and Path(text).expanduser().is_file():
        text = Path(text).expanduser().read_text()
    s = read_data(text, system.get("atom_style", "atomic"))
    tn = list(system["type_names"])
    if len(tn) != len(s.type_names):
        raise ValueError(f"system.type_names: {len(tn)} names for {len(s.type_names)} atom types in the data file")
    s.type_names = tn
    if s.atom_style == "full":
        ff = forcefield_of(study)
        for k, nm in enumerate(tn):
            if nm in ff["types"]:
                s.masses[k] = float(ff["types"][nm]["mass"])
        lab = lambda i: tn[s.types[i] - 1]
        s._raw_topology = {"bonds": [(tuple(lab(i) for i in b[1:]), list(b[1:])) for b in s.bonds],
                           "angles": [(tuple(lab(i) for i in b[1:]), list(b[1:])) for b in s.angles],
                           "dihedrals": [(tuple(lab(i) for i in b[1:]), list(b[1:])) for b in s.dihedrals]}
        assign_forcefield(s, ff, keep_charges=True)
    else:
        for k, nm in enumerate(tn):
            if nm in MASSES and "Masses" not in text:
                s.masses[k] = MASSES[nm]
        q = (study.get("potential") or {}).get("charges")
        if q and s.atom_style == "atomic":
            s.atom_style = "charge"
            s.charges = np.array([float(q.get(nm, 0.0)) for nm in s.names()])
    return s
