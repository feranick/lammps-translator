"""refcalc.py — an independent implementation of the energy, forces and stress of a structure.
Version: 2026.10.25.2

It reads the written data file and the normalised model (not the LAMMPS input), so a mistake in the
translation (types, coefficients, units, cutoffs, charges, topology) shows up as a mismatch with LAMMPS.
Covered: lj, morse, buck (shift, tail), Coulomb cut and long (Ewald, own parameters), eam (funcfl, with
LAMMPS's grid conversion), eam/alloy, eam/fs (LAMMPS's cubic splines), harmonic bonds and angles, OPLS
dihedrals, special-bond weights. Not covered (reported as such): tersoff, sw, meam and other many-body files.
Units are the model's (metal or real); pressures in the units' pressure (bar, atm).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc

from .data import UNITS

SUPPORTED = ("lj", "morse", "buck", "coul", "eam", "eam/alloy", "eam/fs")
MAX_ATOMS = 4000


def supported(model: dict) -> bool:
    return model["kind"] in SUPPORTED


# ------------------------------------------------------------------------------------------- neighbours
def neighbours(pos: np.ndarray, box, rc: float):
    """Full list (i, j, rij = ri - rj_image) of pairs with |rij| < rc over all periodic images."""
    lx, ly, lz, xy, xz, yz = box
    A = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]], float)
    V = abs(np.linalg.det(A))
    widths = [V / np.linalg.norm(np.cross(A[(k + 1) % 3], A[(k + 2) % 3])) for k in range(3)]
    n = [int(math.ceil(rc / w)) for w in widths]
    shifts = np.array([[i, j, k] for i in range(-n[0], n[0] + 1) for j in range(-n[1], n[1] + 1)
                       for k in range(-n[2], n[2] + 1)], float) @ A
    N = len(pos)
    img = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    t0 = cKDTree(pos)
    t1 = cKDTree(img)
    sdm = t0.sparse_distance_matrix(t1, rc, output_type="ndarray")
    i = sdm["i"].astype(int)
    k = sdm["j"].astype(int)
    d = sdm["v"]
    keep = d > 1e-10
    i, k, d = i[keep], k[keep], d[keep]
    j = k % N
    rij = pos[i] - img[k]
    return i, j, rij, np.linalg.norm(rij, axis=1)


# ------------------------------------------------------------------------------------------- results
class Acc:
    def __init__(self, n):
        self.e = {}
        self.f = np.zeros((n, 3))
        self.w = np.zeros((3, 3))

    def pair(self, i, j, rij, fscalar_over_r, half=True):
        """Pair forces F_i = fscalar_over_r * rij (full list: each pair twice, so the virial gets 1/2)."""
        fv = fscalar_over_r[:, None] * rij
        np.add.at(self.f, i, fv)
        fac = 0.5 if half else 1.0
        self.w += fac * np.einsum("ka,kb->ab", rij, fv)

    def add(self, key, v):
        self.e[key] = self.e.get(key, 0.0) + float(v)


# ------------------------------------------------------------------------------------------- pair styles
def _lj(r, eps, sig):
    sr6 = (sig / r) ** 6
    e = 4 * eps * (sr6 * sr6 - sr6)
    f = 24 * eps * (2 * sr6 * sr6 - sr6) / r   # -dE/dr
    return e, f


def _morse(r, D0, a, r0):
    x = np.exp(-a * (r - r0))
    return D0 * (x * x - 2 * x), 2 * a * D0 * (x * x - x)


def _buck(r, A, rho, C):
    ex = A * np.exp(-r / rho)
    return ex - C / r ** 6, ex / rho - 6 * C / r ** 7


def _pair_terms(model, s, acc, i, j, rij, r, factor_lj):
    kind = model["kind"]
    if kind == "coul":
        return
    rc = model["cutoff"]
    ti, tj = s.types[i], s.types[j]
    E = np.zeros(len(r))
    F = np.zeros(len(r))
    for (a, b), v in model["pair"].items():
        sel = ((ti == a) & (tj == b)) | ((ti == b) & (tj == a))
        if not np.any(sel):
            continue
        rr = r[sel]
        if kind == "lj":
            if v["eps"] == 0.0:
                continue
            e, f = _lj(rr, v["eps"], v["sigma"])
            off = _lj(np.array([rc]), v["eps"], v["sigma"])[0][0] if model.get("shift") else 0.0
        elif kind == "morse":
            e, f = _morse(rr, v["D0"], v["alpha"], v["r0"])
            off = _morse(np.array([rc]), v["D0"], v["alpha"], v["r0"])[0][0] if model.get("shift") else 0.0
        else:
            e, f = _buck(rr, v["A"], v["rho"], v["C"])
            off = _buck(np.array([rc]), v["A"], v["rho"], v["C"])[0][0] if model.get("shift") else 0.0
        E[sel] = e - off
        F[sel] = f
    E *= factor_lj
    F *= factor_lj
    acc.add("evdwl", 0.5 * E.sum())
    acc.pair(i, j, rij, F / r)


def _tail(model, s):
    """LAMMPS's long-range corrections (energy*volume and pressure*volume^2, as Pair::etail/ptail)."""
    if not model.get("tail") or model["kind"] not in ("lj", "buck"):
        return 0.0, 0.0
    rc = model["cutoff"]
    counts = {t: int(np.sum(s.types == t)) for t in range(1, len(s.type_names) + 1)}
    et = pt = 0.0
    for (a, b), v in model["pair"].items():
        n2 = counts[a] * counts[b]
        if model["kind"] == "lj":
            sig6 = v["sigma"] ** 6
            rc3 = rc ** 3
            rc6, rc9 = rc3 * rc3, rc3 ** 3
            pre = 8.0 * math.pi * n2 * v["eps"] * sig6 / (9.0 * rc9)
            e, p = pre * (sig6 - 3.0 * rc6), 2.0 * pre * (2.0 * sig6 - 3.0 * rc6)
        else:
            A, rho, C = v["A"], v["rho"], v["C"]
            r2, r3 = rc ** 2, rc ** 3
            e = 2.0 * math.pi * n2 * (A * math.exp(-rc / rho) * rho * (r2 + 2 * rho * rc + 2 * rho ** 2) - C / (3.0 * r3))
            p = (-1 / 3.0) * 2.0 * math.pi * n2 * (-A * math.exp(-rc / rho) * (r3 + 3 * rho * r2 + 6 * rho ** 2 * rc
                                                                               + 6 * rho ** 3) + 2.0 * C / r3)
        mult = 1 if a == b else 2
        et += mult * e
        pt += mult * p
    return et, pt


# ------------------------------------------------------------------------------------------- Coulomb
def _ewald(s, acc, qqr2e, special_pairs, rc):
    q = s.charges
    pos = s.pos
    lx, ly, lz, xy, xz, yz = s.box
    A = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]], float)
    V = abs(np.linalg.det(A))
    tol = 1e-11
    rr = max(rc, 0.4 * min(lx, ly, lz))
    g = math.sqrt(-math.log(tol)) / rr
    i, j, rij, r = neighbours(pos, s.box, rr)
    qq = q[i] * q[j]
    ex = erfc(g * r)
    acc.add("ecoul_long", 0.5 * qqr2e * np.sum(qq * ex / r))
    fr = qqr2e * qq * (ex / r + 2 * g / math.sqrt(math.pi) * np.exp(-(g * r) ** 2)) / r ** 2
    acc.pair(i, j, rij, fr)
    acc.add("ecoul_long", -qqr2e * g / math.sqrt(math.pi) * np.sum(q * q))
    # reciprocal space
    B = 2 * math.pi * np.linalg.inv(A).T            # rows: reciprocal vectors
    kmax = 2 * g * math.sqrt(-math.log(tol))
    nmax = [int(math.ceil(kmax / np.linalg.norm(B[d]))) + 1 for d in range(3)]
    grid = np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing="ij")).reshape(3, -1).T
    K = grid @ B
    k2 = np.einsum("ka,ka->k", K, K)
    sel = (k2 > 0) & (k2 <= kmax * kmax)
    K, k2 = K[sel], k2[sel]
    ak = np.exp(-k2 / (4 * g * g)) / k2
    E_rec, W = 0.0, np.zeros((3, 3))
    F = np.zeros_like(pos)
    for c0 in range(0, len(K), 2000):
        Kc, akc, k2c = K[c0:c0 + 2000], ak[c0:c0 + 2000], k2[c0:c0 + 2000]
        ph = pos @ Kc.T
        cs, sn = np.cos(ph), np.sin(ph)
        Sre, Sim = q @ cs, q @ sn
        S2 = Sre ** 2 + Sim ** 2
        E_rec += np.sum(akc * S2)
        W += np.einsum("k,ab->ab", akc * S2, np.eye(3)) - 2 * np.einsum(
            "k,ka,kb->ab", akc * S2 * (1 / k2c + 1 / (4 * g * g)), Kc, Kc)
        F += q[:, None] * ((sn * Sre[None, :] - cs * Sim[None, :]) * akc[None, :]) @ Kc
    pref = 2 * math.pi / V * qqr2e
    acc.add("ecoul_long", pref * E_rec)
    acc.w += pref * W
    acc.f += 2 * pref * F
    for (a, b, w) in special_pairs:
        if w >= 1.0 or q[a] * q[b] == 0:
            continue
        d = _min_image(pos[a] - pos[b], A)
        rn = np.linalg.norm(d)
        e = (1 - w) * qqr2e * q[a] * q[b] / rn
        acc.add("ecoul_long", -e)
        fv = -(e / rn ** 2) * d
        acc.f[a] += fv
        acc.f[b] -= fv
        acc.w += np.outer(d, fv)


def _min_image(d, A):
    f = np.linalg.solve(A.T, d)
    f -= np.round(f)
    return A.T @ f


def _coul_cut(s, acc, qqr2e, i, j, rij, r, factor_coul):
    qq = s.charges[i] * s.charges[j] * factor_coul
    acc.add("ecoul", 0.5 * qqr2e * np.sum(qq / r))
    acc.pair(i, j, rij, qqr2e * qq / r ** 3)


# ------------------------------------------------------------------------------------------- EAM
def _spline(f: np.ndarray, delta: float) -> np.ndarray:
    """LAMMPS PairEAM::interpolate: 1-based arrays of 7 coefficients (index 0 unused)."""
    n = len(f)
    s = np.zeros((n + 1, 7))
    s[1:, 6] = f
    s[1, 5] = s[2, 6] - s[1, 6]
    s[2, 5] = 0.5 * (s[3, 6] - s[1, 6])
    s[n - 1, 5] = 0.5 * (s[n, 6] - s[n - 2, 6])
    s[n, 5] = s[n, 6] - s[n - 1, 6]
    for m in range(3, n - 1):
        s[m, 5] = ((s[m - 2, 6] - s[m + 2, 6]) + 8.0 * (s[m + 1, 6] - s[m - 1, 6])) / 12.0
    for m in range(1, n):
        s[m, 4] = 3.0 * (s[m + 1, 6] - s[m, 6]) - 2.0 * s[m, 5] - s[m + 1, 5]
        s[m, 3] = s[m, 5] + s[m + 1, 5] - 2.0 * (s[m + 1, 6] - s[m, 6])
    s[n, 4] = s[n, 3] = 0.0
    s[1:, 2] = s[1:, 5] / delta
    s[1:, 1] = 2.0 * s[1:, 4] / delta
    s[1:, 0] = 3.0 * s[1:, 3] / delta
    return s


def _seval(s, x, rdx, nmax, lower1=False):
    p = x * rdx + 1.0
    m = p.astype(int)
    m = np.minimum(m, nmax - 1)
    if lower1:
        m = np.maximum(1, m)
    p = np.minimum(p - m, 1.0)
    c = s[m]
    val = ((c[:, 3] * p + c[:, 4]) * p + c[:, 5]) * p + c[:, 6]
    der = (c[:, 0] * p + c[:, 1]) * p + c[:, 2]
    return val, der


class _Reader:
    """Reads values as LAMMPS's TextFileReader::next_dvector does: line by line, and the rest of the line
    after the last value of a block is discarded (blocks start on new lines)."""

    def __init__(self, lines):
        self.lines, self.k = lines, 0

    def line(self):
        ln = self.lines[self.k]
        self.k += 1
        return ln.split("#")[0].split()

    def values(self, n):
        out = []
        while len(out) < n:
            out += [float(x) for x in self.line()]
        return np.array(out[:n])


def _tokens(path: Path, skip: int):
    lines = Path(path).read_text(errors="replace").splitlines()
    r = _Reader(lines)
    r.k = skip
    return lines[:skip], r


def _read_funcfl(path):
    head, t = _tokens(path, 3)
    h = head[2].split()
    nrho, drho, nr, dr, cut = int(h[0]), float(h[1]), int(h[2]), float(h[3]), float(h[4])
    return {"nrho": nrho, "drho": drho, "nr": nr, "dr": dr, "cut": cut, "frho": t.values(nrho),
            "zr": t.values(nr), "rhor": t.values(nr)}


def _interp4(src: np.ndarray, d_src: float, n_new: int, d_new: float) -> np.ndarray:
    """PairEAM::file2array's 4-point Lagrange interpolation of a funcfl array onto the common grid."""
    nsrc = len(src)
    f = np.concatenate([[0.0], src])     # 1-based
    out = np.zeros(n_new)
    for m in range(1, n_new + 1):
        r = (m - 1) * d_new
        p = r / d_src + 1.0
        k = int(p)
        k = min(k, nsrc - 2)
        k = max(k, 2)
        p -= k
        p = min(p, 2.0)
        c1 = -p * (p - 1.0) * (p - 2.0) / 6.0
        c2 = 0.5 * (p * p - 1.0) * (p - 2.0)
        c3 = -0.5 * p * (p + 1.0) * (p - 2.0)
        c4 = p * (p * p - 1.0) / 6.0
        out[m - 1] = c1 * f[k - 1] + c2 * f[k] + c3 * f[k + 1] + c4 * f[k + 2]
    return out


def eam_tables(model: dict, s, files: dict) -> dict:
    """Spline tables per element as LAMMPS builds them."""
    st = model["kind"]
    els = s.type_names
    if st == "eam":
        fmap = model["files"] or {els[0]: model["file"]}
        fl = {e: _read_funcfl(files[fmap[e]]) for e in els}
        uniq = list(dict.fromkeys(fmap[e] for e in els))
        F = {u: _read_funcfl(files[u]) for u in uniq}
        dr = max(f["dr"] for f in F.values())
        drho = max(f["drho"] for f in F.values())
        rmax = max((f["nr"] - 1) * f["dr"] for f in F.values())
        rhomax = max((f["nrho"] - 1) * f["drho"] for f in F.values())
        nr = int(rmax / dr + 0.5)
        nrho = int(rhomax / drho + 0.5)
        cut = max(f["cut"] for f in F.values())
        frho = {e: _interp4(fl[e]["frho"], fl[e]["drho"], nrho, drho) for e in els}
        rhor = {e: _interp4(fl[e]["rhor"], fl[e]["dr"], nr, dr) for e in els}
        zr = {e: _interp4(fl[e]["zr"], fl[e]["dr"], nr, dr) for e in els}
        z2r = {(a, b): 27.2 * 0.529 * zr[a] * zr[b] for a in els for b in els}
        rho_ij = {(a, b): rhor[a] for a in els for b in els}   # density at b due to a
    else:
        head, t = _tokens(files[model["file"]], 5)
        fel = head[3].split()[1:]
        h = head[4].split()
        nrho, drho, nr, dr, cut = int(h[0]), float(h[1]), int(h[2]), float(h[3]), float(h[4])
        frho_f, rho_f = {}, {}
        for e in fel:
            t.line()                      # Z, mass, a, lattice
            frho_f[e] = t.values(nrho)
            if st == "eam/fs":
                for e2 in fel:
                    rho_f[(e, e2)] = t.values(nr)
            else:
                rho_f[e] = t.values(nr)
        z2_f = {}
        for a_i, a in enumerate(fel):
            for b in fel[:a_i + 1]:
                z2_f[(a, b)] = z2_f[(b, a)] = t.values(nr)
        frho = {e: frho_f[e] for e in els}
        z2r = {(a, b): z2_f[(a, b)] for a in els for b in els}
        if st == "eam/fs":
            rho_ij = {(a, b): rho_f[(a, b)] for a in els for b in els}   # block of neighbour a, sub-function b
        else:
            rho_ij = {(a, b): rho_f[a] for a in els for b in els}
    return {"nr": nr, "nrho": nrho, "dr": dr, "drho": drho, "cut": cut,
            "frho": {e: _spline(v, drho) for e, v in frho.items()},
            "rho": {k: _spline(v, dr) for k, v in rho_ij.items()},
            "z2r": {k: _spline(v, dr) for k, v in z2r.items()}}


def _eam(model, s, acc, files):
    T = eam_tables(model, s, files)
    els = s.type_names
    i, j, rij, r = neighbours(s.pos, s.box, T["cut"])
    sel = r * r < T["cut"] ** 2
    i, j, rij, r = i[sel], j[sel], rij[sel], r[sel]
    rdr, rdrho = 1.0 / T["dr"], 1.0 / T["drho"]
    ei = np.array([els[t - 1] for t in s.types[i]])
    ej = np.array([els[t - 1] for t in s.types[j]])
    rho = np.zeros(s.n)
    rho_ji_d = np.zeros(len(r))   # d/dr of density at i due to j
    rho_ij_d = np.zeros(len(r))   # density at j due to i
    phi = np.zeros(len(r))
    phid = np.zeros(len(r))
    for a in els:
        for b in els:
            m = (ei == b) & (ej == a)            # central b, neighbour a
            if not np.any(m):
                continue
            v, d = _seval(T["rho"][(a, b)], r[m], rdr, T["nr"])
            np.add.at(rho, i[m], v)
            rho_ji_d[m] = d
            v2, d2 = _seval(T["rho"][(b, a)], r[m], rdr, T["nr"])
            rho_ij_d[m] = d2
            z, zd = _seval(T["z2r"][(a, b)], r[m], rdr, T["nr"])
            phi[m] = z / r[m]
            phid[m] = zd / r[m] - phi[m] / r[m]
    fp = np.zeros(s.n)
    Fe = 0.0
    rhomax = (T["nrho"] - 1) * T["drho"]
    for e in els:
        m = np.array([els[t - 1] == e for t in s.types])
        if not np.any(m):
            continue
        v, d = _seval(T["frho"][e], rho[m], rdrho, T["nrho"], lower1=True)
        over = rho[m] > rhomax
        v = v + np.where(over, d * (rho[m] - rhomax), 0.0)
        fp[m] = d
        Fe += v.sum()
    acc.add("evdwl", Fe + 0.5 * phi.sum())
    fpair = -(fp[i] * rho_ji_d + fp[j] * rho_ij_d + phid) / r
    acc.pair(i, j, rij, fpair)
    acc.rho = rho


# ------------------------------------------------------------------------------------------- bonded
def _special(s, weights):
    """(a, b, weight) for 1-2, 1-3, 1-4 neighbours of the bond graph (weights [w12, w13, w14])."""
    nb = [set() for _ in range(s.n)]
    for b in s.bonds:
        nb[b[1]].add(b[2]); nb[b[2]].add(b[1])
    out = {}
    for a in range(s.n):
        d1 = nb[a]
        d2 = set().union(*[nb[x] for x in d1]) - d1 - {a} if d1 else set()
        d3 = set().union(*[nb[x] for x in d2]) - d2 - d1 - {a} if d2 else set()
        for lvl, group in ((0, d1), (1, d2), (2, d3)):
            for b in group:
                if b > a and (a, b) not in out:
                    out[(a, b)] = weights[lvl]
    return [(a, b, w) for (a, b), w in out.items()]


def _e_bond(x, c):
    return c["k"] * (np.linalg.norm(x[0] - x[1]) - c["r0"]) ** 2


def _e_angle(x, c):
    u, v = x[0] - x[1], x[2] - x[1]
    cs = np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)
    return c["k"] * (math.acos(cs) - math.radians(c["theta0"])) ** 2


def _e_dihedral(x, c):
    K = c["k"]
    b1, b2, b3 = x[1] - x[0], x[2] - x[1], x[3] - x[2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    phi = math.atan2(np.cross(n1, n2) @ b2 / np.linalg.norm(b2), n1 @ n2)
    return (0.5 * K[0] * (1 + math.cos(phi)) + 0.5 * K[1] * (1 - math.cos(2 * phi))
            + 0.5 * K[2] * (1 + math.cos(3 * phi)) + 0.5 * K[3] * (1 - math.cos(4 * phi)))


def _bonded(model, s, acc, h=1e-5):
    """Bonded energies (LAMMPS's formulas); forces and virial by central differences of each term
    (O(h^2) ~ 1e-10 relative)."""
    A = s.cell()
    for key, coeffs, fn in (("ebond", model["bonds"], _e_bond), ("eangle", model["angles"], _e_angle),
                            ("edihed", model["dihedrals"], _e_dihedral)):
        items = {"ebond": s.bonds, "eangle": s.angles, "edihed": s.dihedrals}[key]
        for it in items:
            c = coeffs[it[0] - 1]
            idx = list(it[1:])
            x = [s.pos[idx[0]]]
            for a in idx[1:]:
                x.append(x[-1] + _min_image(s.pos[a] - x[-1], A))   # unwrapped chain
            x = np.array(x)
            acc.add(key, fn(x, c))
            for a in range(len(idx)):
                for d in range(3):
                    xp, xm = x.copy(), x.copy()
                    xp[a, d] += h
                    xm[a, d] -= h
                    f = -(fn(xp, c) - fn(xm, c)) / (2 * h)
                    acc.f[idx[a], d] += f
                    acc.w[:, d] += x[a] * f


# ------------------------------------------------------------------------------------------- entry point
def compute(model: dict, s, files: dict | None = None) -> dict:
    """Energy terms, total pe, forces, pressure tensor [xx, yy, zz, yz, xz, xy] (no kinetic part)."""
    if not supported(model):
        return {"supported": False, "why": f"no reference implementation for {model['kind']}"}
    if s.n > MAX_ATOMS:
        return {"supported": False, "why": f"{s.n} atoms > {MAX_ATOMS} (reference check skipped)"}
    U = UNITS[model["units"]]
    acc = Acc(s.n)
    V = s.volume()
    special = _special(s, model["special"]) if model["molecular"] and s.bonds else []
    if model["kind"].startswith("eam"):
        _eam(model, s, acc, files or {})
    else:
        rc = model["cutoff"]
        i, j, rij, r = neighbours(s.pos, s.box, rc)
        flj = np.ones(len(r))
        fcl = np.ones(len(r))
        if special:
            w = {}
            for a, b, ww in special:
                w[(a, b)] = w[(b, a)] = ww
            key = i.astype(np.int64) * s.n + j
            sp = {a * s.n + b: ww for (a, b), ww in w.items()}
            hit = np.array([sp.get(int(k_), 1.0) for k_ in key]) if sp else np.ones(len(r))
            flj = hit.copy()
            fcl = hit.copy()
        _pair_terms(model, s, acc, i, j, rij, r, flj)
        et, pt = _tail(model, s)
        if et:
            acc.add("evdwl", et / V)
        coul = model.get("coulomb", "none")
        if coul == "cut":
            _coul_cut(s, acc, U["qqr2e"], i, j, rij, r, fcl)
        elif coul == "long":
            _ewald(s, acc, U["qqr2e"], special, rc)
        acc.ptail = pt
    if model["molecular"]:
        _bonded(model, s, acc)
    pe = sum(acc.e.values())
    P = acc.w / V * U["nktv2p"]
    pt = getattr(acc, "ptail", 0.0)
    if pt:
        P = P + np.eye(3) * pt / V ** 2 * U["nktv2p"]
    return {"supported": True, "pe": pe, "terms": dict(acc.e), "forces": acc.f,
            "pressure": [P[0, 0], P[1, 1], P[2, 2], P[1, 2], P[0, 2], P[0, 1]],
            "press": float(np.trace(P) / 3.0)}
