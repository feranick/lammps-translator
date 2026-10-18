"""analysis.py — numbers from LAMMPS output: stresses, elastic constants and moduli, Birch-Murnaghan fits,
MD averages with block errors, diffusion, RDF peaks.
Version: 2026.10.18.6
"""
from __future__ import annotations

import math

import numpy as np

from .data import AMU_A3_TO_G_CM3, EV_A2_TO_J_M2, EV_TO_KJ_MOL, UNITS

EV_A3_TO_GPA = 160.21766208


def r(x, n=10):
    return float(f"{float(x):.{n}g}")


def stress_GPa(tag: dict, units: str) -> list[float]:
    """Stress (Voigt xx, yy, zz, yz, xz, xy; tension positive) = - pressure tensor."""
    c = UNITS[units]["p_to_GPa"]
    return [-tag[k] * c for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")]


def to_eV(e: float, units: str) -> float:
    return e * UNITS[units]["e_to_eV"]


# ------------------------------------------------------------------------------------------- elastic
def elastic_constants(tags: dict, eps: float, units: str) -> dict:
    C = np.zeros((6, 6))
    for v in range(1, 7):
        sp = np.array(stress_GPa(tags[f"s{v}p"], units))
        sm = np.array(stress_GPa(tags[f"s{v}m"], units))
        C[:, v - 1] = (sp - sm) / (2 * eps)
    asym = float(np.max(np.abs(C - C.T)))
    Cs = 0.5 * (C + C.T)
    out = {"C_GPa": [[r(x, 8) for x in row] for row in Cs], "asymmetry_GPa": r(asym, 4)}
    for a in range(6):
        for b in range(a, 6):
            out[f"C{a + 1}{b + 1}_GPa"] = r(Cs[a, b], 8)
    ev = np.linalg.eigvalsh(Cs)
    out["born_stable"] = bool(ev.min() > 0)
    out["min_eigenvalue_GPa"] = r(ev.min(), 6)
    BV = (Cs[0, 0] + Cs[1, 1] + Cs[2, 2] + 2 * (Cs[0, 1] + Cs[0, 2] + Cs[1, 2])) / 9
    GV = ((Cs[0, 0] + Cs[1, 1] + Cs[2, 2]) - (Cs[0, 1] + Cs[0, 2] + Cs[1, 2]) + 3 * (Cs[3, 3] + Cs[4, 4] + Cs[5, 5])) / 15
    try:
        S = np.linalg.inv(Cs)
        BR = 1 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    except np.linalg.LinAlgError:
        BR = GR = float("nan")
    B, G = (BV + BR) / 2, (GV + GR) / 2
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
    out.update({"B_voigt_GPa": r(BV, 8), "B_reuss_GPa": r(BR, 8), "G_voigt_GPa": r(GV, 8), "G_reuss_GPa": r(GR, 8),
                "B_GPa": r(B, 8), "G_GPa": r(G, 8), "E_GPa": r(E, 8), "nu": r(nu, 8)})
    # cubic summary (averages of the symmetric-equivalent entries) and anisotropy
    c11 = (Cs[0, 0] + Cs[1, 1] + Cs[2, 2]) / 3
    c12 = (Cs[0, 1] + Cs[0, 2] + Cs[1, 2]) / 3
    c44 = (Cs[3, 3] + Cs[4, 4] + Cs[5, 5]) / 3
    out["cubic"] = {"C11_GPa": r(c11, 8), "C12_GPa": r(c12, 8), "C44_GPa": r(c44, 8),
                    "zener_A": r(2 * c44 / (c11 - c12), 6) if abs(c11 - c12) > 1e-9 else None,
                    "cauchy_C12_minus_C44_GPa": r(c12 - c44, 6)}
    p0 = -np.mean(stress_GPa(tags["ref"], units)[:3])
    out["reference_pressure_GPa"] = r(p0, 6)
    return out


# ------------------------------------------------------------------------------------------- EOS
def birch_murnaghan(V: np.ndarray, E: np.ndarray) -> dict:
    """3rd-order BM: E is a cubic in x = V^(-2/3); fit by linear least squares (V per atom A^3, E eV)."""
    x = V ** (-2.0 / 3.0)
    c = np.polyfit(x, E, 3)
    p = np.poly1d(c)
    dp, d2p, d3p = p.deriv(1), p.deriv(2), p.deriv(3)
    roots = [rt.real for rt in dp.roots if abs(rt.imag) < 1e-12 and rt.real > 0]
    if not roots:
        return {"ok": False, "why": "no minimum in the fitted range"}
    x0 = min(roots, key=lambda t: abs(t - x[np.argmin(E)]))
    V0 = x0 ** (-1.5)
    # derivatives in V: x = V^-2/3; dx/dV = -2/3 x / V
    def dE(Vv):
        xv = Vv ** (-2 / 3); return dp(xv) * (-2 / 3) * xv / Vv
    def d2E(Vv):
        xv = Vv ** (-2 / 3); xp = -2 / 3 * xv / Vv; xpp = 10 / 9 * xv / Vv ** 2
        return d2p(xv) * xp ** 2 + dp(xv) * xpp
    def d3E(Vv):
        xv = Vv ** (-2 / 3); xp = -2 / 3 * xv / Vv; xpp = 10 / 9 * xv / Vv ** 2; xppp = -80 / 27 * xv / Vv ** 3
        return d3p(xv) * xp ** 3 + 3 * d2p(xv) * xp * xpp + dp(xv) * xppp
    B0 = V0 * d2E(V0)
    Bp = -(1 + V0 * d3E(V0) / d2E(V0))
    resid = E - p(x)
    return {"ok": True, "V0_A3": r(V0, 10), "E0_eV": r(p(x0), 12), "B0_GPa": r(B0 * EV_A3_TO_GPA, 8), "B0_prime": r(Bp, 6),
            "fit_rms_eV": r(np.sqrt(np.mean(resid ** 2)), 4), "P_fit_GPa": [-dE(v) * EV_A3_TO_GPA for v in V]}


def work_check(scales: np.ndarray, V: np.ndarray, E: np.ndarray, P_GPa: np.ndarray, abs_tol: float = 0.0) -> dict:
    """E(V_i+2) - E(V_i) = -integral P dV, by Simpson's rule in the (uniform) scale variable f, V = V1 f^3:
    LAMMPS's virial must be the volume derivative of its own energy."""
    V1 = V / scales ** 3
    g = P_GPa / EV_A3_TO_GPA * 3 * V1 * scales ** 2      # P dV/df, eV per atom
    h = scales[1] - scales[0]
    worst, worst_abs, ok = 0.0, 0.0, True
    for i in range(0, len(scales) - 2):
        w = h / 3 * (g[i] + 4 * g[i + 1] + g[i + 2])
        dE = E[i + 2] - E[i]
        mis = abs(dE + w)
        span = float(np.max(E) - np.min(E)) + 1e-12
        worst = max(worst, mis / span)
        worst_abs = max(worst_abs, mis)
        ok = ok and mis <= 2e-2 * span + abs_tol
    return {"max_rel_mismatch": r(worst, 4), "max_mismatch_eV_per_atom": r(worst_abs, 4), "pass": bool(ok), "smooth": bool(worst <= 2e-3),
            "abs_tol_eV_per_atom": r(abs_tol, 4),
            "what": "E(V2) - E(V0) vs -integral P dV (Simpson) per atom; relative to the energy span of the curve"}


# ------------------------------------------------------------------------------------------- MD
def block_average(x: np.ndarray, nblocks: int = 5) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < nblocks * 2:
        return float(np.mean(x)), float("nan")
    b = np.array_split(x, nblocks)
    m = np.array([t.mean() for t in b])
    return float(x.mean()), float(m.std(ddof=1) / math.sqrt(nblocks))


def md_summary(block: dict, meta: dict, units: str, natoms: int, nmol: int, mass_amu: float) -> dict:
    cols = block["columns"]
    R = block["rows"]
    step = R[:, cols.index("step")]
    n0 = int(meta.get("average_from", 0.5) * (step[-1] - step[0])) + step[0] if len(step) else 0
    sel = step >= n0
    U = UNITS[units]
    pc = U["p_to_GPa"]
    ec = U["e_to_eV"]
    get = lambda k: R[sel, cols.index(k)]
    series = {"T_K": get("temp"), "P_GPa": get("press") * pc, "P_bar": get("press") * pc * 1e4,
              "pe_per_atom_eV": get("pe") * ec / natoms, "etotal_per_atom_eV": get("etotal") * ec / natoms,
              "volume_A3": get("vol"), "density_g_cm3": mass_amu / get("vol") * AMU_A3_TO_G_CM3,
              "lx_A": get("lx"), "ly_A": get("ly"), "lz_A": get("lz")}
    for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy"):
        series[f"{k}_GPa"] = get(k) * pc
    if nmol:
        series["pe_per_molecule_kJ_mol"] = get("pe") * ec * EV_TO_KJ_MOL / nmol
    mean, err = {}, {}
    for k, v in series.items():
        mean[k], err[k] = (r(t, 10) for t in block_average(v))
    out = {"mean": mean, "err": err, "samples": int(sel.sum()), "time_ps": r((step[-1] - step[0]) * meta["dt_ps"], 8)}
    if meta.get("msd") and "c_msd[4]" in cols:
        t = (step - step[0]) * meta["dt_ps"]
        msd = R[:, cols.index("c_msd[4]")]
        a, b = np.polyfit(t[sel], msd[sel], 1)
        D = a / 6.0
        out["msd"] = {"D_A2_ps": r(D, 8), "D_m2_s": r(D * 1e-8, 8), "D_cm2_s": r(D * 1e-4, 8),
                      "final_A2": r(msd[-1], 8),
                      "note": "slope of the MSD over the averaging window / 6; the finite box lowers D "
                              "(Yeh-Hummer correction not applied)"}
    if "pe" in cols and "etotal" in cols:
        et = R[:, cols.index("etotal")] * ec / natoms
        tt = (step - step[0]) * meta["dt_ps"]
        if len(tt) > 2 and tt[-1] > 0:
            out["etotal_slope_eV_per_atom_ps"] = r(np.polyfit(tt, et, 1)[0], 6)
    return out


def rdf_summary(blocks: list, meta: dict) -> dict:
    if not blocks:
        return {"error": "no rdf output"}
    B = blocks[-1]
    rr = B[:, 1]
    out = {"r_A": [r(x, 6) for x in rr]}
    pairs = meta["pairs"]
    for k, pr in enumerate(pairs):
        g = B[:, 2 + 2 * k]
        cn = B[:, 3 + 2 * k]
        i = int(np.argmax(g))
        # first minimum after the first peak
        j = i + int(np.argmin(g[i:i + max(3, len(g) // 4)])) if i < len(g) - 1 else i
        out[f"{pr[0]}-{pr[1]}"] = {"g": [r(x, 6) for x in g], "peak_r_A": r(rr[i], 6), "peak_g": r(g[i], 6),
                                   "first_min_r_A": r(rr[j], 6), "coordination": r(cn[j], 6)}
    return out


def surface_energy(e_slab: float, e_bulk: float, area: float, units: str) -> float:
    return (to_eV(e_slab, units) - to_eV(e_bulk, units)) / (2 * area) * EV_A2_TO_J_M2
