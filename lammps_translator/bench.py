"""bench.py — benchmark studies with known answers (closed forms, lattice sums, published values).
Version: 2026.10.18.6

Default cases take seconds; 'slow' ones run only by name. 'chain_feat_cantilever' needs fea-translator
(and CalculiX) in the same environment and is skipped otherwise.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .runner import run_study

MADELUNG = {"rocksalt": 1.747564594633182, "cscl": 1.762674773070988}   # referred to the nearest-neighbour distance


def _lj_lattice_sums(rc_over_r0_max: float):
    """fcc lattice sums over shells within the cutoff (in units of the nearest-neighbour distance)."""
    n = 6
    g = np.array([[i, j, k] for i in range(-n, n + 1) for j in range(-n, n + 1) for k in range(-n, n + 1)
                  if (i + j + k) % 2 == 0 and (i, j, k) != (0, 0, 0)], float) / math.sqrt(2.0)   # nn distance 1
    d = np.linalg.norm(g, axis=1)
    return g, d


def lj_fcc_closed_form(eps: float, sigma: float, rc: float) -> dict:
    """Truncated (unshifted) LJ fcc at zero pressure: nn distance, energy, C11, C12, C44 from lattice sums.
    The shells inside rc must stay inside under the small strains used (rc between shells)."""
    g, d = _lj_lattice_sums(rc / sigma)
    # E(r0)/atom = 2 eps [A12 (s/r0)^12 - A6 (s/r0)^6] with A_n = sum over shells inside the cutoff
    r = 1.09 * sigma
    for _ in range(50):   # shells inside the cutoff at the equilibrium distance (self-consistent)
        inside = d * r < rc
        A12, A6 = np.sum(d[inside] ** -12), np.sum(d[inside] ** -6)
        r0 = (2 * A12 / A6) ** (1 / 6) * sigma
        if np.all((d * r0 < rc) == inside):
            break
        r = r0
    E = 2 * eps * (A12 * (sigma / r0) ** 12 - A6 * (sigma / r0) ** 6)
    R = g[inside] * r0
    r = np.linalg.norm(R, axis=1)
    sr6 = (sigma / r) ** 6
    phi1 = -24 * eps * (2 * sr6 * sr6 - sr6) / r                  # dphi/dr
    phi2 = 24 * eps * (26 * sr6 * sr6 - 7 * sr6) / r ** 2         # d2phi/dr2
    w = (phi2 - phi1 / r) / r ** 2
    omega = (math.sqrt(2) * r0) ** 3 / 4
    x, y = R[:, 0], R[:, 1]
    eV_A3 = 160.21766208
    C11 = np.sum(w * x ** 4) / (2 * omega) * eV_A3
    C12 = np.sum(w * x ** 2 * y ** 2) / (2 * omega) * eV_A3
    gap = min(np.min(d[~inside] * r0) - rc, rc - np.max(d[inside] * r0))
    return {"a": math.sqrt(2) * r0, "E": E, "C11": C11, "C12": C12, "C44": C12, "shell_gap_A": gap}


def cases() -> dict:
    lj = {"eps": 0.0104, "sigma": 3.405, "rc": 3.0 * 3.405}
    lj_cf = lj_fcc_closed_form(lj["eps"], lj["sigma"], lj["rc"])
    out = {
        "lj_fcc_lattice_sums": {
            "what": "LJ fcc (Ar parameters, cutoff 3 sigma between shells): a, energy, C11, C12, C44 vs lattice sums",
            "study": {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [4, 4, 4]},
                      "potential": {"style": "lj", "eps": lj["eps"], "sigma": lj["sigma"], "cutoff": lj["rc"]},
                      "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic", "strain": 1e-4}]},
            "checks": [("minimize.a_A", lj_cf["a"], 1e-7), ("minimize.pe_per_atom_eV", lj_cf["E"], 1e-7),
                       ("elastic.C11_GPa", lj_cf["C11"], 1e-4), ("elastic.C12_GPa", lj_cf["C12"], 1e-4),
                       ("elastic.C44_GPa", lj_cf["C44"], 1e-4)],
            "why_tol": "exact lattice sums; elastic by +/-1e-4 strain: O(strain^2)"},
        "madelung_nacl": {
            "what": "rocksalt point charges +/-1: energy per ion pair = -M e^2/(4 pi eps0 r0), M = 1.747565",
            "study": {"system": {"lattice": "rocksalt", "a": 5.64, "elements": ["Na", "Cl"], "size": [2, 2, 2]},
                      "potential": {"style": "coul", "charges": {"Na": 1, "Cl": -1}, "cutoff": 5.0,
                                    "kspace_accuracy": 1e-10},
                      "stages": [{"kind": "energy"}]},
            "checks": [("energy.pe_per_atom_eV", -MADELUNG["rocksalt"] * 14.399645 / 2.82 / 2, 2e-6)],
            "why_tol": "LAMMPS's coul/long uses a polynomial erfc (relative error ~1e-7 per pair)"},
        "madelung_cscl": {
            "what": "CsCl point charges +/-1: M = 1.762675 (nearest-neighbour distance sqrt(3) a / 2)",
            "study": {"system": {"lattice": "cscl", "a": 4.12, "elements": ["Cs", "Cl"], "size": [3, 3, 3]},
                      "potential": {"style": "coul", "charges": {"Cs": 1, "Cl": -1}, "cutoff": 5.0,
                                    "kspace_accuracy": 1e-10},
                      "stages": [{"kind": "energy"}]},
            "checks": [("energy.pe_per_atom_eV", -MADELUNG["cscl"] * 14.399645 / (math.sqrt(3) * 4.12 / 2) / 2, 2e-6)],
            "why_tol": "as madelung_nacl"},
        "cauchy_mgo": {
            "what": "Buckingham + Ewald MgO (central forces, inversion centres) at zero pressure: C12 = C44; "
                    "bulk modulus from elastic = from the equation of state",
            "study": {"system": {"lattice": "rocksalt", "a": 4.21, "elements": ["Mg", "O"], "size": [2, 2, 2]},
                      "potential": {"style": "buck", "pairs": {"Mg-O": {"A": 1428.5, "rho": 0.2945, "C": 0.0},
                                                             "O-O": {"A": 22764.0, "rho": 0.149, "C": 27.88}},
                                    "charges": {"Mg": 2, "O": -2}, "cutoff": 8.0, "shift": True},
                      "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic"},
                                 {"kind": "eos", "strain": 0.01, "n": 9}]},
            "checks": [("elastic.C44_GPa", "elastic.C12_GPa", 1e-4), ("eos.B0_GPa", "elastic.B_GPa", 0.01)],
            "why_tol": "Cauchy holds exactly; B differs by finite-strain (B' term) and fit effects"},
        "eam_cu_foiles": {
            "what": "Cu_u3.eam (Foiles, Baskes, Daw 1986): a, E_coh, B, relaxed vacancy, (100) surface energy",
            "study": {"system": {"lattice": "fcc", "a": 3.6, "elements": ["Cu"], "size": [4, 4, 4]},
                      "potential": {"style": "eam", "file": "Cu_u3.eam"},
                      "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic"}, {"kind": "vacancy"},
                                 {"kind": "surface", "miller": [1, 0, 0]}]},
            "checks": [("minimize.a_A", 3.615, 0.001), ("minimize.E_coh_eV", 3.54, 0.002),
                       ("elastic.B_GPa", 138.0, 0.01), ("vacancy.E_f_eV", 1.28, 0.02),
                       ("surface_100.gamma_J_m2", 1.28, 0.02)],
            "rename": {"surface": "surface_100"},
            "why_tol": "the paper's rounded values (a, E_coh, B fitted; E_vac, gamma predicted)"},
        "tersoff_si": {
            "what": "Si.tersoff (Tersoff 1988, PRB 37 6991, 'Si(B)'): a, E_coh, B (paper); C11, C12, C44 relaxed "
                    "(as tabulated by Balamane, Halicioglu, Tiller, PRB 46, 2250, 1992): tests internal relaxation",
            "study": {"system": {"lattice": "diamond", "a": 5.4, "elements": ["Si"], "size": [3, 3, 3]},
                      "potential": {"style": "tersoff", "file": "Si.tersoff"},
                      "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic"}]},
            "checks": [("minimize.a_A", 5.432, 0.001), ("minimize.E_coh_eV", 4.63, 0.002), ("elastic.B_GPa", 98.0, 0.01),
                       ("elastic.C11_GPa", 121.0, 0.02), ("elastic.C12_GPa", 86.0, 0.02), ("elastic.C44_GPa", 10.0, 0.05)],
            "why_tol": "published rounded values; C44 is small and printed to 2 digits"},
        "equipartition_lj": {
            "what": "LJ fcc at 4 K (harmonic regime), Nose-Hoover NVT: <pe> - pe0 = (3/2) k T (N-1)/N per atom",
            "study": {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [3, 3, 3]},
                      "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 8.5, "shift": True},
                      "seed": 4321,
                      "stages": [{"kind": "minimize", "box": "iso"},
                                 {"kind": "nvt", "T": 4.0, "steps": 40000, "dt_fs": 4, "tdamp_fs": 400,
                                  "average_from": 0.25}]},
            "checks": [("nvt.mean.pe_per_atom_eV - minimize.pe_per_atom_eV", 1.5 * 8.617343e-5 * 4.0 * 107 / 108, 0.03)],
            "why_tol": "statistical (block error ~1 %) and a small anharmonic part at 4 K"},
        "nve_conservation": {
            "what": "LJ fcc at 40 K, NVE 1 fs: total-energy drift below 1e-5 eV/atom/ps",
            "study": {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [3, 3, 3]},
                      "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 8.5, "shift": True},
                      "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "nve", "T": 80, "steps": 10000}]},
            "checks": [("nve.etotal_slope_eV_per_atom_ps", 0.0, None, 1e-5)],
            "why_tol": "velocity Verlet with a shifted potential"},
        "spce_water": {
            "slow": True,
            "what": "SPC/E water, 216 molecules, 298 K, 1 atm (PPPM, tail): density, potential energy per molecule "
                    "(no polarization correction), O-O first peak",
            "study": {"system": {"molecules": [{"molecule": "spce", "count": 216}], "density_g_cm3": 0.997},
                      "seed": 2024,
                      "stages": [{"kind": "minimize"},
                                 {"kind": "npt", "T": 298, "P_bar": 1.01325, "steps": 40000, "dt_fs": 2},
                                 {"kind": "npt", "name": "prod", "T": 298, "P_bar": 1.01325, "steps": 60000, "dt_fs": 2,
                                  "average_from": 0.0, "rdf": {"pairs": [["OW", "OW"]]}}]},
            "checks": [("prod.mean.density_g_cm3", 0.998, 0.01), ("prod.mean.pe_per_molecule_kJ_mol", -46.7, 0.02),
                       ("prod.rdf.OW-OW.peak_r_A", 2.75, 0.03)],
            "why_tol": "statistical (120 ps production), literature spread of SPC/E values"},
        "chain_feat_cantilever": {
            "what": "lammps-translator elastic (Cu_u3) -> feat_material -> fea-translator cantilever: tip deflection = "
                    "F L^3 / (3 E I) with lmpt's E (shear and Poisson effects within 2 %)",
            "needs": "fea_translator",
            "study": {"system": {"lattice": "fcc", "a": 3.6, "elements": ["Cu"], "size": [3, 3, 3]},
                      "potential": {"style": "eam", "file": "Cu_u3.eam"},
                      "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic"}]}},
    }
    return out


def _get(res: dict, path: str):
    if " - " in path:
        a, b = path.split(" - ")
        return _get(res, a) - _get(res, b)
    v = res
    for p in path.split("."):
        v = v[int(p)] if isinstance(v, list) else v[p]
    return float(v)


def _chain_feat(res: dict, folder: Path) -> dict:
    try:
        import fea_translator.api as feat   # type: ignore
    except ImportError:
        return {"skipped": "fea-translator not installed here"}
    mat = {k: v for k, v in res["feat_material"].items() if k in ("E", "nu", "density")}
    L, b, h, F = 200.0, 10.0, 10.0, 50.0
    study = {"beam": {"part": {"goal": {"features": [{"kind": "base_box", "params": {"w": L, "d": b, "h": h}}]}},
                      "material": mat, "analysis": "static", "mesh": {"size": 2.5},
                      "bcs": [{"kind": "fixed", "name": "wall", "faces": {"nx": -1}},
                              {"kind": "force", "name": "tip", "faces": {"nx": 1}, "fz": -F}],
                      "probes": [{"name": "tip", "faces": {"nx": 1}}]}}
    r = feat.run(study)
    rr = r.get("studies", {}).get("beam", r)
    try:
        uz = abs(float(rr["probes"]["tip"]["mean_u"]["uz"]))
    except (KeyError, TypeError):
        return {"ok": False, "why": "fea-translator result without probes.tip.mean_u.uz", "feat": str(rr)[:500]}
    I = b * h ** 3 / 12
    expect = F * L ** 3 / (3 * mat["E"] * I)
    return {"checks": [{"quantity": "feat tip deflection (mm)", "got": uz, "expected": expect, "tol": 0.02,
                        "pass": abs(uz - expect) <= 0.02 * expect}]}


def prepare(names=None) -> tuple[dict, dict]:
    """The benchmark studies to run ({name: study}) and the rows already known (unknown names, skipped cases)."""
    allc = cases()
    if not names:
        names = [k for k, v in allc.items() if not v.get("slow")]
    studies, pre = {}, {}
    for n in names:
        if n not in allc:
            pre[n] = {"ok": False, "why": f"no benchmark '{n}' ({sorted(allc)})"}
            continue
        c = allc[n]
        if c.get("needs"):
            try:
                __import__(c["needs"])
            except ImportError:
                pre[n] = {"skipped": f"{c['needs']} not installed", "what": c["what"]}
                continue
        st = dict(c["study"])
        if "rename" in c:
            st = {**st, "stages": [dict(s, name=c["rename"].get(s["kind"])) if s["kind"] in c["rename"] else s
                                   for s in st["stages"]]}
        studies[n] = st
    return studies, pre


def evaluate(n: str, res: dict, folder: Path | None = None) -> dict:
    """A benchmark's row from its study result."""
    c = cases()[n]
    secs = (res.get("seconds") or {}).get("wall") if isinstance(res.get("seconds"), dict) else None
    row = {"what": c["what"], "ok": res.get("ok"), "reliable": res.get("reliable"), "seconds": secs}
    job = (res.get("execution") or {}).get("slurm_job") or res.get("job_id")
    if job:
        row["job_id"] = job
    if res.get("state") and res.get("state") != "done":
        row["state"] = res["state"]
    if not res.get("ok"):
        row.update({"passed": False, "why": res.get("why") or res.get("errors")})
        return row
    checks = []
    if n == "chain_feat_cantilever":
        ch = _chain_feat(res, folder or Path("."))
        if "skipped" in ch:
            return {"skipped": ch["skipped"], "what": c["what"]}
        checks = ch.get("checks", [])
        if not checks:
            row.update({"passed": False, "why": ch.get("why")})
            return row
    for chk in c.get("checks", []):
        q, ref, tol = chk[0], chk[1], chk[2]
        try:
            got = _get(res, q)
            if isinstance(ref, str):
                expv = _get(res, ref)
            elif isinstance(ref, tuple):
                expv = _get(res, ref[0]) + ref[1]
            else:
                expv = float(ref)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            checks.append({"quantity": q, "pass": False, "why": f"missing {exc}"})
            continue
        if tol is None:
            ok = abs(got - expv) <= chk[3]
            checks.append({"quantity": q, "got": got, "expected": expv, "abs": chk[3], "pass": ok})
        else:
            rel = abs(got - expv) / max(abs(expv), 1e-30)
            checks.append({"quantity": q, "got": got, "expected": expv, "tol": tol, "rel_error": float(f"{rel:.3g}"),
                           "pass": rel <= tol})
    row["checks"] = checks
    row["why_tol"] = c.get("why_tol")
    row["self_checks"] = res.get("checks")
    row["passed"] = all(x["pass"] for x in checks) and bool(res.get("reliable"))
    if not res.get("reliable"):
        row["unreliable_because"] = res.get("unreliable_because")
    return row


def run(names=None, folder: Path | None = None) -> dict:
    """Runs the benchmarks here, one after another (the local executor)."""
    studies, out = prepare(names)
    for n, st in studies.items():
        t0 = time.time()
        res = run_study(n, st, (folder or Path(".")) / n, 3600)
        row = evaluate(n, res, folder)
        row["seconds"] = round(time.time() - t0, 2)
        out[n] = row
    order = names or [k for k in cases() if k in out]
    return {k: out[k] for k in order if k in out} | {k: v for k, v in out.items() if k not in order}
