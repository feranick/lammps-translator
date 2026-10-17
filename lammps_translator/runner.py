"""runner.py — one study end to end: build, write inputs stage by stage, run LAMMPS, read, analyse, check.
Version: 2026.10.16.1

Folder layout (<request>/<study>/):
    initial.data, potentials/, 00_setup/ (in.lammps, log.lammps, forces0.dump), 01_<stage>/ ..., run_all.sh,
    <study>.json (the result), <study>.tgz (everything but trajectories)
progress.json tells api.status where a running study is.
"""
from __future__ import annotations

import json
import math
import shutil
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from . import analysis as A
from . import deck as D
from .build import System, build_system, read_data, write_data
from .data import AMU_A3_TO_G_CM3, EV_TO_KJ_MOL, UNITS
from .format import MD_KINDS, molecular, validate_study
from .lammps import lmp_info, missing_styles, run as lmp_run
from .potential import commands, model as make_model, resolve
from .refcalc import compute as ref_compute
from .results import read_ave_vector, read_dump, read_log, tags as tag_dict


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def structure_info(s: System, units: str | None = None) -> dict:
    lx, ly, lz, xy, xz, yz = s.box
    out = {"natoms": s.n, "composition": s.composition(), "types": list(s.type_names),
           "box_A": {"lx": A.r(lx), "ly": A.r(ly), "lz": A.r(lz), "xy": A.r(xy), "xz": A.r(xz), "yz": A.r(yz)},
           "volume_A3": A.r(s.volume()), "density_g_cm3": A.r(s.density(), 8), "masses": list(s.masses)}
    if s.mol is not None:
        out["nmolecules"] = s.nmolecules()
        out["bonds"], out["angles"], out["dihedrals"] = len(s.bonds), len(s.angles), len(s.dihedrals)
        out["net_charge_e"] = A.r(float(np.sum(s.charges)), 6)
    if s.lattice:
        out["lattice"] = {k: v for k, v in s.lattice.items() if v is not None}
    return out


def stage_names(stages: list) -> list[str]:
    seen, out = {}, []
    for st in stages:
        base = st.get("name") or st["kind"]
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


class Study:
    def __init__(self, name: str, study: dict, folder: Path, timeout: float, procs: int):
        self.name, self.study, self.folder, self.timeout, self.procs = name, study, folder, timeout, procs
        self.warnings: list[str] = []
        self.checks: dict = {}
        self.unreliable: list[str] = []
        self.t_start = time.time()
        self.lmp_seconds = 0.0
        self.res: dict = {"ok": False, "name": name, "translator": __version__}

    # ----------------------------------------------------------------------------- preparation
    def prepare(self):
        st = self.study
        self.s0 = build_system(st)
        self.warnings += self.s0.warnings
        r = resolve(st, self.s0)
        if r["errors"]:
            raise ValueError("; ".join(r["errors"]))
        self.warnings += r["warnings"]
        self.pot_files = r["files"]
        pot_names = {}
        if self.pot_files:
            (self.folder / "potentials").mkdir(exist_ok=True)
            for nm, path in self.pot_files.items():
                dst = self.folder / "potentials" / Path(path).name
                if not dst.exists():
                    shutil.copyfile(path, dst)
                pot_names[nm] = f"../potentials/{Path(path).name}"
        self.m = make_model(st, self.s0)
        self.units = self.m["units"]
        pre, post = commands(st, self.s0, self.m, pot_names)
        self.ctx = D.Ctx(study=st, model=self.m, pre=pre, post=post, seed=int(st.get("seed", 12345)), initial=self.s0)
        write_data(self.s0, self.folder / "initial.data")
        self.names = stage_names(st["stages"])
        self.needed_styles()

    def needed_styles(self):
        pair = next((ln.split()[1] for ln in self.ctx.post if ln.startswith("pair_style")), None)
        need = {"pair": [pair]}
        if any(ln.startswith("kspace_style") for ln in self.ctx.post):
            need["kspace"] = [next(ln.split()[1] for ln in self.ctx.post if ln.startswith("kspace_style"))]
        for ln in self.ctx.pre:
            if ln.split()[0] in ("bond_style", "angle_style", "dihedral_style"):
                need.setdefault(ln.split()[0].split("_")[0], []).append(ln.split()[1])
        miss = missing_styles(need)
        if miss:
            raise ValueError(f"this LAMMPS ({lmp_info().get('version')}) lacks {miss}: rebuild with the package "
                             "or choose another potential")

    # ----------------------------------------------------------------------------- running
    def progress(self, i: int, plan: list):
        (self.folder / "progress.json").write_text(json.dumps(
            {"current": i, "stages": plan, "started": self.t_start, "stage_started": time.time()}))

    def run_input(self, sid: str, lines: list[str]) -> dict:
        d = self.folder / sid
        d.mkdir(exist_ok=True)
        (d / "in.lammps").write_text("\n".join(lines) + "\n")
        remaining = max(10.0, self.timeout - (time.time() - self.t_start))
        r = lmp_run("in.lammps", "log.lammps", d, self.procs, remaining)
        self.lmp_seconds += r.get("seconds", 0.0)
        log = read_log(d / "log.lammps")
        if r["rc"] != 0 or log["errors"]:
            why = log["errors"][0] if log["errors"] else (r.get("error") or r.get("stderr") or f"exit code {r['rc']}")
            raise RuntimeError(f"LAMMPS failed in {sid}: {why[:600]}")
        log["dir"] = d
        return log

    def run(self) -> dict:
        st = self.study
        res = self.res
        self.prepare()
        res["structure"] = structure_info(self.s0)
        res["units"] = self.units
        info = lmp_info()
        res["lammps"] = {"version": info.get("version"), "exe": info.get("exe"), "procs": self.procs}
        plan = [{"id": "00_setup", "kind": "setup", "steps": 0}]
        for i, (sd, nm) in enumerate(zip(st["stages"], self.names)):
            steps = int(sd.get("steps", 0)) if sd["kind"] in MD_KINDS else (
                int(sd.get("steps", 20000)) * len(sd.get("T", [])) if sd["kind"] == "thermal_expansion" else 0)
            plan.append({"id": f"{i + 1:02d}_{nm}", "kind": sd["kind"], "steps": steps})
        self.plan = plan
        self.progress(0, plan)
        # setup + reference check
        su = D.setup(self.ctx, "../initial.data")
        log = self.run_input("00_setup", su["input"])
        tg = tag_dict(log)
        self.iso = {self.s0.type_names[k - 1]: tg[f"iso{k}"]["pe"] for k in su["meta"]["iso_types"] if f"iso{k}" in tg}
        res["initial"] = self.static_summary(tg["initial"], self.s0)
        self.reference_check(tg["initial"], self.folder / "00_setup" / "forces0.dump")
        d_l, d_o = float(tg["initial"].get("density", 0.0)), self.s0.density()
        self.checks["masses"] = {"density_lammps_g_cm3": A.r(d_l, 10), "density_ours_g_cm3": A.r(d_o, 10),
                                 "pass": abs(d_l - d_o) <= 1e-6 * d_o}
        if not self.checks["masses"]["pass"]:
            self.warnings.append(f"LAMMPS's density {d_l:.6g} g/cm3 differs from ours {d_o:.6g}: the potential file "
                                 "sets other masses")
        state = self.s0
        state_file = "../initial.data"
        for i, (sd, nm) in enumerate(zip(st["stages"], self.names)):
            sid = plan[i + 1]["id"]
            self.progress(i + 1, plan)
            (self.folder / sid).mkdir(exist_ok=True)
            out = self.write_stage(sd, state, state_file, i + 1, self.folder / sid)
            log = self.run_input(sid, out["input"])
            res[nm] = self.analyse(sd, nm, out, log, state)
            if out.get("writes_state"):
                state = self.read_state(self.folder / sid / "state.data", state)
                state_file = f"../{sid}/state.data"
        res["final"] = structure_info(state)
        fm = self.feat_material(res, state)
        if fm:
            res["feat_material"] = fm
        res["ok"] = True
        return res

    def write_stage(self, sd, state, state_file, index, d: Path) -> dict:
        k = sd["kind"]
        if k == "energy":
            return D.energy(self.ctx, sd, state, state_file)
        if k == "minimize":
            return D.minimize(self.ctx, sd, state, state_file)
        if k in MD_KINDS:
            return D.md(self.ctx, sd, state, state_file, index)
        if k == "elastic":
            return D.elastic(self.ctx, sd, state, state_file)
        if k == "eos":
            return D.eos(self.ctx, sd, state, state_file)
        if k == "vacancy":
            return D.vacancy(self.ctx, sd, state, state_file, d)
        if k == "surface":
            return D.surface(self.ctx, sd, state, state_file, d)
        if k == "thermal_expansion":
            return D.thermal_expansion(self.ctx, sd, state, state_file, index)
        raise ValueError(k)

    def read_state(self, path: Path, prev: System) -> System:
        s = read_data(path.read_text(), prev.atom_style)
        for k in ("type_names", "masses", "bond_types", "angle_types", "dihedral_types", "lattice"):
            setattr(s, k, getattr(prev, k))
        if hasattr(prev, "_ff"):
            s._ff = prev._ff
        return s

    # ----------------------------------------------------------------------------- analysis
    def static_summary(self, t: dict, s: System) -> dict:
        u = self.units
        ec = UNITS[u]["e_to_eV"]
        n = int(t["atoms"])
        out = {"pe_eV": A.r(t["pe"] * ec, 12), "pe_per_atom_eV": A.r(t["pe"] * ec / n, 12),
               "pressure_GPa": A.r(t["press"] * UNITS[u]["p_to_GPa"], 8),
               "stress_GPa": [A.r(x, 8) for x in A.stress_GPa(t, u)], "fmax_eV_A": A.r(t["fmax"] * ec, 6),
               "volume_A3": A.r(t["vol"], 12), "volume_per_atom_A3": A.r(t["vol"] / n, 12), "natoms": n}
        if self.m["molecular"] and s.mol is not None:
            nm = s.nmolecules()
            out["pe_per_molecule_kJ_mol"] = A.r(t["pe"] * ec * EV_TO_KJ_MOL / nm, 10)
            out["terms_kcal_mol"] = {k: A.r(t[k], 10) for k in ("evdwl", "ecoul", "elong", "ebond", "eangle", "edihed")}
        if getattr(self, "iso", None) and len(self.iso) == len(s.type_names) and s.n == n:
            e_iso = sum(self.iso[s.type_names[tt - 1]] for tt in s.types)
            out["E_coh_eV"] = A.r((e_iso - t["pe"]) * ec / n, 10)
        return out

    def lattice_constants(self, t: dict, s: System) -> dict:
        s0 = self.s0
        if not s0.lattice:
            return {}
        lat = s0.lattice
        a0 = lat["a"]
        f = (t["vol"] / s0.volume()) ** (1 / 3)
        out = {"a_A": A.r(a0 * f, 10)}
        if lat["kind"] == "hcp":
            out["a_A"] = A.r(a0 * t["lx"] / s0.box[0], 10)
            out["c_A"] = A.r(lat["c"] * t["lz"] / s0.box[2], 10)
            out["c_over_a"] = A.r(out["c_A"] / out["a_A"], 8)
        elif lat["kind"] != "custom":
            out["a_xyz_A"] = [A.r(a0 * t[k] / s0.box[i], 10) for i, k in enumerate(("lx", "ly", "lz"))]
        return out

    def reference_check(self, t: dict, dump: Path):
        m = self.m
        ref = ref_compute(m, self.s0, self.pot_files)
        if not ref.get("supported"):
            self.checks["reference"] = {"available": False, "why": ref.get("why")}
            return
        u = self.units
        n = self.s0.n
        ec = UNITS[u]["e_to_eV"]
        d = read_dump(dump)
        order = np.argsort(d["id"])
        F = np.column_stack([d["fx"][order], d["fy"][order], d["fz"][order]])
        dF = float(np.max(np.abs(F - ref["forces"]))) * ec
        dE = abs(t["pe"] - ref["pe"]) * ec / n
        P_l = np.array([t[k] for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")])
        dP = float(np.max(np.abs(P_l - np.array(ref["pressure"])))) * UNITS[u]["p_to_GPa"]
        coul = m.get("coulomb", "none")
        if coul == "long":
            acc = m["kspace_accuracy"]
            qq = UNITS[u]["qqr2e"] * ec
            qmax2 = max(1.0, float(np.max(np.abs(self.s0.charges))) ** 2)
            ecl = abs(ref["terms"].get("ecoul_long", 0.0)) * ec
            tolF = 10 * acc * qq * qmax2 + 1e-6
            tolE = 0.1 * acc * qq * qmax2 + 2e-6 * ecl / n + 1e-9
            self.kspace_tolE = 0.1 * acc * qq * qmax2
            tolP = 10 * tolE / (self.s0.volume() / n) * A.EV_A3_TO_GPA + 1e-6
            why = (f"long-range Coulomb: kspace accuracy {acc:g} (relative force error) and LAMMPS's polynomial erfc "
                   "(~1e-6 relative energy)")
        else:
            tolE, tolF = 1e-8, 1e-6
            tolP = 1e-6 + 1e-8 * float(np.max(np.abs(P_l))) * UNITS[u]["p_to_GPa"]
            why = "exact (same formulas and tables)"
        chk = {"available": True, "pe_lammps_eV": A.r(t["pe"] * ec, 12), "pe_ref_eV": A.r(ref["pe"] * ec, 12),
               "pe_diff_eV_per_atom": A.r(dE, 4), "max_force_diff_eV_A": A.r(dF, 4), "max_pressure_diff_GPa": A.r(dP, 4),
               "tolerances": {"pe_eV_per_atom": tolE, "force_eV_A": tolF, "pressure_GPa": A.r(tolP, 4), "why": why}}
        if m["molecular"]:
            terms = {"evdwl": t["evdwl"], "ecoul": t["ecoul"] + t["elong"], "ebond": t["ebond"], "eangle": t["eangle"],
                     "edihed": t["edihed"]}
            rt = ref["terms"]
            rc = {"evdwl": rt.get("evdwl", 0.0), "ecoul": rt.get("ecoul", 0.0) + rt.get("ecoul_long", 0.0),
                  "ebond": rt.get("ebond", 0.0), "eangle": rt.get("eangle", 0.0), "edihed": rt.get("edihed", 0.0)}
            chk["terms_diff_kcal_mol"] = {k: A.r(terms[k] - rc[k], 4) for k in terms}
        chk["pass"] = bool(dE <= tolE and dF <= tolF and dP <= tolP)
        self.checks["reference"] = chk
        if not chk["pass"]:
            self.unreliable.append("the reference implementation disagrees with LAMMPS on the initial structure "
                                   "(checks.reference): the input does not mean what the study says")

    def analyse(self, sd: dict, nm: str, out: dict, log: dict, state: System) -> dict:
        k = sd["kind"]
        tg = tag_dict(log)
        u = self.units
        meta = out["meta"]
        if k == "energy":
            return self.static_summary(tg["E"], state)
        if k == "minimize":
            t = tg["final"]
            r = self.static_summary(t, state)
            r.update(self.lattice_constants(t, state))
            r["stop"] = log["stops"]
            r["density_g_cm3"] = A.r(state.total_mass() / t["vol"] * AMU_A3_TO_G_CM3, 8)
            if meta["cycles"]:
                r["cycle_pressures_GPa"] = [A.r(tg[f"cycle{c + 1}"]["press"] * UNITS[u]["p_to_GPa"], 6)
                                            for c in range(meta["cycles"])]
                target = float(sd.get("P_bar", 0.0)) * 1e-4
                dp = abs(r["pressure_GPa"] - target) if sd.get("box") == "iso" else max(
                    abs(-x - target) for x in r["stress_GPa"][:3])
                r["pressure_error_GPa"] = A.r(dp, 4)
                if dp > 0.01:
                    self.unreliable.append(f"{nm}: box relaxation ended {dp:.3g} GPa from the target pressure")
                elif dp > 1e-4:
                    self.warnings.append(f"{nm}: box relaxation ended {dp:.2g} GPa from the target pressure")
            if r["fmax_eV_A"] > 1e-3:
                self.warnings.append(f"{nm}: largest force {r['fmax_eV_A']:.2g} eV/A after minimization "
                                     f"(stop: {log['stops'][-1] if log['stops'] else '?'})")
            if r["fmax_eV_A"] > 0.05:
                self.unreliable.append(f"{nm}: minimization did not converge (largest force {r['fmax_eV_A']:.2g} eV/A)")
            return r
        if k in MD_KINDS:
            blk = max(log["blocks"], key=lambda b: len(b["rows"]))
            r = A.md_summary(blk, meta, u, state.n, state.nmolecules(), state.total_mass())
            r["final"] = self.static_summary(tg["final"], state)
            r["final"].update(self.lattice_constants(tg["final"], state))
            mean, err = r["mean"], r["err"]
            if self.s0.lattice and k == "npt":
                f = (mean["volume_A3"] / self.s0.volume()) ** (1 / 3)
                r["mean"]["a_A"] = A.r(self.s0.lattice["a"] * f, 10)
            if k in ("nvt", "npt"):
                Tt = 0.5 * (meta["T"] + meta["T_end"])
                if abs(mean["T_K"] - Tt) > max(0.05 * Tt, 3 * (err["T_K"] or 0)):
                    self.warnings.append(f"{nm}: mean T {mean['T_K']:.4g} K vs target {Tt:g} K: equilibrate longer")
            if k == "npt":
                Pt = float(sd.get("P_bar", 0.0))
                if err["P_bar"] and abs(mean["P_bar"] - Pt) > 4 * err["P_bar"] + 50:
                    self.warnings.append(f"{nm}: mean P {mean['P_bar']:.4g} bar vs target {Pt:g} bar (err "
                                         f"{err['P_bar']:.2g}): run longer")
            if k == "nve" and r.get("etotal_slope_eV_per_atom_ps") is not None:
                d = abs(r["etotal_slope_eV_per_atom_ps"])
                self.checks.setdefault("energy_drift", {})[nm] = {"eV_per_atom_ps": r["etotal_slope_eV_per_atom_ps"],
                                                                 "pass": d < 1e-3}
                if d > 1e-3:
                    self.warnings.append(f"{nm}: total energy drifts {d:.2g} eV/atom/ps: smaller dt_fs")
                if d > 1e-2:
                    self.unreliable.append(f"{nm}: energy not conserved ({d:.2g} eV/atom/ps)")
            if "rdf" in meta:
                f = log["dir"] / "rdf.txt"
                r["rdf"] = A.rdf_summary(read_ave_vector(f) if f.is_file() else [], meta["rdf"])
            for key in list(r["err"]):
                if r["err"][key] is None:
                    r["err"].pop(key)
            return r
        if k == "elastic":
            r = A.elastic_constants(tg, meta["strain"], u)
            r["strain"] = meta["strain"]
            r["relax"] = meta["relax"]
            Cmax = max(abs(x) for row in r["C_GPa"] for x in row)
            ok_sym = r["asymmetry_GPa"] <= 0.02 * Cmax + 0.05
            chk = {"symmetry": ok_sym, "born_stable": r["born_stable"]}
            P = r["reference_pressure_GPa"]
            if abs(P) > 0.1:
                self.warnings.append(f"{nm}: the cell is at {P:.3g} GPa: these are stress-strain coefficients at that "
                                     "pressure (minimize with box: iso first for zero-pressure constants)")
            if self.m["kind"] in ("lj", "morse", "buck", "coul") and not self.m["molecular"] and self.s0.lattice \
                    and self.s0.lattice["kind"] in ("fcc", "bcc", "sc", "rocksalt", "cscl") and abs(P) < 0.01 \
                    and not self.study["system"].get("substitute"):
                c = r["cubic"]
                chk["cauchy"] = {"C12_minus_C44_GPa": c["cauchy_C12_minus_C44_GPa"],
                                 "pass": abs(c["cauchy_C12_minus_C44_GPa"] - 2 * P) <= 1e-3 * c["C11_GPa"] + 1e-3}
                if not chk["cauchy"]["pass"]:
                    self.unreliable.append(f"{nm}: pair potential breaks the Cauchy relation C12 = C44 (+2P)")
            if not ok_sym:
                self.warnings.append(f"{nm}: C is not symmetric to 2 % (asymmetry {r['asymmetry_GPa']:.3g} GPa): "
                                     "relax the cell first or change strain")
            self.checks[nm] = chk
            return r
        if k == "eos":
            n = state.n
            ec = UNITS[u]["e_to_eV"]
            pts = [tg[f"v{j}"] for j in range(len(meta["scales"]))]
            V = np.array([t["vol"] / n for t in pts])
            E = np.array([t["pe"] * ec / n for t in pts])
            P = np.array([t["press"] * UNITS[u]["p_to_GPa"] for t in pts])
            fit = A.birch_murnaghan(V, E)
            r = {"points": [{"V_per_atom_A3": A.r(v, 10), "E_per_atom_eV": A.r(e, 12), "P_GPa": A.r(p, 8)}
                            for v, e, p in zip(V, E, P)], **{k2: v for k2, v in fit.items() if k2 != "P_fit_GPa"}}
            chk = A.work_check(np.array(meta["scales"]), V, E, P, 2 * getattr(self, "kspace_tolE", 0.0))
            self.checks[nm] = chk
            if not chk["pass"]:
                self.unreliable.append(f"{nm}: the energy change between volumes differs from the work of LAMMPS's "
                                       f"pressure (-integral P dV) by {chk['max_rel_mismatch']:.2g} of the curve's span: "
                                       "energy jumps at the cutoff (use shift: true or a longer cutoff) or an "
                                       "inconsistent virial")
            elif not chk["smooth"]:
                self.warnings.append(f"{nm}: pressure has kinks ({chk['max_rel_mismatch']:.2g} of the span between "
                                     "energy and -integral P dV): neighbour shells cross a cutoff where forces are not "
                                     "zero; B0' is less certain (use a smaller strain)")
            if fit.get("ok"):
                r["fit_max_pressure_error_GPa"] = A.r(float(np.max(np.abs(P - np.array(fit["P_fit_GPa"])))), 4)
                if self.s0.lattice:
                    nb = self.s0.n / self.s0.volume()
                    r["a0_A"] = A.r(self.s0.lattice["a"] * (fit["V0_A3"] * nb) ** (1 / 3), 10)
            return r
        if k == "vacancy":
            ec = UNITS[u]["e_to_eV"]
            Ep, Ev = tg["perfect"]["pe"] * ec, tg["vacancy"]["pe"] * ec
            N = meta["n"]
            r = {"E_f_eV": A.r(Ev - (N - 1) / N * Ep, 10), "E_perfect_eV": A.r(Ep, 12), "E_vacancy_eV": A.r(Ev, 12),
                 "natoms": N, "removed_atom": meta["removed_atom"], "element": meta["element"], "relax": meta["relax"],
                 "note": "constant volume; E_f = E(N-1) - (N-1)/N E(N)"}
            if len(state.type_names) > 1:
                r["note"] += "; for compounds this uses the mean energy per atom, not chemical potentials"
            if N < 100:
                self.warnings.append(f"{nm}: {N} atoms is small for a vacancy (use size >= 4x4x4 cells)")
            return r
        if k == "surface":
            A2 = meta["area_A2"]
            eb = tg["bulk"]["pe"]
            r = {"miller": meta["miller"], "orient": meta["orient"], "area_A2": A.r(A2, 10), "natoms": meta["n"],
                 "thickness_A": A.r(meta["thickness_A"], 8), "vacuum_A": meta["vacuum"],
                 "gamma_unrelaxed_J_m2": A.r(A.surface_energy(tg["slab_unrelaxed"]["pe"], eb, A2, u), 8)}
            if "slab" in tg:
                r["gamma_J_m2"] = A.r(A.surface_energy(tg["slab"]["pe"], eb, A2, u), 8)
                r["gamma_eV_A2"] = A.r(r["gamma_J_m2"] / 16.02176634, 8)
                if r["gamma_J_m2"] > r["gamma_unrelaxed_J_m2"] + 1e-6:
                    self.warnings.append(f"{nm}: relaxed surface energy above the unrelaxed one")
            else:
                r["gamma_J_m2"] = r["gamma_unrelaxed_J_m2"]
            return r
        if k == "thermal_expansion":
            pts = []
            n = state.n
            blocks = [b for b in log["blocks"] if len(b["rows"]) > 5]
            for j, T in enumerate(meta["T"]):
                b = blocks[j]
                sm = A.md_summary(b, meta | {"T": T, "T_end": T}, u, n, state.nmolecules(), state.total_mass())
                V = sm["mean"]["volume_A3"]
                f = (V / self.s0.volume()) ** (1 / 3)
                a = self.s0.lattice["a"] * f if self.s0.lattice else f
                pts.append({"T_K": T, "T_mean_K": sm["mean"]["T_K"], "volume_per_atom_A3": A.r(V / n, 10),
                            "a_A": A.r(a, 10), "P_GPa": sm["mean"]["P_GPa"], "err_volume_A3": sm["err"].get("volume_A3")})
            T = np.array([p["T_K"] for p in pts])
            a = np.array([p["a_A"] for p in pts])
            vv = np.array([p["volume_per_atom_A3"] for p in pts])
            sa = np.polyfit(T, a, 1)[0]
            sv = np.polyfit(T, vv, 1)[0]
            r = {"points": pts, "alpha_linear_1_K": A.r(sa / np.mean(a), 6), "alpha_volume_1_K": A.r(sv / np.mean(vv), 6),
                 "note": "classical MD: no quantum (zero-point) effects; linear fit over the temperatures"}
            return r
        return {}

    def feat_material(self, res: dict, state: System) -> dict | None:
        if self.m["molecular"]:
            return None
        el = [nm for sd, nm in zip(self.study["stages"], self.names) if sd["kind"] == "elastic"]
        if not el:
            return None
        e = res[el[-1]]
        out = {"E": A.r(e["E_GPa"] * 1000.0, 8), "nu": A.r(e["nu"], 6), "density": A.r(state.density() * 1000.0, 8)}
        te = [nm for sd, nm in zip(self.study["stages"], self.names) if sd["kind"] == "thermal_expansion"]
        if te:
            out["alpha"] = res[te[-1]]["alpha_linear_1_K"]
        self.res["feat_material_note"] = ("isotropic (Voigt-Reuss-Hill) from " + el[-1] + "; density of the final "
                                          "state; MPa, kg/m3, 1/K as fea-translator expects (paste as its 'material')")
        return out


def finish(res: dict, st: Study, folder: Path, name: str, study: dict) -> dict:
    res["checks"] = st.checks
    res["warnings"] = list(dict.fromkeys(st.warnings))
    exp = []
    for e in study.get("expect") or []:
        v = res
        try:
            for part in e["quantity"].split("."):
                v = v[int(part)] if isinstance(v, list) else v[part]
            v = float(v)
        except (KeyError, IndexError, ValueError, TypeError):
            exp.append({**e, "pass": False, "why": "no such result"})
            continue
        if "abs" in e:
            ok = abs(v - e["value"]) <= e["abs"]
        else:
            ok = abs(v - e["value"]) <= float(e.get("tol", 0.02)) * max(abs(e["value"]), 1e-30)
        exp.append({**e, "got": v, "rel_error": A.r((v - e["value"]) / e["value"], 4) if e["value"] else None, "pass": ok})
    if exp:
        res["expect"] = exp
    res["reliable"] = bool(res.get("ok")) and not st.unreliable
    if st.unreliable:
        res["unreliable_because"] = st.unreliable
    res["seconds"] = {"lammps": round(st.lmp_seconds, 3), "wall": round(time.time() - st.t_start, 3)}
    # standalone rerun script and archive
    stages = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name[:2].isdigit())
    sh = ["#!/bin/sh", "# reruns this study's LAMMPS inputs in order (lammps-translator " + __version__ + ")",
          'LMP="${LMP:-lmp}"', "set -e", 'cd "$(dirname "$0")"']
    sh += [f'(cd {s} && $LMP -in in.lammps -log log.lammps -screen none)' for s in stages]
    (folder / "run_all.sh").write_text("\n".join(sh) + "\n")
    (folder / "run_all.sh").chmod(0o755)
    tgz = folder.parent / f"{name}.tgz"
    try:
        with tarfile.open(tgz, "w:gz") as tf:
            for p in sorted(folder.rglob("*")):
                if p.is_file() and not p.name.endswith(".lammpstrj"):
                    tf.add(p, arcname=str(Path(name) / p.relative_to(folder)))
    except OSError:
        tgz = None
    res["files"] = {"folder": str(folder), "archive": str(tgz) if tgz else None, "run_all": str(folder / "run_all.sh"),
                    "inputs": [str(folder / s / "in.lammps") for s in stages],
                    "final_state": next((str(folder / s / "state.data") for s in reversed(stages)
                                         if (folder / s / "state.data").is_file()), str(folder / "initial.data"))}
    return _jsonable(res)


def run_study(name: str, study: dict, folder: Path, timeout: float = 3600.0, procs: int | None = None) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    v = validate_study(study)
    if not v["ok"]:
        out = {"ok": False, "name": name, "why": "invalid study", "errors": v["errors"], "warnings": v["warnings"]}
        (folder / f"{name}.json").write_text(json.dumps(out, indent=1))
        return out
    import os
    np_ = procs or int(study.get("procs", os.environ.get("LMPT_PROCS", "1")))
    st = Study(name, study, folder, timeout, np_)
    st.warnings += v["warnings"]
    res = st.res
    try:
        res = st.run()
    except Exception as exc:  # noqa: BLE001
        res.update({"ok": False, "why": str(exc)[:1500]})
        if not isinstance(exc, (ValueError, RuntimeError)):
            res["trace"] = traceback.format_exc()[-3000:]
        if "structure" not in res and hasattr(st, "s0"):
            res["structure"] = structure_info(st.s0)
    out = finish(res, st, folder, name, study)
    (folder / f"{name}.json").write_text(json.dumps(out, indent=1))
    try:
        (folder / "progress.json").unlink()
    except OSError:
        pass
    return out


def inputs(name: str, study: dict, folder: Path) -> dict:
    """Builds the study and writes every stage's input against the initial structure (no LAMMPS run)."""
    folder.mkdir(parents=True, exist_ok=True)
    st = Study(name, study, folder, 0, 1)
    st.prepare = st.prepare  # noqa
    st.s0 = build_system(study)
    r = resolve(study, st.s0)
    if r["errors"]:
        return {"error": "; ".join(r["errors"])}
    st.pot_files = r["files"]
    pot_names = {nm: f"../potentials/{Path(p).name}" for nm, p in st.pot_files.items()}
    st.m = make_model(study, st.s0)
    pre, post = commands(study, st.s0, st.m, pot_names)
    st.ctx = D.Ctx(study=study, model=st.m, pre=pre, post=post, seed=int(study.get("seed", 12345)), initial=st.s0)
    write_data(st.s0, folder / "initial.data")
    names = stage_names(study["stages"])
    texts = {"00_setup/in.lammps": "\n".join(D.setup(st.ctx, "../initial.data")["input"])}
    for i, (sd, nm) in enumerate(zip(study["stages"], names)):
        d = folder / f"{i + 1:02d}_{nm}"
        d.mkdir(exist_ok=True)
        out = st.write_stage(sd, st.s0, "../initial.data" if i == 0 else f"../{i:02d}_{names[i - 1]}/state.data", i + 1, d)
        text = "\n".join(out["input"])
        (d / "in.lammps").write_text(text + "\n")
        texts[f"{d.name}/in.lammps"] = text if len(text) < 12000 else text[:12000] + "\n# ... (truncated)"
    di = (folder / "initial.data").read_text()
    return {"folder": str(folder), "structure": structure_info(st.s0), "inputs": texts,
            "initial_data_head": "\n".join(di.splitlines()[:40]),
            "note": "stages after the first are written against the initial structure here; in a run each is "
                    "written from the state the previous stage leaves"}
