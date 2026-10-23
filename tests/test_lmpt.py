"""Tests for lammps-translator. Version: 2026.10.22.1
Most run anywhere (builder, validation, reference implementation, readers, analysis); the ones marked
'needs LAMMPS' are skipped when no executable is found (set LMPT_LMP)."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from lammps_translator import analysis as A
from lammps_translator import api, bench, deck
from lammps_translator.build import build_system, oriented_cell, read_data, write_data
from lammps_translator.format import study_format, validate_study
from lammps_translator.lammps import find_lmp
from lammps_translator.potential import find_file, model
from lammps_translator.refcalc import compute, neighbours
from lammps_translator.results import read_log

HERE = Path(__file__).parent
EXAMPLES = json.loads((HERE.parent / "examples" / "studies.json").read_text())
needs_lammps = pytest.mark.skipif(find_lmp() is None, reason="LAMMPS not found (set LMPT_LMP)")
needs_cu = pytest.mark.skipif(find_file("Cu_u3.eam") is None, reason="LAMMPS potentials folder not found")

LJ = {"system": {"lattice": "fcc", "a": 5.3, "elements": ["Ar"], "size": [3, 3, 3]},
      "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 8.5}, "stages": [{"kind": "energy"}]}


def test_format_compact_and_examples_valid():
    f = study_format()
    assert len(json.dumps(f)) < 16000          # local LLMs read it whole
    for name, st in {**f["example"], **EXAMPLES}.items():
        v = validate_study(st)
        if "potential" in st and st["potential"].get("style") not in ("lj", "morse", "buck", "coul") \
                and find_file(st["potential"].get("file") or "x") is None and st["potential"].get("files") is None:
            continue   # file potentials need the potentials folder
        assert v["ok"], (name, v)


@pytest.mark.parametrize("bad, words", [
    ({**LJ, "system": {"lattice": "fcx", "a": 3, "elements": ["Ar"]}}, "not one of"),
    ({**LJ, "system": {"lattice": "fcc", "elements": ["Ar"]}}, "system.a"),
    ({**LJ, "system": {"lattice": "rocksalt", "a": 5, "elements": ["Na"]}}, "2 name(s)"),
    ({**LJ, "stages": [{"kind": "nvt", "steps": 100}]}, "T: temperature"),
    ({**LJ, "stages": [{"kind": "minimize", "boxx": "iso"}]}, "unknown keys"),
    ({**LJ, "potential": {**LJ["potential"], "tail": True}, "stages": [{"kind": "minimize", "box": "iso"}]}, "tail"),
    ({**LJ, "stages": [{"kind": "surface", "miller": [0, 0, 0]}]}, "miller"),
    ({**LJ, "potential": {"style": "buck", "pairs": {"Ar-Ar": {"A": 1, "rho": 0.3, "C": 1}}, "cutoff": 8,
                          "charges": {"Ar": 1}}}, "net charge"),
])
def test_validation_messages(bad, words):
    v = validate_study(bad)
    assert not v["ok"]
    assert any(words in e for e in v["errors"]), v["errors"]


def test_lattices_counts_and_neutrality():
    for kind, els, nb in (("fcc", ["Cu"], 4), ("bcc", ["Fe"], 2), ("diamond", ["Si"], 8), ("hcp", ["Ti"], 4),
                          ("rocksalt", ["Na", "Cl"], 8), ("cscl", ["Cs", "Cl"], 2), ("zincblende", ["Ga", "As"], 8),
                          ("fluorite", ["Ca", "F"], 12), ("l12", ["Ni", "Al"], 4)):
        s = build_system({"system": {"lattice": kind, "a": 4.0, "elements": els, "size": [2, 2, 2]},
                          "potential": {"style": "lj", "pairs": {f"{a}-{b}": {"eps": 0.01, "sigma": 2.0} for a in els for b in els},
                                        "cutoff": 5.0}, "stages": [{"kind": "energy"}]})
        assert s.n == 8 * nb, kind
        # no two atoms closer than 0.3 a (no duplicates)
        i, j, rij, r = neighbours(s.pos, s.box, 1.0)
        assert len(r) == 0, kind
    s = build_system({"system": {"lattice": "fluorite", "a": 5.46, "elements": ["Ca", "F"], "size": [2, 2, 2]},
                      "potential": {"style": "coul", "charges": {"Ca": 2, "F": -1}, "cutoff": 6}, "stages": [{"kind": "energy"}]})
    assert abs(s.charges.sum()) < 1e-12 and s.composition() == {"Ca": 32, "F": 64}


def test_orient_cells():
    sysd = {"lattice": "fcc", "a": 1.0, "elements": ["Cu"]}
    L, pos, _ = oriented_cell(sysd, {"x": [1, -1, 0], "y": [1, 1, -2], "z": [1, 1, 1]})
    assert len(pos) * 1 == round(L[0] * L[1] * L[2] * 4)
    assert np.allclose(L, [math.sqrt(2) / 2, math.sqrt(6) / 2, math.sqrt(3)])
    # the oriented cell is the same crystal: every atom has 12 neighbours at a/sqrt(2)
    box = [L[0] * 3, L[1] * 2, L[2] * 2, 0, 0, 0]
    P = np.concatenate([pos + np.array([i * L[0], j * L[1], k * L[2]]) for i in range(3) for j in range(2) for k in range(2)])
    i, j, rij, r = neighbours(P, box, 0.75)
    assert np.allclose(np.bincount(i), 12) and np.allclose(r, 1 / math.sqrt(2))
    o = deck.surface_orient([1, 1, 0])
    assert np.dot(o["x"], o["z"]) == 0 and np.dot(o["y"], o["z"]) == 0


def test_molecule_packing_and_topology():
    st = EXAMPLES["water_spce"]
    s = build_system(st)
    assert s.n == 3 * 216 and s.nmolecules() == 216
    assert abs(s.density() - 0.997) < 1e-9
    assert len(s.bonds) == 432 and len(s.angles) == 216 and s.bond_types == ["HW-OW"] or s.bond_types == ["OW-HW"]
    assert abs(s.charges.sum()) < 1e-9


def test_data_roundtrip(tmp_path):
    s = build_system(EXAMPLES["butane_ua"])
    write_data(s, tmp_path / "d.data")
    t = read_data((tmp_path / "d.data").read_text(), "full")
    assert t.n == s.n and len(t.dihedrals) == len(s.dihedrals) == 64
    assert np.allclose(t.pos, s.pos % np.array(s.box[:3]), atol=1e-9)


def _num_forces(m, s, k, h=1e-5):
    out = []
    for a in k:
        row = []
        for d in range(3):
            sp, sm = s.copy(), s.copy()
            sp.pos[a, d] += h
            sm.pos[a, d] -= h
            row.append(-(compute(m, sp)["pe"] - compute(m, sm)["pe"]) / (2 * h))
        out.append(row)
    return np.array(out)


def test_reference_lj_closed_form_and_forces():
    cf = bench.lj_fcc_closed_form(0.0104, 3.405, 3 * 3.405)
    st = {**LJ, "system": {"lattice": "fcc", "a": cf["a"], "elements": ["Ar"], "size": [4, 4, 4]},
          "potential": {"style": "lj", "eps": 0.0104, "sigma": 3.405, "cutoff": 3 * 3.405}}
    s = build_system(st)
    m = model(st, s)
    r = compute(m, s)
    assert abs(r["pe"] / s.n - cf["E"]) < 1e-12
    assert abs(r["press"]) < 1e-6           # zero pressure at the lattice-sum equilibrium
    s.pos[0] += [0.1, -0.05, 0.07]
    r = compute(m, s)
    assert np.allclose(_num_forces(m, s, [0, 1]), r["forces"][[0, 1]], atol=1e-8)


def test_reference_ewald_madelung_and_forces():
    st = {"system": {"lattice": "rocksalt", "a": 5.64, "elements": ["Na", "Cl"], "size": [2, 2, 2]},
          "potential": {"style": "coul", "charges": {"Na": 1, "Cl": -1}, "cutoff": 5.0}, "stages": [{"kind": "energy"}]}
    s = build_system(st)
    m = model(st, s)
    r = compute(m, s)
    exact = -bench.MADELUNG["rocksalt"] * 14.399645 / 2.82 / 2
    assert abs(r["pe"] / s.n - exact) < 1e-9
    s.pos[3] += [0.05, 0.02, -0.04]
    r = compute(m, s)
    assert np.allclose(_num_forces(m, s, [3, 7]), r["forces"][[3, 7]], atol=1e-6)


@needs_cu
def test_reference_eam_forces_and_pressure():
    st = {"system": {"lattice": "fcc", "a": 3.6, "elements": ["Cu"], "size": [3, 3, 3], "perturb": 0.05},
          "potential": {"style": "eam", "file": "Cu_u3.eam"}, "stages": [{"kind": "energy"}]}
    s = build_system(st)
    m = model(st, s)
    files = {"Cu_u3.eam": str(find_file("Cu_u3.eam"))}
    r = compute(m, s, files)
    num = []
    for d in range(3):
        sp, sm = s.copy(), s.copy()
        sp.pos[5, d] += 1e-5
        sm.pos[5, d] -= 1e-5
        num.append(-(compute(m, sp, files)["pe"] - compute(m, sm, files)["pe"]) / 2e-5)
    assert np.allclose(num, r["forces"][5], atol=1e-6)
    # pressure = -dE/dV under isotropic scaling
    f = 1e-6
    def scaled(x):
        t = s.copy(); t.pos = t.pos * x; t.box = [b * x for b in t.box]; return compute(m, t, files)["pe"]
    V = s.volume()
    dEdV = (scaled(1 + f) - scaled(1 - f)) / (V * ((1 + f) ** 3 - (1 - f) ** 3))
    assert abs(-dEdV * 1.6021765e6 - r["press"]) < 1e-3 * abs(r["press"]) + 1.0


def test_reference_bonded_terms():
    s = build_system(EXAMPLES["butane_ua"])
    m = model(EXAMPLES["butane_ua"], s)
    r = compute(m, s)
    assert set(r["terms"]) >= {"ebond", "eangle", "edihed", "evdwl"}
    s.pos[1] += [0.03, 0.02, -0.01]
    r = compute(m, s)
    assert np.allclose(_num_forces(m, s, [1]), r["forces"][[1]], atol=1e-6)


def test_strained_box_and_birch_murnaghan():
    b = deck.strained_box([10, 11, 12, 0.5, 0.2, 0.1], 6, 0.01)
    assert b[3] == pytest.approx(0.5 + 0.11) and b[4] == pytest.approx(0.2 + 0.001)
    V0, E0, B0, Bp = 12.0, -3.5, 0.8, 4.5      # eV/A^3
    V = np.linspace(11, 13, 9)
    eta = (V0 / V) ** (2 / 3)
    E = E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))
    f = A.birch_murnaghan(V, E)
    assert f["V0_A3"] == pytest.approx(V0, rel=1e-9) and f["B0_GPa"] == pytest.approx(B0 * 160.21766208, rel=1e-7)
    assert f["B0_prime"] == pytest.approx(Bp, rel=1e-6)


def test_elastic_from_synthetic_tags():
    C = np.array([[170, 120, 120, 0, 0, 0], [120, 170, 120, 0, 0, 0], [120, 120, 170, 0, 0, 0],
                  [0, 0, 0, 75, 0, 0], [0, 0, 0, 0, 75, 0], [0, 0, 0, 0, 0, 75]], float)
    eps = 1e-3
    tags = {"ref": {k: 0.0 for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")}}
    for v in range(1, 7):
        for sgn, t in ((1, "p"), (-1, "m")):
            e = np.zeros(6); e[v - 1] = sgn * eps
            sig = C @ e                                     # GPa
            tags[f"s{v}{t}"] = dict(zip(("pxx", "pyy", "pzz", "pyz", "pxz", "pxy"), -sig / 1e-4))   # bar
    r = A.elastic_constants(tags, eps, "metal")
    assert r["C11_GPa"] == pytest.approx(170) and r["C44_GPa"] == pytest.approx(75) and r["born_stable"]
    assert r["B_GPa"] == pytest.approx((170 + 240) / 3)


def test_log_reader(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text("""LAMMPS (4 Jul 2026)
   Step          Temp          PotEng         Press          Volume
         0   0             -9.0974940744  -2.1e-12       3966.2201
       100   12.5          -9.05          120.5          3966.2201
Loop time of 0.0123 on 1 procs for 100 steps with 108 atoms
  Stopping criterion = force tolerance
WARNING: something (src/x.cpp:1)
LMPT final pe -9.05 press 120.5 atoms 108
ERROR: Unknown command: foo (src/input.cpp:314)
Last input line: foo
""")
    r = read_log(log)
    assert r["version"] == "4 Jul 2026" and r["blocks"][0]["columns"] == ["step", "temp", "pe", "press", "vol"]
    assert r["blocks"][0]["rows"].shape == (2, 5) and r["loops"][0]["steps"] == 100
    assert r["tags"][0] == ("final", {"pe": -9.05, "press": 120.5, "atoms": 108.0})
    assert r["stops"] == ["force tolerance"] and r["errors"] and r["warnings"]


def test_inputs_without_lammps(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    r = api.inputs(EXAMPLES["ar_lj_md"], "ar")
    assert "error" not in r, r
    txt = "\n".join(r["inputs"].values())
    assert "pair_style lj/cut" in txt and "fix int all npt" in txt and "LMPT final" in txt


# ------------------------------------------------------------------------------------------- needs LAMMPS
@needs_lammps
def test_bench_default_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    names = ["lj_fcc_lattice_sums", "madelung_nacl", "cauchy_mgo", "equipartition_lj", "nve_conservation"]
    if find_file("Cu_u3.eam"):
        names += ["eam_cu_foiles", "tersoff_si"]
    r = api.bench(names)
    assert not r["failed"], json.dumps(r["cases"], indent=1)[:3000]


@needs_lammps
def test_examples_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    pick = {k: v for k, v in EXAMPLES.items() if k in ("ar_lj_md", "mgo_buckingham", "butane_ua")}
    r = api.run(pick)
    for n, v in r["studies"].items():
        assert v["ok"], (n, v.get("why"))
        assert v["checks"]["reference"]["pass"], (n, v["checks"]["reference"])
        assert all(e["pass"] for e in v.get("expect", [])), (n, v.get("expect"))


@needs_lammps
def test_pending_then_status(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    st = {**LJ, "potential": {**LJ["potential"], "shift": True},
          "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "nvt", "T": 30, "steps": 60000}]}
    r = api.run({"slow": st}, wait=0.5)
    if r["done"]:
        pytest.skip("finished too fast to see pending")
    assert r["pending"] == ["slow"] and "progress" in r and "next" in r
    r = api.status(r["request"], wait=120)
    assert r["done"] and r["studies"]["slow"]["ok"]


@needs_lammps
def test_lammps_error_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("LMPT_OUT", str(tmp_path))
    st = {**LJ, "stages": [{"kind": "nvt", "T": 1e7, "steps": 2000, "dt_fs": 5}]}
    r = api.run({"hot": st})["studies"]["hot"]
    assert r["ok"] is False or r["reliable"] is False or r["warnings"]


@needs_lammps
@needs_cu
def test_rotated_cell_elastic_constants(tmp_path):
    from lammps_translator.runner import run_study
    base = {"potential": {"style": "eam", "file": "Cu_u3.eam"}, "stages": [{"kind": "minimize", "box": "aniso"}, {"kind": "elastic"}]}
    c = run_study("c", {**base, "system": {"lattice": "fcc", "a": 3.615, "elements": ["Cu"], "size": [3, 3, 3]}}, tmp_path / "c")
    o = run_study("o", {**base, "system": {"lattice": "fcc", "a": 3.615, "elements": ["Cu"], "size": [2, 2, 2],
                                           "orient": {"x": [1, -1, 0], "y": [1, 1, -2], "z": [1, 1, 1]}}}, tmp_path / "o")
    k = c["elastic"]["cubic"]
    C11, C12, C44 = k["C11_GPa"], k["C12_GPa"], k["C44_GPa"]
    assert o["elastic"]["C11_GPa"] == pytest.approx((C11 + C12 + 2 * C44) / 2, rel=3e-3)
    assert o["elastic"]["C33_GPa"] == pytest.approx((C11 + 2 * C12 + 4 * C44) / 3, rel=3e-3)
    assert c["checks"]["masses"]["pass"] and c["reliable"] and o["reliable"]
    ex = c["execution"]
    assert {k: ex[k] for k in ("executor", "slurm_job", "procs_requested", "procs_used", "mpi")} == \
        {"executor": "local", "slurm_job": None, "procs_requested": 1, "procs_used": 1, "mpi": None}
    assert ex["limits"]["from_study"] == [] and ex["limits"]["mem"] is None


def test_lammps_not_found_message(tmp_path, monkeypatch):
    from lammps_translator.runner import run_study
    monkeypatch.setenv("LMPT_LMP", str(tmp_path / "no_such_lmp"))
    r = run_study("x", LJ, tmp_path / "x")
    assert r["ok"] is False and "LMPT_LMP" in r["why"] and "not an executable" in r["why"]
