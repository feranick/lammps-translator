"""data.py — tables: element masses, lattices, built-in molecules (with their force fields), unit constants.
Version: 2026.10.25.2
"""
from __future__ import annotations

import math

# LAMMPS's own constants (src/update.cpp, force.cpp), so derived numbers match the program's.
UNITS = {
    "metal": {"boltz": 8.617343e-5, "nktv2p": 1.6021765e6, "qqr2e": 14.399645, "mvv2e": 1.0364269e-4,
              "energy": "eV", "pressure": "bar", "p_to_GPa": 1e-4, "e_to_eV": 1.0, "time_per_fs": 1e-3},
    "real": {"boltz": 0.0019872067, "nktv2p": 68568.415, "qqr2e": 332.06371, "mvv2e": 48.88821291 ** 2,
             "energy": "kcal/mol", "pressure": "atm", "p_to_GPa": 1.01325e-4, "e_to_eV": 4.184 / 96.4853321,
             "time_per_fs": 1.0},
}
EV_TO_KJ_MOL = 96.4853321
EV_A2_TO_J_M2 = 16.02176634
AMU_A3_TO_G_CM3 = 1.66053906660
AVOGADRO = 6.02214076e23

# Standard atomic weights (IUPAC, conventional values), g/mol.
MASSES = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403163, "Ne": 20.1797, "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078, "Sc": 44.955908,
    "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414, "In": 114.818,
    "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196,
    "Ba": 137.327, "La": 138.90547, "Ce": 140.116, "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98040, "Th": 232.0377, "U": 238.02891,
}

# Lattices: conventional cell in units of a (orthorhombic: [1, b/a, c/a]), basis as (site, fx, fy, fz);
# 'site' is the index into the study's 'elements' list.
_FCC = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]


def _shift(pts, d):
    return [tuple((p[i] + d[i]) % 1.0 for i in range(3)) for p in pts]


LATTICES = {
    "sc": {"sites": 1, "cubic": True, "basis": [(0, 0, 0, 0)]},
    "bcc": {"sites": 1, "cubic": True, "basis": [(0, 0, 0, 0), (0, 0.5, 0.5, 0.5)]},
    "fcc": {"sites": 1, "cubic": True, "basis": [(0, *p) for p in _FCC]},
    "diamond": {"sites": 1, "cubic": True,
                "basis": [(0, *p) for p in _FCC] + [(0, *p) for p in _shift(_FCC, (0.25, 0.25, 0.25))]},
    "rocksalt": {"sites": 2, "cubic": True,
                 "basis": [(0, *p) for p in _FCC] + [(1, *p) for p in _shift(_FCC, (0.5, 0, 0))]},
    "cscl": {"sites": 2, "cubic": True, "basis": [(0, 0, 0, 0), (1, 0.5, 0.5, 0.5)]},
    "zincblende": {"sites": 2, "cubic": True,
                   "basis": [(0, *p) for p in _FCC] + [(1, *p) for p in _shift(_FCC, (0.25, 0.25, 0.25))]},
    "fluorite": {"sites": 2, "cubic": True,
                 "basis": [(0, *p) for p in _FCC] + [(1, *p) for p in _shift(_FCC, (0.25, 0.25, 0.25))]
                 + [(1, *p) for p in _shift(_FCC, (0.75, 0.75, 0.75))]},
    "l12": {"sites": 2, "cubic": True, "basis": [(1, 0, 0, 0), (0, 0.5, 0.5, 0), (0, 0.5, 0, 0.5), (0, 0, 0.5, 0.5)]},
    # orthohexagonal cell a x sqrt(3) a x c, 4 atoms
    "hcp": {"sites": 1, "cubic": False,
            "basis": [(0, 0, 0, 0), (0, 0.5, 0.5, 0), (0, 0.5, 5 / 6, 0.5), (0, 0, 1 / 3, 0.5)]},
}
LATTICE_DOC = {
    "sc": "simple cubic [A]", "bcc": "[A]", "fcc": "[A]", "diamond": "[A] (Si, C, Ge)",
    "rocksalt": "B1 [A, B] (NaCl, MgO)", "cscl": "B2 [A, B]", "zincblende": "B3 [A, B] (GaAs, SiC)",
    "fluorite": "C1 [A, B] (CaF2: A on fcc, B on tetrahedral sites)", "l12": "[A, B] A3B (Ni3Al: A=Ni)",
    "hcp": "[A], c = 'c' (default ideal sqrt(8/3) a); orthohexagonal cell a x sqrt(3)a x c, orient not allowed",
    "custom": "'cell': [ax, ay, az] (orthorhombic, A) and 'basis': [[site, fx, fy, fz], ...]; a = 1 scales nothing",
}


def lattice_cell(kind: str, a: float, c: float | None = None) -> list[float]:
    if kind == "hcp":
        return [a, math.sqrt(3.0) * a, c if c else math.sqrt(8.0 / 3.0) * a]
    return [a, a, a]


# Built-in molecules: geometry (A), labels, bonds; force field in LAMMPS 'real' units
# (kcal/mol, A, e). Bond and angle energies are LAMMPS's harmonic: E = K (r - r0)^2, E = K (theta - theta0)^2.
def _water(r, theta_deg):
    t = math.radians(theta_deg) / 2.0
    return [("OW", 0.0, 0.0, 0.0), ("HW", r * math.sin(t), r * math.cos(t), 0.0),
            ("HW", -r * math.sin(t), r * math.cos(t), 0.0)]


MOLECULES = {
    "spce": {
        "doc": "SPC/E water (Berendsen 1987), rigid (SHAKE); use coulomb long and tail corrections",
        "atoms": _water(1.0, 109.47), "bonds": [[0, 1], [0, 2]],
        "forcefield": {
            "types": {"OW": {"mass": 15.9994, "q": -0.8476, "eps": 0.1553, "sigma": 3.166},
                      "HW": {"mass": 1.008, "q": 0.4238, "eps": 0.0, "sigma": 0.0}},
            "bonds": {"OW-HW": {"k": 554.1349, "r0": 1.0}},
            "angles": {"HW-OW-HW": {"k": 45.7696, "theta0": 109.47}},
            "rigid": ["OW-HW", "HW-OW-HW"]},
    },
    "tip3p": {
        "doc": "TIP3P water (Jorgensen 1983), rigid (SHAKE)",
        "atoms": _water(0.9572, 104.52), "bonds": [[0, 1], [0, 2]],
        "forcefield": {
            "types": {"OW": {"mass": 15.9994, "q": -0.834, "eps": 0.1521, "sigma": 3.15061},
                      "HW": {"mass": 1.008, "q": 0.417, "eps": 0.0, "sigma": 0.0}},
            "bonds": {"OW-HW": {"k": 450.0, "r0": 0.9572}},
            "angles": {"HW-OW-HW": {"k": 55.0, "theta0": 104.52}},
            "rigid": ["OW-HW", "HW-OW-HW"]},
    },
    "methane_ua": {
        "doc": "united-atom methane (TraPPE: eps/k = 148 K, sigma 3.73 A)",
        "atoms": [("CH4", 0.0, 0.0, 0.0)], "bonds": [],
        "forcefield": {"types": {"CH4": {"mass": 16.043, "q": 0.0, "eps": 148.0 * 0.0019872067, "sigma": 3.73}}},
    },
}
