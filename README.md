# lammps-translator

Turns an **atomistic study** (a structure, its interactions, an ordered list of stages) into [LAMMPS](https://www.lammps.org) inputs and checked results, deterministically. It includes a command line (`lmpt`), a Python API and an MCP server (`lmpt-mcp`), so LLM-based tools can set up simulations as studies and get the numbers back with their checks.

It is the atomistic counterpart of [build123d-translator](https://github.com/feranick/build123d-translator), [fea-translator](https://github.com/feranick/fea-translator) and [cfd-translator](https://github.com/feranick/cfd-translator). An LLM writes the study (the flexible part). The translator computes everything else: every atom position, type, charge and bond (no `lattice`/`create_atoms` in LAMMPS), every pair coefficient (mixing done here), the strains, volumes and slabs of the property stages, and one LAMMPS input per stage. An independent implementation then recomputes LAMMPS's energy, forces and pressure on the initial structure; each stage checks itself (virial vs energy, Born stability, Cauchy relation, energy drift, convergence); benchmarks compare with closed forms and published values. Results include a fea-translator material (`feat_material`), so atomistic elastic constants go straight into a continuum model.

Scope (first version): **crystals** (fcc, bcc, sc, hcp, diamond, rocksalt, CsCl, zincblende, fluorite, L1₂, custom orthorhombic; orientations, random substitution) with EAM / EAM-alloy / EAM-FS / ADP / Tersoff / SW / MEAM / AIREBO / SNAP files or LJ / Morse / Buckingham with Coulomb (Ewald, PPPM); **molecular liquids** (SPC/E, TIP3P, united-atom methane, or custom molecules with harmonic bonds and angles, OPLS dihedrals, SHAKE); LAMMPS data files. Stages: energy, minimize (box relax), NVE/NVT/NPT MD (averages with errors, RDF, MSD), elastic, eos, vacancy, surface, thermal expansion.

## Install

On the machine that runs LAMMPS (Ubuntu; LAMMPS 4 Jul 2026 from the Debian package, MPI + OpenMP, MANYBODY, KSPACE, MOLECULE, RIGID, MEAM, ML-SNAP, ...):

```
cd ~/Software && git clone https://github.com/feranick/lammps-translator.git    # or: cd lammps-translator && git pull
source ~/Software/build123d-translator/.venv/bin/activate                          # shared venv (alias cad)
pip install -e "$HOME/Software/lammps-translator[mcp,test]"
lmpt status            # LAMMPS found? version, packages, potentials folders, fea-translator
pytest -q              # 26 tests: 21 run anywhere, 5 need LAMMPS
lmpt bench             # benchmarks (seconds); slow ones by name: lmpt bench spce_water
```

The translator finds LAMMPS as `LMPT_LMP` (a command, e.g. `/usr/bin/lmp` or `lmp_mpi`), else `lmp`, `lmp_mpi`, `lmp_serial` or `lammps` on the PATH. Potential files are looked up in `LMPT_POTENTIALS` (`:`-separated), `LAMMPS_POTENTIALS`, `/usr/share/lammps/potentials`, `/usr/local/share/lammps/potentials`, the `lammps` Python package's `share/lammps/potentials`, and next to the executable (`../share/lammps/potentials`); `lmpt status` lists what it found. Without a system LAMMPS, `pip install "lammps-translator[lammps]"` brings the PyPI wheel (it needs `libmpi.so.12`: `pip install mpich` and its `lib` on `LD_LIBRARY_PATH`).

## Use

```
lmpt format > study_format.json          # the study format (what an LLM reads first)
lmpt validate examples/studies.json      # static checks with fixable messages (builds structures, finds files)
lmpt run examples/studies.json           # run, check; results in ~/lmpt_runs/run_*/
lmpt run examples/studies.json cu_properties --json
lmpt inputs examples/studies.json cu_properties   # write the LAMMPS inputs only (inspect, run by hand)
```

Every run writes `<request>/<study>/`: `initial.data`, `potentials/` (copies), one folder per stage (`00_setup`, `01_minimize`, ...) with `in.lammps`, `log.lammps`, `state.data` and stage files (`rdf.txt`, `traj.lammpstrj`, slab and vacancy data), `run_all.sh` (reruns every input in order with plain LAMMPS) and `<study>.json`; plus `<study>.tgz`. Environment: `LMPT_OUT` (~/lmpt_runs), `LMPT_WORKERS` (2), `LMPT_TIMEOUT` (21600 s per study), `LMPT_PROCS` (MPI ranks, 1; or the study's `procs`), `LMPT_MPIRUN` (mpirun).

## A study

```json
{"cu_properties": {
  "system": {"lattice": "fcc", "a": 3.6, "elements": ["Cu"], "size": [4, 4, 4]},
  "potential": {"style": "eam", "file": "Cu_u3.eam"},
  "stages": [{"kind": "minimize", "box": "iso"}, {"kind": "elastic"}, {"kind": "vacancy"},
             {"kind": "surface", "miller": [1, 1, 1]}],
  "expect": [{"quantity": "minimize.a_A", "value": 3.615, "tol": 0.001}]}}
```

- **system** — one of:
  - `{"lattice": kind, "a", "elements", "size", "orient", "substitute", "perturb"}`: kinds `sc bcc fcc diamond rocksalt cscl zincblende fluorite l12 hcp custom` (`elements` by lattice site; hcp takes `c`, default ideal; custom takes `cell` [ax, ay, az] and `basis` [[site, fx, fy, fz]]). `size` defaults to each side ≥ 2 cutoffs and ≥ 3 cells. `orient` {x, y, z} (cubic) builds the smallest orthogonal periodic cell along those directions. `substitute` [{from, to, fraction}] makes a random solid solution (seeded).
  - `{"molecules": [{molecule, count}], "density_g_cm3" | "box"}`: built-in `spce`, `tip3p`, `methane_ua`, or `{atoms: [[label, x, y, z]], bonds: [[i, j]], angles?, dihedrals?}`; angles and dihedrals default to all those of the bond graph. Molecules go on a cubic grid with random orientations; minimize first.
  - `{"data": text or server path, "atom_style": atomic | charge | full, "type_names": [...]}`; for `full`, bond/angle/dihedral types are re-derived from the atom labels and the force field.
- **potential** (atomic systems; LAMMPS units metal: eV, Å): `{"style": "eam", "file": "Cu_u3.eam"}` (one per element: `files {El: file}`), `eam/alloy`, `eam/fs`, `adp`, `tersoff`, `tersoff/mod`, `tersoff/zbl`, `sw`, `meam` (`files [library, parameters]`), `airebo` (`args "3.0 1 1"`), `rebo`, `edip`, `vashishta`, `snap` (`files [coeff, param]`); or analytic `lj` (`pairs {"A-B": {eps, sigma}}`, missing cross pairs mixed geometrically; one element: `eps`, `sigma` directly), `morse` (`D0, alpha, r0`), `buck` (`A, rho, C`; missing pairs: no short-range term), `coul` (charges only), with `cutoff`, `shift`, `tail`, `charges {El: q}`, `coulomb none | cut | long` and `kspace_accuracy` (1e-8; Ewald up to 2000 atoms, else PPPM).
- **forcefield** (molecular systems; units real: kcal/mol, Å, e): `types {label: {mass, q, eps, sigma}}`, `pairs` (explicit cross terms), `mixing geometric | arithmetic`, `bonds {"A-B": {k, r0}}`, `angles {"A-B-C": {k, theta0}}`, `dihedrals {"A-B-C-D": {k: [K1..K4]}}` (OPLS), `rigid` (labels held by SHAKE), `cutoff` (10), `coulomb long | cut | none`, `kspace_accuracy` (1e-5), `tail` (true), `special` [1-2, 1-3, 1-4] weights. LAMMPS's harmonic terms have no ½: E = K (r − r₀)².
- **stages**, in order. `minimize`, `nve`, `nvt`, `npt` change the state (structure and velocities pass to the next stage); the others measure it.

| kind | parameters | results (under the stage's name) |
|---|---|---|
| energy | | `pe_eV`, `pe_per_atom_eV`, `E_coh_eV`, `pressure_GPa`, `stress_GPa` (xx yy zz yz xz xy, tension +), `fmax_eV_A` |
| minimize | `box` fixed / iso / aniso / tri, `P_bar`, `ftol`, `maxiter` | as energy + `a_A` (`c_A`, `c_over_a` for hcp), `density_g_cm3`, `stop`, `pressure_error_GPa` |
| nve / nvt / npt | `T`, `T_end`, `steps`, `dt_fs`, `thermostat` nose_hoover / langevin, `tdamp_fs`, `P_bar`, `coupling` iso / aniso / tri, `pdamp_fs`, `velocities`, `seed`, `average_from` (0.5), `thermo_every`, `rdf {pairs, r_max, bins}`, `msd {of}`, `dump_every` | `mean.*` and `err.*` (block averages): `T_K`, `P_GPa`, `P_bar`, `pe_per_atom_eV`, `etotal_per_atom_eV`, `volume_A3`, `density_g_cm3`, box lengths, pressure tensor, `pe_per_molecule_kJ_mol`, `a_A` (npt); `msd.D_m2_s`; `rdf.<A-B>.g`, `peak_r_A`, `coordination`; `etotal_slope_eV_per_atom_ps`; `final` |
| elastic | `strain` (1e-3), `relax` (true) | `C_GPa` (6×6), `C11_GPa` … `C66_GPa`, `B_GPa`, `G_GPa`, `E_GPa`, `nu` (Voigt–Reuss–Hill), `born_stable`, `cubic` averages, Zener ratio, Cauchy difference |
| eos | `strain` (0.04), `n` (9), `relax` | `points`, Birch–Murnaghan `V0_A3`, `E0_eV`, `B0_GPa`, `B0_prime`, `a0_A` |
| vacancy | `element`, `relax` | `E_f_eV` = E(N−1) − (N−1)/N E(N), constant volume |
| surface | `miller`, `layers`, `vacuum` (15 Å), `relax` | `gamma_J_m2`, `gamma_unrelaxed_J_m2`, `area_A2`, orientation used |
| thermal_expansion | `T` [K, …], `steps` (20000), `P_bar` | `points` (a(T), V(T)), `alpha_linear_1_K`, `alpha_volume_1_K` |

- **expect**: `[{quantity, value, tol | abs}]` with dotted paths (`elastic.C11_GPa`, `nvt.mean.T_K`). **seed** (12345), **procs** (MPI ranks).

### Results

Per stage under its name (default: its kind; repeats get `_2`), plus `structure` (initial: atoms, composition, box, density, lattice), `initial` (energy of the structure as built), `final`, `feat_material` {E (MPa), nu, density (kg/m³), alpha (1/K)} with `feat_material_note`, `checks`, `warnings`, `reliable` (false, with `unreliable_because`, when a hard check failed: do not report the numbers), `expect`, `lammps` (version, command, ranks), `seconds`, `files`.

**Interface** (read by other tools; kept stable): `feat_material.*`, `elastic.C_GPa`, `elastic.{B,G,E}_GPa`, `elastic.nu`, `minimize.a_A`, `<md stage>.mean.*`, `reliable`.

## The checker

**Every run checks itself:**

- **Reference implementation** (`refcalc.py`, on the initial structure, up to 4000 atoms): energy, every force and the pressure tensor recomputed from the written data file and the normalised model, not from the LAMMPS input. It covers lj, morse, buck (shift, tail corrections), Coulomb cut and long (its own Ewald sum, independent parameters), eam (funcfl with LAMMPS's grid conversion), eam/alloy and eam/fs (LAMMPS's splines, read the way LAMMPS reads), harmonic bonds and angles, OPLS dihedrals and special-bond weights. Tolerances: 1e-8 eV/atom and 1e-6 eV/Å for short-range styles (observed: 1e-14 to 1e-16); with long-range Coulomb they follow `kspace_accuracy` and LAMMPS's polynomial erfc (observed 2e-7 relative). Many-body files without a reference (tersoff, sw, meam, airebo, snap) say so in `checks.reference`.
- **Masses**: LAMMPS's density of the initial structure equals ours (`checks.masses`).
- **Minimize**: the stop criterion, the largest force (warning above 1e-3 eV/Å, unreliable above 0.05), and with a box the final pressure (warning above 1e-4 GPa, unreliable above 0.01 GPa).
- **Elastic**: symmetry of C (2 %), Born stability (eigenvalues), the reference pressure (a warning above 0.1 GPa: then these are stress–strain coefficients), and for central pair potentials on centrosymmetric cubic lattices the Cauchy relation C12 = C44 + 2P.
- **EOS**: LAMMPS's virial must be the volume derivative of its own energy: E(V₂) − E(V₀) vs −∫P dV (Simpson) between neighbouring points; above 2 % of the curve's energy span the run is unreliable (energy jumps at a truncated cutoff), above 0.2 % it warns (force kinks at the cutoff).
- **MD**: mean temperature and pressure against the targets (warnings), NVE energy drift (warning above 1e-3, unreliable above 1e-2 eV/atom/ps), errors from 5 block averages.
- Any LAMMPS `ERROR` stops the study with the message and the stage it came from.

**Benchmarks** (`lmpt bench`; results here with LAMMPS 22 Jul 2025 update 4, the PyPI wheel on aarch64 Linux; to rerun with 4 Jul 2026 on the servers):

| case | reference | tolerance | result |
|---|---|---|---|
| lj_fcc_lattice_sums | truncated LJ fcc: nn distance, energy, C11, C12 = C44 from lattice sums (cutoff between shells) | 1e-7 / 1e-4 | pass (a 6e-11, E 1e-14, Cij ≤ 1.3e-7) |
| madelung_nacl | E = −M e²/(4πε₀ r₀), M = 1.747565 | 2e-6 | pass (2.1e-7: LAMMPS's erfc) |
| madelung_cscl | M = 1.762675 | 2e-6 | pass (1.7e-8) |
| cauchy_mgo | Buckingham + Ewald at P = 0: C12 = C44; B(eos) = B(elastic) | 1e-4 / 1 % | pass (2.3e-6, 2.6e-5) |
| eam_cu_foiles | Cu_u3 (Foiles, Baskes, Daw 1986): a 3.615, E_coh 3.54, B 138 GPa, E_vac 1.28 eV, γ(100) 1.28 J/m² | 0.1–2 % | pass (≤ 0.6 %) |
| tersoff_si | Si.tersoff (Tersoff 1988, Si(B)): a 5.432, E_coh 4.63, B 98; C11 121, C12 86, C44 10 GPa (relaxed) | 0.1–5 % | pass (C44 10.3: 3 %) |
| equipartition_lj | LJ fcc at 4 K: ⟨pe⟩ − pe₀ = (3/2) k T (N−1)/N | 3 % | pass (0.23 %) |
| nve_conservation | LJ fcc 40 K, NVE: drift | 1e-5 eV/atom/ps | pass (1e-8) |
| spce_water (slow) | SPC/E 298 K, 1 atm: ρ 0.998 g/cm³, U −46.7 kJ/mol (no polarization correction), O–O peak 2.75 Å | 1 % / 2 % / 3 % | to run (a shorter run here: ρ 0.999 ± 0.003, U −46.78 ± 0.06, peak 2.725) |
| chain_feat_cantilever | lmpt Cu elastic → `feat_material` → fea-translator cantilever: tip = F L³/(3 E I) | 2 % | skipped here (no fea-translator); via the two MCP servers: 1.2786 vs 1.2832 mm (0.36 %) |

**Assumptions about LAMMPS** (each with what catches it; confirmed with 22 Jul 2025, to confirm with 4 Jul 2026):

1. **Tail corrections are not −dE/dV.** LAMMPS's tail pressure is the virial correction, not the volume derivative of its tail energy, so `fix box/relax` with `pair_modify tail yes` stalls ("linesearch alpha is zero"). Validation refuses box relaxation with tail corrections. → test `test_validation_messages`.
2. **EAM files are read line by line.** After the last value of a block, the rest of the line is discarded (NiO.eam.fs has 12000 values per 10000-point block). The reference reads the same way. → reference check (it was off by 0.009 eV/atom before).
3. **EAM files set the masses** (pair_coeff overrides the data file); densities use the file's masses. → `checks.masses` (LAMMPS's density vs ours on the initial structure).
4. **Coulomb long uses a polynomial erfc** (~2e-7 relative), whatever `kspace_accuracy`. → madelung_nacl.
5. **Thermo keywords**: `evdwl` includes the tail energy; `ecoul + elong` is the full Coulomb energy; `pxx…` include the kinetic part (zero for static stages: velocities are set to 0). → molecular reference terms, lattice-sum and Madelung benchmarks.
6. **Stress sign and strain**: pressure positive in compression (stress = −P); `change_box … remap` with final values applied in sequence equals the affine map F = I + ε (upper triangular, engineering shear). → lj_fcc_lattice_sums, the rotated-cell elastic constants of Cu ([1-10]: (C11+C12+2C44)/2, [111]: (C11+2C12+4C44)/3, both within 0.2 %).
7. **write_data / read_data** round-trip positions (wrapped), velocities and topology; atoms are re-sorted by id. → every multi-stage example.
8. **SHAKE removes degrees of freedom** from the temperature. → SPC/E mean T at the target.
9. **The syntax used** (thermo_style custom, `$(…:%.15g)`, write_dump … modify format float, fix box/relax, change_box, compute rdf/msd, fix ave/time vector) is the same in 4 Jul 2026. → `pytest -q` and `lmpt bench` on the servers.

**Limits:** no reference for Tersoff/SW/MEAM/AIREBO/SNAP (checked by published values only); vacancy formation energies of compounds use the mean energy per atom (no chemical potentials); surfaces of compounds may be polar (two terminations averaged); molecular systems cannot be combined with many-body potentials (no hybrid styles); MD diffusion has no finite-size correction; classical thermal expansion (no zero-point effects).

## How it works

1. `format.py` validates the study (static, then builds it and finds the files). Every error names the key and the fix.
2. `build.py` makes the atoms: conventional cells, oriented periodic cells (the smallest translations along the orient vectors, atoms counted against the determinant), supercells, substitutions; molecules on a grid with seeded rotations and their topology; data files (written and read). Boxes are LAMMPS restricted triclinic with the origin at 0.
3. `potential.py` finds potential files, reads their elements, cutoffs and masses, types force fields from atom labels (order-insensitive keys), and writes every pair coefficient explicitly.
4. `deck.py` writes each stage's input from the state the previous stage left (just before it runs). Numbers come back as tagged lines (`print "LMPT <tag> pe $(pe:%.15g) …"`), so nothing depends on LAMMPS's thermo layout.
5. `lammps.py` runs LAMMPS (`-in -log -screen none`, MPI with `mpirun -np`), reads `lmp -h` for version, packages and styles (validation refuses a style the binary lacks).
6. `results.py` reads logs (thermo blocks, tags, errors, warnings, minimizer stops, loop times), dumps, and fix ave/time files. `analysis.py` turns them into stresses, Cij and moduli, Birch–Murnaghan fits, block averages, diffusion, RDF peaks. `refcalc.py` is the independent implementation. `runner.py` ties one study together; `api.py`, `cli.py`, `mcp_server.py` are the interfaces; `bench.py` the benchmarks.

## For LLM tools: the MCP server

Tools: `study_format`, `validate_studies`, `run_studies` (returns within ~45 s; long studies come back `pending` with `progress`: stage, MD steps done of total, s/step, `eta_min_max`), `study_status`, `get_inputs`, `run_benchmarks`, `get_file` (base64, for callers on another machine). In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
"lmpt": {"command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "nicola@carbonio",
                  "/home/nicola/Software/build123d-translator/.venv/bin/lmpt-mcp"]}
```

Polling costs tokens: every `study_status` call is a model turn. The server's instructions tell the model to report the expected finish time and stop when a run needs more than about 5 minutes, and to collect the result later. `lmpt-mcp --http 8769 --host 0.0.0.0` serves streamable HTTP instead of stdio. Results' `feat_material` goes as the `material` of a fea-translator study.

## Files

```
lammps_translator/
  format.py     study format, validation        deck.py      LAMMPS inputs per stage
  data.py       masses, lattices, molecules      lammps.py    runs LAMMPS, reads its capabilities
  build.py      atoms, topology, data files      results.py   log, dump, ave/time readers
  potential.py  files, force fields, commands    analysis.py  Cij, EOS, averages, RDF, MSD
  refcalc.py    reference implementation         runner.py    one study end to end
  bench.py      benchmarks                       api.py, cli.py, mcp_server.py
tests/test_lmpt.py    examples/studies.json    CLAUDE.md (session context)
```

## Versions

| version | changes |
|---|---|
| 2026.10.16.1 | First version: lattices (11 kinds, orient, substitution), packed molecules (SPC/E, TIP3P, methane UA, custom), data files; EAM/ADP/Tersoff/SW/MEAM/AIREBO/SNAP files and LJ/Morse/Buckingham + Coulomb (Ewald, PPPM); force fields with harmonic bonds/angles, OPLS dihedrals, SHAKE; stages energy, minimize, nve/nvt/npt (RDF, MSD), elastic, eos, vacancy, surface, thermal_expansion; reference implementation; 10 benchmarks; feat_material; MCP server with pending runs, progress and ETA. Tested with LAMMPS 22 Jul 2025 (PyPI wheel, aarch64): 26 tests, 8 default benchmarks pass. |

Author: Nicola Ferralis <ferralis@generalflash.com>
