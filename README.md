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
pytest -q              # 63 tests: 53 run anywhere (SLURM ones with fake sbatch/squeue), 10 need LAMMPS
lmpt bench             # benchmarks (~17 s); slow ones by name: lmpt bench spce_water (~9 min, run it from the shell)
```

The translator finds LAMMPS as `LMPT_LMP` (a command, e.g. `/usr/bin/lmp` or `lmp_mpi`), else `lmp`, `lmp_mpi`, `lmp_serial`, `lmp_stable`, `lmp_openmpi` or `lammps` on the PATH, else in `/usr/bin`, `/usr/local/bin`, `/opt/lammps/bin`, `~/.local/bin`. If none is found, `lmpt status` and every run say so (and what to set). The MCP server runs under a non-interactive ssh shell that does not read `~/.bashrc`: put variables in the ssh command (`env LMPT_LMP=... lmpt-mcp`). Potential files are looked up in `LMPT_POTENTIALS` (`:`-separated), `LAMMPS_POTENTIALS`, `/usr/share/lammps/potentials`, `/usr/local/share/lammps/potentials`, the `lammps` Python package's `share/lammps/potentials`, and next to the executable (`../share/lammps/potentials`); `lmpt status` lists what it found. Without a system LAMMPS, `pip install "lammps-translator[lammps]"` brings the PyPI wheel (it needs `libmpi.so.12`: `pip install mpich` and its `lib` on `LD_LIBRARY_PATH`).

## Use

```
lmpt format > study_format.json          # the study format (what an LLM reads first)
lmpt validate examples/studies.json      # static checks with fixable messages (builds structures, finds files)
lmpt run examples/studies.json           # run, check; results in ~/lmpt_runs/run_*/
lmpt run examples/studies.json cu_properties --json
lmpt inputs examples/studies.json cu_properties   # write the LAMMPS inputs only (inspect, run by hand)
lmpt jobs [--active]                     # recent requests with SLURM jobs, their states, the queue
lmpt cancel REQUEST [NAME ...]           # stop a request's studies
lmpt run-study REQUEST NAME              # run one study of a request (what a SLURM job runs)
```

Every run writes `<request>/<study>/`: `initial.data`, `potentials/` (copies), one folder per stage (`00_setup`, `01_minimize`, ...) with `in.lammps`, `log.lammps`, `state.data` and stage files (`rdf.txt`, `traj.lammpstrj`, slab and vacancy data), `run_all.sh` (reruns every input in order with plain LAMMPS) and `<study>.json`; plus `<study>.tgz`. Environment: `LMPT_OUT` (~/lmpt_runs), `LMPT_WORKERS` (2), `LMPT_TIMEOUT` (21600 s per study), `LMPT_PROCS` (MPI ranks, 1; or the study's `procs`), `LMPT_MPIRUN` (mpirun), `LMPT_EXECUTOR`, `LMPT_SLURM_MIN_MINUTES`, `LMPT_SLURM_ARGS` (below).

## Long runs: SLURM

With `LMPT_EXECUTOR=slurm` (in llm-translators' `config/env`), each long study is a SLURM job instead of a worker of the MCP server, following llm-translators' [docs/EXECUTION.md](https://github.com/feranick/llm-translators/blob/main/docs/EXECUTION.md). Studies from every chat then share the machine through one queue, and a job keeps running when the conversation or its ssh connection ends.

- **Which studies.** A study whose time estimate (atoms × steps × a per-style cost, rough) is below `LMPT_SLURM_MIN_MINUTES` (1) runs locally; the others, and those with no estimate, are jobs. `"queue": true` / `false` in a study forces a job / a local run (not with `LMPT_EXECUTOR=local`).
- **The job.** `<request>/<name>/job.sh`, submitted with `sbatch --parsable $LMPT_SLURM_ARGS job.sh`: job name `lmpt:<request>:<name>`, `--ntasks` = the study's `procs` or `LMPT_PROCS`, `--ntasks-per-core=1` (each rank a physical core: `LMPT_PROCS` is sized in cores, and Open MPI binds ranks to cores), `--time` and `--mem` (below), `--export=ALL`, and `exec <the server's python> -m lammps_translator.cli run-study <request> <name>`. Inside, LAMMPS runs with `$LMPT_MPIRUN -np $SLURM_NTASKS lmp` (one rank: `lmp` alone); if the ranks outnumber the physical cores of the allocation, Open MPI gets `--bind-to hwthread` (or `--bind-to none --oversubscribe`) and the result warns. A binding in `LMPT_MPIRUN` is left alone. `job.json` records the id. `sbatch job.sh` reruns it by hand. If `sbatch` fails, the study's result is `ok: false, why: "could not submit: ..."` (never a silent local run).
- **Limits.** Defaults that let several jobs share one node:

  | setting | default | the job asks for |
  |---|---|---|
  | `LMPT_SLURM_MEM` | `auto` | `--mem` = 1 GB + 256 MB per rank + 4 kB per atom, rounded up to 512 MB, at least 2 GB (a 9-rank study of a few thousand atoms: 3.5 GB). Or a size (`8G`), or `none` (SLURM's default: with `CR_Core_Memory` and no `DefMemPerCPU` that is the whole node, so jobs run one at a time, as found on mochi). A `--mem…` in `LMPT_SLURM_ARGS` wins. |
  | `LMPT_SLURM_TIME` | `timeout` | `--time` = ceil(`LMPT_TIMEOUT`/60) + 10 min (6 h 10 min with the default 6 h), as docs/EXECUTION.md says. |
  | `LMPT_SLURM_TIME=estimate` | | `--time` = max(`LMPT_SLURM_TIME_MIN` (20), ceil(`LMPT_SLURM_TIME_FACTOR` (3) × the estimate) + 10) min, at most the above; the study's own timeout is 10 min less. Short limits let SLURM backfill short studies into free cores ahead of long or large jobs. A study that outruns its estimate ends `timed_out`, saying to rerun with `LMPT_SLURM_TIME=timeout` or a larger factor. |
  | cores | `--ntasks-per-core=1` | unless `LMPT_SLURM_ARGS` sets `--hint`, `--threads-per-core`, `--ntasks-per-core` or `--cpus-per-task` |

  `job.json` records `time_min`, `time_from`, `timeout_s`, `mem`, `estimate_min` and `natoms`.
- **Per-study run settings.** `config/env` holds the machine's defaults and only a person edits it; a study can override four of them for itself with `"run": {"timeout_s": 30000, "mem": "12G", "procs": 10, "queue": true}` (all optional; docs/EXECUTION.md, "Per-study run settings"). `timeout_s` replaces `LMPT_TIMEOUT` for that study (LAMMPS's own limit, and the job's `--time` = ceil(timeout_s/60) + 10, whatever `LMPT_SLURM_TIME` says); `mem` is the job's `--mem`, and any `--mem…` in `LMPT_SLURM_ARGS` is dropped for that job while its other arguments stay; `procs` is the ranks (the same as the top-level `procs`, still capped by the atoms); `queue` is the top-level `queue`. No sbatch arguments come from a study. Ceilings, in `config/env`: `LMPT_MAX_TIMEOUT` (s), `LMPT_MAX_MEM` (e.g. `32G`), `LMPT_MAX_PROCS` (also bounds the top-level `procs`); unset means no cap. Validation refuses more than a ceiling (naming it), unknown `run` keys, a malformed `mem` (SLURM sizes: `500M`, `8G`, `1T`), and `run.procs` / `run.queue` that disagree with `procs` / `queue`. `job.json` keeps the study's `run`. Other studies in the same request keep the defaults.
- **Estimates.** `validate_studies` gives each valid study `"estimate": {"minutes", "procs", "mem_mb"}`: the same sizing the SLURM job gets (ranks after the cap, memory as for `--mem`, minutes from the time estimate, an upper bound). llm-translators' router uses them to place runs.
- **Ranks a study can use.** A job asks for min(procs, ceil(atoms / `LMPT_ATOMS_PER_RANK` (2000))) ranks: atoms are known before submitting, so a 256-atom study takes 1 rank, not `LMPT_PROCS`. Locally the same cap applies; a cut is reported in `warnings`, and `execution` shows `procs_requested` and `procs_used`. LAMMPS often scales below 2000 atoms per rank: measure and set it.
- **Run record.** Every result has `execution` (docs/EXECUTION.md): `executor` (`slurm`, `local`, `by hand`), `slurm_job`, `procs_requested`, `procs_used`, `mpi` {`launch`, `binding`, `cores`, `hwthreads`} (null for one rank), and `limits` {`timeout_s`, `mem` (the job's, from `SLURM_MEM_PER_NODE`, e.g. `12288M`; null locally), `from_study` (the `run` keys the study set)}. Results that end without running carry `executor` and `slurm_job`.
- **States.** `study_status` keeps its shape; a pending study's `progress` has `state`: `queued` (`reason`, `expected_start`, `ahead`), `running` (stage, MD steps, `eta_min_max`, `job_elapsed_min`). They come from the files and one `squeue` call, so any server process, after restarts or from another chat, can collect them. A job that leaves the queue without a result ends as `timed_out`, `out_of_memory`, `cancelled` or `failed` (from `sacct`, or from `job.log` when there is no accounting), with `ok: false` and what to change.
- **`jobs` and `cancel`.** `jobs(limit, active)` lists the newest requests with jobs, the queue (`running`, `queued`, `cpus_in_use` of `lmpt:` jobs) and the machine (`sinfo` CPUs). `cancel(request, names)` runs `scancel`; in local mode it stops a study only if this server process runs it.
- **Benchmarks** go through the same path (`lmpt bench --queue`: every case as a job): `lmpt bench` waits for them; the MCP tool returns pending cases with a `request` to pass back as `run_benchmarks(request=...)`.
- **Tested** on carbonio (12 CPUs, 6 cores × 2 threads, LAMMPS 4 Jul 2026, Open MPI 5), 2026.10.18.3: `madelung_nacl` as job 5515 (pass); a 2-rank NVT study as job 5516: whole cores (then `--hint=nomultithread`) gave 2 physical cores (4 CPUs in SLURM's count), Open MPI's default binding, `reliable: true`, mean T 39.8 K for 40 K; `jobs` showed it running with stage and ETA, then done. On mochi (40 CPUs, 20 cores × 2 threads), where 2026.10.18.1 failed with Open MPI's 'allocation-overload': `madelung_nacl` as job 2482 (pass); a 9-rank NVT study as job 2483: 9 physical cores (18 hardware threads), default binding, `reliable: true`, 12.8 s.
- `lmpt status` (`tool_status`) reports `executor: {kind, sbatch, min_minutes, procs, time_limit_min, time, mem, slurm_args, workers, timeout_s, atoms_per_rank, caps: {timeout_s, mem, procs}}` (+ `time_factor`, `time_min_min` with `time: estimate`). `study_format` includes it, so a model sees the defaults and caps before it writes a `run`.

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

**Benchmarks** (`lmpt bench`; LAMMPS 4 Jul 2026 on carbonio (Debian build), 2026-10-16; identical to 22 Jul 2025 (PyPI wheel, aarch64) except where noted; all default cases in 17 s):

| case | reference | tolerance | result |
|---|---|---|---|
| lj_fcc_lattice_sums | truncated LJ fcc: nn distance, energy, C11, C12 = C44 from lattice sums (cutoff between shells) | 1e-7 / 1e-4 | pass (a 6e-11, E 1e-14, Cij ≤ 1.3e-7) |
| madelung_nacl | E = −M e²/(4πε₀ r₀), M = 1.747565 | 2e-6 | pass (2.1e-7: LAMMPS's erfc) |
| madelung_cscl | M = 1.762675 | 2e-6 | pass (1.7e-8) |
| cauchy_mgo | Buckingham + Ewald at P = 0: C12 = C44; B(eos) = B(elastic) | 1e-4 / 1 % | pass (2.3e-6, 2.6e-5) |
| eam_cu_foiles | Cu_u3 (Foiles, Baskes, Daw 1986): a 3.615, E_coh 3.54, B 138 GPa, E_vac 1.28 eV, γ(100) 1.28 J/m² | 0.1–2 % | pass (≤ 0.6 %) |
| tersoff_si | Si.tersoff (Tersoff 1988, Si(B)): a 5.432, E_coh 4.63, B 98; C11 121, C12 86, C44 10 GPa (relaxed) | 0.1–5 % | pass (C44 10.3: 3 %) |
| equipartition_lj | LJ fcc at 4 K: ⟨pe⟩ − pe₀ = (3/2) k T (N−1)/N | 3 % | pass (0.04 %; 0.23 % with 22 Jul 2025: a different random trajectory) |
| nve_conservation | LJ fcc 40 K, NVE: drift | 1e-5 eV/atom/ps | pass (1e-8) |
| spce_water (slow) | SPC/E 298 K, 1 atm: ρ 0.998 g/cm³, U −46.7 kJ/mol (no polarization correction), O–O peak 2.75 Å | 1 % / 2 % / 3 % | pass (ρ 0.9972: 0.08 %; U −46.60 kJ/mol: 0.22 %; peak 2.725 Å: 0.9 %, the 0.05 Å bin centre; 535 s on carbonio) |
| chain_feat_cantilever | lmpt Cu elastic → `feat_material` → fea-translator cantilever: tip = F L³/(3 E I) | 2 % | pass (1.2786 vs 1.2832 mm: 0.36 %; fea-translator 2026.10.16.9 in the same venv) |

**Assumptions about LAMMPS** (each with what catches it; all confirmed with 22 Jul 2025 and 4 Jul 2026: 27 tests and the benchmarks pass on both):

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

Tools: `study_format`, `validate_studies`, `run_studies` (returns within ~45 s; long studies come back `pending` with `progress`: state, stage, MD steps done of total, s/step, `eta_min_max`; with SLURM the job id), `study_status`, `get_inputs`, `jobs`, `cancel`, `run_benchmarks`, `get_file` (base64, for callers on another machine). In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
"lmpt": {"command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "nicola@carbonio",
                  "/home/nicola/Software/build123d-translator/.venv/bin/lmpt-mcp"]}
```

Polling costs tokens: every `study_status` call is a model turn. The server's instructions tell the model to report the expected finish time and stop when a run needs more than about 5 minutes, and to collect the result later; with SLURM, not to poll at all (the jobs do not depend on the conversation) and to find runs again with `jobs()`. `lmpt-mcp --http 8769 --host 0.0.0.0` serves streamable HTTP instead of stdio. Results' `feat_material` goes as the `material` of a fea-translator study.

## Files

```
lammps_translator/
  format.py     study format, validation        deck.py      LAMMPS inputs per stage
  data.py       masses, lattices, molecules      lammps.py    runs LAMMPS, reads its capabilities
  build.py      atoms, topology, data files      results.py   log, dump, ave/time readers
  potential.py  files, force fields, commands    analysis.py  Cij, EOS, averages, RDF, MSD
  refcalc.py    reference implementation         runner.py    one study end to end
  bench.py      benchmarks                       executor.py  local pool or SLURM jobs, states
  api.py, cli.py, mcp_server.py
tests/test_lmpt.py  tests/test_slurm.py  tests/fakeslurm.py  tests/conftest.py    examples/studies.json    context_new_version.md
```

## Versions

| version | changes |
|---|---|
| 2026.10.22.1 | `validate_studies` returns, for each valid study, `estimate`: {`minutes` (upper estimate with these ranks), `procs` (after the atoms cap), `mem_mb` (what the job reserves: `run.mem`, a `--mem`/`--mem-per-cpu` in `LMPT_SLURM_ARGS`, `LMPT_SLURM_MEM`, or auto)}, from the same functions that write `job.sh` (llm-translators docs/EXECUTION.md "Estimates in validation", for the router). 63 tests. |
| 2026.10.20.2 | Tests ignore the machine's settings (`tests/conftest.py` clears `LMPT_*` and `SLURM_*` for every test): on carbonio, `LMPT_PROCS=2` from `config/env` failed `test_rotated_cell_elastic_constants`, and with `LMPT_EXECUTOR=slurm` some tests submitted real jobs. |
| 2026.10.20.1 | Per-study run settings (docs/EXECUTION.md, as cfdt 2026.10.20.1): `run` {`timeout_s`, `mem`, `procs`, `queue`} with ceilings `LMPT_MAX_TIMEOUT`, `LMPT_MAX_MEM`, `LMPT_MAX_PROCS`; validation of caps, unknown keys, sizes and conflicts; `run.mem` replaces `--mem…` in `LMPT_SLURM_ARGS`; `job.json` keeps `run`; `execution.limits`; `executor.timeout_s` and `caps` in status. Ranks capped at atoms / `LMPT_ATOMS_PER_RANK` (2000) before submitting and locally. 62 tests. |
| 2026.10.18.6 | `executor.mem` in `lmpt status` shows a `--mem` from `LMPT_SLURM_ARGS` when it overrides `LMPT_SLURM_MEM`. Mochi, 2026.10.18.5 through the MCP server: two 9-rank NVT studies ran side by side as jobs 2516 and 2517 (9 cores each, default binding, reliable, identical results), and a 64-atom Madelung study in the same request stayed local (estimate under 1 min). |
| 2026.10.18.5 | Job limits: `LMPT_SLURM_MEM` (`auto` by default: memory from ranks and atoms, so jobs share a node whose SLURM gives each job all memory by default), `LMPT_SLURM_TIME=estimate` (`--time` from the study's estimate, for backfill; `LMPT_SLURM_TIME_FACTOR`, `LMPT_SLURM_TIME_MIN`); whole cores with `--ntasks-per-core=1` (docs/EXECUTION.md) instead of `--hint=nomultithread`; the run record `execution` in every result (replaces `lammps.slurm_job` and `lammps.mpi`). 49 tests. |
| 2026.10.18.4 | NUL characters (from Open MPI's messages) removed from LAMMPS error text; `ahead` counts pending jobs by priority, then job id (squeue lists them in another order). Mochi: three 9-rank studies started at once ran as jobs 2484–2486, all reliable; `jobs` from another server process (this conversation's MCP connection) listed and read them. |
| 2026.10.18.3 | `lmpt bench --queue` (`bench(queue=True)`): every case as a SLURM job, whatever `LMPT_SLURM_MIN_MINUTES` says (llm-translators' `bin/run` loads `config/env` after the command line, so `LMPT_SLURM_MIN_MINUTES=0 bin/run ...` is overridden there); benchmark rows show the SLURM job id. |
| 2026.10.18.2 | MPI inside SLURM jobs (first mochi run: Open MPI 5 refused 'allocation-overload': it binds each rank to a core, and the allocation had fewer cores than ranks): jobs ask for whole cores (`--hint=nomultithread`, unless `LMPT_SLURM_ARGS` sets --hint, --threads-per-core, --ntasks-per-core or --cpus-per-task); when ranks still outnumber the cores we may use, Open MPI binds to hardware threads (or none, oversubscribed) with a warning; `lammps.mpi` in results says how ranks were placed. |
| 2026.10.18.1 | SLURM execution (llm-translators docs/EXECUTION.md): `LMPT_EXECUTOR=slurm`, one job per study (`job.sh`, `job.json`; `--ntasks` from `procs`/`LMPT_PROCS`, `--time` = timeout + 10 min), `LMPT_SLURM_MIN_MINUTES` with a time estimate, `"queue"` per study, `LMPT_SLURM_ARGS`; states queued/running/timed_out/out_of_memory/cancelled/failed from squeue, sacct or job.log; `lmpt run-study`, `lmpt jobs`, `lmpt cancel` and the `jobs`, `cancel` MCP tools; `executor` in `tool_status`; benchmarks through the executor; local cancel (LAMMPS stopped, study ends `cancelled`); results written atomically. Tests with fake sbatch/squeue/scancel/sacct/sinfo (18). |
| 2026.10.16.2 | LAMMPS discovery: also `lmp_stable`, `lmp_openmpi` and /usr/bin, /usr/local/bin, /opt/lammps/bin, ~/.local/bin; `LMPT_LMP` with `~`; a missing LAMMPS stops the study with what to set (it was reported as 'no log'), and a run that writes no log reports LAMMPS's own error. 27 tests. Server results (LAMMPS 4 Jul 2026, carbonio): 27 tests, 9/9 benchmarks pass (with the fea-translator chain). |
| 2026.10.16.1 | First version: lattices (11 kinds, orient, substitution), packed molecules (SPC/E, TIP3P, methane UA, custom), data files; EAM/ADP/Tersoff/SW/MEAM/AIREBO/SNAP files and LJ/Morse/Buckingham + Coulomb (Ewald, PPPM); force fields with harmonic bonds/angles, OPLS dihedrals, SHAKE; stages energy, minimize, nve/nvt/npt (RDF, MSD), elastic, eos, vacancy, surface, thermal_expansion; reference implementation; 10 benchmarks; feat_material; MCP server with pending runs, progress and ETA. Tested with LAMMPS 22 Jul 2025 (PyPI wheel, aarch64): 26 tests, 8 default benchmarks pass. |

Author: Nicola Ferralis <ferralis@generalflash.com>
