# lammps-translator: context for a new session

Start a chat with: "Read CLAUDE.md and start."

## What this is

A deterministic **translator + checker** for LLMs: the LLM writes a structured study (the flexible part), the translator computes everything else (no model, no guessing), an independent checker confirms the result, and errors go back to the LLM as fixable messages. Same design as:

- **build123d-translator** (`b3dt`; github.com/feranick/build123d-translator): goal → build123d solid, STEP, script.
- **fea-translator** (`feat`; github.com/feranick/fea-translator): study → Gmsh mesh, CalculiX deck, results.
- **cfd-translator** (`cfdt`; github.com/feranick/cfd-translator): study → OpenFOAM case, results; long runs pending with ETA.

**This tool:**

- Name / CLI / MCP: `lammps-translator` / `lmpt` / `lmpt-mcp`; repo github.com/feranick/lammps-translator
- Domain: atomistic simulation; target LAMMPS 4 Jul 2026 (Debian package build: MPI + OpenMP, MANYBODY, KSPACE, MOLECULE, RIGID, MEAM, ML-SNAP, EXTRA-*; no REAXFF, no GPU), runs remotely on Ubuntu servers (carbonio, mochi; SLURM on a single node).
- Typical requests: "elastic constants and vacancy energy of Cu with EAM"; "surface energy of Si(100) with Tersoff"; "density, RDF and diffusion of SPC/E water at 300 K"; "thermal expansion of Cu-30Ni, then a feat model with that material".
- Study: `system` (lattice | molecules | data) + `potential` (metal units) or `forcefield` (real units) + ordered `stages` (energy, minimize, nve/nvt/npt, elastic, eos, vacancy, surface, thermal_expansion) + `expect`.
- Independent checks: reference implementation (pair styles, Ewald, EAM with LAMMPS's splines, bonded terms) on every run; virial vs energy (eos), Born, Cauchy, energy drift, masses; benchmarks (lattice sums, Madelung, Cauchy, FBD86 Cu, Tersoff Si, equipartition, NVE, SPC/E, feat chain).

## Interconnection

- With **fea-translator** only (not b3dt): results' `feat_material` {E MPa, nu, density kg/m3, alpha 1/K} is a feat `material` as is. Chain benchmark `chain_feat_cantilever` (skipped without `fea_translator`).
- One environment on carbonio: shared venv `~/Software/build123d-translator/.venv` (alias `cad`), `pip install -e ~/Software/lammps-translator[mcp,test]`.
- MCP servers may sit on different machines: pass specs between tools, not paths; `get_file` returns base64.

## Conventions

- Versions `YYYY.MM.DD.x` in every module docstring, `__init__.py`, `pyproject.toml`, and the README's version table; update the README with every change.
- Author: Nicola Ferralis <ferralis@generalflash.com>. I commit and push from the Mac; carbonio clones in `~/Software/`.
- Keep answers concise; give commands ready to paste. Python ≥ 3.10; extras `[mcp, test, lammps]`.

## Lessons (from b3dt, feat, cfdt and this one)

1. Research the target first; list unverified assumptions in the README, each with what catches it.
2. Never rely on the program's numbering: atoms, types, coefficients and strains are ours; LAMMPS only evaluates.
3. Exact checks where possible (reference vs LAMMPS: 1e-14), loose only with a reason (erfc polynomial 2e-7, kspace accuracy).
4. Every run checks itself. 5. Validate before running; every error says the fix.
6. Compute things the way the target does: EAM files are read line by line (rest of line discarded); tail corrections' pressure is not −dE/dV (box relax stalls).
7. Keep `study_format` compact (~11.5 kB). 8. Long runs: pending + progress + ETA; tell the user when to come back.
9. The sandbox kills background processes at the end of each call (~3 min cap): run long benchmarks on the servers.

## Environment

- Mac (Apple Silicon): development, Claude desktop. Sandbox here: aarch64 Ubuntu 22.04, LAMMPS from the PyPI wheel (`pip install lammps mpich`, `LD_LIBRARY_PATH=~/.local/lib`, wrapper as `LMPT_LMP`).
- Servers (Ubuntu, LAMMPS 20260704 at /usr/bin/lmp): MCP over ssh:

```json
"lmpt": {"command": "/usr/bin/ssh",
         "args": ["-T", "-o", "BatchMode=yes", "nicola@carbonio",
                  "/home/nicola/Software/build123d-translator/.venv/bin/lmpt-mcp"]}
```

## State

- 2026.10.20.1: per-study `run` settings and LMPT_MAX_* caps (EXECUTION.md "Per-study run settings", as cfdt 2026.10.20.1); ranks capped at atoms/LMPT_ATOMS_PER_RANK (EXECUTION.md "Sizing jobs"). 62 tests here. Not done from that section: run_benchmarks returning without waiting + bench_status(bench) (cfdt, feat have it; lmpt's run_benchmarks waits up to 45 s and takes request=...); a measured LMPT_ATOMS_PER_RANK on mochi.
- 2026.10.18.6 (status shows LMPT_SLURM_ARGS --mem). Mochi via MCP: jobs 2516/2517 side by side, 9 cores each, reliable; check_local stayed local; config/env there has LMPT_SLURM_ARGS=--mem=8G (overrides auto).
- 2026.10.18.5: LMPT_SLURM_MEM=auto (default), LMPT_SLURM_TIME=estimate (option), --ntasks-per-core=1 (as EXECUTION.md now says), `execution` run record. mochi showed why memory matters: CR_CORE_MEMORY + DefMemPerNode=UNLIMITED gives each job 60000M, so jobs ran one at a time; long time limits (6 h 10 min) also stop backfill.
- 2026.10.18.4: strip NULs from error text. Mochi: three 9-rank NVT at once (jobs 2484-2486) all done and reliable; read by the MCP jobs tool from a different server process. Not yet seen: the third job waiting (PD) while two ran (check sacct Start times), and collecting after an app restart.
- carbonio, 2026.10.18.3, SLURM: madelung_nacl job 5515 pass; NVT 2 ranks job 5516: 2 cores, binding default, reliable, T 39.8 K; jobs/--active worked. mochi, 2026.10.18.3: madelung_nacl job 2482 pass; NVT 9 ranks job 2483: 9 cores, 18 hwthreads, binding default, reliable, 12.8 s (the allocation-overload is fixed). Still to do: two requests from two server processes queueing, MCP reconnect collecting a running job.
- 2026.10.18.3: `lmpt bench --queue`. carbonio (12 CPUs): madelung_nacl passed with 2026.10.18.2, but ran locally: bin/run sources config/env after the command line, so its LMPT_SLURM_MIN_MINUTES=1 beat the inline 0. SLURM path still to confirm on carbonio and mochi.
- 2026.10.18.2: first mochi SLURM run failed in mpirun ('allocation-overload', Open MPI 5/PRRTE: ranks bound to cores, fewer cores than ranks in the allocation; mochi: 40 CPUs, LMPT_PROCS 9). Fix: --hint=nomultithread in job.sh, and a binding fallback (lammps.mpi_plan). To rerun on mochi.
- 2026.10.18.1: SLURM execution per llm-translators docs/EXECUTION.md (executor.py; `lmpt run-study`, `jobs`, `cancel`; states; `executor` in status). Here: 45 tests pass (18 SLURM tests with tests/fakeslurm.py, real LAMMPS 22 Jul 2025 inside the fake jobs). To do on mochi: `LMPT_EXECUTOR=slurm` with `madelung_nacl` and a short NVT run; two requests from two processes queue rather than oversubscribe; close the MCP connection while a job runs, reconnect, collect.
- 2026.10.16.2: clearer 'LAMMPS not found' (first server run reported 'no log' for every benchmark: LAMMPS was not found there; tests skipped). 2026.10.16.1 (first version). Here, with LAMMPS 22 Jul 2025: 27 tests pass; default benchmarks 8/8 pass; reference check exact (≤1e-13) for lj, morse, buck, eam, eam/alloy, eam/fs, bonded terms; Ewald 2e-7 (LAMMPS's erfc). SPC/E short run: ρ 0.999 ± 0.003, U −46.78 kJ/mol. feat chain via MCP: Cu E = 124.7 GPa, ν 0.35 → cantilever 1.2786 vs 1.2832 mm.
- carbonio (LAMMPS 4 Jul 2026 at /usr/bin/lmp, potentials /usr/share/lammps/potentials, fea-translator 2026.10.16.9 in the venv): 27 tests pass; `lmpt bench` 9/9 pass, same numbers as 22 Jul 2025 (equipartition 0.04 %), chain_feat_cantilever 0.36 %.
- spce_water on carbonio: pass (ρ 0.9972 g/cm3, U −46.60 kJ/mol, O–O peak 2.725 Å; 535 s). All 10 benchmarks pass with LAMMPS 4 Jul 2026.
- Seen once in fea-translator 2026.10.16.9: the first run of a cantilever after a pause gave a wrong field (tip 0.968 mm, peak 53 GPa, lateral motion) with the same mesh counts; the identical study rerun gave 1.2786. Its equilibrium check (3.6e-7) did not catch it.
- Open ideas: reference implementations of Tersoff and SW; hybrid (molecules on surfaces); tensile MD (stress–strain); phonons (fix phonon); Green–Kubo viscosity and thermal conductivity; melting point by two-phase coexistence; Yeh–Hummer correction for D; chemical potentials for compound defects.
