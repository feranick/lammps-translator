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

- 2026.10.18.2: first mochi SLURM run failed in mpirun ('allocation-overload', Open MPI 5/PRRTE: ranks bound to cores, fewer cores than ranks in the allocation; mochi: 40 CPUs, LMPT_PROCS 9). Fix: --hint=nomultithread in job.sh, and a binding fallback (lammps.mpi_plan). To rerun on mochi.
- 2026.10.18.1: SLURM execution per llm-translators docs/EXECUTION.md (executor.py; `lmpt run-study`, `jobs`, `cancel`; states; `executor` in status). Here: 45 tests pass (18 SLURM tests with tests/fakeslurm.py, real LAMMPS 22 Jul 2025 inside the fake jobs). To do on mochi: `LMPT_EXECUTOR=slurm` with `madelung_nacl` and a short NVT run; two requests from two processes queue rather than oversubscribe; close the MCP connection while a job runs, reconnect, collect.
- 2026.10.16.2: clearer 'LAMMPS not found' (first server run reported 'no log' for every benchmark: LAMMPS was not found there; tests skipped). 2026.10.16.1 (first version). Here, with LAMMPS 22 Jul 2025: 27 tests pass; default benchmarks 8/8 pass; reference check exact (≤1e-13) for lj, morse, buck, eam, eam/alloy, eam/fs, bonded terms; Ewald 2e-7 (LAMMPS's erfc). SPC/E short run: ρ 0.999 ± 0.003, U −46.78 kJ/mol. feat chain via MCP: Cu E = 124.7 GPa, ν 0.35 → cantilever 1.2786 vs 1.2832 mm.
- carbonio (LAMMPS 4 Jul 2026 at /usr/bin/lmp, potentials /usr/share/lammps/potentials, fea-translator 2026.10.16.9 in the venv): 27 tests pass; `lmpt bench` 9/9 pass, same numbers as 22 Jul 2025 (equipartition 0.04 %), chain_feat_cantilever 0.36 %.
- spce_water on carbonio: pass (ρ 0.9972 g/cm3, U −46.60 kJ/mol, O–O peak 2.725 Å; 535 s). All 10 benchmarks pass with LAMMPS 4 Jul 2026.
- Seen once in fea-translator 2026.10.16.9: the first run of a cantilever after a pause gave a wrong field (tip 0.968 mm, peak 53 GPa, lateral motion) with the same mesh counts; the identical study rerun gave 1.2786. Its equilibrium check (3.6e-7) did not catch it.
- Open ideas: reference implementations of Tersoff and SW; hybrid (molecules on surfaces); tensile MD (stress–strain); phonons (fix phonon); Green–Kubo viscosity and thermal conductivity; melting point by two-phase coexistence; Yeh–Hummer correction for D; chemical potentials for compound defects.
