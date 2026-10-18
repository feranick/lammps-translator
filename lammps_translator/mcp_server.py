"""mcp_server.py — MCP server: lets LLM-based tools run atomistic studies with LAMMPS.
Version: 2026.10.18.6

Tools (all JSON):
  study_format      the study format: systems, potentials, force fields, stages. Read it first.
  validate_studies  static checks of {name: study} (fast; fix errors before running)
  run_studies       runs and checks {name: study} (returns within ~45 s; long studies stay pending)
  study_status      collects the pending studies of a run_studies request
  get_inputs        the LAMMPS inputs and data file of one study, without running
  run_benchmarks    the benchmark studies with known answers (is the toolchain right?)
  jobs              recent and active SLURM runs (also from other conversations), the queue, the machine
  cancel            stops a request's studies
  get_file          a written file as base64 (for callers on another machine)

Run:  lmpt-mcp                       (stdio)
      lmpt-mcp --http 8769 [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse

from . import api, executor

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
    _V2 = True
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    _V2 = False

INSTRUCTIONS = """lammps-translator runs atomistic simulations with LAMMPS: crystals (fcc, bcc, hcp, diamond, rocksalt,
... with EAM, Tersoff, SW, MEAM, LJ, Morse, Buckingham + Ewald potentials) and molecular liquids (SPC/E, TIP3P,
custom molecules with harmonic bonds/angles, OPLS dihedrals). A study is a system, its interactions and ordered
stages: minimize (box relax), nve/nvt/npt MD (averages with errors, RDF, MSD/diffusion), and properties: energy,
elastic (Cij, VRH moduli), eos (Birch-Murnaghan), vacancy, surface energy, thermal_expansion. Results include a
fea-translator material ('feat_material': E, nu, density, alpha) for continuum models.
Workflow: call study_format once and follow its conventions; validate_studies and fix every error; run_studies
(long studies come back as 'pending': poll study_status with the request path). Check checks.reference (an
independent implementation recomputes LAMMPS's energy, forces and pressure on the initial structure), warnings
and 'reliable' before trusting numbers; if 'reliable' is false, do not report the numbers as results (fix what
'unreliable_because' says). MD numbers are statistical: report mean with err; run longer if err is large.
Pending studies report 'progress' with eta_min_max: polling costs tokens, so when a run needs more than about
5 minutes, tell the user when it should finish and stop; collect it later with study_status.
Long studies run as SLURM jobs on the machine (when LMPT_EXECUTOR=slurm): run_studies returns them pending with
job ids. They do not depend on this conversation. Report the expected finish time and stop; do not poll. Later,
study_status(request) collects them; jobs() lists recent and active runs (also those started in other
conversations); cancel(request) stops them.
Units: potentials eV, A (metal); force fields kcal/mol, A (real); T in K, P_bar in bar, dt_fs in fs; results in
eV, A, GPa, K, ps, g/cm3, J/m2."""

server = _Server("lammps-translator", instructions=INSTRUCTIONS)


@server.tool()
def study_format() -> dict:
    """The study format: systems (lattices, molecules, data files), potentials and force fields, stage kinds
    with their parameters, conventions, an example; and the LAMMPS version, packages and potential folders."""
    return api.study_format()


@server.tool()
def validate_studies(studies: dict) -> dict:
    """Static checks of studies given as {name: study}: builds each structure and finds its potential files.
    Returns per study ok, errors (each says what to fix), warnings."""
    return api.validate(studies)


@server.tool()
def run_studies(studies: dict, wait_s: float = 45) -> dict:
    """Builds, runs (LAMMPS) and checks each study. Per study: one result block per stage (energies, lattice
    constants, Cij, moduli, EOS fit, vacancy/surface energies, MD means and errors, RDF, diffusion), 'structure',
    'final', 'feat_material', 'checks' (reference implementation vs LAMMPS, Born stability, Cauchy, virial vs
    energy, energy drift), 'reliable', warnings and files. Returns after at most wait_s seconds: studies still
    running are listed in 'pending' with progress and ETA (with SLURM: state queued/running, job id); collect
    them with study_status. A study's "queue": true/false forces a SLURM job / a local run."""
    return api.run(studies, wait=min(max(float(wait_s), 0.0), 50.0))


@server.tool()
def study_status(request: str, wait_s: float = 45) -> dict:
    """The studies of a run_studies request (its 'request' path): finished ones with their results, the others
    in 'pending' with progress and state (queued: reason, expected_start, ahead; running: stage, eta_min_max,
    job_elapsed_min). Jobs that ended without a result (timed_out, out_of_memory, cancelled, failed) are results
    with ok false and why. Waits up to wait_s seconds (at most 50) for them to finish."""
    return api.status(request, min(max(float(wait_s), 0.0), 50.0))


@server.tool()
def get_inputs(study: dict, name: str = "study") -> dict:
    """Builds one study and writes its LAMMPS inputs without running: the structure summary, every stage's
    in.lammps as text and the head of the data file."""
    return api.inputs(study, name)


@server.tool()
def jobs(limit: int = 20, active: bool = False) -> dict:
    """The newest requests that have SLURM jobs (from any conversation), each study's state, job id and ETA;
    queue {running, queued, cpus_in_use} for this translator's jobs; machine {cpus, cpus_allocated}.
    active=True: only requests with queued or running studies."""
    return api.jobs(limit, active)


@server.tool()
def cancel(request: str, names: list[str] | None = None) -> dict:
    """Stops a request's studies (all, or the named ones): scancel for SLURM jobs; a local study only if this
    server process runs it. Returns what was cancelled and the request's states."""
    return api.cancel(request, names)


@server.tool()
def run_benchmarks(names: list[str] | None = None, request: str | None = None) -> dict:
    """Runs the benchmark studies with known answers (LJ lattice sums, Madelung constants, Cauchy relation,
    Cu EAM and Si Tersoff published values, equipartition, NVE conservation; slow: spce_water; chain with
    fea-translator: chain_feat_cantilever) and reports pass or fail. With SLURM, long cases become jobs: pending
    ones come back with 'request'; call run_benchmarks(request=...) later to evaluate them."""
    return api.bench(names, request, wait=45.0 if executor.kind() == "slurm" or request else None)


@server.tool()
def get_file(path: str, max_mb: float = 20) -> dict:
    """A file written by run_studies (a path from its result) as base64."""
    return api.read_file(path, max_mb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, help="serve MCP over streamable HTTP on this port (default: stdio)")
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.http is None:
        server.run()
    elif _V2:
        server.run(transport="streamable-http", host=a.host, port=a.http)
    else:
        server.settings.host, server.settings.port = a.host, a.http
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
