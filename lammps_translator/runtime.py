"""runtime.py — lmpt's llmt-runtime: the executor code the translators share, with lmpt's settings and hooks.
Version: 2026.11.09.4

    RT                 llmt_runtime.Runtime for lmpt (LMPT_ settings, the `lmpt` CLI, MPI: --ntasks)
    RUNTIME_VERSION    llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.1): workers, out of memory (a LAMMPS run killed for memory is said as such, not as a time-out),
how a job that left the queue ended (ended_state), version recording and ran_with, request_name come from
llmt-runtime. Phase 2 (2026.11.09.3): the SLURM layer (settings, the machine, caps, LMPT_SLURM_TIME, job.sh and
job.json, submission, squeue/sacct states, the queue position by priority, scancel, the machine's CPUs) is
llmt-runtime's too, with lmpt's sizing as hooks: the ranks (executor.ranks_for: atoms / LMPT_ATOMS_PER_RANK and the
node's cores), the memory (executor.mem_auto: 1 GB + 256 MB per rank + 4 kB per atom), job.json's natoms,
estimate_min and procs_requested (executor.job_extra), the progress of running jobs (api.progress), the out-of-memory
advice and the timeout hint. The job's resource lines are the runtime's MPI lines without --nodes=1 (--ntasks=N,
--ntasks-per-core=1 unless LMPT_SLURM_ARGS places the ranks), as lmpt wrote them; cancel()'s flag file keeps lmpt's
name, cancelled.json. Phase 3 (2026.11.09.4): the MCP server's common code is llmt-runtime's server.Server, configured
with RT in stale.py (GUARD: stale servers, the tool wrapper with its 48 s deadline, the server's log, signals).

The hooks are small wrappers that import executor and api when called: executor imports RT from here.
"""
from __future__ import annotations

import json
from pathlib import Path

from llmt_runtime import Hooks, Runtime
from llmt_runtime import __version__ as RUNTIME_VERSION  # noqa: F401

from . import __version__


def oom_advice(folder: Path) -> str:
    """Why and what to ask for when SLURM killed a study's job for memory: the job's --mem and ranks (job.json)."""
    try:
        job = json.loads((Path(folder) / "job.json").read_text())
    except (OSError, ValueError):
        job = {}
    n, mem = int(job.get("procs") or 1), job.get("mem")
    had = f" ({mem} with {n} rank{'s' if n != 1 else ''})" if mem else ""
    atoms = f" for {int(job['natoms']):,} atoms" if job.get("natoms") else ""
    return (f"SLURM killed the job for memory{had}{atoms}: give the study more, \"run\": {{\"mem\": \"<more>\"}} "
            "(within study_format's caps; it replaces a --mem in LMPT_SLURM_ARGS), or use a smaller cell")


def _study_procs(study: dict):
    """The study's own ranks: its top-level procs (lmpt studies have no solve block; 2026.11.09.1 read solve.procs)."""
    return study.get("procs")


def _procs(study: dict) -> int:
    from .executor import atoms, ranks_for
    return ranks_for(study, atoms(study))


def _mem_auto(study: dict) -> str:
    from .executor import mem_auto
    return mem_auto(study)


def _job_extra(study: dict) -> dict:
    from .executor import job_extra
    return job_extra(study)


def _running_progress(folder: Path, study: dict) -> dict:
    from .api import progress
    return progress(Path(folder))


RT = Runtime(prefix="LMPT", cli="lmpt", package="lammps_translator", version=__version__, kind="mpi",
             out_default="~/lmpt_runs", min_minutes=1.0, timeout_default=21600.0, slurm_time_default="timeout",
             cancel_file="cancelled.json",
             hooks=Hooks(study_procs=_study_procs, procs=_procs, mem_auto=_mem_auto, job_extra=_job_extra,
                         oom_advice=oom_advice, running_progress=_running_progress,
                         timeout_hint="or shorten the study (fewer MD steps, a smaller cell), or give it more procs"))
