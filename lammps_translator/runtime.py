"""runtime.py — lmpt's llmt-runtime: the executor code the translators share, with lmpt's settings and hooks.
Version: 2026.11.09.1

    RT                 llmt_runtime.Runtime for lmpt (LMPT_ settings, the `lmpt` CLI, MPI: --ntasks)
    RUNTIME_VERSION    llmt-runtime's version (tool_status().executor.runtime)

Phase 1 (2026.11.09.1): workers, out of memory (a LAMMPS run killed for memory is said as such, not as a time-out),
how a job that left the queue ended (ended_state), version recording and ran_with, request_name come from
llmt-runtime. The SLURM layer (settings, caps, job.sh, states, jobs, cancel) and the MCP server's stale handling
(stale.py) stay until phases 2 and 3.
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


RT = Runtime(prefix="LMPT", cli="lmpt", package="lammps_translator", version=__version__, kind="mpi",
             out_default="~/lmpt_runs", min_minutes=1.0, timeout_default=21600.0, slurm_time_default="timeout",
             hooks=Hooks(oom_advice=oom_advice,
                         timeout_hint="or shorten the study (fewer MD steps, a smaller cell), or give it more procs",
                         study_procs=lambda st: (st.get("solve") or {}).get("procs")))
