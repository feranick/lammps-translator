"""workers.py — lmpt's worker processes end with the server that started them (from llmt-runtime).
Version: 2026.11.09.1

    watch_parent(ppid)   initializer of the study pool (api._Pool): a daemon thread ends the worker, and the LAMMPS
                         process groups it started (lammps.kill_all), when its parent (or the server ppid) is gone;
                         SIGTERM (Pool.terminate, the stale-server exit) does the same

The code is llmt-runtime's workers module (llmt_runtime.workers): polling os.getppid(), not PR_SET_PDEATHSIG (it fires
when the *thread* that created the child exits, and the pool is created from the MCP server's call threads). lmpt's own
list of LAMMPS runs (lammps.kill_all) is the runtime's on_orphan cleanup. Before 2026.11.09.1 the stale-server exit's
pool.terminate() ended the workers at once and their mpirun and LAMMPS ranks, in their own sessions, ran on.
"""
from __future__ import annotations

from llmt_runtime import workers as _W

PARENT_POLL_S = _W.PARENT_POLL_S
_STARTED = _W._STARTED


def _kill_lammps() -> None:
    try:
        from .lammps import kill_all
        kill_all()
    except Exception:  # noqa: BLE001 (the worker ends anyway)
        pass


def watch_parent(ppid: int | None = None) -> None:
    """Starts (once per process) the watcher that ends this worker, and the LAMMPS runs it started, when its parent
    `ppid` (default: the parent now) is gone (llmt_runtime.workers.watch_parent)."""
    _W.PARENT_POLL_S = PARENT_POLL_S       # (a test may set a shorter poll here)
    _W.on_orphan(_kill_lammps)
    _W.watch_parent(ppid, name="lmpt")
