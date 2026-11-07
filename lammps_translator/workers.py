"""workers.py — lmpt's worker processes end with the server that started them.
Version: 2026.11.7.1

    watch_parent(ppid)   initializer of the study pool (api._Pool): a daemon thread ends the worker, and the LAMMPS
                         process groups it started (lammps.kill_all), when its parent is gone

A spawned multiprocessing worker holds both ends of its own task queue, so it never sees EOF when the MCP server
dies (a crash, kill -9, an app restart) and would wait forever, holding its release "in use" for llmt gc; one in the
middle of a local run would also leave mpirun and its LAMMPS ranks running (they are in their own session, so a
signal to the worker does not reach them). The stale-server exit terminates the pool itself (stale.py); this is the
backstop for everything else, as cfdt 2026.11.07.2 and feat do (llm-translators docs/create_new_translators.md).

Not PR_SET_PDEATHSIG: on Linux it fires when the *thread* that created the child exits, and the pool is created
from the MCP server's call threads, which come and go. The worker polls os.getppid() instead (PARENT_POLL_S): when
the parent dies the worker is re-parented and its parent pid changes. SLURM jobs do not depend on the server.
"""
from __future__ import annotations

import os
import threading
import time

PARENT_POLL_S = 1.0
_STARTED = {"pid": None}        # the process the watcher runs in (a fork would inherit the flag)


def _orphaned() -> None:
    try:
        from .lammps import kill_all
        kill_all()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


def watch_parent(ppid: int | None = None) -> None:
    """Starts (once per process) a daemon thread that ends this process, and the LAMMPS runs it started, when its
    parent `ppid` (default: the parent now) is gone."""
    if _STARTED["pid"] == os.getpid():
        return
    _STARTED["pid"] = os.getpid()
    ppid = int(ppid or os.getppid())
    if os.getppid() != ppid:      # the parent died before the worker started
        _orphaned()

    def loop():
        while True:
            time.sleep(PARENT_POLL_S)
            if os.getppid() != ppid:
                _orphaned()
    threading.Thread(target=loop, daemon=True, name="lmpt-parent-watch").start()
