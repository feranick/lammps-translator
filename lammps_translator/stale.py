"""stale.py — the MCP server's guard: llmt-runtime's Server for lmpt (stale servers after a deploy, llm-translators
docs/SHARED.md; the tool wrapper; the server's log; signals).
Version: 2026.11.09.4

    GUARD              llmt_runtime.server.Server(RT, api.__file__, busy=pool_busy, shutdown=stop_workers)
    pool_busy()        studies running in this process's worker pool (they would die with it)
    stop_workers()     the pool's workers stopped (llmt_runtime.workers.stop_pool: SIGTERM, then SIGKILL)

From 2026.11.09.4 (phase 3) this module is a thin layer over llmt-runtime's server.Server; mcp_server wraps every tool
with GUARD.tool(ToolError) and main() calls GUARD.serve(). The former names stay (STALE_MARK, STATE, stale_reason,
check, may_exit, enter_call, leave_call, note, watch, serve) and act on GUARD. As before (omct 2026.10.23.3, cfdt
2026.10.25.4):

- a shared install has <prefix>/releases/<name>/ (with RELEASE.json) and <prefix>/current -> releases/<name>; a server
  keeps the release it started from and is stale when `current` points to another release (in a checkout: when
  lammps_translator/__init__.py on disk carries another version; now also llmt-runtime's __init__.py);
- a call that finds the server idle is refused with a ToolError containing STALE_SERVER, and the server exits;
  llm-translators' router repeats the call on a new one;
- an idle stale server exits by itself within about two LMPT_STALE_CHECK_S periods (10 s; it was 20);
- a busy one answers, with a 'server_stale' note, and exits once idle.

Idle: no call in flight, no study in this server's worker pool, and no background work (the rest of a request's
SLURM submissions, the fea-translator chain of a benchmark: api.background runs them through GUARD.start_background
in a server, so a stale exit waits for them; it used to kill the submissions). The exit stops the pool's workers
(stop_workers, at most llmt_runtime.server.SHUTDOWN_S), then exits whatever happens.

Only a server started by mcp_server.main() ('serving') refuses or exits: tests (llmt deploy --test runs them in a
release that is not current yet) and `lmpt run-study` run normally. LMPT_EXIT_WHEN_STALE: 1 always, 0 never; default:
shared installs only.
"""
from __future__ import annotations

from pathlib import Path

from llmt_runtime.server import STALE_MARK, Server, stale_reason  # noqa: F401 (the former names)
from llmt_runtime.workers import stop_pool

from . import api
from .runtime import RT


def pool_busy() -> bool:
    """Studies running in this process's worker pool (they would die with it). SLURM jobs do not count."""
    return any(not h.ready() for jobs in api._POOL.jobs.values() for h in jobs.values())


def stop_workers() -> None:
    """The local pool's workers end, and their LAMMPS runs with them (SIGTERM, then SIGKILL after a grace period:
    Pool.terminate() alone could wait on a worker inside a long native call)."""
    stop_pool(api._POOL.pool)


GUARD = Server(RT, api.__file__, busy=pool_busy, shutdown=stop_workers)
STATE = GUARD.state           # {inflight, background, exiting, checked, why, shared, serving}


def check() -> str | None:
    """Why this server runs old code, or None (looked at most every 2 s)."""
    return GUARD.stale()


def may_exit() -> bool:
    return GUARD.may_exit()


def enter_call() -> str | None:
    """Counts a call in. Returns a STALE_SERVER message (the call is refused) when this server is stale, may exit
    and has nothing else running; otherwise None."""
    return GUARD.enter_call()


def leave_call() -> None:
    GUARD.leave_call()


def note() -> str | None:
    """The 'server_stale' note for an answer from a stale server that is still serving."""
    return GUARD.note()


def watch() -> None:
    """Exits this server when it is stale, may exit and is idle (every GUARD.period s)."""
    GUARD.watch()


def serve(pkg: str | Path | None = None, period: float | None = None, log: bool = True,
          signals: bool = True) -> None:
    """This process is a server (mcp_server.main() calls GUARD.serve()): pkg and period replace the package folder
    and LMPT_STALE_CHECK_S (tests); log: the server's log in $LMPT_OUT/.server; signals: SIGTERM/SIGHUP/SIGINT stop
    the workers first."""
    if pkg is not None:
        GUARD.pkg_dir = Path(pkg)
    if period is not None:
        GUARD.period = float(period)
    STATE["checked"] = 0.0
    GUARD.serve(log=log, signals=signals)
