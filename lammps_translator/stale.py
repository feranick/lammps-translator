"""stale.py — a server that runs older code than what is installed (llm-translators docs/SHARED.md).
Version: 2026.10.25.1

A shared install has <prefix>/releases/<name>/ (with RELEASE.json) and <prefix>/current -> releases/<name>; a server
keeps the release it started from. It is stale when `current` points to another release (or, in a checkout, when
lammps_translator/__init__.py on disk carries another version). Then, as omct 2026.10.23.3 and cfdt 2026.10.25.4:

- a call that finds the server idle (no other call, no study in this server's worker pool) is refused with a
  ToolError containing STALE_SERVER, and the server exits; llm-translators' router repeats the call on a new one;
- an idle stale server exits by itself within LMPT_STALE_CHECK_S (20) seconds;
- a busy one answers, with a 'server_stale' note, and exits once idle.

Only a server started by mcp_server.main() ('serving') refuses or exits: tests (llmt deploy --test runs them in a
release that is not current yet) and `lmpt run-study` run normally. LMPT_EXIT_WHEN_STALE: 1 always, 0 never;
default: shared installs only.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

STALE_MARK = "STALE_SERVER"
VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)
STATE = {"serving": False, "inflight": 0, "exiting": False, "checked": 0.0, "why": None, "shared": False,
         "pkg": None}
_LOCK = threading.Lock()


def find_release(start: Path) -> Path | None:
    """The release folder (releases/<name>/ with RELEASE.json) that contains start, or None."""
    for p in [start, *start.parents]:
        if (p / "RELEASE.json").is_file() and p.parent.name == "releases":
            return p
    return None


def stale_reason(pkg_dir: Path, running: str) -> tuple[str | None, bool]:
    """(why this process runs old code, or None; is it a shared install?)."""
    rel = find_release(pkg_dir)
    if rel is not None:
        cur = rel.parent.parent / "current"
        try:
            if cur.exists() and cur.resolve() != rel.resolve():
                return f"release {cur.resolve().name} is installed; this server runs {rel.name} ({running})", True
        except OSError:
            pass
        return None, True
    try:
        m = VERSION_RE.search((pkg_dir / "__init__.py").read_text())
    except OSError:
        return None, False
    if m and m.group(1) != running:
        return f"version {m.group(1)} is installed; this server runs {running}", False
    return None, False


def check() -> str | None:
    """stale_reason for this process, looked at most every 2 s."""
    now = time.time()
    if now - STATE["checked"] > 2.0:
        from . import __version__
        pkg = STATE["pkg"] or Path(__file__).resolve().parent
        STATE["why"], STATE["shared"] = stale_reason(Path(pkg), __version__)
        STATE["checked"] = now
    return STATE["why"]


def may_exit() -> bool:
    if not STATE["serving"]:
        return False
    v = os.environ.get("LMPT_EXIT_WHEN_STALE", "").strip()
    return v == "1" or (v != "0" and STATE["shared"])


def pool_busy() -> bool:
    """Studies running in this process's worker pool (they would die with it). SLURM jobs do not count."""
    try:
        from .api import _POOL
    except Exception:  # noqa: BLE001
        return False
    return any(not h.ready() for jobs in _POOL.jobs.values() for h in jobs.values())


def _exit_soon(why: str, delay: float) -> None:
    """Exits after `delay` s, so an answer in flight reaches the caller first. Caller holds _LOCK."""
    if STATE["exiting"]:
        return
    STATE["exiting"] = True
    sys.stderr.write(f"lmpt: {STALE_MARK}: exiting to pick up the new code: {why}\n")
    sys.stderr.flush()

    def go():
        time.sleep(delay)
        try:
            from .api import _POOL
            if _POOL.pool is not None:
                _POOL.pool.terminate()
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)
    threading.Thread(target=go, daemon=True, name="lmpt-exit").start()


def enter_call() -> str | None:
    """Counts a call in. Returns a STALE_SERVER message (the call is refused) when this server is stale, may exit
    and has nothing else running; otherwise None."""
    with _LOCK:
        why = check()
        if why and may_exit() and (STATE["exiting"] or (not STATE["inflight"] and not pool_busy())):
            _exit_soon(why, 1.0)
            return (f"{STALE_MARK}: {why}. This server is exiting so that the next one runs the installed version: "
                    "call again.")
        STATE["inflight"] += 1
        return None


def leave_call() -> None:
    with _LOCK:
        STATE["inflight"] -= 1


def note() -> str | None:
    """The 'server_stale' note for an answer from a stale server that is still serving."""
    if not STATE["why"]:
        return None
    return STATE["why"] + (": this answer comes from the old code; the server restarts when idle" if may_exit()
                           else ": this answer comes from the old code; restart the MCP server")


def watch(period: float | None = None) -> None:
    """Exits this server when it is stale, may exit and is idle."""
    period = float(period if period is not None else os.environ.get("LMPT_STALE_CHECK_S", "20"))

    def loop():
        while True:
            time.sleep(period)
            with _LOCK:
                if check() and may_exit() and not STATE["inflight"] and not pool_busy():
                    _exit_soon(STATE["why"], 0.0)
    threading.Thread(target=loop, daemon=True, name="lmpt-watch-install").start()


def serve(pkg: str | Path | None = None, period: float | None = None) -> None:
    """Called by mcp_server.main(): this process is a server; watch the install."""
    STATE["serving"] = True
    if pkg is not None:
        STATE["pkg"] = Path(pkg)
    STATE["checked"] = 0.0
    watch(period)
