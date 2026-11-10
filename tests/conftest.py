"""Test isolation: the machine's config/env (LMPT_EXECUTOR, LMPT_PROCS, LMPT_SLURM_*, caps) must not change what
the tests see, and no test may submit real SLURM jobs; $USER is set (squeue -u, job listings), and the server's
worker pool is stopped when the session ends (else Pool.__del__ complains at exit). Version: 2026.11.09.6"""
import os

import pytest

_KEYS = ("LMPT_EXECUTOR", "LMPT_PROCS", "LMPT_SLURM_ARGS", "LMPT_SLURM_MEM", "LMPT_SLURM_TIME", "LMPT_SLURM_TIME_FACTOR",
         "LMPT_SLURM_TIME_MIN", "LMPT_SLURM_MIN_MINUTES", "LMPT_MAX_TIMEOUT", "LMPT_MAX_MEM", "LMPT_MAX_PROCS",
         "LMPT_ATOMS_PER_RANK", "LMPT_TIMEOUT", "LMPT_WORKERS", "LMPT_MPIRUN", "LMPT_SLURM_NODE",
         "LMPT_EXIT_WHEN_STALE")


@pytest.fixture(autouse=True)
def _clean_lmpt_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    for k in [k for k in os.environ if k.startswith("SLURM_")]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USER", os.environ.get("USER") or "test")       # unset under cron, some CI and sudo -E


@pytest.fixture(autouse=True, scope="session")
def _stop_server_pool():
    """api._POOL is the server's worker pool, started by the tests that run studies locally: stop it before the
    interpreter exits, so its __del__ does not run against a half torn-down multiprocessing."""
    yield
    from lammps_translator import api
    pool = api._POOL.pool
    if pool is not None:
        pool.terminate()
        pool.join()
        api._POOL.pool = None
        api._POOL.jobs.clear()
