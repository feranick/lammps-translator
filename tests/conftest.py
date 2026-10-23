"""Test isolation: the machine's config/env (LMPT_EXECUTOR, LMPT_PROCS, LMPT_SLURM_*, caps) must not change what
the tests see, and no test may submit real SLURM jobs. Version: 2026.10.22.1"""
import os

import pytest

_KEYS = ("LMPT_EXECUTOR", "LMPT_PROCS", "LMPT_SLURM_ARGS", "LMPT_SLURM_MEM", "LMPT_SLURM_TIME", "LMPT_SLURM_TIME_FACTOR",
         "LMPT_SLURM_TIME_MIN", "LMPT_SLURM_MIN_MINUTES", "LMPT_MAX_TIMEOUT", "LMPT_MAX_MEM", "LMPT_MAX_PROCS",
         "LMPT_ATOMS_PER_RANK", "LMPT_TIMEOUT", "LMPT_WORKERS", "LMPT_MPIRUN")


@pytest.fixture(autouse=True)
def _clean_lmpt_env(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    for k in [k for k in os.environ if k.startswith("SLURM_")]:
        monkeypatch.delenv(k, raising=False)
