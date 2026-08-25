# Copyright 2025 - ALE-Bench no-docker (host) backend entry point
"""Host (no-docker) backend for the ALE-Bench reward / eval chain.

This is the RL-side entry point of the gpublaze no-docker judging port. The
actual sandbox implementation lives in the machine wrapper layer
(``scripts/gpublaze/pysite/ale_host_backend.py``) next to the other
gpublaze-specific patches; it is auto-installed by that directory's
``sitecustomize.py`` whenever ``ALE_BENCH_CONTAINER_BACKEND=host`` and the
pysite dir is on ``PYTHONPATH`` (both gpublaze wrappers -- the eval client and
the RL launcher -- arrange this).

What the host backend changes (and what it does NOT):
  * It swaps ONLY the container runtime: ``ale_bench.utils.docker_client`` is
    replaced with a bwrap-based local sandbox (exported image rootfs with the
    identical g++ 12.2.0 toolchain, 1-core taskset pinning, RLIMIT_AS memory
    cap, no network). The official ALE-Bench harness flow -- compile ->
    per-case run -> rust tester scoring -> standings rank -> performance -- is
    byte-for-byte the same code as the docker path.
  * Failure semantics are preserved: infra failures (missing bwrap / rootfs /
    spawn failure) raise out of ``containers.run()`` and surface through
    ``compute_score`` as ``AleInfraError``; model-side CE/WA/RE/TLE remain
    ordinary zero-score results. The existing ``ale_compile_selftest``
    exercises this backend too (it compiles a known-good program through the
    same ``run_compile_container`` path), so with ``backend=host`` the one-shot
    selftest IS the host-backend substrate check.

Usage from the RL side (identical to the eval side): run the trainer with
``ALE_BENCH_CONTAINER_BACKEND=host`` and the gpublaze pysite dir on PYTHONPATH;
``sitecustomize`` installs the backend in every interpreter. This module's
helpers let reward code assert that state explicitly.
"""

from __future__ import annotations

import os

_ENV_VAR = "ALE_BENCH_CONTAINER_BACKEND"


def host_backend_requested() -> bool:
    """True when the environment asks for the no-docker host backend."""
    return os.environ.get(_ENV_VAR, "").lower() == "host"


def host_backend_active() -> bool:
    """True when the host backend is actually installed in this interpreter."""
    try:
        import ale_host_backend
    except ImportError:
        return False
    return ale_host_backend.installed()


def install_host_backend() -> bool:
    """Explicitly install the host backend (normally done by sitecustomize).

    Raises RuntimeError when the wrapper-layer module is not importable --
    fail loud, mirroring the AleInfraError contract: a requested-but-missing
    backend must never silently degrade to docker or to zero scores.
    """
    try:
        import ale_host_backend
    except ImportError as exc:
        msg = (
            "ALE_BENCH_CONTAINER_BACKEND=host was requested but the gpublaze host "
            "backend (scripts/gpublaze/pysite/ale_host_backend.py) is not importable. "
            "Put scripts/gpublaze/pysite on PYTHONPATH (the gpublaze wrappers do this)."
        )
        raise RuntimeError(msg) from exc
    return ale_host_backend.install()
