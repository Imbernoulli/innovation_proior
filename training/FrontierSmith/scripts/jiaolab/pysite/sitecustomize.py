"""jiaolab wrapper-layer patches for ALE-Bench.

Loaded automatically (python imports `sitecustomize` from sys.path) when the
jiaolab wrappers prepend scripts/jiaolab/pysite to PYTHONPATH. No historical
file -- and no gpublaze file -- is edited.

One patch, because jiaolab has exactly one usable sandbox:

  ALE_BENCH_CONTAINER_BACKEND=apptainer -> install the EAGER, CPU-PINNED
  apptainer backend (ale_apptainer_backend.py). docker on this box needs sudo
  (unavailable) and there is no bwrap, so the gpublaze `host` backend cannot run
  here at all.

Why not simply rely on ALE-Bench's own shipped apptainer backend
(ale_bench.utils._ApptainerContainer)?  Two reasons, both in the module
docstring of ale_apptainer_backend.py:

  * the shipped one is LAZY (spawns inside wait()), and every exception out of
    wait() is converted by the harness into a COMPILATION_ERROR CaseResult --
    i.e. a broken sandbox would be scored as a silent zero.  Ours spawns inside
    containers.run(), so infra failures surface as AleInfraError;
  * the shipped one drops the harness's `cpu_quota=100000` (1 CPU) request.
    ALE-Bench scoring is wall-clock sensitive and jiaolab is a 128-core box
    shared with another user, so each sandbox leases a distinct core.

If this file is NOT on PYTHONPATH the harness still works -- it falls back to
ALE-Bench's shipped apptainer backend -- but with the weaker failure semantics
above. The jiaolab eval client always sets PYTHONPATH, and refuses to run if the
backend cannot be installed.
"""
from __future__ import annotations

import os

if os.environ.get("ALE_BENCH_CONTAINER_BACKEND", "").lower() == "apptainer":
    try:
        import ale_apptainer_backend

        ale_apptainer_backend.install()
    except Exception as exc:  # sitecustomize exceptions only print; make it loud
        import sys
        import traceback

        traceback.print_exc()
        print(
            f"sitecustomize: FATAL: ALE_BENCH_CONTAINER_BACKEND=apptainer but the apptainer "
            f"backend failed to install ({exc!r}). ALE scoring on this interpreter is NOT safe.",
            file=sys.stderr,
        )

if os.environ.get("ALE_BENCH_CONTAINER_BACKEND", "").lower() == "host":
    import sys

    print(
        "sitecustomize: REFUSING ALE_BENCH_CONTAINER_BACKEND=host on jiaolab: the gpublaze host "
        "backend needs bwrap, which is not installed here. Use apptainer.",
        file=sys.stderr,
    )
