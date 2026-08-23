# Copyright 2025 - ALE-Bench compile substrate self-test
"""One-time COMPILE STARTUP SELF-TEST for the ALE-Bench scoring path.

RESIDUAL VERIFIER RISK this closes
----------------------------------
ALE-Bench scores a model's C++ by compiling it inside a container (Docker or,
on the cluster, an apptainer ``.sif``) via
``ale_bench.tool_wrappers.case_runner.run_compile_container``. The compile-bug
audit already made a *missing* ``a.out`` map to a clean ``COMPILATION_ERROR``
(reward 0) instead of crashing, and made genuine infra failures raise
``AleInfraError`` instead of being silently scored 0.

But there is a residual, *systematic* failure mode: if a compute node's compile
substrate is broken for EVERY program (bad apptainer image path, missing
compiler, unwritable bind mount, broken container runtime), then every single
submission -- including a perfectly valid one -- is classified
``COMPILATION_ERROR`` -> reward 0. Across a ~45-model campaign that reads as
"all models are weak" (a fake uniform-zero) rather than what it is: an infra
outage on that node.

This module compiles a MINIMAL, KNOWN-GOOD C++ program (``int main(){return 0;}``)
through the *same* compile-container code path used for real scoring, BEFORE any
scoring begins. If that trivial program does not produce a binary, the substrate
is broken and we refuse to score (raising the caller's ``AleInfraError``) instead
of emitting a fake all-zeros result.

It runs ONCE per process (module-level cached flag) and only does the compile
step -- it does NOT run test cases, start a session, or load problem data -- so it
is fast (a single container compile, ~a few seconds) and adds zero per-sample
overhead.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Type

logger = logging.getLogger(__name__)

# Minimal known-good C++17 program. If THIS does not compile, the substrate is
# broken (not the model). Kept trivial on purpose so a failure is unambiguous.
_KNOWN_GOOD_CPP = "int main(){return 0;}\n"

# Module-level one-shot cache. ``True`` once the self-test has passed in this
# process; subsequent calls are a no-op. Guarded by a lock so a thread pool that
# fans out scoring cannot run the compile probe more than once.
_SELFTEST_PASSED = False
_SELFTEST_LOCK = threading.Lock()


class AleCompileSelfTestError(RuntimeError):
    """Default error type raised when the compile self-test fails.

    Callers that already have their own infra-error type (``alebench_full`` and
    ``alebench`` each define an ``AleInfraError``) pass it via ``error_cls`` so
    the failure surfaces with the type the caller's error handling expects.
    """


def _run_known_good_compile() -> str | None:
    """Compile the known-good program through the real compile-container path.

    Returns ``None`` on success, or a human-readable reason string on failure.
    Exercises exactly the functions ``run_cases`` uses (``setup_paths_compile``,
    ``get_compile_volumes``, ``build_compile_command``, ``run_compile_container``)
    and the same container backend (``docker_client`` -> docker or apptainer),
    so a pass here means the real scoring compile path works on this node.
    """
    # Imported lazily so importing this helper never forces the ALE-Bench /
    # docker stack to import (e.g. in unit tests that never score ALE).
    from ale_bench.code_language import CodeLanguage, JudgeVersion
    from ale_bench.tool_wrappers.case_runner import (
        build_compile_command,
        get_compile_volumes,
        run_compile_container,
        setup_paths_compile,
    )

    code_language = CodeLanguage.CPP17
    judge_version = JudgeVersion(os.environ.get("ALEBENCH_JUDGE_VERSION", "202301"))

    with tempfile.TemporaryDirectory(prefix="ale-compile-selftest-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        host_paths_compile = setup_paths_compile(temp_dir, _KNOWN_GOOD_CPP, code_language, judge_version)
        compile_volumes = get_compile_volumes(host_paths_compile, temp_dir)
        compile_command = build_compile_command(
            code_language, judge_version, host_paths_compile.object_file.relative_to(temp_dir)
        )
        # run_compile_container returns None on success, or a CaseResult
        # (judge_result == COMPILATION_ERROR) if the build artifact is missing /
        # the compiler exited non-zero / the container timed out. For a trivial
        # valid program that can only mean a broken substrate.
        compile_result = run_compile_container(
            code_language,
            judge_version,
            host_paths_compile,
            compile_volumes,
            compile_command,
        )

    if compile_result is None:
        return None
    return getattr(compile_result, "message", str(compile_result))


def ale_compile_selftest(error_cls: Type[Exception] = AleCompileSelfTestError, *, force: bool = False) -> None:
    """Verify the ALE compile substrate ONCE per process; raise if broken.

    Compiles ``int main(){return 0;}`` through the same compile-container path
    used for real scoring. On success it sets a module-level cached flag so
    subsequent calls return immediately (no per-sample overhead). On failure --
    the trivial program did not produce a binary, or the container backend itself
    errored -- it raises ``error_cls`` so the caller refuses to score instead of
    silently emitting a fake uniform-zero.

    Args:
        error_cls: Exception type to raise on failure. Callers pass their own
            ``AleInfraError`` so the failure flows through existing handling.
        force: Re-run even if a previous call already passed (used by validation
            harnesses to exercise the raise path). Does not bypass the cache for
            normal callers.
    """
    global _SELFTEST_PASSED

    if _SELFTEST_PASSED and not force:
        return

    with _SELFTEST_LOCK:
        # Re-check under the lock: another thread may have just passed it.
        if _SELFTEST_PASSED and not force:
            return

        logger.info("Running ALE compile self-test (known-good C++ through the real compile container)...")
        try:
            failure_reason = _run_known_good_compile()
        except Exception as exc:
            # The container backend / wrapper itself blew up (e.g. apptainer image
            # missing, docker daemon down, bind-mount unwritable). That is a broken
            # substrate, full stop.
            logger.exception("ALE compile self-test raised before completing")
            raise error_cls(
                "ALE compile self-test FAILED -- compile substrate is broken; refusing to score "
                f"(would produce fake uniform-zero). Backend error: {exc!r}"
            ) from exc

        if failure_reason is not None:
            logger.error("ALE compile self-test classified a known-good program as a compile error: %s", failure_reason)
            raise error_cls(
                "ALE compile self-test FAILED -- compile substrate is broken; refusing to score "
                f"(would produce fake uniform-zero). A trivial valid C++ program failed to compile: {failure_reason}"
            )

        _SELFTEST_PASSED = True
        logger.info("ALE compile self-test PASSED -- compile substrate is healthy.")


def reset_selftest_cache() -> None:
    """Reset the one-shot cache. For tests / validation harnesses only."""
    global _SELFTEST_PASSED
    with _SELFTEST_LOCK:
        _SELFTEST_PASSED = False
