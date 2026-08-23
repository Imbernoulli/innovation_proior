# Copyright 2025 - ALE-Bench reward for VERL RL training
"""ALE-Bench Docker judge reward for the full problem set via public eval."""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any, Optional

from verl.utils.reward_score.ale_selftest import ale_compile_selftest
from verl.utils.reward_score.frontiercs import extract_cpp
from verl.utils.reward_score.reward_norm import get_problem_reference, normalize_alebench_score

logger = logging.getLogger(__name__)

_LITE = False
_JUDGE_VERSION = os.environ.get("ALEBENCH_JUDGE_VERSION", "202301")
_CODE_LANGUAGE = "cpp17"
_NUM_WORKERS = int(os.environ.get("ALEBENCH_NUM_WORKERS", "4"))
_SESSION_DURATION_SECONDS = float(os.environ.get("ALEBENCH_PUBLIC_SESSION_DURATION_SECONDS", 365 * 24 * 3600))
_SESSION_CACHE: dict[str, Any] = {}


class AleInfraError(RuntimeError):
    """Raised when ALE-Bench evaluation fails for INFRASTRUCTURE reasons (Docker
    unavailable, compile container produced no ``a.out`` -> FileNotFoundError,
    session start failed, OOM, etc.) rather than a legitimate model-side result.

    This is loud on purpose. A legitimate model failure (code that does not
    compile, WA, RE, TLE) is reported by ALE as a *valid* ``Result`` with
    ``overall_absolute_score == 0`` and flows through as reward 0.0. An infra
    failure throws an exception out of ``public_eval``; silently mapping that to
    reward 0.0 (the old behavior) is indistinguishable from genuine model
    weakness and silently poisons every ALE reward in the rollout."""


def _empty_result() -> dict[str, float | None]:
    return {
        "score": 0.0,  # normalized 0-100 reward (matches frontiercs.py scale)
        "performance": 0.0,
        "rank": None,
        "overall_absolute_score": 0.0,  # raw, kept for logging
        "overall_relative_score": None,
        "raw_score": 0.0,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _start_session(problem_id: str) -> Any:
    import ale_bench

    return ale_bench.start(
        problem_id=problem_id,
        lite_version=_LITE,
        num_workers=_NUM_WORKERS,
        run_visualization_server=False,
        session_duration=_SESSION_DURATION_SECONDS,
    )


def _get_session(problem_id: str) -> Any:
    session = _SESSION_CACHE.get(problem_id)
    if session is None:
        session = _start_session(problem_id)
        _SESSION_CACHE[problem_id] = session
    return session


def _close_cached_sessions() -> None:
    for problem_id, session in list(_SESSION_CACHE.items()):
        try:
            session.close()
        except Exception:
            logger.exception("ALE-Bench full cached session close failed for %s", problem_id)
    _SESSION_CACHE.clear()


atexit.register(_close_cached_sessions)


def _drop_session(problem_id: str) -> None:
    session = _SESSION_CACHE.pop(problem_id, None)
    if session is not None:
        try:
            session.close()
        except Exception:
            logger.exception("ALE-Bench full cached session close failed for %s", problem_id)


def _public_eval(problem_id: str, code: str) -> Any:
    session = _get_session(problem_id)
    return session.public_eval(
        code,
        _CODE_LANGUAGE,
        judge_version=_JUDGE_VERSION,
        skip_local_visualization=True,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Evaluate C++ solution on ALE-Bench full train set using public eval."""
    if data_source != "alebench_full":
        raise ValueError(f"data_source must be 'alebench_full', got {data_source}")

    # COMPILE STARTUP SELF-TEST (once per process, cached). If this node's compile
    # substrate is systematically broken, EVERY sample would be classified
    # COMPILATION_ERROR -> reward 0, indistinguishable from genuine model weakness.
    # Probe the real compile-container path with a known-good program before
    # scoring anything; raise AleInfraError (not a fake 0) if it is broken.
    ale_compile_selftest(AleInfraError)

    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)
    # Empty extraction is a real model-side 0 (no code produced), not infra.
    if not code:
        return _empty_result()

    # Run the eval. Any exception escaping public_eval is an INFRASTRUCTURE
    # failure (Docker down, compile container produced no a.out -> FileNotFoundError,
    # session error, OOM). It must be LOUD, not silently scored 0.0. A legitimate
    # model failure does NOT throw -- ALE returns a valid Result with
    # overall_absolute_score == 0, which we map to reward 0.0 below.
    try:
        try:
            result = _public_eval(problem_id, code)
        except Exception as exc:
            if "session is finished" not in str(exc).lower():
                raise
            # Stale cached session: drop and retry once with a fresh session.
            _drop_session(problem_id)
            result = _public_eval(problem_id, code)
    except Exception as exc:
        # Drop the (possibly corrupted) cached session so the next sample restarts
        # clean instead of reusing a broken /tmp / container state.
        _drop_session(problem_id)
        logger.exception("ALE-Bench full public eval INFRA failure for %s", problem_id)
        raise AleInfraError(
            f"ALE-Bench public_eval failed for problem {problem_id}: {exc!r}"
        ) from exc

    raw_score = float(result.overall_absolute_score)
    # NOTE: public_eval never populates per-case relative scores, so
    # result.overall_relative_score is always None here. The raw absolute
    # score is unbounded (1e4..1e13) and inverted for minimize-type problems,
    # which dominates and destabilizes mixed GRPO. Normalize per-problem to a
    # bounded, direction-aware 0-100 reward (same scale as frontiercs.py).
    norm_score = normalize_alebench_score(problem_id, raw_score)
    ref_entry = get_problem_reference(problem_id) or {}
    return {
        "score": norm_score,  # <- bounded 0-100 reward used by GRPO
        "performance": raw_score,  # raw, for logging / comparison to ALE leaderboard
        "rank": None,
        "overall_absolute_score": raw_score,  # raw, kept for logging
        "overall_relative_score": _optional_float(result.overall_relative_score),
        "raw_score": raw_score,
        "ale_reference": float(ref_entry.get("reference", 0.0)),
        "ale_score_type": str(ref_entry.get("score_type", "")),
    }
