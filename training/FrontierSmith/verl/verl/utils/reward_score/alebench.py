# Copyright 2025 - ALE-Bench reward for VERL RL training (validation only)
"""ALE-Bench Docker judge reward: run C++ code via ALE-Bench private eval.

FIDELITY TO THE OFFICIAL ALE-Bench PROTOCOL
-------------------------------------------
This reward scores ONE model response (one C++ solution) per call, because that
is the contract GRPO/verl requires (reward is a pure function of a single
rollout). It is therefore intentionally a SINGLE-SUBMISSION scorer, NOT the full
official agentic loop (repeated-sampling -> median-selection -> self-refinement
with public-eval feedback -> power-of-two private-eval checkpoints; see
ALE-Bench/src/ale_bench_eval/scaffolds.py + __main__.py). For a faithful
end-to-end agentic evaluation of a served model, use
FrontierSmith/scripts/run_alebench_faithful.py.

What IS faithful here (matches the official harness exactly):
  * We call the official ``session.private_eval`` (ALE-Bench/src/ale_bench/
    session.py:565), so the raw-score -> case-accept -> overall_absolute_score ->
    standings-rank -> rank->performance mapping is the OFFICIAL one
    (ale_bench/data.py Standings.get_new_rank + RankPerformanceMap.get_performance).
    ``performance`` is the AtCoder-style rating estimate (higher is better), the
    same headline number the official ``analyze_results`` aggregates.
  * CE/WA/RE/TLE flow through as a valid Result with overall_absolute_score == 0
    (never an infra crash), matching official semantics.

Known DELIBERATE divergences (documented, not bugs):
  * Single submission vs. the official iterative loop (see above).
  * ``lite_version`` defaults True (ALEBENCH_LITE) -> 5 public + 10% private
    seeds, matching the official "lite" seed regime; set ALEBENCH_LITE=false for
    the full seed set (required for a leaderboard-valid rank/performance, cf.
    ALE-Bench/README.md).
  * ``code_language`` now defaults to ``cpp20`` (env ALEBENCH_CODE_LANGUAGE) to
    match the official driver default (ale_bench_eval/__main__.py:261) instead of
    the previous hardcoded ``cpp17``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from verl.utils.reward_score.ale_selftest import ale_compile_selftest
from verl.utils.reward_score.frontiercs import extract_cpp

logger = logging.getLogger(__name__)


class AleInfraError(RuntimeError):
    """Raised when ALE-Bench evaluation fails for INFRASTRUCTURE reasons (Docker
    unavailable, session start failed, compile container produced no a.out, OOM)
    rather than a legitimate model-side result. Loud on purpose: a model failure
    (CE/WA/RE/TLE) is a valid ALE Result with overall_absolute_score == 0 and
    flows through as reward 0.0; an infra failure must not be silently scored 0.0
    and mistaken for model weakness."""


_LITE = os.environ.get("ALEBENCH_LITE", "true").lower() in ("1", "true")
_JUDGE_VERSION = os.environ.get("ALEBENCH_JUDGE_VERSION", "202301")
# Official ale_bench_eval default code_language is cpp20 (ale_bench_eval/__main__.py:261).
# Overridable for the judge_version whose default differs (201907->cpp17, 202510->cpp23).
_CODE_LANGUAGE = os.environ.get("ALEBENCH_CODE_LANGUAGE", "cpp17")  # cpp17 image exists; cpp20 image not built (set ALEBENCH_CODE_LANGUAGE=cpp20 after building it for full fidelity)
_NUM_WORKERS = int(os.environ.get("ALEBENCH_NUM_WORKERS", "4"))


def _empty_result() -> dict[str, float | None]:
    return {
        "score": 0.0,
        "performance": 0.0,
        "rank": None,
        "overall_absolute_score": 0.0,
        "overall_relative_score": None,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _start_session(problem_id: str):
    import ale_bench

    # private_eval is limited to one call per session, so each scored sample
    # must get a fresh session instead of reusing a cached public-eval session.
    return ale_bench.start(
        problem_id=problem_id,
        lite_version=_LITE,
        num_workers=_NUM_WORKERS,
        run_visualization_server=False,
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Evaluate C++ solution on ALE-Bench and return native ALE metrics.

    Returns dict with:
        score: ALE performance, used as the scalar reward/validation core metric
        performance: same as score, for explicit logging
        rank: estimated standings rank, lower is better
        overall_absolute_score: private absolute score
        overall_relative_score: private relative score when available
    """
    if data_source != "alebench":
        raise ValueError(f"data_source must be 'alebench', got {data_source}")

    # COMPILE STARTUP SELF-TEST (once per process, cached). A systematically
    # broken compile substrate would classify EVERY submission as a compile
    # error -> reward 0 (a fake uniform-zero). Probe the real compile-container
    # path with a known-good program before scoring; raise AleInfraError if broken.
    ale_compile_selftest(AleInfraError)

    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)
    # Empty extraction is a real model-side 0 (no code produced), not infra.
    if not code:
        return _empty_result()

    session = None
    try:
        # Any exception here (session start, private_eval, Docker / a.out failure)
        # is an INFRASTRUCTURE failure and must be LOUD, not silently scored 0.0.
        # A model failure (CE/WA/RE/TLE) does NOT throw: ALE returns a valid Result
        # with overall_absolute_score == 0 and a real rank-mapped performance.
        try:
            session = _start_session(problem_id)
            result, rank, performance = session.private_eval(code, _CODE_LANGUAGE, judge_version=_JUDGE_VERSION)
        except Exception as exc:
            logger.exception("ALE-Bench eval INFRA failure for %s", problem_id)
            raise AleInfraError(
                f"ALE-Bench private_eval failed for problem {problem_id}: {exc!r}"
            ) from exc
        performance = float(performance)
        return {
            "score": performance,
            "performance": performance,
            "rank": float(rank),
            "overall_absolute_score": float(result.overall_absolute_score),
            "overall_relative_score": _optional_float(result.overall_relative_score),
        }
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("ALE-Bench session close failed for %s", problem_id)
