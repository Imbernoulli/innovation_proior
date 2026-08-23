# Copyright 2025 - Frontier-CS reward for VERL RL training
"""Frontier-CS judge reward: submit C++ code to judge API, return score."""

import logging
import os
import re
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


def _default_judge_url() -> str:
    """Resolve the Frontier-CS judge base URL.

    Order: ``FRONTIERCS_JUDGE_URL`` env > ``http://localhost:$PORT`` (the judge
    launcher binds ``$PORT``, which the canonical recipe sets to 8092) > 8082.

    This makes the reward function follow the judge the launcher actually started.
    Previously the URL was hard-pinned to :8082 while the launcher binds :8092,
    so every frontiercs row raised JudgeInfraError (ECONNREFUSED). Callers can
    still override per-call via ``judge_url`` arg or ``extra_info['judge_url']``.
    """
    explicit = os.environ.get("FRONTIERCS_JUDGE_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("PORT") or "8082"
    return f"http://localhost:{port}"


DEFAULT_JUDGE_URL = _default_judge_url()
POLL_INTERVAL = 2.0
# Per-problem judge wait budget. Raised from the original hard-coded 300.0s
# because a SINGLE slow judge call (the judge serves each submission in 11-20s and
# can queue behind others) was tripping the 300s ceiling and crashing the whole RL
# run. Overridable via FRONTIERCS_JUDGE_MAX_WAIT; default bumped to 900s.
MAX_WAIT = float(os.environ.get("FRONTIERCS_JUDGE_MAX_WAIT", "900"))
# Fail-soft: when set (default ON), a JudgeInfraError on ONE problem is logged and
# scored 0.0 for that sample instead of propagating up and killing the trainer. Set
# FRONTIERCS_JUDGE_FAIL_SOFT=0 to restore the old loud-crash behavior (e.g. for
# offline judge debugging where you WANT a down judge to halt loudly).
FAIL_SOFT = os.environ.get("FRONTIERCS_JUDGE_FAIL_SOFT", "1") not in ("0", "false", "False", "")
HARDTEST_PREFIXES = ("hardtest_smp_", "hardtest_orig_")
HARDTEST_FULL_SCORE = 100.0


class JudgeInfraError(RuntimeError):
    """Raised when the Frontier-CS judge is unreachable / fails for infrastructure
    reasons (connection refused, HTTP 5xx, timeout, malformed response). This is a
    LOUD failure on purpose: a down/OOM-killed judge must not be silently mapped to
    reward 0.0, which would be indistinguishable from genuine model weakness and
    would silently poison every rollout (the original go-judge ECONNREFUSED bug)."""


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_cpp(response_text: str) -> str:
    """Extract C++ code from model response (markdown or raw), ignoring <think> blocks."""
    if not response_text:
        return ""

    response_text = strip_think(response_text)
    if not response_text:
        return ""

    code = response_text.strip()

    # Try to extract from ```cpp blocks
    cpp_pattern = r'```(?:cpp|c\+\+)?\s*\n(.*?)```'
    matches = re.findall(cpp_pattern, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Fallback: strip markdown if present
    if code.startswith("```cpp"):
        code = code[6:].strip()
    elif code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def is_hardtest_problem(ground_truth: Any) -> bool:
    """Whether this problem id is one of the packaged hardtest variants."""
    return str(ground_truth).startswith(HARDTEST_PREFIXES)


def normalize_frontiercs_score(raw_score: float, ground_truth: Any) -> float:
    """Map hardtest judge scores to binary reward while preserving default behavior otherwise."""
    if is_hardtest_problem(ground_truth):
        return 100.0 if raw_score >= HARDTEST_FULL_SCORE else 0.0
    return raw_score


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    judge_url: Optional[str] = None,
    **kwargs,
) -> float:
    """
    Submit C++ solution to Frontier-CS judge and return score.

    Hardtest package ids (hardtest_smp_* / hardtest_orig_*) are treated as
    strict binary reward: only full score counts as 100, partial score counts
    as 0.0. All other Frontier-CS problems keep the original 0-100 score.

    Args:
        data_source: Must be "frontiercs"
        solution_str: Model's raw response (may contain ```cpp ... ```)
        ground_truth: problem_id (str)
        judge_url: Judge API base URL (default: http://localhost:8082)

    Returns:
        float: Reward score. A returned 0.0 is always a *legitimate model-side 0*
        (empty/non-compiling/failed code). Infrastructure failures (judge
        unreachable, HTTP error, timeout) RAISE ``JudgeInfraError`` instead of
        being silently scored 0, so a broken judge halts training loudly rather
        than poisoning the rollout with fake zeros.

    Raises:
        JudgeInfraError: on any judge/infrastructure failure.
    """
    if data_source != "frontiercs":
        raise ValueError(f"data_source must be 'frontiercs', got {data_source}")

    # Fail-soft wrapper: a JudgeInfraError reflects a transient/per-problem judge
    # failure (a single slow/timed-out submission, a momentary connection blip). With
    # FAIL_SOFT on (default), we log it and score THIS sample 0.0 so one bad problem
    # never propagates up to kill the whole multi-hour RL run. The inner
    # _compute_score_inner keeps the original loud-raise contract intact, and
    # FAIL_SOFT=0 restores it end-to-end.
    if not FAIL_SOFT:
        return _compute_score_inner(
            data_source, solution_str, ground_truth, extra_info, judge_url, **kwargs
        )
    try:
        return _compute_score_inner(
            data_source, solution_str, ground_truth, extra_info, judge_url, **kwargs
        )
    except JudgeInfraError as exc:
        logger.warning(
            "FrontierCS judge infra error on problem %s -> scoring this sample 0.0 "
            "and continuing (FAIL_SOFT). %s",
            str(ground_truth),
            exc,
        )
        return 0.0


def _compute_score_inner(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    judge_url: Optional[str] = None,
    **kwargs,
) -> float:
    """Core judge submit/poll. Raises JudgeInfraError on infrastructure failure;
    compute_score wraps it to optionally fail soft (see FAIL_SOFT)."""
    # Resolve judge URL: explicit arg > extra_info > env/default. Resolve the default
    # at CALL time (not import time) so the reward Ray worker honors the PORT /
    # FRONTIERCS_JUDGE_URL env the launcher sets (the judge binds $PORT=8092, not the
    # legacy hard-pinned 8082). NOTE: the previous expression
    # `(judge_url or extra_info.get(...) if extra_info else None)` had a precedence
    # bug -- when extra_info was None it discarded an explicitly-passed judge_url.
    url = judge_url or (extra_info or {}).get("judge_url") or _default_judge_url()
    url = url.rstrip("/")
    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)

    # Empty extraction is a real model-side 0 (no code produced), not infra.
    if not code:
        return 0.0

    try:
        r = requests.post(
            f"{url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": problem_id, "lang": "cpp"},
            timeout=30,
        )
        r.raise_for_status()
        sid = r.json().get("sid")
    except Exception as exc:  # connection refused, 5xx, timeout, bad JSON -> infra
        raise JudgeInfraError(
            f"Frontier-CS judge submit failed for problem {problem_id} at {url}: {exc!r}"
        ) from exc
    if not sid:
        raise JudgeInfraError(
            f"Frontier-CS judge returned no submission id for problem {problem_id} at {url}"
        )

    start = time.time()
    while time.time() - start < MAX_WAIT:
        try:
            r2 = requests.get(f"{url}/result/{sid}", timeout=10)
            if r2.status_code == 404:
                time.sleep(POLL_INTERVAL)
                continue
            r2.raise_for_status()
            res = r2.json()
        except Exception as exc:
            raise JudgeInfraError(
                f"Frontier-CS judge poll failed for problem {problem_id} (sid={sid}) "
                f"at {url}: {exc!r}"
            ) from exc
        status = res.get("status")
        if status == "done":
            raw_score = float(res.get("score", 0))
            return normalize_frontiercs_score(raw_score, problem_id)
        if status == "error":
            # Judge engine reports a code-side failure (compile/runtime/WA) -> real 0.
            return 0.0
        time.sleep(POLL_INTERVAL)
    raise JudgeInfraError(
        f"Frontier-CS judge timed out after {MAX_WAIT}s for problem {problem_id} "
        f"(sid={sid}) at {url}"
    )
