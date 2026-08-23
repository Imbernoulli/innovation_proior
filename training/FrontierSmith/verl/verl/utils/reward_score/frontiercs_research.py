# Copyright 2025 - frontiercs_research reward for VERL RL training
"""frontiercs_research reward: score a candidate research-track solution by running
the OFFICIAL Frontier-CS research evaluator.py (the exact same harness used for the
research EVAL), and return a 0-100 score.

MIRRORS the frontiersmith_synth reward in *interface* and fail-soft contract:
  compute_score(data_source, solution_str, ground_truth, extra_info, ...) -> float in [0,100].

Scoring: reuse scripts/frontiercs_research_eval.py::evaluate_research_solution, which
copies the problem dir into a tempdir, writes the extracted Python code to solution.py,
runs `python evaluator.py --solution-path ... --output-path result.json`, and returns
{score, score_unbounded, status, ...} with score on 0..100. A genuine 0.0 is a real
model-side 0 (empty/infeasible/failing solution). Infrastructure failures (missing
problem dir, evaluator crash, TIMEOUT) raise ResearchInfraError; we wrap fail-soft
(default ON) so one bad problem never kills the RL run.

RL cost control: the research evaluators have a heavy tail (median ~2.6s but some run
minutes). For RL we cap each evaluation at FRONTIERCS_RESEARCH_TIMEOUT (default 180s
here) so a slow rollout is scored 0 (fail-soft) rather than stalling the whole batch.
The subprocess python is FRONTIERCS_RESEARCH_PYTHON (an env with the simulator deps,
e.g. envs/sft_lf); the reward worker itself only needs a light import of the harness.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---- locate the research eval harness (FrontierSmith/scripts/) ----
#   this file: <fs>/FrontierSmith/verl/verl/utils/reward_score/frontiercs_research.py
#   harness:   <fs>/FrontierSmith/scripts/frontiercs_research_eval.py
_THIS = Path(__file__).resolve()
_FS_ROOT = _THIS.parents[4]          # <fs>/FrontierSmith
_SCRIPTS = _FS_ROOT / "scripts"

# Tighter default timeout for RL than the eval defaults (bound the slow tail).
os.environ.setdefault("FRONTIERCS_RESEARCH_CPU_TIMEOUT", "180")
os.environ.setdefault("FRONTIERCS_RESEARCH_TIMEOUT", "180")
# Fail-soft on infra errors (default ON) -- same knob shape as frontiersmith_synth.
FAIL_SOFT = os.environ.get("FRONTIERCS_RESEARCH_FAIL_SOFT", "1") not in ("0", "false", "False", "")

_HARNESS = None
_CPU = None


def _harness():
    """Lazily import the base research eval harness (for extract_python_code + the
    GPU-path evaluator). Cached. Adds scripts/ to sys.path."""
    global _HARNESS
    if _HARNESS is None:
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        import frontiercs_research_eval as R  # noqa: E402
        _HARNESS = R
    return _HARNESS


def _cpu():
    """Lazily import the CPU-family eval module (per-family official CLI dispatch,
    reads JSON score; the module the research CPU EVAL actually used). Cached."""
    global _CPU
    if _CPU is None:
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        import frontiercs_research_cpu_eval as C  # noqa: E402
        _CPU = C
    return _CPU


# ---- host-RAM guard (2026-07-30) ------------------------------------------------
# Each evaluation spawns a simulator subprocess that can eat many GB. verl runs 8
# RewardLoopWorkers, each dispatching MANY compute_score calls concurrently, so an
# unbounded pool of evaluators OOM-killed the training node twice (research2 on
# 07-27 after 4 steps; research_sp8 on 07-30 in step 1: Ray "worker killed due to
# node running low on memory" while optimizer offload holds ~288G). Cap concurrent
# evaluator subprocesses PER WORKER PROCESS with a threading semaphore.
# 8 workers x default 2 = <=16 simulators in flight. Override: FRONTIERCS_RESEARCH_MAX_CONC.
import threading as _threading
_EVAL_SEM = _threading.Semaphore(int(os.environ.get("FRONTIERCS_RESEARCH_MAX_CONC", "1")))

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Score a research-track rollout. ground_truth = problem_id (e.g.
    'cant_be_late/high_availability_loose_deadline_large_overhead'). Returns 0..100."""
    try:
        R = _harness()
    except Exception as e:  # harness import failure = infra
        if FAIL_SOFT:
            logger.warning("frontiercs_research harness import failed (fail-soft 0.0): %s", e)
            return 0.0
        raise

    # Extract the Python code from the (think + prose + code) response.
    try:
        code = R.extract_python_code(solution_str or "")
    except Exception:
        code = ""
    if not code or not code.strip():
        return 0.0

    pid = str(ground_truth)
    # CPU-family problems (all 64 in research.parquet) use the per-family official
    # CLI dispatch in the cpu_eval module; GPU-family falls back to the base harness.
    try:
        C = _cpu()
        use_cpu = C.is_cpu_family(pid)
    except Exception:
        C = None
        use_cpu = False
    try:
        with _EVAL_SEM:
            if use_cpu:
                t = int(os.environ.get("FRONTIERCS_RESEARCH_CPU_TIMEOUT", "180"))
                res = C.evaluate_cpu_research_solution(pid, code, timeout=t)
            else:
                t = int(os.environ.get("FRONTIERCS_RESEARCH_TIMEOUT", "180"))
                res = R.evaluate_research_solution(pid, code, timeout=t)
    except Exception as e:  # ResearchInfraError or any harness fault = infra, fail-soft
        if FAIL_SOFT:
            logger.warning("frontiercs_research infra error (fail-soft 0.0) for %s: %s", pid, e)
            return 0.0
        raise

    score = (res or {}).get("score")
    if score is None:
        return 0.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0
    # Clamp to [0, 100]; a legitimate 0 is a real model-side 0.
    return max(0.0, min(100.0, score))
