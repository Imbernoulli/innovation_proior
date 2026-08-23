# Copyright 2025 - ThetaEvolve (AlphaEvolve-style program search) reward for VERL RL training
"""ThetaEvolve reward: extract a Python program from the model response, run the
official OpenEvolve evaluator offline (in an isolated subprocess), read the raw
objective value, and normalize to a 0-100 in-task reward.

Design (mirrors the fcs+ale 0-100 convention; thinking allowed in the response):

* The model is asked to *reason* then emit a Python program inside a ```python
  fence. The program must, when run, call ``save_search_results(...)`` writing the
  candidate solution into the temp workspace (exactly as the OpenEvolve initial
  programs do).
* We hand that program file to the task's official ``evaluator_modular.evaluate``
  (``ObjectiveBasedProblemEvaluator`` + ``verify.py``). This runs the program with
  a hard timeout in a fresh temp dir, validates the solution, and recomputes the
  raw objective (anti-cheat). We read ``result.metrics['objective_value']``.
* We normalize the raw objective to 0-100 using the per-task reference range and
  optimization direction (taken from the task's own ``score_transform`` config so
  the口径 matches the OpenEvolve pipeline):
    - MINIMIZE: score = 100 * (worst - raw) / (worst - best), clamped to [0, 100]
      where best = score_range_min (SOTA/target), worst = score_range_max (baseline)
    - MAXIMIZE: score = 100 * (raw - worst) / (best - worst), clamped to [0, 100]
      where best = score_range_max (SOTA/target), worst = score_range_min

Per-task reference values (verified by running the official initial programs):

| data_source        | task                       | dir      | best(=100) | worst(=0) | initial-prog raw |
|--------------------|----------------------------|----------|------------|-----------|------------------|
| thetaevolve_ac3    | third_autocorr_inequality  | minimize | 1.4557     | 3.2       | ~3.159 (~2.3/100)|
| thetaevolve_circle | circle_packing_modular n=26| maximize | 2.635      | 0.0       | ~0.96 (~36/100)  |

Robustness: any failure (no code, uncompilable, runtime error, timeout, invalid
solution, evaluator crash) yields score 0.0 and never raises out of compute_score.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

from verl.utils.reward_score.frontiercs import strip_think

logger = logging.getLogger(__name__)

# Root of the ThetaEvolve OpenEvolve adaptation (importable repo).
THETAEVOLVE_ROOT = os.environ.get(
    "THETAEVOLVE_ROOT",
    "/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/openevolve_adapted",
)

# Per-task registry. Keyed by VERL data_source.
#   eval_dir   : dir containing evaluator_modular.py + verify.py
#   config     : OpenEvolve YAML providing problem_type + score_transform
#   direction  : "minimize" | "maximize"
#   best       : raw objective that maps to score 100 (SOTA/target)
#   worst      : raw objective that maps to score 0 (baseline / no-effort)
#   timeout_s  : hard wall-clock for one program evaluation (subprocess)
_TASKS: dict[str, dict[str, Any]] = {
    "thetaevolve_ac3": {
        "eval_dir": os.path.join(
            THETAEVOLVE_ROOT,
            "examples/third_autocorr_inequality/evaluators",
        ),
        "config": os.path.join(
            THETAEVOLVE_ROOT,
            "examples/third_autocorr_inequality/configs/"
            "config_third_autocorr_inequality_qwen35_local_smoke.yaml",
        ),
        "direction": "minimize",
        "best": 1.4557,
        "worst": 3.2,
        "timeout_s": 90.0,
    },
    "thetaevolve_circle": {
        "eval_dir": os.path.join(
            THETAEVOLVE_ROOT,
            "examples/circle_packing_modular/evaluators",
        ),
        "config": os.path.join(
            THETAEVOLVE_ROOT,
            "examples/circle_packing_modular/configs/"
            "config_circle_packing_modular_qwen35_local_smoke.yaml",
        ),
        "direction": "maximize",
        "best": 2.635,
        "worst": 0.0,
        "timeout_s": 90.0,
    },
}


def extract_python(response_text: str) -> str:
    """Extract a Python program from a model response, ignoring <think> blocks.

    Prefers fenced ```python blocks; falls back to any fenced block; finally to
    the raw post-think text if it looks like code.
    """
    import re

    if not response_text:
        return ""
    text = strip_think(response_text)
    if not text:
        return ""
    text = text.strip()

    # ```python ... ``` (also accepts ```py and bare ```)
    for pat in (r"```(?:python|py)\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        matches = re.findall(pat, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    # Fallback: strip a leading/trailing fence if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# Runner executed in an isolated subprocess. It imports the task evaluator with
# the correct OPENEVOLVE_CONFIG_PATH so the score_transform/direction bind to the
# right task, runs evaluate(program_path), and prints the metrics as JSON.
_RUNNER_SRC = r'''
import json, os, sys
eval_dir = sys.argv[1]
program_path = sys.argv[2]
sys.path.insert(0, eval_dir)
try:
    import evaluator_modular as em
    res = em.evaluate(program_path)
    metrics = dict(res.metrics) if res is not None and res.metrics else {}
    out = {
        "ok": True,
        "objective_value": metrics.get("objective_value"),
        "combined_score": metrics.get("combined_score"),
        "validity": metrics.get("validity"),
        "rl_normalized_reward": metrics.get("rl_normalized_reward"),
    }
except Exception as exc:  # never let the runner crash silently
    import traceback
    out = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
print("THETAEVOLVE_RESULT_JSON:" + json.dumps(out))
'''


def _normalize(raw: float, direction: str, best: float, worst: float) -> float:
    """Linear map raw objective -> [0, 100], clamped, accounting for direction."""
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if raw != raw or raw in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    if direction == "minimize":
        denom = worst - best
        if denom <= 0:
            return 0.0
        score = 100.0 * (worst - raw) / denom
    else:  # maximize
        denom = best - worst
        if denom <= 0:
            return 0.0
        score = 100.0 * (raw - worst) / denom
    return max(0.0, min(100.0, score))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Evaluate a ThetaEvolve candidate program and return a 0-100 reward dict.

    Args:
        data_source: one of the registered ThetaEvolve data_sources.
        solution_str: raw model response (may contain <think>...</think> + ```python).
        ground_truth: task key (matches data_source); kept for parity / overrides.
        extra_info: optional overrides (best, worst, direction, config, timeout_s).

    Returns:
        dict with key "score" (float in [0, 100]) plus diagnostics. Never raises.
    """
    task = _TASKS.get(data_source)
    if task is None:
        # Allow ground_truth to select the task if data_source is generic.
        task = _TASKS.get(str(ground_truth))
    if task is None:
        return {"score": 0.0, "raw_objective": None, "error": f"unknown task {data_source}"}

    # Per-row overrides (rarely needed; keeps reward configurable from data).
    if extra_info:
        task = {**task, **{k: extra_info[k] for k in (
            "best", "worst", "direction", "config", "eval_dir", "timeout_s") if k in extra_info}}

    code = extract_python(solution_str)
    if not code or "save_search_results" not in code and "def construct" not in code and "def run" not in code:
        # No usable program. (We still try if there is *some* code, but empty -> 0.)
        if not code:
            return {"score": 0.0, "raw_objective": None, "error": "no_code"}

    env = dict(os.environ)
    env["OPENEVOLVE_CONFIG_PATH"] = task["config"]
    # Ensure the openevolve package and the evaluator dir are importable.
    extra_py = THETAEVOLVE_ROOT
    env["PYTHONPATH"] = extra_py + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    tmp_prog = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_candidate.py", delete=False
        ) as fh:
            fh.write(code)
            tmp_prog = fh.name

        timeout_s = float(task.get("timeout_s", 90.0))
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER_SRC, task["eval_dir"], tmp_prog],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s + 30.0,  # subprocess budget > inner program timeout
        )
        result_obj = None
        for line in proc.stdout.splitlines():
            if line.startswith("THETAEVOLVE_RESULT_JSON:"):
                result_obj = json.loads(line[len("THETAEVOLVE_RESULT_JSON:"):])
                break
        if result_obj is None:
            return {
                "score": 0.0,
                "raw_objective": None,
                "error": "no_result",
                "stderr_tail": (proc.stderr or "")[-500:],
            }
        if not result_obj.get("ok"):
            return {"score": 0.0, "raw_objective": None, "error": result_obj.get("error")}

        raw = result_obj.get("objective_value")
        # validity 0 or missing objective -> the program failed/invalid -> 0.
        if raw is None or not result_obj.get("validity"):
            return {"score": 0.0, "raw_objective": raw, "error": "invalid_or_failed"}

        score = _normalize(raw, task["direction"], task["best"], task["worst"])
        return {
            "score": float(score),
            "raw_objective": float(raw),
            "direction": task["direction"],
            "best": task["best"],
            "worst": task["worst"],
        }
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "raw_objective": None, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001 - reward must never crash the worker
        logger.exception("ThetaEvolve reward failed for %s", data_source)
        return {"score": 0.0, "raw_objective": None, "error": str(exc)}
    finally:
        if tmp_prog and os.path.exists(tmp_prog):
            try:
                os.remove(tmp_prog)
            except OSError:
                pass
