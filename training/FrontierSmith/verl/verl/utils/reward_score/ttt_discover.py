# Copyright 2025 - TTT-Discover (math discovery) reward for VERL RL training
"""TTT-Discover reward: extract a constructive solution from the model response,
score it with the official offline numpy verifier, and normalize to 0-100.

Currently implemented tasks (in-task normalized, 0-100, thinking allowed):

| data_source | task             | dir      | best(=100) | worst(=0) | notes                       |
|-------------|------------------|----------|------------|-----------|-----------------------------|
| ttt_erdos   | erdos_min_overlap| minimize | 0.3808     | 0.5       | C5 overlap bound, h:[0,2]->[0,1] |
| ttt_ac1     | ac1 autocorr     | minimize | 1.5030     | 2.0       | C1 upper bound              |
| ttt_ac2     | ac2 autocorr     | maximize | 0.97       | 0.80      | C2 lower bound              |

Reference values (from TTT-Discover/scripts/verify_released_math_results.py, run
offline on the released sequences):
  erdos raw=0.380875 (n=600), ac1 raw=1.502863 (n=30000), ac2 raw=0.959180 (n=50000).
The "worst(=0)" anchors are easily-reachable trivial baselines (erdos: constant
h=0.5 -> C5=0.5; ac1: a flat box -> ~2.0; ac2: ~0.80) so a no-effort answer gets
~0 and SOTA gets 100, with clamping.

Candidate format accepted from the model (inside a ```python or ```json fence):
  1. A direct sequence: a JSON / Python list of floats, e.g. ```json\n[0.5, 0.5, ...]\n```
     or a ```python fence whose code assigns ``sequence = [...]`` / ``h = [...]``.
  2. A program defining ``run(...)`` (erdos) returning ``(h_values, c5_bound, n_points)``
     or returning / assigning a 1D sequence. Executed in an isolated subprocess
     with a timeout; only the returned/produced sequence is scored (we recompute
     the objective ourselves -- anti-cheat).

Robustness: any failure -> score 0.0; never raises out of compute_score.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Optional

import numpy as np

from verl.utils.reward_score.frontiercs import strip_think

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LEN = 200_000  # guard against pathological outputs


# ----------------------------- official scorers ----------------------------- #
def evaluate_erdos(sequence) -> float:
    """C5 overlap bound (minimize). Mirrors TTT verify_released_math_results.evaluate_erdos."""
    h = np.asarray(sequence, dtype=np.float64)
    if h.ndim != 1 or h.size == 0:
        return float("inf")
    if not np.all(np.isfinite(h)):
        return float("inf")
    if np.any(h < 0) or np.any(h > 1):
        return float("inf")
    n = h.shape[0]
    target_sum = n / 2.0
    cur = float(np.sum(h))
    if cur != target_sum:
        if cur <= 0:
            return float("inf")
        h = h * (target_sum / cur)
        if np.any(h < 0) or np.any(h > 1):
            return float("inf")
    dx = 2.0 / n
    return float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))


def evaluate_ac1(sequence) -> float:
    """C1 autocorrelation upper bound (minimize). Mirrors TTT evaluate_ac1."""
    seq = list(sequence)
    if not seq:
        return float("inf")
    xs = [min(1000.0, max(0.0, float(x))) for x in seq]
    total = float(np.sum(xs))
    if total < 0.01:
        return float("inf")
    conv = np.convolve(xs, xs)
    return float(2 * len(xs) * np.max(conv) / (total ** 2))


def evaluate_ac2(sequence) -> float:
    """C2 autocorrelation lower bound (maximize). Mirrors TTT evaluate_ac2."""
    seq = list(sequence)
    if not seq:
        return float("-inf")
    xs = [min(1000.0, max(0.0, float(x))) for x in seq]
    if float(np.sum(xs)) < 0.01:
        return float("-inf")
    conv = np.convolve(xs, xs)
    num_points = len(conv)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], conv, [0]))
    l2 = 0.0
    for i in range(len(conv) + 1):
        y1 = y_points[i]
        y2 = y_points[i + 1]
        hh = x_intervals[i]
        l2 += (hh / 3) * (y1 ** 2 + y1 * y2 + y2 ** 2)
    norm_1 = np.sum(np.abs(conv)) / (len(conv) + 1)
    norm_inf = np.max(np.abs(conv))
    if norm_1 <= 0 or norm_inf <= 0:
        return float("-inf")
    return float(l2 / (norm_1 * norm_inf))


_TASKS: dict[str, dict[str, Any]] = {
    "ttt_erdos": {"scorer": evaluate_erdos, "direction": "minimize", "best": 0.3808, "worst": 0.5,
                  "entrypoint": "run"},
    "ttt_ac1": {"scorer": evaluate_ac1, "direction": "minimize", "best": 1.5030, "worst": 2.0,
                "entrypoint": "run"},
    "ttt_ac2": {"scorer": evaluate_ac2, "direction": "maximize", "best": 0.97, "worst": 0.80,
                "entrypoint": "run"},
}


# ----------------------------- extraction ----------------------------- #
def _extract_fenced(text: str, langs) -> list[str]:
    out = []
    for lang in langs:
        out.extend(re.findall(rf"```(?:{lang})?\s*\n(.*?)```", text, re.DOTALL))
    return out


def _try_parse_sequence(blob: str):
    """Parse a JSON/Python literal list of numbers from a string. Returns list or None."""
    blob = blob.strip()
    # direct JSON
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(blob)
        except Exception:
            obj = None
        if obj is not None:
            seq = _coerce_seq(obj)
            if seq is not None:
                return seq
    # find the largest [...] span and parse that
    m = re.search(r"\[(?:[^\[\]]|\n)*\]", blob, re.DOTALL)
    if m:
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(m.group(0))
            except Exception:
                obj = None
            seq = _coerce_seq(obj) if obj is not None else None
            if seq is not None:
                return seq
    return None


def _coerce_seq(obj):
    if isinstance(obj, dict):
        for k in ("sequence", "h_values", "h", "data", "heights"):
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, (list, tuple)):
        # erdos run() returns (h_values, c5, n); take the first list-like element.
        if obj and isinstance(obj[0], (list, tuple, np.ndarray)):
            obj = obj[0]
        try:
            seq = [float(x) for x in obj]
        except (TypeError, ValueError):
            return None
        if 0 < len(seq) <= MAX_SEQUENCE_LEN:
            return seq
    return None


# Subprocess runner: executes a candidate program that defines run() (or assigns a
# sequence), then prints the resulting sequence as JSON. We score it ourselves.
_RUNNER_SRC = r'''
import json, sys, ast
import numpy as np
entrypoint = sys.argv[1]
prog_path = sys.argv[2]
src = open(prog_path).read()
g = {"np": np, "numpy": np}
seq = None
try:
    exec(compile(src, "<candidate>", "exec"), g)
    if entrypoint in g and callable(g[entrypoint]):
        res = g[entrypoint]()
        if isinstance(res, (list, tuple)) and res and isinstance(res[0], (list, tuple, np.ndarray)):
            seq = list(np.asarray(res[0]).ravel())
        else:
            seq = list(np.asarray(res).ravel())
    else:
        for name in ("sequence", "h_values", "h", "result", "data", "heights"):
            if name in g:
                seq = list(np.asarray(g[name]).ravel())
                break
    if seq is not None:
        seq = [float(x) for x in seq]
        out = {"ok": True, "sequence": seq}
    else:
        out = {"ok": False, "error": "no sequence produced"}
except Exception as exc:
    import traceback
    out = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()[-800:]}
print("TTT_RESULT_JSON:" + json.dumps(out))
'''


def _run_program(code: str, entrypoint: str, timeout_s: float):
    """Execute candidate program in an isolated subprocess; return produced sequence or None."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix="_ttt.py", delete=False) as fh:
            fh.write(code)
            tmp = fh.name
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER_SRC, entrypoint, tmp],
            capture_output=True, text=True, timeout=timeout_s,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("TTT_RESULT_JSON:"):
                obj = json.loads(line[len("TTT_RESULT_JSON:"):])
                if obj.get("ok"):
                    return obj.get("sequence")
                return None
        return None
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _normalize(raw: float, direction: str, best: float, worst: float) -> float:
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if raw != raw or raw in (float("inf"), float("-inf")):
        return 0.0
    if direction == "minimize":
        denom = worst - best
        if denom <= 0:
            return 0.0
        score = 100.0 * (worst - raw) / denom
    else:
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
    """Score a TTT-Discover constructive answer. Returns dict with 0-100 "score"."""
    task = _TASKS.get(data_source) or _TASKS.get(str(ground_truth))
    if task is None:
        return {"score": 0.0, "raw_objective": None, "error": f"unknown task {data_source}"}
    if extra_info:
        task = {**task, **{k: extra_info[k] for k in ("best", "worst", "direction") if k in extra_info}}

    timeout_s = float((extra_info or {}).get("timeout_s", 120.0))
    scorer = task["scorer"]

    text = strip_think(solution_str or "")
    if not text.strip():
        return {"score": 0.0, "raw_objective": None, "error": "empty"}

    candidate_seq = None

    # 1) Try direct sequence from json / python / plain fences (then whole text).
    blocks = _extract_fenced(text, ["json", "python", "py", ""]) or [text]
    for blk in blocks:
        seq = _try_parse_sequence(blk)
        if seq is not None:
            candidate_seq = seq
            break

    # 2) If a python program defines run()/assigns a sequence, execute it.
    if candidate_seq is None:
        py_blocks = _extract_fenced(text, ["python", "py"]) or _extract_fenced(text, [""])
        for code in py_blocks:
            if "def %s" % task["entrypoint"] in code or "def run" in code or "=" in code:
                seq = _run_program(code, task["entrypoint"], timeout_s)
                if seq is not None:
                    candidate_seq = seq
                    break

    if candidate_seq is None:
        return {"score": 0.0, "raw_objective": None, "error": "no_sequence"}

    try:
        raw = scorer(candidate_seq)
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "raw_objective": None, "error": f"scorer:{exc}"}

    score = _normalize(raw, task["direction"], task["best"], task["worst"])
    return {
        "score": float(score),
        "raw_objective": (None if raw in (float("inf"), float("-inf")) else float(raw)),
        "direction": task["direction"],
        "best": task["best"],
        "worst": task["worst"],
        "n": len(candidate_seq),
    }
