# Copyright 2025 - Within-task (per-task) reward normalization for mixed multi-task GRPO.
"""Per-task reward normalization so every task contributes on a comparable scale.

WHY
---
The mixed RL run trains on five task families whose per-task scorers all *nominally*
emit a 0-100 reward, but whose *realized* reward distributions differ a lot:

  data_source        scorer                                  realized reward
  ----------------   -------------------------------------   --------------------
  frontiercs         judge pass-rate / hardtest binary        {0,100} or 0-100
  alebench_full      saturating ratio vs per-problem ref      0-100 (usually small)
  thetaevolve_circle linear (raw-worst)/(best-worst)*100      0-100 (~36 baseline)
  thetaevolve_ac3    linear (worst-raw)/(worst-best)*100      0-100
  ttt_erdos          linear (worst-raw)/(worst-best)*100      0-100 (near 0)

GRPO already does per-PROMPT-GROUP (per-uid) standardization a_i=(r_i-mu_g)/sig_g,
which is affine-invariant *within a group*. But across a mixed batch the different
task SCALES still bias which task's gradient dominates the global policy update
(group std differs, and a task whose rewards cluster near a constant produces near-
zero advantages while a task with a wide spread dominates). The user wants each task
internally normalized ("每个任务内部都应该去做归一化") so no single task's raw scale
dominates.

WHAT THIS DOES
--------------
A deterministic, STATELESS, per-item affine map from a task's raw scorer score onto
a common target range (default [0, 1]):

    normed = clamp( (raw - lo) / (hi - lo) ) * target

where (lo, hi) are the task's known reward bounds (the SAME bounds the per-task
scorer uses to *evaluate* the task, see the table above -> consistent with eval).

This is intentionally NOT a running z-score: the reward is computed one item at a
time inside ``RewardLoopWorker`` (``run_single``), across independent Ray workers,
so there is no reliable shared batch state, and a running statistic would not be
reproducible at eval time. A fixed, bounded affine map keyed on ``data_source`` is
reproducible, monotone (preserves the per-group ranking GRPO needs), and matches
exactly how each task is scored.

CONFIG
------
Controlled by environment variables (so it is on/off-able from the launcher without
touching verl config plumbing):

  FS_PERTASK_REWARD_NORM=1            enable (default: 0 / off -> identity passthrough)
  FS_PERTASK_REWARD_NORM_TARGET=1.0  target scale; 1.0 -> [0,1], 100.0 -> [0,100]

Per-row overrides: if ``extra_info`` carries ``reward_norm_lo`` / ``reward_norm_hi``
those win over the registry (lets the data parquet pin custom bounds per problem).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Per-task nominal reward bounds. These are the SAME bounds the per-task scorer
# maps onto, so normalizing by them is consistent with how the task is evaluated:
#   * frontiercs / alebench_full: 0-100 convention (frontiercs.py / alebench_full.py)
#   * thetaevolve_* / ttt_*: their scorers clamp to [0,100] (thetaevolve.py/ttt_discover.py)
# Every registered task already emits in [lo, hi]; the affine map just rescales.
_TASK_BOUNDS: dict[str, tuple[float, float]] = {
    "frontiercs": (0.0, 100.0),
    "alebench": (0.0, 100.0),
    "alebench_full": (0.0, 100.0),
    "thetaevolve_circle": (0.0, 100.0),
    "thetaevolve_ac3": (0.0, 100.0),
    "ttt_erdos": (0.0, 100.0),
    "ttt_ac1": (0.0, 100.0),
    "ttt_ac2": (0.0, 100.0),
}

# Default bounds for an unregistered task (assume the project-wide 0-100 convention).
_DEFAULT_BOUNDS = (0.0, 100.0)


def is_enabled() -> bool:
    """Whether per-task reward normalization is turned on (env FS_PERTASK_REWARD_NORM)."""
    return os.environ.get("FS_PERTASK_REWARD_NORM", "0").lower() in ("1", "true", "yes", "on")


def target_scale() -> float:
    """Target output scale: rewards land in [0, target]. Default 1.0 -> [0,1]."""
    try:
        return float(os.environ.get("FS_PERTASK_REWARD_NORM_TARGET", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def get_bounds(data_source: str, extra_info: Optional[dict] = None) -> tuple[float, float]:
    """Resolve (lo, hi) reward bounds for a task; per-row extra_info overrides win."""
    lo, hi = _TASK_BOUNDS.get(str(data_source), _DEFAULT_BOUNDS)
    if extra_info:
        if extra_info.get("reward_norm_lo") is not None:
            lo = float(extra_info["reward_norm_lo"])
        if extra_info.get("reward_norm_hi") is not None:
            hi = float(extra_info["reward_norm_hi"])
    return lo, hi


def normalize(
    data_source: str,
    raw_score: float,
    extra_info: Optional[dict] = None,
    *,
    target: Optional[float] = None,
) -> float:
    """Map a task's raw scorer score onto the common target range, clamped.

    Args:
        data_source: VERL ``data_source`` key identifying the task.
        raw_score: the score returned by that task's per-task scorer.
        extra_info: optional per-row overrides (reward_norm_lo / reward_norm_hi).
        target: output scale (defaults to ``target_scale()``).

    Returns:
        float in [0, target]. Monotone in ``raw_score``; never raises.
    """
    try:
        raw = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    if raw != raw or raw in (float("inf"), float("-inf")):  # NaN / inf -> 0
        return 0.0

    lo, hi = get_bounds(data_source, extra_info)
    denom = hi - lo
    if denom <= 0:
        # Degenerate bounds: fall back to identity so we never blow up.
        logger.warning("pertask_norm: bad bounds (%s, %s) for %s; passthrough.", lo, hi, data_source)
        return raw
    unit = (raw - lo) / denom
    unit = max(0.0, min(1.0, unit))  # clamp to [0,1]
    tgt = target_scale() if target is None else float(target)
    return unit * tgt


def maybe_normalize(
    data_source: str,
    raw_score: float,
    extra_info: Optional[dict] = None,
) -> dict[str, Any]:
    """Apply per-task normalization iff enabled; return reward + diagnostics.

    Returns a dict:
      {
        "reward": <normalized-or-raw score used for GRPO>,
        "reward_raw": <raw scorer score>,
        "pertask_norm_applied": bool,
        "pertask_norm_lo": lo, "pertask_norm_hi": hi, "pertask_norm_target": tgt,
      }
    """
    raw = float(raw_score) if isinstance(raw_score, (int, float)) and raw_score == raw_score else 0.0
    if not is_enabled():
        return {
            "reward": raw_score,
            "reward_raw": raw,
            "pertask_norm_applied": False,
        }
    lo, hi = get_bounds(data_source, extra_info)
    tgt = target_scale()
    normed = normalize(data_source, raw_score, extra_info, target=tgt)
    return {
        "reward": normed,
        "reward_raw": raw,
        "pertask_norm_applied": True,
        "pertask_norm_lo": lo,
        "pertask_norm_hi": hi,
        "pertask_norm_target": tgt,
    }
