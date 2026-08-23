# Copyright 2025 - Shared reward-normalization helpers for multi-task GRPO RL.
"""Shared, bounded reward normalization so heterogeneous task rewards live on a
common ~0-100 scale (matching frontiercs.py) for stable mixed GRPO training.

Why this exists
---------------
Mixed GRPO trains on FrontierCS (reward 0-100, bounded) together with ALE-Bench
full (raw ``overall_absolute_score`` = 1e4 .. 1e13, UNBOUNDED, and *inverted*
for minimize-type problems). The raw ALE magnitude dominates the batch reward
statistics and the per-group GRPO std, collapsing the FrontierCS signal over
RL steps. We map every ALE raw score to a bounded, monotone, direction-aware
0-100 value with a smooth saturating ratio against a per-problem reference.

Mapping
-------
For a per-problem positive reference ``R`` (roughly the human-best raw score on
the same case scale) and the raw public-eval ``overall_absolute_score`` ``s``:

    maximize:  norm = 100 * s / (s + R)        (increasing, 0 at s=0, 50 at s=R)
    minimize:  norm = 100 * R / (s + R)         (decreasing, 100 at s->0+, 50 at s=R)

Properties:
  * bounded in (0, 100) for s>0, exactly 0 for failures / non-accepted runs,
  * strictly monotone -> preserves the per-problem ranking signal GRPO uses,
  * never hard-clips, so it keeps gradient everywhere (unlike min(s/R,1)*100),
  * direction-aware: minimize problems (lower raw = better) are inverted so a
    better solution always gets a higher reward,
  * reference is the "half-credit" anchor; its exact value is non-critical for
    GRPO (per-group std-normalization is affine-invariant) but the scale keeps
    ALE rewards comparable to FrontierCS.

The per-problem reference table is precomputed once (raw human-absolute scale,
from ALE-Bench standings / relative_results) and stored as JSON; see
``scripts/prepare_alebench_score_references.py``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bounded reward range (matches frontiercs.py: 0-100).
REWARD_MIN = 0.0
REWARD_MAX = 100.0

# Default location of the precomputed per-problem ALE reference table.
_DEFAULT_REF_PATH = os.environ.get(
    "ALEBENCH_SCORE_REFERENCES",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "data",
        "alebench",
        "ale_score_references.json",
    ),
)

_REF_CACHE: Optional[dict[str, dict[str, Any]]] = None


def _load_references(path: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Load (and cache) the per-problem ALE reference table.

    Returns an empty dict if the table is missing so the caller can fall back
    gracefully instead of crashing the rollout.
    """
    global _REF_CACHE
    if _REF_CACHE is not None and path is None:
        return _REF_CACHE
    ref_path = os.path.abspath(path or _DEFAULT_REF_PATH)
    try:
        with open(ref_path) as fh:
            table = json.load(fh)
    except FileNotFoundError:
        logger.warning("ALE reference table not found at %s; using fallback scale.", ref_path)
        table = {}
    except Exception:
        logger.exception("Failed to load ALE reference table at %s; using fallback.", ref_path)
        table = {}
    if path is None:
        _REF_CACHE = table
    return table


def get_problem_reference(problem_id: str, path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return the reference entry ({'score_type', 'reference', ...}) for a problem id."""
    return _load_references(path).get(str(problem_id))


def saturating_normalize(raw_score: float, reference: float, score_type: str) -> float:
    """Map a raw ALE absolute score to a bounded 0-100 reward.

    Args:
        raw_score: ``result.overall_absolute_score`` (>=0; 0 means failure/non-AC).
        reference: positive per-problem reference (half-credit anchor).
        score_type: "maximize" (higher raw is better) or "minimize" (lower is better).

    Returns:
        float in [0, 100].
    """
    raw = float(raw_score)
    ref = float(reference)
    # Failure / non-accepted run -> 0 raw -> 0 reward (both directions).
    if raw <= 0.0 or ref <= 0.0:
        return REWARD_MIN
    if str(score_type).lower().startswith("min"):
        # lower raw is better
        norm = REWARD_MAX * ref / (raw + ref)
    else:
        # maximize: higher raw is better
        norm = REWARD_MAX * raw / (raw + ref)
    # Numerical safety clamp.
    return float(min(REWARD_MAX, max(REWARD_MIN, norm)))


def normalize_alebench_score(
    problem_id: str,
    raw_score: float,
    *,
    references_path: Optional[str] = None,
) -> float:
    """Normalize an ALE-Bench raw absolute score to a bounded 0-100 reward.

    Falls back to the raw score's own magnitude as the reference (half-credit at
    the observed value, treated as maximize) when the per-problem reference is
    unavailable; this still yields a bounded value rather than an unbounded one.
    """
    entry = get_problem_reference(problem_id, references_path)
    if entry is None:
        # Fallback: bounded but uninformed. Use the raw value as its own scale so
        # the output stays in (0, 100) instead of exploding. Logged once-ish.
        logger.warning("No ALE reference for problem_id=%s; using self-scaled fallback.", problem_id)
        raw = float(raw_score)
        if raw <= 0.0:
            return REWARD_MIN
        # raw/(raw+raw) == 50 for any positive raw -> bounded constant-ish fallback.
        return saturating_normalize(raw, raw, "maximize")
    return saturating_normalize(raw_score, entry.get("reference", 0.0), entry.get("score_type", "maximize"))
