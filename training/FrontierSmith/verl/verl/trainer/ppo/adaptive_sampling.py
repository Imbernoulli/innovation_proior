# Copyright 2026 - Per-prompt adaptive resampling ("group deepening") for GRPO.
"""Deepen the groups that produce no gradient, instead of replacing their prompts.

THE MEASURED PROBLEM
--------------------
GRPO standardizes the advantage *within* the n samples drawn for one prompt::

    a_i = (r_i - mean_g) / (std_g + eps)

If all n samples of a group score identically -- in practice, all zero -- then
``std_g == 0`` and every ``a_i`` is exactly 0. The group emits **no gradient**;
its n rollouts are pure waste. Across our 4-arm campaign this hit 885 / 5,120
groups (17.3%), i.e. 14,160 wasted rollouts.

Those dead groups are dominated by *truncation*, not by genuine impossibility:
at base step 10 dead groups ran 85.5% truncated vs 32.4% for live groups, and
P(dead | zero of 16 samples completed) = 0.93. Their median length sits pinned
at exactly the 32,768 cap with slope +-0.0 tokens/step over 20 steps. The
samples that most need a "be shorter" signal are precisely the ones emitting no
gradient at all.

WHY DEEPENING, NOT DAPO's FILTER-AND-REFILL
-------------------------------------------
Upstream DAPO's dynamic sampling ("filter_groups") oversamples the *prompt*
stream: it generates ``gen_batch_size > train_batch_size`` prompts, throws away
the flat groups, and backfills with **fresh prompts** until the train batch is
full. That answers a different question ("give me a batch of prompts that
already have spread") and it structurally *avoids* the hard prompts -- the very
ones we care about.

What we want instead: when a prompt's group has no spread, **spend more samples
on that same prompt**. A problem the policy solves 1 time in 128 is not a
useless problem; it is a problem whose signal we simply failed to observe at
n=16. Deepening 16 -> 32 -> 64 -> 128 recovers that signal from the prompt we
already chose, and leaves the prompt distribution untouched.

(For the record: ``algorithm.filter_groups`` exists as a dataclass in
``verl/trainer/config/algorithm.py:583,611`` but has **no consumer** anywhere in
``RayPPOTrainer``; setting ``FILTER_GROUPS_ENABLE=True`` is a silent no-op
today. This module does not resurrect it -- it implements the different,
deepening behaviour that was asked for.)

THE GROUP-SIZE / DENOMINATOR DECISION  (read this before changing defaults)
--------------------------------------------------------------------------
An enlarged group cannot simply be dropped into the update. Three things break:

1. ``ppo_mini_batch_size``. ``DPActor.update_policy`` does
   ``data.split(self.config.ppo_mini_batch_size)`` (``verl/workers/actor/
   dp_actor.py:551``) where the config value has already been multiplied by
   ``rollout.n``. A batch of ``B*n + extra`` rows produces a ragged final
   mini-batch and shifts every mini-batch boundary.
2. The token-mean denominator. With ``loss_agg_mode=token-mean`` the loss is
   ``sum_tokens(adv * ratio) / total_tokens_in_minibatch``, and
   ``loss_scale_factor = response_mask.shape[0] / ppo_mini_batch_size``
   (``dp_actor.py:593``). A 128-sample group carries ~8x the tokens of a
   16-sample group and would silently dominate the update -- we would have
   "fixed" gradient starvation by introducing gradient capture.
3. Sequence balancing / packing. ``_balance_batch`` and dynamic-bsz packing
   assume a fixed row count per step; variable group sizes make the per-rank
   token load, and therefore the number of micro-batches, vary step to step.

DECISION (default ``keep="subsample"``): **enlarge for measurement, fold back to
exactly n for the update.** Concretely, for a group deepened to K samples:

  * the GRPO baseline ``(mean, std)`` is computed over **all K** samples -- a
    strictly lower-variance baseline estimate than the n-sample one;
  * exactly **n** of the K rows are carried into the update, so the batch is
    ``B*n`` rows, byte-for-byte the shape the trainer sees today. Points 1-3
    above are untouched;
  * the n retained rows are drawn by *stratified* sampling that guarantees the
    informative minority samples survive (a 1-in-128 success has only a 12%
    chance of surviving a uniform 16-of-128 draw -- that would throw away the
    thing we paid for);
  * stratification biases the sample, so each retained row carries a
    Horvitz-Thompson weight that removes the bias exactly::

        w_i = n / (K * p_i)

    where ``p_i`` is row i's inclusion probability. The weight multiplies the
    advantage (the policy-gradient objective is linear in the advantage, so
    this is exact and denominator-agnostic). Under uniform sampling
    ``p_i = n/K`` and ``w_i == 1``, i.e. the unstratified case is the
    identity -- a useful invariant to test.

  ``sum_{i in S} w_i == n`` holds identically by construction, so a deepened
  group carries exactly the same total mass into the update as any other group.
  Deepening therefore buys **signal where there was none**, at zero change to
  the update's shape, denominator, or balance.

Worked example (the dominant case: a group that was 0/16 dead turns out to be
1/128). Scores ``[s, 0 x 127]``: ``mean = s/128``, ``std = s/sqrt(128)``, so the
successful sample gets ``a = 11.22`` and weight ``w = 16/(128*1) = 0.125`` ->
effective ``1.40``; the 15 retained zeros get ``a = -0.088``, ``w = 1.058`` ->
``-1.40`` total. The group's contributions sum to ~0, as GRPO requires, and the
group now emits a real gradient where before it emitted exactly zero.

Note this is *smaller* than the ``3.75`` the same success would receive in a
1-of-16 group -- and that is correct, not a bug. At 1/128 the policy is far
worse at this prompt than a 1/16 observation would imply; GRPO's std
normalization inflates rare successes when the group is small. Deepening buys an
honest baseline, not a louder one. The win is that the group is **alive at all**.

.. warning::
   The ``1.40`` effective advantage DEPENDS ON GRPO's division by the group
   std. With Dr.GRPO-style mean-only centering (``norm_adv_by_std_in_grpo=
   False``) the same 1-in-128 positive gets ``(1 - 1/128) * 0.125 ~= 0.12`` --
   more than an order of magnitude smaller, and easily drowned out by ordinary
   groups whose advantages keep their natural scale. If anyone removes std
   normalization later, deepened positives silently collapse; the HT weights
   would then need to be rescaled by something like the group std to
   compensate. Do not flip that switch without revisiting this file.

``keep="all"`` is implemented for experiments but is **not** the default,
precisely because of points 1-3. It logs a loud warning.

INTERACTION WITH THE OVERLONG PENALTY (verl/utils/reward_score/overlong_penalty.py)
----------------------------------------------------------------------------------
With the penalty enabled, a group of truncated-but-differently-long samples
acquires real spread from the length term alone, so it is no longer dead and
deepening skips it -- and the gradient it emits says "be shorter", which is
exactly the missing signal. A group where every sample hits the cap *exactly*
still gets an identical penalty and stays dead, so deepening still applies. The
two features are complementary, and the epsilon below is compared against the
*post-penalty* score spread.

V2: THE no_positive TRIGGER AND THE DROP RULE
---------------------------------------------
Production exposed a v1 flaw: with the overlong penalty on, an all-failed group
acquires variance from the penalty ALONE, so the zero-variance trigger stops
firing -- and those penalty-only groups are actively harmful: in a group of 16
all-failed samples a complete-but-wrong sample (raw score 0) sits above the
penalty-dragged group mean (~-0.2) and receives POSITIVE advantage. We were
uptraining wrong-but-short answers with no correct exemplar in sight.

The v2 pipeline (ADAPTIVE_N_TRIGGER=no_positive, the default) is sequenced so
the penalty NEVER participates in the deepening decision and never trains a
group that lacks a correct exemplar:

  1. Generate the initial n; judge the trigger on RAW scores (pre-penalty):
     does any sample have raw > pos_eps?
  2. NO -> deepen the same prompt in rounds (n -> 2n -> ... -> max_n, budget
     capped). Re-judge on raw scores each round; stop the moment a positive
     appears.
  3. Positive found -> fold back to n with every positive GUARANTEED retained
     (protected stratum), and only the kept group's PENALIZED scores feed the
     GRPO baseline (mean/std over all K penalized scores) and advantages. The
     penalty therefore shapes length only where a correct answer anchors the
     ordering: correct > wrong-short > truncated.
  4. No positive at budget -> DROP the group entirely: its response_mask is
     zeroed, so it contributes NO advantage and NO tokens to the token-mean
     loss denominator (the audit showed dead tokens diluting effective lr by
     ~25%; a dropped group must not dilute).

Note on where the penalty is applied: the penalty is a deterministic per-row
function of response length, so "apply it only to kept groups at fold-back"
and "apply it in the rollout worker, then judge every decision on the raw
score and fully mask non-kept groups" produce byte-identical training signal.
The implementation keeps the worker-side application (rm_scores arrive
penalized) and reconstructs the raw score from reward_pre_overlong (or by
recomputing the deterministic penalty) for every trigger decision.

ADAPTIVE_N_TRIGGER=zero_variance selects the v1 behaviour exactly (deepen on
flat post-penalty scores, never drop).

CONFIG (env vars, matching the FS_*/launcher convention used by pertask_norm)
----------------------------------------------------------------------------
  ADAPTIVE_N_ENABLE=0        master switch. 0 (default) -> this module is a
                             total no-op and the trainer takes its original path.
  ADAPTIVE_N_TRIGGER=no_positive
                             no_positive (v2): deepen while max(raw) <= pos_eps,
                             drop at budget exhaustion. zero_variance (v1):
                             deepen flat groups, never drop.
  ADAPTIVE_N_POS_EPS=0.0     raw score strictly above this counts as positive.
  ADAPTIVE_N_MAX=128         ceiling on the deepened group size K.
  ADAPTIVE_N_GROWTH=2.0      per-round growth factor (16 -> 32 -> 64 -> 128).
  ADAPTIVE_N_EPS=1e-6        zero_variance trigger: flat when max-min <= eps.
  ADAPTIVE_N_KEEP=subsample  subsample (default, see above) | all.
  ADAPTIVE_N_STRATIFY=1      stratified retention + HT weights. 0 -> uniform
                             (positives are still protected under no_positive).
  ADAPTIVE_N_MAX_PROMPTS=0   cap deepened prompts per step (0 = no cap).
  ADAPTIVE_N_MAX_EXTRA=0     cap extra rollouts per step (0 = no cap). Matters a
                             lot for mlsbench_agent rows: one extra sample is a
                             whole multi-minute episode, not one generate call.
  ADAPTIVE_N_AGENTS=         comma-separated agent_name allow-list (empty = all).
                             e.g. "single_turn_agent" to deepen only the cheap rows.
  ADAPTIVE_N_SEED=0          RNG seed for the retention draw.

V2.1 knobs (literature pass: DAPO / VAPO / Dr.GRPO / Knapsack-RL / SPEED / PKPO / NSR):
  ADAPTIVE_CLAMP_ZERO_RAW=1  safety net closing the residual hole penalty-gating
                             leaves: in a deepened group with one positive and
                             many penalty-bearing truncations the group mean can
                             sit BELOW zero, so a wrong-but-complete row (raw 0)
                             clears the mean and would be uptrained again. The
                             clamp forces adv <= 0 for every row whose RAW
                             reward is not positive, in every group, deepened or
                             natural. Active whenever adaptive resampling or the
                             overlong penalty is on (with both off it is
                             mathematically inert for our non-negative scorers).
  ADAPTIVE_REQUEUE_ENABLE=1  bounded requeue instead of hard drop: the loader is
                             single-epoch-no-repeat, so a hard-dropped prompt is
                             permanently lost -- the worst possible spend of up
                             to 128 generations. First-time no-positive-at-budget
                             prompts enter a FIFO and are re-injected into a
                             later gen batch ONCE; only a second failure drops
                             them for good.
  ADAPTIVE_REQUEUE_AFTER_STEPS=1
                             minimum steps a prompt waits in the FIFO before it
                             is eligible for re-injection. (The literature memo
                             suggested >=200 for long runs; our production runs
                             are 20-40 steps total, so the default is "next
                             eligible slot".)
  ADAPTIVE_REQUEUE_MAX_FRAC=0.25
                             cap on re-injected prompts per step, as a fraction
                             of the train batch. The actual count is rounded
                             DOWN to a multiple of ppo_mini_batch_size so the
                             mini-batch split never sees a ragged batch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveSamplingConfig",
    "GroupPlan",
    "flat_group_mask",
    "has_positive",
    "group_needs_deepening",
    "group_indices",
    "next_round_target",
    "plan_retention",
    "adaptive_advantage_values",
    "warn_unknown_env",
]

# Every env var the adaptive-n / overlong stack actually reads. A typo'd knob
# (ADAPTIVE_N_TRIGER=..., FS_OVERLONG_PENALTY_FACTO=...) silently reverts to
# the default, which for safety-critical flags is the worst failure mode --
# so from_env scans for the prefixes and warns on anything unrecognized.
KNOWN_ENV_KEYS = {
    "ADAPTIVE_N_ENABLE",
    "ADAPTIVE_N_TRIGGER",
    "ADAPTIVE_N_POS_EPS",
    "ADAPTIVE_N_MAX",
    "ADAPTIVE_N_GROWTH",
    "ADAPTIVE_N_EPS",
    "ADAPTIVE_N_KEEP",
    "ADAPTIVE_N_STRATIFY",
    "ADAPTIVE_N_MAX_PROMPTS",
    "ADAPTIVE_N_MAX_EXTRA",
    "ADAPTIVE_N_MAX_EXTRA_PER_WORKER",  # v2.5 in-wave: the workers enforce this share
    "ADAPTIVE_N_OVERLAP",
    "ADAPTIVE_N_INWAVE",
    "ADAPTIVE_N_AGENTS",
    "ADAPTIVE_N_SEED",
    "ADAPTIVE_CLAMP_ZERO_RAW",
    "ADAPTIVE_REQUEUE_ENABLE",
    "ADAPTIVE_REQUEUE_AFTER_STEPS",
    "ADAPTIVE_REQUEUE_MAX_FRAC",
    "ADAPTIVE_REQUEUE_BUFFER_CAP",
    "FS_OVERLONG_PENALTY",
    "FS_OVERLONG_BUFFER_LEN",
    "FS_OVERLONG_PENALTY_FACTOR",
    "FS_OVERLONG_MAX_RESP_LEN",
    "FS_OVERLONG_LOG",
}
_ENV_PREFIXES = ("ADAPTIVE_N_", "ADAPTIVE_CLAMP_", "ADAPTIVE_REQUEUE_", "FS_OVERLONG_")
_warned_unknown_env = False


def warn_unknown_env() -> None:
    """Warn (once) about env vars that look like ours but are not recognized."""
    global _warned_unknown_env
    if _warned_unknown_env:
        return
    _warned_unknown_env = True
    unknown = sorted(
        k for k in os.environ if k.startswith(_ENV_PREFIXES) and k not in KNOWN_ENV_KEYS
    )
    if unknown:
        logger.warning(
            "[adaptive-n] UNRECOGNIZED env var(s) %s -- these are silently ignored. "
            "Known knobs: %s",
            unknown,
            sorted(KNOWN_ENV_KEYS),
        )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[adaptive-n] bad float for %s=%r, using %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        logger.warning("[adaptive-n] bad int for %s=%r, using %s", name, raw, default)
        return default


@dataclass
class AdaptiveSamplingConfig:
    """Knobs for per-prompt adaptive resampling. Defaults reproduce vanilla GRPO."""

    enable: bool = False
    trigger: str = "no_positive"  # "no_positive" (v2) | "zero_variance" (v1)
    pos_eps: float = 0.0  # raw score strictly above this counts as positive
    max_n: int = 128
    growth: float = 2.0
    eps: float = 1e-6
    keep: str = "subsample"  # "subsample" | "all"
    stratify: bool = True
    max_prompts: int = 0  # 0 = unlimited
    max_extra: int = 0  # 0 = unlimited
    agents: tuple[str, ...] = field(default_factory=tuple)  # empty = all
    seed: int = 0
    clamp_zero_raw: bool = True  # v2.1(A): adv <= 0 for non-positive-raw rows
    requeue_enable: bool = True  # v2.1(B): FIFO retry before a group is lost
    requeue_after_steps: int = 1
    requeue_max_frac: float = 0.25
    requeue_buffer_cap: int = 256  # oldest evicted beyond this (deepcopied rows)
    # v2.4 (2026-08-12, user directive "深挖和主生成不应该分开串行"): pipelined
    # deepening. The synchronous ladder issues one extra generate_sequences per
    # round AFTER the main wave has fully drained -- measured 124 min/step of
    # low-occupancy GPU time. In overlap mode no extra wave is issued at all:
    # a group without a positive is masked out of this step and its prompt is
    # re-emitted in the NEXT step's main wave at the deeper size, so every
    # deepening token is generated inside a full-width wave.
    overlap: bool = False
    # v2.5 (2026-08-12, user directive "发现全错就开始roll"): in-wave deepening.
    # The extra samples are launched by the AgentLoopWorker the moment a group's
    # own samples come back with no positive, so they generate ALONGSIDE the
    # groups still in flight instead of in a narrow post-wave tail. Everything
    # else (ladder, ceiling, budget, fold-back to n with HT weights) is the
    # existing trainer-side machinery -- only the launch timing moves.
    inwave: bool = False

    @classmethod
    def from_env(cls, env: dict | None = None) -> "AdaptiveSamplingConfig":
        prev = None
        if env is not None:
            prev = dict(os.environ)
            os.environ.update({k: str(v) for k, v in env.items()})
        try:
            agents_raw = os.environ.get("ADAPTIVE_N_AGENTS", "") or ""
            cfg = cls(
                enable=_env_flag("ADAPTIVE_N_ENABLE", False),
                trigger=(os.environ.get("ADAPTIVE_N_TRIGGER", "no_positive") or "no_positive").strip().lower(),
                pos_eps=_env_float("ADAPTIVE_N_POS_EPS", 0.0),
                max_n=_env_int("ADAPTIVE_N_MAX", 128),
                growth=_env_float("ADAPTIVE_N_GROWTH", 2.0),
                eps=_env_float("ADAPTIVE_N_EPS", 1e-6),
                keep=(os.environ.get("ADAPTIVE_N_KEEP", "subsample") or "subsample").strip().lower(),
                stratify=_env_flag("ADAPTIVE_N_STRATIFY", True),
                max_prompts=_env_int("ADAPTIVE_N_MAX_PROMPTS", 0),
                max_extra=_env_int("ADAPTIVE_N_MAX_EXTRA", 0),
                agents=tuple(a.strip() for a in agents_raw.split(",") if a.strip()),
                seed=_env_int("ADAPTIVE_N_SEED", 0),
                clamp_zero_raw=_env_flag("ADAPTIVE_CLAMP_ZERO_RAW", True),
                requeue_enable=_env_flag("ADAPTIVE_REQUEUE_ENABLE", True),
                requeue_after_steps=_env_int("ADAPTIVE_REQUEUE_AFTER_STEPS", 1),
                requeue_max_frac=_env_float("ADAPTIVE_REQUEUE_MAX_FRAC", 0.25),
                requeue_buffer_cap=_env_int("ADAPTIVE_REQUEUE_BUFFER_CAP", 256),
                overlap=_env_flag("ADAPTIVE_N_OVERLAP", False),
                inwave=_env_flag("ADAPTIVE_N_INWAVE", False),
            )
            warn_unknown_env()
        finally:
            if prev is not None:
                os.environ.clear()
                os.environ.update(prev)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.requeue_enable and self.requeue_buffer_cap < 1:
            raise ValueError(
                f"ADAPTIVE_REQUEUE_BUFFER_CAP must be >= 1 when requeue is enabled, got {self.requeue_buffer_cap}"
            )
        if self.keep not in ("subsample", "all"):
            raise ValueError(f"ADAPTIVE_N_KEEP must be 'subsample' or 'all', got {self.keep!r}")
        if self.trigger not in ("no_positive", "zero_variance"):
            raise ValueError(
                f"ADAPTIVE_N_TRIGGER must be 'no_positive' or 'zero_variance', got {self.trigger!r}"
            )
        if self.growth <= 1.0:
            raise ValueError(f"ADAPTIVE_N_GROWTH must be > 1.0, got {self.growth}")
        if self.max_n < 1:
            raise ValueError(f"ADAPTIVE_N_MAX must be >= 1, got {self.max_n}")
        if self.eps < 0:
            raise ValueError(f"ADAPTIVE_N_EPS must be >= 0, got {self.eps}")
        if self.inwave and self.overlap:
            raise ValueError("ADAPTIVE_N_INWAVE and ADAPTIVE_N_OVERLAP are mutually exclusive")
        if self.inwave:
            if self.trigger != "no_positive":
                raise ValueError("ADAPTIVE_N_INWAVE=1 is only defined for trigger=no_positive")
            if self.keep != "subsample":
                raise ValueError("ADAPTIVE_N_INWAVE=1 requires ADAPTIVE_N_KEEP=subsample")
        if self.overlap:
            # Pipelined deepening re-emits the prompt in a later batch, which is
            # exactly what the requeue FIFO does; without it a starving group
            # would be masked out and never come back.
            if not self.requeue_enable:
                raise ValueError("ADAPTIVE_N_OVERLAP=1 requires ADAPTIVE_REQUEUE_ENABLE=1")
            if self.trigger != "no_positive":
                raise ValueError("ADAPTIVE_N_OVERLAP=1 is only defined for trigger=no_positive")
            if self.keep != "subsample":
                raise ValueError("ADAPTIVE_N_OVERLAP=1 requires ADAPTIVE_N_KEEP=subsample")

    def describe(self) -> str:
        return (
            f"enable={self.enable} trigger={self.trigger} pos_eps={self.pos_eps} max_n={self.max_n} "
            f"growth={self.growth} eps={self.eps} keep={self.keep} stratify={self.stratify} "
            f"max_prompts={self.max_prompts} max_extra={self.max_extra} "
            f"agents={list(self.agents) or 'ALL'} seed={self.seed} "
            f"clamp_zero_raw={self.clamp_zero_raw} requeue={self.requeue_enable} "
            f"requeue_after={self.requeue_after_steps} requeue_frac={self.requeue_max_frac} "
            f"overlap={self.overlap} inwave={self.inwave}"
        )


def group_indices(uids: np.ndarray) -> dict:
    """Map uid -> np.ndarray of row positions, preserving first-seen uid order."""
    out: dict = {}
    for pos, uid in enumerate(uids):
        out.setdefault(uid, []).append(pos)
    return {uid: np.asarray(idx, dtype=np.int64) for uid, idx in out.items()}


def flat_group_mask(scores: np.ndarray, eps: float) -> bool:
    """True when the group carries no usable score spread.

    Uses peak-to-peak rather than std so the threshold is expressed in reward
    units (an eps of 1e-6 means "the best and worst sample differ by less than a
    millionth of a reward point"), and so a group of size 1 is trivially flat.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        return True
    return bool((scores.max() - scores.min()) <= eps)


def has_positive(raw_scores: np.ndarray, pos_eps: float = 0.0) -> bool:
    """True when the group contains at least one correct exemplar (raw > pos_eps).

    MUST be judged on RAW (pre-penalty) scores: a truncated sample's penalized
    score is negative and a complete-but-wrong sample's is 0, but neither is a
    correct answer, and the penalty must never influence this decision.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if raw_scores.size == 0:
        return False
    return bool(raw_scores.max() > pos_eps)


def group_needs_deepening(
    raw_scores: np.ndarray,
    penalized_scores: np.ndarray,
    cfg: AdaptiveSamplingConfig,
) -> bool:
    """Does this group need more samples?

    no_positive (v2): deepen while there is no correct exemplar, judged on RAW
    scores only. A group whose variance comes entirely from the overlong penalty
    still deepens -- that variance would otherwise uptrain wrong-but-short
    answers against a baseline no correct answer anchors.

    zero_variance (v1): deepen while the (penalized) scores are flat.
    """
    if cfg.trigger == "no_positive":
        return not has_positive(raw_scores, cfg.pos_eps)
    return flat_group_mask(penalized_scores, cfg.eps)


def next_round_target(current: int, cfg: AdaptiveSamplingConfig) -> int:
    """Group size to grow to next, or ``current`` when the ceiling is reached."""
    if current >= cfg.max_n:
        return current
    target = int(np.ceil(current * cfg.growth))
    target = max(target, current + 1)
    return min(target, cfg.max_n)


@dataclass
class GroupPlan:
    """Which rows of a K-sample group survive into the update, and their weights."""

    keep_idx: np.ndarray  # local positions within the group, length n
    weights: np.ndarray  # Horvitz-Thompson weights, same length
    mean: float  # baseline mean over ALL K samples
    std: float  # baseline std  over ALL K samples (unbiased, matches torch.std)
    group_size: int  # K
    enlarged: bool


def _unbiased_std(x: np.ndarray) -> float:
    """Sample std with ddof=1 to match ``torch.std`` (verl's GRPO baseline)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 1.0
    return float(np.std(x, ddof=1))


def plan_retention(
    scores: np.ndarray,
    n: int,
    cfg: AdaptiveSamplingConfig,
    rng: np.random.Generator | None = None,
    protect: np.ndarray | None = None,
) -> GroupPlan:
    """Choose which n of K samples to train on, plus unbiasing weights.

    The baseline (mean/std) is always computed over all K samples: with
    stratified retention the retained subset is deliberately non-representative,
    so using the retained mean would bias the baseline itself, not just the
    weights.

    ``protect`` (bool array of length K): rows that MUST survive the fold. Under
    the no_positive trigger this is the positive-raw-score rows -- the whole
    point of deepening was to find them, so losing one to the draw would throw
    away exactly what we paid for. Protected rows are their own stratum with
    inclusion probability 1 (as long as they fit in n), and the HT weights keep
    the estimate unbiased: w_i = n / (K * p_i) with p_prot = 1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    K = scores.size
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)

    mean = float(scores.mean()) if K else 0.0
    std = _unbiased_std(scores)

    if K <= n:
        return GroupPlan(
            keep_idx=np.arange(K, dtype=np.int64),
            weights=np.ones(K, dtype=np.float64),
            mean=mean,
            std=std,
            group_size=K,
            enlarged=False,
        )

    # Protected stratum takes precedence over both the uniform and the modal
    # paths: a guaranteed-retained positive is a correctness requirement, not a
    # variance optimization, so it applies even with stratify=0.
    if protect is not None and np.asarray(protect).any():
        protect = np.asarray(protect, dtype=bool)
        prot_idx = np.flatnonzero(protect)
        rest_idx = np.flatnonzero(~protect)

        n_prot = min(prot_idx.size, n)
        n_rest = n - n_prot
        keep_prot = prot_idx if prot_idx.size <= n else np.sort(rng.choice(prot_idx, size=n, replace=False))
        keep_rest = (
            np.sort(rng.choice(rest_idx, size=n_rest, replace=False)) if n_rest > 0 else np.empty(0, dtype=np.int64)
        )

        p = np.zeros(K, dtype=np.float64)
        p[prot_idx] = n_prot / prot_idx.size  # 1.0 whenever every positive fits
        if rest_idx.size:
            p[rest_idx] = n_rest / rest_idx.size

        keep = np.sort(np.concatenate([keep_prot, keep_rest]).astype(np.int64))
        kept_p = np.where(p[keep] <= 0, 1.0, p[keep])
        weights = n / (K * kept_p)
        # sum(w) == n identically when both strata are sampled; the only exception
        # is the degenerate n_prot == n case (more positives than the fold can
        # hold), where the unsampled rest-stratum breaks the identity -- normalize
        # so a deepened group can never carry more mass than any other group.
        w_sum = weights.sum()
        if w_sum > 0 and not np.isclose(w_sum, n):
            weights = weights * (n / w_sum)

        return GroupPlan(
            keep_idx=keep,
            weights=weights,
            mean=mean,
            std=std,
            group_size=K,
            enlarged=True,
        )

    if not cfg.stratify:
        keep = np.sort(rng.choice(K, size=n, replace=False))
        # uniform inclusion p = n/K  =>  w = n / (K * n/K) = 1
        return GroupPlan(
            keep_idx=keep,
            weights=np.ones(n, dtype=np.float64),
            mean=mean,
            std=std,
            group_size=K,
            enlarged=True,
        )

    # Stratify around the modal score: everything that is NOT the modal value is
    # "informative" and must be protected from the draw. For the dominant
    # all-zero-but-one case this is exactly {the one success} vs {the zeros}.
    rounded = np.round(scores, 12)
    values, counts = np.unique(rounded, return_counts=True)
    modal = values[int(np.argmax(counts))]
    majority = np.flatnonzero(rounded == modal)
    minority = np.flatnonzero(rounded != modal)

    if minority.size == 0:
        # still flat at K samples: nothing to protect, fall back to uniform
        keep = np.sort(rng.choice(K, size=n, replace=False))
        return GroupPlan(
            keep_idx=keep,
            weights=np.ones(n, dtype=np.float64),
            mean=mean,
            std=std,
            group_size=K,
            enlarged=True,
        )

    # Keep as much of the minority as fits in half the budget, then fill from the
    # majority. Halving keeps the retained group from becoming all-minority (which
    # would push every HT weight far from 1 and inflate variance in the other
    # direction).
    n_min = min(minority.size, max(1, n // 2))
    n_maj = n - n_min
    if n_maj > majority.size:
        n_maj = majority.size
        n_min = min(minority.size, n - n_maj)

    keep_min = rng.choice(minority, size=n_min, replace=False) if n_min else np.empty(0, dtype=np.int64)
    keep_maj = rng.choice(majority, size=n_maj, replace=False) if n_maj else np.empty(0, dtype=np.int64)

    p = np.zeros(K, dtype=np.float64)
    if minority.size:
        p[minority] = n_min / minority.size
    if majority.size:
        p[majority] = n_maj / majority.size

    keep = np.sort(np.concatenate([keep_min, keep_maj]).astype(np.int64))
    kept_p = p[keep]
    # guard: a stratum sampled at rate 0 must never be retained
    kept_p = np.where(kept_p <= 0, 1.0, kept_p)
    weights = n / (K * kept_p)

    return GroupPlan(
        keep_idx=keep,
        weights=weights,
        mean=mean,
        std=std,
        group_size=K,
        enlarged=True,
    )


def adaptive_advantage_values(
    scores: np.ndarray,
    mean: "float | np.ndarray",
    std: "float | np.ndarray",
    weights: np.ndarray,
    norm_adv_by_std: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """GRPO advantage against the K-sample baseline, times the HT weight.

    Mirrors ``core_algos.compute_grpo_outcome_advantage`` exactly for the
    unweighted / non-enlarged case, so a group that was never deepened comes out
    bit-identical to today's value.
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if norm_adv_by_std:
        adv = (scores - mean) / (std + eps)
    else:
        adv = scores - mean
    return adv * weights
