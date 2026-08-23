# Copyright 2026 - DAPO soft overlong punishment, applied on the agent-loop path.
"""Length penalty for the streaming agent-loop reward path.

WHY THIS FILE EXISTS INSTEAD OF JUST USING ``DAPORewardManager``
---------------------------------------------------------------
The repo already ships two DAPO reward managers with the soft overlong
punishment. **Neither is usable on our path**, for different reasons:

1. ``verl/workers/reward_manager/dapo.py`` (the colocate manager, registered in
   ``verl/workers/reward_manager/__init__.py``). Its ``__call__`` opens with::

       verl/workers/reward_manager/dapo.py:62-64
           reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
           if reward_from_rm_scores is not None:
               return reward_from_rm_scores

   ``_extract_reward_from_rm_scores`` (``verl/workers/reward_manager/
   abstract.py:64-66``) returns early whenever ``"rm_scores" in data.batch``.
   On the async agent-loop path ``rm_scores`` is *always* present -- it is
   written by ``AgentLoopWorker._postprocess`` at ``verl/experimental/
   agent_loop/agent_loop.py:774-780`` -- so the overlong block at
   ``dapo.py:121`` is **unreachable**. (It is also latently broken: line 121
   reads ``self.overlong_buffer_cfg.enable`` unguarded, which raises
   ``AttributeError`` when the cfg is ``None``, unlike its sibling at
   ``verl/experimental/reward_loop/reward_manager/dapo.py:108`` which checks for
   ``None`` first.)

2. ``verl/experimental/reward_loop/reward_manager/dapo.py`` (the *streaming*
   manager) does implement the penalty correctly at lines 108-117, and it does
   sit on our path -- but it only covers *half* our rows, and adopting it costs
   us a feature:

   * MLS rows never reach it. ``AgentLoopWorker._compute_score`` dispatches to a
     ``RewardLoopWorker`` only when the agent did not already produce a score::

         verl/experimental/agent_loop/agent_loop.py:715
             if output.reward_score is None and enable_async_reward:

     ``mlsbench_agent`` computes its reward in-loop and sets ``reward_score``,
     so those rows bypass every reward manager. A penalty installed there would
     silently apply to ``single_turn_agent`` rows only, making the two halves of
     our mixed batch incomparable -- which is exactly what
     ``FS_PERTASK_REWARD_NORM`` exists to prevent.
   * Switching ``reward_manager=dapo`` would drop our per-task normalization:
     that lives in the *naive* streaming manager
     (``verl/experimental/reward_loop/reward_manager/naive.py:104``) and has no
     counterpart in the dapo one, which also lacks the ``tool_extra_fields`` /
     ``__num_turns__`` plumbing at ``naive.py:46-53``.

So the penalty is applied at the one choke point every row passes through with
its length known and its reward still a scalar: ``AgentLoopWorker._postprocess``.
The formula below is DAPO's, transcribed verbatim from
``verl/experimental/reward_loop/reward_manager/dapo.py:109-113`` -- a linear ramp
inside the buffer, not a cliff.

THE FORMULA
-----------
With ``L`` the number of valid response tokens, ``M`` the max response length and
``b`` the buffer length::

    expected = M - b
    exceed   = L - expected
    penalty  = min(-exceed / b * factor, 0)

so ``penalty`` is 0 while ``L <= M - b``, then ramps down linearly across the
buffer, reaching ``-factor`` exactly at ``L == M``. Reward becomes
``score + penalty``.

SCALE NOTE
----------
DAPO's ``penalty_factor=1.0`` was tuned against a {-1, +1} accuracy reward. Our
rewards live in [0, 1] once ``FS_PERTASK_REWARD_NORM=1`` rescales them, so
``factor=1.0`` means a fully-capped response can be driven to -1.0, i.e. the
penalty spans twice the reward range. That is a deliberate, strong "stop hitting
the cap" signal, but it is worth choosing consciously: ``factor`` in 0.5-1.0 is
the sensible band on the [0,1] scale. The default here is DAPO's 1.0 so the
implementation matches the paper; the penalty is off unless explicitly enabled.

CONFIG (env vars; same convention as ``pertask_norm``, read inside Ray workers)
------------------------------------------------------------------------------
  FS_OVERLONG_PENALTY=0        master switch. 0 (default) -> exact no-op.
  FS_OVERLONG_BUFFER_LEN=4096  buffer width b, in tokens.
  FS_OVERLONG_PENALTY_FACTOR=1.0
  FS_OVERLONG_MAX_RESP_LEN=    max response length M. Defaults to the value the
                               caller passes in (``data.max_response_length``);
                               set it only to override.
  FS_OVERLONG_LOG=1            emit ``overlong_reward`` / ``overlong`` into
                               reward_extra_info for logging.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["OverlongPenaltyConfig", "overlong_penalty", "apply_overlong_penalty"]


_FLAG_TRUE = ("1", "true", "yes", "on")
_FLAG_FALSE = ("0", "false", "no", "off", "")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _FLAG_TRUE:
        return True
    if val in _FLAG_FALSE:
        return False
    # Loud (review m4): a typo like FS_OVERLONG_PENALTY=ture must not silently
    # disable the penalty AND the zero-raw advantage clamp gated on it.
    raise ValueError(f"malformed boolean env {name}={raw!r}; use one of {_FLAG_TRUE + _FLAG_FALSE}")


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        # Loud (review m4): silently substituting the default here would run
        # production with a penalty geometry the operator did not ask for.
        raise ValueError(f"malformed numeric env {name}={raw!r}") from None


@dataclass
class OverlongPenaltyConfig:
    """DAPO soft overlong punishment. Disabled by default -> exact no-op."""

    enable: bool = False
    buffer_len: int = 4096
    penalty_factor: float = 1.0
    max_resp_len: int | None = None
    log: bool = True

    @classmethod
    def from_env(cls, max_resp_len: int | None = None) -> "OverlongPenaltyConfig":
        cfg = cls(
            enable=_env_flag("FS_OVERLONG_PENALTY", False),
            buffer_len=int(_env_float("FS_OVERLONG_BUFFER_LEN", 4096.0)),
            penalty_factor=float(_env_float("FS_OVERLONG_PENALTY_FACTOR", 1.0)),
            max_resp_len=_env_float("FS_OVERLONG_MAX_RESP_LEN", None) or max_resp_len,
            log=_env_flag("FS_OVERLONG_LOG", True),
        )
        if cfg.enable:
            cfg.validate()
        return cfg

    def validate(self) -> None:
        # Same invariants DAPORewardManager asserts (workers/reward_manager/dapo.py:45-56),
        # so a misconfiguration fails at startup rather than silently mis-scaling rewards.
        if self.max_resp_len is None:
            raise ValueError(
                "FS_OVERLONG_PENALTY=1 requires a max response length: pass one in or set "
                "FS_OVERLONG_MAX_RESP_LEN."
            )
        if self.buffer_len <= 0:
            raise ValueError(f"FS_OVERLONG_BUFFER_LEN must be positive when enabled, got {self.buffer_len}")
        if self.max_resp_len < self.buffer_len:
            raise ValueError(
                f"FS_OVERLONG_MAX_RESP_LEN ({self.max_resp_len}) must be >= "
                f"FS_OVERLONG_BUFFER_LEN ({self.buffer_len})"
            )

    def describe(self) -> str:
        return (
            f"enable={self.enable} buffer_len={self.buffer_len} factor={self.penalty_factor} "
            f"max_resp_len={self.max_resp_len} log={self.log}"
        )


def overlong_penalty(valid_response_length, max_resp_len, buffer_len, penalty_factor: float = 1.0) -> float:
    """DAPO's soft overlong punishment. Returns a value in ``[-factor, 0]``.

    Transcribed from ``verl/experimental/reward_loop/reward_manager/dapo.py:109-113``:
    zero until ``max_resp_len - buffer_len``, then a linear ramp reaching
    ``-penalty_factor`` at ``max_resp_len``. Not a cliff.
    """
    expected_len = max_resp_len - buffer_len
    exceed_len = float(valid_response_length) - expected_len
    return float(min(-exceed_len / buffer_len * penalty_factor, 0.0))


def apply_overlong_penalty(
    score: float,
    valid_response_length,
    cfg: OverlongPenaltyConfig,
) -> tuple[float, dict]:
    """Return ``(reward, extra_info)``. A pure no-op when ``cfg.enable`` is False.

    ``extra_info`` uses float flags rather than bools on purpose: these values are
    wrapped in ``np.array(...)`` and later ``json.dumps``'d into the rollout JSONL,
    and ``np.bool_`` is not JSON serializable (same trap documented at
    ``verl/experimental/reward_loop/reward_manager/naive.py:110-112``).
    """
    if not cfg.enable:
        return float(score), {}
    penalty = overlong_penalty(
        valid_response_length=valid_response_length,
        max_resp_len=cfg.max_resp_len,
        buffer_len=cfg.buffer_len,
        penalty_factor=cfg.penalty_factor,
    )
    reward = float(score) + penalty
    if not cfg.log:
        return reward, {}
    return reward, {
        "overlong_reward": float(penalty),
        "overlong": 1.0 if penalty < 0 else 0.0,
        "reward_pre_overlong": float(score),
    }
