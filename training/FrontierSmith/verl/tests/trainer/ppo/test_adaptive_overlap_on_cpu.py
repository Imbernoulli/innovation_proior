# Copyright 2026 -- CPU tests for v2.4 pipelined ("overlap") deepening.
"""No GPU, no Ray: exercise the two pieces of the overlap path that can silently
corrupt a run -- the per-prompt wave expansion and the carry/requeue bookkeeping.

Run: python -m pytest verl/tests/trainer/ppo/test_adaptive_overlap_on_cpu.py -q
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.adaptive_sampling import AdaptiveSamplingConfig, next_round_target
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def make_batch(num_prompts: int) -> DataProto:
    return DataProto(
        batch=TensorDict(
            {"input_ids": torch.arange(num_prompts * 4).reshape(num_prompts, 4)},
            batch_size=(num_prompts,),
        ),
        non_tensor_batch={"uid": np.array([f"u{i}" for i in range(num_prompts)], dtype=object)},
        meta_info={},
    )


class FakeTrainer:
    """Bind the real methods to a minimal object (RayPPOTrainer.__init__ needs a
    live cluster). Only the state the methods touch is provided."""

    _expand_gen_batch = RayPPOTrainer._expand_gen_batch

    def __init__(self, cfg: AdaptiveSamplingConfig, pending: dict | None = None):
        self.adaptive_cfg = cfg
        self._pending_deepen = dict(pending or {})


def cfg_overlap(**kw) -> AdaptiveSamplingConfig:
    base = dict(enable=True, trigger="no_positive", overlap=True, max_n=128, growth=2.0, max_extra=0)
    base.update(kw)
    return AdaptiveSamplingConfig(**base)


def test_expansion_without_pending_is_exactly_repeat_interleave():
    t = FakeTrainer(cfg_overlap())
    b = make_batch(4)
    out, idx, m = t._expand_gen_batch(b, 16)
    assert len(out) == 64
    np.testing.assert_array_equal(idx, np.repeat(np.arange(4), 16))
    ref = b.repeat(repeat_times=16, interleave=True)
    torch.testing.assert_close(out.batch["input_ids"], ref.batch["input_ids"])
    assert m == {}


def test_carried_prompt_is_expanded_in_wave():
    # prompt 2 was starving last step and must come back at n=32
    t = FakeTrainer(cfg_overlap(), pending={2: 32})
    out, idx, m = t._expand_gen_batch(make_batch(4), 16)
    assert len(out) == 16 * 3 + 32
    counts = np.bincount(idx, minlength=4)
    np.testing.assert_array_equal(counts, [16, 16, 32, 16])
    assert m["adaptive/overlap_extra_rows"] == 16.0
    assert m["adaptive/overlap_prompts_deepened"] == 1.0
    assert m["adaptive/overlap_max_group_size"] == 32.0


def test_pending_is_consumed_once():
    t = FakeTrainer(cfg_overlap(), pending={0: 32})
    t._expand_gen_batch(make_batch(2), 16)
    _, idx, _ = t._expand_gen_batch(make_batch(2), 16)
    np.testing.assert_array_equal(np.bincount(idx, minlength=2), [16, 16])


def test_budget_clip_stays_on_the_n_grid():
    # 3 carried prompts want +16 each but only 24 extra rows are allowed:
    # 16 fits, the 8-row remainder must be refused (never a ragged wave).
    t = FakeTrainer(cfg_overlap(max_extra=24), pending={0: 32, 1: 32, 2: 32})
    out, idx, m = t._expand_gen_batch(make_batch(3), 16)
    assert m["adaptive/overlap_extra_rows"] == 16.0
    assert len(out) % 16 == 0
    counts = np.bincount(idx, minlength=3)
    assert sorted(counts.tolist()) == [16, 16, 32]


def test_ceiling_is_respected():
    t = FakeTrainer(cfg_overlap(max_n=32), pending={0: 64})
    _, idx, _ = t._expand_gen_batch(make_batch(2), 16)
    assert np.bincount(idx, minlength=2)[0] == 32  # clipped to max_n, not 64


def test_overlap_mode_requires_requeue_and_no_positive_trigger():
    with pytest.raises(ValueError, match="ADAPTIVE_REQUEUE_ENABLE"):
        AdaptiveSamplingConfig(enable=True, overlap=True, requeue_enable=False).validate()
    with pytest.raises(ValueError, match="trigger=no_positive"):
        AdaptiveSamplingConfig(enable=True, overlap=True, trigger="zero_variance").validate()
    with pytest.raises(ValueError, match="ADAPTIVE_N_KEEP"):
        AdaptiveSamplingConfig(enable=True, overlap=True, keep="all").validate()
    AdaptiveSamplingConfig(enable=True, overlap=True).validate()  # defaults are fine


def test_ladder_targets_match_the_sequential_mode():
    cfg = cfg_overlap()
    assert next_round_target(16, cfg) == 32
    assert next_round_target(32, cfg) == 64
    assert next_round_target(64, cfg) == 128
    assert min(next_round_target(128, cfg), cfg.max_n) == 128  # ceiling


def test_oversized_wave_must_fold_back_to_n():
    """The bug this guards: in overlap mode `extra_outputs` is always empty, so the
    early return skipped the fold and a carried group entered the loss at 2n rows
    (double weight, double share of the token-mean denominator)."""
    import inspect
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    src = inspect.getsource(RayPPOTrainer._adaptive_deepen)
    assert "oversized" in src, "the ragged-wave guard is gone"
    # the early return must be conditioned on BOTH no extra rounds AND uniform width
    assert "if not extra_outputs and not oversized:" in src
    # and the folded batch must be described by a uniform index, not the ragged one
    assert "folded_index = np.repeat(" in src
    assert "return folded, folded_index, metrics" in src


def test_fold_keeps_exactly_n_rows_per_prompt():
    """plan_retention is the fold primitive: a 32-sample group folds to n=16 with
    Horvitz-Thompson weights summing to n, and a positive is never dropped."""
    import numpy as np
    from verl.trainer.ppo.adaptive_sampling import plan_retention

    cfg = cfg_overlap()
    rng = np.random.default_rng(0)
    scores = np.zeros(32)
    scores[7] = 1.0  # the single 1-in-32 success
    plan = plan_retention(scores, n=16, cfg=cfg, rng=rng, protect=scores > 0)
    assert len(plan.keep_idx) == 16
    assert 7 in plan.keep_idx.tolist(), "the protected positive must survive the fold"
    assert abs(float(np.sum(plan.weights)) - 16.0) < 1e-9, "HT weights must sum to n"
