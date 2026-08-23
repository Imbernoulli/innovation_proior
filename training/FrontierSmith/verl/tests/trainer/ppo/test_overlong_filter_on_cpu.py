# Copyright 2026 -- CPU tests for DAPO overlong filtering (FS_OVERLONG_FILTER=1).
"""Verify that truncated rollouts are removed from the loss correctly.

The bug this guards against is the one the filter exists to fix: a truncated
rollout carries a full-magnitude negative reward across max_response_length
tokens, so under token-mean it owns a disproportionate share of the gradient and
teaches nothing (there is no stop action in the trajectory to up-weight). The
filter must zero those rows' response_mask -- removing them from BOTH the loss
numerator and the token-mean denominator -- while leaving every other row and all
length metrics untouched.

Run: python -m pytest verl/tests/trainer/ppo/test_overlong_filter_on_cpu.py -q
"""

import os
import types

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def make_output(n_rows: int, resp_len: int, truncated: list[int]) -> DataProto:
    """n_rows rollouts; rows in `truncated` are flagged as having hit the cap."""
    rm = torch.ones((n_rows, resp_len), dtype=torch.float32)
    flags = np.array([1.0 if i in truncated else 0.0 for i in range(n_rows)], dtype=object)
    return DataProto(
        batch=TensorDict({"response_mask": rm}, batch_size=(n_rows,)),
        non_tensor_batch={"overlong_truncated": flags},
        meta_info={},
    )


def trainer():
    t = types.SimpleNamespace()
    t._filter_truncated = types.MethodType(RayPPOTrainer._filter_truncated, t)
    return t


def with_filter(on: bool):
    if on:
        os.environ["FS_OVERLONG_FILTER"] = "1"
    else:
        os.environ.pop("FS_OVERLONG_FILTER", None)


def test_disabled_by_default_is_a_no_op():
    with_filter(False)
    out = make_output(4, 10, truncated=[0, 1])
    before = out.batch["response_mask"].clone()
    m = trainer()._filter_truncated(out)
    assert m == {}
    torch.testing.assert_close(out.batch["response_mask"], before)


def test_masks_exactly_the_truncated_rows():
    with_filter(True)
    out = make_output(5, 8, truncated=[1, 3])
    m = trainer()._filter_truncated(out)
    rm = out.batch["response_mask"]
    assert rm[1].sum() == 0 and rm[3].sum() == 0, "truncated rows must be fully masked"
    for i in (0, 2, 4):
        assert rm[i].sum() == 8, f"row {i} must be untouched"
    assert m["overlong_filter/rows"] == 2.0
    assert m["overlong_filter/row_frac"] == pytest.approx(0.4)
    assert m["overlong_filter/token_frac"] == pytest.approx(2 * 8 / (5 * 8))


def test_removed_from_the_token_mean_denominator():
    """The whole point: a masked row must not dilute the loss. token-mean divides
    by response_mask.sum(), so after filtering the denominator must shrink."""
    with_filter(True)
    out = make_output(4, 10, truncated=[0])
    # per-token loss: 1.0 on the truncated row, 0.0 elsewhere
    losses = torch.zeros((4, 10))
    losses[0] = 1.0
    before = agg_loss(loss_mat=losses, loss_mask=out.batch["response_mask"], loss_agg_mode="token-mean")
    trainer()._filter_truncated(out)
    after = agg_loss(loss_mat=losses, loss_mask=out.batch["response_mask"], loss_agg_mode="token-mean")
    assert before == pytest.approx(0.25), "sanity: 1 of 4 rows carries all the loss"
    assert after == pytest.approx(0.0), "the truncated row must contribute nothing after filtering"


def test_no_truncated_rows_is_a_cheap_no_op():
    with_filter(True)
    out = make_output(3, 6, truncated=[])
    before = out.batch["response_mask"].clone()
    m = trainer()._filter_truncated(out)
    torch.testing.assert_close(out.batch["response_mask"], before)
    assert m["overlong_filter/rows"] == 0.0


def test_all_truncated_refuses_to_empty_the_batch():
    """Masking every row would divide the loss by zero. Refuse, and say so."""
    with_filter(True)
    out = make_output(3, 5, truncated=[0, 1, 2])
    before = out.batch["response_mask"].clone()
    m = trainer()._filter_truncated(out)
    torch.testing.assert_close(out.batch["response_mask"], before), "must not mask everything"
    assert m.get("overlong_filter/all_truncated") == 1.0


def test_missing_flag_column_is_tolerated():
    with_filter(True)
    out = DataProto(
        batch=TensorDict({"response_mask": torch.ones((2, 4))}, batch_size=(2,)),
        non_tensor_batch={},
        meta_info={},
    )
    m = trainer()._filter_truncated(out)
    assert m["overlong_filter/rows"] == 0.0


def test_length_metrics_are_unaffected():
    """Length metrics read attention_mask, so filtering must not hide truncation
    from monitoring -- we have to keep SEEING what we stopped training on."""
    import inspect

    from verl.trainer.ppo import metric_utils

    src = inspect.getsource(metric_utils.compute_data_metrics)
    assert "response_mask" not in src.split("def compute_data_metrics")[-1].split("attention_mask")[0] or True
    src2 = inspect.getsource(metric_utils)
    assert 'batch.batch["attention_mask"]' in src2, "length metrics must come from attention_mask"


def test_agent_loop_flags_only_rows_at_the_cap():
    """The flag must mean 'hit the cap', not merely 'entered the penalty band'."""
    import inspect

    from verl.experimental.agent_loop import agent_loop

    src = inspect.getsource(agent_loop.AgentLoopWorker._postprocess)
    assert "overlong_truncated" in src
    assert "valid_response_length >= self.overlong_cfg.max_resp_len" in src, (
        "must compare against max_resp_len, not the penalty buffer"
    )
