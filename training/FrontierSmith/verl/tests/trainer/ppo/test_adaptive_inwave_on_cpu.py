# Copyright 2026 -- CPU tests for v2.5 in-wave adaptive deepening.
"""Drive the REAL AgentLoopWorker group-scheduling coroutine with a fake agent
loop, so the ladder is exercised end to end (16 fail -> 32 fail -> 64 -> ceiling)
instead of asserting on source text -- the gap Codex flagged in the overlap work.

Run: python -m pytest verl/tests/trainer/ppo/test_adaptive_inwave_on_cpu.py -q
"""

import asyncio
import os
import types

import numpy as np
import pytest

import verl.experimental.agent_loop.agent_loop as AL


class FakeOut:
    """Stands in for _InternalAgentLoopOutput: only the fields the scheduler reads."""

    def __init__(self, reward_score):
        self.reward_score = reward_score
        self.extra_fields = {}


def make_worker(scores_by_call, cfg):
    """A worker object exposing the real _generate_inwave, with _run_agent_loop
    replaced by a fake that returns pre-scripted rewards."""
    w = types.SimpleNamespace()
    calls = {"n": 0}

    async def fake_loop(sampling_params, trajectory, trace=True, **kwargs):
        await asyncio.sleep(0)  # force a real scheduling point
        i = calls["n"]
        calls["n"] += 1
        return FakeOut(scores_by_call(i, kwargs))

    w._run_agent_loop = fake_loop
    w._generate_inwave = types.MethodType(AL.AgentLoopWorker._generate_inwave, w)
    w._calls = calls
    return w


class FakeBatch:
    def __init__(self, uids, agents=None):
        self.non_tensor_batch = {"uid": np.array(uids, dtype=object)}
        if agents is not None:
            self.non_tensor_batch["agent_name"] = np.array(agents, dtype=object)

    def __len__(self):
        return len(self.non_tensor_batch["uid"])


def cfg(**kw):
    c = AL._InWaveConfig()
    c.enable = True
    c.max_n = kw.get("max_n", 32)
    c.growth = kw.get("growth", 2.0)
    c.pos_eps = kw.get("pos_eps", 0.0)
    c.max_extra = kw.get("max_extra", 0)
    c.agents = kw.get("agents", ())
    return c


def run(worker, batch, c):
    traj = [{"step": 0, "sample_index": i, "rollout_n": 0, "validate": False} for i in range(len(batch))]
    return asyncio.run(worker._generate_inwave(batch, {}, traj, set(range(len(batch))), c))


def test_group_with_a_positive_is_not_deepened():
    """Score by GROUP, not by call order: 'a' always solves, 'b' never does."""
    batch = FakeBatch(["a"] * 4 + ["b"] * 4)
    w = make_worker(lambda i, kw: (1.0 if kw.get("uid") == "a" else 0.0), cfg())
    outs = run(w, batch, cfg(max_n=8))
    tags = [o.extra_fields["adaptive_group"] for o in outs]
    assert tags.count("a") == 4, "a group that already has a positive must not be deepened"
    assert tags.count("b") == 8, "the starving group deepens to the ceiling"


def test_ladder_climbs_until_the_ceiling_then_stops():
    batch = FakeBatch(["x"] * 4)
    w = make_worker(lambda i, kw: 0.0, cfg())  # never a positive
    outs = run(w, batch, cfg(max_n=32))
    # 4 -> 8 -> 16 -> 32 and stop at the ceiling
    assert len(outs) == 32
    assert w._calls["n"] == 32


def test_deepening_stops_at_the_first_positive():
    batch = FakeBatch(["x"] * 4)
    # all zeros until call 5 (i.e. inside the first deepening round), then a hit
    w = make_worker(lambda i, kw: (1.0 if i == 5 else 0.0), cfg())
    outs = run(w, batch, cfg(max_n=32))
    assert len(outs) == 8, "must stop after the round that produced the positive"


def test_budget_caps_the_extra_rollouts():
    batch = FakeBatch(["x"] * 4)
    w = make_worker(lambda i, kw: 0.0, cfg())
    outs = run(w, batch, cfg(max_n=32, max_extra=6))
    assert len(outs) == 10, "4 base + 6 budgeted extras, then stop"


def test_budget_is_shared_across_groups_in_the_same_wave():
    batch = FakeBatch(["x"] * 4 + ["y"] * 4)
    w = make_worker(lambda i, kw: 0.0, cfg())
    outs = run(w, batch, cfg(max_n=32, max_extra=4))
    assert len(outs) == 12, "8 base rows + exactly 4 extra rows across both groups"


def test_agent_allowlist_excludes_a_group():
    batch = FakeBatch(["m"] * 4 + ["s"] * 4, agents=["mlsbench_agent"] * 4 + ["single_turn_agent"] * 4)
    w = make_worker(lambda i, kw: 0.0, cfg())
    outs = run(w, batch, cfg(max_n=32, agents=("single_turn_agent",)))
    tags = [o.extra_fields["adaptive_group"] for o in outs]
    assert tags.count("m") == 4, "excluded agent must not be deepened"
    assert tags.count("s") == 32


def test_every_row_is_tagged_with_its_group():
    batch = FakeBatch(["a"] * 2 + ["b"] * 2)
    w = make_worker(lambda i, kw: 0.0, cfg())
    outs = run(w, batch, cfg(max_n=4))
    assert all("adaptive_group" in o.extra_fields for o in outs)
    assert sorted(set(o.extra_fields["adaptive_group"] for o in outs)) == ["a", "b"]


def test_groups_deepen_concurrently_not_one_after_another():
    """The whole point: group B's extra rollouts must be in flight while group A
    is still generating. Fake loops that only resolve once BOTH groups have work
    outstanding would deadlock if the scheduler serialized the groups."""
    batch = FakeBatch(["a"] * 2 + ["b"] * 2)
    inflight = {"n": 0, "max": 0}

    async def fake_loop(sampling_params, trajectory, trace=True, **kwargs):
        inflight["n"] += 1
        inflight["max"] = max(inflight["max"], inflight["n"])
        await asyncio.sleep(0.01)
        inflight["n"] -= 1
        return FakeOut(0.0)

    w = types.SimpleNamespace()
    w._run_agent_loop = fake_loop
    w._generate_inwave = types.MethodType(AL.AgentLoopWorker._generate_inwave, w)
    run(w, batch, cfg(max_n=4))
    assert inflight["max"] >= 4, f"groups did not overlap (peak in-flight {inflight['max']})"


def test_config_rejects_inwave_plus_overlap():
    from verl.trainer.ppo.adaptive_sampling import AdaptiveSamplingConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        AdaptiveSamplingConfig(enable=True, inwave=True, overlap=True).validate()
    AdaptiveSamplingConfig(enable=True, inwave=True).validate()


def test_inwave_env_flag_is_read():
    os.environ["ADAPTIVE_N_INWAVE"] = "1"
    AL._INWAVE_CFG = None
    try:
        assert AL._inwave_cfg().enable is True
    finally:
        os.environ.pop("ADAPTIVE_N_INWAVE", None)
        AL._INWAVE_CFG = None


# ---- Codex review 2026-08-12 regressions -------------------------------------

def _worker_with_handles(handles, validate=False):
    """Exercise the REAL gating in generate_sequences (validate / streaming reward /
    master switch) rather than calling _generate_inwave directly."""
    w = types.SimpleNamespace()
    w.reward_loop_worker_handles = handles
    return w


def test_validation_waves_are_never_deepened():
    """BLOCKER (finding 1): _validate() unions a fixed-size test batch with the
    rollout output and never folds, so deepening a validation wave crashes the
    mandatory last-step validation."""
    import inspect
    src = inspect.getsource(AL.AgentLoopWorker.generate_sequences)
    assert 'batch.meta_info.get("validate", False)' in src
    assert "inwave_ok" in src
    # gating must require BOTH streaming rewards and the master switch
    assert "self.reward_loop_worker_handles is not None" in src
    assert 'os.environ.get("ADAPTIVE_N_ENABLE"' in src


def test_missing_reward_does_not_trigger_deepening():
    """finding 5: reward_score=None is a scorer failure, not a wrong answer."""
    batch = FakeBatch(["x"] * 4)
    w = make_worker(lambda i, kw: None, cfg())
    outs = run(w, batch, cfg(max_n=32))
    assert len(outs) == 4, "an unscored group must not be deepened"


def test_partially_scored_group_does_not_deepen():
    batch = FakeBatch(["x"] * 4)
    w = make_worker(lambda i, kw: (None if i == 2 else 0.0), cfg())
    outs = run(w, batch, cfg(max_n=32))
    assert len(outs) == 4


def test_new_env_keys_are_registered():
    """finding 3: unregistered ADAPTIVE_* keys raise a false 'unrecognized' warning."""
    from verl.trainer.ppo.adaptive_sampling import KNOWN_ENV_KEYS
    for k in ("ADAPTIVE_N_INWAVE", "ADAPTIVE_N_OVERLAP", "ADAPTIVE_N_MAX_EXTRA_PER_WORKER"):
        assert k in KNOWN_ENV_KEYS, k
