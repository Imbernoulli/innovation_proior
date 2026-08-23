# Copyright 2026 - tests for per-prompt adaptive resampling and the overlong penalty.
"""CPU unit tests for the two anti-dead-group features.

Run:  pytest -q verl/tests/trainer/ppo/test_adaptive_sampling_on_cpu.py
"""

import os
from unittest import mock

import numpy as np
import pytest
import torch

from verl.trainer.ppo.adaptive_sampling import (
    AdaptiveSamplingConfig,
    adaptive_advantage_values,
    flat_group_mask,
    group_indices,
    group_needs_deepening,
    has_positive,
    next_round_target,
    plan_retention,
)
from verl.utils.reward_score.overlong_penalty import (
    OverlongPenaltyConfig,
    apply_overlong_penalty,
    overlong_penalty,
)

GRPO_EPS = 1e-6


def _cfg(**kw) -> AdaptiveSamplingConfig:
    # trigger defaults to zero_variance in these fixtures so the v1 tests keep
    # exercising exactly the v1 semantics; v2 tests opt in explicitly.
    base = dict(
        enable=True, trigger="zero_variance", max_n=128, growth=2.0, eps=1e-6,
        keep="subsample", stratify=True, seed=0,
    )
    base.update(kw)
    return AdaptiveSamplingConfig(**base)


def _cfg_v2(**kw) -> AdaptiveSamplingConfig:
    base = dict(
        enable=True, trigger="no_positive", pos_eps=0.0, max_n=128, growth=2.0,
        eps=1e-6, keep="subsample", stratify=True, seed=0,
    )
    base.update(kw)
    return AdaptiveSamplingConfig(**base)


# =====================================================================
# Disabled-by-default: both features must be exact no-ops out of the box
# =====================================================================


class TestDisabledByDefault:
    def test_adaptive_disabled_with_clean_env(self):
        adaptive_env = {k: v for k, v in os.environ.items() if not k.startswith("ADAPTIVE_N_")}
        with mock.patch.dict(os.environ, adaptive_env, clear=True):
            cfg = AdaptiveSamplingConfig.from_env()
        assert cfg.enable is False
        assert cfg.max_n == 128 and cfg.keep == "subsample" and cfg.stratify is True

    def test_overlong_disabled_with_clean_env(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FS_OVERLONG_")}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = OverlongPenaltyConfig.from_env(max_resp_len=32768)
        assert cfg.enable is False

    def test_overlong_disabled_is_identity(self):
        cfg = OverlongPenaltyConfig(enable=False, buffer_len=4096, penalty_factor=1.0, max_resp_len=32768)
        # even a maximally-overlong response must come back untouched
        reward, extra = apply_overlong_penalty(score=0.375, valid_response_length=32768, cfg=cfg)
        assert reward == 0.375
        assert extra == {}

    def test_adaptive_disabled_config_never_validates_away_defaults(self):
        # a disabled overlong cfg must not raise even with a missing max_resp_len
        clean = {k: v for k, v in os.environ.items() if not k.startswith("FS_OVERLONG_")}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = OverlongPenaltyConfig.from_env(max_resp_len=None)
        assert cfg.enable is False  # and no exception


# =====================================================================
# FEATURE 2 -- DAPO soft overlong punishment: the ramp, not a cliff
# =====================================================================


class TestOverlongPenaltyRamp:
    M = 32768
    B = 4096
    F = 1.0

    def p(self, length, factor=None):
        return overlong_penalty(length, self.M, self.B, self.F if factor is None else factor)

    def test_no_penalty_below_the_buffer(self):
        assert self.p(0) == 0.0
        assert self.p(1000) == 0.0
        assert self.p(self.M - self.B - 1) == 0.0

    def test_zero_exactly_at_the_buffer_edge(self):
        # L == M - b  =>  exceed == 0  =>  penalty == 0 (start of the ramp)
        assert self.p(self.M - self.B) == 0.0

    def test_linear_ramp_inside_the_buffer(self):
        quarter = self.p(self.M - self.B + self.B // 4)
        half = self.p(self.M - self.B // 2)
        three_q = self.p(self.M - self.B + 3 * self.B // 4)
        assert quarter == pytest.approx(-0.25)
        assert half == pytest.approx(-0.5)
        assert three_q == pytest.approx(-0.75)
        # equal steps in length produce equal steps in penalty -> it is a ramp
        assert (half - quarter) == pytest.approx(three_q - half)

    def test_full_penalty_at_the_cap(self):
        assert self.p(self.M) == pytest.approx(-self.F)

    def test_penalty_keeps_ramping_past_the_cap(self):
        # DAPO does not clamp the lower end; a longer-than-max response (possible
        # when the buffer maths is fed a different M) keeps getting worse.
        assert self.p(self.M + self.B) == pytest.approx(-2.0)

    def test_penalty_factor_scales_the_ramp(self):
        assert self.p(self.M, factor=0.5) == pytest.approx(-0.5)
        assert self.p(self.M - self.B // 2, factor=0.5) == pytest.approx(-0.25)

    def test_monotone_non_increasing_in_length(self):
        lengths = np.arange(0, self.M + 1, 251)
        vals = [self.p(int(x)) for x in lengths]
        assert all(b <= a + 1e-12 for a, b in zip(vals[:-1], vals[1:], strict=True))

    def test_matches_shipped_dapo_arithmetic_verbatim(self):
        """Byte-compatible with verl/experimental/reward_loop/reward_manager/dapo.py:109-113."""
        for length in [0, 1, 12345, 28671, 28672, 30000, 32767, 32768, 40000]:
            # transcription of the shipped implementation
            overlong_buffer_len = self.B
            expected_len = self.M - overlong_buffer_len
            exceed_len = length - expected_len
            overlong_penalty_factor = self.F
            reference = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            assert self.p(length) == pytest.approx(float(reference)), f"mismatch at {length}"

    def test_torch_scalar_length_is_accepted(self):
        # _postprocess hands us an int() of a tensor element; make sure a raw tensor
        # scalar also works, since that is an easy future refactor to make.
        assert overlong_penalty(torch.tensor(32768), self.M, self.B, self.F) == pytest.approx(-1.0)


class TestOverlongPenaltyApplication:
    def test_enabled_penalty_is_added_to_the_score(self):
        cfg = OverlongPenaltyConfig(enable=True, buffer_len=4096, penalty_factor=1.0, max_resp_len=32768)
        reward, extra = apply_overlong_penalty(score=0.8, valid_response_length=32768, cfg=cfg)
        assert reward == pytest.approx(-0.2)  # 0.8 + (-1.0)
        assert extra["overlong_reward"] == pytest.approx(-1.0)
        assert extra["overlong"] == 1.0
        assert extra["reward_pre_overlong"] == pytest.approx(0.8)

    def test_short_response_is_untouched_but_still_logged(self):
        cfg = OverlongPenaltyConfig(enable=True, buffer_len=4096, penalty_factor=1.0, max_resp_len=32768)
        reward, extra = apply_overlong_penalty(score=0.8, valid_response_length=1000, cfg=cfg)
        assert reward == pytest.approx(0.8)
        assert extra["overlong_reward"] == 0.0
        assert extra["overlong"] == 0.0

    def test_log_flags_are_floats_not_numpy_bools(self):
        # np.bool_ is not JSON serializable and crashes the rollout JSONL dump;
        # see verl/experimental/reward_loop/reward_manager/naive.py:110-112.
        cfg = OverlongPenaltyConfig(enable=True, buffer_len=4096, penalty_factor=1.0, max_resp_len=32768)
        _, extra = apply_overlong_penalty(score=0.0, valid_response_length=32768, cfg=cfg)
        import json

        json.dumps(extra)  # must not raise
        assert isinstance(extra["overlong"], float)

    def test_log_off_suppresses_extra_info_but_not_the_penalty(self):
        cfg = OverlongPenaltyConfig(
            enable=True, buffer_len=4096, penalty_factor=1.0, max_resp_len=32768, log=False
        )
        reward, extra = apply_overlong_penalty(score=1.0, valid_response_length=32768, cfg=cfg)
        assert reward == pytest.approx(0.0)
        assert extra == {}

    def test_validation_rejects_a_buffer_larger_than_max(self):
        with pytest.raises(ValueError, match="must be >="):
            OverlongPenaltyConfig(enable=True, buffer_len=40000, penalty_factor=1.0, max_resp_len=32768).validate()

    def test_validation_requires_a_max_resp_len(self):
        with pytest.raises(ValueError, match="max response length"):
            OverlongPenaltyConfig(enable=True, buffer_len=4096, max_resp_len=None).validate()

    def test_env_roundtrip(self):
        env = {
            "FS_OVERLONG_PENALTY": "1",
            "FS_OVERLONG_BUFFER_LEN": "8192",
            "FS_OVERLONG_PENALTY_FACTOR": "0.5",
        }
        with mock.patch.dict(os.environ, env):
            cfg = OverlongPenaltyConfig.from_env(max_resp_len=32768)
        assert cfg.enable and cfg.buffer_len == 8192 and cfg.penalty_factor == 0.5
        assert cfg.max_resp_len == 32768


# =====================================================================
# FEATURE 1 -- flat-group detection and the round schedule
# =====================================================================


class TestFlatGroupDetection:
    def test_all_zero_group_is_flat(self):
        assert flat_group_mask(np.zeros(16), eps=1e-6) is True

    def test_all_identical_nonzero_group_is_flat(self):
        # the other dead-group shape: every sample solves it perfectly
        assert flat_group_mask(np.ones(16), eps=1e-6) is True

    def test_one_success_in_sixteen_is_live(self):
        s = np.zeros(16)
        s[3] = 0.42
        assert flat_group_mask(s, eps=1e-6) is False

    def test_eps_boundary(self):
        s = np.zeros(16)
        s[0] = 1e-6
        assert flat_group_mask(s, eps=1e-6) is True  # <= eps counts as flat
        s[0] = 1.1e-6
        assert flat_group_mask(s, eps=1e-6) is False

    def test_singleton_group_is_flat(self):
        assert flat_group_mask(np.array([0.7]), eps=1e-6) is True

    def test_tiny_penalty_induced_spread_counts_as_live(self):
        # with the overlong penalty on, two truncated-but-different-length samples
        # differ slightly -> a real "be shorter" gradient, so NOT dead.
        s = np.zeros(16)
        s[0] = -1.0
        s[1] = -0.9998
        assert flat_group_mask(s, eps=1e-6) is False


class TestRoundSchedule:
    def test_doubling_schedule_to_the_ceiling(self):
        cfg = _cfg(max_n=128, growth=2.0)
        seq, cur = [], 16
        for _ in range(6):
            nxt = next_round_target(cur, cfg)
            if nxt == cur:
                break
            seq.append(nxt)
            cur = nxt
        assert seq == [32, 64, 128]

    def test_ceiling_is_a_fixed_point(self):
        cfg = _cfg(max_n=128)
        assert next_round_target(128, cfg) == 128
        assert next_round_target(200, cfg) == 200

    def test_ceiling_clamps_a_partial_round(self):
        cfg = _cfg(max_n=100, growth=2.0)
        assert next_round_target(64, cfg) == 100

    def test_ceiling_256(self):
        cfg = _cfg(max_n=256, growth=2.0)
        seq, cur = [], 16
        while True:
            nxt = next_round_target(cur, cfg)
            if nxt == cur:
                break
            seq.append(nxt)
            cur = nxt
        assert seq == [32, 64, 128, 256]

    def test_growth_must_exceed_one(self):
        with pytest.raises(ValueError, match="GROWTH"):
            AdaptiveSamplingConfig(enable=True, growth=1.0).validate()

    def test_keep_mode_is_validated(self):
        with pytest.raises(ValueError, match="ADAPTIVE_N_KEEP"):
            AdaptiveSamplingConfig(enable=True, keep="everything").validate()


class TestGroupIndices:
    def test_interleaved_uids_group_correctly(self):
        uids = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
        g = group_indices(uids)
        assert list(g.keys()) == ["a", "b", "c"]
        np.testing.assert_array_equal(g["b"], np.array([2, 3]))


# =====================================================================
# FEATURE 1 -- retention plan: the Horvitz-Thompson invariants
# =====================================================================


class TestRetentionPlan:
    def test_group_not_enlarged_is_the_identity(self):
        scores = np.array([0.0] * 15 + [1.0])
        plan = plan_retention(scores, n=16, cfg=_cfg())
        assert plan.enlarged is False
        np.testing.assert_array_equal(plan.keep_idx, np.arange(16))
        np.testing.assert_allclose(plan.weights, np.ones(16))

    def test_baseline_uses_all_K_samples_and_matches_torch_std(self):
        rng = np.random.default_rng(7)
        scores = rng.normal(size=128)
        plan = plan_retention(scores, n=16, cfg=_cfg())
        assert plan.group_size == 128
        assert plan.mean == pytest.approx(float(scores.mean()))
        # verl's GRPO baseline uses torch.std (ddof=1); ours must agree exactly
        assert plan.std == pytest.approx(float(torch.std(torch.tensor(scores)).item()))

    def test_uniform_retention_gives_weights_of_exactly_one(self):
        """p_i = n/K  =>  w_i = n/(K * n/K) = 1. The unstratified case is the identity."""
        scores = np.concatenate([np.zeros(120), np.ones(8)])
        plan = plan_retention(scores, n=16, cfg=_cfg(stratify=False))
        assert plan.enlarged is True
        assert len(plan.keep_idx) == 16
        np.testing.assert_allclose(plan.weights, np.ones(16))

    def test_stratified_weights_sum_to_exactly_n(self):
        """sum_{i in S} w_i == n identically: a deepened group carries the same
        total mass into the update as any other group -- it cannot dominate."""
        for K, n_pos in [(32, 1), (64, 3), (128, 1), (128, 40), (128, 127)]:
            scores = np.concatenate([np.ones(n_pos), np.zeros(K - n_pos)])
            plan = plan_retention(scores, n=16, cfg=_cfg())
            assert len(plan.keep_idx) == 16, (K, n_pos)
            assert plan.weights.sum() == pytest.approx(16.0), (K, n_pos)

    def test_stratification_always_retains_the_rare_success(self):
        """A 1-in-128 success survives a uniform 16-of-128 draw only ~12% of the
        time. Stratification must keep it every single time."""
        scores = np.zeros(128)
        scores[99] = 0.75
        for seed in range(50):
            plan = plan_retention(scores, n=16, cfg=_cfg(seed=seed), rng=np.random.default_rng(seed))
            assert 99 in set(plan.keep_idx.tolist()), f"lost the success at seed {seed}"

    def test_worked_example_weights(self):
        """K=128, one success: w_min = 16/(128*1) = 0.125, w_maj = 16/(128*(15/127))."""
        scores = np.zeros(128)
        scores[0] = 1.0
        plan = plan_retention(scores, n=16, cfg=_cfg(), rng=np.random.default_rng(0))
        w = dict(zip(plan.keep_idx.tolist(), plan.weights.tolist(), strict=True))
        assert w[0] == pytest.approx(0.125)
        majority_w = [v for k, v in w.items() if k != 0]
        assert all(x == pytest.approx(16 / (128 * (15 / 127))) for x in majority_w)
        assert len(majority_w) == 15

    def test_minority_capped_at_half_the_budget(self):
        # 40 successes out of 128: keep at most n//2 = 8 of them, so the retained
        # group does not become all-minority.
        scores = np.concatenate([np.ones(40), np.zeros(88)])
        plan = plan_retention(scores, n=16, cfg=_cfg(), rng=np.random.default_rng(1))
        kept_minority = sum(1 for i in plan.keep_idx if scores[i] == 1.0)
        assert kept_minority == 8
        assert plan.weights.sum() == pytest.approx(16.0)

    def test_still_flat_at_K_falls_back_to_uniform(self):
        # deepened to 128 and STILL all zeros -> nothing to protect, weights == 1
        plan = plan_retention(np.zeros(128), n=16, cfg=_cfg())
        assert plan.enlarged is True
        np.testing.assert_allclose(plan.weights, np.ones(16))

    def test_retention_indices_are_sorted_and_unique(self):
        scores = np.concatenate([np.ones(5), np.zeros(123)])
        plan = plan_retention(scores, n=16, cfg=_cfg(), rng=np.random.default_rng(3))
        idx = plan.keep_idx.tolist()
        assert idx == sorted(idx)
        assert len(set(idx)) == len(idx)


# =====================================================================
# FEATURE 1 -- advantage math over enlarged groups
# =====================================================================


def _reference_grpo_advantage(scores: np.ndarray, norm_by_std=True) -> np.ndarray:
    """Independent transcription of core_algos.compute_grpo_outcome_advantage
    for a single group, used as the ground truth for the no-op guarantee."""
    t = torch.tensor(scores, dtype=torch.float64)
    mean = torch.mean(t)
    std = torch.std(t)
    if norm_by_std:
        return ((t - mean) / (std + GRPO_EPS)).numpy()
    return (t - mean).numpy()


class TestAdvantageMath:
    def test_non_enlarged_group_is_bit_identical_to_vanilla_grpo(self):
        """The no-op guarantee: a group that was never deepened must produce
        exactly the advantage the untouched trainer produces."""
        rng = np.random.default_rng(11)
        scores = rng.normal(size=16)
        plan = plan_retention(scores, n=16, cfg=_cfg())
        ours = adaptive_advantage_values(scores, plan.mean, plan.std, plan.weights)
        np.testing.assert_allclose(ours, _reference_grpo_advantage(scores), rtol=0, atol=1e-12)

    def test_dead_group_produces_exactly_zero_advantage(self):
        """This is the bug being fixed: 16 identical scores -> zero gradient."""
        scores = np.zeros(16)
        plan = plan_retention(scores, n=16, cfg=_cfg())
        adv = adaptive_advantage_values(scores, plan.mean, plan.std, plan.weights)
        np.testing.assert_allclose(adv, np.zeros(16))
        assert np.abs(adv).sum() == 0.0

    def test_deepening_turns_a_dead_group_into_a_live_one(self):
        """0/16 dead -> deepened to 1/128 -> real, nonzero gradient."""
        deepened = np.zeros(128)
        deepened[64] = 0.6
        plan = plan_retention(deepened, n=16, cfg=_cfg(), rng=np.random.default_rng(0))
        kept_scores = deepened[plan.keep_idx]
        adv = adaptive_advantage_values(kept_scores, plan.mean, plan.std, plan.weights)
        assert np.abs(adv).sum() > 0.0
        # the retained success carries positive advantage, the zeros negative
        pos = adv[kept_scores > 0]
        neg = adv[kept_scores == 0]
        assert pos.size == 1 and pos[0] > 0
        assert np.all(neg < 0)

    def test_worked_example_numbers_from_the_design_doc(self):
        """scores=[s, 0x127] => a_pos=11.22, w=0.125 => 1.40; contributions ~sum to 0."""
        s = 1.0
        scores = np.zeros(128)
        scores[0] = s
        plan = plan_retention(scores, n=16, cfg=_cfg(), rng=np.random.default_rng(0))
        assert plan.mean == pytest.approx(s / 128)
        assert plan.std == pytest.approx(s / np.sqrt(128), rel=1e-9)

        raw_pos = (s - plan.mean) / (plan.std + GRPO_EPS)
        assert raw_pos == pytest.approx(11.22, abs=0.02)

        kept = scores[plan.keep_idx]
        adv = adaptive_advantage_values(kept, plan.mean, plan.std, plan.weights)
        weighted_pos = adv[kept > 0][0]
        assert weighted_pos == pytest.approx(1.40, abs=0.02)
        # a well-formed GRPO group has contributions that cancel
        assert adv.sum() == pytest.approx(0.0, abs=1e-9)

    def test_deepened_success_is_quieter_than_a_1_of_16_success(self):
        """Not a bug: at 1/128 the policy really is worse than 1/16 implies, and
        GRPO's std normalization inflates rare successes in small groups. The win
        is that the group is alive at all, not that it is louder."""
        shallow = np.zeros(16)
        shallow[0] = 1.0
        shallow_adv = _reference_grpo_advantage(shallow)[0]
        assert shallow_adv == pytest.approx(3.75, abs=0.02)

        deep = np.zeros(128)
        deep[0] = 1.0
        plan = plan_retention(deep, n=16, cfg=_cfg(), rng=np.random.default_rng(0))
        kept = deep[plan.keep_idx]
        deep_adv = adaptive_advantage_values(kept, plan.mean, plan.std, plan.weights)[kept > 0][0]
        assert deep_adv < shallow_adv

    def test_horvitz_thompson_estimator_is_unbiased(self):
        """E_S[ (1/n) sum_{i in S} w_i a_i x_i ] == (1/K) sum_{i=1..K} a_i x_i.

        Averaged over many retention draws, the stratified estimator must recover
        the full-group gradient it stands in for.
        """
        K, n = 64, 16
        rng_scores = np.random.default_rng(5)
        scores = np.where(rng_scores.random(K) < 0.15, 1.0, 0.0)
        assert 0 < scores.sum() < K  # genuinely mixed

        plan0 = plan_retention(scores, n=n, cfg=_cfg())
        mean, std = plan0.mean, plan0.std
        full_adv = (scores - mean) / (std + GRPO_EPS)
        # an arbitrary per-sample "gradient direction" to weight against
        x = np.random.default_rng(6).normal(size=K)
        truth = float(np.mean(full_adv * x))

        estimates = []
        for seed in range(4000):
            plan = plan_retention(scores, n=n, cfg=_cfg(seed=seed), rng=np.random.default_rng(seed))
            kept = plan.keep_idx
            adv = adaptive_advantage_values(scores[kept], mean, std, plan.weights)
            estimates.append(float(np.mean(adv * x[kept])))

        assert float(np.mean(estimates)) == pytest.approx(truth, abs=0.02 * max(1.0, abs(truth)) + 0.01)

    def test_stratified_has_lower_variance_than_uniform(self):
        """The point of stratifying: the rare success is always present, so the
        per-step gradient stops being a coin flip."""
        K, n = 128, 16
        scores = np.zeros(K)
        scores[0] = 1.0
        plan0 = plan_retention(scores, n=n, cfg=_cfg())
        mean, std = plan0.mean, plan0.std

        def spread(stratify):
            vals = []
            for seed in range(300):
                p = plan_retention(scores, n=n, cfg=_cfg(stratify=stratify, seed=seed), rng=np.random.default_rng(seed))
                adv = adaptive_advantage_values(scores[p.keep_idx], mean, std, p.weights)
                vals.append(float(adv.sum()))
            return float(np.std(vals))

        assert spread(True) < spread(False)

    def test_norm_by_std_off_is_mean_centering_only(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        plan = plan_retention(scores, n=4, cfg=_cfg())
        adv = adaptive_advantage_values(scores, plan.mean, plan.std, plan.weights, norm_adv_by_std=False)
        np.testing.assert_allclose(adv, scores - 0.5)

    def test_broadcasting_over_per_row_baselines(self):
        """_apply_adaptive_advantage passes per-row mean/std arrays; make sure the
        vectorized form agrees with the scalar one."""
        scores = np.array([0.0, 1.0, 0.0, 1.0])
        means = np.full(4, 0.5)
        stds = np.full(4, 0.5)
        w = np.ones(4)
        vec = adaptive_advantage_values(scores, means, stds, w)
        scalar = adaptive_advantage_values(scores, 0.5, 0.5, w)
        np.testing.assert_allclose(vec, scalar)


# =====================================================================
# FEATURE 1 -- trainer-level helpers (no Ray required)
# =====================================================================


class TestTrainerHelpers:
    def test_group_scores_slices_interleaved_rows(self):
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        prompt_index = np.array([0, 0, 0, 1, 1, 1])
        groups = RayPPOTrainer._group_scores(scores, prompt_index, 2)
        np.testing.assert_allclose(groups[0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(groups[1], [3.0, 4.0, 5.0])

    def test_apply_adaptive_advantage_only_touches_enlarged_rows(self):
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        rows, seqlen = 4, 3
        token_level_rewards = torch.zeros(rows, seqlen, dtype=torch.float32)
        token_level_rewards[1, -1] = 1.0  # a success in the enlarged group
        response_mask = torch.ones(rows, seqlen, dtype=torch.float32)
        original_adv = torch.full((rows, seqlen), 7.0, dtype=torch.float32)

        batch = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_mask": response_mask,
                "advantages": original_adv.clone(),
                "returns": original_adv.clone(),
            },
            non_tensors={
                # rows 0,1 were deepened; rows 2,3 were not
                "adaptive_enlarged": np.array([1.0, 1.0, 0.0, 0.0]),
                "adaptive_group_mean": np.array([0.25, 0.25, 0.0, 0.0]),
                "adaptive_group_std": np.array([0.5, 0.5, 1.0, 1.0]),
                "adaptive_weight": np.array([2.0, 0.5, 1.0, 1.0]),
            },
        )
        out = RayPPOTrainer._apply_adaptive_advantage(None, batch, True)
        adv = out.batch["advantages"]

        # untouched rows keep the value compute_advantage produced
        assert torch.allclose(adv[2], torch.full((seqlen,), 7.0))
        assert torch.allclose(adv[3], torch.full((seqlen,), 7.0))
        # enlarged rows are recomputed against the K-sample baseline, times weight
        expected_row0 = (0.0 - 0.25) / (0.5 + GRPO_EPS) * 2.0
        expected_row1 = (1.0 - 0.25) / (0.5 + GRPO_EPS) * 0.5
        assert adv[0, 0].item() == pytest.approx(expected_row0, rel=1e-5)
        assert adv[1, 0].item() == pytest.approx(expected_row1, rel=1e-5)
        # returns track advantages for GRPO
        assert torch.allclose(out.batch["returns"][0], adv[0])

    def test_apply_adaptive_advantage_is_a_noop_without_the_columns(self):
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.zeros(2, 3),
                "response_mask": torch.ones(2, 3),
                "advantages": torch.full((2, 3), 5.0),
            }
        )
        out = RayPPOTrainer._apply_adaptive_advantage(None, batch, True)
        assert torch.allclose(out.batch["advantages"], torch.full((2, 3), 5.0))

    def test_apply_adaptive_advantage_is_a_noop_when_nothing_was_enlarged(self):
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        batch = DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.zeros(2, 3),
                "response_mask": torch.ones(2, 3),
                "advantages": torch.full((2, 3), 5.0),
            },
            non_tensors={
                "adaptive_enlarged": np.array([0.0, 0.0]),
                "adaptive_group_mean": np.array([0.0, 0.0]),
                "adaptive_group_std": np.array([1.0, 1.0]),
                "adaptive_weight": np.array([1.0, 1.0]),
            },
        )
        out = RayPPOTrainer._apply_adaptive_advantage(None, batch, True)
        assert torch.allclose(out.batch["advantages"], torch.full((2, 3), 5.0))

    def test_masked_positions_stay_zero(self):
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        response_mask = torch.tensor([[1.0, 1.0, 0.0]])
        batch = DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.tensor([[0.0, 0.0, 1.0]]),
                "response_mask": response_mask,
                "advantages": torch.zeros(1, 3),
            },
            non_tensors={
                "adaptive_enlarged": np.array([1.0]),
                "adaptive_group_mean": np.array([0.5]),
                "adaptive_group_std": np.array([0.5]),
                "adaptive_weight": np.array([1.0]),
            },
        )
        out = RayPPOTrainer._apply_adaptive_advantage(None, batch, True)
        assert out.batch["advantages"][0, 2].item() == 0.0
        assert out.batch["advantages"][0, 0].item() != 0.0


# =====================================================================
# End-to-end shape contract: fold-back must preserve the batch shape
# =====================================================================


class TestV2NoPositiveTrigger:
    """V2: the trigger judges RAW (pre-penalty) scores; penalty variance must
    never satisfy it, and default config selects it."""

    def test_default_trigger_is_no_positive(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("ADAPTIVE_N_")}
        with mock.patch.dict(os.environ, clean, clear=True):
            cfg = AdaptiveSamplingConfig.from_env()
        assert cfg.trigger == "no_positive"
        assert cfg.pos_eps == 0.0

    def test_trigger_is_validated(self):
        with pytest.raises(ValueError, match="ADAPTIVE_N_TRIGGER"):
            AdaptiveSamplingConfig(enable=True, trigger="sometimes").validate()

    def test_has_positive_semantics(self):
        assert has_positive(np.array([0.0, 0.0, 0.3])) is True
        assert has_positive(np.zeros(16)) is False
        assert has_positive(np.array([-1.0, -0.5, 0.0])) is False  # penalties are not positives
        assert has_positive(np.array([]), 0.0) is False
        assert has_positive(np.array([0.05]), pos_eps=0.1) is False  # eps is strict

    def test_penalty_variance_group_still_triggers_deepening(self):
        """THE v1 blind spot: 16 raw-zero samples whose PENALIZED scores have
        spread (different truncation depths). v1's zero-variance trigger goes
        blind; v2 must still deepen -- there is no correct exemplar."""
        raw = np.zeros(16)
        penalized = raw + np.linspace(-1.0, 0.0, 16)  # penalty-only spread
        assert flat_group_mask(penalized, 1e-6) is False  # v1 would NOT fire
        assert group_needs_deepening(raw, penalized, _cfg_v2()) is True  # v2 fires
        # and under the explicit v1 mode, the old behaviour is preserved
        assert group_needs_deepening(raw, penalized, _cfg(trigger="zero_variance")) is False

    def test_group_with_one_positive_does_not_deepen(self):
        raw = np.zeros(16)
        raw[7] = 0.4
        penalized = raw.copy()
        penalized[3] = -0.5  # a truncated sample
        assert group_needs_deepening(raw, penalized, _cfg_v2()) is False

    def test_all_zero_no_penalty_group_triggers_both_modes(self):
        """zero-variance groups are a strict subset of no-positive groups."""
        raw = np.zeros(16)
        assert group_needs_deepening(raw, raw, _cfg_v2()) is True
        assert group_needs_deepening(raw, raw, _cfg(trigger="zero_variance")) is True


class TestV2PositiveRetentionGuarantee:
    """Spec item 3: stratified retention MUST keep every positive sample."""

    def test_one_in_128_positive_always_survives_the_fold(self):
        raw = np.zeros(128)
        raw[99] = 0.75
        penalized = raw - 0.2  # some uniform penalty shift; protect uses raw
        for seed in range(200):
            plan = plan_retention(
                penalized, n=16, cfg=_cfg_v2(seed=seed), rng=np.random.default_rng(seed),
                protect=raw > 0.0,
            )
            assert 99 in set(plan.keep_idx.tolist()), f"lost the positive at seed {seed}"

    def test_every_positive_survives_when_several_exist(self):
        raw = np.zeros(64)
        pos_idx = [3, 17, 40]
        for i in pos_idx:
            raw[i] = 0.5
        for seed in range(100):
            plan = plan_retention(
                raw, n=16, cfg=_cfg_v2(seed=seed), rng=np.random.default_rng(seed), protect=raw > 0.0
            )
            kept = set(plan.keep_idx.tolist())
            assert set(pos_idx).issubset(kept), f"seed {seed} lost a positive"

    def test_protected_weights_still_sum_to_n(self):
        raw = np.zeros(128)
        raw[[5, 60]] = 1.0
        plan = plan_retention(raw, n=16, cfg=_cfg_v2(), rng=np.random.default_rng(0), protect=raw > 0.0)
        assert plan.weights.sum() == pytest.approx(16.0)
        # protected rows carry p=1 -> w = n/K exactly
        w = dict(zip(plan.keep_idx.tolist(), plan.weights.tolist(), strict=True))
        assert w[5] == pytest.approx(16 / 128)
        assert w[60] == pytest.approx(16 / 128)

    def test_protection_applies_even_with_stratify_off(self):
        """Protection is a correctness requirement, not a variance optimization."""
        raw = np.zeros(128)
        raw[42] = 0.9
        for seed in range(100):
            plan = plan_retention(
                raw, n=16, cfg=_cfg_v2(stratify=False, seed=seed),
                rng=np.random.default_rng(seed), protect=raw > 0.0,
            )
            assert 42 in set(plan.keep_idx.tolist())

    def test_more_positives_than_n_keeps_n_and_normalizes(self):
        raw = np.concatenate([np.ones(20), np.zeros(108)])
        plan = plan_retention(raw, n=16, cfg=_cfg_v2(), rng=np.random.default_rng(1), protect=raw > 0.0)
        assert len(plan.keep_idx) == 16
        assert all(raw[i] == 1.0 for i in plan.keep_idx)  # all kept rows are positives
        assert plan.weights.sum() == pytest.approx(16.0)


class TestV2DropSemantics:
    """Spec item 4: a group with no positive at budget is dropped -- masked out
    of BOTH the advantage and the token-mean loss denominator."""

    def _batch_two_groups(self, n=4, seqlen=6):
        """Group 0 rows 0..3 (will be dropped), group 1 rows 4..7 (kept)."""
        from verl.protocol import DataProto

        rows = 2 * n
        response_mask = torch.ones(rows, seqlen)
        return DataProto.from_dict(
            tensors={"response_mask": response_mask},
            non_tensors={
                "adaptive_weight": np.ones(rows),
                "adaptive_enlarged": np.zeros(rows),
                "adaptive_group_mean": np.zeros(rows),
                "adaptive_group_std": np.ones(rows),
            },
        )

    def test_dropped_group_tokens_leave_the_denominator(self):
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        n, seqlen = 4, 6
        out = self._batch_two_groups(n=n, seqlen=seqlen)
        prompt_index = np.array([0] * n + [1] * n)
        metrics = {}
        RayPPOTrainer._mask_dropped_groups(None, out, prompt_index, {0}, metrics)

        mask = out.batch["response_mask"]
        assert mask[:n].sum().item() == 0.0  # dropped group fully masked
        assert mask[n:].sum().item() == n * seqlen  # kept group untouched

        # token-mean over the batch == token-mean over the KEPT group only:
        # the dropped group does not dilute the denominator.
        loss_mat = torch.ones(2 * n, seqlen) * 3.0
        full = agg_loss(loss_mat=loss_mat, loss_mask=mask, loss_agg_mode="token-mean")
        kept_only = agg_loss(loss_mat=loss_mat[n:], loss_mask=mask[n:], loss_agg_mode="token-mean")
        assert full.item() == pytest.approx(kept_only.item())
        assert full.item() == pytest.approx(3.0)  # NOT 1.5, which dilution would give

        assert metrics["adaptive/dropped_rows"] == float(n)
        assert metrics["adaptive/dropped_token_frac"] == pytest.approx(0.5)
        np.testing.assert_array_equal(
            out.non_tensor_batch["adaptive_dropped"], np.array([1.0] * n + [0.0] * n)
        )

    def test_all_dropped_masks_everything_and_loss_is_exact_zero(self):
        """Every group dropped -> ALL rows masked (no sentinel). The agg_loss
        denominator floor turns the fully-masked batch into loss exactly 0 for
        pg, KL and entropy alike -- a true no-op, unlike the old sentinel which
        kept its tokens in the KL term (review m1)."""
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        n, seqlen = 4, 6
        out = self._batch_two_groups(n=n, seqlen=seqlen)
        prompt_index = np.array([0] * n + [1] * n)
        metrics = {}
        RayPPOTrainer._mask_dropped_groups(None, out, prompt_index, {0, 1}, metrics)

        mask = out.batch["response_mask"]
        assert mask.sum().item() == 0  # nothing survives; no sentinel exemption

        # token-mean over the fully-masked batch is exactly 0, not NaN
        loss_mat = torch.rand(2 * n, seqlen)
        loss = agg_loss(loss_mat=loss_mat, loss_mask=mask.float(), loss_agg_mode="token-mean")
        assert torch.isfinite(loss)
        assert loss.item() == 0.0

    def test_mandated_end_to_end_case(self):
        """Coordinator-mandated: a group of 16 raw-zero samples with nonzero
        penalties (a) still triggers deepening under v2 and (b) at budget
        exhaustion contributes ZERO tokens to the loss denominator."""
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        raw = np.zeros(16)
        penalized = np.linspace(-1.0, 0.0, 16)  # nonzero penalties, raw all zero

        # (a) triggers deepening despite penalty-made variance
        assert group_needs_deepening(raw, penalized, _cfg_v2()) is True

        # (b) budget exhausted with no positive -> dropped -> zero denominator share
        from verl.protocol import DataProto

        seqlen = 5
        live_rows = 16
        response_mask = torch.ones(32, seqlen)
        out = DataProto.from_dict(
            tensors={"response_mask": response_mask},
            non_tensors={
                "adaptive_weight": np.ones(32),
                "adaptive_enlarged": np.zeros(32),
                "adaptive_group_mean": np.zeros(32),
                "adaptive_group_std": np.ones(32),
            },
        )
        prompt_index = np.array([0] * 16 + [1] * 16)
        metrics = {}
        RayPPOTrainer._mask_dropped_groups(None, out, prompt_index, {0}, metrics)
        mask = out.batch["response_mask"]
        assert mask[:16].sum().item() == 0.0
        denom = mask.sum().item()
        assert denom == live_rows * seqlen  # denominator counts ONLY the live group
        loss = agg_loss(torch.ones(32, seqlen), mask, "token-mean")
        assert loss.item() == pytest.approx(1.0)  # undiluted


class TestV21AdvantageClamp:
    """v2.1(A): a row with non-positive RAW reward never gets positive advantage."""

    def _trainer_shim(self, pos_eps=0.0):
        import types

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        shim = types.SimpleNamespace()
        shim.adaptive_cfg = _cfg_v2(pos_eps=pos_eps)
        shim.config = types.SimpleNamespace(data=types.SimpleNamespace(max_response_length=6))
        shim._raw_scores_np = types.MethodType(RayPPOTrainer._raw_scores_np, shim)
        shim._clamp_nonpositive_raw_advantage = types.MethodType(
            RayPPOTrainer._clamp_nonpositive_raw_advantage, shim
        )
        return shim

    def _batch(self, raw, penalized, adv_rowvals, seqlen=6):
        """Rows carry token_level_scores (penalized, one-hot at last token),
        reward_pre_overlong (raw column) and a constant advantage per row."""
        from verl.protocol import DataProto

        rows = len(raw)
        scores = torch.zeros(rows, seqlen)
        scores[:, -1] = torch.tensor(penalized, dtype=torch.float32)
        adv = torch.tensor(adv_rowvals, dtype=torch.float32).unsqueeze(-1).expand(rows, seqlen).clone()
        return DataProto.from_dict(
            tensors={
                "token_level_scores": scores,
                "advantages": adv,
                "returns": adv.clone(),
            },
            non_tensors={"reward_pre_overlong": np.asarray(raw, dtype=np.float64)},
        )

    def test_mandated_case_penalty_dragged_mean(self):
        """THE mandated test: deepened group {1 positive, many truncated}, group
        mean < 0 -> raw-0 rows sit above the mean with positive advantage; the
        clamp must force every raw-0 row to adv <= 0 while leaving the positive
        row alone."""
        # 8 rows: 1 positive (raw 0.9), 5 truncated (raw 0, penalty -1 -> pen -1.0),
        # 2 complete-but-wrong (raw 0, no penalty -> pen 0.0)
        raw = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pen = [0.9, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0]
        mean = float(np.mean(pen))
        assert mean < 0  # the hole exists: raw-0-complete rows clear this mean
        std = float(np.std(pen, ddof=1))
        adv = [(x - mean) / (std + 1e-6) for x in pen]
        assert adv[6] > 0 and adv[7] > 0  # wrong-but-complete UPTRAINED without the clamp

        shim = self._trainer_shim()
        batch = self._batch(raw, pen, adv)
        out, m = shim._clamp_nonpositive_raw_advantage(batch)
        a = out.batch["advantages"]
        # every raw-0 row is now <= 0 ...
        for i in range(1, 8):
            assert torch.all(a[i] <= 0), f"row {i} still positive: {a[i]}"
        # ... the positive row is untouched ...
        assert torch.allclose(a[0], torch.full_like(a[0], adv[0]))
        # ... negative advantages (truncated rows) keep their true value
        assert torch.allclose(a[1], torch.full_like(a[1], adv[1]))
        # returns clamped in lockstep
        assert torch.all(out.batch["returns"][6] <= 0)
        assert m["adaptive/clamped_rows"] == 2.0

    def test_clamp_keeps_negative_gradients(self):
        """Clamp is min(adv, 0), not zeroing: 'do less of this' survives."""
        raw = [0.0, 0.0]
        pen = [-0.5, -1.0]
        adv = [0.7, -0.9]
        shim = self._trainer_shim()
        out, _ = shim._clamp_nonpositive_raw_advantage(self._batch(raw, pen, adv))
        assert torch.all(out.batch["advantages"][0] == 0.0)  # positive part forfeited
        assert torch.allclose(out.batch["advantages"][1], torch.full_like(out.batch["advantages"][1], -0.9))  # untouched

    def test_natural_group_is_also_covered(self):
        """'in every group, deepened or natural': no adaptive columns needed."""
        raw = [0.0, 1.0]
        pen = [0.0, 1.0]
        adv = [0.3, 1.2]  # raw-0 row somehow positive
        shim = self._trainer_shim()
        out, m = shim._clamp_nonpositive_raw_advantage(self._batch(raw, pen, adv))
        assert torch.all(out.batch["advantages"][0] == 0.0)
        assert torch.allclose(out.batch["advantages"][1], torch.full_like(out.batch["advantages"][1], 1.2))
        assert m["adaptive/clamped_rows"] == 1.0

    def test_no_positive_rows_touched_when_all_positive(self):
        raw = [0.5, 0.8]
        pen = [0.5, 0.8]
        adv = [-1.0, 1.0]
        shim = self._trainer_shim()
        out, m = shim._clamp_nonpositive_raw_advantage(self._batch(raw, pen, adv))
        assert torch.allclose(out.batch["advantages"][0], torch.full_like(out.batch["advantages"][0], -1.0))
        assert torch.allclose(out.batch["advantages"][1], torch.full_like(out.batch["advantages"][1], 1.0))
        assert m["adaptive/clamped_rows"] == 0.0

    def test_raw_reconstruction_without_the_column(self):
        """FS_OVERLONG_LOG=0 path: raw is reconstructed from the deterministic
        penalty, so the clamp still knows raw == 0 for a truncated row."""
        import types

        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        M, B, seqlen, plen = 6, 4, 6, 2
        env = {"FS_OVERLONG_PENALTY": "1", "FS_OVERLONG_BUFFER_LEN": str(B), "FS_OVERLONG_MAX_RESP_LEN": str(M)}
        with mock.patch.dict(os.environ, env):
            shim = types.SimpleNamespace()
            shim.adaptive_cfg = _cfg_v2()
            shim.config = types.SimpleNamespace(data=types.SimpleNamespace(max_response_length=M))
            shim._raw_scores_np = types.MethodType(RayPPOTrainer._raw_scores_np, shim)

            # one row, fully truncated (resp len 6 == M): penalty = -1, penalized = -1, raw = 0
            prompts = torch.zeros(1, plen, dtype=torch.long)
            attn = torch.ones(1, plen + seqlen, dtype=torch.long)
            scores = torch.zeros(1, seqlen)
            scores[0, -1] = -1.0
            batch = DataProto.from_dict(
                tensors={"prompts": prompts, "attention_mask": attn, "rm_scores": scores},
            )
            raw = shim._raw_scores_np(batch, scores.sum(dim=-1).numpy().astype(np.float64))
            assert raw[0] == pytest.approx(0.0)


class TestV21SaturatedTies:
    """v2.1(D)(iii): all-positive tied groups are masked out of the denominator
    exactly like dropped groups -- DAPO filters both ends."""

    def test_saturated_group_masked_like_dropped(self):
        from verl.protocol import DataProto
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        n, seqlen = 4, 5
        out = DataProto.from_dict(
            tensors={"response_mask": torch.ones(2 * n, seqlen)},
            non_tensors={
                "adaptive_weight": np.ones(2 * n),
                "adaptive_enlarged": np.zeros(2 * n),
                "adaptive_group_mean": np.zeros(2 * n),
                "adaptive_group_std": np.ones(2 * n),
            },
        )
        prompt_index = np.array([0] * n + [1] * n)
        metrics = {}
        # group 0 saturated (all-positive tie) -> passed in the same mask set
        RayPPOTrainer._mask_dropped_groups(None, out, prompt_index, {0}, metrics)
        mask = out.batch["response_mask"]
        assert mask[:n].sum().item() == 0.0
        loss = agg_loss(torch.full((2 * n, seqlen), 2.0), mask, "token-mean")
        assert loss.item() == pytest.approx(2.0)  # undiluted by the saturated group

    def test_saturation_judged_on_penalized_scores(self):
        """If the penalty differentiates equally-correct answers by length, the
        group has a real 'prefer shorter' gradient and must NOT be masked. This
        is the detection predicate used in _adaptive_deepen."""
        raw = np.array([0.8, 0.8, 0.8, 0.8])
        pen_tied = raw.copy()
        pen_spread = np.array([0.8, 0.6, 0.8, 0.7])  # penalty separates them
        cfg = _cfg_v2()
        is_saturated_tied = bool(np.all(raw > cfg.pos_eps)) and flat_group_mask(pen_tied, cfg.eps)
        is_saturated_spread = bool(np.all(raw > cfg.pos_eps)) and flat_group_mask(pen_spread, cfg.eps)
        assert is_saturated_tied is True
        assert is_saturated_spread is False


class TestV21Requeue:
    """v2.1(B): bounded requeue -- retry once, then drop for good."""

    def _shim(self, batch_len=8, mini=4, **cfg_kw):
        import types

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        shim = types.SimpleNamespace()
        shim.adaptive_cfg = _cfg_v2(**cfg_kw)
        shim.global_steps = 10
        shim.config = types.SimpleNamespace(
            actor_rollout_ref=types.SimpleNamespace(
                actor=types.SimpleNamespace(ppo_mini_batch_size=mini),
                rollout=types.SimpleNamespace(n=4),
            ),
            data=types.SimpleNamespace(max_response_length=8),
        )
        shim._requeue_buffer = []
        shim._requeue_retry_uids = set()
        shim._requeue_second_drops = 0
        shim._requeue_retried_total = 0
        shim._requeue_snapshot = None
        shim._inject_requeued_prompts = types.MethodType(RayPPOTrainer._inject_requeued_prompts, shim)
        return shim

    def _row(self, tag):
        from verl.protocol import DataProto

        return DataProto.from_dict(
            tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)},
            non_tensors={"uid": np.array([tag], dtype=object)},
        )

    def _batch_of(self, k):
        from verl.protocol import DataProto

        return DataProto.from_dict(
            tensors={"input_ids": torch.zeros(k, 4, dtype=torch.long)},
            non_tensors={"uid": np.array([f"fresh{i}" for i in range(k)], dtype=object)},
        )

    def test_injection_respects_mini_batch_divisibility(self):
        shim = self._shim(batch_len=8, mini=4, requeue_max_frac=1.0)
        for i in range(6):
            shim._requeue_buffer.append({"row": self._row(f"q{i}"), "step": 1})
        batch = self._batch_of(8)
        out, num = shim._inject_requeued_prompts(batch)
        # 6 eligible, cap 8, rounded DOWN to a multiple of mini=4 -> inject 4
        assert num == 4
        assert len(out) == 12
        assert len(shim._requeue_buffer) == 2  # the rest wait for the next slot

    def test_wait_period_respected(self):
        shim = self._shim(requeue_max_frac=1.0)
        shim.adaptive_cfg = _cfg_v2(requeue_after_steps=200, requeue_max_frac=1.0)
        shim._requeue_buffer.append({"row": self._row("q0"), "step": 9})  # 1 step old
        out, num = shim._inject_requeued_prompts(self._batch_of(8))
        assert num == 0 and len(out) == 8

    def test_injection_below_mini_defers(self):
        shim = self._shim(mini=4, requeue_max_frac=1.0)
        for i in range(3):  # fewer than one mini-batch worth
            shim._requeue_buffer.append({"row": self._row(f"q{i}"), "step": 1})
        out, num = shim._inject_requeued_prompts(self._batch_of(8))
        assert num == 0 and len(out) == 8
        assert len(shim._requeue_buffer) == 3  # nothing lost, just deferred

    def test_disabled_requeue_is_identity(self):
        shim = self._shim()
        shim.adaptive_cfg = _cfg_v2(requeue_enable=False)
        shim._requeue_buffer.append({"row": self._row("q0"), "step": 1})
        out, num = shim._inject_requeued_prompts(self._batch_of(8))
        assert num == 0 and len(out) == 8

    def test_zero_variance_mode_never_requeues(self):
        shim = self._shim()
        shim.adaptive_cfg = _cfg(trigger="zero_variance")  # v1 semantics
        shim._requeue_buffer.append({"row": self._row("q0"), "step": 1})
        out, num = shim._inject_requeued_prompts(self._batch_of(8))
        assert num == 0

    def test_uid_stripped_rows_concat_into_uidless_batch(self):
        """The injection-time schema: the fresh batch has NO uid yet (assigned
        after injection), so stashed rows must not carry one. protocol.py:208
        asserts keys against the FIRST piece; a stale uid would crash, a missing
        key would silently mis-align columns."""
        from verl.protocol import DataProto

        def uidless_row(tag):
            return DataProto.from_dict(
                tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)},
                non_tensors={"raw_prompt": np.array([tag], dtype=object)},
            )

        def fresh_batch(k):
            return DataProto.from_dict(
                tensors={"input_ids": torch.zeros(k, 4, dtype=torch.long)},
                non_tensors={"raw_prompt": np.array([f"f{i}" for i in range(k)], dtype=object)},
            )

        shim = self._shim(mini=4, requeue_max_frac=1.0)
        for i in range(4):
            shim._requeue_buffer.append({"row": uidless_row(f"q{i}"), "step": 1})
        out, num = shim._inject_requeued_prompts(fresh_batch(8))
        assert num == 4
        assert len(out) == 12
        assert "uid" not in out.non_tensor_batch
        # every column has the concatenated length -- no silent mis-alignment
        for key, col in out.non_tensor_batch.items():
            assert len(col) == 12, key

    def test_schema_mismatch_fails_loudly_not_silently(self):
        from verl.protocol import DataProto

        stale = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)},
            non_tensors={
                "raw_prompt": np.array(["q"], dtype=object),
                "uid": np.array(["stale-uid"], dtype=object),  # stash forgot to strip
            },
        )
        fresh = DataProto.from_dict(
            tensors={"input_ids": torch.zeros(8, 4, dtype=torch.long)},
            non_tensors={"raw_prompt": np.array([f"f{i}" for i in range(8)], dtype=object)},
        )
        shim = self._shim(mini=1, requeue_max_frac=1.0)
        for i in range(1):
            shim._requeue_buffer.append({"row": stale, "step": 1})
        with pytest.raises(RuntimeError, match="schema mismatch"):
            shim._inject_requeued_prompts(fresh)


class TestPostprocessIntegration:
    """Exercise the REAL AgentLoopWorker._postprocess, not just the pure penalty
    function -- this is the code path both agents actually take."""

    MAX_RESP = 4096
    BUFFER = 2048
    PROMPT_LEN = 8

    def _make_item(self, valid_response_len, score):
        from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, _InternalAgentLoopOutput

        total = self.PROMPT_LEN + self.MAX_RESP
        attention_mask = torch.zeros(1, total, dtype=torch.long)
        attention_mask[0, : self.PROMPT_LEN] = 1
        attention_mask[0, self.PROMPT_LEN : self.PROMPT_LEN + valid_response_len] = 1
        response_mask = torch.zeros(1, self.MAX_RESP, dtype=torch.long)
        response_mask[0, :valid_response_len] = 1
        return _InternalAgentLoopOutput(
            prompt_ids=torch.zeros(1, self.PROMPT_LEN, dtype=torch.long),
            response_ids=torch.zeros(1, self.MAX_RESP, dtype=torch.long),
            input_ids=torch.zeros(1, total, dtype=torch.long),
            position_ids=torch.zeros(1, total, dtype=torch.long),
            response_mask=response_mask,
            attention_mask=attention_mask,
            reward_score=score,
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=0.1, tool_calls=0.0, num_preempted=0),
            extra_fields={},
        )

    def _worker(self, enable):
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

        class _Stub:
            _postprocess = AgentLoopWorker._postprocess

            def __init__(self, cfg):
                self.reward_loop_worker_handles = ["stub"]  # non-None -> streaming path
                self.overlong_cfg = cfg

        return _Stub(
            OverlongPenaltyConfig(
                enable=enable,
                buffer_len=self.BUFFER,
                penalty_factor=1.0,
                max_resp_len=self.MAX_RESP,
                log=True,
            )
        )

    def test_penalty_reaches_rm_scores(self):
        cases = [(500, 0.8), (self.MAX_RESP - self.BUFFER, 0.8), (self.MAX_RESP - self.BUFFER // 2, 0.8),
                 (self.MAX_RESP, 0.0), (self.MAX_RESP, 1.0)]
        out = self._worker(True)._postprocess([self._make_item(length, s) for length, s in cases])
        finals = out.batch["rm_scores"].sum(dim=-1).numpy()
        expected = []
        for length, raw in cases:
            exceed = length - (self.MAX_RESP - self.BUFFER)
            expected.append(raw + min(-exceed / self.BUFFER, 0.0))
        np.testing.assert_allclose(finals, expected, atol=1e-5)

    def test_penalty_flows_into_reward_extra_keys(self):
        out = self._worker(True)._postprocess([self._make_item(self.MAX_RESP, 0.0)])
        keys = set(out.meta_info.get("reward_extra_keys", []))
        assert {"overlong", "overlong_reward", "reward_pre_overlong"}.issubset(keys)
        for k in keys:
            assert k in out.non_tensor_batch

    def test_disabled_postprocess_passes_scores_through_untouched(self):
        out = self._worker(False)._postprocess(
            [self._make_item(self.MAX_RESP, 0.7), self._make_item(100, 0.7)]
        )
        np.testing.assert_allclose(out.batch["rm_scores"].sum(dim=-1).numpy(), [0.7, 0.7], atol=1e-6)
        assert "overlong_reward" not in out.non_tensor_batch
        assert not out.meta_info.get("reward_extra_keys")

    def test_group_pinned_at_the_cap_is_still_detected_as_dead(self):
        """Identical penalties cancel out -> deepening still applies. Documented
        interaction between the two features."""
        out = self._worker(True)._postprocess([self._make_item(self.MAX_RESP, 0.0) for _ in range(4)])
        scores = out.batch["rm_scores"].sum(dim=-1).numpy()
        assert flat_group_mask(scores, 1e-6) is True

    def test_differently_truncated_group_becomes_alive_and_says_be_shorter(self):
        lengths = [self.MAX_RESP, self.MAX_RESP - 200, self.MAX_RESP - 700, self.MAX_RESP - 1500]
        out = self._worker(True)._postprocess([self._make_item(x, 0.0) for x in lengths])
        scores = out.batch["rm_scores"].sum(dim=-1).numpy()
        assert flat_group_mask(scores, 1e-6) is False
        # the shortest response carries the highest reward -> gradient says "be shorter"
        assert int(np.argmax(scores)) == 3


class TestFoldBackShapeContract:
    def test_every_group_folds_back_to_exactly_n(self):
        """The whole point of keep='subsample': ppo_mini_batch_size slicing, the
        token-mean denominator and seq-balancing must all see B*n rows."""
        n, B = 16, 8
        rng = np.random.default_rng(0)
        total = 0
        for p in range(B):
            K = [16, 32, 64, 128][p % 4]
            scores = np.zeros(K)
            if p % 2 == 0:
                scores[rng.integers(K)] = 1.0
            plan = plan_retention(scores, n=n, cfg=_cfg(), rng=rng)
            assert len(plan.keep_idx) == min(K, n) if K < n else len(plan.keep_idx) == n
            total += len(plan.keep_idx)
        assert total == B * n

    def test_weights_are_one_when_no_group_was_deepened(self):
        """Nothing fired -> the update is arithmetically identical to today."""
        n = 16
        for p in range(4):
            scores = np.zeros(n)
            scores[0] = float(p)
            plan = plan_retention(scores, n=n, cfg=_cfg())
            np.testing.assert_allclose(plan.weights, np.ones(n))
            assert plan.enlarged is False
