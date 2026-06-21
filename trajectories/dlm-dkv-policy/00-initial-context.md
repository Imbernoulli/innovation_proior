## Research question

A diffusion language model does not generate left-to-right. LLaDA-8B-Instruct begins a response as a block of `[MASK]` tokens and, over many denoising steps, runs a full bidirectional forward over the whole prompt-plus-response sequence, predicts every still-masked position at once, commits a few of the most confident ones, and re-masks the rest — until nothing is masked. Quality is competitive with an autoregressive model of the same size, but the per-token cost is not: an autoregressive model reuses its prefix KV cache and does one forward per token, while the diffusion model recomputes the entire sequence's keys, values, attention, and feed-forward outputs at every layer on every step. The object of design is the **cache policy** — a declarative plan, evaluated step by step against a fixed LLaDA rollout, that decides which token positions and which layers get recomputed and which reuse cached state, so that redundant denoising work is cut while the benchmark-native final-task score is preserved. Everything else about the host model, the loop, and the scoring is fixed.

## Prior art / Background / Baselines

- **Autoregressive transformer LMs with a prefix KV cache.** Causal attention makes each token depend only on earlier positions, so once a position's key and value are computed they never change; generation appends at the end and the prefix cache is exact and append-only. Gap: it relies on a fixed causal prefix, which a bidirectional denoiser does not have, so the cache is structurally inapplicable here.
- **Discrete / absorbing-state diffusion.** A forward corruption process masks tokens toward an all-`[MASK]` endpoint; the model learns to reverse it under a variational bound, and the masking case reduces the per-step objective to masked-token cross-entropy. Gap: the framework defines a sound generative process but says nothing about reusing computation across reverse steps — every step is a full forward.
- **LLaDA.** A bidirectional transformer trained with `1/t`-weighted masked cross-entropy and sampled by a reverse process from a fully masked response with low-confidence re-masking and semi-autoregressive block decoding. Gap: the published rollout is the uncached one — a full bidirectional forward over the whole sequence every step, reusing nothing, because committing one mask changes every other token's context.
- **Confidence-based parallel decoding of masked predictors.** Predict all masked positions, keep the most confident, re-mask the least confident, and repeat. Gap: it fixes which tokens commit, not which computation is reused; it is the transfer rule the rollout already uses, orthogonal to caching.

## Fixed substrate / Code framework

The harness owns everything except the cache policy. It loads `LLaDA-8B-Instruct` and the public benchmark, lays the response region as mask tokens after the prompt, and runs one fixed denoising rollout whose mechanics — the bidirectional forward, the per-layer KV/feature cache buffers, the attention-rollout accumulator, the certainty-density and drift probes, the layer-reset bookkeeping, and the token-transfer commit — are all implemented in fixed harness code below the editable region. The policy never calls a paper repository's generation function; it returns plain dicts of flags and parameters, and the harness executes the corresponding mechanism. Some mechanisms need extra forward arguments (active query rows, tracked-token positions, eager attention weights); the harness loads task-local compatibility model classes to expose those hooks, but the outer rollout stays policy-driven.

Three workloads are fixed (`WORKLOAD_CONFIGS`): MATH-500 (`gen_length=256, block_length=32, num_steps=256`), HumanEval (`512, 32, 512`), ARC-Challenge (`64, 32, 64`); the mask token id is `126336`. The rollout is deterministic at the fixed seed.

## Editable interface

Exactly one region is editable — the `DLMRefreshPolicy` class in `dLLM-cache/custom_dlm_eval.py` (the compatibility class name; semantically it is a DLM cache-plan policy). Every method on the ladder is a fill of this same six-hook contract, each hook returning a dict the harness consumes:

- `block_schedule(request_meta)` — generation length, block length, steps per block, and whether a block opens with a full warm forward.
- `query_plan(step_meta, mask_state, cache_state)` — which positions are forwarded/recomputed this step: `full_sequence`, `current_block`, `active_query_rows`, `tracked_window`, or a masked window.
- `cache_refresh_plan(layer_meta, step_meta, token_stats, cache_state)` — per-layer recompute vs reuse: whether the feature cache is on, the prompt/generation refresh intervals, the selected-row fraction, the row selector, the KV-overwrite mode, and any layer reset.
- `attention_probe_plan(layer_meta, step_meta)` — whether attention weights / similarity probes are needed, and probe parameters (`rollout_p`, `current_k`, `gamma`, `track_num`, `sigma`, `inflate_w`).
- `token_transfer_plan(logits, mask_state, step_meta)` — which masked tokens are committed back to the global denoising state (low-confidence top-k, or a confidence threshold).
- `after_step(step_meta, logits, attention_stats, transfer_state, cache_state)` — updates rollout state (active query masks, attention rollout, tracked tokens, density scores, layer-reset boundaries).

The starting point is the scaffold default: the uncached control — full-sequence forward, no feature cache, full refresh every step, standard low-confidence transfer. Each later method replaces exactly this class and nothing else.

```python
# EDITABLE region of dLLM-cache/custom_dlm_eval.py — default fill (uncached control)
class DLMRefreshPolicy:
    """Default shared-hook policy: uncached LLaDA denoising rollout.

    The participant-facing surface is a cache-plan interface over one fixed
    rollout, not a selector for paper-specific backend modules.
    """

    policy_name = "vanilla_uncached"

    def block_schedule(self, request_meta):
        wl = WORKLOAD_CONFIGS[request_meta["workload"]]
        return {
            "gen_length": wl["gen_length"],
            "block_length": wl["block_length"],
            "num_steps": wl["num_steps"],
            "warmup_forward": False,
        }

    def query_plan(self, step_meta, mask_state, cache_state):
        return {
            "query_scope": "full_sequence",
            "query_positions": None,
            "track_positions": [],
            "masked_window": None,
        }

    def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
        return {
            "use_feature_cache": False,
            "prompt_refresh_interval": 1,
            "gen_refresh_interval": 1,
            "transfer_ratio": 0.0,
            "row_selector": "none",
            "kv_update": "full_refresh",
            "layer_reset": None,
        }

    def attention_probe_plan(self, layer_meta, step_meta):
        return {
            "need_attention_weights": False,
            "rollout_p": 0.0,
            "current_k": 0,
            "gamma": None,
            "track_num": 0,
        }

    def token_transfer_plan(self, logits, mask_state, step_meta):
        return {
            "mode": "low_confidence",
            "scope": "current_block",
            "num_transfer_tokens": step_meta["default_num_transfer_tokens"],
            "threshold": None,
            "force_one": True,
        }

    def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
        return cache_state
```

## Evaluation settings

Three workloads: `math` (MATH-500 test split, exact final-answer accuracy), `humaneval` (OpenAI HumanEval, pass@1 execution accuracy), `lm-eval` (ARC-Challenge test split, exact answer-letter accuracy). One predeclared policy is used across all workloads (no per-benchmark hyperparameter search). One seed (42). Per workload the harness records `final_score` (0–100, higher better; the canonical quality metric), `reuse_ratio` (higher better; fraction of generated-token cache work reused), `refresh_ratio` (= `1 − reuse_ratio`, lower better), `tokens_per_s` (higher better, decode throughput), `peak_memory_mb` (lower better), `n_examples`, and `elapsed`. The task is an efficiency benchmark: each workload applies the final score as a near-lossless soft quality gate (`math ≥ 35`, `humaneval ≥ 40`, `lm-eval ≥ 84`), and once the gate is satisfied ranks cache reuse (weight 0.75) and decode throughput (weight 0.25, normalized against the visible baseline envelope); the task score is the geometric mean across the three workloads.
