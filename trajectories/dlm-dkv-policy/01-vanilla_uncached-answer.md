**Problem.** A diffusion LM generates by denoising with full bidirectional attention, recomputing the
entire sequence's keys, values, attention, and feed-forward outputs at every layer on every step. The
autoregressive prefix KV cache is mathematically inapplicable — there is no causal prefix whose states are
frozen and no fixed append order — so the exactly-correct rollout reuses nothing. The task pays for reuse
and throughput, so this exact rollout is the efficiency floor.

**Key idea (the uncached control).** Run the model's native rollout at full cost: a full bidirectional
forward over the whole sequence at every denoising step, no feature cache, refresh everything every step,
standard low-confidence token transfer over the current block. It is the only policy on the ladder that is
exactly the model's intended behavior; everything later trades a bounded slice of that exactness for
compute saved.

**Why it is the floor.** By construction `reuse_ratio = 0` on every workload, so the dominant efficiency
term (weight 0.75) is zero everywhere; throughput is the slowest of the field (most work per step), so the
throughput term is near zero too; the quality gate passes comfortably because the rollout is exact. Under
the geometric mean of near-zero per-workload efficiency, this is the lowest score on the ladder — and it
establishes the three references every later rung needs: the native-accuracy quality ceiling, the
throughput floor, and the reuse floor at 0.

**Step-1 edit.** Leave `DLMRefreshPolicy` at the scaffold default: `query_scope = "full_sequence"`,
`use_feature_cache = False`, prompt/generation refresh intervals 1, `kv_update = "full_refresh"`, no
attention probes, low-confidence transfer, identity `after_step`.

**What to watch.** Same shape on all three workloads — quality passed, reuse 0, throughput at the bottom;
longer generations (MATH, HumanEval) slowest, short ARC-Challenge fastest but still cache-free. The
diagnosis is already an efficiency problem the floor refuses to address, forcing a real feature cache at
step 2.

```python
class DLMRefreshPolicy:
    """No-cache control: full LLaDA denoising forward every step."""

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
