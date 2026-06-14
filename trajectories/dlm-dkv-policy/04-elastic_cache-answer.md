**Problem (from step 3).** dLLM-Cache kept every gate intact (MATH back to 35.2, over the gate) but bought
only moderate reuse (0.49–0.66) and essentially no throughput (~16–26 tokens/s, untouched) because it
refreshed on a **clock** and refreshed **every layer**, and ran **all queries** every step. The throughput
term, and reuse on the short workload, are leaving most of the headroom on the table.

**Key idea.** Refresh *when* the state moved and only the *layers* that moved, and stop forwarding distant
masks. (1) **Depth-aware reset:** KV drift grows with depth (residual recursion `Δ̄^{ℓ+1} ≤ λ_ℓ Δ̄^ℓ`,
`λ_ℓ>1`), so a boundary `ℓ*` separates settled shallow from active deep layers — reuse shallow, recompute
from `ℓ*` to `L`. (2) **Attention-drift trigger:** measure change on the *attention weights* (the cause of
KV change, not polluted by caching error) via the most-attended decoded token, whose excess drift is bounded
by `O(√d_k/(R_ℓ√N))` — a conservative probe; cosine-similarity below `γ` at a layer sets `ℓ*` to the next
layer, adaptively per step. (3) **Sliding window:** forward only a window of the leftmost live masks (plus
the tracked probe), block-caching distant length-bias masks.

**Why it beats step 3.** Three multiplicative cuts to the per-step forward — windowed queries, drift-triggered
(not clocked) refresh, layer-reset from `ℓ*` (not full depth) — finally move the throughput term, while the
conservative trigger and confidence-0.9 threshold keep accuracy native so the gates stay intact.

**This task vs the generic method.** Faithful to the method: `window_length = 16`, query `tracked_window`
after step 0, `row_selector = "tracked_tokens_and_masked_window"`, `kv_update = "tracked_window_layer_reset"`,
`layer_reset = "attention_similarity"`, `need_attention_weights = True` (eager attention for the trigger),
`gamma = 0.9`, `track_num = 1`; transfer `mode = "confidence_threshold"`, `threshold = 0.9`,
`num_transfer_tokens = 1`. One predeclared `γ = 0.9` and window 16 across all workloads — no per-benchmark
search.

**Hyperparameters.** `window_length = 16`, `gamma = 0.9`, `track_num = 1`, confidence `threshold = 0.9`.

**What to watch (the bar to clear).** Throughput should jump *multiplicatively* (first policy to cut the
per-step forward itself), largest on short ARC. Reuse should recover toward d2Cache's range on the long
workloads (~0.86–0.88), nearer ~0.5 on short ARC. Gates must hold (MATH ~36–37, HumanEval ~47, ARC ~84) — the
conservative trigger and threshold exist for exactly that. Net: the only policy to move all three terms at once,
clearing dLLM-Cache's profile (gates passed, reuse 0.49–0.66, throughput stuck) on every axis.

```python
class DLMRefreshPolicy:
    """Elastic-Cache: tracked-token windows with attention-similarity reset."""

    policy_name = "elastic_cache"

    def block_schedule(self, request_meta):
        wl = WORKLOAD_CONFIGS[request_meta["workload"]]
        return {
            "gen_length": wl["gen_length"],
            "block_length": wl["block_length"],
            "window_length": 16,
            "num_steps": wl["num_steps"],
            "warmup_forward": False,
        }

    def query_plan(self, step_meta, mask_state, cache_state):
        return {
            "query_scope": "full_sequence" if step_meta["step"] == 0 else "tracked_window",
            "query_positions": None,
            "track_positions": cache_state.get("track_positions", []),
            "masked_window": (mask_state["block_start"], mask_state["block_end"]),
        }

    def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
        return {
            "use_feature_cache": False,
            "prompt_refresh_interval": 1,
            "gen_refresh_interval": 1,
            "transfer_ratio": 0.0,
            "row_selector": "tracked_tokens_and_masked_window",
            "kv_update": "tracked_window_layer_reset",
            "layer_reset": "attention_similarity",
        }

    def attention_probe_plan(self, layer_meta, step_meta):
        return {
            "need_attention_weights": True,
            "rollout_p": 0.0,
            "current_k": 0,
            "gamma": 0.9,
            "track_num": 1,
        }

    def token_transfer_plan(self, logits, mask_state, step_meta):
        return {
            "mode": "confidence_threshold",
            "scope": "masked_window",
            "num_transfer_tokens": 1,
            "threshold": 0.9,
            "force_one": True,
        }

    def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
        return cache_state
```
