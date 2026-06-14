**Problem (from step 1).** The uncached floor reuses nothing (reuse 0 everywhere) and is the slowest of
the field, earning zero efficiency credit. The redundancy is real — most steps flip only a few masked
positions — but a segment-and-clock cache cannot capture it: one rule per segment both reuses moving KV
(lost quality) and refreshes settled KV (wasted compute), at the wrong granularity.

**Key idea.** Decide reuse per token, per step. A masked token's KV moves only in the rapid-change phase
right before it is decoded, so refresh masked tokens by a Gaussian certainty density times confidence
`D(i)·s^i` (top-`current_k`), which predicts imminence from the locality of decoding. Prompt and decoded
tokens have no confidence, so refresh them by attention-rollout importance (`W = normalize(E+I)`, `C ← W·C`,
column-sum, nucleus mass `rollout_p`). The recompute set is the union plus any freshly transferred token;
everyone else reuses cached K/V.

**Why it works.** The two stages match the two token populations with genuinely different dynamics — masked
tokens change a lot but briefly and locally (a density signal finds them), prompt/decoded tokens change
little but are heavily attended (an attention signal finds them) — and neither signal can find the other.
It is training-free: density comes from the mask layout, confidence and attention from the forward already
run.

**This task vs the generic method.** Run as **pure diffusion** — `block_length = gen_length`,
`num_steps = gen_length`, one token per step, no semi-AR blocks (the `D·s` score already gives a
quasi-left-to-right order that curbs premature EOS). Query narrows to `active_query_rows` after step 0;
`row_selector = "certainty_density_attention_rollout"`, `kv_update = "active_q_mask"`. Attention probes are
required (`need_attention_weights = True`, eager attention for the rollout matrix). `inflate_w = 0` — no gap
inflation, keeping the active set strictly minimal for maximum reuse (the generic default inflates).

**Hyperparameters.** `rollout_p = 0.1`, `current_k = 32`, `sigma = 10.0`, `inflate_w = 0`.

**What to watch.** Reuse_ratio should jump hard off 0.0 (highest on the board, well above 0.8 on long
workloads); throughput should rise only modestly above the floor (active queries still attend over the whole
sequence; eager attention and rollout add overhead). The risk is the MATH quality gate: aggressive active-row
reuse on a long, unforgiving answer can bleed a few points, and MATH has the least headroom (38.4 native,
gate 35); if it slides under, the gate penalty multiplies the math workload down and the geometric mean
suffers despite top reuse.

```python
class DLMRefreshPolicy:
    """d2Cache: active query rows plus attention-rollout top-up."""

    policy_name = "d2cache"

    def block_schedule(self, request_meta):
        wl = WORKLOAD_CONFIGS[request_meta["workload"]]
        return {
            "gen_length": wl["gen_length"],
            "block_length": wl["gen_length"],
            "num_steps": wl["gen_length"],
            "warmup_forward": False,
        }

    def query_plan(self, step_meta, mask_state, cache_state):
        return {
            "query_scope": "full_sequence" if step_meta["step_in_block"] == 0 else "active_query_rows",
            "query_positions": cache_state.get("active_q_mask"),
            "track_positions": [],
            "masked_window": (mask_state["block_start"], mask_state["block_end"]),
        }

    def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
        return {
            "use_feature_cache": False,
            "prompt_refresh_interval": 1,
            "gen_refresh_interval": 1,
            "transfer_ratio": 0.0,
            "row_selector": "certainty_density_attention_rollout",
            "kv_update": "active_q_mask",
            "layer_reset": None,
        }

    def attention_probe_plan(self, layer_meta, step_meta):
        return {
            "need_attention_weights": True,
            "rollout_p": 0.1,
            "current_k": 32,
            "gamma": None,
            "track_num": 0,
            "sigma": 10.0,
            "inflate_w": 0,
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
