A diffusion language model like LLaDA-8B-Instruct does not generate left-to-right. It lays the response region as a block of `[MASK]` tokens after the prompt and, over many denoising steps, runs a full bidirectional forward over the whole sequence, predicts every masked position at once, commits the most confident ones, and re-masks the rest until nothing is masked. The cache policy is the whole point of this task, but it sits on top of that rollout, and before I can relax anything I need an unambiguous floor: the model's native behaviour at full cost, reusing nothing. The reason that floor matters is also precisely the reason caching is hard here, so let me make it exact rather than merely lazy.

I propose the *uncached control* — the only policy on the ladder that is exactly the model's intended behaviour: a full bidirectional forward over the entire current sequence at every single denoising step, recomputing every key, value, attention output, and feed-forward output at every layer, with no feature cache and no skipped positions. What makes this the *correct* reference, and not just the conservative one, comes straight out of the masked-diffusion construction. LLaDA's predictor is time-independent — a masked position's clean value depends only on the unmasked context around it, which is literally clean data, so the network needs no timestep input and is a plain bidirectional transformer with no causal mask — and its attention is bidirectional: every position attends to every other, masked ones included. The autoregressive prefix KV cache (Vaswani et al. 2017; Radford et al. 2019) is free precisely because causal attention makes token $i$ depend only on tokens $\le i$, so once $K_i, V_i$ are computed they never change and generation appends at the end. Both pillars fail here. Because attention is bidirectional, a token's key and value are a function of the *whole* sequence, so the moment any masked token is unmasked anywhere, the context — and hence the keys and values — of every other token shifts, including tokens committed earlier. And there is no append order: the transfer rule commits whichever masked positions are most confident, anywhere in the response, so I cannot even pre-decide whose states to refresh. The exact prefix cache is not inconvenient here; it is mathematically inapplicable.

There *is* a genuine redundancy that every later rung will exploit: across a whole rollout the network sees at most $L$ distinct conditioning states, because most steps flip only a handful of masked positions and leave the rest of the input identical, and the output on an unchanged input is identical. But that reuse is licensed *only* when a position's conditioning state has not changed, and in a bidirectional model the moment any mask reveals it changes the context for every other position — there is no causal prefix whose states are provably frozen. A stale cached value can therefore be silently wrong, and since committed tokens are frozen for the rest of the rollout, a wrong commit caused by a stale feature is unrecoverable. So the floor *declines* every shortcut deliberately, and in this task's policy vocabulary each decline is one dict value: the block schedule takes the workload's own `gen_length`, `block_length`, and `num_steps` unchanged and opens no warm forward (there is no cache to warm); `query_scope = "full_sequence"` forwards every position; `use_feature_cache = False` with both refresh intervals at 1, `row_selector = "none"`, `kv_update = "full_refresh"`, no layer reset; no attention probes (the floor needs no importance or drift signal because it reuses nothing); standard low-confidence transfer over the current block, forcing one; and an identity `after_step`, since there is no rollout state to carry when nothing is cached.

Why this is the *floor* on this task specifically falls out of how the score is built. The score is an efficiency score: each workload first applies the final task accuracy as a near-lossless soft quality gate (`math ≥ 35`, `humaneval ≥ 40`, `lm-eval ≥ 84`), then — once the gate passes — rewards cache reuse (the dominant term, weight $0.75$) and decode throughput (weight $0.25$, normalized against the visible baseline envelope), geometric-mean across the three workloads. By construction the control reuses nothing, so `reuse_ratio = 0` on every workload and the dominant term is zero everywhere; its throughput is the slowest of the field because it does the most work per step, so the throughput term is near zero too; and the quality gate passes comfortably, because it is the exact rollout and its accuracy is the model's native accuracy. Under a geometric mean of near-zero per-workload efficiency, that is the lowest score on the ladder. The control is the weakest policy not because it is wrong — it is the only exactly-correct one — but because the metric pays for reuse and throughput, and the exact rollout deliberately spends both. What it earns me is the set of references every later rung is measured against: the native-accuracy quality ceiling, the throughput floor, and the reuse floor at exactly $0$, which means the entire interval from $0$ to $1$ of reuse headroom is there to win. The diagnosis it hands forward is already pointed: this is not a quality problem, it is an efficiency problem the floor refuses to address, and the fix at step 2 is to stop recomputing what has not changed.

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
