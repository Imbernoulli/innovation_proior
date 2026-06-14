**Problem (from step 2).** d2Cache won the most reuse (0.81–0.90) but its active-row narrowing on a
pure-diffusion rollout bled MATH from 38.4 to 25.0 — under the 35 gate — and the gate penalty multiplied the
math workload down; throughput barely moved (eager-attention + rollout overhead). Top reuse with a broken
gate loses to moderate reuse with all gates intact.

**Key idea.** Keep the **full top-level rollout** (forward all queries every step, so every prediction is
computed from a complete input — the gate safeguard) and cache only internal features. Refresh the
quasi-static prompt on a long interval `K_p`; refresh the whole response on a short interval `K_r`; between
response refreshes, recompute only the `transfer_ratio` fraction of response tokens whose features actually
moved, judged by **cosine similarity of the cheap Value projection** to its cached Value (lowest similarity
= most changed), scattering their K/attention/FFN back and reusing the rest.

**Why it beats step 2 where it matters.** d2Cache's `D·s` is a structural imminence proxy that misses drift
from distant commits; the V-verify measures whether the feature *actually* changed, catching drift from any
cause — including the distant-commit drift that flipped MATH digits. Running all queries costs more compute
(less raw reuse than d2Cache) but protects the accuracy the gate punishes, so no workload is multiplied down.

**This task vs the generic method.** Query stays `full_sequence` every step (no active-row narrowing).
`use_feature_cache = True`, `prompt_refresh_interval = 50`, per-workload `gen_refresh_interval`
(math 7 / humaneval 8 / lm_eval 3), `transfer_ratio = 0.25`, `row_selector =
"lowest_value_feature_similarity"`, `kv_update = "scatter_refresh"`. No attention probes
(`need_attention_weights = False`) — cheap Value test, so no eager-attention overhead. The generic method
tunes `(K_p, K_r)` per dataset; the task pins one predeclared policy, so `K_p = 50` is fixed and the
per-workload `K_r` is baked into presets read from `step_meta`.

**Hyperparameters.** `prompt_refresh_interval = 50`; `gen_refresh_interval` = 7 (math) / 8 (humaneval) /
3 (lm_eval); `transfer_ratio = 0.25`.

**What to watch.** Quality should recover everywhere and MATH should climb back over 35 (the whole point of
dropping query narrowing). Reuse_ratio should be *lower* than d2Cache (run all queries, skip only features) —
expect ~0.49–0.66 — but throughput should not crater on ARC the way d2Cache's did. Net: outrank d2Cache by
keeping every gate intact, not by winning reuse. If all gates pass but efficiency stays middling, the
remaining headroom (especially throughput, untouched so far) is refreshing only *when* and only the *layers*
that moved.

```python
class DLMRefreshPolicy:
    """dLLM-Cache: interval feature reuse plus low-similarity row refresh."""

    policy_name = "dllm_cache"
    _PRESETS = {
        "math": (50, 7, 0.25),
        "humaneval": (50, 8, 0.25),
        "lm_eval": (50, 3, 0.25),
    }

    def _preset(self, request_meta):
        return self._PRESETS.get(request_meta["workload"], (50, 4, 0.25))

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
        request_meta = {"workload": step_meta.get("workload", ""), "step_budget": "final"}
        prompt_interval, gen_interval, transfer_ratio = self._preset(request_meta)
        return {
            "use_feature_cache": True,
            "prompt_refresh_interval": prompt_interval,
            "gen_refresh_interval": gen_interval,
            "transfer_ratio": transfer_ratio,
            "row_selector": "lowest_value_feature_similarity",
            "kv_update": "scatter_refresh",
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
