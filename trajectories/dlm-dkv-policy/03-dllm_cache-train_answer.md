d2Cache did exactly what I most worried it would, and the numbers say it sharply. Reuse jumped as predicted — $0.8655 / 0.8976 / 0.8102$, the highest on the board — confirming the per-token, per-step selection finds the inactive bulk and narrows the active query rows hard. Throughput barely moved ($20.97 / 19.84 / 22.66$ tokens/s, and on ARC it actually *fell* from the floor's $35.7$, because eager attention plus the per-step $D\cdot s$ and rollout overhead cost more than the short 64-token generation saves). But the real verdict is the risk I flagged: MATH quality collapsed from the floor's $38.4$ to **25.0**, far under the $35$ gate, while HumanEval held at $40.2$ and ARC at $84.2$. That is the worst possible place for the bleed, because the gate penalty multiplies the whole MATH workload toward zero and the geometric mean drags the entire policy below a blander one. Top reuse with a broken gate loses to moderate reuse with all gates intact.

Why MATH broke while the others survived is the design lesson. MATH answers are long, strictly sequential, and unforgiving — one wrong committed digit ruins the exact-match answer, and commits are frozen for the rest of the rollout. d2Cache's $D\cdot s$ selector reuses the cached KV of any token it underrates, and density is a *structural* proxy for imminence, not a measurement of whether the token's representation actually moved. On a long sequential answer, a token whose neighborhood looks sparse can still have a feature that shifted because of a *distant* commit that bidirectional attention propagated, and the density selector happily leaves it on stale cache; that stale feature flips a digit, the commit freezes, MATH falls under the gate. HumanEval (shorter spans, execution-checked, more local) and ARC (a single letter, 64 tokens) tolerate the same staleness. So the failure is not "reuse is bad," it is "reuse driven by a structural imminence proxy bleeds quality on exactly the workload with the tightest gate." I propose *dLLM-Cache*, which keeps aggressive reuse but drives the refresh decision off a signal tied to whether the feature *actually changed*, and is far more conservative about query narrowing.

The first move is to back off the aggression that cost MATH: keep the **full top-level rollout**, forwarding the whole sequence each step so every position's prediction is computed from a complete, current input, and cache only the *internal features* I am entitled to skip. This is the structural difference from d2Cache — it narrows *which queries run*; I run all queries but reuse cached intermediate features (keys, values, attention output, feed-forward output) for the positions whose features did not move. Running all queries is more compute than d2Cache's active-row scheme, so I expect *less* raw reuse, but it protects the prediction quality the gate punishes. The redundancy I then exploit is the two-part structure read at the segment-plus-row level the diagnostic supports. The prompt is quasi-static: prompt tokens are never masked, never change, and their features drift only as the response fills in around them — a slow second-order effect — so adjacent-step feature similarity in the prompt sits near one across many steps. So I cache the prompt and refresh it on a *long* interval, `prompt_refresh_interval = 50`, recomputing prompt features once and reusing them for ~50 steps, paying the prompt forward $K/50$ times instead of $K$ — a large saving for negligible error. The response is not uniform: most response tokens are also nearly frozen step-to-step, but a small minority genuinely move. So I refresh the *whole* response cache on a *short* interval `gen_refresh_interval` to bound staleness, and between those full refreshes recompute only the small set of response tokens that actually changed. The short interval is workload-specific because responses churn at different rates — $7$ for MATH, $8$ for HumanEval, $3$ for ARC — shorter where the gate is tighter or the sequence shorter.

What should rescue MATH is *how* the between-refresh partial update decides which response tokens moved — and unlike d2Cache's structural density proxy, I want a signal tied to the feature itself. The trap is circular: to know a token's expensive attention/feed-forward output changed I would have to compute it, which is the thing I am skipping. So I need a *cheap* feature whose adjacent-step change predicts the expensive downstream change, and the Value projection is exactly that: it is cheap (a single matmul, no attention, no FFN) and it is *what the attention reads out* — a position's attention output is a weighted sum of Values, so if a token's own Value barely changed and its neighborhood barely changed, its attention output and the FFN on top cannot have changed much. So I compute the response Values cheaply, take the *cosine* similarity of each token's current Value to its cached Value (cosine, not L2, because I care about directional/semantic change, not magnitude), and recompute the `transfer_ratio = 0.25` fraction with the *lowest* similarity — the most-changed tokens — scattering their fresh keys, attention, and feed-forward outputs back into the cache and reusing the rest. This is the `lowest_value_feature_similarity` row selector with the `scatter_refresh` KV update. The crucial contrast: d2Cache's $D\cdot s$ asks "is this token structurally about to be decoded?", a proxy that misses distant-commit-induced drift, while the V-verify asks "did this token's value actually move?", which catches drift from any cause, including the distant commits that flipped MATH digits. Refresh is driven by measured change, not predicted imminence, and that is the quality safeguard.

In the task's hooks, the block schedule keeps the workload defaults (`block_length = 32`, the workload's `num_steps`), because here I am *not* narrowing queries and I keep standard semi-autoregressive block decoding — the interval cache rides on top of the normal block-wise rollout, it does not replace it. The query plan stays `full_sequence` every step, the deliberate conservatism. The cache-refresh plan turns the feature cache on (`use_feature_cache = True`), sets `prompt_refresh_interval = 50`, the per-workload `gen_refresh_interval`, `transfer_ratio = 0.25`, `row_selector = "lowest_value_feature_similarity"`, `kv_update = "scatter_refresh"`, no layer reset. No attention probes are needed (`need_attention_weights = False`): the selection is a cheap Value-similarity test, not an attention-rollout importance, so I avoid the eager-attention overhead that hurt d2Cache's throughput — a second reason this should be faster per step despite running all queries. Transfer stays low-confidence over the current block; `after_step` is identity because the harness maintains the feature cache internally. The one thing this task pins where the generic method tunes per dataset is the interval choice: the published method lists different $(K_p, K_r)$ for GSM8K vs HumanEval vs MMLU, but the task uses one predeclared policy, so I fix $K_p = 50$ and bake the per-workload $K_r$ into presets ($7/8/3$) read from `step_meta` rather than searching per benchmark.

I expect quality to recover across the board, MATH specifically climbing back over its $35$ gate — that is the entire reason to trade away d2Cache's query narrowing — back into the mid-30s, with HumanEval and ARC holding near the floor. Reuse should be *lower* than d2Cache's $0.81$–$0.90$ because I run all queries and only skip features, I would expect $0.49$–$0.66$, clearly above the floor's $0.0$ but visibly below d2Cache. Throughput should be similar to d2Cache or a touch better per step on the long workloads, and on ARC it should not crater the way d2Cache's did, since the short interval-3 cache still skips real work without the rollout cost. The net: I expect this to *outrank* d2Cache not by winning reuse — it loses reuse — but by keeping every gate intact, so no workload is multiplied down and the geometric mean stays whole. What I will watch is whether modest reuse and still-modest throughput leave a large gap to the ceiling: if all three gates pass but the efficiency terms are only middling, the diagnosis for the next step is that I have been refreshing on a *clock* and refreshing *every layer*, and the real headroom — especially the throughput term no policy has touched — is in refreshing only *when* the state moved and only the *layers* that moved.

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
