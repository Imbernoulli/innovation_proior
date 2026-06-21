dLLM-Cache validated the trade and then exposed the next ceiling, both in the numbers. The trade worked: MATH recovered from d2Cache's $25.0$ to $35.2$ — back over the $35$ gate, by a hair — HumanEval climbed to $46.3$, ARC held at $84.0$, so all three gates are intact and no workload is multiplied down, which is exactly why it outranks d2Cache. But reuse fell to $0.6328 / 0.6562 / 0.4922$ (well below d2Cache's $0.81$–$0.90$) because I ran *all* queries and only skipped internal features, and the throughput term no policy on this ladder has yet moved barely budged: $20.84 / 15.58 / 25.95$ tokens/s, on HumanEval actually slower than the floor. dLLM-Cache bought safety and moderate reuse but essentially no speed, and the geometric mean is middling because the throughput term everywhere, and reuse on ARC, leave most of the headroom on the table.

The speed never came because I have been refreshing on a **clock** and refreshing **every layer**, and running **all queries**. The prompt interval ($50$) and response interval ($7/8/3$) recompute on a fixed schedule whether or not anything changed, and when an interval fires it recomputes *all layers* of the refreshed region, paying full depth for layers that settled long ago; the V-verify is adaptive about which tokens but still runs every step and triggers a full-depth refresh; and forwarding all queries keeps the per-step forward essentially full-cost. To win throughput I need the opposite on two axes at once — refresh *when the state moved*, not on a clock, and refresh only the *layers* that moved, not the whole stack — and stop forwarding the distant masked tokens that contribute nothing. Each is a multiplicative cut to the per-step compute, which is what throughput rewards. I propose *Elastic-Cache*, which makes all three cuts at once.

Take the depth axis first, the largest lever. The step-to-step KV change is not uniform across depth: it is *larger in deeper layers than shallow ones*. That matches the old probing folklore — early layers lock onto local lexical structure and settle fast, deep layers carry global semantics that keep shifting — and the architecture forces it: in a residual block the adjacent-step hidden-state change satisfies a recursion $\bar\Delta^{\ell+1} \le \lambda_\ell\,\bar\Delta^{\ell}$ with $\lambda_\ell > 1$ for an expressive layer, so the *bound* on drift grows multiplicatively with depth, and combined with shallow decoded-token drift vanishing while deep drift stays bounded away from zero, there is a boundary layer $\ell^*$ separating settled shallow layers from active deep ones. So when I refresh I should not recompute layer 1 — its cache has converged — I should reuse the shallow layers and recompute only from $\ell^*$ onward. Skipping the shallow half of the stack on most refreshes is a near-$2\times$ cut to the refresh cost, and it is where the throughput hides; d2Cache and dLLM-Cache treat every layer the same.

I will not hand-pick $\ell^*$ — a hand-set constant is the clock problem in another guise, and $\ell^*$ should move (shallow early in decoding when much is changing, deep late when things have settled) — so $\ell^*$ has to fall out of the model's own signal at runtime, which forces the *when* axis. I need a cheap, reliable test that the cached state has drifted enough to refresh, and ideally the same test produces the boundary. Comparing hidden states $H^\ell$ to last step's is circular and noisy — while I reuse cache the $H$ I carry is itself built from stale KV, so the difference conflates genuine change with my own caching error, compounding with depth — so I need a signal *upstream* of the KV change and not polluted by my caching. The source of the KV change is the bidirectional attention itself: when a newly decoded token suddenly receives real attention, it rewrites the attention output earlier tokens computed when it was still masked, and that shifts their hidden state, hence their K and V. So the change in the **attention weights** is the *cause*, not the symptom, and it tracks the KV change closely. Measuring every token's attention change every step would defeat the purpose, so I want one cheap probe whose movement lower-bounds everyone else's: among already-decoded tokens, the *most-attended* one has the *smallest* drift, because for it to remain most-attended its lead over the runner-up must absorb its own attention swing, which bounds its excess drift by $O(\sqrt{d_k}/(R_\ell\sqrt{N}))$, vanishing for long sequences. So the most-attended token is a *conservative* trigger — if even the most stable token's attention pattern has broken, everything less stable has moved at least as much. I track it (`track_num = 1`, the conservative top-1 signal), and at each layer in reuse mode recompute just the tracked token's attention row and compare it to last step's by **cosine** similarity (scale-free, the pattern is what matters). If the similarity at some layer drops below a threshold $\gamma$, the pattern there broke: restore the cached full-sequence hidden at the *next* layer and full-recompute from there to the last layer; if no layer ever drops below $\gamma$, keep reusing the whole cache and pay almost nothing. The same per-layer test answers both axes — the first layer to break *is* $\ell^*$, adapting per step and per input, and the refresh runs from $\ell^*{+}1$ to $L$. The single knob is $\gamma = 0.9$: higher is stricter, lower laxer.

One more axis the dynamics hand me is the distant masked tokens. They are barely attended — a length-bias prior holding the sequence's shape, not informing the current prediction — so I should not forward them at all; dLLM-Cache forwarded every query including these, while a rigid block cache would freeze masked tokens right at a block edge the current prediction still leans on. The fix is a **sliding window** of the leftmost masked positions, `window_length = 16`, that moves through the response as tokens commit: the nearby masks that genuinely attend to each other stay live and current, only the truly distant ones get block-cached. So the set I forward each step is the window plus the tracked-token rows the trigger needs — a small fraction of the sequence, the third multiplicative cut and the one that finally moves throughput, because the per-step forward is now over ~16 positions plus a probe, not the whole response.

In the task's hooks, the block schedule keeps `block_length = 32` but adds `window_length = 16`. The query plan forwards `full_sequence` at step 0 to fill the cache, then `tracked_window` on every later step, handing the harness the `track_positions` and the masked window. The cache-refresh plan declares `row_selector = "tracked_tokens_and_masked_window"`, `kv_update = "tracked_window_layer_reset"`, and crucially `layer_reset = "attention_similarity"` — the depth-aware reset the harness drives off the per-layer cosine trigger, the mechanism no prior policy used. The attention-probe plan sets `need_attention_weights = True` (the trigger needs the tracked token's attention row, so eager attention again) and carries `gamma = 0.9`, `track_num = 1`. The token-transfer plan switches from low-confidence top-k to `mode = "confidence_threshold"` with `threshold = 0.9`, scope the masked window, `num_transfer_tokens = 1`, force-one: confidence-aware parallel decoding commits every windowed position whose top probability clears $0.9$ (and at least one), which pairs naturally with $\gamma$ — confident, stable predictions keep attention patterns stable, uncertain revisions make the trigger fire. `after_step` is identity; the harness maintains the per-layer cache, the tracked positions, and the per-step layer-reset boundary. Faithful to the method, I keep one predeclared $\gamma = 0.9$ and `window_length = 16` across all three workloads, no per-benchmark search.

So the delta from dLLM-Cache is three multiplicative compute cuts aimed exactly at the throughput term it left untouched: forward only a sliding window of live masks plus a probe instead of all queries; trigger refresh on a measured attention-drift signal instead of a clock; and on refresh recompute only the layers from the adaptively-found $\ell^*$ onward instead of the whole stack. The headline should be throughput — the first policy that cuts the per-step forward itself — so tokens/s should jump *multiplicatively*, well past dLLM-Cache's $\sim 20/16/26$, plausibly several-fold, largest on the short ARC workload where each forward is tiny relative to the per-step overhead. Reuse should rise back toward d2Cache's neighborhood on the long workloads (windowed forward plus shallow-layer reuse skips a lot), around $0.86$–$0.88$ on MATH and HumanEval, recovering the ground dLLM-Cache gave up, though on ARC with only 64 tokens and a 16-window it may land nearer $0.5$. The gates are the bar I must not break, having watched d2Cache fall under MATH's: the conservative trigger and the confidence-$0.9$ threshold exist precisely to keep accuracy native, so I expect MATH near $36$–$37$, HumanEval near $47$, ARC near $84$, all passing. If that holds, this policy wins the task not on any single term but by being the only one to move all three — reuse high on the long workloads, throughput multiplied everywhere, gates intact — so its per-workload efficiency scores are high and the geometric mean clears every prior rung.

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
