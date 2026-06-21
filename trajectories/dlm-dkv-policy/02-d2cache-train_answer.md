The uncached floor told me what is missing in numbers: the quality gate is clear everywhere (MATH 38.4, HumanEval 43.9, ARC 84.0) but `reuse_ratio` is a flat $0.0$ with `refresh_ratio` $1.0$, and throughput is the slowest of the field ($19.5 / 11.5 / 35.7$ tokens/s) because the control does the most work per step. The diagnosis is not quality, it is that the policy spends compute it does not have to. The redundancy is real — across a rollout there are at most $L$ distinct conditioning states, and most steps flip only a handful of masked positions — and the floor captures none of it. So the move is forced: reuse the keys and values of positions whose context did not meaningfully change this step, and recompute only the positions that did. The entire question is *which* positions, at *what granularity*.

The crude answer — a segment-and-clock cache that refreshes the prompt on a long interval and the response on a short one — fails because a segment is not a unit of KV dynamics. Inside one response segment at one step some tokens are dead stable and some are about to lurch, and a single interval treats them identically, so it does both wrong things at once: it refreshes settled tokens (the floor's wasted compute in miniature) and reuses moving tokens (lost quality, which a gated metric like MATH punishes). I need the dynamics at the granularity they actually live at, which is per token, per step. So I propose *d2Cache*: decide reuse per token per step, recomputing only the union of the tokens that genuinely moved and reusing cached K/V for everyone else.

What makes that decidable is that a masked token's KV trajectory is not a smooth drift but three phases — a gradual-change phase early, a *rapid-change* phase in the handful of steps right before the token is decoded, and a stable phase after — and the lurch has a nameable cause: it happens when the token's local context fills in, because a neighbor flipping from a `[MASK]` embedding to a concrete token is a large change to the context this token attends to, larger the closer the neighbor. So I only need to recompute a masked token's KV during its rapid-change phase; in the gradual and stable phases I reuse it and lose nothing. But identifying a token *during* its rapid-change phase means identifying it *before* it is decoded, which is circular. The decoding order breaks the circle: the model overwhelmingly decodes near the token it just decoded — the front is spatially local — so imminence is the *density of known tokens in the local neighborhood* of a masked position, which depends only on which positions are currently known. I make it distance-aware with a Gaussian, since a known token right next to my masked position should count for far more than one ten away, and a Gaussian has no hard cutoff: for masked position $i$,
$$D(i) = \sum_{j} \exp\!\left(-\frac{|i-j|^2}{2\sigma^2}\right)\,\mathbb{1}\{j \text{ known}\},$$
summing over prompt and already-decoded positions, with $\sigma = 10$ chosen so the Gaussian's reach matches the empirically observed decoding-locality scale of about ten positions. Density gives structural imminence; the model's own per-position prediction confidence $s^i$ comes free after each forward and gives predictive certainty, and these are different (a token can sit in a dense neighborhood yet be predicted unsurely). I want both, so I multiply — the calibrated score $D(i)\cdot s^i$ is a soft logical AND — and refresh the top-`current_k` masked tokens by it, with a small fixed budget `current_k = 32` because the front is narrow.

That handles masked tokens but is blind to the prompt and already-decoded tokens, which have no confidence $s^i$. Their KV moves little — that is why caching them helps — but "little" is not "nothing," and never refreshing them accumulates error that would eventually push a gated score under threshold. They need a different reason to refresh, and the right one is that *other* tokens depend on them: a token that many queries attend to strongly is one whose KV I cannot let drift. So a second stage selects by attention importance, and to measure it faithfully through a deep stack I do not read the last layer — I compose attention across all layers with the residual path folded in, $W = \mathrm{normalize}(E + I)$, $C \leftarrow W\cdot C$, then take the column sum $c_j = \sum_i C_{ij}$ as the influence flowing into token $j$, selecting by nucleus mass with `rollout_p = 0.1`. A tenth of the cumulative influence already captures the salient tokens because dLLM attention is concentrated, and adjacent-step rollout maps are stable enough that this step's importance predicts next step's recompute set. The recompute set is the union: the imminent masked tokens by $D\cdot s$, the salient known tokens by rollout, plus any token freshly transferred this step (its embedding just changed fundamentally, so it must be refreshed once). Everything else reuses cache. The two stages match the two token populations that have genuinely different dynamics — masked tokens change a lot but briefly and locally, prompt/decoded tokens change little but are heavily attended — and neither signal can find the other. It is training-free: density comes from the mask layout, confidence and attention from the forward already run.

Landing this in the task's hooks differs from the generic method in two places I have to get right. First, the block schedule: the generic method preserves a frame/delta outer loop with semi-autoregressive blocks, but *this task runs d2Cache as pure diffusion* — `block_length = gen_length`, `num_steps = gen_length`, one token per step, no block boundaries. That is legitimate here because $D\cdot s$ is doubling as a decoding criterion: it already prefers tokens next to known tokens, and known tokens grow outward from the prompt, so decoding by $D\cdot s$ produces a quasi-left-to-right order on its own and curbs the premature-EOS pathology that block decoding existed to fix. Second, the query plan: step 0 forwards the `full_sequence` to fill the cache, every later step narrows to `active_query_rows`, handing the harness the `active_q_mask` and the masked window; `row_selector = "certainty_density_attention_rollout"` and `kv_update = "active_q_mask"` (inactive rows reuse cached K/V), with the feature cache *off* because the saving here is narrowing the query *rows*, not interval-skipping features. The attention-probe plan must set `need_attention_weights = True` — the rollout needs the explicit $\mathrm{softmax}(QK^\top)$ matrix, which fused kernels never materialize, so the harness routes attention through eager mode — and carries `rollout_p = 0.1`, `current_k = 32`, `sigma = 10.0`. The one knob I deliberately set against the generic default is `inflate_w = 0`: the method offers a gap-inflation that fills small holes between selected rows to keep the attention kernel contiguous, but I keep the active set strictly minimal, trading some kernel efficiency for the smallest recompute set and the highest reuse_ratio. Transfer stays low-confidence over the current block, forcing one; `after_step` is where the harness updates the active query mask, attention rollout, tracked positions, and density scores.

I expect reuse to jump hard off $0.0$ — the active set is a small fraction of the sequence, so on long pure-diffusion responses most tokens are inactive most steps, and I would expect well above $0.8$, the highest on the board. Throughput should *not* leap: the active queries still attend over the whole sequence, the eager-attention requirement adds real overhead, and computing $D\cdot s$ plus the rollout matrix every step is not free, so tokens/s should only inch up from the floor. The risk I can already feel is the MATH quality gate. MATH answers are long, sequential, and unforgiving — a single stale feature that flips a committed digit is unrecoverable — and MATH has the least headroom ($38.4$ native, gate $35$). If aggressive active-row narrowing reuses the KV of any token $D\cdot s$ underrates and bleeds even a few points, MATH slides under $35$, the gate penalty multiplies that whole workload down, and the geometric mean drags the policy below a blander one despite top reuse. If that is what happens, the next step's job is already written: keep selective reuse but refresh on a signal tied to whether the *prediction* actually changed, not a structural density proxy.

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
