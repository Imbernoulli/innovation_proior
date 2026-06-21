The full-attention anchor pinned the five accuracies I now have to defend — hotpotqa 37.1, passage retrieval 62.4, repobench 47.6, LongBench v2 29.0, gsm8k 31.8 — at retained $1.0$, which is four times over the $0.25$ budget tolerance, so as a budgeted submission it is worthless. The task is now sharp: get as close to those accuracies as I can while throwing away roughly 80% of every layer's prefill cache, using only positions and cached tensors, because the hook hands me keys, values, and hidden states but no attention matrix and no query. So I reach for the cheapest rule that could plausibly hold quality and read its gap to the anchor per workload, since *which* workloads collapse will tell me whether a positional retention rule is even the right shape.

The obvious cheap rule is recent-only: cache the last $L$ tokens, evict the oldest as the window rolls. It needs nothing but position, which is exactly what this hook allows. But a bare recent window does not fail gently — it falls off a cliff. The known perplexity curve is flat and healthy while the cache still holds everything, and then the instant the window fills and the *first* token of the sequence is evicted, perplexity spikes and the model produces nonsense. That is bizarre: the evicted token is thousands of positions from what is being predicted, the window is full of good recent tokens, yet dropping the ancient first token detonates the model. So this is not lost long-range *information* — the collapse is tied specifically and abruptly to evicting the *initial* tokens. Looking at what the model actually does, beyond the bottom couple of layers nearly every head dumps a huge fraction of its attention mass — often more than half on long sequences — onto the first few positions. Replacing those first tokens with meaningless filler does not move the attention away from them, and reintroducing fillers after an eviction restores quality, so it is absolute position, not content, that the model is using: it has learned to treat whatever sits in the first few slots as a fixed destination for attention.

The mechanism is the softmax. Within a head the weights are $a_{ti} = \exp(x_{ti}) / \sum_j \exp(x_{tj})$, and the defining property is that they sum to one — there is no "none of the above." A query whose information is essentially self-contained, with no strong match anywhere, still must produce a distribution summing to one, so it dumps the leftover mass somewhere cheap: a token whose value it can mix in harmlessly without corrupting the residual stream. The model manufactures a few such *attention sinks* — tokens that collect large attention precisely because they are *not* semantically important, just to soak up the surplus the softmax forces it to allocate. And the initial tokens become the sinks because of causal visibility: a query at position $t$ can attend only to positions $\le t$, so the only positions visible to *every* later query are the earliest ones. The first few tokens are the one common dumping ground every position can reach.

Now the cliff is mechanical. Split the attended positions into a sink set $S$ and the ordinary rest $R$. Before eviction a non-sink token's weight is $\exp(x_j) / (\sum_S \exp + \sum_R \exp)$, and the sinks dominate that denominator. Evict $S$ and the same logit now divides by only $\sum_R \exp$: every remaining weight is multiplied by a large factor even though no remaining logit improved, the sink-value contribution to the output vanishes, and the ordinary values are over-amplified. Downstream layers receive a distributional shape they never see in normal inference. The recent window threw away the denominator anchor. So the rescue writes itself, and I propose *StreamingLLM*: pin a small fixed set of the very first tokens permanently, and slide the budget's remaining capacity over the most recent tokens, dropping the middle. The cache becomes two pieces — sinks plus a recent window — with no attention read and no query dependence, which fits this hook precisely because the hook forbids exactly the things this rule does not need. As for how many sinks: a model trained with one dedicated fixed token would need one slot, but an off-the-shelf instruct model spreads the sink role across the first *few* positions, and sweeping the count has the shape where one or two does not recover quality, four does, beyond four is marginal. So $n_{\text{sink}} = 4$, and the recent window is whatever the budget leaves after the four sinks. The retention decision is entirely positional and static — fixed before looking at a single attention value.

There is one subtlety the naive "keep first four plus last $L$" rule gets silently wrong on this model, and it matters because Qwen uses RoPE. RoPE makes the query-key inner product depend only on the *relative* offset: with $q_m = R_m W_q x_m$, $k_n = R_n W_k x_n$, $R$ orthogonal and additive in the index, $q_m^\top k_n = x_m^\top W_q^\top R_{n-m} W_k x_n$, a function of $n - m$ alone. So the model sees the *gaps* between kept tokens. If I keep original positions $[0,1,2,3,\dots,m-1,m]$ with a hole where the middle was, the relative distance between the pinned sinks and the rolling window grows without bound as the stream advances, and those huge relative distances are exactly the regime RoPE was never trained on. I would be re-importing the length-extrapolation failure I am trying to dodge, through the back door, and on long retrieval prompts that is where it hurts most.

The fix is to assign positions by index *within the cache*, not by original text position. If the cache holds $n_{\text{kept}}$ tokens, treat them as occupying contiguous positions $0, 1, \dots, n_{\text{kept}}-1$ regardless of where they came from, so every relative distance the model sees stays small and contiguous, inside the trained range, no matter how far the stream has gone. Concretely, the keys are stored already rotated to their original positions — a key for original position $p$ is $R_p (W_k x)$ — and I want it to behave as if it sat at new cache position $p'$. Since rotations compose, $R_{p'} = R_{p'-p} R_p$, so I left-multiply the stored, already-rotated key by $R_{p'-p}$, i.e. re-rotate it by $\delta = p' - p$; I never need the un-rotated key. In the efficient RoPE form, $R_m x = x \cos(m\theta) + \text{rotate\_half}(x)\sin(m\theta)$, where $\text{rotate\_half}$ swaps the two halves with a sign flip, so to re-rotate by $\delta$ I form the angles $\delta\theta$ from the module's rotary inv-freq table, take cos and sin, and apply the same formula. After sorting the kept indices chronologically, the $j$-th kept token's $\delta$ is $j$ minus its original index — generally negative, a backward rotation that closes the gaps. Values carry no position, so they are just gathered. Because this policy always re-rotates, `rerotate_selected_keys = True`, and the harness advances decode positions from the re-rotated, contiguous cache length rather than the original sequence length.

Mapping onto the three-method hook, I respect what *this* harness exposes. `retention_plan` hands the sink count and the budget — I read `compression_ratio` from `cache_meta`, but since the harness force-overrides it at the call site, declaring it is only provenance; the harness enforces its own value. `score_tokens` produces the static positional mask the harness wants: it keeps the top-$n_{\text{kept}}$ by score, so I score $1$ everywhere and $0$ on the middle block to prune. With $n_{\text{kept}} = \lfloor k_{\text{len}}(1-r)\rfloor$, the number to prune is $n_{\text{pruned}} = k_{\text{len}} - \lfloor k_{\text{len}}(1-r)\rfloor$, and the zero slice starts immediately after the sinks, at $[n_{\text{sink}} : n_{\text{sink}} + n_{\text{pruned}}]$, leaving exactly the first $n_{\text{sink}}$ and the most recent $n_{\text{kept}} - n_{\text{sink}}$ tokens one-scored. One harness detail: the re-rotation builds its gather index from `keys.shape[-1]` (the actual stored head dimension) rather than a `module.head_dim` attribute, so the gather width always matches the cache, and the rotary table comes from `module.rotary_emb.inv_freq`, which the harness wires onto every attention module. `select_cache` does the top-k, sorts the kept indices chronologically (required for the contiguous re-positioning to make sense), re-rotates the kept keys by their deltas, and gathers the values.

I expect retained to land at $\sim 0.20$ across all five workloads, clearing the penalty the anchor failed, and runtime near the anchor's or a touch below since the decode cache is a fifth the size. The accuracy bets follow from the rule's shape: it keeps recent context and the denominator anchor but discards the middle *blindly*, so it should hold where the answer lives near the end or is locally inferable and bleed where the answer can sit anywhere in a long passage. Passage retrieval and hotpotqa should drop below the anchor — those are needle-in-a-haystack workloads where a positional rule drops the needle in the middle — while LongBench v2 should roughly hold given its head-tail truncation and multiple-choice format. The number I most worry about is gsm8k: the reasoning prefix is exactly what gets compressed, so if streaming shreds the chain-of-thought, gsm8k falls hard from 31.8, possibly to near zero. That would be the loud signal that a purely positional rule is the wrong shape and that I need a *content-aware* score able to decide which middle tokens to keep — and that gsm8k gap, read against 31.8, is the falsifiable hinge that decides the next step.

```python
# EDITABLE region of custom_selection_eval.py (lines 40-101) — step 2: StreamingLLM
class SelectionPolicy:
    """StreamingLLM: keep attention sinks and the most recent tokens."""

    method_name = "streamingllm"
    rerotate_selected_keys = True

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        k_len = int(keys.shape[2])
        n_sink = int(plan.get("sink_tokens", 4))
        ratio = float(plan["compression_ratio"])
        assert k_len > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
        n_pruned = k_len - int(k_len * (1.0 - ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, n_sink : n_sink + n_pruned] = 0
        return scores

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rerotate_cache_keys(self, module, indices, keys):
        bsz, num_key_value_heads, n_kept = indices.shape
        device = indices.device
        device_type = keys.device.type
        dtype = keys.dtype
        inv_freq = module.rotary_emb.inv_freq[None, None, :, None].float().expand(
            bsz, num_key_value_heads, -1, 1
        )
        new_positions = torch.arange(0, n_kept, device=device).unsqueeze(0)[:, None, :].float()
        new_positions = new_positions.expand(bsz, num_key_value_heads, n_kept)
        delta_pos = (new_positions - indices.float()).unsqueeze(2)
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        gathered = keys.gather(2, gather_idx).contiguous()
        return (gathered * cos) + (self.rotate_half(gathered) * sin)

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices
        indices = torch.sort(indices, dim=2).values
        selected_keys = self.rerotate_cache_keys(module, indices, keys)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        selected_values = values.gather(2, gather_idx).contiguous()
        return selected_keys, selected_values
```
