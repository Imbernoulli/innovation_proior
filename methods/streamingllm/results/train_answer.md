We want to keep an already-trained decoder-only Transformer running over a token stream that never ends — a multi-round chat assistant up over day-long sessions, a log monitor, a running transcript — decoding token after token, in principle without bound. Two walls make this impossible with a vanilla model. The first is mechanical: autoregressive decoding caches the Key and Value of every past token in every layer, the KV cache, so each new query attends to all of history without recomputing it. That cache grows linearly with the number of tokens and the per-token attention cost grows with it, so on a stream of millions of tokens we exhaust GPU memory and the latency climbs until the system is unusable. The second wall is deeper: even with infinite memory, quality falls apart once the input is longer than the model's pretraining window (4K for Llama-2), because the position encodings and attention patterns were never exercised at those distances. So the precise goal is an inference-time scheme that lets the model decode from a *fixed-budget* slice of the KV cache — constant memory, constant per-token latency — while keeping language-modeling quality intact for sequences far beyond the trained window, on the model as it ships, no finetuning, on standard dense-attention kernels.

The obvious thing to try is a sliding window: cache only the most recent $L$ tokens, evict the oldest as new ones arrive. Memory and latency go flat once the window fills, and intuitively recent context is what matters for the next token. But the perplexity curve betrays it. For the first $L$ tokens the cache holds everything, so this is exactly dense attention and perplexity is flat and healthy; then token $L{+}1$ arrives, the cache is full, and to make room we evict the oldest token — token number one, the very first token of the text — and at exactly that step perplexity spikes. Not a gentle slope of forgetting; a cliff. The evicted tokens are thousands of positions away from what we are predicting and should be irrelevant, the recent window is full of perfectly good context, yet dropping the ancient first token detonates the model. Re-encoding the recent window from scratch each step (sliding window with recomputation) does preserve quality and is the practical oracle, but it costs $O(L^2)$ attention per generated token and is far too slow for real-time streaming. Heavy-hitter eviction (H2O) keeps the top-K tokens by accumulated attention, but it reads the realized attention matrix at decision time, adds dynamic machinery to the decode loop, and is built to compress within a window, not to extrapolate without bound. Length-extrapolation position tricks (position interpolation, NTK-aware scaling, YaRN) push the usable window out a finite amount but give neither constant memory nor unbounded streaming; they are orthogonal to the cache-budget problem. None of these does all of it at once.

The fix has to come from understanding the cliff, so look at what the model actually does with its attention. Across all layers and heads of Llama-2, MPT, Falcon, and Pythia, beyond the bottom couple of layers nearly every head dumps a huge fraction of its attention mass onto the first few token positions — on length-4096 sequences the attention from the last token back to the very first token often exceeds *half* of the total mass, in most layers. The discriminating test is to replace the first four tokens of a real text with a meaningless linebreak token: if content drove the attention, garbage tokens should not attract it, but they do, and reintroducing them after an eviction restores perplexity just as the originals would. So it is not semantics — it is *absolute position*. The model has learned to use whatever sits in the first few slots as a fixed destination for attention.

I propose StreamingLLM. The reason for the position-bound concentration is the softmax. Within a head the attention weights are $\mathrm{SoftMax}(x)_i = e^{x_i} / \sum_j e^{x_j}$ and are forced to sum to one over all attended tokens — there is no "abstain." When a query is essentially self-contained and has no strong match anywhere, the surplus probability mass must still land *somewhere*, and the cheapest place to dump it is a token whose value vector can be mixed in harmlessly, one that is always present and easy to find. The model therefore manufactures a few such dumping grounds; I call them attention sinks — tokens that collect large attention scores precisely *because* they are not semantically important. This rhymes with the quantization-side observation of persistent outlier activations traced to the same forced allocation, which prompted the off-by-one variant $\mathrm{SoftMax}_1(x)_i = e^{x_i} / (1 + \sum_j e^{x_j})$ whose extra $+1$ lets the weights sum to less than one. Why the *initial* tokens specifically? Causal masking: a token at position $t$ is visible only to queries at positions $\ge t$, so token zero is the one slot every later query can reach. A shared dumping ground must be a universal back-reference, and only the earliest positions qualify, so gradient descent drives the first few tokens into the sink role.

This explains the cliff exactly. Write the attended positions as a sink set $S$ plus the ordinary remainder $R$. Before eviction a non-sink token $j \in R$ has weight
$$a_j = \frac{e^{x_j}}{\sum_{s \in S} e^{x_s} + \sum_{r \in R} e^{x_r}},$$
and after evicting $S$ the same logit gives $a'_j = e^{x_j} / \sum_{r \in R} e^{x_r}$, so
$$a'_j = a_j \cdot \frac{\sum_{s \in S} e^{x_s} + \sum_{r \in R} e^{x_r}}{\sum_{r \in R} e^{x_r}}.$$
When the sinks dominate the denominator, every remaining weight is multiplied by a large factor even though none of the remaining logits improved; the attention output also loses the sink-value contribution and amplifies the ordinary values, handing the downstream layers a distributional shape they never see in normal dense inference. The window had perfectly good recent tokens but threw away the denominator anchor — that is the cliff. So the rescue is forced: do not evict the sinks. Keep the first few tokens' KV permanently, pinned, and slide the window over everything else, dropping the middle. The cache becomes two pieces — a small fixed set of initial tokens plus a rolling window of the most recent tokens. The sinks anchor the attention computation and hold the denominator close to its normal shape; the rolling window carries the actually-useful recent context. No finetuning, no architecture change, just a cache-management rule on top of standard attention.

How many initial tokens to pin? If the model had been trained with one fixed token always at position zero, a single sink slot would suffice. But standard models do not have that — Llama-2's `<s>` is prepended before chunking, so the token actually at position zero of a training chunk is mostly random — and lacking a consistent starting token the model spreads the sink role across the first *few* positions. That predicts a saturating shape for a perplexity-vs-sink-count sweep: one or two pinned tokens should leave much of the spread-out sink mass stranded outside the cache, the recovery should be roughly complete once the handful of positions carrying the role are captured, and pinning still more should buy little. Absent that sweep in front of me, I take the saturation point to be small — around four rather than one or twenty — and default to $n_{\text{sink}} = 4$ for off-the-shelf models, with the recent window taking whatever the budget leaves.

The subtlety the naive "first four plus last $L$" rule gets wrong is position encoding. Most of these models use relative encodings — RoPE in Llama-2, Falcon, Pythia; ALiBi in MPT — and the trap is to keep each kept token labeled with its *original* text position. RoPE rotates query and key by an angle proportional to absolute position, $q_m = R_{\Theta,m} W_q x_m$ and $k_n = R_{\Theta,n} W_k x_n$, with $R_{\Theta,m}$ block-diagonal from frequencies $\theta_i = 10000^{-2(i-1)/d}$, orthogonal ($R_{\Theta,m}^{\top} = R_{\Theta,-m}$) and additive ($R_{\Theta,m} R_{\Theta,m'} = R_{\Theta,m+m'}$), so the inner product depends only on relative distance:
$$q_m^{\top} k_n = x_m^{\top} W_q^{\top} R_{\Theta,m}^{\top} R_{\Theta,n} W_k x_n = x_m^{\top} W_q^{\top} R_{\Theta,n-m} W_k x_n.$$
What matters is the *gaps* between kept tokens. If I keep original positions $[0,1,2,3,6,7,8]$ with a hole where evicted tokens $4,5$ were, then as the stream rolls forward the recent window's original positions keep climbing, so the relative distance between the pinned sinks and the rolling window grows without bound — straight into the un-trained RoPE regime, reintroducing the very length-extrapolation failure I am dodging. The fix is to assign positions by index *within the cache*, not original text position: treat the kept set as occupying contiguous positions $[0,1,2,\dots]$ and the new query as the next contiguous slot, so every relative distance the model ever sees is small, contiguous, and inside the trained range no matter how far the stream has progressed. The physical gap in the original text is simply erased from the position bookkeeping.

Concretely, for RoPE I must re-rotate. A key cached *after* rotation at original position $p$ is $R_{\Theta,p}(W_k x)$, and I want it to behave as if at new cache position $p'$, i.e. $R_{\Theta,p'}(W_k x)$. Since rotations compose, $R_{\Theta,p'} = R_{\Theta,p'-p} R_{\Theta,p}$, so I left-multiply the stored, already-rotated key by $R_{\Theta,p'-p}$ — rotate it by the *difference* $\delta = p' - p$. I never need the un-rotated key. In the efficient form $R_{\Theta,m} x = x \odot \cos(m\theta) + \mathrm{rotate\_half}(x) \odot \sin(m\theta)$, with $\mathrm{rotate\_half}([x_1, x_2]) = [-x_2, x_1]$, I form the angles $\delta\theta$ and apply the same formula. After top-k selection and sorting into chronological order, the new positions are $0, 1, \dots, n_{\text{kept}}-1$, so $\delta = \text{new\_index} - \text{original\_index}$, generally negative — rotating *backward* to close the gaps, which the signed orthogonal rotations make exactly consistent. Values carry no position and are simply gathered. For ALiBi the principle is even simpler: apply a *contiguous* linear distance bias over the cache positions instead of a "jumping" bias that reflects the original-text gap; no re-rotation.

So no clever per-token scoring rule is needed. The linebreak test already showed the initial tokens are important by position, not content, and the retention rule that follows from the diagnosis is purely *positional and static*: keep the first $n_{\text{sink}}$ tokens, keep the most recent $(\text{budget} - n_{\text{sink}})$, drop the middle, regardless of attention values — no per-step attention read. In the harness's form, that is a score of $1$ on everything and $0$ on the middle block to prune: with $k$ cached tokens and compression ratio $r$, the retained count is $n_{\text{kept}} = \lfloor k(1-r) \rfloor$, the prune count is $n_{\text{pruned}} = k - n_{\text{kept}}$, and the zero slice is $[n_{\text{sink}} : n_{\text{sink}} + n_{\text{pruned}}]$, leaving exactly the sinks and the recent window. Top-k keeps that set, sorting restores chronological order for the contiguous re-positioning, and `select_cache` gathers the kept values and gathers-and-re-rotates the kept keys.

One further idea is prevention rather than rescue. We needed four sinks only because off-the-shelf models never had a dedicated sink and spread the role over a few positions. If at pretraining we prepend a single *learnable* sink token to every sample, the model gets one consistent, always-present slot to offload surplus attention onto, and at streaming time only that one token need be pinned; it costs nothing on normal tasks, since it absorbs the leftover mass softmax was already wasting on initial tokens. A weaker variant is the *zero sink*: replace softmax with $\mathrm{SoftMax}_1(x)_i = e^{x_i} / (1 + \sum_j e^{x_j})$, whose $+1$ is exactly a prepended token with all-zero key and value — it contributes $e^{q\cdot 0} = 1$ to the denominator and zero to the output — giving an abstain option, but it is weaker than a learnable sink because the model can still lean partly on other initial tokens, whereas a content-bearing learnable sink gives it a full destination to route to cleanly. So for any model trained from scratch the recommendation is the dedicated learnable sink; the four-sink pinning is the no-retraining rescue for models that already exist.

```python
import torch


class SelectionPolicy:
    """StreamingLLM: keep attention sinks and the most recent tokens; drop the middle.
    Re-rotate kept keys to their positions within the cache (RoPE relative distances
    stay inside the trained range). Values carry no position, so they are just gathered."""

    method_name = "streamingllm"
    rerotate_selected_keys = True

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,                       # 4 for off-the-shelf models
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        # Static positional mask: 1 on sinks + recent, 0 on the middle block to prune.
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
        # R_{p}(W_k x) -> R_{p'}(W_k x) = R_{p'-p} R_{p}(W_k x); rotate by delta = new - old.
        bsz, num_key_value_heads, n_kept = indices.shape
        device = indices.device
        device_type = keys.device.type
        dtype = keys.dtype
        inv_freq = module.rotary_emb.inv_freq[None, None, :, None].float().expand(
            bsz, num_key_value_heads, -1, 1
        )
        new_positions = torch.arange(0, n_kept, device=device).unsqueeze(0)[:, None, :].float()
        new_positions = new_positions.expand(bsz, num_key_value_heads, n_kept)
        delta_pos = (new_positions - indices.float()).unsqueeze(2)  # delta = new - old (<= 0)
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        gathered = keys.gather(2, gather_idx).contiguous()
        return (gathered * cos) + (self.rotate_half(gathered) * sin)

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices    # sinks + recent window
        indices = torch.sort(indices, dim=2).values      # chronological order
        selected_keys = self.rerotate_cache_keys(module, indices, keys)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        selected_values = values.gather(2, gather_idx).contiguous()
        return selected_keys, selected_values
```
