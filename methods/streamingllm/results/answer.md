# StreamingLLM, distilled

StreamingLLM lets an already-trained decoder-only LLM decode over effectively unbounded token
streams with constant memory and per-token latency, and no finetuning, by managing the KV cache
as two pinned regions: a few **attention-sink** tokens at the very start, kept permanently, plus
a **rolling window** of the most recent tokens. The middle is dropped. Kept tokens are
re-positioned by their index *within the cache* (RoPE keys are re-rotated; ALiBi biases are made
contiguous) so all relative distances stay inside the model's trained range.

## Problem it solves

Streaming deployment (multi-round dialogue, day-long sessions) of an LLM with a fixed
pretraining window. Caching every past token's KV makes memory and latency grow without bound,
and quality collapses once the input exceeds the pretraining length. The goal is fixed-budget
decoding (constant memory/latency) that preserves language-modeling quality for sequences far
beyond the training window, on off-the-shelf models running standard dense-attention kernels.

## Key idea

- **Window attention fails as a cliff.** Keeping only a recent window is healthy until the
  sequence outgrows the cache — then perplexity spikes abruptly, exactly when the *first* tokens
  are evicted, not gradually as old context is lost.
- **Attention sinks.** Beyond the bottom couple of layers, almost every head dumps a large
  fraction of its attention mass onto the first few token positions regardless of content (often
  more than half the total mass goes to token zero). Replacing the first tokens with linebreaks
  doesn't move this, so it is **absolute position, not semantics**, that matters.
- **Why.** Softmax forces attention weights to sum to one — there is no abstain — so when a query
  has no strong match, the surplus mass must land somewhere. Causal masking makes the initial
  tokens the only ones visible to *every* later query, so training drives them into the role of a
  shared dumping ground ("attention sinks"). If `S` is the sink set and `R` the remaining cache,
  evicting `S` changes a non-sink weight from `e^{x_j}/(sum_S e^{x_s}+sum_R e^{x_r})` to
  `e^{x_j}/sum_R e^{x_r}`. When the sink denominator is large, every remaining value is amplified
  and the attention output shifts sharply downstream — that is the cliff.
- **Fix.** The sinks are valuable for the attention/denominator mass they hold, not their content,
  so pin them: keep the first `n_sink` tokens' KV permanently and slide the window over the rest.
  The retention rule is **static and positional** (no attention read needed): keep sinks + recent,
  drop the middle. `n_sink = 4` for off-the-shelf models because, lacking a consistent fixed start
  token in pretraining, they spread the sink role over a few initial positions (1-2 insufficient,
  4 enough, more marginal).
- **Position within the cache.** Assign positions by cache index, not original text index, so
  relative distances stay contiguous and inside the trained range as the stream rolls forward.

## RoPE re-rotation (the load-bearing detail)

RoPE rotates query/key by absolute position, `q_m = R_{Θ,m} W_q x_m`, `k_n = R_{Θ,n} W_k x_n`,
with `R` block-diagonal, orthogonal (`R_{Θ,m}^T = R_{Θ,-m}`), and additive
(`R_{Θ,m} R_{Θ,m'} = R_{Θ,m+m'}`). The inner product depends only on relative position:
`q_m^T k_n = x_m^T W_q^T R_{Θ,n-m} W_k x_n`. A key cached (after rotation) at original position
`p` is `R_{Θ,p}(W_k x)`; to make it act at a new cache position `p'`, left-multiply by
`R_{Θ,p'-p}` (since `R_{Θ,p'} = R_{Θ,p'-p} R_{Θ,p}`). With the efficient form
`R_{Θ,m} x = x ⊙ cos(m θ) + rotate_half(x) ⊙ sin(m θ)`, where
`rotate_half([x_1, x_2]) = [-x_2, x_1]`, re-rotate by `delta = p' - p` using `cos(delta θ)`,
`sin(delta θ)`. After top-k and sorting, new positions are `0..n_kept-1`, so `delta =
new_index - original_index` (`<= 0`: rotate backward to close gaps). Values carry no position and
are simply gathered. For ALiBi, apply a contiguous linear distance bias instead of a "jumping" one
— no re-rotation.

## Pretraining fix (prevent rather than rescue)

The four-sink pinning is needed only because off-the-shelf models lack a dedicated sink. Prepend
a single **learnable sink token** to every pretraining sample and the model gets one consistent
slot to offload surplus attention; then a *single* pinned sink is the intended streaming
configuration. A weaker variant is the **zero sink** — replace softmax with
`SoftMax_1(x)_i = e^{x_i} / (1 + sum_j e^{x_j})`, whose `+1` is exactly a prepended token with
all-zero key and value (it contributes `e^0 = 1` to the denominator and zero to the output) —
which gives an abstain option but still lets the model lean on other initial tokens.

## Final algorithm

```
budget B (cache size), n_sink = 4, recent window L = B - n_sink
maintain cache = [first n_sink tokens' KV]  ++  [most recent L tokens' KV]   # drop the middle
on each new token:
    append its KV to the recent window
    if cache size > B:
        evict from the recent window only (never the n_sink sinks)
    assign positions by index within the cache (contiguous), not original text index
    for RoPE: re-rotate each kept key by delta = (new cache position) - (original position)
    for ALiBi: use a contiguous linear distance bias
```

## Working code

A direct implementation of the selection-policy slot:

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

An equivalent cache-then-slice implementation keeps `[0:n_sink]` concatenated with
`[seq_len - L : seq_len]` of K and V, with RoPE applied at the contiguous cache positions
(`apply_rotary_pos_emb` keyed on `arange(kv_seq_len)`) at each decode step.
