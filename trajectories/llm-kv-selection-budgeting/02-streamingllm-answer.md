**Problem.** The full-attention anchor pins the quality ceiling but retains 1.0 and fails the budget.
I need a compressor that keeps ~20% of each layer's prefill cache with minimal quality loss, using only
positions / cached tensors — the hook passes no attention matrix and no query.

**Key idea.** A bare recent-only window collapses at a cliff the moment the *first* tokens are evicted,
because the model routes a huge share of attention onto the initial positions regardless of content
(attention sinks: softmax's sum-to-one forces surplus mass somewhere, and causal visibility makes the
earliest tokens the only common dumping ground). The sinks hold the softmax denominator, so evicting
them corrupts every downstream layer. Fix: pin the first `n_sink` tokens permanently and slide the
remaining budget over the most recent tokens, dropping the middle — a static, positional, attention-free
retention rule.

**Why it works (and where it won't).** It keeps the denominator anchor and the recent context, so it
holds where the answer is recent or locally inferable. But it discards the middle *blindly*, so a fact
sitting in the middle of a long context is dropped — the structural weakness a content-aware score must
later fix.

**RoPE re-rotation.** With RoPE the query-key product depends only on relative offset, so keeping
original positions with a hole leaves an unbounded sink-to-window gap that drifts outside the trained
range. Re-label kept tokens to contiguous cache positions `0..n_kept-1` and re-rotate each kept key by
`delta = new - old` (since `R_{p'} = R_{p'-p} R_p`, left-multiply the stored rotated key by `R_delta`).
Values carry no position — just gather them. `rerotate_selected_keys = True` so the harness advances
decode positions from the contiguous re-rotated cache.

**Step-2 edit.** `score_tokens` returns a static mask: 1 everywhere, 0 on the middle block
`[n_sink : n_sink + n_pruned]` with `n_pruned = k_len - int(k_len*(1-ratio))`; `compression_ratio` is
read from `cache_meta` (the harness force-overrides it anyway). `select_cache` does top-k, sorts kept
indices chronologically, re-rotates keys (gather width from `keys.shape[-1]`, rotary table from
`module.rotary_emb.inv_freq`), gathers values.

**Hyperparameters.** `sink_tokens = 4` (off-the-shelf models spread the sink role over the first few
positions; four holds it, more is marginal); recent-window size implied by the budget (`~20%` after the
4 sinks).

**What to watch.** Retained ~0.20 (clears the budget penalty the anchor failed). Expect hotpotqa /
passage retrieval to drop below the anchor (middle needles dropped), LongBench v2 to roughly hold, and
gsm8k to be the loud signal: if the chain-of-thought prefix is shredded it falls hard from 31.8 — the
hinge that says a positional rule is the wrong shape and a content-aware score is needed.

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
