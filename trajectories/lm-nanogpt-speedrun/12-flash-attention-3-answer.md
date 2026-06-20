**Problem (from step 11).** The sparse-attention-gate record reached 2.812 min (168,938 ms, 1695 steps,
~99.7 ms/step). With the architecture lean and the optimizer good, the dominant per-step cost is the
attention kernel itself, which is FlexAttention — a *general* mechanism that synthesizes a kernel from an
arbitrary block-mask function and leaves H100 throughput on the table compared to a hand-tuned kernel.

**Key idea (FlashAttention-3 with varlen packing + window schedule).** Swap the FlexAttention call for
**FlashAttention-3**, the Hopper-tuned kernel that overlaps softmax with warp-group-async matmuls and TMA
copies for far higher SM utilization than FlashAttention-2 (and than compiled FlexAttention). FA3 takes
only `causal`, `window_size`, and `cu_seqlens`, so re-express the three masks: the **document mask** becomes
free **varlen packing** — each FineWeb document is its own variable-length sequence delimited by
`cu_seqlens`, so the kernel never attends across a boundary; the **sliding window** becomes
`window_size=(bm_size, 0)`; the **window warmup** becomes a short discrete `ws_schedule = (3, 7, 11)` (via
`get_ws(step)`) so FA3 recompiles only a handful of times instead of on every window value. Documents are
clipped to `train_max_seq_len = 2048` so each fixed 393216-token batch packs a greater diversity of
documents, and `cu_seqlens` is padded to a fixed max-document count for static shapes.

**Why it works.** Attention is the fattest op per step; FA3's hand-tuned asynchronous kernel does the
attention matmuls and softmax materially faster than a compiled general kernel on Hopper, so the per-step
time drops directly. Varlen packing makes document masking free (no masked-out cross-document scores), and
clipping to 2048 increases per-step document diversity, trimming a few steps. Needs a compile-compatible FA3
build (the official module graph-breaks under `torch.compile`) pinned to a specific torch nightly; warmup
iterations are bumped to 30 to warm each window graph, a one-time cost outside the raced wallclock.

**Change / code.** FlexAttention call → `flash_attn_varlen_func` with `cu_seqlens` and `window_size`.

```python
# compile-compatible FlashAttention-3 build, replacing the FlexAttention call
flash_attn_interface = get_kernel('kernels-community/flash-attn3', version=1).flash_attn_interface

def forward(self, x, attn_args, qkvo_w):
    B, T = x.size(0), x.size(1)
    assert B == 1, "varlen sequences requires B == 1"
    seqlens, bm_size = attn_args.seqlens, attn_args.bm_size          # cu_seqlens, current window size
    train_max_seq_len, yarn = attn_args.train_max_seq_len, attn_args.yarn
    q, k, v = F.linear(x, qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
    max_len = train_max_seq_len if self.training else val_max_len
    q, k = norm(q), norm(k)
    q, k = yarn.rotary(q), yarn.rotary(k)
    # use flash_attn over flex_attn @varunneal; flash_attn_varlen suggested by @YouJiacheng
    y = flash_attn_interface.flash_attn_varlen_func(
        q[0], k[0], v[0],
        cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,                  # document boundaries (free document mask)
        max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=True, softmax_scale=yarn.attn_scale,
        window_size=(bm_size, 0),                                    # causal sliding window
    )
    y = y.view(B, T, self.num_heads, self.head_dim).contiguous().view(B, T, self.num_heads * self.head_dim)
    return F.linear(y, qkvo_w[self.dim * 3:].type_as(y))

# varlen packing + discrete window schedule:
#   train_max_seq_len = 2048           # clip each document -> more document diversity per fixed token budget
#   tokens/step = 393216; cu_seqlens padded to a fixed max-document count
#   ws_schedule = (3, 7, 11); get_ws(step); block_size = 128; warmup_iters 10 -> 30
```

This is the endpoint of the ladder. The bar it clears: the sparse-attention-gate record's 168,938 ms
(≈2.812 min) at val_loss ≤ 3.28 — the FA3 kernel speedup plus the clipping-driven document diversity bring
the record under it while the validated mean stays below 3.28.
