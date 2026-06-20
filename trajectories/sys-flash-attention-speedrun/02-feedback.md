What this record changed: same exact-attention math, re-partitioned. Parallelized over the sequence (query)
dimension so the grid fills the GPU even at small batch / long context; reduced the non-matmul FLOPs by deferring
the softmax `1/ℓ` normalization out of the inner loop (one division per row in the epilogue) and using base-2
`exp2`; and split the per-block work across warps along queries instead of keys, removing the cross-warp
reductions and `__syncthreads` from the hot loop. Output unchanged (bit-for-bit exact), memory still linear.

Measured numbers (higher speedup / TFLOPs/s / MFU is better). PROVENANCE: the repo ships no machine-readable
result files for this rung; the headline figures live in the repo's README text and changelog and in the
FlashAttention-2 paper (arXiv:2307.08691). The *code* (the sequence-parallel grid, the deferred normalization,
the query-split warp layout) is the in-repo kernel excerpted above.

| comparison | figure | source |
|---|---|---|
| FlashAttention-2 vs. FlashAttention v1 | **~2× faster** | README changelog "### 2.0: Complete rewrite, 2x faster"; FA2 paper arXiv:2307.08691 ("around 2× speedup compared to FlashAttention") |
| forward attention throughput, A100 | **50–73% of theoretical max FLOPs/s** (up to ~230 TFLOPs/s) | FA2 paper arXiv:2307.08691 (abstract) |
| end-to-end GPT-style training, per A100 | **up to 225 TFLOPs/s = 72% model FLOPs utilization** | README "reaching up to 225 TFLOPs/sec per A100, equivalent to 72% model FLOPs utilization"; FA2 paper arXiv:2307.08691 |

The headline for this rung is the README's "225 TFLOPs/sec per A100, 72% MFU" and the changelog's "2x faster"
over v1; the per-curve A100 sweeps (50–73% of peak across seqlens and head dims) are in the FA2 paper. The ~2× and
the jump to 72% MFU are the direct consequence of the three partitioning changes above — filling the SMs,
cutting the non-matmul stall, and deleting the in-loop cross-warp synchronization.
