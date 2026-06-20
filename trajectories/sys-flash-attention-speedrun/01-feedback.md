What this record changed: fused `QKᵀ`, softmax, and `P·V` into a single IO-aware tiled kernel with a streaming
(online) softmax, so the `N×N` score matrix never touches HBM; HBM traffic dropped from `O(N²)` to `O(N)` and
the backward recomputes the scores (saving only the per-row log-sum-exp) instead of storing `P`. The output is
bit-for-bit exact.

Measured numbers (higher speedup / throughput is better). PROVENANCE: the repo ships no machine-readable result
files for FlashAttention v1; these headline numbers are from the FlashAttention paper (arXiv:2205.14135), whose
*code* is faithfully reflected in the kernel excerpts above (`flash_attn/flash_attn_triton.py`,
`csrc/flash_attn/src/softmax.h`).

| comparison | figure | source |
|---|---|---|
| end-to-end GPT-2 training, seq. length 1K | **3× speedup** | FlashAttention paper, arXiv:2205.14135 (abstract) |
| BERT-large (seq. length 512) end-to-end vs. the MLPerf 1.1 training speed record | **15% wall-clock speedup** (a new speed record) | FlashAttention paper, arXiv:2205.14135 (abstract) |
| long-range arena (seq. length 1K–4K) | **2.4× speedup** | FlashAttention paper, arXiv:2205.14135 (abstract) |

The 3× on GPT-2 and the 15% over the MLPerf-1.1 BERT-large record are the headline figures for this rung; both
come from the paper rather than an in-repo benchmark file, because no such file exists. The mechanism behind the
numbers — eliminating the quadratic HBM round-trip of the score matrix — is exactly the fused, streaming-softmax
tiling in the code above, and it is what makes attention compute-bound and linear-memory rather than
bandwidth-bound and quadratic.
