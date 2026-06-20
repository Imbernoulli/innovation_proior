**Problem (from step 0).** The textbook three-pass attention writes the `N×N` score matrix `S = QKᵀ` to HBM, reads
it back for the softmax, writes `P`, and reads `P` back for `P·V` — `~4·N²` of HBM traffic for an intermediate
that is immediately discarded. The two matmuls are cheap on the tensor cores; the kernel is bandwidth-bound on
the quadratic score matrix, and the saved-for-backward `P` makes memory `O(N²)` too. Attention is slow because of
how it touches memory, not because of its FLOPs.

**Key idea — FlashAttention: IO-aware tiling + online (streaming) softmax + recompute-in-backward.** Fuse `QKᵀ`,
softmax, and `P·V` into one kernel and never write the `N×N` matrix to HBM. Tile the queries; loop over key/value
blocks; compute each score block in registers and fold it into the softmax with a **streaming** recurrence that
maintains a running max `m`, running normalizer `ℓ`, and running output `õ`, rescaling `õ` and `ℓ` by
`exp(m_old − m_new)` whenever a block raises the max. Sizing the tiles to fit in SRAM makes the fusion possible.
In the backward, save only the per-row log-sum-exp `L = m + log ℓ` (an `O(N)` vector) and **recompute** the
scores from `Q, K` instead of reading back a stored `P`.

**Why it works.** The online-softmax update is exact — `exp(xⱼ − m_new) = exp(xⱼ − m_old)·exp(m_old − m_new)`, so
rescaling the partial sums by a single scalar reassociates the softmax with no approximation; the output is
bit-for-bit the textbook result. Because the score block is born and consumed in SRAM/registers, HBM traffic drops
from `O(N²)` to `O(N)` (read Q, K, V once; write O once), so the kernel stops stalling on memory and the tensor
cores run the two GEMMs back-to-back. Recompute-in-backward trades cheap re-GEMM FLOPs for the avoided readback of
an `N×N` matrix — the right trade on a bandwidth-bound problem — keeping the backward linear in memory. Causal
masking lets a query block skip all key blocks past its diagonal, roughly halving the work.

**Change / code.** The online-softmax tiling loop (the readable Triton forward, `flash_attn/flash_attn_triton.py`):
each key/value block updates the running max, rescales the output accumulator, and adds its contribution; the row
is normalized once at the end and only the log-sum-exp is stored for the backward.

```python
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
        p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        v = tl.load(v_ptrs + start_n * stride_vn)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    tl.store(out_ptrs, acc_o)
```

The same online-softmax rescale, in the production CUDA kernel, is `Softmax::softmax_rescale_o` — for each new
key block it takes the new running max, rescales the running sum and the output accumulator by
`exp((m_prev − m_cur)·scale)`, then exponentiates the new scores against the new max (`csrc/flash_attn/src/softmax.h`):

```cpp
    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        if (Is_first) {
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };
```

The backward never reads back `P`: it recomputes the scores from `Q, K` and rebuilds the probabilities from the
saved `lse` (`flash_attn/flash_attn_triton.py`):

```python
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, k, trans_b=True)
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        # ... rebuild p = exp(qk * softmax_scale - lse_i) from the saved log-sum-exp, then accumulate dq, dk, dv ...
```
