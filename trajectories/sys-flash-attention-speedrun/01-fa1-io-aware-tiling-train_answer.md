I start from a correct but slow attention kernel, and the first thing I do is count bytes rather than FLOPs. The math is $O = \mathrm{softmax}(QK^\top/\sqrt d)\,V$, two matmuls the tensor cores should devour at hundreds of TFLOPs/s — so if attention is slow, the matmuls are not the bottleneck. The textbook recipe runs three passes: a GEMM forms the $N\times N$ score matrix $S = QK^\top$ and writes it to HBM, a softmax kernel reads $S$ back and writes $P = \mathrm{softmax}(S)$, and a second GEMM reads $P$ back to produce $O = P\,V$. The matmul work is $\sim\!4N^2d$ FLOPs, but the HBM traffic for the score matrix alone is write $S$, read $S$, write $P$, read $P$ — $\sim\!4N^2$ elements shuttled across the slow memory just to move an intermediate that is immediately discarded. For $N=8\mathrm{K}$ in FP16 that intermediate is 128 MB *per head*. The tensor cores idle while the kernel waits on HBM: attention is memory-bound, and the thing flooding the bus is the materialized score matrix. The backward makes it worse, because the obvious implementation stashes the same $N\times N$ $P$ to reuse, so the activation footprint is $O(N^2)$ too.

The cure is to never write $S$. I propose **FlashAttention**: fuse $QK^\top$, the softmax, and $P\,V$ into a single IO-aware tiled kernel, compute each score block where it is born and consumed — in registers and shared memory — and keep the whole thing exact and linear in memory. The obstacle is the softmax in the middle. For a row of scores $x_1\dots x_N$ I want $o = \sum_j \big(e^{x_j-m}/\ell\big)\,v_j$ with $m=\max_j x_j$ and $\ell=\sum_j e^{x_j-m}$; the max is there only so $\exp$ does not overflow, but both $m$ and $\ell$ are *global* reductions over the entire row. If I have to see all $N$ scores before normalizing the first one, I am forced to hold a full row resident, which is exactly what forces materialization.

What makes the fusion possible is an **online (streaming) softmax** — a recurrence that maintains running statistics and never needs the whole row at once. Process the keys block by block, carrying a running max $m^{(t)}$, a running normalizer $\ell^{(t)}=\sum e^{x_j-m^{(t)}}$ over keys seen so far, and a running unnormalized output $\tilde o^{(t)}=\sum e^{x_j-m^{(t)}}\,v_j$. When a new block arrives with a larger local max, the running max climbs to $m^{(t+1)}=\max(m^{(t)},m_{\text{local}})$, and everything accumulated against the old max is too large by exactly $e^{m^{(t)}-m^{(t+1)}}$. Because $\exp$ of a difference factors cleanly, $e^{x_j-m^{(t+1)}}=e^{x_j-m^{(t)}}\cdot e^{m^{(t)}-m^{(t+1)}}$, I can correct the old accumulators with a single scalar multiply. With $\alpha = e^{m^{(t)}-m^{(t+1)}}\in(0,1]$,
$$\ell^{(t+1)} = \alpha\,\ell^{(t)} + \!\!\sum_{j\in\text{block}}\!\! e^{x_j-m^{(t+1)}}, \qquad \tilde o^{(t+1)} = \alpha\,\tilde o^{(t)} + \!\!\sum_{j\in\text{block}}\!\! e^{x_j-m^{(t+1)}}\,v_j.$$
After the last block the answer is $\tilde o_{\text{final}}/\ell_{\text{final}}$. Since the max only ever rises, $\alpha\le 1$ and nothing overflows, and the recurrence is algebraically *identical* to the whole-row softmax — just reassociated — so the output is bit-for-bit the textbook result, not an approximation.

That dissolves the obstacle and the kernel structure follows. Tile the queries: each threadblock owns a block of $B_r$ query rows, loads that $Q$ tile into SRAM where it stays, and loops over key/value blocks of $B_c$. Per block it loads $K_j, V_j$, computes the score tile $S_{ij}=Q_iK_j^\top$ as a small tensor-core GEMM living in registers, takes the block's row maxima, performs the online-softmax update of $(m,\ell)$ and the rescale of $\tilde o$, then accumulates $\tilde o\mathrel{+}= P_{ij}V_j$. The load-bearing design choice is *IO-awareness*: I pick $B_r, B_c$ so that $Q_i$, $K_j$, $V_j$ and the $B_r\times B_c$ score block all fit in the SM's shared memory and registers at once — sizing the tiles to the real memory hierarchy rather than pretending memory is flat. Every byte of $K$ and $V$ is read from HBM exactly once, reused across all queries in the block, and dropped; the $N\times N$ matrix is never written anywhere. The surviving HBM traffic is read $Q,K,V$ once and write $O$ once — $O(Nd)$, linear in $N$ — while the matmul FLOPs are unchanged but now run back-to-back out of SRAM with no stall, so the kernel is finally compute-bound.

The backward is where materialization wants to creep back in, and I refuse to save $P$. To rebuild the probabilities I need only the scores and one scalar per row — the log-sum-exp $L = m + \log\ell$, since $P_{ij}=e^{S_{ij}-L_i}$. So the forward stashes just that $O(N)$ vector. In the backward I *recompute* the scores $S_{ij}=Q_iK_j^\top$ tile by tile, the same fused tiled GEMM as the forward, and recover $P_{ij}=e^{S_{ij}-L_i}$ on the fly. This trades a cheap re-GEMM against the avoided readback of an $N\times N$ matrix — exactly the right trade on a bandwidth-bound problem — and keeps the backward linear in memory too. Causal masking is then a free bonus: a query block whose rows all precede a key block can skip that block entirely, so the key loop only runs up to the diagonal and causal attention does roughly half the work with no materialized mask.

The online-softmax tiling loop, the readable Triton forward (`flash_attn/flash_attn_triton.py`): each key/value block updates the running max, rescales the output accumulator, and adds its contribution; the row is normalized once at the end and only the log-sum-exp is stored for the backward.

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

The same online-softmax rescale, in the production CUDA kernel, is `Softmax::softmax_rescale_o` — for each new key block it takes the new running max, rescales the running sum and the output accumulator by `exp((m_prev − m_cur)·scale)`, then exponentiates the new scores against the new max (`csrc/flash_attn/src/softmax.h`):

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

The backward never reads back `P`: it recomputes the scores from `Q, K` and rebuilds the probabilities from the saved `lse` (`flash_attn/flash_attn_triton.py`):

```python
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, k, trans_b=True)
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        # ... rebuild p = exp(qk * softmax_scale - lse_i) from the saved log-sum-exp, then accumulate dq, dk, dv ...
```
