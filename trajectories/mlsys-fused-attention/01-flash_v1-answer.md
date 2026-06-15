**Problem.** Standard attention materializes the `N x N` score matrix `S = Q K^T` and the probability
matrix `P` in HBM, so it is memory-bound: its wall-clock is dominated by writing and re-reading those
`N x N` matrices across the slow HBM bus, not by the two matmuls. On the harness's long-sequence
configs that traffic is enormous, and approximate-attention tricks do not help (they cut FLOPs, the
wrong cost, and break the `max_diff < 1e-2` gate). The kernel must compute *exact* `softmax(Q K^T) V`
while removing the `N^2` HBM traffic.

**Key idea (the floor).** Tile `Q, K, V` and stream them through SRAM, never letting an `N x N` object
touch HBM. The numerically-stable softmax depends only on a running max `m` and normalizer `l`; the
online-normalizer identity combines block partials exactly by rescaling with `e^{m_old - m_new}`, so
the denominator — and, pushed through `* V`, the whole output — is built one key-block at a time. One
program owns one query block of one (batch, head); it loops over causal key blocks accumulating the
unnormalized output `acc` and `(m, l)` in fp32, and divides once at the end. Exact, `O(N^2 d)` FLOPs,
`O(N)` extra memory, `N^2` traffic gone.

**Why it is the weakest rung (deliberately).** This is the basic FA1 fill — correct and fused but with
no scheduling optimization. It (1) applies the elementwise causal mask `tl.where` at *every* iterated
key block, including the ~half strictly below the diagonal that need no mask; (2) multiplies the
softmax scale inside the loop rather than folding it into `Q` once at load; (3) uses one uniform block
shape `64 x 64` for every head dim, so it cannot trade larger blocks for fewer iterations on small
heads nor shrink them to relieve register pressure on `hdim256`. It is forward-only (no log-sum-exp,
no backward) because the harness benchmarks only the forward pass. Those un-stripped non-matmul
operations and the one-size block are exactly the headroom the next rungs take.

**Hyperparameters.** `BLOCK_M = BLOCK_N = 64` (uniform); fp32 accumulators (`m_i`, `l_i`, `acc`); `P`
cast to fp16 for the `P @ V` matmul; `exp2` with `log2(e) = 1.44269504`; grid
`(cdiv(seqlen, BLOCK_M), batch * nheads)`; `sm_scale` default `1/sqrt(headdim)`.

**What to watch.** Should clear `correct = 1` on all three configs (exact math) and land in the
few-hundred-TFLOPs band. Expect it to lag SDPA most on `hdim256_seq16k` — large head dim, batch 1, the
occupancy-starved, register-pressured corner where the uniform block and the unstripped inner loop
hurt most. The failure is a scheduling failure, which forces the next fill.

```python
@triton.jit
def _custom_attn_fwd(
    Q, K, V, Out,
    sm_scale,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """FA1-style: single-pass tiling + online softmax, causal mask every block."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else seqlen
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """FA1-style wrapper with uniform block sizes."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    _custom_attn_fwd[grid](
        q, k, v, o, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o
```
