**Problem (from step 1).** The basic fused kernel is exact and IO-light but scheduled badly: every
config is slower than SDPA (`speedup_vs_sdpa` 0.94 / 0.94 / 0.81). The bottleneck has moved from HBM
traffic to non-matmul work in the inner loop — on Tensor-Core GPUs a non-matmul FLOP costs ~16x a
matmul FLOP, so the softmax/mask/scale operations dominate the time even though they are a small
fraction of FLOPs.

**Key idea (three forward-side, output-preserving levers).** (1) *Two-pass causal.* Split the key loop
into a first pass over blocks strictly below the diagonal (no mask) and a second pass over the single
diagonal-crossing block (mask applied), with the upper triangle never iterated — removing the
elementwise `tl.where` from the bulk of the work. (2) *Fuse the scale into Q.* Pre-multiply
`q <- q * (sm_scale * 1.44269504)` in the wrapper, so `Q K^T` comes out already in base-2 log units and
the inner loop is pure `exp2` with no per-element scale multiply; the kernel no longer takes `sm_scale`
at all. (3) *Per-headdim block sizes.* `128 x 64` for the small head dims (64, 128) to feed fuller
matmuls and amortize loop overhead, `64 x 64` for headdim 256 to stay within registers/shared memory.
Deferred normalization (divide once at the end) was already in the floor.

**Why it works.** All three levers move or remove non-matmul work without changing the output: the
two-pass split only skips entirely-masked blocks and omits the mask where nothing would be masked, the
fused scale is an algebraic identity, and the block-size choice rearranges where arithmetic happens,
not what. The output is still exactly `softmax(Q K^T) V`.

**What the harness does NOT expose (vs full FA2).** The published FA2 method also adds split-Q warp
partitioning (to kill the intra-block shared-memory reduction) and stores a log-sum-exp for the
backward pass. This task benchmarks only the forward pass, so there is no backward to feed an LSE, and
Triton's `tl.dot` schedules warps itself — so this fill is the three forward-side levers only.

**Hyperparameters.** `BLOCK_M, BLOCK_N = 128, 64` for `headdim <= 128`, else `64, 64`; scale +
`log2(e)` fused into `Q` in the wrapper; fp32 accumulators; `P` cast to `v.dtype` for `P @ V`; grid
`(cdiv(seqlen, BLOCK_M), batch * nheads)`.

**What to watch.** `hdim64_seq4k` should clearly beat the floor's 269.8 and be the first to cross
`speedup_vs_sdpa > 1.0`. `hdim256_seq16k` keeps `64 x 64`, so only the stripped loop helps — roughly
flat at batch 1. `hdim128_seq8k` is the risk: the `128 x 64` block at headdim 128 may spill registers
or starve occupancy and *regress below* the floor's 297.4 — the tell that block sizes must be searched,
not tabulated.

```python
@triton.jit
def _flash_v2_fwd(
    Q, K, V, Out,
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
    """FA2-style: two-pass causal, scale fused into Q, tuned block sizes."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q and pre-multiply scale (FA2: fuse scale into Q)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # --- Pass 1: non-causal blocks (all positions valid, skip masking) ---
    if IS_CAUSAL:
        causal_boundary = start_m * BLOCK_M
        non_causal_end = (causal_boundary // BLOCK_N) * BLOCK_N
    else:
        non_causal_end = seqlen
        causal_boundary = seqlen

    for start_n in range(0, non_causal_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # --- Pass 2: causal boundary blocks (need masking) ---
    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
    else:
        hi = non_causal_end

    for start_n in range(non_causal_end, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
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
    """FA2-style wrapper with per-headdim block sizes and fused scale."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    # FA2 optimization: fuse sm_scale into Q (saves one mul per element in inner loop)
    q = (q * (sm_scale * 1.44269504)).contiguous()
    o = torch.empty_like(q)
    # FA2: per-headdim block sizes for better tensor core utilization
    if headdim <= 64:
        BLOCK_M, BLOCK_N = 128, 64
    elif headdim <= 128:
        BLOCK_M, BLOCK_N = 128, 64
    else:
        BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    _flash_v2_fwd[grid](
        q, k, v, o,
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
