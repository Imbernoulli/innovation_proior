**Problem (from step 2).** The FA2 fill's fixed per-headdim block table backfired: `128 x 64` at
headdim 128 spilled registers / starved occupancy, regressing `hdim128_seq8k` from 297.4 to 200.2
TFLOPs (speedup 0.94 -> 0.64). The binding constraint is register pressure, a joint function of
`BLOCK_M`, `BLOCK_N`, and `headdim` — no table indexed by head dim alone can be right. Block shape must
be searched, not tabulated.

**Key idea (autotune the schedule, add software pipelining).** Wrap the kernel in `@triton.autotune`
over eight `(BLOCK_M, BLOCK_N, num_stages, num_warps)` candidates keyed on
`(seqlen, BLOCK_DMODEL, IS_CAUSAL)`; the compiler compiles each, benchmarks it on the real launch
shape, and caches the fastest. Crucially this also searches `num_stages` (3, 4): Triton's
`num_stages > 1` is software pipelining — it prefetches the next iteration's K/V loads while the
current iteration computes, the DSL-level shadow of Hopper's async load/compute overlap (FA3's central
idea). The exact algorithm (two-pass causal, scale fused into Q in the wrapper, `exp2`, deferred
normalization) is carried over verbatim from FA2.

**Why it works.** The autotuner chooses only among configurations of the same exact kernel body, so the
output is unchanged; it picks the block shape that does not spill on each config (closing the FA2
regression, since `64 x 64` is a candidate) and the pipeline depth that overlaps loads with compute (a
real new win the floor never had).

**What the harness does NOT expose (vs full FA3).** FA3's mechanism — hand-written warp-specialized
producer/consumer warpgroups, GMMA, TMA descriptors, FP8 matmul, incoherent (Hadamard) processing for
FP8 accuracy — is not accessible from Triton; the task description says so. This fill expresses only
the algorithmic structure plus the compiler's `num_stages`/`num_warps` knobs. The `num_stages`
pipelining is the automatic compiler pipeline, not hand-scheduled warpgroup ping-pong, and does not
recover FA3's full Hopper utilization.

**Hyperparameters.** Autotune configs: `{128x128/s3/w8, 128x64/s3/w8, 128x64/s4/w8, 64x64/s3/w4,
64x64/s4/w8, 64x128/s3/w8, 128x32/s3/w4, 64x32/s4/w4}`; key `['seqlen','BLOCK_DMODEL','IS_CAUSAL']`;
scale + `log2(e)` fused into `Q`; grid `lambda META: (cdiv(seqlen, META['BLOCK_M']), batch*nheads)`.

**What to watch.** The `hdim128_seq8k` regression should vanish and clear the floor's 297.4 (a result
near 400 would confirm the pipelining pays off); `hdim64_seq4k` should hold above `1.0x` SDPA. The
honest limit: `hdim256_seq16k` (headdim 256, batch 1) is register-starved for every candidate, so
expect a plateau near 240, still under SDPA — the gap that only the below-DSL FA3 levers (warp
specialization, FP8, TMA) could close. Within this Triton surface no published lever is clearly
stronger, so the ladder ends here.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
    ],
    key=['seqlen', 'BLOCK_DMODEL', 'IS_CAUSAL'],
)
@triton.jit
def _flash_v3_fwd(
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
    """FA3-inspired: autotuned two-pass causal with software pipelining."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q with scale already fused (done in wrapper)
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # --- Pass 1: non-causal blocks (no mask, better pipelining) ---
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

    # --- Pass 2: causal boundary blocks ---
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
    """FA3-inspired: autotuned pipelining + fused scale + two-pass causal."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    # Fuse scale into Q
    q = (q * (sm_scale * 1.44269504)).contiguous()
    o = torch.empty_like(q)
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch * nheads)
    _flash_v3_fwd[grid](
        q, k, v, o,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o
```
