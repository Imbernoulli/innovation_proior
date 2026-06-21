## Research question

Self-attention is the throughput bottleneck of a Transformer forward pass. For `Q, K, V` in `R^{N x d}`, the operation is `O = softmax(Q K^T) V` computed row-wise. A naive implementation materializes the `N x N` score matrix in HBM, so it is memory-bound and slow on long sequences. The goal is a **fused attention forward kernel**: a Triton `@triton.jit` kernel (plus Python launcher) that computes exact attention while maximizing achieved throughput (TFLOPs/s) on an H100, subject to a hard correctness constraint (max abs diff from PyTorch SDPA `< 1e-2`). Everything else — benchmark harness, FLOP accounting, reference, configs — is fixed. Each rung on the ladder fills the same kernel slot.

## Prior art / Background / Baselines

Attention has been accelerated along several axes.

- **Standard (materialized) attention.** Form `S = Q K^T` as an `N x N` matrix, write it to HBM, read it back to softmax into `P`, then read `P` and `V` to form `O = P V`. The operation is exact and moves `Theta(N d + N^2)` bytes across HBM; the `N^2` term dominates for `N` in the thousands with `d` only 64-256.

- **Approximate attention (Reformer, Linformer, Performer, Longformer, BigBird).** Sparsify or low-rank/kernel-approximate the score matrix to cut FLOPs to near-linear in `N`.

- **Online / lazy softmax (Milakov & Gimelshein; Rabe & Staats).** A numerically-stable softmax is fully determined by a running max `m` and normalizer `l`; the online-normalizer identity combines block partials exactly by rescaling with `e^{m_old - m_new}`, so the output can be built one block at a time with `O(1)` memory per query.

## Fixed substrate / Code framework

A benchmark harness is frozen and must not be touched: it builds `(batch, nheads, seqlen, headdim)` FP16 tensors for `Q, K, V` on H100, computes the PyTorch-SDPA reference, checks `max_diff < 1e-2`, then times both the custom kernel and SDPA (25 warmup, 100 timed runs, median latency). It reports, per config, `tflops` (achieved TFLOPs/s, primary, higher better), `latency_ms` (lower better), `max_diff`/`correct` (hard constraint), and `speedup_vs_sdpa` (SDPA latency / custom latency — the cross-GPU-comparable ratio). The causal FLOP count is fixed at `4 * batch * seqlen^2 * nheads * headdim / 2`. Available imports inside the editable region: `torch`, `triton`, `triton.language as tl`, `math`, `torch.nn.functional as F`.

## Editable interface

Exactly one region is editable — the Triton kernel `_custom_attn_fwd` and the Python wrapper `custom_attention_forward` in `flash-attention/custom_triton_bench.py` (lines 29-119). The contract the harness calls is the wrapper:

```
custom_attention_forward(q, k, v, causal=True, sm_scale=None) -> output
    q, k, v : (batch, nheads, seqlen, headdim), contiguous, FP16
    causal  : if True, key_pos <= query_pos
    sm_scale: softmax scale (default 1/sqrt(headdim))
    returns : (batch, nheads, seqlen, headdim), same dtype, == softmax(q k^T) v
```

Every method on the ladder replaces exactly this kernel + wrapper and nothing else: the kernel may be renamed, helper kernels may be defined, block sizes may be tuned, `@triton.autotune` may be used. The starting point is the scaffold below — a basic tiled kernel with online softmax: one program per query block per (batch, head), a single pass over all causal key blocks with the mask applied at every block, the scale folded inside the loop, uniform `BLOCK_M = BLOCK_N = 64`, online-softmax accumulators in fp32, the probability tile cast to fp16 for the `P @ V` matmul, and a single normalization at the end. This is forward-only (the harness benchmarks only the forward pass): no log-sum-exp statistic is stored and there is no backward kernel.

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
    """Fused self-attention forward kernel (default: basic flash attention)."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q tile [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    # Online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K/V tiles
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else seqlen
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Load K tile [BLOCK_N, BLOCK_DMODEL]
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        # S = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        # Load V tile and accumulate [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize and store
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """Python wrapper for the custom Triton attention kernel."""
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

## Evaluation settings

Three causal configurations, total tokens fixed at 16384, all FP16, causal, H100 80GB SXM5, seed 42:

| Config | Batch | SeqLen | Heads | HeadDim |
|---|---|---|---|---|
| `hdim64_seq4k` | 4 | 4096 | 32 | 64 |
| `hdim128_seq8k` | 2 | 8192 | 16 | 128 |
| `hdim256_seq16k` | 1 | 16384 | 8 | 256 |

Per config, four reported quantities: `tflops` (primary, higher better), `latency_ms` (lower better), `correct` (1 iff `max_diff < 1e-2`, a hard gate), and `speedup_vs_sdpa` (SDPA latency / custom latency). The three configs probe distinct regimes: small head dim with the largest batch (`hdim64_seq4k`), the middle ground (`hdim128_seq8k`), and a large head dim with batch 1 — the occupancy-starved, register-pressured corner (`hdim256_seq16k`).
