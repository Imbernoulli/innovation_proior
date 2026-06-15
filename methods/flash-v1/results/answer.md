# FlashAttention: fast and memory-efficient exact attention

FlashAttention computes exact self-attention `O = softmax(Q K^T) V` (`Q, K, V in R^{N x d}`) as a
single fused GPU kernel that **never materializes the `N x N` score or probability matrix in HBM**.
It tiles `Q, K, V` into SRAM-sized blocks, combines per-block softmax partials with the online
(running-max) softmax recurrence, and accumulates the output in place. Attention is memory-bound,
so the win comes from cutting HBM accesses — not FLOPs — while keeping the same mathematical
attention map, up to ordinary floating-point differences.

## Problem it solves

Standard attention materializes `S = Q K^T` and `P = softmax(S)`, both `N x N`, to HBM, costing
`O(N^2)` memory and `Theta(Nd + N^2)` HBM accesses. Because softmax/masking/dropout are
memory-bound elementwise/reduction ops, the runtime is dominated by moving those `N x N` matrices
across the slow HBM bus, and the quadratic memory caps the usable sequence length. Approximate
attention cuts FLOPs but, optimizing the wrong quantity, often yields no wall-clock speedup and
changes the function. The goal: exact attention, faster and lower-memory, on real GPUs.

## Key idea

Make attention **IO-aware** — minimize reads/writes between HBM and the small/fast SRAM, and never
put an `N x N` object in HBM. Two techniques:

1. **Tiling + online softmax.** Split `Q` into row-blocks of `B_r`, and `K, V` into row-blocks of
   `B_c`. In the IO-counting algorithm, load one key/value block and sweep the query blocks,
   computing each `B_r x B_c` score block on-chip and folding it into running per-row state — the
   max `m`, the normalizer `l`, and the output accumulator. The online-softmax identity
   `m_new = max(m, m~)`, `l_new = e^{m - m_new} l + e^{m~ - m_new} l~` (Milakov & Gimelshein 2018),
   extended to rescale the output accumulator by `e^{m - m_new}`, combines the block partials into
   the exact result. Defer the single `1/l` normalization to the end.
2. **Recomputation in the backward pass.** Store only `O` and row softmax statistics `(m, l)` or
   an equivalent log-sum-exp value (plus the dropout RNG state), not `S`/`P`. In the backward pass,
   recompute `S, P` on-chip from `Q, K, V` blocks. This is selective gradient checkpointing, but
   because the saved cost is HBM traffic (not FLOPs), it makes the backward pass *faster*, not a
   memory-for-speed trade — which is what lets the whole forward pass fuse into one kernel with
   masking/dropout free on-chip.

## Forward algorithm (exact)

Block sizes `B_c = ceil(M / 4d)`, `B_r = min(ceil(M / 4d), d)` for SRAM size `M` (the 4 accounts
for `Q, K, V, O` blocks sharing SRAM; cap `B_r <= d` so the `B_r x B_c` score block fits).
Initialize `O = 0`, `l = 0`, `m = -inf`. In the IO-counting order, for each key/value block `j`,
sweep query blocks `i`:

```
S_ij   = tau * Q_i K_j^T                                  # tau = 1/sqrt(d); on chip
m~     = rowmax(S_ij);   P~ = exp(S_ij - m~);   l~ = rowsum(P~)
m_new  = max(m_i, m~)
l_new  = e^{m_i - m_new} * l_i + e^{m~ - m_new} * l~
O_i   <- diag(l_new)^{-1} ( diag(l_i) e^{m_i - m_new} O_i + e^{m~ - m_new} P~ V_j )
m_i, l_i <- m_new, l_new
```

(In the practical kernel, parallelize over query blocks — one program per query block, loop over
K/V inside — keep an unnormalized accumulator `acc`, rescale `acc *= e^{m_i - m_new}` each block,
and divide by `l_i` once at the end.)

**Correctness (induction on key-blocks `j`).** Invariant after `j` blocks: `m^(j) =
rowmax(S_{:,:j})`, `l^(j) = rowsum(exp(S_{:,:j} - m^(j)))`, `O^(j) = softmax(S_{:,:j}) V_{:j}`.
Base `j=0` trivial. Step: the `m, l` updates are the online-softmax identity; substituting
`O^(j) = diag(l^(j))^{-1} exp(S_{:,:j} - m^(j)) V_{:j}` into the `O` update, the `diag(l^(j))`
cancels and both base-shifts `e^{m^(j) - m^(j+1)} exp(S - m^(j)) = exp(S - m^(j+1))` collapse to
`O^(j+1) = softmax(S_{:,:j+1}) V_{:j+1}`. At `j = T_c`, `O = softmax(QK^T) V`. FLOPs `O(N^2 d)`,
extra memory `O(N)`.

## IO complexity

With `B_c = Theta(M/d)`, `B_r = Theta(min(M/d, d))`, the number of passes over `Q, O` is
`T_c = N/B_c = Theta(Nd/M)`; `K, V` are loaded once. HBM accesses:

```
Theta(Nd * T_c) = Theta(N^2 d^2 / M)   vs   standard  Theta(Nd + N^2).
```

Since `d^2 << M` for typical `d` (64-128), `M` (~100 KB), this is many-fold fewer HBM accesses —
the source of the speedup. The backward pass has the same `Theta(N^2 d^2 / M)`.

**Lower bound.** No exact attention algorithm achieves `o(N^2 d^2 / M)` HBM accesses for all
`M in [d, Nd]`: at `M = Theta(Nd)` that is `o(Nd)`, but inputs+output of size `Nd` in HBM force
`Omega(Nd)`. So FlashAttention is IO-optimal up to constants in this uniform-over-`M` sense.

## Backward pass (exact, recomputed)

With scaled scores `S_ij = tau q_i.k_j`, `P_ij = e^{S_ij}/L_i`, and
`o_i = sum_j P_ij v_j`:

```
dV: dv_j = sum_i P_ij do_i                 (dV = P^T dO)
dP_ij = do_i . v_j                         (dP = dO V^T)
D_i   = P_{i:}^T dP_{i:} = do_i . o_i       (length-N reduction -> length-d dot product)
dS_ij = P_ij (dP_ij - D_i)                  (softmax Jacobian diag(P) - P P^T)
dQ: dq_i = tau * sum_j dS_ij k_j
dK: dk_j = tau * sum_i dS_ij q_i
```

Recompute `P_ij` on-chip from `Q, K, V` blocks and saved row statistics `(m, l)` or LSE; regenerate
the dropout mask from the saved RNG state. In the `exp2` kernel below, `Lse` is stored as the
base-2 row log-sum-exp `m_i + log2(l_i)`, matching the base-2 score units. Block-wise, `O(N)`
extra memory.

## Working code (fused forward kernel, Triton)

One program per query block, loop over K/V, online softmax into an fp32 accumulator, single
normalization at the end, and store one row log-sum-exp buffer for backward recomputation. The
practical kernel stores scores in base-2 units by multiplying the softmax scale by
`log2(e) = 1.44269504`, then uses `exp2(x) = 2^x` instead of `exp`; causal masking skips
fully-masked key-blocks and masks only the boundary block; fp32 accumulate, fp16 tensor-core
matmul.

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out, Lse,
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
    start_m = tl.program_id(0)          # this program owns BLOCK_M query rows
    off_hz = tl.program_id(1)           # for one (batch, head)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh
    lse_offset = off_hz * seqlen

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # load the query block once; keep it on chip for the whole loop
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    qk_scale = sm_scale * 1.44269504                         # log2(e), for exp2 softmax

    # running online-softmax state (fp32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running normalizer
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # running UNnormalized output

    hi = tl.minimum((start_m + 1) * BLOCK_M, seqlen) if IS_CAUSAL else seqlen
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # S_ij = Q_i K_j^T * scale * log2(e), on chip (never written to HBM)
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where((start_n + offs_n)[None, :] < seqlen, qk, float("-inf"))
        if IS_CAUSAL:                                            # boundary-block mask only
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)                       # 2^{m_old - m_new}
        p = tl.math.exp2(qk - m_new[:, None])                   # 2^{S - m_new}
        l_i = l_i * alpha + tl.sum(p, axis=1)                    # rescale + add block normalizer
        acc = acc * alpha[:, None]                               # rescale running output
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)                       # acc += P~ V_j
        m_i = m_new

    acc = acc / l_i[:, None]                                     # single normalization at the end
    lse_ptrs = Lse + lse_offset + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < seqlen)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < seqlen)


def attention_forward(q, k, v, causal=True, sm_scale=None):
    """Fused exact attention forward. q,k,v: (batch, nheads, seqlen, headdim), FP16, contiguous."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    lse = torch.empty((batch * nheads, seqlen), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 64, 64                                    # tiles sized to fit SRAM
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)        # one program per query block
    _attn_fwd[grid](
        q, k, v, o, lse, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o, lse
```

Besides the input tile loads, the output `O` and one `O(N)` row-statistics buffer cross the HBM bus
at the end; the score and probability matrices never do.

## Relation to prior work

- **Online softmax** (Milakov & Gimelshein 2018): the running-max/normalizer recurrence that makes
  the per-block softmax combine exactly; FlashAttention extends the rescaling to the output
  accumulator and runs it across tiles.
- **Lazy-softmax / memory-efficient attention** (Rabe & Staats 2021; Jang et al. 2019): exact,
  reduces memory *footprint* via deferred division + running max, but leaves HBM *accesses*
  quadratic (so ~same speed), keeps a separate partial output per block, and uses generic gradient
  checkpointing in the backward pass. FlashAttention instead targets HBM accesses, updates a single
  output in place, and derives the backward pass analytically so recomputation is a strict win.
- **IO-complexity model** (Aggarwal & Vitter 1988) and the roofline / arithmetic-intensity view
  (Williams et al. 2009): the framing that counts HBM accesses, which is the metric FlashAttention
  optimizes and proves optimal over a range of SRAM sizes.
