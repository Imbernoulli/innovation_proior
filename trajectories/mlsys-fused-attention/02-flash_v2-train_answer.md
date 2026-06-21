The basic fused kernel did exactly its one job and then stalled. It cleared $\text{correct} = 1$ on all three configs and landed where a correct-but-untuned fused kernel lands — $\text{hdim64\_seq4k}$ 269.8 TFLOPs, $\text{hdim128\_seq8k}$ 297.4, $\text{hdim256\_seq16k}$ 234.2 — but the cross-GPU-comparable column is brutal: $\text{speedup\_vs\_sdpa}$ of 0.94, 0.94, 0.81, every config slower than PyTorch SDPA, worst in exactly the corner I flagged. So the bottleneck has moved: this is no longer a memory problem (the $N^2$ traffic is gone) and not a correctness problem (the math is exact) — it is a scheduling problem. The kernel is fused and IO-light but it leaves the Tensor Cores idle, and SDPA's fused kernel does not. The cost model says why. The H100 does on the order of 990 TFLOPs/s of FP16 matmul on its Tensor Cores but only $\sim$67 TFLOPs/s of FP32 on the regular ALUs that run the softmax — a non-matmul FLOP costs roughly 16 matmul-FLOPs of wall-clock. In attention the matmuls are $Q K^\top$ and $P V$; everything else — the exponentials, row-maxes, rescalings, divisions, mask comparisons — is non-matmul, small by FLOP count but, at $\sim$16$\times$ each, large by time. The floor's inner loop is full of exactly this work.

What I propose is FlashAttention v2: the same exact output, with three forward-side, output-preserving levers that strip the non-matmul work out of the hot loop and size the blocks to the regime. The first and biggest is a two-pass causal split. The floor applies the elementwise mask `tl.where(row >= col, qk, -inf)` at every iterated key block, but since I already work in blocks I can reason at the block level. Fix a query row block covering rows $[\text{start\_m}\cdot\text{BLOCK\_M},\ (\text{start\_m}+1)\cdot\text{BLOCK\_M})$; a key block covers columns $[\text{start\_n},\ \text{start\_n}+\text{BLOCK\_N})$. Three kinds exist. A block entirely above the diagonal masks to zero and contributes nothing — already skipped by bounding $\text{hi} = (\text{start\_m}+1)\cdot\text{BLOCK\_M}$. A block entirely below the diagonal has nothing masked, so its `tl.where` is pure wasted work. Only the block straddling the diagonal actually needs the elementwise mask, and for square blocks that is exactly one block per row block. So I split the column loop into two passes: a first pass over the strictly-below-diagonal blocks with no mask at all, and a second pass over just the boundary block with `tl.where`. The split point is

$$\text{non\_causal\_end} = \left\lfloor \frac{\text{start\_m}\cdot\text{BLOCK\_M}}{\text{BLOCK\_N}} \right\rfloor \cdot \text{BLOCK\_N},$$

rounding the first query row of this tile down to a key-block boundary; everything before it is safely below the diagonal. On a long causal sequence the boundary block is one of many, so this removes essentially all the masking cost.

The second lever folds the scale. The floor multiplies `qk = tl.dot(q, trans(k)) * sm_scale` inside the loop — one multiply per element of every score tile, on the slow side — but `sm_scale` is constant for the whole kernel, and so is the $\log_2 e = 1.44269504$ factor the `exp2` softmax needs. So I fold them together and push them onto $Q$ at load time, in the wrapper: $q \leftarrow q \cdot (\text{sm\_scale} \cdot 1.44269504)$. Then $Q K^\top$ already comes out in base-2 log units, the inner loop carries no per-element scale multiply at all, and the running-max comparison and subtraction happen consistently in log2 units — the per-element multiply over the $N \times N$ score matrix collapses to one multiply over the $\text{BLOCK\_M} \times d$ query tile, done once. This even changes the kernel signature: with the scale fully fused into $Q$, the kernel no longer needs `sm_scale` as a parameter, and the inner loop is pure `exp2` on an already-scaled `qk`.

The third lever sizes the blocks per head dim. The floor's uniform $64 \times 64$ cannot be right across the three regimes. Bigger blocks mean fewer loop iterations — fewer per-element rescales, fewer loads per output element — but need more registers and shared memory to hold the tiles, and past a point the register file spills (catastrophic) or the kernel will not launch. The sweet spot depends on the head dim, because $d$ sets how wide each tile is. For the small head dims (64 and 128) I can afford $\text{BLOCK\_M} = 128$ with $\text{BLOCK\_N} = 64$; for the large head dim (256) I stay at $64 \times 64$ to fit registers and shared memory. The larger row blocks amortize loop overhead and feed the Tensor Cores fuller $Q K^\top$ and $P V$ matmuls where the head dim leaves register room.

Every move is justified as answer-unchanged, and the $\text{max\_diff}$ gate is unforgiving, so the exactness has to hold on each one. The deferred normalization (divide once at the end) was already banked at the floor, proved exact by the telescoping online-max corrections; it is not new here. The fused scale is an algebraic identity — multiplying $Q$ by a constant before $Q K^\top$ is the same as multiplying the scores after, and folding $\log_2 e$ in is just $e^x = 2^{x \log_2 e}$. The two-pass split only skips blocks that are entirely masked (they contribute zero) and only omits the mask on blocks where no element would be masked, so it computes the same thing as masking everywhere — the boundary block still gets the full `tl.where`. The per-headdim sizes rearrange where arithmetic happens, not what. So the kernel still returns $O = \mathrm{softmax}(Q K^\top) V$ exactly, in $O(N^2 d)$ FLOPs with $O(N)$ extra memory, with the non-matmul work pulled out of the hot loop. One honest caveat on scope: the published FA2 method also adds split-Q warp partitioning (to kill the intra-block shared-memory reduction) and stores a log-sum-exp for the backward pass, but this harness exposes neither — it benchmarks only the forward, so there is no backward to feed an LSE, and Triton's `tl.dot` schedules the warps itself rather than letting me hand-partition them. So this fill is exactly the three forward-side levers.

The cost model makes specific, falsifiable predictions. The masking-split and fused-scale wins scale with the number of blocks, so they should help most where the loop is longest, and the per-headdim block change helps only where it actually changes the block — $\text{hdim64}$ and $\text{hdim128}$ (which go to $128 \times 64$), and not at all for $\text{hdim256}$ (which stays $64 \times 64$). So I expect $\text{hdim64\_seq4k}$ to clearly beat 269.8 — bigger row block and stripped inner loop both pulling up — and to be the first config to cross $\text{speedup\_vs\_sdpa} > 1.0$, since it was already closest at 0.94. $\text{hdim256\_seq16k}$ keeps $64 \times 64$, so only the stripped loop helps and at batch 1 it stays roughly flat. The genuine coin toss is $\text{hdim128\_seq8k}$: moving to $\text{BLOCK\_M} = 128$ at a head dim already twice as wide as $\text{hdim64}$ doubles the register footprint of the $Q$ tile and the accumulator. If that tips the kernel into spilling — or just lowers occupancy because each program hogs more of the register file — the $128 \times 64$ choice could regress below the floor's 297.4. The two effects fight: leaner loop (faster) versus heavier block (lower occupancy / possible spill, slower), and the cost model alone cannot call it on this card. If that regression shows up, the diagnosis for the next rung is already written: block-size selection cannot be a fixed per-headdim table; it has to be searched per regime, along with the pipeline depth and warp count, so the kernel finds the configuration that does not spill instead of betting on one table that happens to be wrong for the middle case.

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
