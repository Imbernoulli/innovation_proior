The FA2 fill confirmed the worry I closed the previous rung on, in the sharpest possible way. Two configs moved as the cost model predicted: $\text{hdim64\_seq4k}$ jumped 269.8 $\to$ 308.3 TFLOPs and crossed $\text{speedup\_vs\_sdpa}$ from 0.94 to **1.09**, the first config to beat SDPA, and $\text{hdim256\_seq16k}$ ticked up modestly to 240.4, still under SDPA at 0.83 since it keeps $64 \times 64$. But $\text{hdim128\_seq8k}$ did the thing I flagged as the genuine coin toss and *regressed*: 297.4 down to **200.2**, $\text{speedup\_vs\_sdpa}$ collapsing 0.94 $\to$ 0.64 — a 33% throughput loss from one block-shape change, on the exact config where I said $128 \times 64$ at headdim 128 might spill or starve occupancy. The mechanism is specific: at headdim 128 a $\text{BLOCK\_M} = 128$ query tile is $128 \times 128$ fp16 resident in registers for the whole loop, plus a $128 \times 128$ fp32 accumulator, plus the running stats, plus the $128 \times 64$ score tile. The register file per SM is finite; when one program demands that many registers, either the compiler spills to local memory (HBM-backed — catastrophic, it reintroduces exactly the traffic I worked to remove) or occupancy drops because fewer programs fit per SM. At headdim 64 the same block is only $128 \times 64$, half the footprint, affordable. So the binding constraint, register pressure, is a joint function of $\text{BLOCK\_M}$, $\text{BLOCK\_N}$, and $\text{headdim}$ — no table indexed by head dim alone can be right. Block shape must be searched, not tabulated.

What I propose for this rung is the FA3-inspired fill: keep the FA2 algorithm exactly and autotune the schedule while adding software pipelining. I wrap the kernel in `@triton.autotune` over a list of eight `triton.Config` candidates, keyed on the parameters that change the right answer — $\text{seqlen}$, $\text{BLOCK\_DMODEL}$ (the head dim), and $\text{IS\_CAUSAL}$. At compile time Triton compiles each candidate, benchmarks it on the actual launch shape, and caches the fastest. Instead of betting on $128 \times 64$ for headdim 128 and losing, I hand the compiler a spread of block shapes — $128\times128$, $128\times64$, $64\times64$, $64\times128$, $128\times32$, $64\times32$ — and let it discover which one does not spill on this card. That alone closes the regression by construction: whatever it picks is no slower than the best candidate, and $64 \times 64$ — the floor's choice that scored 297.4 — is on the list, so the search floor already sits above the FA2 hole.

But block shape is not the only knob the chip exposes, and this is where the rung goes beyond merely fixing FA2's table. There is a second axis the FA2 loop left completely untouched: how it overlaps memory with compute. On Hopper the matmul is asynchronous — a warp can issue a Tensor-Core matmul and keep going while it runs — and there is a dedicated memory-copy unit that can prefetch the next tile from HBM into shared memory concurrently with the current tile's compute. My synchronous loop never asks for this: it loads $K_j, V_j$, waits, computes, loads $K_{j+1}$, waits. But loading is on the copy unit and computing is on the Tensor Cores — different units — so if the load of tile $j+1$ runs *while* the compute of tile $j$ is in flight, the loads hide under the compute and the Tensor Cores stop waiting on HBM at the top of every iteration. In hand-written Hopper code this is producer/consumer warp specialization with a multi-stage circular shared-memory buffer; I cannot write warpgroup barriers or TMA descriptors in this DSL. But Triton exposes the *principle* through a compiler knob, `num_stages`: marking a loop with $\text{num\_stages} > 1$ makes the compiler restructure it into a software pipeline that prefetches the next iteration's tile loads while the current iteration computes — precisely the producer/consumer-plus-circular-buffer idea expressed at the DSL level, with `num_stages` setting how many tiles ahead the prefetch runs. So I put `num_stages` (values 3 and 4) into the search, because the right depth trades prefetch coverage against register/shared-memory pressure — each extra stage needs another buffer slot resident — and that trade depends on the same regime variables as the block shape. One more knob in the same family is `num_warps`, the warps per program: more warps spread the work across more execution units and can raise occupancy but split the register file more ways, so I pair `num_warps` of 8 with the larger 128-row blocks and 4 with the smaller ones. The search is over the joint space $(\text{BLOCK\_M}, \text{BLOCK\_N}, \text{num\_stages}, \text{num\_warps})$, eight candidates, enough to cover the sensible corners without making compile time explode.

I want to be precise about what this is and is not, because the temptation is to oversell it. This is FA3-inspired in the exact sense that FA3's central contribution is overlapping the GEMMs with the softmax and the loads through Hopper's asynchrony — but FA3's *mechanism*, hand-written warp specialization with producer/consumer warpgroups, GMMA, TMA descriptors, FP8 matmul, and incoherent (Hadamard) processing for FP8 accuracy, is not accessible from Triton at all. What I can actually express is the algorithmic structure I already have — two-pass causal, scale fused into $Q$ in the wrapper, `exp2`, deferred normalization, all carried over verbatim from FA2 and all still exact — plus the compiler's `num_stages` software pipelining and `num_warps`, with the whole $(\text{block}, \text{stages}, \text{warps})$ space autotuned per launch shape. The $\text{num\_stages} > 1$ pipelining is the Triton-level shadow of FA3's load/compute overlap; it is real and it helps, but it is the compiler's automatic pipeline, not hand-scheduled warpgroup ping-pong, and I will not pretend it recovers FA3's full Hopper utilization. The FP8 path, the Hadamard rotation, the in-kernel $V$ transpose — none of that is in this fill, because none of it is expressible here, and the $\text{max\_diff}$ gate would punish a botched FP8 quantization anyway.

Correctness is untouched, and it must be, since every prior rung's wins were answer-unchanged. The autotuner only chooses *among* configurations of the *same* kernel body; it does not change what the kernel computes, and that body is the FA2 two-pass causal loop with the scale fused into $Q$ — exact, as established. `num_stages` software pipelining is a compiler transformation of loop *scheduling*: it reorders when loads and matmuls issue, not what they compute, and the online-softmax recurrence lands at the true softmax-weighted output regardless of how the iterations are scheduled, because each fold is an identity. `num_warps` changes how a program's tile work is divided across warps, and the rows are the axis of independence, so partitioning couples nothing. So the fill still returns $O = \mathrm{softmax}(Q K^\top) V$ exactly, clears $\text{max\_diff} < 10^{-2}$, and keeps $O(N^2 d)$ FLOPs with $O(N)$ extra memory — the only change from FA2 is that the compiler now picks the block shape, pipeline depth, and warp count per launch instead of me tabulating them.

The predictions are sharp, and since this is the strongest rung I want both the bar it must clear and an honest read of where it falls short. The regression must vanish: $\text{hdim128\_seq8k}$ was the FA2 disaster at 200.2, and autotuning over a list that includes $64 \times 64$ (the floor's 297.4 shape) plus the pipelined larger blocks should not merely recover but clear the floor, landing well above 297.4 — a result back near 400 would confirm the $\text{num\_stages}$ prefetch is paying off and the regression was purely the wrong fixed block. $\text{hdim64\_seq4k}$ should hold or extend its lead over 308.3 and stay above $1.0\times$ SDPA, since the search contains FA2's winning shape and adds pipelining on top. The honest limit is $\text{hdim256\_seq16k}$: at headdim 256, batch 1, every candidate block is register-starved (the head dim alone is 256-wide), so the autotuner is choosing among configurations that are all occupancy-bound, and $\text{num\_stages}$ pipelining costs registers it cannot spare. I expect it to plateau near 240, still under SDPA's $\sim$0.84$\times$. That plateau is the falsifiable claim defining what this kernel buys: autotuning plus software pipelining fixes the FA2 regression and pushes the small- and mid-head configs past SDPA, but it cannot close the large-head, batch-1 gap, because the mechanism that would close it — warp specialization, FP8, TMA — lives below the DSL. Within this harness's Triton surface there is no published lever clearly stronger than the autotuned, software-pipelined kernel this rung lands, so this is where the ladder honestly ends.

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
