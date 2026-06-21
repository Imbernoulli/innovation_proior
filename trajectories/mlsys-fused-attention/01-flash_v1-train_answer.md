Attention is the throughput bottleneck of the Transformer forward pass, and the reason is not arithmetic but traffic. The map I have to compute is $O = \mathrm{softmax}(Q K^\top) V$ for $Q, K, V \in \mathbb{R}^{N \times d}$, and the obvious implementation forms the score matrix $S = Q K^\top$ — an $N \times N$ object — writes it to HBM, reads it back to softmax row-wise into $P$, writes $P$, then reads $P$ and $V$ to form $O$. For the configs the harness actually runs (seqlen 4096, 8192, 16384 with $d$ only 64, 128, 256) that $N \times N$ matrix dwarfs everything else, and I pay for it twice — write then read — and again for $P$. The two matmuls $Q K^\top$ and $P V$ are dense and the H100's Tensor Cores devour them; the softmax is a row-max, a row-sum, an elementwise exponential and a divide, almost no arithmetic per number, just a flood of numbers streamed through. With HBM at $\sim$3.35 TB/s against on-chip SRAM an order of magnitude faster but only a couple hundred kilobytes per SM, the wall-clock of standard attention is set by shuttling $S$ and $P$ across the slow bus, not by the GEMMs. That reframes the whole task: I am not trying to do fewer operations, I am trying to do fewer reads and writes to HBM. This is exactly why the approximate-attention lineage — Reformer's hashing (Kitaev et al. 2020), Linformer, Performer, Longformer, BigBird — does not help here: it cuts FLOPs, which were never the bottleneck, so its measured wall-clock is often no better than dense attention at these lengths, and worse, it changes the function, which the harness's hard $\text{max\_diff} < 10^{-2}$ gate would reject outright. I want the function to stay exact and the speedup to come from removing the $N^2$ traffic.

What I propose for this rung is the basic tiled, IO-aware fused kernel — FlashAttention v1 — that never lets an $N \times N$ object touch HBM. The one obstacle between me and that is softmax's column coupling: the denominator of row $i$ is $\sum_j e^{S_{ij}}$, a sum over all $N$ keys, so naively I need the whole row before I can normalize any entry. (The other classic obstacle, storing $S$ and $P$ for the backward pass, does not bite — the harness benchmarks only the forward, so I build a forward-only kernel and never recompute anything for gradients.) The seed of the answer is already inside the numerically stable softmax: I never form $e^{x_i}$ directly (it overflows past $x \approx 89$); I subtract the row max, $m = \max_i x_i$, take $f_i = e^{x_i - m}$, $l = \sum_i f_i$, and $\mathrm{softmax}_i = f_i / l$. Softmax is therefore fully determined by two scalars per row, the max $m$ and the normalizer $l$, plus the raw scores. The question is whether I can build $m$ and $l$ incrementally, seeing the scores a block at a time. I can, by the online-normalizer identity (Milakov & Gimelshein 2018). Suppose I have processed one chunk with local max $m^{(1)}$ and $l^{(1)} = \sum e^{x - m^{(1)}}$, and a second arrives with $m^{(2)}, l^{(2)}$. The combined max is $m = \max(m^{(1)}, m^{(2)})$; the two normalizers were each taken relative to their own local max so they cannot be added directly, but each rescales exactly, since $e^{x - m} = e^{x - m^{(k)}} e^{m^{(k)} - m}$, giving

$$l = e^{m^{(1)} - m}\, l^{(1)} + e^{m^{(2)} - m}\, l^{(2)},$$

exact, and with both exponents $\le 0$ the rescaling factors lie in $(0, 1]$, so still no overflow. I push this through the $\cdot\, V$: keep a running unnormalized output accumulator $\text{acc}$ and a running normalizer $l$, both starting at zero, with running max $m = -\infty$; as each key/value block arrives, exponentiate its scores relative to $m$, add $e^{\text{score} - m} V_{\text{block}}$ into $\text{acc}$ and $e^{\text{score} - m}$ into $l$, and whenever the max grows, multiply both $\text{acc}$ and $l$ by $\alpha = e^{m_{\text{old}} - m_{\text{new}}}$ before folding in the new block. Divide $\text{acc}$ by $l$ once at the very end. That is exact attention with $O(1)$ extra state per query — the deferred-normalization trick that makes the whole pipeline a single kernel.

That forces the shape of the algorithm. The only place an $N \times N$ thing may legally exist is on-chip, transiently, so I tile everything: split $Q$ into row-blocks of $\text{BLOCK\_M}$ queries, $K, V$ into row-blocks of $\text{BLOCK\_N}$, and a single score block $Q_i K_j^\top$ is $\text{BLOCK\_M} \times \text{BLOCK\_N}$, sized to fit SRAM. Correctness is an induction: after processing $j$ key-blocks the running $(m, l, \text{acc})$ equals the true max, normalizer, and unnormalized output over the first $j$ blocks; the base case is the empty sum, each step is the online identity, so at the last block $\text{acc} / l$ is exactly $\mathrm{softmax}(Q_i K^\top) V$. The FLOP count stays $O(N^2 d)$ — the same arithmetic as dense attention, as it must be for an exact method — while the HBM traffic drops to the inputs, the output, and the $O(N)$ running stats. The $N^2$ traffic is gone by construction; that is the entire source of the speedup.

A few practical decisions pin down the kernel. The unit of parallelism is the query block: one program per $(\text{start\_m}, \text{batch}\cdot\text{head})$, looping over key/value blocks inside, so each program owns its query rows entirely, keeps $m_i, l_i, \text{acc}$ resident for its whole life, and writes $O_i$ exactly once — the launch grid is $(\lceil \text{seqlen}/\text{BLOCK\_M} \rceil,\ \text{batch}\cdot\text{nheads})$. For the exponential I use the hardware's fast base-2 special function: since $e^x = 2^{x \log_2 e}$ with $\log_2 e = 1.44269504$, I keep the softmax in base-2 units, $\alpha = \text{exp2}((m_i - m_{\text{new}})\cdot 1.44269504)$ and $p = \text{exp2}((qk - m_{\text{new}})\cdot 1.44269504)$ — the same softmax expressed in the units the fast `exp2` wants. I keep the scores, running max, normalizer, and output accumulator in fp32 (the sums and rescalings need the dynamic range, and fp16 accumulation over many blocks would risk the $\text{max\_diff}$ gate), with inputs and output in fp16; I cast the probability tile $p$ back to fp16 before the $P V$ matmul so it runs on the Tensor Cores at full rate.

Deliberately, this rung is the floor and leaves three obvious economies untouched, because the floor should honestly be the floor. For the causal mask, query $i$ attends only to keys $j \le i$, so I bound the inner loop, $\text{hi} = (\text{start\_m}+1)\cdot\text{BLOCK\_M}$, never iterating blocks strictly above the diagonal — but I still apply the elementwise mask `tl.where(offs_m >= start_n + offs_n, qk, -inf)` at *every* iterated block, including the roughly half that sit strictly below the diagonal where nothing is masked. I also multiply the scale inside the loop, `qk = tl.dot(q, trans(k)) * sm_scale`, one multiply per element of every score tile, rather than folding `sm_scale` into $Q$ once at load. And I use one uniform $64 \times 64$ block for every head dim, with no per-headdim specialization and no autotuning. Each of these is wasted non-matmul work or a missed tuning lever — and that is precisely the headroom the next rungs take. So this rung removes the $N^2$ HBM traffic and stays exact, but it is fused-and-untuned: it will clear $\text{correct} = 1$ on all three configs and land in the few-hundred-TFLOPs band, while lagging SDPA worst on $\text{hdim256\_seq16k}$ — large head dim, batch 1, the occupancy-starved, register-pressured corner — where the uniform block and the unstripped inner loop hurt most. That is a scheduling failure, not a memory or correctness one, and it is what the next fill attacks.

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
