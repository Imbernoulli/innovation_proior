# FlashAttention: fast and memory-efficient exact attention

## Problem

Self-attention computes O = softmax(QK^T)V for Q, K, V ∈ ℝ^{N×d}. Standard implementations
materialize the N×N score matrix S = QK^T and probability matrix P = softmax(S) in GPU
high-bandwidth memory (HBM), costing O(N²) memory and, more importantly, a number of HBM accesses
that grows like N². Attention is **memory-bound**: most of its runtime is spent moving the N×N
matrices across the slow HBM bus, not doing arithmetic. FLOP-reducing approximate-attention methods
therefore often fail to deliver wall-clock speedup.

## Key idea

Make attention **IO-aware**: minimize reads/writes between HBM and the small, fast on-chip SRAM,
and never materialize the N×N matrix in HBM. Two techniques achieve this for *exact* attention:

1. **Tiling.** Split Q, K, V into blocks that fit in SRAM and stream over them. Use the online
   (single-pass) softmax recurrence — a running row-max m and running denominator ℓ, rescaling
   partial results by e^{m_old − m_new} whenever the max changes — generalized so that it rescales
   the running **output accumulator** as well as the denominator. The exact output is built block
   by block on-chip; the whole computation (QK^T → mask → softmax → dropout → PV) fuses into one
   GPU kernel, so inputs are read from HBM once and only O is written back.

2. **Recomputation.** Instead of storing S and P for the backward pass, store only O and the O(N)
   softmax statistics (and the dropout RNG state). In the backward pass, recompute the needed
   attention block on-chip from the inputs and the saved statistics. This adds FLOPs but removes
   the N×N HBM round-trip, so it is a net speedup. The key identity D_i = P_{i:}·dP_{i:} = do_i·o_i
   turns a length-N row reduction into a size-d dot product, keeping the backward on-chip and O(N)
   in memory.

## Forward algorithm

Block sizes B_c = ⌈M/4d⌉, B_r = min(⌈M/4d⌉, d) for SRAM size M. Initialize O = 0, ℓ = 0, m = −∞.
Outer loop over K,V blocks j (loaded once); inner loop over Q blocks i:

- S_ij = τ Q_i K_j^T (on chip); apply mask.
- m̃_ij = rowmax(S_ij); P̃_ij = exp(S_ij − m̃_ij); ℓ̃_ij = rowsum(P̃_ij).
- m_i^new = max(m_i, m̃_ij); ℓ_i^new = e^{m_i − m_i^new} ℓ_i + e^{m̃_ij − m_i^new} ℓ̃_ij.
- O_i ← diag(ℓ_i^new)^{-1} ( diag(ℓ_i) e^{m_i − m_i^new} O_i + e^{m̃_ij − m_i^new} P̃_ij V_j ).

Returns O = softmax(τQK^T)V exactly, with O(N²d) FLOPs and O(N) extra memory.

## Backward algorithm

Saved from forward: O, the statistics (m, ℓ) — equivalently lse = m + log ℓ — and the RNG state.
Per row, D_i = rowsum(O_i ∘ dO_i). Outer loop over K,V blocks j (accumulate dK_j, dV_j on chip),
inner loop over Q blocks i:

- Recompute P_ij = diag(ℓ_i)^{-1} exp(S_ij − m_i) on chip (regenerate dropout mask from RNG).
- dV_j += P_ij^T dO_i;  dP_ij = dO_i V_j^T;  dS_ij = P_ij ∘ (dP_ij − D_i).
- dQ_i += τ dS_ij K_j;  dK_j += τ dS_ij^T Q_i.

## IO complexity

FlashAttention performs **Θ(N²d²/M)** HBM accesses versus **Θ(Nd + N²)** for standard attention.
Since head dim d (64–128) gives d² ≪ M (~100 KB), this is a many-fold reduction, which is the
source of the wall-clock speedup. The backward pass has the same Θ(N²d²/M) complexity. In the
uniform-across-M sense, no exact attention algorithm can be o(N²d²/M) for all SRAM sizes
M ∈ [d, Nd]: at M = Θ(Nd) any algorithm must read inputs and write output, ≥ Ω(Nd), so
o(N²d²/M) = o(Nd) is impossible. **Block-sparse FlashAttention** (skip zero blocks of a
block-sparsity mask) achieves Θ(Nd + N²d²s/M), where s is the fraction of nonzero blocks.

## Working code

The fused-kernel running-statistics / accumulator-rescale pattern: the forward keeps a
temporary running max and saves the final log-sum-exp statistic for the backward.

```python
import math
import torch


def flash_attention_forward(Q, K, V, block_n=128, scale=None, causal=False):
    """Exact attention without materializing the N x N matrix.

    Q, K, V: (N, d). Returns O = softmax(scale * Q K^T) V and the per-row
    log-sum-exp statistic needed by the backward pass.
    """
    N, d = Q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    device = Q.device

    O = torch.zeros_like(Q)
    m = torch.full((N,), float("-inf"), device=device, dtype=Q.dtype)      # running row max
    lse = torch.full((N,), float("-inf"), device=device, dtype=Q.dtype)    # running log-sum-exp
    acc = torch.zeros_like(Q)                # output accumulator (un-normalized)

    for j0 in range(0, N, block_n):          # outer loop: K, V blocks (loaded once)
        Kj = K[j0:j0 + block_n]              # (B_c, d)
        Vj = V[j0:j0 + block_n]
        S = (Q @ Kj.T) * scale              # (N, B_c) score block, "on chip"
        if causal:                          # mask: set disallowed entries to -inf
            idx_q = torch.arange(N, device=device)[:, None]
            idx_k = (j0 + torch.arange(Kj.shape[0], device=device))[None, :]
            S = S.masked_fill(idx_k > idx_q, float("-inf"))

        m_blk = S.max(dim=1).values          # block row max
        m_new = torch.maximum(m, m_blk)      # updated running max
        P = torch.exp(S - m_new[:, None])    # exp(S - m_new), on chip
        l_blk = P.sum(dim=1)                 # this block's denominator piece

        acc = acc * torch.exp(m - m_new)[:, None]   # rescale old accumulator to new max
        acc = acc + P @ Vj                          # add this block's P V contribution

        # fold block into running log-sum-exp:  L_new = e^{lse-m_new} + l_blk
        lse = m_new + torch.log(torch.exp(lse - m_new) + l_blk)
        m = m_new

    O = acc * torch.exp(m - lse)[:, None]    # single final normalization by 1 / L_i
    return O, lse


def flash_attention_backward(Q, K, V, O, dO, lse, block_n=128, scale=None, causal=False):
    """Backward pass: recompute P on-chip from lse; D_i = do_i . o_i."""
    N, d = Q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    device = Q.device

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    D = (O * dO).sum(dim=1)                   # D_i = rowsum(O_i .* dO_i) = do_i . o_i

    for j0 in range(0, N, block_n):
        Kj = K[j0:j0 + block_n]
        Vj = V[j0:j0 + block_n]
        S = (Q @ Kj.T) * scale
        if causal:
            idx_q = torch.arange(N, device=device)[:, None]
            idx_k = (j0 + torch.arange(Kj.shape[0], device=device))[None, :]
            S = S.masked_fill(idx_k > idx_q, float("-inf"))

        P = torch.exp(S - lse[:, None])      # recomputed P block, no N x N stored
        dV[j0:j0 + Vj.shape[0]] += P.T @ dO  # dV += P^T dO
        dP = dO @ Vj.T                       # dP_ij = dO V^T
        dS = P * (dP - D[:, None]) * scale   # dS_ij = P_ij (dP_ij - D_i), with scale
        dQ += dS @ Kj                        # dQ_i += dS_ij K_j
        dK[j0:j0 + Kj.shape[0]] += dS.T @ Q  # dK_j += dS_ij^T Q_i

    return dQ, dK, dV
```

In production this is a single fused CUDA/Triton kernel (one program per query block) so that the
loop body runs entirely in SRAM and only Q, K, V, O and the O(N) statistics cross the HBM bus.
