# FlashAttention-2

## Problem

Exact self-attention is the Transformer's runtime/memory bottleneck, quadratic in sequence length `N`. An IO-aware tiled kernel (FlashAttention) already makes memory linear in `N` and runs 2–4× faster than standard attention with no approximation, but its forward pass reaches only ~30–50% of a GPU's peak FLOPs/s (backward ~25–35%), versus 80–90% for a tuned matmul. The gap is poor work partitioning between thread blocks (low occupancy) and between warps (shared-memory traffic), plus time spent on costly non-matmul operations. Goal: close the gap to GEMM throughput while keeping the exact output and linear memory.

## Key idea

Keep tiling + online softmax (compute exact attention in SRAM-sized blocks, never materializing the `N×N` score matrix; recompute in the backward), then re-engineer the work layout around the GPU: (1) cut non-matmul FLOPs, (2) parallelize over the sequence-length dimension, (3) partition warps to avoid inter-warp reduction. Roughly another 2× over FlashAttention.

## Prior art recap (FlashAttention)

Online softmax processes each query row in column blocks, carrying a running max `m` and denominator `ℓ`. When a new block raises the max, rebase the partial result by `e^{m_prev - m_new}`. This gives the exact softmax with no full-row materialization, enabling tiling. Backward recomputes `S, P` from SRAM-resident input tiles instead of storing the `N×N` matrices, so memory is `O(N)`.

## The three improvements

**1. Fewer non-matmul FLOPs.** On an A100, matmul runs at 312 TFLOPs/s but non-matmul at 19.5 TFLOPs/s — a non-matmul FLOP costs ~16×. FlashAttention rescaled the output accumulator by `1/ℓ` on every inner step (elementwise, non-matmul). Instead, carry an **un-normalized** accumulator, rescaling only by the max-correction each step, and divide by `ℓ` once at the very end:
`Õ^(j) = e^{m^(j-1)-m^(j)} Õ^(j-1) + e^{S^(j)-m^(j)} V_j`, then `O = Õ^(last)/ℓ^(last)`.
Also store only the log-sum-exp `L = m + log(ℓ)` (one vector) for the backward, which recomputes `P = e^{S-L}`.

**2. Parallelize over sequence length.** FlashAttention parallelized only over batch × heads (one thread block per head), starving the SMs when long context forces small batches. Query row blocks are independent (each row's softmax normalizes within itself), so swap the loops — **outer loop over Q row blocks, inner over K/V column blocks** — making each row block an independent thread block, and add sequence length as a third grid axis. Backward parallelizes over column blocks, with `dQ` accumulated by atomic adds.

**3. Warp partitioning (split-Q).** FlashAttention's "split-K" split `K, V` across warps with `Q` shared, forcing warps to reduce partial `QKᵀ·V` products through shared memory. FlashAttention-2 splits `Q` across warps with `K, V` shared: each warp computes its query rows of `QKᵀ` then multiplies by the shared `V` to get its own complete output rows — no inter-warp communication.

**Extras.** Causal masking skips entirely-masked blocks (~half for long `N`; ~1.7–1.8× speedup) and applies the elementwise mask only on the diagonal block. MQA/GQA index a shared K/V head implicitly (no duplication), summing `dK, dV` across query heads in the backward. Block sizes `{64,128}×{64,128}` per head dimension and SRAM budget. Output is exact: `O = softmax(QKᵀ)V`, `O(N²d)` FLOPs, `O(N)` extra memory.

## Code

```python
import torch

def flash_attention2_forward(Q, K, V, B_r, B_c):
    N, d = Q.shape
    O = torch.zeros(N, d); L = torch.zeros(N)
    T_r = (N + B_r - 1) // B_r; T_c = (N + B_c - 1) // B_c
    for i in range(T_r):                                 # outer: Q row blocks (parallel grid axis)
        q = Q[i*B_r:(i+1)*B_r]
        O_i = torch.zeros(q.shape[0], d)                 # un-normalized accumulator
        m_i = torch.full((q.shape[0],), float("-inf")); l_i = torch.zeros(q.shape[0])
        for j in range(T_c):                             # inner: K/V column blocks
            k = K[j*B_c:(j+1)*B_c]; v = V[j*B_c:(j+1)*B_c]
            S = q @ k.T
            m_new = torch.maximum(m_i, S.max(dim=1).values)
            P = torch.exp(S - m_new[:, None])
            alpha = torch.exp(m_i - m_new)               # max-correction (<= 1)
            l_i = alpha * l_i + P.sum(dim=1)
            O_i = alpha[:, None] * O_i + P @ v
            m_i = m_new
        O[i*B_r:(i+1)*B_r] = O_i / l_i[:, None]          # normalize once
        L[i*B_r:(i+1)*B_r] = m_i + torch.log(l_i)        # store log-sum-exp
    return O, L

def flash_attention2_backward(Q, K, V, O, dO, L, B_r, B_c):
    N, d = Q.shape
    dQ = torch.zeros(N, d); dK = torch.zeros(N, d); dV = torch.zeros(N, d)
    D = (dO * O).sum(dim=1)
    T_r = (N + B_r - 1) // B_r; T_c = (N + B_c - 1) // B_c
    for j in range(T_c):                                 # parallelize over K/V columns
        k = K[j*B_c:(j+1)*B_c]; v = V[j*B_c:(j+1)*B_c]
        dK_j = torch.zeros(k.shape[0], d); dV_j = torch.zeros(v.shape[0], d)
        for i in range(T_r):
            q = Q[i*B_r:(i+1)*B_r]
            S = q @ k.T
            P = torch.exp(S - L[i*B_r:(i+1)*B_r][:, None])     # recompute P
            dV_j += P.T @ dO[i*B_r:(i+1)*B_r]
            dP = dO[i*B_r:(i+1)*B_r] @ v.T
            dS = P * (dP - D[i*B_r:(i+1)*B_r][:, None])         # P ⊙ (dP - D)
            dQ[i*B_r:(i+1)*B_r] += dS @ k                       # atomic add
            dK_j += dS.T @ q
        dK[j*B_c:(j+1)*B_c] = dK_j; dV[j*B_c:(j+1)*B_c] = dV_j
    return dQ, dK, dV
```

The production kernel fuses this into a single CUDA/Triton kernel launched over a grid of `(row blocks, batch·heads)` thread blocks, with `Q` split across warps inside each block; the code above is the exact computation it carries out.
