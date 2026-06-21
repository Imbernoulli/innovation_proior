Self-attention in Transformers computes O = softmax(QK^T)V for query, key, and value matrices Q, K, V of size N × d. The standard implementation materializes the N × N score matrix S = QK^T and the probability matrix P = softmax(S) in GPU high-bandwidth memory (HBM). Because softmax, masking, dropout, and the surrounding matrix multiplications are memory-bound operations, most of attention's wall-clock time is spent reading and writing these N × N matrices across the slow HBM bus rather than performing arithmetic. Reducing FLOPs alone is therefore insufficient: approximate and sparse attention methods often fail to deliver real speedups because they still move large intermediate matrices through HBM and sometimes introduce irregular access patterns that hurt throughput.

The right objective is to make attention IO-aware. We want to minimize transfers between HBM and the small, fast on-chip SRAM, and we want to avoid writing the full N × N attention matrix to HBM entirely, while still computing exact attention so model quality is unchanged. The two obstacles are that softmax requires a row-wise maximum and denominator, which seems to need the whole row at once, and that training's backward pass normally needs the N × N matrices S and P for gradients.

The method I propose is FlashAttention. It solves both problems by tiling the computation to run on-chip and using recomputation during the backward pass. The core idea is to generalize the online softmax recurrence so that it updates not only a running row maximum and denominator, but also a running output accumulator. We split K and V into blocks that fit in SRAM and stream over them. For each query block, we maintain a running maximum m_i, a running denominator ℓ_i, and an unnormalized output accumulator acc_i. When a new key/value block arrives, we compute the local score block S_ij on-chip, obtain its local max and denominator, then rescale the existing accumulator and denominator by the maximum shift and add the new block's contribution. At the end we normalize once per row. This produces exactly softmax(QK^T)V by induction: rescaling by e^{m_old − m_new} rebases both the accumulated numerator and denominator onto the new maximum, so partial results glue together exactly.

For the backward pass, FlashAttention does not store S or P. Instead it stores only the output O and O(N) softmax statistics, typically as the log-sum-exp lse_i = m_i + log ℓ_i. During the backward it recomputes the needed attention block on-chip from Q, K, and lse. The key identity D_i = P_{i:}^T dP_{i:} = do_i · o_i turns the length-N row reduction that appears in the softmax Jacobian into a simple dot product of two length-d vectors, keeping the backward pass on-chip and O(N) in extra memory. Because attention is memory-bound, the extra FLOPs from recomputation are cheaper than the HBM round-trips they avoid, so the backward is faster as well as more memory-efficient.

The IO complexity is Θ(N²d²/M) HBM accesses for SRAM size M, versus Θ(Nd + N²) for standard attention. With head dimension d around 64–128 and M around 100 KB, d² ≪ M, so this is a large constant-factor reduction. A matching lower-bound argument shows no exact attention algorithm can be asymptotically better across the natural range of M. The same tiling framework also supports block-sparse attention by simply skipping zero blocks.

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
    m = torch.full((N,), float("-inf"), device=device, dtype=Q.dtype)
    lse = torch.full((N,), float("-inf"), device=device, dtype=Q.dtype)
    acc = torch.zeros_like(Q)

    for j0 in range(0, N, block_n):
        Kj = K[j0:j0 + block_n]
        Vj = V[j0:j0 + block_n]
        S = (Q @ Kj.T) * scale
        if causal:
            idx_q = torch.arange(N, device=device)[:, None]
            idx_k = (j0 + torch.arange(Kj.shape[0], device=device))[None, :]
            S = S.masked_fill(idx_k > idx_q, float("-inf"))

        m_blk = S.max(dim=1).values
        m_new = torch.maximum(m, m_blk)
        P = torch.exp(S - m_new[:, None])
        l_blk = P.sum(dim=1)

        acc = acc * torch.exp(m - m_new)[:, None]
        acc = acc + P @ Vj

        lse = m_new + torch.log(torch.exp(lse - m_new) + l_blk)
        m = m_new

    O = acc * torch.exp(m - lse)[:, None]
    return O, lse


def flash_attention_backward(Q, K, V, O, dO, lse, block_n=128, scale=None, causal=False):
    """Backward pass: recompute P on-chip from lse; D_i = do_i . o_i."""
    N, d = Q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    device = Q.device

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    D = (O * dO).sum(dim=1)

    for j0 in range(0, N, block_n):
        Kj = K[j0:j0 + block_n]
        Vj = V[j0:j0 + block_n]
        S = (Q @ Kj.T) * scale
        if causal:
            idx_q = torch.arange(N, device=device)[:, None]
            idx_k = (j0 + torch.arange(Kj.shape[0], device=device))[None, :]
            S = S.masked_fill(idx_k > idx_q, float("-inf"))

        P = torch.exp(S - lse[:, None])
        dV[j0:j0 + Vj.shape[0]] += P.T @ dO
        dP = dO @ Vj.T
        dS = P * (dP - D[:, None]) * scale
        dQ += dS @ Kj
        dK[j0:j0 + Kj.shape[0]] += dS.T @ Q

    return dQ, dK, dV
```
