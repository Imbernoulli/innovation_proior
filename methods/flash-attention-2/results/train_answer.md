Self-attention becomes the dominant runtime and memory bottleneck in Transformers as sequence length grows, because the score matrix between queries and keys has N^2 entries. Standard implementations compute S = QK^T with a GEMM, write it to high-bandwidth memory, read it back for softmax, write the N x N probability matrix, and read that matrix again for PV. They also store the probability matrix for the backward pass. When N is much larger than the head dimension d, this quadratic memory footprint and the repeated HBM traffic dominate execution. The IO-aware FlashAttention kernel already fixes the memory problem by loading tiles into SRAM, applying online softmax one block at a time, and recomputing tiles during the backward pass, giving exact attention with only O(N) extra memory. Once that big memory win is in place, however, the kernel still sits well below GEMM throughput because of low occupancy on long sequences, unnecessary inter-warp communication, and scalar non-matmul work such as elementwise normalizations, exponentials, reductions, and causal masks.

The remaining gap is a scheduling and micro-optimization problem rather than a mathematical one. On hardware like an A100, Tensor Core matmul runs about 16 times faster per FLOP than scalar FP32 reductions, so every division, renormalization, shared-memory exchange, and masked operation matters. The baseline tiled kernel normalizes the output accumulator after every key/value block, forces warps to reduce partial results through shared memory, and exposes too few independent thread blocks when batch size is small and the sequence is long. The goal is therefore to keep exact softmax attention while reshaping the parallel work so that more thread blocks run independently, warps own complete output rows, and scalar housekeeping is minimized.

The method is FlashAttention-2. It preserves the exact output O = softmax(softmax_scale * QK^T)V and the linear auxiliary memory of FlashAttention, but changes the loop order, the online-softmax accumulator, the launch grid, and the warp-level work partitioning. The key idea is to stop normalizing the output after every block. Instead, each query-row block carries an unnormalized accumulator A, a running row max m, and a running denominator ell. When a new score block S_j arrives, the new max is m_new = max(m, rowmax(S_j)), the old accumulator is rescaled by alpha = exp(m - m_new), and the new unnormalized contributions are added in. Only at the end of the row block do we divide A by ell to produce O and store L = m + log(ell) for the backward pass. This removes redundant per-block divisions and keeps the number of scalar normalizations minimal.

The second change is parallelization. In the forward pass, FlashAttention-2 makes each query row block an independent unit of work. The launch grid is conceptually (num_query_blocks, batch, heads), so a long sequence with small batch still creates enough independent thread blocks to saturate the GPU. Each block loads its query tile, sweeps all key/value tiles, and writes its own O_i and L_i without communicating with other blocks. In the backward pass the complementary axis is used: each key/value column block becomes an independent worker that owns dK_j and dV_j and sweeps all query blocks. The shared dQ accumulator is updated with atomics or a separate accumulation buffer. Inside a forward block, query rows are split across warps rather than key/value slices, so each warp computes complete output rows and avoids the inter-warp reduction needed by split-K partitioning.

For causal attention, FlashAttention-2 applies the same case analysis to minimize masking work. If a key block lies entirely above the diagonal relative to the query block, it is skipped entirely. If it lies entirely below the diagonal, no causal mask is applied at all. Only the boundary block that straddles the diagonal needs an elementwise key_index > query_index mask. The backward pass recomputes probability tiles from L as P = exp(S - L[:, None]) and then computes dV_j += P^T dO_i, dP = dO_i V_j^T, dS = P * (dP - D[:, None]) where D_i = sum_k dO_ik O_ik, and finally accumulates dQ and dK with the attention scale applied to the score gradients.

```python
import torch


def flash_attention2_forward(Q, K, V, B_r, B_c, *, causal=False, softmax_scale=1.0):
    """Exact O = softmax(softmax_scale * Q K^T) V, tiled with O(N) extra memory."""
    N, d = Q.shape
    O = torch.empty_like(Q)
    L = torch.empty(N, dtype=torch.float32, device=Q.device)
    T_r = (N + B_r - 1) // B_r
    T_c = (N + B_c - 1) // B_c

    for i in range(T_r):
        r0, r1 = i * B_r, min((i + 1) * B_r, N)
        q = Q[r0:r1]
        rows = torch.arange(r0, r1, device=Q.device)
        acc = torch.zeros((r1 - r0, d), dtype=torch.float32, device=Q.device)
        m = torch.full((r1 - r0,), -float("inf"), dtype=torch.float32, device=Q.device)
        ell = torch.zeros((r1 - r0,), dtype=torch.float32, device=Q.device)

        for j in range(T_c):
            c0, c1 = j * B_c, min((j + 1) * B_c, N)
            if causal and c0 >= r1:
                continue

            k = K[c0:c1]
            v = V[c0:c1]
            S = (q @ k.T).float() * softmax_scale

            if causal and c1 > r0:
                cols = torch.arange(c0, c1, device=Q.device)
                S = S.masked_fill(cols[None, :] > rows[:, None], -float("inf"))

            m_new = torch.maximum(m, S.max(dim=1).values)
            P_tilde = torch.exp(S - m_new[:, None])
            alpha = torch.exp(m - m_new)
            ell = alpha * ell + P_tilde.sum(dim=1)
            acc = alpha[:, None] * acc + P_tilde @ v.float()
            m = m_new

        O[r0:r1] = (acc / ell[:, None]).to(O.dtype)
        L[r0:r1] = m + torch.log(ell)

    return O, L


def flash_attention2_backward(Q, K, V, O, dO, L, B_r, B_c, *, causal=False, softmax_scale=1.0):
    """Backward pass recomputing tiled probabilities from L."""
    N, d = Q.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    D = (dO.float() * O.float()).sum(dim=1)
    T_r = (N + B_r - 1) // B_r
    T_c = (N + B_c - 1) // B_c

    for j in range(T_c):
        c0, c1 = j * B_c, min((j + 1) * B_c, N)
        cols = torch.arange(c0, c1, device=Q.device)
        k = K[c0:c1]
        v = V[c0:c1]
        dK_j = torch.zeros_like(k, dtype=torch.float32)
        dV_j = torch.zeros_like(v, dtype=torch.float32)

        for i in range(T_r):
            r0, r1 = i * B_r, min((i + 1) * B_r, N)
            if causal and c0 >= r1:
                continue

            q = Q[r0:r1]
            rows = torch.arange(r0, r1, device=Q.device)
            S = (q @ k.T).float() * softmax_scale
            if causal and c1 > r0:
                S = S.masked_fill(cols[None, :] > rows[:, None], -float("inf"))

            P = torch.exp(S - L[r0:r1, None])
            dO_i = dO[r0:r1].float()
            dV_j += P.T @ dO_i
            dP = dO_i @ v.float().T
            dS = P * (dP - D[r0:r1, None])

            dQ[r0:r1] += (dS @ k.float() * softmax_scale).to(dQ.dtype)
            dK_j += dS.T @ q.float() * softmax_scale

        dK[c0:c1] = dK_j.to(dK.dtype)
        dV[c0:c1] = dV_j.to(dV.dtype)

    return dQ, dK, dV
```
