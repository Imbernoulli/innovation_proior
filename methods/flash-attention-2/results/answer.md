# FlashAttention-2

FlashAttention-2 keeps the exact tiled attention idea from FlashAttention but changes the GPU work schedule and the online-softmax accumulator. It returns the same `O = softmax(QK^T)V` as standard attention, keeps `O(N)` auxiliary memory, and targets the remaining throughput gap to GEMM.

## Method

1. Carry an unnormalized output accumulator. For each query row block and key/value block:

```text
m_new = max(m_old, rowmax(S_j))
P_tilde_j = exp(S_j - m_new[:, None])
alpha = exp(m_old - m_new)          # <= 1
ell_new = alpha * ell_old + rowsum(P_tilde_j)
A_new = alpha[:, None] * A_old + P_tilde_j V_j
```

Normalize only once:

```text
O_i = A_last / ell_last[:, None]
L_i = m_last + log(ell_last)
```

The sign is important: the old accumulator is multiplied by `exp(m_old - m_new)`, not by its reciprocal.

2. Store only `L = logsumexp(S)` per query row for the backward pass. Backward recomputes each probability tile as `P_ij = exp(S_ij - L_i)` and uses

```text
D_i = sum_k dO_ik O_ik
dS_ij = P_ij * (dP_ij - D_i)
dP_ij = dO_i V_j^T
```

then accumulates `dV_j += P_ij^T dO_i`, `dQ_i += dS_ij K_j`, and `dK_j += dS_ij^T Q_i` with the attention scale applied to `dQ,dK` when scores use `softmax_scale * QK^T`.

3. Make query row blocks the forward unit of parallel work. The forward launch grid is conceptually `(num_query_blocks, batch, heads)`, so long sequences create many independent thread blocks even when batch size is small. Each block owns one query tile, sweeps the key/value tiles, and writes that tile's `O` and `L`.

4. Make key/value column blocks the backward unit of parallel work. Each worker owns `dK_j,dV_j` and contributes to shared `dQ`; production kernels use atomics or accumulation buffers for those `dQ` contributions.

5. Split work across warps by query rows rather than by key/value slices. With `Q` split across warps and `K,V` shared, each warp computes complete output rows, avoiding the forward inter-warp reduction required by split-K work partitioning.

6. For causal attention, skip blocks entirely above the diagonal, apply no elementwise mask below the diagonal, and apply the elementwise causal mask only to diagonal/boundary blocks. For MQA/GQA, index shared `K,V` heads rather than physically duplicating them; in backward, sum `dK,dV` across query heads sharing the same KV head.

## Reference Pseudocode

```python
import torch

def flash_attention2_forward(Q, K, V, B_r, B_c, *, causal=False, softmax_scale=1.0):
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

The production CUDA implementation uses specialized CuTe/CUTLASS layouts, reverse traversal of key/value tiles for masking efficiency, tuned block sizes such as 64 or 128 on the sequence axes, and atomics/accumulation buffers in the backward. Those are implementation refinements of the same recurrence and work partitioning above.
