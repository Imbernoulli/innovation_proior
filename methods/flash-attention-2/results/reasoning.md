Let me start from the kernel that already works, because the remaining problem is no longer the big `N x N` memory footprint. Standard attention writes `S = QK^T` to HBM, reads it for softmax, writes `P`, then reads `P` for `PV`. Since `N` is usually much larger than the head dimension, those `N x N` reads and writes dominate. A tiled exact kernel fixes that by loading blocks into SRAM, updating each row's softmax online, and recomputing the tiles in the backward pass. So the first wall has already been crossed: I can get exact attention with linear extra memory.

The next question is why this exact tiled kernel is still far from a good GEMM. Once the HBM traffic is reduced, the profile is about occupancy, warp communication, and scalar work. On an A100, matmul and non-matmul are not comparable currencies: Tensor Core matmul is about 16 times faster per FLOP than FP32 scalar/reduction work (312 vs 19.5 TFLOPs/s). So a kernel can have the right asymptotic FLOPs and still waste time on normalizations, exponentials, reductions, masks, and shared-memory exchanges.

I need to be precise about the online softmax before changing it. For one query row, suppose I have already processed some key/value blocks with running max `m_old`, denominator `ell_old`, and an unnormalized accumulator `A_old = sum exp(S_old - m_old) V_old`. A new score block `S_j` arrives. The new max is `m_new = max(m_old, rowmax(S_j))`. To express the old contribution on the new exponential scale, I multiply by `exp(m_old - m_new)`. This factor is less than or equal to one. The updated denominator and accumulator are

```text
P_tilde_j = exp(S_j - m_new[:, None])
ell_new = exp(m_old - m_new) * ell_old + rowsum(P_tilde_j)
A_new = exp(m_old - m_new)[:, None] * A_old + P_tilde_j V_j
```

At the end, `O = A_last / ell_last[:, None]`. This is exact because after the rescale the old term becomes `exp(S_old - m_new) V_old`, while the new term is already `exp(S_j - m_new) V_j`. Dividing by the corresponding sum of exponentials gives the same row-wise softmax as if I had seen the full row at once.

This also fixes a sign trap. If I ever multiply the old accumulator by `exp(m_new - m_old)`, I am growing the old contribution exactly when the max has increased, which is backwards. The correction factor has to be `alpha = exp(m_old - m_new)`, or equivalently in base two `exp2((m_old - m_new) * log2(e))` if the exponentials are computed in base two. Either way it is a shrink factor, never a growth factor.

With that settled, I can remove one source of scalar work. The earlier tiled formula keeps the output normalized after each block, so each block update includes denominator divisions and a rescaling of the already normalized output. But normalization by `ell` is only needed once the final denominator is known. I should carry `A`, the unnormalized accumulator, rebase it only by the max-correction factor each block, and divide by `ell` once at the end of the row block. For the backward pass I do not need to save both `m` and `ell`; storing `L = m + log(ell)` is enough because each probability tile can be recomputed as `P = exp(S - L[:, None])`.

Now I need more independent work. If one thread block handles one batch/head pair, then long-context training hurts occupancy: long sequences force small batches, and `batch * heads` may be far below the number of SMs. The independent axis is the query row block. A row's softmax normalizes only across keys for that same row, so row block `i` does not need row block `i + 1`. Therefore the forward kernel should make a query row block the outer unit of work, and each thread block should sweep all key/value blocks for that query block. This means the launch grid includes the row-block axis in addition to batch and heads.

That requires a loop-order change relative to the older IO-aware schedule. If the outer loop is over key/value blocks, a given output row block is repeatedly touched by different outer iterations, so it is not a clean independent unit. If the outer loop is over query row blocks, one worker owns `Q_i`, sweeps `K_j,V_j`, finishes `O_i` and `L_i`, and writes them out. No other worker needs to communicate with it in the forward pass.

The backward has the complementary shape. A key/value column block owns `dK_j` and `dV_j`, and it must visit all query row blocks. Every column block contributes to `dQ`, so `dQ` is the shared gradient. The conceptual algorithm can show `dQ += dS K_j`, but a real parallel kernel needs accumulation through atomics or a separate accumulation buffer when multiple column-block workers update the same query rows.

Inside a thread block, I also want the output rows partitioned cleanly. The old split over `K,V` makes several warps produce partial contributions to the same output rows, so they must write partial products through shared memory, synchronize, and reduce. If instead I split the query rows across warps and keep `K,V` available to all of them, each warp owns complete output rows. It computes its rows of `QK^T`, multiplies by `V`, and does not need another warp's partial result in the forward pass. This is the right direction because it moves communication out of the hottest path.

Causal masking has three cases. If a key block starts at or after the end of the query row block, every pair in the block has key index greater than query index, so the whole block is masked and can be skipped. If a key block ends at or before the start of the query row block, every key is allowed and no elementwise causal mask is needed. Only the boundary block that overlaps the diagonal needs the elementwise `key_index > query_index` mask. This case distinction matters because masking is scalar work and because skipped upper-triangular blocks are a large fraction of the causal problem.

Putting the pieces together, the forward computation I want is this. The production CUDA kernel may traverse the key/value blocks in reverse for masking and register-pressure reasons, but the recurrence is order-equivalent.

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
```

The backward recomputes probabilities from `L`; it does not use a stored `N x N` probability matrix. The row scalar `D = rowsum(dO * O)` is the softmax-gradient correction. Again, this is the mathematical structure; the production kernel uses specialized layouts, accumulation buffers, and atomics to make the parallel writes legal and fast.

```python
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

This preserves the exact attention operator. The improvements are all about schedule and representation: keep the unnormalized accumulator until the end, save logsumexp instead of max plus denominator, launch forward workers over query row blocks, launch backward workers over key/value column blocks with `dQ` accumulation, and split query rows across warps so the forward path avoids inter-warp reductions.
