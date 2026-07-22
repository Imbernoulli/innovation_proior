Let me start from the kernel that already works, because the remaining problem is no longer the big `N x N` memory footprint. Standard attention writes `S = QK^T` to HBM, reads it for softmax, writes `P`, then reads `P` for `PV`. Since `N` is usually much larger than the head dimension, those `N x N` reads and writes dominate. A tiled exact kernel fixes that by loading blocks into SRAM, updating each row's softmax online, and recomputing the tiles in the backward pass. So the memory wall is already behind me: I can get exact attention with linear extra memory, and the question is what is still slow.

The next question is why this exact tiled kernel is still far from a good GEMM. Once the HBM traffic is reduced, the profile is about occupancy, warp communication, and scalar work. On an A100, matmul and non-matmul are not comparable currencies: Tensor Core matmul is about 16 times faster per FLOP than FP32 scalar/reduction work (312 vs 19.5 TFLOPs/s). So a kernel can have the right asymptotic FLOPs and still waste time on normalizations, exponentials, reductions, masks, and shared-memory exchanges. That tells me where to push: cut the per-block scalar work, and find more independent work to keep the SMs busy.

Before I change anything, I want to be sure I actually understand the online softmax, because every later decision rests on it being exact. For one query row, suppose I have already processed some key/value blocks with running max `m_old`, denominator `ell_old`, and an unnormalized accumulator `A_old = sum exp(S_old - m_old) V_old`. A new score block `S_j` arrives. The new max is `m_new = max(m_old, rowmax(S_j))`. The old exponentials are based at `m_old`; to rebase them to `m_new` I multiply by `exp(m_old - m_new)`. The updated state should be

```text
P_tilde_j = exp(S_j - m_new[:, None])
ell_new = exp(m_old - m_new) * ell_old + rowsum(P_tilde_j)
A_new = exp(m_old - m_new)[:, None] * A_old + P_tilde_j V_j
```

I do not want to take "this is exact" on faith, so let me run one row by hand and compare against a single-pass softmax. Take a row whose scores arrive in two blocks: block 1 has scores `[1, 3]` with values `[[1,0],[0,1]]`, block 2 has scores `[2, 5]` with values `[[1,1],[2,0]]`. The single-pass answer first: the global max is `5`, the exponentials of `[1,3,2,5]` rebased to `5` are `[e^-4, e^-2, e^-3, e^0] = [0.0183, 0.1353, 0.0498, 1]`, summing to `Z = 1.2034`, and weighting the four value rows gives `O = [1.7185, 0.1538]`.

Now the blocked version. After block 1, `m = 3`, `ell = e^-2 + e^0 = 1.1353`, and `A = e^-2[1,0] + e^0[0,1] = [0.1353, 1]`. Block 2 arrives with `rowmax = 5`, so `m_new = 5` and the rebase factor is `alpha = exp(3 - 5) = 0.1353`. The new exponentials are `exp([2,5] - 5) = [0.0498, 1]`. Then `ell = 0.1353 * 1.1353 + (0.0498 + 1) = 1.2034`, matching `Z`. The accumulator is `A = 0.1353 * [0.1353, 1] + (0.0498[1,1] + 1[2,0]) = 0.1353*[0.1353,1] + [2.0498, 0.0498] = [2.0681, 0.1851]`, and `O = A / ell = [1.7185, 0.1538]`. That equals the single-pass `[1.7185, 0.1538]`, so the recurrence is exact on this row, not just plausibly exact.

Running that example also lets me catch a sign trap that is easy to get backwards. The rebase factor multiplies the *old* accumulator, and it has to shrink it, because the old terms were based at the smaller max and must come down to the larger one. If I instead used `alpha = exp(m_new - m_old)`, I would be *growing* the old contribution exactly when the max increased. Let me confirm that actually breaks the number rather than just being inelegant. With the wrong factor `exp(5 - 3) = 7.389`, block 1's contribution would be inflated: `ell = 7.389 * 1.1353 + 1.0498 = 9.438` and `A = 7.389*[0.1353,1] + [2.0498,0.0498] = [3.050, 7.439]`, giving `O = [0.323, 0.788]`. That is nowhere near the reference `[1.7185, 0.1538]`, so the wrong sign is genuinely wrong, not a wash. The correct factor is `alpha = exp(m_old - m_new)`, always `<= 1`; in base two it is `exp2((m_old - m_new) * log2(e))` if the exponentials are computed in base two, but it is the same shrink factor either way.

With the recurrence pinned down, I can look for scalar work to remove. The older tiled formula keeps the output normalized after *each* block, so every block update divides by the running `ell` and rescales an already-normalized output. But normalization by `ell` is only correct once the final denominator is known, and the example above shows I never needed the intermediate `O` — I only needed `A` and `ell`, dividing once at the end. So I should carry the unnormalized accumulator `A`, rebase it by `alpha` each block, and divide by `ell` exactly once per query row block. That deletes one division and one rescale from every inner iteration, which is exactly the non-matmul work the 16x cost ratio punishes.

For the backward pass I want to avoid storing the `N x N` probability matrix, so I need to recompute each `P` tile from something small. The natural per-row scalar is `L = m + log(ell)`, the logsumexp of the row, because then `P = exp(S - L[:, None])`. Let me check that this recompute really reproduces softmax and that the backward built on it matches autograd, rather than assuming it. On a random `6x4` example I compute reference attention and its gradients with autograd, then store only `L = logsumexp(S)`, recompute `P = exp(S - L)`, and form the backward by hand: `D = rowsum(dO * O)`, `dP = dO V^T`, `dS = P * (dP - D[:, None])`, `dQ = dS K`, `dK = dS^T Q`, `dV = P^T dO`. The recomputed `P` matches the direct softmax to about `5e-17`, and `dQ`, `dK`, `dV` each match autograd to about `4e-16`. So storing one scalar `L` per row is enough to reconstruct both the probabilities and all three gradients exactly; I do not need to save `m` and `ell` separately, and I never need the stored `P`.

Now I need more independent work, which is the occupancy half of the problem. If one thread block handles one batch/head pair, then long-context training hurts: long sequences force small batches, and `batch * heads` may be far below the number of SMs, leaving them idle. I need an axis of parallelism that long sequences *create* rather than destroy. A row's softmax normalizes only across keys for that same row — the example confirmed each row is self-contained — so row block `i` never needs row block `i + 1`. That makes the query row block an independent unit. So the forward kernel should make a query row block the outer unit of work, each thread block sweeping all key/value blocks for its query block, and the launch grid should carry the row-block axis in addition to batch and heads. Now a longer sequence means more row blocks, which means more thread blocks and better occupancy.

That requires a loop-order change relative to the older IO-aware schedule. If the outer loop is over key/value blocks, a given output row block is repeatedly touched by different outer iterations, so it is not a clean independent unit and workers would have to coordinate on shared `O`. If the outer loop is over query row blocks, one worker owns `Q_i`, sweeps `K_j,V_j`, finishes `O_i` and `L_i`, and writes them out, with no other worker needing to communicate with it in the forward pass. The query-outer order is the one that makes the row block a true independent unit.

The backward has the complementary shape. There the recomputation needs each key block to see all query rows that attend to it, so a key/value column block owns `dK_j` and `dV_j` and visits all query row blocks. The asymmetry is in `dQ`: every column block contributes `dS K_j` to `dQ`, so `dQ` is shared across workers. The conceptual algorithm writes `dQ += dS K_j`, but a real parallel kernel cannot let many column-block workers race on the same query rows, so it needs atomics or a separate accumulation buffer for those `dQ` updates. So forward parallelizes over query blocks with private outputs, and backward parallelizes over key/value blocks with one shared accumulator.

Inside a thread block, I also want the output rows partitioned cleanly across warps. The old split over `K,V` makes several warps each produce a partial contribution to the *same* output rows, so they must write partials through shared memory, synchronize, and reduce them — and shared-memory exchange plus sync is exactly the non-matmul tax I am trying to cut. If instead I split the query rows across warps and keep `K,V` visible to all of them, each warp owns complete output rows: it computes its rows of `QK^T`, exponentiates, multiplies by `V`, and never needs another warp's partial result in the forward pass. Communication moves off the hot path. I expect the boundary case to need care — at the row tile's edge a warp still reduces over its own keys — but no cross-warp reduction remains for the output, which is the part that was costing synchronization every inner block.

Causal masking splits into three cases by where a key block sits relative to the query row block. If a key block starts at or after the end of the query row block (`c0 >= r1`), every pair has key index greater than query index, so the whole block is masked and can be skipped entirely — no matmul, no exponential. If a key block ends at or before the start of the query row block, every key is allowed and no elementwise mask is needed. Only the boundary block straddling the diagonal needs the elementwise `key_index > query_index` mask. This matters for two reasons: masking is scalar work, and the skipped upper-triangular blocks are close to half of a causal problem, so skipping them is a large constant factor, not a micro-optimization.

Putting the pieces together, the forward computation I want is below. The production CUDA kernel may traverse the key/value blocks in reverse for masking and register-pressure reasons, but the recurrence is order-equivalent, and the hand-worked row above is exactly this loop with `T_c = 2`.

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

The backward recomputes probabilities from `L`; it does not use a stored `N x N` probability matrix. The row scalar `D = rowsum(dO * O)` is the softmax-gradient correction, and the random-example check above is exactly this code path, matching autograd to roundoff. Again, this is the mathematical structure; the production kernel uses specialized layouts, accumulation buffers, and atomics to make the parallel writes legal and fast.

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
