Let me start from where exact attention actually burns time on a GPU, because that's the thing I want to fix, and it isn't the FLOP count. The standard implementation computes `S = QKᵀ`, an `N×N` matrix, writes it to HBM; reads it back to take the row-wise softmax `P`, writes `P` to HBM; reads `P` back to compute `O = PV`. Since `N ≫ d` — `N` is 1k–8k, `d` is 64–128 — that `N×N` intermediate is enormous, and the whole thing is bottlenecked not by the matmuls but by shuttling `S` and `P` across the slow HBM bus. The GPU's compute units sit idle waiting on memory. So the first move isn't algebraic, it's about *where the data lives*: keep the giant intermediate off HBM entirely.

The obstacle to that is the softmax. The output for one query row is `O_i = Σ_j softmax(S_{ij}) V_j`, and the softmax over row `i` couples every entry: I need the row's max (for numerical stability) and the row's sum-of-exponentials before I can normalize any single weight. That seems to force me to have the whole row of `S` in hand at once — which is exactly the `N×N` materialization I'm trying to avoid. So the real question is whether I can compute a softmax-weighted sum *incrementally*, seeing the row in blocks, without ever holding all of it.

The online-softmax trick (Milakov & Gimelshein 2018; Rabe & Staats 2021) is the way through. Walk the row in column blocks, carrying two running statistics: the max seen so far `m`, and the running denominator `ℓ`. Take a row block split into two column blocks `S^(1)` and `S^(2)`. Process `S^(1)`: `m^(1) = rowmax(S^(1))`, `ℓ^(1) = rowsum(e^{S^(1)-m^(1)})`, and a partial output `O^(1) = (1/ℓ^(1)) e^{S^(1)-m^(1)} V^(1)`. Now `S^(2)` arrives and might contain a larger value, so the max updates: `m^(2) = max(m^(1), rowmax(S^(2)))`. Every exponential I computed against the old max `m^(1)` is now mis-based — it should have been against `m^(2)`. But rebasing is just a multiply: `e^{S^(1)-m^(1)}` becomes `e^{S^(1)-m^(2)}` by multiplying by `e^{m^(1)-m^(2)}`, a factor `≤ 1` since `m^(2) ≥ m^(1)`. So the denominator rebases as `ℓ^(2) = e^{m^(1)-m^(2)} ℓ^(1) + rowsum(e^{S^(2)-m^(2)})`, and the partial output rebases the same way before the new block is folded in. At the end of the row, the running statistics equal the true full-row max and sum, so the result is the *exact* softmax — no approximation, computed in a stream.

That unlocks tiling. Load a block of `Q`, a block of `K`, a block of `V` into SRAM (the fast on-chip memory), compute that block's slice of `S` right there in SRAM, update the running `m`, `ℓ`, and output, and move to the next block — never writing `S` or `P` to HBM. For the backward pass I don't store `P` either; once the input blocks are back in SRAM I just *recompute* `S` and `P` from them. Memory drops from `O(N²)` to `O(N)`, and the HBM traffic plummets. This is the IO-aware kernel that already exists, and it's 2–4× faster than standard attention with the identical output.

But here's the thing that bothers me when I actually profile it: it's still only hitting 30–50% of the device's peak FLOPs/s on the forward pass, 25–35% on the backward — while a well-tuned GEMM hits 80–90%. The memory problem is solved; the kernel is now *compute*-limited, and it's wasting the compute. I want to know exactly where, because the IO win was the easy 4×, and the next 2× has to come from how the work is laid out on the chip. Three things turn up.

The first is subtle and it's about *which FLOPs*. The Tensor Cores make matrix-multiply absurdly cheap relative to everything else: an A100 does 312 TFLOPs/s of FP16 matmul but only 19.5 TFLOPs/s of non-matmul FP32. So a single non-matmul FLOP costs about 16× a matmul FLOP. Now look at the online-softmax inner loop: at every block step it rescales the output accumulator by `1/ℓ` (the `diag(ℓ)⁻¹` normalization). That division is elementwise — non-matmul — and it happens on every one of the inner iterations. Even though it's a small fraction of the total FLOP *count*, at 16× the per-FLOP cost it eats a real slice of wall-clock. Do I actually need to normalize by `ℓ` on every step? The normalization is just a final division by the row's denominator; the *running* part that genuinely must happen each step is rebasing to the new max. So I'll carry an **un-normalized** output accumulator — fold in each block as `Õ ← e^{m^(prev)-m^(new)} Õ + e^{S^(new)-m^(new)} V^(new)`, where the only per-step rescale is the max-correction factor `e^{m^(prev)-m^(new)}` (which I need anyway) — and divide by the final `ℓ` exactly once, after the inner loop finishes: `O = (1/ℓ^(last)) Õ^(last)`. The per-step `1/ℓ` division is gone. While I'm at it, for the backward pass I don't need to stash both `m` and `ℓ` per row; I can store just the log-sum-exp `L = m + log(ℓ)`, a single vector, and recompute the probabilities in the backward as `P = e^{S - L}`. Fewer non-matmul ops, less stored state.

Let me sanity-check that the un-normalized accumulator is still exact for two blocks. `Õ^(1) = e^{S^(1)-m^(1)} V^(1)`. After block 2 with `m = m^(2)`: `Õ^(2) = e^{m^(1)-m^(2)} Õ^(1) + e^{S^(2)-m^(2)} V^(2) = e^{m^(1)-m^(2)} e^{S^(1)-m^(1)} V^(1) + e^{S^(2)-m^(2)} V^(2) = e^{S^(1)-m} V^(1) + e^{S^(2)-m} V^(2)`. Dividing by `ℓ^(2) = rowsum(e^{S^(1)-m}) + rowsum(e^{S^(2)-m})` gives exactly `softmax([S^(1) S^(2)]) [V^(1); V^(2)]`. Exact, and the max-correction factor is `e^{m^(prev)-m^(new)} ≤ 1`, which is the natural "shrink the old contribution because the scale grew" direction.

The second problem is occupancy. The existing kernel parallelizes over batch × heads — one thread block per attention head — and schedules thread blocks onto the GPU's ~108 SMs. That's fine when `batch × heads` is large, ≥80 or so, because then there are enough thread blocks to keep every SM busy. But the whole point is *long context*, and long sequences force *small* batches (memory). With a small batch and a handful of heads, there are only a dozen thread blocks for a hundred SMs — most of the chip is idle. I need another axis to parallelize over, and the sequence length is sitting right there. Different query *row* blocks are independent: the softmax for row `i` is normalized entirely within row `i`, so row block A never needs anything from row block B. So I can hand each row block to its own thread block and add the sequence-length dimension to the grid, on top of batch and heads. Now even a batch of one with few heads spawns plenty of thread blocks, and the SMs fill up.

For that to be clean, the loop order matters, and the IO-aware kernel had it the inconvenient way: its outer loop ran over `K, V` *column* blocks and the inner loop over `Q` row blocks. With the outer loop over columns, a single row's output is touched by every outer iteration, so the rows aren't independent units of work — you can't just peel a row block off to its own thread block. Swap it: outer loop over `Q` row blocks, inner loop over `K, V` column blocks. Now each outer iteration owns one row block start to finish — load `Q_i`, sweep all the `K_j, V_j` updating that block's running `m_i, ℓ_i, O_i`, write `O_i` and `L_i` out — and these outer iterations share nothing, so they map to independent thread blocks with no communication. The outer loop is embarrassingly parallel. (This loop swap plus sequence-length parallelism is the structure Phil Tillet implemented in Triton.) For the backward pass the natural independent unit is the *column* block — each `K_j, V_j` accumulates `dK_j, dV_j` over all rows — so I parallelize the backward over column blocks; the one piece of shared state is `dQ`, which every column block contributes to, so I accumulate it with atomic adds to HBM.

The third problem lives *inside* a thread block, in how the ~4–8 warps split the work. The IO-aware kernel uses a "split-K" layout: it splits `K` and `V` across the four warps and keeps `Q` shared. Each warp computes a slice of `QKᵀ`, then has to multiply by its slice of `V` — but since each warp only holds part of the contraction, the warps must write their partial products to shared memory, synchronize, and reduce them together to form the output. That shared-memory round trip plus the synchronization is pure overhead, and it happens for every block. Flip the split: keep `K` and `V` shared across all warps and split `Q` across the warps instead. Now each warp owns a slice of the query *rows*; it computes its rows of `QKᵀ`, then multiplies by the shared full `V` to produce *its own complete rows of the output*. No warp needs any other warp's partial result — the output rows are partitioned, not the contraction — so there's no inter-warp reduction and no shared-memory traffic for it. The backward still needs some synchronization because the dependencies among `Q, K, V, O, dO, dQ, dK, dV` are tangled, but it too avoids split-K and its shared-memory reductions.

A couple of things make this even faster in the common cases. Causal masking, for autoregressive LMs, sets `S_{ij} = -∞` for `j > i`. Since I'm already working in blocks, any block whose column indices are entirely greater than its row indices is *entirely* masked out — for a long sequence that's roughly half the blocks — so I just skip computing them, which alone is ~1.7–1.8× faster than the unmasked kernel. And a block whose row indices are all strictly greater than its column indices needs *no* mask at all (everything's allowed); only the diagonal block per row actually needs the elementwise mask applied. For multi-query and grouped-query attention, where many query heads share one key/value head, I don't physically duplicate `K, V`; I just index into the shared head, and in the backward I sum the `dK, dV` contributions across the query heads that shared it. Block sizes are chosen in `{64,128}×{64,128}` depending on the head dimension and how much shared memory the device has — too large and the kernel spills registers or overflows shared memory and either crawls or won't launch.

Putting it together, here is the forward pass — outer loop over query row blocks, inner over key/value blocks, an un-normalized accumulator rescaled only by the max-correction each step and divided by `ℓ` once at the end, storing the log-sum-exp for the backward. In production this fuses into a single CUDA/Triton kernel; written out in tiles, the computation is:

```python
import torch

def flash_attention2_forward(Q, K, V, B_r, B_c):
    # Q,K,V: (N, d). Exact O = softmax(Q K^T) V; never materializes the N×N matrix.
    N, d = Q.shape
    O = torch.zeros(N, d)
    L = torch.zeros(N)                                   # log-sum-exp per row, for backward
    T_r = (N + B_r - 1) // B_r
    T_c = (N + B_c - 1) // B_c
    for i in range(T_r):                                 # OUTER loop over Q row blocks (parallel)
        q = Q[i*B_r:(i+1)*B_r]                            # load Q_i to SRAM
        O_i = torch.zeros(q.shape[0], d)                 # un-normalized output accumulator
        m_i = torch.full((q.shape[0],), float("-inf"))   # running row max
        l_i = torch.zeros(q.shape[0])                    # running denominator
        for j in range(T_c):                             # INNER loop over K/V column blocks
            k = K[j*B_c:(j+1)*B_c]
            v = V[j*B_c:(j+1)*B_c]
            S = q @ k.T                                  # (B_r, B_c), in SRAM (matmul)
            m_new = torch.maximum(m_i, S.max(dim=1).values)
            P = torch.exp(S - m_new[:, None])            # un-normalized probs (no 1/l yet)
            alpha = torch.exp(m_i - m_new)               # max-correction factor, <= 1
            l_i = alpha * l_i + P.sum(dim=1)             # rebase denominator, add this block
            O_i = alpha[:, None] * O_i + P @ v           # rebase accumulator, add P@V (matmul)
            m_i = m_new
        O[i*B_r:(i+1)*B_r] = O_i / l_i[:, None]          # ONE normalization at the very end
        L[i*B_r:(i+1)*B_r] = m_i + torch.log(l_i)        # store log-sum-exp only
    return O, L
```

The backward recomputes `S` and `P` from the stored `L` (no `N×N` matrix kept), and accumulates the gradients in tiles — parallelized over column blocks, with `dQ` gathered by atomic adds:

```python
def flash_attention2_backward(Q, K, V, O, dO, L, B_r, B_c):
    N, d = Q.shape
    dQ = torch.zeros(N, d); dK = torch.zeros(N, d); dV = torch.zeros(N, d)
    D = (dO * O).sum(dim=1)                               # row-wise, used in dS
    T_r = (N + B_r - 1) // B_r
    T_c = (N + B_c - 1) // B_c
    for j in range(T_c):                                 # parallelize backward over K/V columns
        k = K[j*B_c:(j+1)*B_c]; v = V[j*B_c:(j+1)*B_c]
        dK_j = torch.zeros(k.shape[0], d); dV_j = torch.zeros(v.shape[0], d)
        for i in range(T_r):
            q = Q[i*B_r:(i+1)*B_r]
            S = q @ k.T
            P = torch.exp(S - L[i*B_r:(i+1)*B_r][:, None])    # recompute P from log-sum-exp
            dV_j += P.T @ dO[i*B_r:(i+1)*B_r]
            dP = dO[i*B_r:(i+1)*B_r] @ v.T
            dS = P * (dP - D[i*B_r:(i+1)*B_r][:, None])        # softmax-grad: P ⊙ (dP - D)
            dQ[i*B_r:(i+1)*B_r] += dS @ k                      # atomic add to shared dQ
            dK_j += dS.T @ q
        dK[j*B_c:(j+1)*B_c] = dK_j; dV[j*B_c:(j+1)*B_c] = dV_j
    return dQ, dK, dV
```

The causal mask drops in as a block-level skip (skip blocks entirely above the diagonal, apply the elementwise mask only on the diagonal block); the warp layout is the implementation-level split-Q that keeps each warp's output rows self-contained.

The causal chain: standard attention is slow because it ships an `N×N` matrix through HBM, so tiling plus online softmax keeps the computation in SRAM and recomputes in the backward — exact, linear-memory, 2–4× faster, but now compute-bound at ~30–50% of peak. The remaining gap is work layout. Because a non-matmul FLOP costs ~16× a matmul one, I carry an un-normalized output accumulator and normalize by `ℓ` only once at the end (storing only the log-sum-exp), cutting the per-step elementwise rescales. Because query row blocks are independent, I swap the loops to put the row block on the outer loop and parallelize over sequence length as a third grid axis, filling the SMs even when long context forces small batches. And because split-K forces warps to reduce partial products through shared memory, I split `Q` across warps instead so each warp produces complete output rows with no inter-warp communication. Same exact output as before; roughly another 2×, closing in on GEMM-level throughput.
