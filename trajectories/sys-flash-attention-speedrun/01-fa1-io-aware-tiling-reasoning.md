I have a correct attention kernel and it is slow, and I want to understand *why* before I touch anything. The math is `O = softmax(QKᵀ/√d) · V` and there are two matmuls in there that the tensor cores should chew through at hundreds of TFLOPs/s. So if attention is slow, the matmuls aren't the bottleneck — something else is. Let me count the bytes, not the FLOPs.

The standard way I compute this is three passes. First a GEMM forms `S = QKᵀ`, an `N×N` matrix, and writes it to HBM. Then a softmax kernel reads `S` back, computes the row maxima, the exponentials, the row sums, normalizes, and writes `P = softmax(S)` to HBM. Then a second GEMM reads `P` back and multiplies by `V` to get `O`. Stare at the memory traffic of that. The matmul FLOPs are `~2·N²·d` for each GEMM, so `~4·N²·d` total. But the HBM traffic for the score matrix alone is: write `S` (`N²` elements), read `S` (`N²`), write `P` (`N²`), read `P` (`N²`) — `~4·N²` elements of HBM traffic just to shuttle the intermediate that I immediately throw away. The arithmetic intensity — FLOPs per byte — of the whole attention is low because the dominant cost isn't the `d`-deep contraction, it's moving an `N×N` matrix across the slow memory four times. For `N=8K`, FP16, that `N×N` is 128 MB *per head*. The kernel spends its life waiting on HBM while the tensor cores idle. Attention is memory-bound, and the thing flooding the memory bus is the materialized score matrix.

So the obvious target is: don't materialize `S`. If I never write the `N×N` matrix to HBM at all, that `4·N²` of traffic vanishes and I'm left moving Q, K, V, and O — which are each only `N·d`, *linear* in `N`. The whole quadratic term in the memory cost is an artifact of how I staged the computation, not of the computation itself.

The way to not materialize it is to fuse the three passes into one kernel, computing `S`, its softmax, and `P·V` for a tile of the output without ever sending the scores to global memory — keep them in registers and shared memory, where they're born and where they die. Fine, that's the goal. The trouble is the softmax in the middle. Softmax over a row needs the *whole* row: I subtract the row max for stability and divide by the row sum, and the max and sum are reductions over all `N` keys. If I have to see all `N` scores of a row before I can normalize even the first one, then I'm forced to hold a full row of `S` resident — and if I tile the keys to keep memory bounded, I only ever have a *block* of the row at a time. How do I take a softmax over a row when I only ever see it a block at a time?

Let me think about what a softmax actually requires. For a row of scores `x₁ … x_N`, I want `oᵢ = Σⱼ (exp(xⱼ − m) / ℓ) · vⱼ` where `m = maxⱼ xⱼ` and `ℓ = Σⱼ exp(xⱼ − m)`. The `m` is there only for numerical stability — mathematically I could use any constant, but in floating point I must subtract the max or `exp` overflows. The problem is that `m` and `ℓ` are global over the row, and I want to process the keys in blocks `1, 2, 3, …` and accumulate `o` incrementally.

What if I keep *running* statistics? Suppose after processing the first block of keys I have a running max `m⁽¹⁾` (the max so far), a running normalizer `ℓ⁽¹⁾ = Σ exp(xⱼ − m⁽¹⁾)` over the keys seen so far, and a running unnormalized output `õ⁽¹⁾ = Σ exp(xⱼ − m⁽¹⁾) · vⱼ`. Now a second block arrives with its own local max. The running max updates to `m⁽²⁾ = max(m⁽¹⁾, m_local)`. But all my accumulated quantities were computed relative to the *old* max `m⁽¹⁾`. If the new block raised the max, every `exp(xⱼ − m⁽¹⁾)` I already summed is too large by exactly a factor of `exp(m⁽¹⁾ − m⁽²⁾)` relative to what it should be against the new max. And `exp` of a difference factors cleanly: `exp(xⱼ − m⁽²⁾) = exp(xⱼ − m⁽¹⁾) · exp(m⁽¹⁾ − m⁽²⁾)`. So I can *correct* the old accumulators by a single scalar multiply. Let `α = exp(m⁽¹⁾ − m⁽²⁾)` — a number in `(0,1]` since `m⁽²⁾ ≥ m⁽¹⁾`. Then

    ℓ⁽²⁾ = α · ℓ⁽¹⁾ + Σ_{j in block 2} exp(xⱼ − m⁽²⁾)
    õ⁽²⁾ = α · õ⁽¹⁾ + Σ_{j in block 2} exp(xⱼ − m⁽²⁾) · vⱼ

The old partial output and old normalizer just get rescaled by `α` and then I add the new block's contribution computed against the new max. The max only ever goes up, so `α ≤ 1` and nothing overflows. After the last block, the true normalized output is `õ_final / ℓ_final`. This is a streaming softmax: one pass over the keys, maintaining `(m, ℓ, õ)`, rescaling the running output whenever the max grows. The full row of scores never has to exist at once — I only ever hold a block of `S` plus three small running quantities per query row. And critically, the *answer is exact*: this is algebraically identical to computing the softmax over the whole row, just reassociated. No approximation. The output is bit-for-bit the textbook attention.

That dissolves the obstacle. Now I can structure the kernel around it. Tile the queries: each threadblock takes a block of `B_r` query rows, loads that `Q` tile into SRAM, and it stays there. Then loop over key/value blocks of `B_c`: load `K_j`, `V_j` tiles into SRAM, compute the block of scores `S_ij = Q_i K_jᵀ` (a small tensor-core GEMM, `B_r × B_c`, living in registers), find the block's row maxima, do the online-softmax update of `(m, ℓ)` and the rescale of the running output `õ`, then accumulate `õ += P_ij V_j` (the second small GEMM). Pick `B_r`, `B_c` so that `Q_i`, `K_j`, `V_j`, and the `B_r × B_c` score block all fit in the SM's shared memory and registers at once — that's the constraint that makes the fusion possible. Every byte of `K` and `V` is read from HBM exactly once, streamed through SRAM, reused across all the queries in the block, and dropped. The `N×N` matrix is never written anywhere. This is what makes it *IO-aware*: I sized the tiles to the actual memory hierarchy — what fits in fast SRAM — instead of pretending memory is flat, and the whole design follows from minimizing HBM reads/writes rather than minimizing FLOPs.

Let me sanity-check the cost. HBM traffic is now: read `Q`, `K`, `V` once each (`3·N·d`), write `O` once (`N·d`). That's `O(N·d)`, linear in `N`. The `4·N²` of score-matrix traffic is gone entirely. The matmul FLOPs are unchanged — I still do `~4·N²·d` of them — but now they run back-to-back out of SRAM with no HBM stall in between, so the kernel is finally compute-bound on the tensor cores, which is where attention *should* be bound. For a long sequence the speedup is large precisely because the term I eliminated grew as `N²` while what's left grows as `N`.

Now the backward pass, and here's a second place the materialization wants to creep back in. The gradients through attention need the probabilities `P`. The naive forward would have saved `S` or `P` for the backward to reuse — but I just spent all this effort *not* writing `P`, and saving it would reintroduce the `O(N²)` activation, now living across the whole forward-backward and blowing the memory budget again. So I refuse to save it. Instead: what do I actually need to reconstruct `P` in the backward? The softmax of a row is determined by the scores and one scalar per row — the log-sum-exp `L = m + log ℓ`, since `Pᵢⱼ = exp(Sᵢⱼ − Lᵢ)`. If I save just that one number per query row from the forward (an `O(N)` vector, negligible), then in the backward I can *recompute* the scores `S_ij = Q_i K_jᵀ` tile by tile — the same fused, tiled GEMM as the forward — and recover `P_ij = exp(S_ij − L_i)` on the fly, block by block, without ever having stored the matrix. Recomputing `S` costs extra FLOPs, but FLOPs are cheap and HBM bandwidth is the bottleneck; trading a re-GEMM (compute) to avoid reading back an `N×N` matrix (bandwidth) is exactly the right trade on a memory-bound problem. So the backward is also fused and tiled, also linear in memory, recomputing the scores from `(Q, K)` and the saved `L` instead of reading a saved `P`. This recompute-in-backward is the other half of keeping the whole thing `O(N)` in memory.

So the kernel I want is: for each query block, load `Q_i`; loop over key/value blocks `K_j, V_j`; compute `S_ij = Q_i K_jᵀ`; update the running max `m`, rescale the running normalizer `ℓ` and the running output accumulator `acc_o` by `exp(m_old − m_new)`; add the new block `exp(S_ij − m_new)` into `ℓ` and `exp(S_ij − m_new) · V_j` into `acc_o`; at the end divide `acc_o` by `ℓ` and write `O_i`, and stash `L_i = m + log ℓ`. The score block lives in registers; `K_j, V_j` stream through SRAM; nothing quadratic touches HBM.

Let me write the forward as the fused tile loop. The cleanest expression of the online-softmax update is the running-`(m, ℓ)` recurrence with the output-accumulator rescale done in place each iteration:

```python
# fused attention forward, one threadblock owns a block of BLOCK_M query rows.
# acc_o is the running output accumulator; m_i, lse_i are the running max and log-normalizer.
lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
m_i   = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
q = tl.load(q_ptrs)                                   # Q tile stays in SRAM for the whole loop

for start_n in range(0, end_n, BLOCK_N):              # stream over key/value blocks
    k = tl.load(k_ptrs + start_n * stride_kn)
    qk = tl.dot(q, k, trans_b=True)                   # S_ij = Q_i K_jᵀ  (lives in registers)
    if IS_CAUSAL:
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
    m_ij  = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)   # new running max
    p     = tl.exp(qk * softmax_scale - m_ij[:, None])         # exp against the new max
    l_ij  = tl.sum(p, 1)
    acc_o_scale = tl.exp(m_i - m_ij)                  # α = exp(m_old − m_new), the rescale factor
    acc_o = acc_o * acc_o_scale[:, None]              # correct the running output for the raised max
    v = tl.load(v_ptrs + start_n * stride_vn)
    acc_o += tl.dot(p.to(v.dtype), v)                 # add this block's exp(S)·V
    m_i = m_ij
    l_i_new = tl.exp(lse_i - m_ij) + l_ij             # update the running normalizer
    lse_i   = m_ij + tl.log(l_i_new)

o_scale = tl.exp(m_i - lse_i)                         # final normalization: divide by the row sum
acc_o = acc_o * o_scale[:, None]
tl.store(out_ptrs, acc_o)
tl.store(lse_ptrs, lse_i)                             # save ONLY the log-sum-exp for the backward
```

and the backward never reads a stored `P` — it reloads `Q`, `K` and recomputes the scores, then rebuilds the probabilities from the saved `lse`:

```python
# fused attention backward: recompute, don't read back the N×N matrix.
q = tl.load(q_ptrs)
k = tl.load(k_ptrs)
qk = tl.dot(q, k, trans_b=True)                       # recompute S_ij = Q_i K_jᵀ
if IS_CAUSAL:
    qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
lse_i = tl.load(lse_ptrs)                             # the one scalar per row saved by the forward
p = tl.exp(qk * softmax_scale - lse_i[:, None])       # P_ij rebuilt on the fly, no HBM round-trip
# ... gradients dq, dk, dv accumulated from p and the upstream grad, tile by tile ...
```

The causal case is a bonus: when the mask zeroes the upper triangle, a query block whose rows all precede a key block can skip that key block entirely — `end_n` only runs up to the diagonal — so causal attention does roughly half the work for free, with no materialized mask.

Stepping back to the causal chain: attention was slow not because of its FLOPs but because the textbook three-pass staging writes an `N×N` score matrix to HBM and reads it back, making the kernel bandwidth-bound on a quadratic intermediate. The online-softmax recurrence lets me compute an exact row-softmax in a single streaming pass over key blocks, maintaining a running max, normalizer, and output, rescaling whenever the max grows — so the score matrix never has to be resident as a whole. Sizing the query/key/value tiles to fit in SRAM fuses `QKᵀ`, softmax, and `P·V` into one kernel that reads Q, K, V once and writes O once: HBM traffic drops from `O(N²)` to `O(N)`, the tensor cores stop stalling on memory, and the result is bit-for-bit exact. Saving only the per-row log-sum-exp and recomputing the scores in the backward keeps that pass linear in memory too. The kernel is now compute-bound, exact, and linear in sequence length — and what's left to attack is no longer *whether* the matrix hits HBM but how efficiently the surviving on-chip work is partitioned across the GPU's warps.
