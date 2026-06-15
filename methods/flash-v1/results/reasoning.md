Let me start from what actually hurts, which is that attention is slow on long sequences and I cannot make it fast no matter how I rearrange the arithmetic. The function is simple: for `Q, K, V` in `R^{N x d}`, I want `O = softmax(Q K^T) V`. The obvious implementation forms `S = Q K^T`, an `N x N` matrix, writes it out, reads it back to apply softmax row-wise into `P = softmax(S)`, writes `P`, then reads `P` and `V` to form `O = P V`. For `N` in the thousands and `d` only 64 or 128, that `N x N` matrix is enormous compared to everything else, and I keep paying for it twice — once to store it, once to read it back — and again for `P`.

So the first thing I should do is figure out *where the time actually goes*, because if I guess wrong I will optimize the wrong thing. The matmuls `Q K^T` and `P V` are dense and the GPU eats those for breakfast; the chip has far more arithmetic throughput than I can feed it. The softmax, on the other hand, is a row-wise max, a row-wise sum, and an elementwise exponential and divide — almost no arithmetic per number, just a lot of numbers streamed through. On this hardware the HBM runs at maybe 1.5-2 TB/s while the on-chip SRAM runs around 19 TB/s but is tiny, a couple hundred kilobytes per streaming multiprocessor. Compute has been getting faster relative to memory for generations. So the right model here is arithmetic intensity: an operation is compute-bound if it does many FLOPs per byte moved, memory-bound if it does few. The softmax, and the whole act of writing and re-reading an `N x N` matrix, is memory-bound. The wall-clock time of standard attention is dominated by shuttling `S` and `P` across the slow HBM bus, not by the matmuls. That reframes the entire problem: I am not trying to do fewer operations, I am trying to do fewer *reads and writes to HBM*.

This is worth being precise about because it tells me which prior work is barking up the wrong tree. There is a whole industry of approximate attention — sparsify the score matrix so most of it is zero (Reformer's hashing, Kitaev et al. 2020), or approximate `softmax(QK^T)` by a low-rank or kernel-feature product so you never form the `N x N` matrix at all (Linformer, Performer, Katharopoulos's linear attention), or hybrids of the two (Longformer, BigBird). These cut the FLOP count to near-linear in `N`, and from the FLOP count alone they look like the answer. But the wall-clock measurements are much less kind: many are no faster than dense attention at the sequence lengths I care about, and some are slower. Why? Because they were optimizing FLOPs, and FLOPs are not the bottleneck. A method can have a beautiful linear FLOP count and still be memory-bound, still moving data around, still not beating a well-tuned dense matmul that the hardware is built to run. And on top of that they pay a quality cost, because they have changed the function. I want the function to stay exact — the same `softmax(QK^T)V` map, up to ordinary floating-point differences — and I want the speedup to come from the thing that actually matters, the HBM traffic.

Two obstacles stand between me and "never touch the `N x N` matrix in HBM." The first is softmax itself: the denominator of row `i` is `sum_j e^{S_ij}`, a sum over *all* `N` keys, so naively I need the whole row of `S` in hand before I can normalize any entry of it. The softmax couples the columns. The second is the backward pass: to get gradients I conventionally need `S` and `P` again, so even if I avoid storing them in the forward pass I seem to need them later, which pushes me right back to writing `N x N` out. Let me take the softmax-coupling obstacle first, because if I cannot break that, nothing else matters.

The numerically stable softmax already carries the seed of the answer. I never compute `e^{x_i}` directly — that overflows to infinity the moment any score exceeds about 89 in float. Instead I subtract the row max first: let `m = max_i x_i`, then `f_i = e^{x_i - m}`, `l = sum_i f_i`, and `softmax_i = f_i / l`. Now the exponent is always `<= 0`, so `f_i in (0, 1]`, no overflow. That is standard. But staring at it, the structure I need is right there: softmax is fully determined by two scalars per row, the max `m` and the normalizer `l`, plus the raw scores. If I had a way to compute `m` and `l` *incrementally*, seeing the scores a chunk at a time and never holding the whole row, the coupling would be broken.

Can I? Suppose I have already seen the first chunk of a row, with its max `m^(1)` and its normalizer `l^(1) = sum over chunk 1 of e^{x - m^(1)}`. Now a second chunk arrives, with max `m^(2)` and `l^(2) = sum over chunk 2 of e^{x - m^(2)}`. The true max of both is `m = max(m^(1), m^(2))`. The trouble is that `l^(1)` and `l^(2)` were each computed relative to their *own* local max, not the common `m`, so I cannot just add them. But I can fix each one up. The normalizer I really want for chunk 1 is `sum e^{x - m}`, and I have `sum e^{x - m^(1)}`; the two differ by a constant factor in the exponent: `e^{x - m} = e^{x - m^(1)} * e^{m^(1) - m}`, so `sum e^{x - m} = e^{m^(1) - m} * l^(1)`. Same for chunk 2. So the combined normalizer is

```
l = e^{m^(1) - m} * l^(1) + e^{m^(2) - m} * l^(2),    m = max(m^(1), m^(2)).
```

That is exact, and `m^(1) - m <= 0` and `m^(2) - m <= 0`, so the rescaling factors are in `(0, 1]` — still no overflow. This is the online-normalizer trick (Milakov & Gimelshein 2018): carry `(m, l)` as running state, and whenever the running max jumps, multiply the running normalizer by `e^{m_old - m_new}` before folding in the new chunk's contribution. They prove by a clean induction that after sweeping the whole row this gives exactly `max_k x_k` and `sum_k e^{x_k - m}`. It is a reduction with a tiny carried summary — algebraic aggregation. So the softmax denominator no longer needs the whole row at once. Good. That is the wall broken on the denominator.

But I do not actually want the denominator in isolation — I want `O = P V`, the softmax-weighted sum of value vectors. So I need to push this incremental idea one step further, through the `* V`. Write the unnormalized output of row `i` over the keys seen so far. Lazy-softmax (Rabe & Staats 2021, who note it goes back to Jang et al. 2019) makes exactly this move: defer the `1/l` division to the very end by the distributive law. Keep a running unnormalized vector and a running normalizer, both starting at zero, and a running max starting at `-inf`; as each key/value `(k_j, v_j)` arrives, compute the score, exponentiate it relative to the running max, add `e^{score - m} * v_j` into the running vector and `e^{score - m}` into the running normalizer, renormalizing both by `e^{m_old - m_new}` whenever the max grows; divide at the end. That gives exact attention with `O(1)` memory per query. So *both* coupling obstacles dissolve with the same running-max rescaling, applied to the output accumulator as well as to the normalizer.

So why is this not already the answer? Because Rabe & Staats were solving a *different* problem than mine. They were minimizing the total memory footprint — the peak bytes resident — and they got that down beautifully, `O(1)` or `O(log n)`. But the number of *HBM accesses* in their implementation is still quadratic, so on a GPU it runs at about the same speed as standard attention, sometimes a touch slower. Footprint and traffic are not the same axis, and traffic is the one that sets my wall-clock time. There is also a structural wrinkle I want to avoid: they summarize each block into its *own* temporary output and only combine all the temporaries at the very end, which means holding one partial output per block. If I am going to fuse this into a single kernel, I would rather keep *one* output per query and update it in place. So the idea is right — incremental softmax over blocks — but I have to re-engineer it around HBM traffic and around an in-place, single-kernel update.

Let me make the traffic concrete by reasoning about it the way the IO-complexity literature does (Aggarwal & Vitter 1988): count reads/writes between fast and slow memory as *the* cost. Standard attention writes `S` (`N^2`), reads it, writes `P` (`N^2`), reads `P` and `V`, writes `O`. That is `Theta(Nd + N^2)` HBM accesses, the `N^2` term swamping the rest. My target is to get the `N^2` term out of the HBM count entirely — to never let an `N x N` object touch HBM. The only place an `N x N` thing can legally exist, then, is on-chip, in SRAM, transiently. And SRAM is tiny. So I cannot hold a whole row of `S` either; I have to hold a *block*.

That forces the shape of the algorithm: tile everything. Split `Q` into row-blocks of `B_r` queries, split `K` and `V` into row-blocks of `B_c` keys/values. A single block of scores `S_ij = Q_i K_j^T` is `B_r x B_c`, which I size to fit in SRAM. For the clean IO-counting version, I load one key/value block `K_j, V_j` into SRAM and sweep all query blocks `Q_i`: each inner step loads `Q_i`, the current `O_i`, and the row statistics `(m_i, l_i)`, computes the score block on-chip, folds it into those statistics and `O_i` with the rescaling identity, and writes only the updated row state back. `S` and `P` never go to HBM. The `N x N` traffic is gone by construction; the only HBM traffic is the inputs, the output, and the `O(N)` row statistics.

Now I need the in-place output update written out exactly, because this is the load-bearing line and a sign error here silently corrupts everything. After processing some key-blocks I hold `O_i`, the *normalized* attention output over the keys seen so far, plus `(m_i, l_i)`. A new key-block gives me `S_ij`, and from it the block-local `m~ = rowmax(S_ij)`, `P~ = exp(S_ij - m~)`, `l~ = rowsum(P~)`. The new running max is `m_new = max(m_i, m~)`. The new running normalizer is `l_new = e^{m_i - m_new} l_i + e^{m~ - m_new} l~`, exactly the online identity. For the output: `O_i` currently equals `(1/l_i) * (sum over old keys of e^{S - m_i} v)`. I want `O_new` to equal `(1/l_new) * (sum over old keys of e^{S - m_new} v + sum over new block of e^{S - m_new} v)`. The old sum, rescaled from base `m_i` to base `m_new`, is `e^{m_i - m_new}` times the old unnormalized output, and the old unnormalized output is `l_i * O_i`. The new block's contribution, rescaled from base `m~` to base `m_new`, is `e^{m~ - m_new} * (P~ V_j)`. So

```
O_new = diag(l_new)^{-1} ( diag(l_i) e^{m_i - m_new} O_i  +  e^{m~ - m_new} P~ V_j ).
```

I should check this actually telescopes to the truth, not just for one step but all the way, because "looks right" is how sign errors survive. Induct on the number of processed key-blocks `j`. Claim: after `j` blocks, `m^(j) = rowmax(S_{:,:j})`, `l^(j) = rowsum(exp(S_{:,:j} - m^(j)))`, and `O^(j) = softmax(S_{:,:j}) V_{:j}` over the first `j` blocks of keys. Base `j = 0`: `O = 0`, `l = 0`, `m = -inf`, vacuously the empty sum. Step: assume it for `j`. The `m, l` updates are the online identity, which gives `m^(j+1) = rowmax(S_{:,:j+1})` and `l^(j+1) = rowsum(exp(S_{:,:j+1} - m^(j+1)))` — that part is exactly Milakov-Gimelshein. For the output, substitute `O^(j) = diag(l^(j))^{-1} exp(S_{:,:j} - m^(j)) V_{:j}` into the update:

```
O^(j+1) = diag(l^(j+1))^{-1} ( diag(l^(j)) e^{m^(j) - m^(j+1)} diag(l^(j))^{-1} exp(S_{:,:j} - m^(j)) V_{:j}
                               + e^{m~ - m^(j+1)} exp(S_{j:j+1} - m~) V_{j:j+1} ).
```

The `diag(l^(j))` and its inverse cancel. In the first term, `e^{m^(j) - m^(j+1)} exp(S_{:,:j} - m^(j)) = exp(S_{:,:j} - m^(j+1))` — the base shifts cleanly from `m^(j)` to `m^(j+1)`. In the second, `e^{m~ - m^(j+1)} exp(S_{j:j+1} - m~) = exp(S_{j:j+1} - m^(j+1))` — same. So

```
O^(j+1) = diag(l^(j+1))^{-1} ( exp(S_{:,:j} - m^(j+1)) V_{:j} + exp(S_{j:j+1} - m^(j+1)) V_{j:j+1} )
        = diag(l^(j+1))^{-1} exp(S_{:,:j+1} - m^(j+1)) V_{:j+1}
        = softmax(S_{:,:j+1}) V_{:j+1}.
```

The claim holds for `j+1`, so by induction at `j = T_c` (all key-blocks), `O = softmax(QK^T) V`, exactly. No approximation anywhere — every step is an identity. And the FLOP count: each inner body is `O(B_r B_c d)` for the two matmuls (`Q_i K_j^T` and `P~ V_j`), run `T_c T_r = (N/B_c)(N/B_r)` times, totaling `O(N^2 d)` — the same arithmetic as dense attention, as it must be for an exact method. The only extra memory is the `O(N)` for `(m, l)`. So I have traded *no* asymptotic FLOPs for a complete removal of `N x N` HBM traffic. That is the whole game.

How big should the blocks be? They are set by the SRAM budget `M`, and I want them as large as will fit, because larger blocks mean fewer passes and fewer HBM accesses. What has to be resident on-chip at once: the key-block `K_j` (`B_c x d`), the value-block `V_j` (`B_c x d`), the query-block `Q_i` (`B_r x d`), and the output-block `O_i` (`B_r x d`) — four `*-by-d` tiles. Budgeting them as the four occupants of `M` gives `B_c = ceil(M / 4d)` and `B_r = ceil(M / 4d)`. There is one more constraint: the score block `S_ij` is `B_r x B_c`, which also has to fit, so `B_r B_c = O(M)`; with `B_c ~ M/d` that forces `B_r <= d`, so I cap `B_r = min(ceil(M/4d), d)`. Concretely with the constraints `B_c = Theta(M/d)`, `B_r = Theta(min(M/d, d))`, the number of passes over `Q` and `O` is `T_c = N/B_c = Theta(Nd/M)`. Each element of `K, V` is loaded once (`Theta(Nd)`), and I make `T_c` passes over `Q, O` loading `Theta(Nd)` each, so total HBM accesses are `Theta(Nd * T_c) = Theta(N^2 d^2 / M)`. Compare standard attention's `Theta(N^2)`: the ratio is about `d^2 / M`, and with `d` in the 64-128 range and `M` around a hundred kilobytes, `d^2` is many times smaller than `M`. That is the source of the speedup — many-fold fewer HBM accesses on the exact same function. I can even predict the block-size sweep before running it: as `B_c` grows, `T_c` shrinks, HBM accesses drop, runtime drops — until the block gets large enough that I am no longer memory-bound and other costs take over, or large enough that it stops fitting in SRAM. So there is a sweet spot, not "bigger is always better."

Is `Theta(N^2 d^2 / M)` actually good, or could a cleverer algorithm do better? Let me lower-bound it. Suppose some exact algorithm did `o(N^2 d^2 / M)` HBM accesses for *all* `M` in `[d, Nd]`. Take `M = Theta(Nd)` — SRAM as big as the whole input. Then that bound is `o(N^2 d^2 / Nd) = o(Nd)`. But the inputs `Q, K, V` and the output `O` are each `Nd` and they live in HBM; just reading the input and writing the output is `Omega(Nd)`. Contradiction. So no exact attention algorithm can uniformly beat `Theta(N^2 d^2 / M)` across that whole range of SRAM sizes — what I have is IO-optimal up to constants in that uniform-over-`M` sense. That settles that I am not leaving an asymptotic factor on the table.

Now the second obstacle I deferred: the backward pass. If I have to read an `N x N` `P` from HBM to compute gradients, I have reintroduced exactly the traffic I just eliminated. I do not store `S` or `P` in the forward pass — only `O`, the row softmax statistics `(m, l)` or equivalently a log-sum-exp per row, and (if I dropout) the RNG state. So in the backward pass I will *recompute* `S` and `P` on-chip from blocks of `Q, K, V` and the saved statistics. This is a form of selective gradient checkpointing (Griewank & Walther 2008; Chen et al. 2016), but with a twist that matters: ordinary checkpointing trades *speed* for memory — you recompute, so you do more work and run slower, in exchange for a smaller footprint. Here the recomputation is *also faster*, because the thing I am avoiding is not a few FLOPs, it is reading a giant `N x N` matrix from slow HBM. Recomputing `P` from small SRAM-resident blocks costs extra arithmetic but saves enormous HBM traffic, and on a memory-bound operation that is a net win. The same logic — count HBM accesses, not FLOPs — flips checkpointing from a memory-vs-speed tradeoff into a strict improvement.

Let me actually derive the backward gradients, because I want them exact and I want them expressible in blocks with `O(N)` extra memory. Let the loss be `phi`, the output gradient `dO = d phi / d O`, and the scaled score be `S_ij = tau q_i . k_j`. I want `dQ, dK, dV`. With `L_i = sum_j e^{S_ij}` the full softmax normalizer for row `i`, and `P_ij = e^{S_ij} / L_i`, the output is `o_i = sum_j P_ij v_j`. The easy one: `O = P V`, so `dV = P^T dO`, i.e. `dv_j = sum_i P_ij do_i`. Since I can recompute `P_ij` block by block from the saved statistics, this is a running sum, no `N x N` stored. Next, `dP`: from `o_i = sum_j P_ij v_j`, `dP = dO V^T`, so `dP_ij = do_i . v_j`. Then I need to push `dP` back through the row-softmax. The Jacobian of `y = softmax(x)` is `diag(y) - y y^T`, so `dS_{i:} = (diag(P_{i:}) - P_{i:} P_{i:}^T) dP_{i:} = P_{i:} ∘ dP_{i:} - (P_{i:}^T dP_{i:}) P_{i:}`, where `∘` is elementwise. Define `D_i = P_{i:}^T dP_{i:}`. This looks like it needs a length-`N` reduction over the row, which would be awkward on-chip. But watch:

```
D_i = sum_j P_ij dP_ij = sum_j P_ij (do_i . v_j) = do_i . (sum_j P_ij v_j) = do_i . o_i.
```

The length-`N` reduction collapses into a single length-`d` dot product `do_i . o_i` — and I already have `o_i` from the forward pass. That is the key simplification that keeps the backward pass on-chip. So `dS_ij = P_ij (dP_ij - D_i)`. Finally, since `S_ij = tau q_i . k_j`, `dq_i = tau * sum_j dS_ij k_j` and `dk_j = tau * sum_i dS_ij q_i`. Every one of these — `dV, D, dQ, dK` — is a sum I can accumulate block by block, recomputing `P_ij` on-chip from `Q, K, V` blocks and the saved row statistics, with only `O(N)` (really `O(d)` per block plus the `D` vector) extra memory. And the dropout mask, which would be `N x N`, I never store: I save the RNG state from the forward pass and regenerate the exact same mask in the backward pass. So the backward pass is a block version of the forward derivation, same `Theta(N^2 d^2 / M)` HBM accesses, exact gradients.

There is one more thing fusion buys me that I should not gloss over. Tiling lets me put the *entire* pipeline — load `Q, K, V` blocks, matmul, mask, softmax-rescale, matmul, write `O` — into a *single* GPU kernel. The reason ordinary kernel fusion failed for attention is that in training you have to write the intermediates to HBM for the backward pass, so fusion does not actually remove the traffic. But because I have arranged to *recompute* the intermediates in the backward pass rather than store them, I am free to fuse the whole forward pass with no intermediate spills at all. The masking and dropout, which would otherwise be separate memory-bound elementwise kernels each round-tripping `N x N` through HBM, now happen on the score block while it is already in SRAM, for free. Fusion and recomputation are not two independent tricks; recomputation is what *makes* the fusion pay off in training.

Now to land this in real kernel code, and a couple of practical decisions fall out that the clean math does not dictate. First, the loop order. The correctness proof I did had the key/value blocks on the outer loop and the query blocks on the inner — that is natural for the IO accounting, "load each `K_j, V_j` once, sweep all `Q_i`." But if I write the kernel that way, many programs (one per `(i, j)`) all want to update the same output rows `O_i`, which means writing partial `O_i` back to HBM and reloading it, or coordinating across programs. Cleaner: make the *query block* the unit of parallelism — one kernel program per query-block `start_m` — and loop over the key/value blocks *inside* that program. Then each program owns its query rows entirely: it keeps `m_i, l_i, acc` (the running output) in registers/SRAM for the whole life of the program, sweeps all `K_j, V_j`, and writes `O_i` exactly once at the end. No cross-program writes to `O`, no atomics, no reloading partial outputs. Same computation, much friendlier to the hardware.

Second, since each program normalizes its own rows only at the very end, I can defer the `1/l` division entirely. Instead of recomputing `diag(l_new)^{-1}` on every block, I keep the *unnormalized* accumulator `acc` and the running `l_i`, and on each new block I rescale `acc` by `alpha = e^{m_i - m_new}` (the same factor that fixes `l`) and add `P~ V_j`; only after the loop do I divide `acc` by `l_i` once. Concretely per block: `m_new = max(m_i, rowmax(S_ij))`, `alpha = exp(m_i - m_new)`, `p = exp(S_ij - m_new)`, `l_i = l_i * alpha + rowsum(p)`, `acc = acc * alpha + p @ V_j`, `m_i = m_new`; then at the end `acc = acc / l_i`. For training I also need a row statistic for the backward recomputation, and the compact one is the log-sum-exp of the scaled score row: in base-2 units that is `m_i + log2(l_i)`. This is the same telescoping update — I have just pulled the single normalization out of the loop, which removes a divide per block and a bunch of non-matmul work. Fewer non-matmul operations matter because they are the part that does not run on the fast tensor-core matmul units.

Third, the exponential. Calling a generic `exp` (natural-base) per element is not the cheapest thing the hardware offers; GPUs have a fast base-2 exponential `exp2(x) = 2^x` as a near-native instruction. Since `e^x = 2^{x log2 e}` with `log2(e) = 1.44269504...`, I can keep the kernel's score state in base-2 units: compute `qk = (QK^T) * sm_scale * 1.44269504`, track row maxima in those same units, then use `alpha = exp2(m_i - m_new)` and `p = exp2(qk - m_new)`. This is the same softmax, just expressed in the units the fast exponential wants.

Fourth, causality. For a causal mask, query `i` may only attend to keys `j <= i`. Rather than compute every block and then mask the upper triangle to `-inf`, I bound the inner loop: for query-block `start_m`, the last key that can possibly be attended is at position `(start_m + 1) * BLOCK_M - 1`, so I only iterate `start_n` up to `hi = (start_m + 1) * BLOCK_M`. Key-blocks entirely above the diagonal are skipped, not masked — that nearly halves the work for causal attention. Within the boundary block that straddles the diagonal, I still need the per-element mask, setting `S_ij` to `-inf` where `query_pos < key_pos` so those entries contribute zero after the exponential.

Fifth, precision. The scores, the running max, the normalizer, and the output accumulator I keep in fp32 — the softmax sums and the rescaling need the dynamic range, and accumulating an output over many blocks in fp16 would lose bits. The inputs `Q, K, V` and the final `O` are fp16, and I cast the probability block `p` back to fp16 before the `p @ V_j` matmul so it runs on the tensor cores at full throughput. Block sizes I take uniform, `BLOCK_M = BLOCK_N = 64`, which comfortably fits the working set in SRAM for the head dimensions in play; the launch grid is one program per query-block per (batch, head), `grid = (cdiv(seqlen, BLOCK_M), batch * nheads)`.

Putting all of that into the kernel, filling the empty slot from the harness — one program per query block, loop over key/value blocks, online-softmax rescale into an fp32 accumulator, a stored row log-sum-exp for backward recomputation, and a single normalization at the end:

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out, Lse,
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
    # one program owns one block of BLOCK_M query rows, for one (batch, head)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh
    lse_offset = off_hz * seqlen

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # this program's query positions
    offs_n = tl.arange(0, BLOCK_N)                       # key positions within a K/V block
    offs_d = tl.arange(0, BLOCK_DMODEL)                  # head dimension

    # load this query block ONCE; it stays on chip for the whole loop
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    qk_scale = sm_scale * 1.44269504                         # log2(e), for exp2 softmax

    # running online-softmax state, all in fp32
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")   # running row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running normalizer
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)    # running UNnormalized output

    # causal: only key-blocks up to the diagonal can contribute -> skip the rest
    hi = tl.minimum((start_m + 1) * BLOCK_M, seqlen) if IS_CAUSAL else seqlen
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # S_ij = Q_i K_j^T * scale * log2(e), computed on chip, never written to HBM
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where((start_n + offs_n)[None, :] < seqlen, qk, float("-inf"))
        if IS_CAUSAL:                                            # boundary-block masking only
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # online softmax: grow the running max, rescale, fold in this block
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)                       # 2^{m_old - m_new}
        p = tl.math.exp2(qk - m_new[:, None])                   # 2^{S - m_new}
        l_i = l_i * alpha + tl.sum(p, axis=1)                    # rescale normalizer, add block
        acc = acc * alpha[:, None]                               # rescale running output
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)                      # + P~ V_j (fp16 matmul, fp32 acc)
        m_i = m_new

    acc = acc / l_i[:, None]                                     # single 1/l normalization at the end
    lse_ptrs = Lse + lse_offset + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < seqlen)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < seqlen)   # write O once


def attention_forward(q, k, v, causal=True, sm_scale=None):
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    lse = torch.empty((batch * nheads, seqlen), device=q.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N = 64, 64                                    # tiles sized to fit SRAM
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)        # one program per query block
    _attn_fwd[grid](
        q, k, v, o, lse, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o, lse
```

Let me trace the causal chain back. I started stuck: attention is slow on long sequences, and the approximate-attention crowd cut FLOPs without cutting wall-clock time. Profiling the standard kernel against the arithmetic-intensity model showed why — attention is memory-bound, dominated by writing and re-reading the `N x N` matrices `S` and `P` across slow HBM, so the cost that matters is HBM accesses, not FLOPs. The plan became: compute exact attention while never letting an `N x N` object touch HBM. Softmax's row-coupling was the obstacle; the numerically-stable softmax depends only on a running max and normalizer, and the online-normalizer identity lets me combine block partials exactly by rescaling with `e^{m_old - m_new}` — so the denominator, and then (pushing through `* V`) the whole output, can be built one block at a time. That forced a tiled algorithm: split `Q, K, V` into SRAM-sized blocks, accumulate `(m, l, O)` over key-blocks with the rescaling identity, never spilling scores. An induction confirmed the in-place update telescopes to exactly `softmax(QK^T)V` with `O(N^2 d)` FLOPs and `O(N)` extra memory; the IO analysis gave `Theta(N^2 d^2 / M)` HBM accesses, many-fold below the standard `Theta(N^2)`, and a lower-bound argument ruled out uniformly improving that asymptotic over the whole SRAM range. The backward pass threatened to reintroduce `N x N` traffic, but storing only `O` and row softmax statistics and recomputing `S, P` on-chip — with the simplification `D_i = do_i . o_i` collapsing a length-`N` reduction to a length-`d` dot product — keeps it block-wise and exact; and because the saved cost is HBM traffic not FLOPs, this recomputation is faster, not a memory-for-speed trade, which in turn is what lets the forward pipeline fuse without spilling the score or probability matrices. Realizing it as code, I made the query block the unit of parallelism so each program owns its output rows and normalizes once at the end, used `exp2` with `log2(e)` for the exponential, stored the base-2 log-sum-exp for recomputation, skipped fully-masked key-blocks under causality, and kept fp32 accumulators with fp16 tensor-core matmuls — landing on a single fused attention kernel that computes exact attention with `N x N` traffic eliminated.
