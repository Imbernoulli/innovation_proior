# FlashAttention-2 synthesis (grounded)

## Verified
- arXiv 2307.08691, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", Tri Dao.
- Canonical impl: Dao-AILab/flash-attention; algorithmic reference = OpenAI Triton tutorial 06-fused-attention.py (loop swap + seq parallelism first done by Phil Tillet in Triton).
- Prior art: FlashAttention (FA1), Dao et al. 2022, arXiv 2205.14135.

## GPU model (background, knowable pre-method)
- Memory hierarchy: HBM (40-80GB, ~1.5-2 TB/s) vs on-chip SRAM/shared memory (192KB/SM on A100, ~19 TB/s, 108 SMs). SRAM ~10x faster, tiny.
- Tensor Cores: A100 312 TFLOPs/s FP16/BF16 matmul vs 19.5 TFLOPs/s non-matmul FP32 -> a non-matmul FLOP is ~16x more expensive than a matmul FLOP. KEY for FA2 motivation.
- Execution: kernel -> threads -> warps (32 threads) -> thread blocks -> SMs. Warps in a block share via shared memory (slow-ish, needs sync); threads in a warp share via fast shuffle. Thread blocks on different SMs are independent. Occupancy = fraction of GPU resources used; need many thread blocks (>=~80 on A100) to fill SMs.

## Standard attention (baseline)
- Q,K,V in R^{N×d}. S=QK^T (N×N), P=softmax(S) rowwise, O=PV (N×d). Backward: dV=P^T dO, dP=dO V^T, dS=dsoftmax(dP), dQ=dS K, dK=Q dS^T. softmax grad: if p=softmax(s), ds=(diag(p)-pp^T)dp.
- Materializes S,P to HBM = O(N^2) memory + many HBM reads/writes; memory-bandwidth-bound -> slow. Must store P (N×N) for backward.

## FlashAttention (FA1) — prior art to derive in full
- Two classical techniques: TILING (process Q,K,V in blocks that fit SRAM) + RECOMPUTATION (recompute S,P in backward instead of storing).
- Online softmax (Milakov & Gimelshein 2018; Rabe & Staats 2021): compute softmax blockwise and rescale running output, exact, no approximation.
- The math (2 column blocks S^(1), S^(2) of a row block):
  m^(1)=rowmax(S^(1)); l^(1)=rowsum(e^{S^(1)-m^(1)}); O^(1)=diag(l^(1))^{-1} e^{S^(1)-m^(1)} V^(1)
  m^(2)=max(m^(1), rowmax(S^(2))); l^(2)=e^{m^(1)-m^(2)} l^(1) + rowsum(e^{S^(2)-m^(2)})
  O^(2)=diag(l^(1)/l^(2))^{-1} O^(1) + diag(l^(2))^{-1} e^{S^(2)-m^(2)} V^(2) = O (correct full softmax)
- FA1 rescales the *output accumulator* O by diag(l)^{-1} at EVERY block step (the diag(l^(1)/l^(2))^{-1} factor).
- FA1 forward: OUTER loop over K,V blocks (columns), INNER over Q blocks (rows). Parallelizes over batch × heads only (one thread block per head). Within block: "split-K" warp partition — K,V split across 4 warps, Q shared; each warp gets a slice of QK^T then must multiply by V slice and COMMUNICATE through shared memory to sum -> shared mem reads/writes + sync.
- Results: 2-4x speedup over standard attention, 10-20x memory saving (memory linear in N). But forward only 30-50% of peak FLOPs/s, backward 25-35%, vs GEMM 80-90%.

## Why FA1 leaves performance on the table (the FA2 motivation)
1. Too many non-matmul FLOPs: rescaling the output by diag(l)^{-1} every step is elementwise (non-matmul) work; non-matmul is 16x costlier per FLOP, so even a "small fraction" of non-matmul FLOPs dominates wall-clock.
2. Low occupancy for long sequences: parallelizing only over batch×heads means few thread blocks when batch is small (long seq -> small batch), so SMs sit idle.
3. Wasteful warp communication: split-K forces all warps to write partial QK^T·V results to shared memory and sync to reduce.

## FA2 — the three improvements (derivation)
### (1) Reduce non-matmul FLOPs (algorithm tweaks)
- Don't rescale BOTH terms by diag(l^(2))^{-1} every step. Keep an UNSCALED output accumulator:
  Õ^(2) = diag(e^{m^(1)-m^(2)})^{-1} Õ^(1) + e^{S^(2)-m^(2)} V^(2)   [only multiply old acc by the max-correction e^{m^(1)-m^(2)}, NOT by l]
  Only at the very end: O = diag(l^(last))^{-1} Õ^(last). Saves the per-step division by l.
- Store only the logsumexp L^(j) = m^(j) + log(l^(j)) for backward, instead of both m and l. (Backward recomputes P_i^(j) = exp(S_ij - L_i).)
- Forward alg (Alg 1): outer loop i over Q row blocks; init O_i^(0)=0, l_i^(0)=0, m_i^(0)=-inf; inner loop j over K/V col blocks:
  S_i^(j) = Q_i K_j^T
  m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
  P̃_i^(j) = exp(S_i^(j) - m_i^(j))   [pointwise]
  l_i^(j) = e^{m_i^(j-1) - m_i^(j)} l_i^(j-1) + rowsum(P̃_i^(j))
  O_i^(j) = diag(e^{m_i^(j-1) - m_i^(j)})^{-1} O_i^(j-1) + P̃_i^(j) V_j   [NOTE: diag(e^{m diff})^{-1} = multiply by e^{m_i^(j-1)-m_i^(j)} ... wait sign: it's diag(e^{m^(j-1)-m^(j)})^{-1}? Alg line 85 writes diag(e^{m_i^(j-1)-m_i^(j)})^{-1} O_i^(j-1). Since m^(j)>=m^(j-1), m^(j-1)-m^(j)<=0, e^{...}<=1, and inverse makes it >=1?? Re-check below.]
  After inner loop: O_i = diag(l_i^(T_c))^{-1} O_i^(T_c); L_i = m_i^(T_c) + log(l_i^(T_c)).
- SIGN CHECK on the O accumulator rescale: We want old acc (carrying e^{S^(j-1)-m^(j-1)}) re-based to the new max m^(j). e^{S^(j-1)-m^(j-1)} -> e^{S^(j-1)-m^(j)} requires multiplying by e^{m^(j-1)-m^(j)} (a factor <=1, since m^(j)>=m^(j-1)). The Triton kernel does exactly: alpha = exp(m_i - m_ij) where m_ij=new max>=m_i=old -> alpha<=1, acc = acc*alpha. So the correct factor is e^{m^(j-1)-m^(j)} (NOT its inverse). The paper's Alg-1 line writes diag(e^{m^(j-1)-m^(j)})^{-1} which would be e^{m^(j)-m^(j-1)}>=1 — this is a known typo/notation slip in the algorithm box; the body text eq (line 58) and the Triton ref both give Õ^(2)=diag(e^{m^(1)-m^(2)})^{-1} Õ^(1)+... Hmm the body ALSO writes ^{-1}. Let me resolve: body line 33 (FA1 form) O^(2)=diag(l^(1)/l^(2))^{-1}O^(1)+diag(l^(2))^{-1}e^{S^(2)-m^(2)}V^(2). FA1 O^(1)=diag(l^(1))^{-1}e^{S^(1)-m^(1)}V^(1) is ALREADY scaled by l and by m^(1). The diag(l^(1)/l^(2))^{-1}=diag(l^(2)/l^(1)) rebases the l-normalization; the m-rebase e^{m^(1)-m^(2)} is folded in because... Actually use the UNSCALED form (line 58): Õ^(1)=e^{S^(1)-m^(1)}V^(1) (no l). Then Õ^(2)=diag(e^{m^(1)-m^(2)})^{-1} Õ^(1) + e^{S^(2)-m^(2)}V^(2), and the text's own simplification on line 58 states this equals e^{s^(1)-m}V^(1)+e^{s^(2)-m}V^(2) with m=m^(2). For that equality: diag(e^{m^(1)-m^(2)})^{-1} · e^{S^(1)-m^(1)} must = e^{S^(1)-m^(2)}=e^{S^(1)-m}. So diag(e^{m^(1)-m^(2)})^{-1}·e^{-m^(1)} = e^{-m^(2)} => diag(e^{m^(1)-m^(2)})^{-1}=e^{-m^(2)+m^(1)}=e^{m^(1)-m^(2)}. So the paper's "diag(e^{m^(1)-m^(2)})^{-1}" NOTATION actually DENOTES the scalar e^{m^(1)-m^(2)} (they treat diag(x)^{-1} loosely). The numerically correct multiplicative factor is e^{m^(1)-m^(2)} = e^{m_prev - m_new} <= 1. CONFIRMED consistent with Triton alpha=exp(m_i - m_ij). GOOD. In reasoning I'll write the factor cleanly as e^{m^(j-1)-m^(j)} (= alpha, <=1) to avoid the confusing diag-inverse notation.

### (2) Parallelize over sequence length + swap loop order
- FA2 makes the OUTER loop over Q row blocks (rows), inner over K/V blocks. This is the swap from FA1 (which looped outer over K/V).
- Why swap: with outer loop over rows, each Q row block is INDEPENDENT (its softmax normalization is self-contained per row), so different row blocks -> different thread blocks, no communication. The outer loop is "embarrassingly parallel".
- Add a third parallelization axis: sequence length (row blocks), on top of batch and heads. Fills SMs when batch×heads is small (long-seq regime). Grid = (num row blocks, batch*heads).
- Backward: parallelize over column blocks (each thread block = one K/V column block); the only shared state is dQ accumulation across column blocks -> use atomic adds to HBM.
- Loop swap + seq-parallelism credited to Phil Tillet's Triton fused-attention.

### (3) Warp work partitioning: split-Q instead of split-K
- FA1 "split-K": split K,V across the 4 warps, Q shared. Each warp computes a slice of QK^T, multiplies by its V slice, then all warps write partials to shared memory + sync + reduce -> shared-mem traffic.
- FA2 "split-Q": split Q across the 4 warps, K,V shared by all warps. Each warp computes its rows of QK^T, then multiplies by the (shared) full V to get its own output rows directly -> NO inter-warp communication, no shared-mem reduction.
- Backward still needs some sync (more complex dependencies among Q,K,V,O,dO,dQ,dK,dV) but also avoids split-K.

## Other details
- Causal masking: blocks where all col idx > all row idx are entirely masked -> SKIP them (~half the blocks for large N) -> 1.7-1.8x speedup. Blocks where row idx strictly > col idx need NO mask. Only the diagonal block per row needs the elementwise mask applied.
- MQA/GQA: don't duplicate K/V heads; manipulate head indices implicitly; backward sums dK,dV across the query heads sharing each KV head.
- Block sizes: {64,128}×{64,128} depending on head dim d and SRAM size; too big -> register spilling or exceeds shared memory.
- Correctness: exact O = softmax(QK^T)V, no approximation. O(N^2 d) FLOPs, O(N) extra memory (store L).

## Design-decision -> why
- Unscaled O accumulator + final rescale: removes per-step elementwise division (non-matmul, 16x costly) -> more time on matmul.
- Store logsumexp L=m+log(l) only: one vector not two; backward recomputes P=exp(S-L).
- Outer loop over Q rows: makes row blocks independent so they parallelize across thread blocks with zero communication.
- Parallelize over seq length: occupancy when batch×heads small (long context).
- split-Q warp partition: each warp owns output rows fully, no cross-warp reduction through shared memory.
- Causal block skipping: ~half the work is provably zero, skip it.

## Scaffold correspondence (pre-method)
- Pre-method scaffold: an exact-attention primitive with forward(Q,K,V)->O and a backward, plus a hardware notion of "tile/block" loops and a parallel grid. Bodies (how to loop, what to rescale, how to assign work to blocks/warps) are the empty slots.
- Final code: blocked online-softmax forward with outer loop over Q tiles, inner over K/V tiles, unscaled accumulator + final normalize, store logsumexp; grid over (row blocks, batch*heads); split-Q warps. (Expressed in PyTorch/Triton-style pseudocode grounded in Triton tutorial + flash-attention repo.)
