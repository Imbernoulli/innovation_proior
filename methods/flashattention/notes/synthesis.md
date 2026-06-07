# FlashAttention — Synthesis notes (Phase 1.5)

## Pain point / research question
Self-attention on sequence length N: S = QK^T (N×N), P = softmax(S) row-wise, O = PV.
Time AND memory are O(N^2). The memory materialization of S, P to HBM is the real killer.
By 2022 Transformers want long context but attention is the bottleneck. Many "efficient
attention" methods reduce FLOPs to linear/near-linear (sparse, low-rank) but DON'T give
wall-clock speedup and aren't widely adopted. Why? They optimize FLOPs, ignore memory IO.

Key diagnostic: most attention ops (softmax, dropout, masking — elementwise/reduction) are
MEMORY-BOUND, not compute-bound. Standard attention reads/writes the N×N matrices S and P to
HBM repeatedly. The runtime is dominated by HBM traffic, not by the matmul FLOPs.

So the right objective is not "fewer FLOPs" but "fewer HBM accesses" — IO-awareness.

## GPU memory hierarchy (the load-bearing hardware fact)
- A100: HBM 40-80GB @ 1.5-2.0 TB/s; SRAM 192KB per SM × 108 SMs @ ~19 TB/s.
- SRAM ~10× faster bandwidth but orders of magnitude smaller.
- Compute has outpaced memory bandwidth → memory-bound ops dominate.
- Arithmetic intensity (Williams roofline 2009): FLOPs per byte. Compute-bound = big matmul/conv;
  memory-bound = elementwise + reductions (softmax, layernorm, dropout).
- Kernel fusion is the standard remedy for memory-bound ops: load input once, do all ops, write
  once. BUT in training, intermediates must be saved to HBM for the backward pass — which defeats
  naive fusion. This is the gap FlashAttention must close.

## Ancestors (load-bearing)
1. **Standard attention** (Vaswani 2017): the baseline. Materializes S, P → O(N^2) memory and
   Θ(Nd + N^2) HBM accesses. Backward needs S, P stored → again O(N^2) memory / Θ(Nd+N^2) IO.
2. **Online softmax / online normalizer calculation** (Milakov & Gimelshein 2018): compute
   softmax in ONE pass over the row by maintaining a running max m and running denominator d.
   Recurrence when reading a new value x_j:
       m_new = max(m, x_j),  d_new = d * e^{m - m_new} + e^{x_j - m_new}.
   This is the algebraic-aggregation trick. It decouples the columns so softmax need not see the
   whole row at once. (FlashAttention generalizes this to blocks of columns, and also rescales the
   *output accumulator*, not just the denominator.)
3. **Self-attention does not need O(n^2) memory** (Rabe & Staats 2021): used online softmax to
   show attention needs only linear EXTRA memory. BUT they target memory FOOTPRINT, not memory
   ACCESSES; their method is ~same speed or slightly slower than standard attention. Their backward
   uses gradient checkpointing (recompute via the forward), and they keep K temporary outputs (one
   per block) then combine. FlashAttention differs on all three: targets IO not footprint;
   incrementally updates ONE output copy (no K copies); derives the backward analytically instead
   of generic checkpointing.
4. **IO complexity / external-memory model** (Aggarwal & Vitter 1988): count transfers between
   slow and fast memory. The lens that makes "IO-aware" rigorous.
5. **Reformer / Kitaev 2020**: sparse-approximation, cited as also using the softmax-scaling trick.
6. **Block-sparse / butterfly** (Child 2019 sparse transformer; Zaheer BigBird; Dao pixelated
   butterfly 2021, kaleidoscope 2020): the sparsity patterns for the block-sparse extension. s =
   fraction of nonzero blocks; common s = N^{-1/2} or N^{-1} log N.
7. Gradient checkpointing (Griewank 2008; Chen 2016): recompute activations in backward to save
   memory — FlashAttention's recomputation is a selective form of this, but it's a SPEEDUP here
   because the recompute happens on-chip and avoids HBM reads of the N×N matrix.

## The two technical challenges and the two techniques
Goal: never read/write the N×N attention matrix to/from HBM.
Requires (i) compute softmax reduction without the whole row; (ii) don't store N×N for backward.
- (i) → **TILING**: split Q,K,V into blocks, make passes, incrementally do softmax via online-
  softmax rescaling of both the denominator AND the output accumulator.
- (ii) → **RECOMPUTATION**: store O and the softmax stats (m, ℓ) — O(N) — and recompute S,P
  on-chip in the backward from blocks of Q,K,V in SRAM. More FLOPs but far fewer HBM accesses → faster.
- **Kernel fusion**: tiling lets the whole thing be ONE CUDA kernel (matmul→softmax→mask→
  dropout→matmul) so inputs are loaded once.

## Forward algorithm (Alg 1) — exact statics
Block sizes: B_c = ceil(M/4d), B_r = min(ceil(M/4d), d). [the /4 and the min(...,d) are the
SRAM-budget constants — see "why these block sizes" below.]
T_r = ceil(N/B_r) row blocks of Q; T_c = ceil(N/B_c) col blocks of K,V.
Init O=0 (N×d), ℓ=0 (N), m=-inf (N) in HBM.
Outer loop j over K,V blocks; inner loop i over Q blocks:
  S_ij = Q_i K_j^T  (B_r × B_c), on chip
  m̃_ij = rowmax(S_ij); P̃_ij = exp(S_ij - m̃_ij); ℓ̃_ij = rowsum(P̃_ij)
  m_i^new = max(m_i, m̃_ij)
  ℓ_i^new = e^{m_i - m_i^new} ℓ_i + e^{m̃_ij - m_i^new} ℓ̃_ij
  O_i ← diag(ℓ_i^new)^{-1} ( diag(ℓ_i) e^{m_i - m_i^new} O_i + e^{m̃_ij - m_i^new} P̃_ij V_j )
  write ℓ_i ← ℓ_i^new, m_i ← m_i^new
The O_i update is the crux: the previous accumulator O_i was normalized by the OLD ℓ_i, so
multiply back by diag(ℓ_i) to un-normalize, rescale by e^{m_i - m_i^new} for the new max, add the
new block's contribution e^{m̃_ij - m_i^new} P̃_ij V_j, then renormalize by diag(ℓ_i^new)^{-1}.

Full forward (Alg 2) adds: softmax scale τ (=1/√d), mask (set entries to -inf), dropout (with
saved RNG state R), and saves O, ℓ, m, R for backward.

## Correctness proof (induction on j) — the inline derivation
Claim after j-th outer iter, with K_:j, V_:j first jB_c rows:
  m^(j) = rowmax(S_:,:j),  ℓ^(j) = rowsum(exp(S_:,:j - m^(j))),  O^(j) = P_:,:j V_:j.
Base j=0 trivial. Inductive step: m^(j+1)=max(m^(j), m̃)=rowmax over j+1 blocks. ℓ likewise by the
algebraic-aggregation identity. For O, substitute O^(j)=P_:,:j V_:j and expand:
  O^(j+1) = diag(ℓ^(j+1))^{-1}( diag(ℓ^(j)) e^{m^(j)-m^(j+1)} O^(j) + e^{m̃-m^(j+1)} exp(S_{j:j+1}-m̃) V_{j:j+1} )
substitute O^(j) = diag(ℓ^(j))^{-1} exp(S_:,:j - m^(j)) V_:j; the diag(ℓ^(j)) cancels; the
e^{m^(j)-m^(j+1)} and e^{-m^(j)} combine into e^{-m^(j+1)}; the second term's e^{m̃-m^(j+1)}e^{-m̃}=
e^{-m^(j+1)}. Pull out e^{-m^(j+1)}, recombine the two column-slabs into the full exp(S_:,:j+1 -
m^(j+1)) times [V_:j; V_{j:j+1}] = softmax(S_:,:j+1) V_:j+1. QED. At j=T_c, O = softmax(QK^T)V.
FLOPs O(N^2 d), extra memory O(N).

## Memory-efficient backward derivation (the analytic backward, Appendix)
Loss φ; dO = ∂φ/∂O. L_i = Σ_j e^{q_i·k_j} (normalizer, omitting max-shift for derivation).
o_i = Σ_j (e^{q_i·k_j}/L_i) v_j.
- dV = P^T dO  ⇒  dv_j = Σ_i P_ij do_i = Σ_i (e^{q_i·k_j}/L_i) do_i.
- dP = dO V^T  ⇒  dP_ij = do_i · v_j.
- Softmax Jacobian: y=softmax(x), J = diag(y) - y y^T. So
  dS_i: = (diag(P_i:) - P_i: P_i:^T) dP_i: = P_i: ∘ dP_i: - (P_i:^T dP_i:) P_i:.
- Define D_i = P_i:^T dP_i: = Σ_j (e^{q_i·k_j}/L_i) do_i·v_j = do_i · (Σ_j (e^{q_i·k_j}/L_i) v_j)
  = do_i · o_i.  ← KEY: D_i is just the dot product of do_i and o_i (size-d vectors), no need to
  reduce over the size-N row P_i:. This is what makes the backward memory-efficient.
- Then dS_ij = P_ij (dP_ij - D_i).
- dq_i = Σ_j dS_ij k_j = Σ_j P_ij (dP_ij - D_i) k_j.
- dk_j = Σ_i dS_ij q_i = Σ_i P_ij (dP_ij - D_i) q_i.
With τ scaling, dq and dk pick up the factor τ. Backward is O(N^2) FLOPs, O(N) extra memory.

Backward FlashAttention (Alg 4): outer loop j over K,V blocks (accumulate dK_j, dV_j in SRAM);
inner loop i over Q blocks. Recompute P_ij = diag(ℓ_i)^{-1} exp(S_ij^masked - m_i) from stored
ℓ_i, m_i (no need to store P!). Regenerate dropout mask Z_ij from RNG state R.
  dV_j += (P_ij^drop)^T dO_i
  dP_ij^drop = dO_i V_j^T ; dP_ij = dP_ij^drop ∘ Z_ij
  D_i = rowsum(dO_i ∘ O_i)
  dS_ij = P_ij ∘ (dP_ij - D_i)
  dQ_i += τ dS_ij K_j  (accumulate to HBM)
  dK_j += τ dS_ij^T Q_i  (accumulate on SRAM)
Two observations: (1) don't store dropout mask, regen from RNG → O(N) memory; (2) compute D_i =
do_i·o_i not by reducing over size-N rows.

## IO complexity (the central theorem) — full argument
Standard attention: S=QK^T reads Q,K (Nd) writes S (N^2) → Θ(Nd+N^2); softmax reads S writes P →
Θ(N^2); O=PV reads P,V writes O → Θ(Nd+N^2). Total Θ(Nd + N^2).
FlashAttention: each element of K,V loaded once (outer loop). T_c passes over Q and O, each pass
loads all of Q,O = Θ(Nd). So HBM accesses = Θ(Nd + Nd·T_c) = Θ(Nd·T_c).
Block-size constraints (all must fit in SRAM of size M): B_c d = O(M) ⇒ B_c = O(M/d); B_r d = O(M)
⇒ B_r = O(M/d); B_r B_c = O(M). So set B_c = Θ(M/d), B_r = Θ(min(M/d, M/B_c)) = Θ(min(M/d, d)).
Then T_c = N/B_c = Θ(Nd/M). HBM accesses = Θ(Nd · Nd/M) = Θ(N^2 d^2 / M).
Since typically d^2 ≪ M (d=64-128, M~100KB), N^2 d^2/M ≪ N^2 → many-fold fewer HBM accesses
(up to ~9×). Backward same: Θ(N^2 d^2/M).

Lower bound (Prop): no exact attention algorithm does o(N^2 d^2/M) HBM accesses for ALL M in
[d, Nd]. Proof: at M=Θ(Nd), o(N^2d^2/M)=o(Nd), but inputs Q,K,V and output O have total size Θ(Nd)
and start in HBM, so ANY exact algorithm must touch ≥ Ω(Nd) — contradiction. (Subrange lower bound,
standard in streaming-algorithm literature, Woodruff 2004.)

## Block-sparse extension
Mask M̃ ∈ {0,1}^{N×N} in block form. Skip zero blocks in the loops. IO complexity
Θ(Nd + N^2 d^2 M^{-1} s), s = fraction of nonzero blocks. With s=N^{-1/2} → Θ(N√N); s=N^{-1}logN
→ Θ(N log N). Use butterfly sparsity (Dao pixelated). The Nd term remains because O must still be
written.

## Why these design choices (design-decision → why table)
- **Optimize HBM accesses, not FLOPs**: because attention is memory-bound; FLOP-reduction methods
  (sparse/low-rank) failed to give wall-clock speedup precisely because they ignored IO. The
  micro-benchmark: FlashAttention has MORE GFLOPs (75.2 vs 66.6 due to recompute) yet far fewer
  HBM GB (4.4 vs 40.3) and 5.7× faster (7.3 vs 41.7 ms). HBM access is the determining factor.
- **Tiling (block the softmax)**: softmax couples a whole row, which seems to force materializing
  the row. Online softmax breaks the coupling: maintain running (m, ℓ) and rescale. Generalize from
  rescaling a scalar denominator to rescaling the whole output accumulator O_i.
- **Rescale the OUTPUT accumulator, not just recombine at the end** (vs Rabe & Staats): keeping one
  running O_i avoids storing K temporary outputs → less total memory and fits the fused kernel.
- **Recomputation in backward**: storing S,P costs O(N^2) memory and HBM traffic. Recomputing from
  Q,K,V blocks in SRAM (using stored m,ℓ) is cheaper in HBM accesses even though it's more FLOPs —
  because the recompute lives in fast SRAM. Inverts the usual "checkpointing trades speed for
  memory": here it BUYS speed.
- **Store (m, ℓ) [or just log-sum-exp lse]**: O(N) statistics let you (a) finish the forward
  normalization and (b) reconstruct P in backward without ever storing the N×N matrix. Canonical
  Triton impl fuses m and ℓ into one stat lse = m + log(ℓ): then P = exp(S·scale - lse) directly,
  and final O scale = exp(m_i - lse_i).
- **Save RNG state for dropout, regen mask in backward**: storing the N×N dropout mask would be
  O(N^2); regenerating from the saved PRNG state R keeps it O(N).
- **Block sizes B_c=ceil(M/4d), B_r=min(ceil(M/4d), d)**: the kernel must hold blocks of Q_i, K_j,
  V_j, O_i (each B·d) plus S_ij (B_r·B_c) in SRAM of size M. The /4 reserves SRAM budget across the
  ~4 simultaneously-resident d-wide blocks (Q,K,V,O). B_r capped at d so that B_r·B_c = O(M) holds
  (with B_c~M/d, B_r·B_c ~ M·B_r/d ≤ M needs B_r ≤ d). Asymptotically B_c=Θ(M/d), B_r=Θ(min(M/d,d)).
- **Compute D_i = do_i · o_i**: avoids reducing P_i: ∘ dP_i: over the size-N row (which wouldn't fit
  SRAM); the identity D_i = P_i:^T dP_i: = do_i^T o_i turns an N-reduction into a d-dot-product.
- **One fused CUDA kernel**: tiling makes matmul→softmax→mask→dropout→matmul a single kernel so
  inputs are read once; otherwise intermediates round-trip through HBM (the training-fusion gap).
- **Block-sparse via butterfly**: butterfly products can express any structured matrix with near-
  optimal params/runtime and are hardware-friendlier than arbitrary sparsity (hardware lottery).

## Canonical implementation (for grounding code)
Official: github.com/Dao-AILab/flash-attention. The Triton reference (flash_attn_triton.py)
_fwd_kernel: lse_i, m_i init -inf; acc_o init 0; loop over K/V blocks: qk=q@k^T; m_ij=max(rowmax(qk)
*scale, lse_i); p=exp(qk*scale - m_ij); l_ij=sum(p); acc_o *= exp(m_i - m_ij); acc_o += p@v;
m_i=m_ij; lse_i = m_ij + log(exp(lse_i - m_ij) + l_ij). Final: o_scale=exp(m_i - lse_i); acc_o *=
o_scale; store Out, Lse. (Note: this variant folds m and ℓ into lse and uses lse_i as the running
max-or-sum slot; mathematically equivalent to the (m, ℓ) pair in the paper.)
Backward: preprocess D_i = rowsum(O ∘ dO); per col block recompute p from lse, compute dV=p^T dO,
dP=dO V^T, dS=p∘(dP - D_i)*scale, dK=dS^T Q, dQ += dS K (atomic add).
