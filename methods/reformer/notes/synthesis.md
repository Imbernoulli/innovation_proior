# Reformer synthesis notes

## Pain point (the world before the method)
Transformers on long sequences blow up memory in THREE distinct ways. Diagnostic
calculation from the intro: 0.5B params/layer = 2GB; activations for 64K tokens,
d=1024, batch 8 = 64K*1K*8 = 0.5B floats = 2GB. Per-layer this fits on one
accelerator. Yet we can't even fine-tune. Why? Three multipliers the back-of-envelope
ignored:
1. Activations stored for backprop -> memory x N_layers (the n_l factor).
2. FFN intermediate dim d_ff (=4096) >> d_model (=1024) -> the FFN dominates per-layer memory.
3. Attention QK^T is [b, L, L] -> O(L^2) memory & time. At L=64K, single head, fp32 = 16GB.

Goal: keep Transformer quality but make memory ~ per-layer (independent of depth)
and attention ~ O(L log L), so a big model fits on ONE accelerator at L=64K.

## Ancestors (load-bearing)
- Transformer (Vaswani 2017): scaled dot-product attention softmax(QK^T/sqrt(d_k))V,
  multi-head (h projections to d_k,d_k,d_v), residual + LayerNorm sublayers, FFN 4x wide.
  The O(L^2) attention and per-layer activation storage are the things to kill.
- Memory-efficient attention (folklore / used as baseline): QK^T need not be materialized;
  compute softmax(q_i K^T)V per query, recompute on backward. Memory ~ L not L^2 but TIME still L^2.
- Angular LSH (Andoni et al. 2015, "Practical and optimal LSH for angular distance"):
  random projections give a hash that is locality-sensitive for angular/cosine distance.
  Concretely random rotation R [d_k, b/2], h(x)=argmax([xR; -xR]) over the 2*(b/2)=b signed
  axes. Nearby (small angle) vectors land in the same argmax bucket w.h.p.
- RevNet (Gomez et al. 2017): reversible residual block y1=x1+F(x2), y2=x2+G(y1),
  invertible: x2=y2-G(y1), x1=y1-F(x2). Recompute activations on backward instead of storing
  -> activation memory independent of depth. Originally for ResNet image classification.
- Sparse Transformer (Child 2019): factorized fixed sparse attention; reduces L^2 but the
  sparsity pattern is fixed/handcrafted, not content-based.
- Adafactor (Shazeer & Stern 2018): sublinear-memory optimizer state (used in training).

## LSH ATTENTION derivation (verify each step)
- softmax(QK^T) is dominated by the few largest q_i.k_j. So for query q_i only its nearest
  keys matter; the rest contribute ~0. Pick, say, 32-64 nearest keys instead of all L.
- Need nearest neighbors fast in high-dim -> LSH. Use angular LSH above: R [d_k, n_buckets/2],
  rotated = x R (then concat with -xR), bucket = argmax over the n_buckets signed axes.
  n_buckets even.
- Shared-QK: we need a notion of "q_i close to k_j" but Q and K are different projections.
  Set Q=K (same linear layer A->QK, separate V). Sharing QK does not hurt quality (shown empirically).
- Single-query attention rewrite:
    o_i = sum_{j in P_i} exp(q_i.k_j - z(i,P_i)) v_j,  P_i = {j: i>=j} (causal),
    z = log-partition. For batching use extended set P~_i with mask m(j,P_i)=inf if j not in P_i else 0.
- LSH restricts: P_i = {j: h(q_i)=h(k_j)}.
- Bucket problems: uneven sizes; a bucket may have queries but no keys. Fix by k_j = q_j/||q_j||
  so h(k_j)=h(q_j) (queries and keys in same bucket trivially since same direction).
- Sort tokens by (bucket, position): permutation i -> s_i. Same-bucket pairs cluster on diagonal.
- Chunk into chunks of m consecutive sorted tokens; each chunk attends to itself and one chunk back.
    extP_i = {j : floor(s_i/m)-1 <= floor(s_j/m) <= floor(s_i/m)}.
  "one back" because a bucket may straddle a chunk boundary.
  If max_i|P_i| < m then P_i subset extP_i (so chunking loses nothing as long as buckets < m).
  Set m = 2l/n_buckets; average bucket = l/n_buckets, assume P(bucket > 2*avg) is low.
- Complexity: n_buckets ~ l/const so chunk size const; cost ~ l * chunk * n_rounds = O(l).
  Conceptually attention goes O(L^2) -> O(L log L) (the log from #buckets scaling / sort).
  Table: LSH memory max(b n_h l d_k, b n_h l n_r (4l/n_c)^2), with n_c=l/32 -> 4l/n_c=128, c=128^2.
- Multi-round: hashing can miss neighbors. Do n_rounds independent hashes h^(1..r),
  P_i = union_r P_i^(r), P_i^(r)={j: h^(r)(q_i)=h^(r)(q_j)}.
- Causal masking under sorting: carry a position index, permute it with the SAME sort, compare s-indices.
- Self-exclusion: in shared-QK, q_i.q_i is almost always the max, so a token attending to ITSELF
  dominates. Forbid self-attention (set its logit to large-but-FINITE -1e5 / TOKEN_SELF_ATTN_VALUE),
  EXCEPT when a token has no other valid target (e.g. first token) -> finite so it can still attend
  to self if nothing else is available.

## Multi-round combination (APPENDIX - verify carefully)
Goal: combine the n_rounds parallel LSH attentions into ONE attention over the union P_i,
WITHOUT double counting a (i,j) pair that shows up in several rounds.
- N_{i,j} = |{r' : j in P_i^(r')}| = number of rounds in which j is a neighbor of i.
- m^(r)_{i,j} = inf if j not in P_i^(r); 1e5 if i=j (self, large finite); log N_{i,j} otherwise.
- o_i = sum_{j in extP_i} exp(q_i.k_j - m(j,P_i) - z(i,P_i)) v_j
      = sum_{r=1}^{n_rounds} exp(z(i,P_i^(r)) - z(i,P_i)) o_i^(r),
  where o_i^(r) = sum_{j in extP_i^(r)} exp(q_i.k_j - m^(r)_{i,j} - z(i,P_i^(r))) v_j.
  The 1/N_{i,j} = exp(-log N_{i,j}) folded into m^(r) corrects the over-counting: a pair present
  in N rounds is summed N times, each scaled 1/N -> counted once total.
  The outer exp(z^(r) - z) re-weights each round's local softmax to the global partition.
  In code this is done via per-round logsumexp (slogits) then a softmax across rounds (probs)
  and out = sum_r o^(r) * probs^(r).

## REVERSIBLE TRANSFORMER derivation
- RevNet block applied to Transformer: F=Attention, G=FeedForward.
    Y1 = X1 + Attention(X2),  Y2 = X2 + FeedForward(Y1).
  LayerNorm moves INSIDE the residual functions (pre-norm inside F,G).
  Reverse: X2 = Y2 - FeedForward(Y1),  X1 = Y1 - Attention(X2).
- Both X1,X2 sized d_model (so total stays 2*d_model -> equal-param vs Transformer when handled right).
- Backward: don't store per-layer activations; recompute by inverting block-by-block from outputs.
  Memory independent of n_l. Code: ReversibleBlock.forward under no_grad; backward_pass recomputes
  G(y1), gets x2=y2-G(y1), backprops; then F(x2), x1=y1-F(x2). RNG saved/restored (Deterministic)
  so dropout etc. recompute identically.
- Chunking FFN: FFN is position-wise independent, so split sequence into c chunks and run FFN
  chunk-by-chunk: Y2 = [X2^(1)+FF(Y1^(1)); ...; X2^(c)+FF(Y1^(c))]. Removes the d_ff memory spike.
  Numerically identical to unchunked. Also chunk the output log-prob/loss for large vocab.
- Parameter swap to CPU: with chunking+reversible, activation mem is depth-independent but PARAM
  mem still grows with depth; swap params to/from CPU since b*l is huge so compute amortizes transfer.

## Final complexity table (assume d_ff>=d_model, n_c=l/32, c=128^2)
Transformer:            mem max(b l d_ff, b n_h l^2) n_l
Reversible Transformer: mem max(b l d_ff, b n_h l^2)        (n_l gone)
Chunked Reversible:     mem max(b l d_model, b n_h l^2)     (d_ff gone)
LSH Transformer:        mem max(b l d_ff, b n_h l n_r c) n_l
Reformer (all):         mem max(b l d_model, b n_h l n_r c)

## Canonical code map (lucidrains reformer-pytorch -> deliverable)
- LSHAttention.hash_vectors: angular LSH, random_rotations, argmax over [rot; -rot], per-round offsets.
- LSHAttention.forward: bucket+ticker sort, batched_index_select, reshape into chunks, look_one_back,
  einsum dots, masks (input, causal, self via TOKEN_SELF_ATTN_VALUE), dup-count log-subtract,
  logsumexp softmax, unsort, cross-round softmax (probs) combine.
- LSHSelfAttention: toqk (shared), tov, merge heads, full-attn fallback for short seqs, to_out.
- reversible.py: Deterministic (RNG save), ReversibleBlock.forward/backward_pass, _ReversibleFunction,
  ReversibleSequence (reverse only if seq long enough).
- Reformer.forward: x=cat([x,x]) to make (x1,x2); ReversibleSequence; mean of the two halves.
- FeedForward + Chunk wrapper = chunked FFN.

## Design-decision -> why
- argmax([xR;-xR]) not sign hashing: gives b buckets from b/2 rotations, angular-distance sensitive (Andoni).
- different R per round: independent hashes -> union reduces miss prob.
- shared-QK + k=q/||q||: makes h(k)=h(q), guarantees nonempty buckets, no quality loss.
- sort then chunk: turns ragged bucket attention into fixed-size batched matmuls (GPU friendly).
- one chunk back: bucket can straddle chunk boundary.
- m=2l/n_buckets: covers a bucket up to 2x average size.
- finite self-mask (-1e5) not -inf: lets a no-other-target token still attend to itself.
- log N_{i,j} mask: dedup across rounds (count once).
- F=Attn,G=FF in revnet: natural 2-sublayer split; reverse recovers both.
- chunk FFN: position independence -> no quality change, kills d_ff spike.
</content>
</invoke>
