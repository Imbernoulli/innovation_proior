# Transformer-XL synthesis

## Pain point / research question
Autoregressive LM: P(x) = prod P(x_t | x_<t). Want to model LONG-term dependency.
- RNN/LSTM: standard, but vanishing/exploding gradients; empirically LSTM-LM uses ~200 context words (Khandelwal 2018). Hard to optimize over long range even with gating + grad clipping.
- Self-attention (Vaswani 2017): direct connections between any pair of positions -> short gradient paths, no recurrence-distance decay. Al-Rfou et al. 2018 trained deep (64-layer) char Transformer LM with auxiliary losses, beat LSTM. BUT trained on FIXED-LENGTH segments of a few hundred chars, no info flow across segments.

Two limitations of fixed-length segment training ("vanilla model"):
1. Max dependency length upper-bounded by segment length L. So the optimization advantage of attention (no vanishing grad) is wasted.
2. Context fragmentation: segments are cut as consecutive chunks ignoring sentence/semantic boundaries; first few tokens of each segment have no/little context -> bad prediction, inefficient optimization.

Eval: vanilla model shifts window by ONE position each step, recomputes whole segment from scratch -> uses max context but extremely expensive (Transformer-XL later up to 1800x faster).

## Ancestors / baselines
- Vanilla Transformer LM on fixed segments (Al-Rfou 2018): the thing being fixed.
- Absolute sinusoidal positional encoding (Vaswani 2017): U in R^{Lmax x d}, row i = sinusoid of absolute position i; added to word embeddings at input. inv_freq = 1/10000^(2k/d), concat[sin, cos].
- RNN/LSTM (Hochreiter 1997) + truncated BPTT (Mikolov 2010): cache LAST hidden state of prev segment as fixed forward input; gradient stays within segment. XL is the attention analogue but caches a SEQUENCE of states, not just the last one.
- Shaw et al. 2018 relative position (machine translation): only terms (a)+(b); merges W_k R into single trainable matrix \hat{R} -> loses sinusoid inductive bias -> no length generalization.
- Memory-augmented nets (Graves 2014 NTM, Weston 2014 memory nets): motivates calling cached states "memory".

## Segment-level recurrence with state reuse (derivation)
Two consecutive segments s_tau, s_{tau+1} of length L. h_tau^n in R^{L x d} = layer-n hidden states for segment tau.
Cache prev segment's layer-(n-1) states, STOP-GRADIENT, concat along length:
  h~_{tau+1}^{n-1} = [SG(h_tau^{n-1}) o h_{tau+1}^{n-1}]
  q = h_{tau+1}^{n-1} Wq^T   (query from CURRENT segment only)
  k,v = h~_{tau+1}^{n-1} Wk^T, h~_{tau+1}^{n-1} Wv^T   (keys/values from EXTENDED context)
  h_{tau+1}^n = TransformerLayer(q,k,v)
Key point: keys/values conditioned on extended context including cached h_tau^{n-1}.
- Recurrence shifts ONE layer DOWN per segment (h_{tau+1}^n depends on h_tau^{n-1}), unlike same-layer RNN recurrence. => max dependency length grows O(N x L) (N layers).
- SG: gradient stays within current segment (only forward info flows back). Like truncated BPTT but cache a sequence.
- Generalize: cache M old states (possibly multiple segments) as memory m_tau^n in R^{M x d}. M = L in training, larger at eval.
- Eval speedup: reuse representations instead of recomputing.

## Why absolute positional encoding breaks under recurrence
If we keep absolute U: h_tau = f(h_{tau-1}, E_{s_tau} + U_{1:L}), h_{tau+1} = f(h_tau, E_{s_{tau+1}} + U_{1:L}).
Both segments get the SAME U_{1:L}. So x_{tau,j} and x_{tau+1,j} are indistinguishable in position -> the model can't tell which segment a cached state came from. Positions collide. => need RELATIVE positions.

Conceptual: positional encoding = a temporal "bias"/clue about where to attend. Inject it into each layer's ATTENTION SCORE rather than statically into the initial embedding, and define it RELATIVELY: query q_{tau,i} attending to key k_{tau,j} only needs the relative distance i-j, not absolute positions. Create R in R^{Lmax x d}, row i = sinusoid of relative distance i. Absolute position recoverable from relative distances, so no info lost.

## 4-term decomposition (the heart)
Standard Transformer absolute attention score q_i . k_j with q from (E_{x_i}+U_i), k from (E_{x_j}+U_j):
A^abs_{i,j} = (E_{x_i}+U_i)^T Wq^T Wk (E_{x_j}+U_j), expand:
  (a) E_{x_i}^T Wq^T Wk E_{x_j}    -- content x content
  (b) E_{x_i}^T Wq^T Wk U_j        -- content(query) x position(key)
  (c) U_i^T Wq^T Wk E_{x_j}        -- position(query) x content(key)
  (d) U_i^T Wq^T Wk U_j            -- position x position

Re-parameterize to relative:
  (a) E_{x_i}^T Wq^T W_{k,E} E_{x_j}
  (b) E_{x_i}^T Wq^T W_{k,R} R_{i-j}      [U_j -> R_{i-j}]
  (c) u^T W_{k,E} E_{x_j}                 [U_i^T Wq^T -> u^T]
  (d) v^T W_{k,R} R_{i-j}                 [U_i^T Wq^T -> v^T, U_j -> R_{i-j}]

Three changes:
1. Replace key-side absolute U_j (terms b,d) with relative sinusoid R_{i-j}. R is sinusoid, NO learnable params (keeps Vaswani inductive bias -> length generalization).
2. Query-side absolute term U_i^T Wq^T is the same for every query position -> replace with a SINGLE learnable global vector: u (in term c), v (in term d). "Attentive bias toward a word/distance should be the same regardless of query position."
3. Separate Wk into W_{k,E} (content keys) and W_{k,R} (location keys), because content and position are now different kinds of input (embeddings vs sinusoids) and want different projections.

Meanings: (a) content addressing, (b) content-dependent positional bias, (c) global content bias (word prior), (d) global positional bias (distance prior).

vs Shaw 2018: they only have (a)+(b), drop (c),(d); merge Wk R into one trainable \hat{R}, losing sinusoid -> no generalization to longer attention.

## Full per-layer recurrence (n=1..N)
h~^{n-1} = [SG(m^{n-1}) o h^{n-1}]
q,k,v = h^{n-1} Wq^T, h~^{n-1} W_{k,E}^T, h~^{n-1} Wv^T   (note: content key uses W_{k,E})
A_{i,j} = q_i^T k_j + q_i^T W_{k,R} R_{i-j} + u^T k_j + v^T W_{k,R} R_{i-j}
  = (q_i + u)^T k_j  [=AC]  +  (q_i + v)^T (W_{k,R} R_{i-j})  [=BD]
a = MaskedSoftmax(A) v
o = LayerNorm(Linear(a) + h^{n-1})   (post-LN residual)
h^n = PositionwiseFF(o)
h^0 = E_{s}.

## Efficient computation (Appendix B)
Naive: compute W_{k,R} R_{i-j} for all (i,j) -> quadratic.
i-j ranges 0..M+L-1 only. Define Q = [R_{M+L-1}; ...; R_0] W_{k,R}^T in R^{(M+L) x d}, reversed order so Q_k = W_{k,R} R_{M+L-1-k}.
Term (b) matrix B (L x (M+L)) has entries q_i^T W_{k,R} R_{...}; observe B_tilde = q Q^T has all the same dot products but each row of B is a LEFT-SHIFT of the corresponding row of B_tilde. So compute B_tilde via one matmul, then shift. The _rel_shift trick: pad a zero column on the left, reshape (klen+1, qlen) -> drop first row -> reshape back = shifts each row appropriately. Same for term (d): d_tilde = (Q v)^T, then shift. Cost linear.

## Code mapping (kimiyoung/transformer-xl pytorch, attn_type=0 = paper default = RelPartialLearnable)
- PositionalEmbedding: sinusoid R, inv_freq = 1/10000^(arange(0,d,2)/d), concat[sin,cos]. pos_seq = arange(klen-1,-1,-1) (relative distances, descending).
- r_w_bias = u (n_head x d_head), r_r_bias = v.
- RelPartialLearnableMultiHeadAttn:
  qkv_net: Wq,W_{k,E},Wv combined; r_net: W_{k,R}.
  cat = [mems, w]; w_head_q = q[-qlen:] (query from current only); k,v over extended.
  rw_head_q = w_head_q + r_w_bias; AC = einsum(rw_head_q, w_head_k)  -> terms (a)+(c)
  rr_head_q = w_head_q + r_r_bias; BD = einsum(rr_head_q, r_head_k); BD = _rel_shift(BD) -> terms (b)+(d)
  attn_score = (AC+BD) * scale (1/sqrt(d_head)); masked softmax over klen; *v; o_net; residual+LayerNorm.
- _rel_shift: zero-pad col, reshape, drop, view_as.
- _update_mems: torch.no_grad, cat[mems,hids], slice last mem_len, detach (= SG).
- MemTransformerLM._forward: builds pos_emb once, triangular causal mask with diagonal=1+mlen, loops layers passing mems_i.
- scale 1/sqrt(d_head): standard, keep dot-product variance ~1 so softmax not saturated.
- post-LN (default pre_lnorm=False).

## Design-choice -> why table
- Cache states + SG: carry info across segments without backprop blowup / staleness; gradient bounded to one segment (truncated-BPTT analogue). Cache SEQUENCE not last state (vs RNN BPTT) because attention can address any past position.
- Recurrence one-layer-down: that's just how stacking caches works; gives O(N L) range.
- Relative pos: absolute collides across segments under reuse.
- R sinusoid no params (not merged learnable \hat R like Shaw): inductive bias -> generalize to longer memory at eval.
- u,v global vectors replace query-side absolute: query position has no absolute meaning anymore; the per-position query term U_i was the only thing carrying absolute query position, and it should be constant across positions.
- Separate W_{k,E}, W_{k,R}: content keys vs location keys are different input types.
- Inject pos into every layer's score (not just input): each layer re-decides where to attend; static input injection would be washed out and is absolute-flavored.
- M=L train, larger eval: generalization test + cheap longer context at eval.
- 1/sqrt(d_head) scale, post-LN, FFN, residual: inherited Transformer.
