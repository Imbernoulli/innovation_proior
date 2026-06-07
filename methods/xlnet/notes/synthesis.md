# XLNet synthesis (pre-Phase-2)

## Pain point / research question
Pretrain-then-finetune for NLU. Want a pretraining objective that (a) uses DEEP BIDIRECTIONAL context (downstream NLU needs both sides), (b) does NOT inject artificial [MASK] tokens (pretrain-finetune discrepancy), (c) does NOT assume masked targets are conditionally independent given the context (BERT factorizes the joint over masked tokens as a product = independence assumption, ignores dependency among masked tokens), and (d) ideally fits in the AR density-estimation framework so AR progress (Transformer-XL) transfers.

## Ancestors (load-bearing)
- **AR LM** (Dai&Le 2015 semi-supervised seq learning; Peters et al. 2018 ELMo; Radford et al. 2018 GPT): p(x)=Π_t p(x_t|x_<t), forward (or backward). Trained max Σ log softmax(h_θ(x_{1:t-1})^T e(x_t)). Pro: exact product rule, no [MASK], no independence assumption. Con: unidirectional only — h only sees left context; ELMo concatenates an independently-trained fwd + bwd LM = shallow, no deep joint interaction.
- **BERT** (Devlin et al. 2018): denoising AE. Corrupt x→x̂ by replacing ~15% tokens with [MASK]; reconstruct masked tokens x̄. Objective max log p(x̄|x̂) ≈ Σ_t m_t log p(x_t|x̂) — the ≈ is the independence assumption: factorizes joint over masked tokens as a product, each masked token reconstructed separately. Pro: bidirectional (H_θ(x̂)_t sees both sides). Cons: (1) independence assumption (can't model dependency between two masked tokens, e.g. "New"/"York"); (2) [MASK] absent at finetune → discrepancy; partially replacing [MASK] with real tokens doesn't fix it because it must stay rare or the objective trivializes.
- **NADE/MADE** (Uria et al. 2016 orderless NADE; Germain et al. 2015): density estimators that train over MULTIPLE/random factorization orders to get an order-agnostic model. Motivation there = better density estimation via orderless inductive bias; they rely on implicit position-awareness in MLP architecture; essentially orderless (bag-of-words-ish). KEY DIFFERENCE for us: we want order-AWARENESS (positional encodings) and the motive is bidirectional context for AR pretraining, not orderless density estimation.
- **Transformer-XL** (Dai et al. 2019): SOTA AR LM. Two pieces we integrate: (i) segment recurrence — cache hidden states of previous segment as memory, extend effective context, no gradient through memory; (ii) relative positional encoding — decompose attention logit into content-content (a), content-position (b), and two global bias terms; concretely A_{ij} = (q_i+u)^T k_j + (q_i+v)^T W_R r_{i-j}, where u (=r_w_bias) and v (=r_r_bias) are global learnable bias vectors, r_{i-j} sinusoidal relative-distance embedding. rel_shift trick computes the relative term efficiently.
- **Transformer/self-attention** (Vaswani et al. 2017): the backbone. Q,K,V; softmax(QK^T/√d)V; multi-head; residual+LayerNorm; position-wise FFN.

## The derivation chain
1. Keep AR's product form (no independence assumption, no [MASK]) but make it bidirectional by maximizing over a RANDOM FACTORIZATION ORDER z. Objective: max_θ E_{z~Z_T}[ Σ_t log p_θ(x_{z_t} | x_{z_<t}) ]. Same params shared across all T! orders → in expectation each position conditions on every other position on both sides. Permutes only the FACTORIZATION ORDER (the attention mask), NOT the sequence order (positions/positional encodings stay original) — because finetuning always sees natural order.
2. Parameterization problem: naive softmax p(x_{z_t}=x | x_{z_<t}) = exp(e(x)^T h_θ(x_{z_<t})) / Σ... is INDEPENDENT of the target position z_t. Two permutations z^(1),z^(2) with same prefix z_<t but z_t=i vs j give IDENTICAL distributions. But ground-truth dist of position i and position j differ → wrong / can't learn. So h must be replaced by g_θ(x_{z_<t}, z_t) that takes target POSITION z_t as input. p(x_{z_t}=x|x_{z_<t}) = exp(e(x)^T g_θ(x_{z_<t}, z_t))/Σ.
3. How to build g? Stand at position z_t, use the position to query the context x_{z_<t}. Contradiction in a standard single-stream Transformer: (1) to PREDICT x_{z_t}, g must NOT see content x_{z_t} (else trivial); (2) to serve as context for a LATER prediction x_{z_j}, j>t, the representation at position z_t MUST encode content x_{z_t}. One hidden state can't be both. → TWO STREAMS:
   - content stream h_{z_t}: attends to x_{z_≤t} (includes z_t's own content) — like a normal Transformer hidden state. Init h_i^(0)=e(x_i).
   - query stream g_{z_t}: attends to x_{z_<t} and position z_t only, NOT content x_{z_t}. Init g_i^(0)=w (shared trainable vector).
   Shared params. Per layer m: g_{z_t}^(m) ← Attn(Q=g_{z_t}^(m-1), KV=h_{z_<t}^(m-1)); h_{z_t}^(m) ← Attn(Q=h_{z_t}^(m-1), KV=h_{z_≤t}^(m-1)). Use last-layer g_{z_t}^(M) for the softmax. Content stream = exactly standard self-attention → at finetune drop query stream, use content stream as normal Transformer(-XL).
4. Partial prediction: full PLM is hard to optimize / slow convergence. Only predict the LAST tokens in the order (the ones with the longest context). Split z into z_≤c (context) and z_>c (targets). Objective max E_z[ log p(x_{z_>c} | x_{z_≤c}) ] = E_z[ Σ_{t=c+1}^{|z|} log p(x_{z_t}|x_{z_<t}) ]. K ≈ |z|/(|z|-c) ≈ 6 (predict ~1/K). For unpredicted tokens, no query stream needed → saves compute/memory.
5. Integrate Transformer-XL: relative positional encoding based on ORIGINAL positions (straightforward since perm is just masking). Recurrence: cache content reps h̃^(m) of previous segment; next segment attends KV=[h̃^(m-1), h_{z_≤t}^(m-1)]. Crucially positional encodings depend only on actual positions, so the attention update is INDEPENDENT of the previous segment's factorization order z̃ — can cache & reuse memory without knowing z̃.
6. Multiple segments: input [CLS, A, SEP, B, SEP] like BERT but NSP dropped (no consistent gain). Relative SEGMENT encoding: for positions i,j use s_+ if same segment else s_-; attention term a_{ij}=(q_i+b)^T s_{ij} added to logit. Benefits: relative inductive bias generalizes; supports >2 segments at finetune.

## Two-stream attention with TXL backbone (appendix, exact)
Init: h_t=e(x_t), g_t=w.
Per layer m:
  ĥ_{z_t}^(m) = LN( h_{z_t}^(m-1) + RelAttn(h_{z_t}^(m-1), [h̃^(m-1), h_{z_≤t}^(m-1)]) )
  h_{z_t}^(m) = LN( ĥ_{z_t}^(m) + PosFF(ĥ_{z_t}^(m)) )
  ĝ_{z_t}^(m) = LN( g_{z_t}^(m-1) + RelAttn(g_{z_t}^(m-1), [h̃^(m-1), h_{z_<t}^(m-1)]) )   # note z_<t for query stream
  g_{z_t}^(m) = LN( ĝ_{z_t}^(m) + PosFF(ĝ_{z_t}^(m)) )
p(X_{z_t}=x|x_{z_<t}) = exp(e(x)^T g_{z_t}^(M)) / Σ_x' exp(e(x')^T g_{z_t}^(M))

Key impl detail (code): both streams SHARE keys/values from the CONTENT stream (k_head_h,v_head_h from cat=[mems,h]); they differ only in (a) which query head (q from h vs q from g) and (b) the attention MASK — content uses non_tgt_mask (can see self), query uses attn_mask (cannot see self). perm_mask realized via rev_index comparison: position i can attend j iff i comes after j in the order (or j is a non-masked context token). target_mapping maps num_predict targets onto the query stream.

## Code grounding (zihangdai/xlnet, TF1)
- rel_attn_core: ac=(q+r_w_bias)·k_h, bd=rel_shift((q+r_r_bias)·k_r), ef=seg_mat·((q+r_s_bias)·seg_embed); attn_score=(ac+bd+ef)*scale - 1e30*mask; softmax; ·v.
- two_stream_rel_attn: compute shared k_h,v_h,k_r from content; h-stream q from h with attn_mask_h(=non_tgt_mask); g-stream q from g with attn_mask_g(=attn_mask), target_mapping einsum.
- transformer_xl: builds attn_mask from perm_mask+input_mask, non_tgt_mask (subtract eye so content can see self), word_emb_k + mask_emb for query stream init, relative pos enc, loops layers calling two_stream_rel_attn then PosFF.
- lm_loss + two_stream_loss: logits=h·softmax_w+b; sparse softmax CE; total_loss = Σ(loss*tgt_mask)/Σ tgt_mask.

## Hyperparams (XLNet-Large): 24 layers, d_model 1024, 16 heads, d_head 64, FFN 4096, GeLU, dropout 0.1, K=6, seqlen 512, batch 8192, lr 4e-4, 500K steps, 40K warmup, Adam eps 1e-6, wd 0.01.

## Design-decision → why
- Random factorization order (vs fixed fwd/bwd): only way to get bidirectional context while keeping AR product form. Each pos conditions on both sides in expectation.
- Permute order not tokens: finetune sees natural order; keep positional encodings on original positions.
- Target-position-aware g (vs naive h): naive softmax is position-blind → two distinct target positions get identical predictions, provably (same prefix). Must inject z_t.
- Two streams (vs one): single state can't simultaneously hide its own content (needed to predict it) and expose it (needed as context for later predictions). Resolve with content (sees self) + query (doesn't).
- Query init w (shared) vs per-position: position info comes from relative attention; content init e(x_i).
- Partial prediction K=6: full PLM converges slowly; predict only long-context tail; ~1/K targets; skip query stream for non-targets to save compute.
- TXL recurrence + relative pos: extend context, and relative-pos makes memory reuse independent of previous order. Relative > absolute pos here because permutation needs order to come from masking not from absolute position embeddings tied to factorization.
- Relative segment encoding (vs BERT absolute): generalizes, supports >2 segments.
- Drop NSP: no consistent gain in ablation.
