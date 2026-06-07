# LLaMA synthesis (Phase 1.5)

## Verified
- arXiv 2302.13971, "LLaMA: Open and Efficient Foundation Language Models". Title verified via source tar.
- Canonical impl: official meta-llama repo `llama_v1` tag `llama/model.py` (no GQA — correct for LLaMA-1). Saved to code/llama_v1_model.py.

## The core thesis / pain point
- Prevailing assumption (GPT-3, Gopher, PaLM line): more parameters → better. Scaling efforts chase larger N.
- Chinchilla (Hoffmann 2022) showed: for a fixed TRAINING compute budget C, optimal is NOT the biggest model but a smaller model trained on MORE data; N ∝ C^a, D ∝ C^b with a≈b≈0.5. Chinchilla recommends e.g. ~10B model on ~200B tokens.
- LLaMA's reframing: Chinchilla optimizes *training* compute and ignores *inference* compute. At deployment, you serve the model billions of times. Given a target quality, the model you want is the one cheapest *at inference* = the smallest one that reaches that quality. A larger model may be cheaper to *train* to a quality level, but a smaller model trained on more tokens is cheaper to *serve*.
- Key empirical observation (motivating, about training dynamics, allowed in context): a 7B model keeps improving past 1T tokens — well beyond the Chinchilla-optimal token count for that size. So "train past the training-compute optimum" is a free lunch for inference cost.
- Goal: best achievable performance at various *inference* budgets → train a family (7B–65B) on far more tokens than compute-optimal (1.0–1.4T), using only publicly available data.

## Architecture: standard pre-norm decoder Transformer + 3 borrowed improvements
All grounded; each ancestor + why:

### Pre-normalization + RMSNorm [from GPT-3 pre-norm; RMSNorm from Zhang & Sennrich 2019, arXiv 1910.07467]
- Post-norm (original Transformer) normalizes the *output* of each sublayer (after residual add). Pre-norm normalizes the *input* of each sublayer → gradient flows cleanly through the residual highway → far more stable deep training. (GPT-2/GPT-3 adopt pre-norm for this reason.)
- LayerNorm does two things: re-center (subtract mean) and re-scale (divide by std). Zhang & Sennrich hypothesize re-centering is dispensable — most of LN's benefit is the re-scaling (implicit learning-rate adaptation via scale invariance). Residual stream + learned projections can absorb mean shifts themselves.
- RMSNorm: x / sqrt(mean(x^2)+eps) * g. No mean subtraction, no bias. Re-scaling invariant but not re-centering invariant. 7%–64% faster than LN, comparable quality.
- Code: `x * rsqrt(mean(x^2,-1)+eps) * weight`, computed in fp32 then cast back. eps=1e-6 (code) / 1e-5 default in ModelArgs.

### SwiGLU FFN [from PaLM; SwiGLU from Shazeer 2020, arXiv 2002.05202]
- Original FFN: W2 · ReLU(W1 x). Two matrices, hidden dim 4d.
- GLU (Dauphin 2017): elementwise product of two linear projections, one gated by a nonlinearity. GLU variants (Shazeer): replace sigmoid gate with Swish/SiLU → SwiGLU: FFN(x) = W2 ( SiLU(W1 x) ⊙ W3 x ). Three matrices.
- Shazeer found GEGLU/SwiGLU give best perplexity at constant compute.
- Parameter matching: GLU has THREE matrices vs two. To keep params/compute constant, reduce hidden dim to 2/3 of the original. So 4d → (2/3)·4d = (8/3)d. LLaMA states "2/3 · 4d instead of 4d". (Shazeer: "reduce d_ff to 2/3 its original value to keep the parameter count and compute constant since GLU variants have three weight matrices instead of two".)
- Code FeedForward: hidden_dim = 4*dim; then hidden = int(2*hidden/3); then round up to multiple_of (256). SiLU(w1 x) * w3 x → w2.

### RoPE rotary position embeddings [from GPTNeo usage; RoPE from Su et al. 2021, arXiv 2104.09864]
- Absolute position embeddings (added to input, Transformer/BERT/GPT) inject position once at the bottom; attention sees absolute positions. Want relative-position dependence and good extrapolation.
- RoPE goal: find position functions f_q(x,m), f_k(x,n) such that <f_q, f_k> = g(x_m, x_n, m-n) depends only on relative offset m-n.
- 2D solution: multiply each 2D sub-vector by rotation by angle m·θ. In complex form (Wq x)⊙e^{imθ}. Then <R_m q, R_n k> = q^T R_{n-m} k depends on n-m only. d-dim: block-diagonal rotation R^d_{Θ,m}, frequencies θ_i = 10000^{-2(i-1)/d} (i=1..d/2). Low i → fast rotation (short range), high i → slow (long range) → long-term decay of attention with distance.
- Applied at EVERY layer to q and k (not v), unlike absolute PE added once.
- Code: precompute_freqs_cis (freqs = 1/(theta^(arange(0,dim,2)/dim)), theta=10000, outer with positions, polar → complex), apply_rotary_emb (view q,k as complex pairs, multiply by freqs_cis, back to real).

## Optimizer / training (grounded Table + §Optimizer)
- AdamW, β1=0.9, β2=0.95, weight decay 0.1, grad clip 1.0.
- Cosine LR schedule, final LR = 10% of max. 2000 warmup steps.
- Batch size 4M tokens. LR: 3e-4 (7B,13B), 1.5e-4 (33B,65B).
- Model table: 7B d=4096 h=32 L=32; 13B d=5120 h=40 L=40; 33B d=6656 h=52 L=60; 65B d=8192 h=64 L=80.
- Tokenizer: BPE (SentencePiece), split numbers into digits, byte fallback for unknown UTF-8. ~1.4T tokens.

## Efficient implementation (grounded §Efficient implementation)
- Memory-efficient causal attention from xformers (Rabe & Saunders 2021 self-attention does not need O(n^2) memory; backward from FlashAttention Dao 2022): don't store attention weights, don't compute masked (upper-triangular) scores.
- Activation checkpointing: save expensive activations (linear outputs), recompute the rest; manual backward for transformer layers instead of autograd.
- Model + sequence parallelism (Korthikanti 2022), overlap all_reduce comm with compute. ~380 tokens/sec/GPU on 2048 A100-80GB for 65B.

## Scaffold ↔ final code correspondence
- RMSNorm class (norm + scale) ← context stub: a normalization module (slot)
- precompute_freqs_cis / apply_rotary_emb ← context stub: position-handling function (slot)
- Attention (wq,wk,wv,wo; kv cache; rotary applied to q,k) ← context stub: self-attention module
- FeedForward (w1,w2,w3, SiLU gate, 2/3·4d) ← context stub: position-wise FFN module
- TransformerBlock (pre-norm: x + attn(norm(x)); h + ffn(norm(h))) ← context stub: residual block
- Transformer (embeddings, layers, final norm, output proj, no learned PE) ← context stub: decoder LM

## Things deliberately OUT of scope (eval experiments)
- All benchmark numbers (commonsense/QA/MMLU/code/math tables), instruction-finetuning results, carbon footprint, bias/toxicity. Stop at architecture+optimizer+efficient-impl.
