# ALiBi synthesis notes (pre-Phase-2)

## The pain point
- Transformer LM trained on input length L. At inference, segment long text into L-length subsequences (nonoverlapping inference). Want to feed L_valid > L at inference to (a) use more context, (b) reduce the "early token curse" (early positions in a subsequence have little left context → high ppl).
- Observation (diagnostic): sinusoidal-position transformers DO NOT extrapolate. Train L=512, test L+k: ppl improves only up to ~k=20, then degrades sharply. Vaswani speculated sinusoidal *might* extrapolate; empirically it barely does (a few dozen tokens).
- Learned absolute position embeddings: literally cannot represent positions > L (no embedding exists) → zero extrapolation.

## Why position encodings fail to extrapolate (the diagnosis)
- Absolute position info is injected as a signal added to token embeddings (sinusoidal/learned) or applied as rotation (rotary). The model learns to interpret the *specific* position signals it saw in [0, L). At position > L the signal is out-of-distribution: sinusoidal vectors at new phases/combinations the model never trained on; rotary rotations at angles never seen; learned embeddings don't exist at all. The model has no learned response to OOD position signal → degradation.
- Key insight from rotary/T5: they inject position at every layer into keys/queries only, NOT into values → output of attention sublayer carries no explicit absolute position. This "segregation" seems good for extrapolation. ALiBi borrows it.

## Baseline ladder (each leaves a gap)
1. Sinusoidal (Vaswani §3.5): fixed sin/cos vectors added at input layer. Cheap, no params. Barely extrapolates (~+20 tokens then degrades).
2. Learned absolute (Vaswani alt; GPT-3, Jurassic-1): a learned vector per position. No extrapolation at all (undefined beyond L).
3. Rotary (Su et al. 2021 RoFormer; GPT-J): rotate q,k by position-dependent angle at every layer; relative; not added to values. Extrapolates a bit better (~+100-200 tokens) but slower & more memory than sinusoidal.
4. T5 relative bias (Raffel et al. 2020; Shaw et al. 2018): add a *learned* scalar bias to each q·k score depending on bucketed relative distance, shared across layers per head; buckets so far distances share a bias (helps extrapolation). Best extrapolation of baselines (~+600-800 tokens) BUT ≥2× slower than sinusoidal — kills the efficiency motive (no point training short to extrapolate if the method itself is slow).

So: extrapolation is *achievable by changing position method* (T5 proves it), but no existing method extrapolates *efficiently*. Gap = a position method that extrapolates AND is as cheap as sinusoidal AND parameter-free.

## The method (ALiBi)
- Remove all position embeddings.
- After q·k dot product, before softmax, add a static non-learned bias to the score for query i, key j (j≤i): m·(j−i) = −m·(i−j). I.e. softmax(q_i Kᵀ + m·[−(i−1),…,−2,−1,0]).
- m = head-specific slope, fixed before training, NOT learned, NOT multiplied by 1/√d_k scaling.
- Penalty grows linearly with distance (i−j): a recency inductive bias. Each head has a different slope so heads have different "effective windows."

## Slope schedule (derive)
- For n=8 heads: geometric sequence 1/2^1, 1/2^2, …, 1/2^8. Start = ratio = 1/2.
- General n heads: geometric sequence starting at 2^(−8/n), ratio 2^(−8/n). So m_h = (2^(−8/n))^h = 2^(−8h/n) for h=1..n.
- Check n=8: 2^(−8/8)=2^−1=1/2; sequence (1/2)^h = 1/2,1/4,...,1/256. ✓
- n=16: start 2^(−8/16)=2^(−1/2)=1/√2, ratio 1/√2 → 2^{−0.5},2^{−1},...,2^{−8} (interpolates the 8-head set by geometric-averaging consecutive pairs). ✓
- Code form: get_slopes_power_of_2(n): start = 2^(−2^−(log2(n)−3)); note −2^−(log2 n −3) = −2^(3−log2 n) = −8/n. ✓ ratio=start; m_i = start·ratio^i = start^(i+1).
- Non-power-of-2 n: take closest lower power of 2's slopes, then interpolate from the 2× set, taking every other one — a workaround to keep the "good properties" (slopes in (0,1), denser near 0).
- WHY this schedule: manual search over ~10 sets. Insight: best slope sets lie in (0,1) with density increasing toward 0. Geometric with ratio 2^{−8/n} gives exactly that — many heads with small slopes (long effective range / near-uniform attention) and a few with large slopes (sharp recency). Trainable slopes did NOT extrapolate well (and +3% slowdown) → keep fixed. Robust: even sampling slopes from exponential dist works (high variance). So fix once, reuse across sizes/datasets like sinusoidal hyperparams.

## Why linear distance penalty extrapolates while learned embeddings don't
- The bias is a simple, monotone function of relative distance |i−j| with NO learned parameters tied to absolute position. At distance d > anything seen, the value m·(−d) is just a larger negative number — the function is *defined and smooth everywhere*, extrapolating by construction. There is nothing to be "out of distribution": softmax is translation-invariant, so only relative distances matter, and the penalty for a new larger distance is a natural continuation of the same line.
- Contrast: learned/sinusoidal absolute encodings present the model with novel input vectors at new positions → OOD. ALiBi presents the model only with attention scores shaped by a recency penalty that monotonically continues.
- The recency bias also means very distant tokens get strongly down-weighted; the model effectively attends within a soft window, so adding far-away tokens at inference doesn't disrupt the local computation it learned.

## Implementation facts (faithful to repo)
- Implemented by folding the linear bias INTO the causal mask matrix that's added to q·kᵀ before softmax. So zero extra ops at runtime.
- Causal mask: upper triangle (strictly above diagonal) = −inf; add alibi bias tensor of shape (n_heads,1,maxpos) broadcast.
- Memory: mask grows from L×L to n_heads×L×L (different slope per head). Negligible (≤100MB).
- Trick in code: `slopes.view(...)*torch.arange(maxpos)` produces rows all identical (= m·[0,1,...,L-1]) rather than the exact diagonal-difference matrix of Fig 3, but softmax translation-invariance makes it equivalent because the bias is linear (each row gets a constant shift that cancels in softmax). The sign: arange is positive, then it's ADDED to a future_mask whose allowed entries are 0 and disallowed are −inf; effectively the bias decreases with distance to the left. (In the simplest correct standalone form: bias[i,j] = −m·(i−j) for j≤i.)
- Slope NOT scaled by 1/√d_k.
- 8 heads → 1/2^1..1/2^8. 16 heads (CC100 model, 25 layers dim 2048 ffn 8192, 1.3B) → 1/√2 start.

## Evaluation settings (pre-method facts; NO outcomes)
- WikiText-103 (Merity 2016): ~103M training tokens, English Wikipedia. Baevski & Auli (2018) adaptive-input LM: 16 layers, dim 1024, 8 heads, ffn 4096, tied input/softmax embeddings (Press&Wolf 2017, Inan 2017). 205 epochs.
- Toronto Book Corpus (Zhu 2015).
- CC100+RoBERTa corpus (461GB): RoBERTa corpus (BookCorpus, Wikipedia, CC-News, OpenWebText, Stories) 161GB + CC-100 English 300GB. 1.3B model, 50k updates, 128 V100.
- Metric: perplexity. Protocol: nonoverlapping inference (segment into L); sliding-window inference (stride S, slow) as analysis tool. Train length L_train vs valid length L_valid; extrapolation = L_valid > L_train.
- Fairseq (PyTorch) on GPUs.

## Code framework scaffold (pre-method, generic)
- A causal decoder-only transformer LM harness: token embedding (no learned position table necessarily), stack of N decoder layers (self-attn + ffn + layernorm), tied output projection, cross-entropy LM loss, nonoverlapping-inference eval loop.
- Attention sublayer: scores = q·kᵀ / √d; add causal mask; softmax; ·v. The slot: how (if at all) position enters. Stubs: `positional_signal(...) -> ?  # TODO` and `build_attn_mask(seqlen, ...) -> mask  # TODO` where the contribution goes.
