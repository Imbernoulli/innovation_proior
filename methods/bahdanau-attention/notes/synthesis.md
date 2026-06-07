# Synthesis — Bahdanau Attention (RNNsearch / "Jointly Learning to Align and Translate")

## The pain point (what existed and where it fell short)

Neural MT in 2014 = encoder–decoder (Sutskever et al. 2014; Cho et al. 2014). Encoder RNN reads source x=(x_1..x_Tx), squashes it into ONE fixed-length vector c (Sutskever: c = h_Tx, the final LSTM state). Decoder RNN generates y conditioned on c:
- p(y) = ∏_t p(y_t | y_1..y_{t-1}, c)
- p(y_t | ..., c) = g(y_{t-1}, s_t, c)   — same c at EVERY step.

Central difficulty: a fixed-length c of dimension n must hold ALL information of a source sentence of arbitrary length. Information-theoretic bottleneck. Worse for long sentences and for sentences longer than training.

DIAGNOSTIC FINDING (motivating, pre-method, from Cho et al. 2014b "On the properties of NMT"): basic encoder–decoder BLEU degrades rapidly as source length grows; falls off a cliff past ~20–30 words. This is the empirical observation the method reacts to. (This is a finding about EXISTING systems → context, not a proposed-method result.)

## Load-bearing ancestors

1. **Sutskever et al. 2014 (seq2seq)** — deep LSTM encoder → fixed vector → deep LSTM decoder. Reversed source order trick. SOTA-close on EN-FR. Limit: single fixed c; degrades on long inputs.
2. **Cho et al. 2014a (RNN Encoder–Decoder + GRU)** — proposed the encoder–decoder framing for MT scoring; introduced the gated recurrent unit (GRU): update gate z, reset gate r,
   - s̃ = tanh(W e(y) + U[r∘s_{prev}] + C c)
   - z = σ(W_z e(y)+U_z s_{prev}+C_z c); r = σ(...); s = (1−z)∘s_{prev} + z∘s̃.
   GRU keeps gradient paths with product-of-derivatives ≈ 1 (mitigates vanishing gradient — Hochreiter 91, Bengio et al. 94, Pascanu et al. 2013). Limit: still one fixed c.
3. **Cho et al. 2014b** — the diagnostic that long-sentence performance collapses. The smoking gun.
4. **Alignment in statistical MT (IBM models, Brown et al.; phrase-based Koehn et al. 2003; Koehn 2010)** — translation modeled with an explicit alignment a as a LATENT variable (which source word generates which target word), trained by EM. NULL token to map words to/from nowhere. Hard, discrete alignment. Limit for NMT: discrete latent var → not differentiable, can't be trained jointly by backprop; NULL hack is counter-intuitive.
5. **Bidirectional RNN (Schuster & Paliwal 1997; Graves et al. 2013 speech)** — forward RNN gives →h_t (summary of x_1..x_t), backward RNN gives ←h_t (summary of x_t..x_Tx). Concatenate h_t=[→h_t; ←h_t] → annotation summarizing whole sentence centered on position t. Needed because a per-position annotation must see both left and right context.
6. **Graves 2013 (handwriting synthesis)** — a related "learning to align": Gaussian-kernel location-based attention, but modes move MONOTONICALLY in one direction. Limit for MT: reordering (EN→DE, adjective-noun swaps) needs non-monotonic alignment. Motivates a content-based, unconstrained alignment.
7. **Deep output / maxout (Pascanu et al. 2014; Goodfellow et al. 2013)** — the output layer detail.

## The key idea (insight-before-method form)

If one vector can't hold a long sentence, stop forcing it to. Encode the source into a SEQUENCE of vectors (one per position) and let the decoder pull a DIFFERENT, freshly-built summary at each output step — focused on the source positions relevant right now. The decoder thus offloads the "remember everything" burden onto a re-readable memory.

Two pieces fall out:
- **Per-step context** c_i (distinct for each target i), so p(y_i|y_1..y_{i-1},x) = g(y_{i-1}, s_i, c_i), s_i=f(s_{i-1},y_{i-1},c_i).
- **How to build c_i**: a weighted sum (expected annotation) over the source annotations, weights = a soft, differentiable, content-based alignment.

## Core math (ALL must be derived inline in reasoning.md)

Annotations (BiRNN encoder): →h_j, ←h_j by GRU; h_j = [→h_j ; ←h_j] ∈ R^{2n}.

Alignment SCORE (additive / MLP):
- e_ij = a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j)
- W_a ∈ R^{n'×n}, U_a ∈ R^{n'×2n}, v_a ∈ R^{n'}.
WHY additive MLP and not dot product: query s_{i-1}∈R^n and key h_j∈R^{2n} have DIFFERENT dims → a raw dot product isn't even shape-compatible. The MLP projects both into a common R^{n'} space (W_a, U_a) and scores with v_a; tanh gives a learnable nonlinear comparator. (Scaled dot-product came later; not knowable here.) Efficiency note: U_a h_j doesn't depend on i → precompute once per sentence; only W_a s_{i-1} recomputed per step. Still O(Tx·Ty) evaluations.

WHY query with s_{i-1} (previous decoder state, before emitting y_i): it summarizes what's been generated so far → "what do I need next?"; using s_i would be circular (s_i needs c_i needs the query).

SOFTMAX → soft alignment weights:
- α_ij = exp(e_ij) / Σ_{k=1}^{Tx} exp(e_ik).
Probability that y_i aligns to / is translated from x_j.

CONTEXT vector (expected annotation):
- c_i = Σ_{j=1}^{Tx} α_ij h_j.
This is E[h] over the alignment distribution α_i·. Key contrast with SMT: alignment is NOT a discrete latent variable trained by EM; it's a deterministic, differentiable function → gradient of the loss backprops THROUGH α into the alignment MLP and the whole net. Joint training. No NULL hack; soft weights naturally handle many-to-one and length mismatch.

Reduction: fix c_i = →h_Tx (constant) ⇒ recover the plain RNN encoder–decoder. So the method strictly generalizes the baseline.

Decoder GRU with c_i injected:
- s̃_i = tanh(W E y_{i-1} + U[r_i ∘ s_{i-1}] + C c_i)
- z_i = σ(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i)
- r_i = σ(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)
- s_i = (1−z_i)∘s_{i-1} + z_i∘s̃_i
- s_0 = tanh(W_s ←h_1)   (init from backward state of first word).

Deep output (maxout) for p(y_i):
- t̃_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i
- t_i = maxout pairs: t_i = [ max(t̃_{i,2j-1}, t̃_{i,2j}) ]_{j=1..l}
- p(y_i | s_i, y_{i-1}, c_i) ∝ exp(y_i^T W_o t_i), softmax.

Training: minibatch SGD (80 sentences) + Adadelta (ε=1e-6, ρ=0.95); grad L2-norm clipped to ≤1 (Pascanu 2013). Orthogonal init for recurrent matrices; W_a,U_a ~ N(0,0.001²); V_a, biases=0; others ~ N(0,0.01²). Sizes: n=1000, m=620, maxout l=500, n'=1000. Beam search at decode.

## Design-decision → why table

| choice | why this, not the alternative |
|---|---|
| sequence of annotations vs 1 vector | fixed c is an info bottleneck; long-sentence BLEU collapse (Cho 2014b) is the evidence |
| per-step c_i | each target word needs different source info; one c forces averaging-away |
| weighted sum (expected annotation) | differentiable surrogate for "pick relevant positions"; avoids hard discrete selection |
| soft (deterministic) alignment vs latent var + EM (IBM) | soft = differentiable → joint backprop training; discrete latent needs EM, not end-to-end |
| additive MLP score (not dot product) | query dim n ≠ key dim 2n; MLP projects both to R^{n'}; learnable nonlinear comparator |
| query = s_{i-1} | pre-emission state = "what I need next"; s_i would be circular |
| BiRNN encoder | annotation must summarize BOTH left and right context around x_j |
| concat [→h;←h] | keep both directional summaries; recency bias focuses h_j near x_j |
| GRU units | gated paths, derivative-product≈1, fights vanishing gradient; simpler than LSTM |
| inject c into s̃,z,r AND output | context conditions both the recurrence and the emission |
| no NULL token | soft weights handle many-to-one & length mismatch naturally |
| precompute U_a h_j | independent of i → cut Tx·Ty cost roughly in half |
| grad clip ≤1 | RNN exploding gradients (Pascanu 2013) |
| orthogonal recurrent init | preserve gradient norm through recurrence |

## Canonical implementation (grounding)

bentrevett/pytorch-seq2seq tutorial 3 (downloaded → code/bentrevett_code.py). Faithful: bidirectional GRU encoder (`nn.GRU(..., bidirectional=True)`), `Attention` = additive MLP `attn_fc` then `v_fc` (v_a, bias-free) → softmax; context via `torch.bmm(a, encoder_outputs)`; decoder GRU input = [embedded ; weighted-context]; output projection over [output ; weighted ; embedded]; decoder hidden init = tanh(linear(cat(fwd_last, bwd_last))). Teacher forcing, Adam, CrossEntropy(ignore pad), grad clip. d2l AdditiveAttention cross-checks the score formula. The scaffold in context.md = this with the attention/context/per-step-c bodies hollowed to TODO.

## In-frame reminders
- Never say "this paper"/authors/arXiv/RNNsearch-as-paper. May name "soft attention / alignment model" as the thing being built (mostly answer.md).
- Cite ancestors (Sutskever 2014, Cho 2014, Schuster&Paliwal 1997, Graves 2013, Koehn 2003, Pascanu 2013) freely.
- context.md: NO outcome numbers (no BLEU table, no RNNsearch wins). The long-sentence-collapse diagnostic about the EXISTING encoder–decoder IS allowed (motivating finding).
- reasoning.md ends before the RNNsearch benchmark; no fabricated wins.
