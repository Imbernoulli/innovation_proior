# Synthesis — outlier-driven rescaling (method short-name: `outlier-rescaling`)

## What pain point existed?

Transformer LLMs reliably grow **outliers** — values orders of magnitude larger than the rest:
- **Attention sinks**: a few tokens (usually the first / BOS) get enormous attention logits and absorb a large fraction of the softmax mass (Xiao 2023, StreamingLLM).
- **Massive activations (MA)**: a few coordinates of a few special tokens hit ~1000s in magnitude (Sun 2024).
- **Residual / outlier-feature dimensions**: a *fixed set of dimensions* carries persistently large activations across *most* tokens, input-independent (Dettmers 2022 "GPT3.int8()"; Kovaleva 2021 BERT busters; He 2024).

These outliers are practically harmful: they dominate the dynamic range under quantization (W4A4/W8A8 fail), and cause larger floating-point error. The obvious fix — clip them, or remove the normalization that seems to create them — *destroys* model quality. So there is a tension: outliers hurt deployment yet seem load-bearing for training. The open question: **what functional role do these outliers play, and can we get the benefit without the large values?**

## The central object / claim

Unify all of these as **outlier-driven rescaling**: an outlier is large *not because it carries information downstream*, but because it acts as a **scale knob inside a normalization**. Normalization (softmax in attention; RMSNorm in the residual) divides by a statistic of the whole vector. If one coordinate/token is huge, it dominates that statistic, which **shrinks every other (non-outlier) component after normalization**. So the outlier sets the relative scale of the useful components. Both sink types are the *same mechanism* expressed through two different normalizers.

Key falsifiable predictions, each validated:
1. **Outliers + normalization are a unit.** Remove the normalization → outliers vanish but training stability/perf degrade (DyT for RMSNorm; sigmoid/linear attention for softmax). Keep the normalization but clip the outliers → also degrades / diverges. → outliers function *jointly* with normalization to rescale.
2. **Outliers are scale factors, not contributors.** The RMSNorm weight `λ` on an outlier dimension is tiny (e.g. 0.004–0.06 vs ~1): the network deliberately *dampens* the outlier's direct contribution right after using it for scaling. Mirror of the attention-sink fact: sink tokens' value vectors have small norm.
3. **Outliers can be moved into parameters.** A learnable pre-norm scaling vector `λ1` (PreAffine) amplifies the fixed dimensions, so the residual stream stays small but the input to RMSNorm still has the large values needed for rescaling — analogous to learnable sink tokens/biases that absorb attention sinks.
4. **If rescaling is the purpose, give it explicitly → outliers shrink.** Insert a low-rank sigmoid self-gate after each norm (GatedNorm): the network gets an explicit rescaling pathway and stops needing outliers. Loss preserved/improved, activations smooth, quantization robust.
5. **Reduced reliance on outliers → robustness to architecture choices.** With GatedNorm, DyT converges and sigmoid-GLU matches/beats SwiGLU — because the only reason SwiGLU beat GLU was that swish produces larger activations that *feed* outlier-driven rescaling.

## The appendix proof (worked, and a correction)

Setup: input feature `h ∈ R^D`, RMSNorm/LN with affine weight `λ`, one outlier dim `d`. Two assumptions, both motivated by observation:
- the outlier dim has a *small* affine weight: `|λ_d| ≤ ε·||λ||∞` (observed: 0.004 vs 1);
- the outlier accounts for ratio `r` of the feature norm: `r = |h_d| / ||h||₂`.

Define `u = h / ||h||_rms`, so `||u||_rms = 1`, i.e. `(1/D)Σ u_i² = 1` ⇒ `||u||₂² = D`. Also `|u_d| = r·... ` — careful: the paper writes `|u_d| = r`. Note `h/||h||_rms` scales by `√D` relative to `h/||h||₂`, so let me track it as the paper does.

RMSNorm output: `LN(h) = (λ ⊙ h)/||h||_rms`. Then
`||LN(h)||₂ = ||λ⊙h||₂ / ||h||_rms = √D · ||λ⊙h||₂/||h||₂ = √D · ||λ⊙u||₂` where here `u = h/||h||₂` (unit-2-norm version).
With `|u_d| = r`, `Σ_{i≠d} u_i² = 1 − r²`:
`||λ⊙u||₂² = λ_d² r² + Σ_{i≠d} λ_i² u_i² ≤ λ_d² r² + ||λ_{-d}||∞²(1−r²)`.
So
`||LN(h)||₂ ≤ √D · √( ||λ_{-d}||∞²(1−r²) + λ_d² r² )`.    (the actually-derived bound)

Converting to RMS norm of the output (divide by √D): `||LN(h)||_rms ≤ √( ||λ_{-d}||∞²(1−r²) + λ_d² r² )`. Using `|λ_d| ≤ ε||λ||∞` and bounding `||λ_{-d}||∞ ≤ ||λ||∞`:
`||LN(h)||_rms ≤ ||λ||∞ · √( (1−r²) + ε² r² )`.    (the stated abstract-level bound)

Since `ε < 1`, the bracket `(1−r²) + ε²r² = 1 − (1−ε²)r²` **decreases as r grows**. So a bigger outlier (larger `r`) ⇒ smaller post-norm feature norm. The outlier is a *dial* on the output magnitude. NOTE for reasoning: the tighter intermediate form keeps `||λ_{-d}||∞` and `λ_d²` separate; the clean `||λ||∞√(1−r²+ε²r²)` is the relaxed, presentable version. Both correct; the relaxation just upper-bounds `||λ_{-d}||∞ ≤ ||λ||∞` and `λ_d² ≤ ε²||λ||∞²`.

## Design-decision → why table (with rejected alternatives)

| Decision | Why this | Rejected alternative & its failure |
|---|---|---|
| Frame both sinks as *rescaling*, not *bias/feature* | Sink value-vectors have small norm; outlier-dim RMSNorm weights are ~0.004 → the network suppresses their direct contribution, so they can't be carrying content; what's left is scale. | "Outliers are important features" → contradicted by tiny λ and tiny value-norm. "Just artifacts to delete" → clipping diverges. |
| Diagnose via *removing normalization* (DyT, linear/sigmoid attn) | If the outlier exists to drive a normalizer, killing the normalizer should kill the outlier — clean causal test. | — |
| Diagnose via *clipping* while keeping norm | Separates "outlier" from "normalization": if clipping (norm intact) also hurts, the *large value itself* is needed by the norm, not incidental. | — |
| GatedNorm = **low-rank** (r=16) sigmoid self-gate after norm | Cheap (3.7M params ≈ 2% of 2B; <5% latency, shrinks with scale); rank-16 enough because rescaling is low-dimensional. | Full-rank gate = expensive, no quality gain. |
| **sigmoid** activation in the gate (not tanh/SiLU/identity) | Bounded in (0,1) and fine-grained near 0 → stable, monotone down-scaling; matches the "outlier shrinks the rest" semantics. Empirically tanh/SiLU/none → unstable outlier dynamics; tensorwise+non-sigmoid → divergence. | tanh/SiLU/identity (adaLN-style) → unstable/divergent. |
| **elementwise** gate (score shape d), not tensorwise (shape 1) | Per-dimension rescaling is finer modulation → better loss; tensorwise reduces sinks similarly but only matches baseline. | tensorwise: mitigates outliers but no perf gain. |
| Gate form `σ(W_up·swish(W_down·y)) ⊙ y` | Two-layer low-rank bottleneck gives input-dependent per-dim scale; swish inside is the standard cheap nonlinearity in the bottleneck; outer sigmoid does the bounded gating. | linear gate `σ(W y)` is the GatedAttention form; the bottleneck MLP gives more capacity at low cost. |
| **PreAffine**: `RMSNorm(λ1 ⊙ x)` with trainable `λ1∈R^d` | Sinks live in *fixed dims across tokens* ⇒ a static per-dim vector can amplify exactly those dims, so the *residual* stays small but the RMSNorm input still has the large values that drive rescaling — "absorb the outlier into a parameter." | Just clip → loses rescaling. Standard RMSNorm `λ` (post-norm) can't do it: post-norm scaling is pointwise and can't change the RMS the division uses. |
| Keep RMSNorm (don't adopt DyT as the fix) | DyT is pointwise → no cross-dim coupling → can't rescale → unstable, +0.259 loss. Want to *keep* the rescaling, just make it explicit & outlier-free. | DyT as replacement → degrades. |
| Reduce FFN width to keep param count constant when adding gate | Fair comparison: any gain must come from rescaling, not extra params. | — |
| Why SwiGLU>GLU explained, not assumed | swish is unbounded → larger FFN activations → more outlier headroom → feeds outlier-driven rescaling. With GatedNorm supplying rescaling explicitly, the advantage evaporates (GLU ≈ or > SwiGLU). Answers Shazeer 2020's "we offer no explanation … divine benevolence." | Believing SwiGLU is intrinsically better → wrong; it's a proxy for rescaling capacity. |

## Load-bearing ancestors (write-ups)

- **Softmax attention / Transformer (Vaswani 2017).** Attention = softmax(QKᵀ/√d)V. Softmax forces weights to sum to 1 — no "attend to nothing" option, so the network parks excess mass on a fixed token. This *normalization-forces-a-sink* property is the root of attention sinks.
- **StreamingLLM / attention sinks (Xiao 2023).** First tokens get persistent huge attention; keeping their KV ("sink tokens") restores windowed-attention quality. Establishes the phenomenon + that sinks are functionally necessary (removing them collapses output). Gap: descriptive, no mechanism beyond "softmax dumps mass."
- **Massive activations (Sun 2024).** A few coordinates of special tokens are ~constant, input-agnostic, act as *implicit bias*; co-locate with attention sinks; fixed by a learnable attention bias / sink token. Gap: treats them as bias; this work reframes as scale and extends to dimension-wise residual sinks.
- **When Attention Sink Emerges (Gu 2024)** & **Systematic Outliers (An 2025).** Sinks tie to softmax normalization; An 2025 + Bondarenko 2023 argue attention outliers are *context-aware scale factors* and that gating in attention removes them. Direct conceptual parent of "outlier-driven rescaling"; gap: only attention side.
- **Quantizable Transformers / GatedAttention (Bondarenko 2023; Qiu 2025 "Gated Attention").** A head-wise/element-wise sigmoid gate `attn_out * σ(x Wθ)` after SDPA lets heads "do nothing" without a sink → removes attention sinks, improves stability/perf, deployed in Qwen3-Next. This is the *attention-side template* that GatedNorm copies onto the residual side.
- **RMSNorm (Zhang & Sennrich 2019).** `y = x/√(mean(x²)+ε) ⊙ λ`. Drops LayerNorm's re-centering; keeps re-scaling invariance. The division-by-RMS is exactly the coupling that lets one big dim shrink the rest — the residual-side normalizer in which residual sinks operate.
- **DyT / Transformers without Normalization (Zhu 2025); Chen 2025 stronger NF-transformers.** `DyT(x)=γ·tanh(αx)+β`, pointwise, drop-in for LN/RMSNorm. Removes outliers but is *pointwise* — no cross-dimension coupling — so it cannot rescale; used here as the "remove normalization" probe (diverges / +0.259).
- **He 2024 "outlier features in transformer training."** Outlier features (kurtosis of activation norms); attributes them to normalization; proposes Outlier-Protected (un-normalized) block. Establishes normalization↔outlier link + that removing norm reduces outliers; gap: removes the rescaling benefit.
- **GLU / SwiGLU (Shazeer 2020).** `SwiGLU(x)=W_down((W_up x) ⊙ swish(W_gate x))`. Beats ReLU/GELU/sigmoid-GLU, "no explanation … divine benevolence." Here explained: swish's larger activations feed rescaling.
- **StyleGAN2 demodulation (Karras 2020) & adaLN/FiLM (Perez 2018, Peebles 2023).** Normalization-as-source-of-outliers in vision; input-dependent gating replicates the scaling effect *without* normalization. Cross-domain precedent that gating can stand in for normalization-driven rescaling.
- **Quantization difficulty (Dettmers 2022 GPT3.int8(); SmoothQuant Xiao 2023; outlier suppression Wei 2022/2023; NVFP4).** Outliers dominate dynamic range → migrate difficulty activation→weight (SmoothQuant) or Hadamard-rotate. Motivates *why we care*: smoother activations = better low-bit.

## Code grounding (1.4)
- `code/modeling_qwen3_gated.py` (Qiu 2025 official): exact RMSNorm (`Qwen3RMSNorm`), pre-norm decoder layer (`input_layernorm` before attn, `post_attention_layernorm` before MLP, residual adds), and gated attention: gate dims split off `q_proj`, `attn_output = attn_output * torch.sigmoid(gate_score)` before `o_proj`. SwiGLU = `down(act(gate(x)) * up(x))`.
- `code/dynamic_tanh.py` (Zhu 2025 official): `tanh(alpha*x)*weight+bias`.
Final code: GatedNorm = RMSNorm subclass adding `W_down(d→r) → swish → W_up(r→d) → sigmoid`, multiplied elementwise into the normalized output; PreAffine = RMSNorm with extra `λ1` applied to input before computing RMS.
