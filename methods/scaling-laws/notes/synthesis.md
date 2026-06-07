# Scaling Laws (Kaplan et al.) synthesis (Phase 1.5)

## Verified
- arXiv 2001.08361, "Scaling Laws for Neural Language Models". Title verified from source.
- No code repo (OpenAI, closed). The reproducible artifact = the power-law functional forms + their fitting procedure + the compute-optimal allocation derivation. Code deliverable = parameter/compute counting + curve-fitting of the power laws + the closed-form optimal allocation. Cross-check the constants against the paper's tables.

## Pain point / question
- Transformers improve with scale, but it's all anecdotal/ad hoc. Want: a PREDICTIVE, quantitative theory — how does test loss depend on model size N, data D, compute C, training time S, batch B? If loss is predictable, you can decide model size / data / compute allocation BEFORE training.
- Key empirical phenomenon (motivating, in scope): loss falls as a clean power law in each of N, D, C over many orders of magnitude, when not bottlenecked by the others. Architecture details (depth, width, #heads) barely matter at fixed N — performance is set by SCALE, not shape.

## Definitions / counting (DERIVE)
- Transformer hyperparams: n_layer, d_model, d_ff, d_attn, n_heads, n_ctx (=1024 default).
- Non-embedding param count (exclude vocab/positional embeddings → cleaner laws):
  N ≈ 2 d_model n_layer (2 d_attn + d_ff) = 12 n_layer d_model^2  with standard d_attn = d_ff/4 = d_model.
- Forward compute per token: C_forward ≈ 2N + 2 n_layer n_ctx d_model. (factor 2 = multiply-accumulate.) For d_model >> n_ctx/12 the context term is negligible → C_forward ≈ 2N.
- Backward ≈ 2× forward → C ≈ 6N FLOPs per training token. So over D tokens, C ≈ 6ND.

## The three single-variable power laws (the basic fits)
- L(N) = (N_c/N)^{α_N}, α_N ~ 0.076, N_c ~ 8.8e13 (non-embed params).
- L(D) = (D_c/D)^{α_D}, α_D ~ 0.095, D_c ~ 5.4e13 (tokens).
- L(C_min) = (C_c^min/C_min)^{α_C^min}, α_C^min ~ 0.050, C_c^min ~ 3.1e8 PF-days.
- These hold over 6+ orders in N, 2+ in D, 8 in C_min. Weak dependence on shape.
- Doubling N → loss × 2^{-α_N} = 0.95.
- Fit method: train models, take converged/early-stopped test loss, linear fit in log-log (log L vs log X → slope = -α_X).

## Combined L(N,D) ansatz (DERIVE the form from principles)
- L(N,D) = [ (N_c/N)^{α_N/α_D} + D_c/D ]^{α_D}.
- Three principles forcing this form:
  1. Vocab/tokenization rescales loss by overall factor → must allow rescaling (absorb into N_c, D_c) → they have no fundamental meaning.
  2. Limits: fix D, N→∞ → L→L(D); fix N, D→∞ → L→L(N). (Knowing L(N) at D=∞ and L(D) at N=∞ fully determines all params.)
  3. Analytic at D=∞: a 1/D series expansion with integer powers (overfitting variance ∝ 1/D). This breaks the N↔D symmetry — a symmetric form [(N_c/N)^{α_N}+(D_c/D)^{α_D}]^β has no clean 1/D expansion and needs an extra param.
- Check principle 2: as N→∞, (N_c/N)^{α_N/α_D}→0, so L→(D_c/D)^{α_D}=L(D). ✓. As D→∞, D_c/D→0, L→[(N_c/N)^{α_N/α_D}]^{α_D}=(N_c/N)^{α_N}=L(N). ✓.
- Fit: α_N=0.076, α_D=0.103, N_c=6.4e13, D_c=1.8e13.
- Overfitting governed by ratio N^{α_N/α_D}/D. Equal-loss tradeoff: increase D sublinearly, D ∝ N^{α_N/α_D} ~ N^{0.74}, to avoid overfitting.
- δL = L(N,D)/L(N,∞) - 1 ≈ (1 + (N/N_c)^{α_N/α_D} D_c/D)^{α_D} - 1.

## L(N,S) — model size and training steps (infinite data limit)
- L(N, S_min) = (N_c/N)^{α_N} + (S_c/S_min)^{α_S}, α_N=0.077, α_S=0.76, N_c=6.5e13, S_c=2.1e3.
- S_min = adjusted steps (at B>>B_crit). Note this is ADDITIVE (sum of two power laws), unlike L(N,D) which is the bracket-power form — because here it's infinite data, the two terms are capacity-limit + optimization-limit and just add.

## Critical batch size (Bcrit) — needed for compute-optimal
- From gradient-noise-scale theory (McCandlish et al. 1812.06162): below B_crit, increasing B costs ~no extra compute but saves steps; above, diminishing returns. Tradeoff law: (S/S_min - 1)(E/E_min - 1) = 1, B_crit ≡ E_min/S_min.
- B_crit(L) = B_*/L^{1/α_B}, B_* ~ 2e8 tokens, α_B ~ 0.21. Depends on loss only, not model size. Doubles per 13% loss decrease.
- S_min(S) = S/(1 + B_crit/B); C_min(C) = C/(1 + B/B_crit). C_min = compute if trained at B<<B_crit (min compute); S_min = steps if trained at B>>B_crit (min steps).

## Compute-optimal allocation (THE key result — DERIVE)
- C_min ≡ 6 N B_crit S_min. Want N(C_min) minimizing loss at fixed compute.
- Substitute S_min = C_min/(6 N B) into L(N,S_min) and minimize over N.
- Result: optimal scalings N ∝ C^{α_C^min/α_N}, B ∝ C^{α_C^min/α_B}, S ∝ C^{α_C^min/α_S}, D=B·S, with
  α_C^min = 1/(1/α_S + 1/α_B + 1/α_N).
- Derivation intuition: at the optimum each contribution to the loss scales with the same power of C; the combined exponent is the harmonic-style combination (reciprocal of sum of reciprocals) of the three exponents, because the three "directions" of spending compute (size, batch, steps) each have power-law efficiency and compose multiplicatively under C = 6 N B S.
- Numerics: 1/α_S=1.316, 1/α_B=4.762, 1/α_N=12.99 (α_N=0.077); sum=19.07; α_C^min=0.0524≈0.054 (paper quotes ~0.05).
- N(C_min) ∝ C^{α_C^min/α_N} = C^{0.0524/0.076} ≈ C^{0.71} (paper); empirically C^{0.73}.
- B ∝ C^{0.24}, S ∝ C^{0.03} (≈ constant). D = B·S ∝ C^{0.27}.
- CONCLUSION: spend extra compute PRIMARILY on bigger models (N ∝ C^0.73), modest batch growth, almost no extra steps; data grows only ~C^0.27. Bigger models are more sample-efficient. (This is the recipe later revised by the compute-optimal correction — but that's posterior, OUT of scope.)

## Contradiction / conjecture
- L(C_min) and L(D) extrapolations cross far beyond studied scales (data grows too slowly with compute to keep feeding the compute-optimal model). Laws must break before then; intersection conjectured to mark max achievable performance / where the simple picture ends.

## Scaffold ↔ final code correspondence
- transformer_param_count(...) / forward_flops_per_token(...) ← context stub: counting helpers
- fit_power_law(X, L) — log-log linear fit → (X_c, alpha) ← context stub: single-variable fit
- L_ND(N,D,params) and fit_L_ND(runs) ← context stub: joint loss model + fit
- compute_optimal_exponents(α_N,α_S,α_B) → α_C^min and N,B,S exponents ← context stub: allocation solver

## OUT of scope
- LSTM-vs-Transformer context-position curves, transfer-to-other-distributions specifics (these are mostly motivating/diagnostic and can be touched lightly), sample-efficiency plots. The proposed-method "results" here ARE the laws themselves (this paper's contribution is the empirical laws + derivations), so those are in scope; only skip pure benchmark-style comparison figures.
