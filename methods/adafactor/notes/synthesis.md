# Adafactor synthesis

## Pain point / research question
- Adam keeps 2 extra accumulators per param (m, v) → triples memory. v alone matches model size.
- For huge weight/embedding matrices (billions of params, MoE, large MT models) this auxiliary storage becomes the binding constraint on model size, since memory capacity grew slower than compute.
- Goal: keep Adam-style per-coordinate adaptivity (the divide-by-sqrt(second moment)) but at sublinear extra memory for matrix params, with comparable quality.

## Core derivation chain (discovery order)
1. v_t is an EMA of g_t^2, has same shape as the matrix W in R^{n×m}. Want to store O(n+m) not O(nm).
2. Idea: store low-rank factors R∈R^{n×k}, S∈R^{k×m}, V≈RS. Need k small; rank-1 (k=1) gives O(n+m).
3. Wall #1: standard low-rank approx = truncated SVD (Eckart-Young, Frobenius-optimal). But:
   - SVD factors don't decompose over addition → incompatible with exponential smoothing (moving avg of factors ≠ factors of moving avg).
   - SVD not guaranteed nonnegative → but we need V̂≥0 to take 1/sqrt(V̂). Blocker.
4. Need: nonnegative + linear-in-V (so EMA commutes with factoring). Turn to NMF cost functions.
   Generalized KL / I-divergence: d(p,q) = p log(p/q) − p + q. Nonneg, =0 iff p=q (from x log x ≥ x−1).
5. Minimize Σ_ij d(V_ij, [RS]_ij) over R≥0, S≥0. General rank-k is hard (alternating min, Finesso-Spreij). But rank-1 has closed form.
6. Lemma (rank-1): [RS]_ij = R_i S_j. Expand loss:
   Σ V log V − Σ V log R_i − Σ V log S_j − Σ V + Σ R_i S_j.
   ∂/∂R_i = 0: −Σ_j V_ij/R_i + Σ_j S_j = 0 → R_i = (Σ_j V_ij)/(Σ_j S_j).
   ∂/∂S_j = 0: −Σ_i V_ij/S_j + Σ_i R_i = 0 → S_j = (Σ_i V_ij)/(Σ_i R_i).
   Scale ambiguity (αR, S/α). Fix Σ R_i = Σ_ij V_ij ⇒ R_i = Σ_j V_ij (row sums), S_j = (Σ_i V_ij)/(Σ_ij V_ij) (col sums normalized).
   Vector form: R = V 1_m, S = 1_n^T V / (1_n^T V 1_m). Reconstruction V̂ = R S = V 1_m 1_n^T V / (1_n^T V 1_m).
   Elementwise V̂_ij = R_i C_j / (Σ_k R_k) with C = 1_n^T V the col sums.
7. KEY property: V̂ depends on V only through row sums (V 1_m) and col sums (1_n^T V), which are LINEAR in V. So EMA of row sums = row sums of EMA. Exponential smoothing now commutes → maintain only R_t∈R^n, C_t∈R^m. O(n+m). Exact recovery if V is already rank-1.
8. Algorithm 2 (factored Adam, β1=0): R_t = β2 R_{t-1} + (1−β2)(G²)1_m ; C_t = β2 C_{t-1} + (1−β2)1_n^T(G²); V̂_t = (R_t C_t/1_n^T R_t)/(1−β2^t); X_t = X − α G/(sqrt(V̂)+ε). Note 1_n^T R_t = C_t 1_m, symmetric.

## Additional pieces
9. Drop first moment (β1=0) to save the OTHER accumulator → vectors/scalars now zero extra, matrices O(n+m). But removing momentum without warmup → instability (BLEU collapse 23.1→0.1 without warmup).
10. Diagnose instability: define RMS(U_t) = sqrt(mean_x (g²/v̂)). If v̂ tracks g² well, ratio≈1. With slow decay (β2=0.999) and no warmup, RMS(U_t) fluctuates >>1 → larger-than-desired updates → divergence. (Reddi et al. 2018 / "On the convergence of Adam": slow decay = out-of-date estimator.) Fast decay (β2=0.9) keeps RMS≈1 but Reddi shows fast decay hurts convergence. Tension.
11. Remedy A — UPDATE CLIPPING: clip the unscaled update U = G/sqrt(V̂) by its RMS:
    Û_t = U_t / max(1, RMS(U_t)/d), d=1. Caps the actual step RMS, unlike gradient clipping which caps only the gradient norm (adaptive scaling can still blow up the update). Cures the no-warmup instability (d=1 works, d=2 doesn't).
12. Remedy B — INCREASING DECAY β̂2_t. Adam's bias correction IS an increasing decay: β̂2_t = β2(1−β2^{t-1})/(1−β2^t), starts 0, →β2. Generalize: propose β̂2_t = 1 − t^{-c}, c>0.
    - Proof it removes bias correction: expand v_t = Σ_i (1−β̂2_i) Π_{j>i} β̂2_j g_i². For E[v_t]=E[g_t²] in stationary case need Σ_i (1−β̂2_i)Π_{j>i}β̂2_j = 1. Induction: t=1 gives 1−β̂2_1=1 (since β̂2_1=0). Step holds. Works for any schedule with β̂2_1=0.
    - Need past-gradient weight →0: lim_t (1−β̂2_i)Π_{j=i+1}^t β̂2_j = 0 ∀i. With β̂2_j=1−1/j^c reduces to Π(1−1/j^c)→0 iff Σ 1/j^c diverges iff c≤1. So 0<c≤1. c=1 → simple arithmetic average v_t = (Σ g_i²)/t. Recommend c=0.8.
13. RELATIVE STEP SIZE: Adam's α is absolute target step. Hinton intuition: param updates should be ~1e-2..1e-3 × param magnitude (relative). Define scale = RMS(X), lower-bounded by ε2=1e-3 (so zero-init params can escape 0). α_t = max(ε2, RMS(X_{t-1}))·ρ_t. Robust to differently-scaled embeddings; removes need for the "clever" embedding rescale-by-sqrt(d_model) trick.

## Final Adafactor (Alg 6 matrix / 7 vector + HP)
- α_t = max(ε2, RMS(X_{t-1})) ρ_t
- R_t = β̂2_t R_{t-1} + (1−β̂2_t)(G²+ε1 1_n1_m^T)1_m ; C_t = β̂2_t C_{t-1}+(1−β̂2_t)1_n^T(G²+ε1...)
- V̂_t = R_t C_t / 1_n^T R_t   (NO bias correction — handled by β̂2_1=0 schedule)
- U_t = G_t/sqrt(V̂_t) ; Û_t = U_t/max(1, RMS(U_t)/d) ; X_t = X − α_t Û_t
- Vector case: V̂_t = β̂2_t V̂_{t-1}+(1−β̂2_t)(G²+ε1) (no factoring), rest same.
- HP: ε1=1e-30, ε2=1e-3, d=1, ρ_t=min(1e-2, 1/sqrt(t)), β̂2_t=1−t^{-0.8}.
- ε1 only prevents div-by-zero in the squared gradient; added INSIDE the accumulator.

## Code grounding (transformers/fairseq + T2T)
- factored = ndim>=2. Store exp_avg_sq_row (shape[:-1]), exp_avg_sq_col (shape[:-2]+shape[-1:]).
- Implementations track row/col MEANS (reduce_mean over last/second-last axis) not sums; reconstruction:
  r_factor = rsqrt(vr / mean(vr, last dim)); c_factor = rsqrt(vc); update = grad * r_factor[...,None]*c_factor[...,None,:].
  Mean-vs-sum: V̂_ij = R_i C_j/(Σ R). With means vr_i = (1/m)Σ_j V_ij, vc_j=(1/n)Σ_i V_ij; rsqrt(vr_i/mean_i(vr)) * rsqrt(vc_j) reproduces 1/sqrt(V̂_ij) up to the per-matrix constant absorbed by clipping/lr. Mathematically equivalent to sums form.
- _rms = norm2/sqrt(numel). lr = param_scale*rel_step. beta2t = 1 - step^decay_rate (decay_rate=-0.8). clip: update /= clamp(rms(update)/d, min=1). beta1 optional first moment.

## In-frame discipline
- Never name "Adafactor paper"/authors/arXiv. Method name "Adafactor" OK in answer.md as the thing being built.
- Adam, RMSProp, Adagrad, Adadelta, NMF (Lee-Seung), Eckart-Young, Finesso-Spreij, Reddi et al., Pascanu (grad clipping), Hinton intuition, Shazeer MoE appendix D = prior-art citations, fine.
</content>
</invoke>
