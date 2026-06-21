# Performer (FAVOR+) — synthesis

## Pain point
Softmax attention: A = exp(QKᵀ/√d), Att = D⁻¹AV, D=diag(A1_L). Time O(L²d), space O(L²+Ld) because A is materialized. Quadratic in sequence length L → blocks long sequences (proteins L=8192, ImageNet64 L=12288, PG-19).

## Load-bearing ancestors
- **Vaswani et al. 2017 (Transformer)**: defines dot-product attention; the O(L²) object to fix.
- **Sparse/local attention** (Sparse Transformer Child et al. 2019; Longformer; Image Transformer Parmar 2018; Routing Transformer k-means): restrict attention to neighborhoods/learned sparse patterns. Don't approximate full softmax; need hand-built sparsity patterns / custom CUDA/TVM kernels; no rigorous guarantee on representation power.
- **Reformer (Kitaev et al. 2020)**: LSH to bucket similar tokens → O(L log L). Requires shared Q=K; approximation, not unbiased; relies on sparsity prior.
- **Linformer (Wang et al. 2020)**: low-rank projection of K,V to k rows → O(Lk). Biased, non-causal only, large MSE; assumes attention is low-rank.
- **Linear Transformer / Transformers-are-RNNs (Katharopoulos et al. 2020, trans-rnns)**: replace softmax with φ(q)ᵀφ(k), φ=elu+1, reassociate to linear cost. But φ chosen ad hoc; does NOT approximate softmax; numerically unstable (exploding gradients/NaN observed).
- **Rahimi & Recht 2007 (Random Fourier Features, fourierapprox)**: shift-invariant kernel K(x-y)=E_ω[cos]; Gaussian kernel via φ=(sin,cos) of ωᵀx, ω~N(0,I). The RFF template.
- **Yu et al. 2016 (Orthogonal Random Features, ort) / Choromanski et al. (unreas, geom)**: make ω_i orthogonal to reduce MC variance; previously only asymptotic-in-d guarantees.

## The derivation chain (discovery order)
1. Softmax is quadratic only because A is built then multiplied. If A(i,j) = φ(q_i)ᵀφ(k_j) (a dot product of feature maps), then AV = Q'(K'ᵀV) by associativity → O(Lrd). Need: write exp(qᵀk) as E[φ(q)ᵀφ(k)].
2. Generalized kernelizable attention: A(i,j)=K(q_i,k_j)=E[φ(q_i)ᵀφ(k_j)], φ:R^d→R_+^r. Att̂ = D̂⁻¹(Q'((K')ᵀV)), D̂=diag(Q'((K')ᵀ1_L)). Space O(Lr+Ld+rd), time O(Lrd).
3. RFF template: φ(x)=(h(x)/√m)(f_1(ω_1ᵀx),...,f_l(ω_mᵀx)). Trig: h=exp(‖x‖²/2), f=(sin,cos) gives unbiased SM estimate (SM=exp(‖x‖²/2)·K_gauss·exp(‖y‖²/2)).
4. WALL: trig features can be negative. Attention is a convex combination weighted by normalized kernel scores; need non-negative A. Worse: variance of trig estimator blows up as SM→0 (many low-relevance entries). D̂⁻¹ can go negative → unstable / NaN. (Empirically: trig softmax training unstable.)
5. POSITIVE features. Want exp form. Gaussian integral: exp(xᵀy)=exp(-‖x‖²/2)exp(‖x+y‖²/2)exp(-‖y‖²/2); and exp(‖x+y‖²/2)=E_{ω~N(0,I)}[exp(ωᵀx)exp(ωᵀy)] (complete the square). So SM(x,y)=E[exp(ωᵀx-‖x‖²/2)·exp(ωᵀy-‖y‖²/2)]. φ⁺(u)=(exp(-‖u‖²/2)/√m)(exp(ω_1ᵀu),...,exp(ω_mᵀu)): unbiased, strictly positive.
6. hyp+: split exp(±u), h=exp(-‖x‖²/2)/√2, halves variance (cancels covariance of cosh).
7. MSE lemma (App H.2):
   - MSE(trig)=(1/2m)exp(‖z‖²)SM⁻²(1-exp(-‖Δ‖²))², z=x+y, Δ=x-y. → ∞ as SM→0.
   - MSE(+)=(1/m)exp(‖z‖²)SM²(1-exp(-‖z‖²)). → 0 as SM→0 (since SM²→0). This is the punchline: positive estimator is accurate exactly where it matters.
   - MSE(hyp+)=½(1-exp(-‖z‖²))MSE(+).
8. ORTHOGONAL features (FAVOR+). Entangle ω_i exactly orthogonal within a Gaussian block (Gram-Schmidt/QR), marginals unchanged after chi-length scaling → still unbiased for isotropic D. A single block has m≤d, while implementations stack independent d×d blocks plus a partial block when more features are requested. "Beautiful function" F_{Ω,g}(z)=E[g(ωᵀz)], g entire with nonneg power-series coeffs (exp qualifies). Theorem (general-var): MSE(ort) ≤ MSE(iid) - (1-1/m)(2/(d+2))(F(z)-a_0)². For SM, F-a_0 = SM·exp((‖x‖²+‖y‖²)/2) - 1 form → gives main-text Thm 3 gap. Holds for ALL d, not just asymptotically. Positivity (a_i≥0) is exactly what makes the τ≤d/(d+2) bound give a strictly positive gap.
9. SMREG (regularized softmax): replace ω by √d·ω/‖ω‖ (sample on sphere radius √d). Thm: SMREG≤SM and ratio ≥ 1 - 2/d^{1/3}+o(...). So SMREG is a tight, universal lower-bound proxy; ORF concentration even sharper for it.
10. Unidirectional/causal: need tril(Q'K'ᵀ)C without forming L×L. Prefix-sum: G_j = K'_j C_jᵀ (outer product, M×(d+1)), G^PS_i = Σ_{j≤i}G_j, output row i = G^PS_i × Q'_i. O(Lmd) time, parallel prefix sum O(log L).
11. Uniform approximation uses m = Θ((d/δ²) log(4 d^{3/4}R/δ)) under bounded query/key norm R, so the feature count depends on dimension, radius, and precision but not directly on sequence length L.
12. Generalized attention: any nonnegative φ can be used; the Google reference default is deterministic φ(x)=ReLU(x)+10^{-3}, with iid/orthogonal projected variants optional. Drop-in.

## Design decisions → why
- Non-negative features: convex-combination semantics of attention; negative scores → negative D̂⁻¹ → NaN.
- exp (not sin/cos): unbiased AND positive AND variance→0 as kernel→0.
- orthogonal ω: variance reduction provable for all d; positivity needed for the bound.
- ORF is exact within d-row blocks; uniform-approximation feature count depends on d, query/key radius R, and error tolerance, not directly on L.
- normalize by d^{-1/4} per vector (so qᵀk/√d folded into renorming q,k): code data_normalizer = dim^{-0.25}.
- numerical stabilizer: subtract max in exponent (log-sum-exp style) before exp; +eps.
- redraw features periodically: avoid unlucky fixed projection.
- ReLU generalized kernel: Google reference default is deterministic ReLU plus 10^{-3}; no softmax d^{-1/4} normalization unless explicitly configured.

## Code grounding
Google reference implementation (`code/google-research`, commit `4fde028f6017e16aefcbc2b6d3f77f70b9f6b421`): `nonnegative_softmax_kernel_feature_creator` (φ⁺ with query/key max subtraction and stabilizer), `generalized_kernel_feature_creator` (deterministic ReLU default, optional projections), `GaussianOrthogonalRandomMatrix` (block QR plus chi-distributed row norms or sqrt(d) scaling), and `FastAttentionviaLowRankDecomposition` (noncausal reassociation, causal scan numerator/denominator, denominator stabilizer, feature redraw policy).
