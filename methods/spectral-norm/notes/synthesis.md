# Synthesis — Spectral Normalization for GANs (slug: spectral-norm, arXiv 1802.05957)

## Pain point / research question
GAN training is unstable. The generator only learns through the discriminator D. The
quality of the *function class* D is drawn from controls everything. Two concrete failures:
- In high-dim space the density-ratio estimate D learns is inaccurate/unstable; G fails to
  capture multimodal structure.
- (Arjovsky & Bottou 2017) when supports of p_G and q_data are disjoint, there exists a
  perfect D; once found, ∇_x D = 0 a.e., so the generator gradient vanishes — training stops.
=> Need to *restrict* the discriminator's function class. The successful restriction in the
literature: control the **Lipschitz constant** of D.

## Load-bearing ancestors (lineage)
- **Goodfellow et al. 2014** — original GAN, min_G max_D V; optimal D*_G = q/(q+p);
  f* = log q - log p; ∇_x f* = ∇q/q - ∇p/p can be unbounded/incomputable => need regularity
  on the derivative.
- **Arjovsky & Bottou 2017 (towards principled)** — disjoint-support => perfect D, zero
  gradient. Diagnostic motivating finding.
- **Arjovsky, Chintala, Bottou 2017 (WGAN)** — Wasserstein-1 distance via Kantorovich-
  Rubinstein duality requires D to be 1-Lipschitz; enforce by **weight clipping** w←clip(w,-c,c).
  Limitation: clipping pushes weights to ±c => low-rank, "capacity underuse"; slow.
- **Gulrajani et al. 2017 (WGAN-GP)** — replace clipping with **gradient penalty**
  λ E[(‖∇_x̂ D‖_2 - 1)^2] at interpolated points x̂ = εx + (1-ε)x̃. Limitation: only
  regularizes on the support of *current* batch; support drifts as G changes => unstable
  with high LR; also doubles cost (gradient-of-gradient = extra fwd/bwd round).
- **Salimans & Kingma 2016 (weight normalization)** — normalize each row: w̄_i = w_i/‖w_i‖.
  Implicit constraint Σσ_t^2 = d_o => to maximize ‖W̄h‖ the optimizer drives W̄ to rank one
  (one singular value √d_o, rest 0). Same capacity-underuse pitfall. Frobenius norm
  W/‖W‖_F: Σσ_t^2 = 1, same argument.
- **Brock et al. 2016 (orthonormal reg)** — penalty ‖W^T W - I‖_F^2 sets ALL singular values
  to 1 => destroys spectrum, forces use of all dims even useless ones.
- **Yoshida & Miyato 2017 (spectral norm regularization)** — adds a penalty on σ(W) to the
  loss (data-independent, like L2). Provided the power-iteration σ-estimate technique.
  Difference from SN: it *penalizes*, doesn't *set* σ to a value.

## The derivation chain (insight order)
1. Want ‖f‖_Lip ≤ K. f is a composition of linear maps and 1-Lipschitz activations.
2. Lipschitz constant of a map g = sup_h σ(∇g(h)) where σ = spectral norm = largest singular
   value. For a linear layer g(h)=Wh, ∇g = W, so ‖g‖_Lip = σ(W).
3. Composition bound: ‖g_1∘g_2‖_Lip ≤ ‖g_1‖_Lip·‖g_2‖_Lip. With ‖a_l‖_Lip = 1 (ReLU/lReLU),
   ‖f‖_Lip ≤ Π_l σ(W^l).
4. So bound each layer's σ. Hard constraint per layer: divide by σ.
   **W̄_SN = W/σ(W)** => σ(W̄_SN)=1 => ‖f‖_Lip ≤ 1.
5. Cost: σ(W) = top singular value. SVD every step too expensive.
   **Power iteration** with a *persistent* u reused across SGD steps; ONE iteration/update:
     v ← W^T u/‖W^T u‖,  u ← W v/‖W v‖,  σ ≈ u^T W v.
   Justification: W changes little per SGD step, so warm-started u stays near u_1.
6. Why SN beats the others:
   - vs weight clipping / weight norm / Frobenius: those constrain Σσ_t^2 => bias to low rank
     (rank-1 optimum) => discriminator uses few features. SN constrains only σ_1 => spectrum
     free => keeps many features. (Confirmed by singular-value histograms: WC/WN concentrate,
     SN broad.)
   - vs orthonormal: that sets all σ_t=1, destroying spectrum; SN only scales so max=1.
   - vs WGAN-GP: GP regularizes on sample points (support-dependent, drifts, unstable at high
     LR, 2x cost); SN regularizes the operator itself, support-independent, cheap.

## Gradient analysis (appendix-level, must be derived inline)
∂σ(W)/∂W_{ij} = [u_1 v_1^T]_{ij}  (derivative of top singular value).
∂W̄_SN/∂W_{ij} = (1/σ)E_{ij} - (1/σ^2)[u_1v_1^T]_{ij} W = (1/σ)(E_{ij} - [u_1v_1^T]_{ij} W̄_SN).
With δ := (∂V/∂(W̄_SN h))^T,
∂V/∂W = (1/σ)( Ê[δ h^T] - λ u_1 v_1^T ),  λ := Ê[δ^T W̄_SN h].
First term = ordinary unnormalized gradient. Second term = adaptive penalty on the first
singular component; λ>0 when δ and W̄_SN h align => prevents column space collapsing into one
direction => keeps the layer from becoming over-sensitive in one direction.
General form: for W̄ = W/N(W), ∂V/∂W = (1/N)(∇_{W̄}V - trace(∇_{W̄}V^T W̄) ∇_W N). For N=‖W‖_F,
∇_W N = W̄; for N=σ, ∇_W N = u_1 v_1^T.

Reparametrization: W̃ = γ W̄_SN, γ scalar learned — relaxes the 1-Lipschitz constraint, needs
another Lipschitz control (e.g. GP), improves WGAN-GP.

## Conv weights
W ∈ R^{d_out×d_in×h×w} treated as 2-D matrix d_out × (d_in·h·w) for σ.

## Code grounding
PyTorch torch.nn.utils.spectral_norm (code/spectral_norm_pytorch.py): forward_pre_hook recomputes
W̄ each forward: power iteration on persistent buffers u,v under no_grad; sigma=u·(Wv); W=W/sigma.
n_power_iterations=1. Matches Algorithm 1.

## Two flagged equations re-derived (self-check, Codex gate later)
- Power iteration: W=Σσ_t u_t v_t^T; (W^TW)^k u amplifies σ_1^2 mode => u→u_1,v→v_1;
  u^T W v → σ_1. One step/update OK since W slow-moving + warm start. CORRECT.
- SN gradient: chain rule above; signs/factors checked. CORRECT.

## Scaffold (pre-method, generic GAN/Lipschitz)
Bare GAN harness: G (z→x), D (Linear/Conv layers + lReLU), adversarial V, Adam, alternating
d/g updates. ONE empty slot: a hook/transform applied to a layer's weight that enforces a
norm constraint — # TODO. Baselines section gets WGAN clipping, WGAN-GP gradient penalty,
weight normalization described.
