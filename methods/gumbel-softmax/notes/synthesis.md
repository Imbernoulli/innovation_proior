# Synthesis — Gumbel-Softmax / Concrete

## Pain point / research question
We want stochastic neural nets with **discrete (categorical) latent variables** trained by SGD.
Objective L(θ) = E_{z~p_θ(z)}[f(z)], need ∇_θ. z is a categorical draw → sampling/argmax is
non-differentiable, blocks backprop. Want a low-variance, single-sample gradient estimator.

## Load-bearing ancestors (cite by author/year, elaborate)
1. **Reparameterization trick / VAE** (Kingma & Welling 2013; Rezende et al. 2014). For
   reparameterizable continuous dists, write z = g(θ, ε), ε independent of θ. Then
   ∂/∂θ E_z[f(z)] = E_ε[∂f/∂g · ∂g/∂θ]. Gaussian: z = μ + σ·ε, ε~N(0,1). Low variance, exact.
   FAILS for categorical: a categorical sample is one_hot(argmax(...)); argmax has zero gradient
   a.e., so there's no continuous g(θ,ε) whose output is a discrete one-hot AND is differentiable.
2. **Score-function / REINFORCE / likelihood-ratio** (Williams 1992; Glynn; Fu). Identity
   ∇_θ p_θ(z) = p_θ(z) ∇_θ log p_θ(z) ⇒ ∇_θ E[f] = E_z[f(z) ∇_θ log p_θ(z)]. Unbiased, only needs
   p_θ continuous in θ, no backprop through f or z. BUT high variance — variance scales ~linearly
   with sample dimensionality; needs control variates/baselines (NVIL, DARN, MuProp, VIMCO) to be
   usable. Still noisy for categoricals.
3. **Gumbel-Max trick** (Gumbel 1954; Maddison/Tarlow/Minka 2014). Exact categorical sampling:
   z = one_hot(argmax_i [log π_i + g_i]), g_i ~ Gumbel(0,1), g_i = -log(-log u_i), u_i~U(0,1).
4. **Biased path-derivative / Straight-Through** (Bengio, Léonard, Courville 2013). For
   non-reparameterizable z, approximate ∇_θ z ≈ ∇_θ m(θ) for a differentiable proxy m. ST for
   Bernoulli: ∇_θ z ≈ 1 (pass gradient through as if sample = mean). Biased; fwd/bwd mismatch
   raises variance.
5. **Sibling / parallel prior-art context (NOT the target): Concrete distribution**
   (Maddison, Mnih, Whye Teh 2016) — independently discovered the same relaxation. Cite as
   concurrent parallel work, present the relaxation as the object both arrive at.
6. **Semi-supervised VAE** (Kingma et al. 2014) — joint (Gaussian style z, categorical class y).
   Unlabeled ELBO requires marginalizing y over all k classes → O(D + k(I+G)) per step. With a
   differentiable categorical sample, single-sample backprop → O(D+I+G).

## Gumbel-Max correctness (derive inline)
g_i~Gumbel(0,1), CDF F(g)=exp(-e^{-g}). x_i = log π_i + g_i is Gumbel with location log π_i:
P(x_i ≤ t) = exp(-π_i e^{-t}); pdf f_i(t)=π_i e^{-t} exp(-π_i e^{-t}).
P(argmax = k) = ∫ f_k(t) Π_{i≠k} P(x_i ≤ t) dt
 = ∫ π_k e^{-t} exp(-(Σ_i π_i) e^{-t}) dt. Sub s=e^{-t}: = ∫_0^∞ π_k exp(-(Σπ_i)s) ds = π_k/Σπ_i.
So argmax draws EXACTLY from Categorical(π). (Works with unnormalized π too — normalizer cancels.)
Why Gumbel and not another noise: max-stability — max of Gumbels is Gumbel; the additive
log π_i + g_i pushes the categorical structure into an argmax-of-perturbed-logits.

## The relaxation (derive inline)
argmax still non-differentiable. Replace argmax by softmax with temperature τ:
y_i = exp((log π_i + g_i)/τ) / Σ_j exp((log π_j + g_j)/τ).  y ∈ Δ^{k-1}, smooth for τ>0.
∂y/∂π well-defined ⇒ reparameterized: noise g is independent of π, path-derivative estimator.

## Temperature behavior / bias-variance (derive inline)
- τ→0: softmax → argmax, y → one-hot, recovers exact categorical samples. But gradients ∂y/∂π
  blow up (the softmax becomes a step; Jacobian entries explode) → high gradient variance.
- τ large: y → uniform (1/k), smooth, low-variance gradients, but biased (samples far from the
  true categorical; expectation drifts to uniform).
- Tradeoff: low τ = low bias / high variance; high τ = high bias / low variance.
- Practice: anneal τ from high to a small positive value, e.g. τ = max(0.5, exp(-r t)).
  Optionally learn τ → interpretable as entropy regularization.

## Straight-Through Gumbel-Softmax
When a hard discrete sample is required (RL action, quantization): forward z = one_hot(argmax y),
backward use ∂z/∂θ ≈ ∂y/∂θ. Implementation: ret = y_hard - y_soft.detach() + y_soft (value is
one-hot, gradient is the soft Jacobian). Lower variance than plain ST because y is a true
differentiable proxy of z (no sample-independent-mean fwd/bwd discrepancy).

## Appendix density (derive inline in reasoning)
p_{π,τ}(y) = Γ(k) τ^{k-1} (Σ_i π_i / y_i^τ)^{-k} Π_i (π_i / y_i^{τ+1}).
Derivation: softmax not invertible (loses 1 dof) → center by subtracting (x_k+g_k); marginalize
g_k out of centered Gumbel via v=e^{-g_k} substitution giving
p(u_{1:k-1}) = Γ(k) Π exp(x_i-u_i) (Σ exp(x_i-u_i))^{-k}; then change of variables y=h(u),
h^{-1}_i = τ(log y_i - log y_k), Jacobian det = τ^{k-1} Π_{j=1}^k y_j^{-1}
(via det(I+uv^T)=1+u^Tv). Compose → density above.

## Canonical code (1.4)
torch.nn.functional.gumbel_softmax: gumbels=-log(-log... via .exponential_().log()); add logits,
/τ; softmax; hard branch = straight-through. Snapshot in code/. Scaffold = generic VAE with a
stochastic categorical latent node (sample_latent stub).

## Design decisions → why
- one-hot encoding of z on simplex corners: lets relaxation live in same space Δ^{k-1}; defines E[z]=π.
- use log π_i (logits) inside argmax: required for Gumbel-Max correctness (location = log π).
- softmax as argmax relaxation (not e.g. sparsemax): differentiable everywhere, exact argmax as τ→0.
- temperature τ shared scalar, annealed: cheapest knob to trade bias/variance.
- ST variant: only when hard samples mandatory; otherwise soft is lower-variance.
