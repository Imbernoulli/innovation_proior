# Mip-NeRF: a multiscale, anti-aliased neural radiance field

## Problem

A point-sampled neural radiance field casts one infinitesimally thin ray per pixel and
encodes single points along it. A point sample carries no information about the pixel's
footprint — the size of the region of space the pixel integrates over. So two cameras
observing the same surface point at different distances produce the same input feature and
the model cannot serve both: it blurs close-up views and aliases distant ones, and the
positional-encoding degree L has no setting that is right at all scales. Mip-NeRF makes the
representation scale-aware so a single recovered model renders correctly at any resolution,
without brute-force supersampling.

## Key idea

Cast a **cone** from each pixel instead of a ray, divide it into **conical frustums**, and
feed the MLP an **integrated positional encoding (IPE)** of each frustum — the *expected*
positional encoding over the volume the pixel sees, rather than the encoding of a single
point. The exact expectation over a frustum has no closed form, so the frustum's uniform
distribution is approximated by a multivariate Gaussian matched to its first two moments;
under a Gaussian the expected sinusoid is closed-form, which yields a soft per-frequency
low-pass that automatically anti-aliases. Because the feature now encodes scale, the two
separate "coarse" and "fine" networks collapse into one multiscale MLP.

## Final method

**Cone geometry.** Apex at camera center o, axis along d through the pixel center, radius at
the image plane ṙ = (pixel width) · 2/√12, chosen so the cone cross-section's variance
matches the pixel footprint's variance.

**Conical-frustum Gaussian.** For a frustum between t₀ and t₁, with midpoint t_μ = (t₀+t₁)/2
and half-width t_δ = (t₁−t₀)/2 (used for numerical stability), the moments of a point drawn
uniformly from the frustum are
- μ_t = t_μ + 2 t_μ t_δ² / (3 t_μ² + t_δ²),
- σ_t² = t_δ²/3 − 4 t_δ⁴ (12 t_μ² − t_δ²) / [15 (3 t_μ² + t_δ²)²],
- σ_r² = ṙ² ( t_μ²/4 + 5 t_δ²/12 − 4 t_δ⁴ / [15 (3 t_μ² + t_δ²)] ).

Lift to world space: μ = o + μ_t d and Σ = σ_t² (d dᵀ) + σ_r² (I − d dᵀ/‖d‖²), with diagonal
diag(Σ) = σ_t² (d∘d) + σ_r² (1 − d∘d/‖d‖²).

**Integrated positional encoding.** With the Fourier-feature form γ(x) = [sin(Px); cos(Px)]
where P scales each axis by 2⁰,…,2^{L−1}, under x ∼ N(μ, Σ) the lifted mean is μ_γ = Pμ and
the lifted variances are diag(Σ_γ) = [diag(Σ), 4 diag(Σ), …, 4^{L−1} diag(Σ)]. Using
E[sin v] = sin(m) e^{−s²/2}, E[cos v] = cos(m) e^{−s²/2},
IPE(μ, Σ) = [ sin(μ_γ) ∘ exp(−½ diag(Σ_γ)); cos(μ_γ) ∘ exp(−½ diag(Σ_γ)) ].
Frequencies whose period exceeds the frustum's spread pass through (variance ≈ 0); those that
oscillate across it are damped toward zero. Plain PE is the degenerate Σ = 0 case, so L
ceases to need tuning (set it large, e.g. 16, and forget it).

**Single multiscale MLP and rendering.** One MLP maps each frustum's IPE (plus encoded
direction) to (σ, c); composite with the standard alpha rule C = Σ_k T_k(1−exp(−σ_k δ_k)) c_k,
T_k = exp(−Σ_{k'<k} σ_{k'} δ_{k'}). Hierarchical sampling queries the same MLP twice: a
stratified coarse pass, then a fine pass whose distances are inverse-transform-sampled from
the coarse compositing weights after a 2-tap-max + 2-tap-blur (blurpool) widening and an
α = 0.01 floor: w'_k = ½(max(w_{k−1},w_k) + max(w_k,w_{k+1})) + α.

**Objective.** min_Θ Σ_rays [ λ‖C_true − C_coarse‖² + ‖C_true − C_fine‖² ], with λ = 0.1.
Density uses a shifted softplus log(1+exp(x−1)); color a slightly widened sigmoid
(1+2ε)/(1+e^{−x}) − ε with ε = 0.001. For multiscale training, scale each pixel's loss by its
footprint area. Adam, batch 4096, learning rate annealed logarithmically 5·10⁻⁴ → 5·10⁻⁶
over 1M iterations; 128 samples per level. Half the parameters of the two-network baseline,
slightly faster.

## Code

```python
import torch

def conical_frustum_to_gaussian(d, t0, t1, base_radius):
    mu = (t0 + t1) / 2.0
    hw = (t1 - t0) / 2.0
    den = 3.0 * mu**2 + hw**2
    t_mean = mu + 2.0 * mu * hw**2 / den
    t_var  = hw**2 / 3.0 - (4.0/15.0) * hw**4 * (12.0*mu**2 - hw**2) / den**2
    r_var  = base_radius**2 * (mu**2/4.0 + 5.0*hw**2/12.0 - (4.0/15.0)*hw**4/den)
    return t_mean, t_var, r_var

def lift_gaussian(d, t_mean, t_var, r_var, origins):
    mean = origins[..., None, :] + t_mean[..., None] * d[..., None, :]
    d_sq = d**2
    d_norm_sq = d_sq.sum(-1, keepdim=True).clamp_min(1e-10)
    diag = (t_var[..., None] * d_sq[..., None, :]
            + r_var[..., None] * (1.0 - d_sq[..., None, :] / d_norm_sq[..., None, :]))
    return mean, diag                       # mu and diag(Sigma)

def integrated_pos_enc(mean, diag, L=16):
    scales = 2.0 ** torch.arange(L, device=mean.device)
    mu_g  = (mean[..., None, :] * scales[:, None]).reshape(*mean.shape[:-1], -1)
    var_g = (diag[..., None, :] * (scales**2)[:, None]).reshape(*diag.shape[:-1], -1)
    damp  = torch.exp(-0.5 * var_g)
    return torch.cat([torch.sin(mu_g) * damp, torch.cos(mu_g) * damp], dim=-1)

def cast_cone(origins, directions, base_radius, t_vals):
    t0, t1 = t_vals[..., :-1], t_vals[..., 1:]
    t_mean, t_var, r_var = conical_frustum_to_gaussian(directions, t0, t1, base_radius)
    return lift_gaussian(directions, t_mean, t_var, r_var, origins)

def volume_render(sigma, rgb, t_vals):
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1.0 - alpha + 1e-10], -1), -1)[..., :-1]
    w = alpha * T
    return (w[..., None] * rgb).sum(-2), w

def resample(t_vals, w):                    # widen + floor the coarse PDF, then inverse-transform sample
    w_pad = torch.cat([w[..., :1], w, w[..., -1:]], -1)
    w_max = torch.maximum(w_pad[..., :-1], w_pad[..., 1:])
    w_blur = 0.5 * (w_max[..., :-1] + w_max[..., 1:])
    weights = w_blur + 0.01
    return inverse_transform_sample(t_vals, weights)

def render_pixel(mlp, origins, directions, base_radius, n=128, near=2.0, far=6.0):
    t_c = stratified(near, far, n + 1, origins.shape[:-1])
    mean, diag = cast_cone(origins, directions, base_radius, t_c)
    sigma, rgb = mlp(integrated_pos_enc(mean, diag), directions)
    C_coarse, w = volume_render(sigma, rgb, t_c)
    t_f = resample(t_c, w)
    mean, diag = cast_cone(origins, directions, base_radius, t_f)
    sigma, rgb = mlp(integrated_pos_enc(mean, diag), directions)
    C_fine, _ = volume_render(sigma, rgb, t_f)
    return C_coarse, C_fine

def loss_fn(C_coarse, C_fine, C_true, lam=0.1):
    return lam * ((C_coarse - C_true)**2).mean() + ((C_fine - C_true)**2).mean()
```
