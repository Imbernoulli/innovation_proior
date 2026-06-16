OK, let me think about what's actually breaking. I have the point-sampled volumetric pipeline working: an MLP maps an encoded 3D position to a density and color, I cast one ray per pixel, sample points along it, composite. On the standard benchmark — every camera the same distance from the object, same focal length, same resolution — it's photorealistic. I'm happy. But the instant I let the cameras live at different distances, or train on images at a mix of resolutions, it falls apart. Close-up renderings come out mushy and blurred; distant renderings come out aliased, with crawling jagged edges. And there's no single setting that fixes both. Push the positional-encoding degree L down and the close-ups get worse; push it up and the distant views shimmer harder. So before I reach for any machinery, I need to understand *why* one model can't serve two scales.

Let me stare at a single pixel. A pixel is not a point. It's a little square on the sensor, and the light it records is the integral of all the radiance arriving through the solid angle its footprint subtends — a thin cone fanning out from the camera center through the pixel's area into the world. The true pixel color is an *average* over that cone. But my pipeline casts one infinitesimally thin ray straight through the pixel center and samples *points* on it. A point has no width. So the feature I hand the MLP — the positional encoding of r(t_k) — knows the *location* of the sample but nothing about how big a chunk of the world this pixel is actually looking at.

Now run the thought experiment that makes the failure precise. Two cameras look at the exact same surface point P. One is up close, so its pixel footprint at P is tiny — it sees a square millimeter. The other is far away (or it's the same image box-downsampled), so its pixel footprint at P is huge — it integrates over a square centimeter. Both cast a ray through P, both sample the point P, both compute γ(P). Identical input. Identical MLP output. But the two pixels *should* have different colors — one is a sharp detail, the other is the average of a whole neighborhood. The representation is blind to scale. During training, the near and far images are handing the MLP contradictory targets for the same input, and the best it can do is average them: blur. That's the close-up blur and it's not a tuning problem, it's an *ambiguity* baked into point sampling.

And the aliasing is the same coin's other face. The positional encoding carries frequencies up to 2^{L−1}. For a distant pixel whose footprint is large, those high frequencies are way above the rate at which the pixel can sample the scene without folding — classic undersampling, the high frequencies alias down into spurious low-frequency garbage, the shimmer and the jaggies. The encoding offers the network all frequencies up to L regardless of what the pixel can actually resolve. So a single L is wrong: it's too high for far pixels (alias) and, if I lower it, too low for near pixels (blur). The point sample and its full-bandwidth encoding are the disease.

So what would a cure look like? The pixel integrates over a cone; I should *featurize the cone*, not a point on its axis. The feature I feed the MLP should depend on the *region of space the pixel sees over an interval*, including its size and shape — so that a near pixel and a far pixel looking at P get *different* features and can produce different colors, and so that when the footprint is large the high frequencies are automatically suppressed rather than aliased. That's the whole game: make the input feature scale-aware.

Graphics already solved aliasing, twice, and the two solutions tell me which road to take. Supersampling: cast many rays per pixel, raise the sampling rate toward Nyquist, average. It works, but cost is linear in the number of rays, and each of my rays already costs hundreds of MLP evaluations — so a 16× supersample is a 16× tax on an already-expensive renderer. Dead on arrival for training. The other solution is prefiltering — the mipmap: precompute the texture at a pyramid of downsampled scales, and at render time look up the level whose footprint matches the pixel's projected footprint. Equivalently, trace a cone per pixel and read the scene at the scale the cone's footprint dictates. Prefiltering moves the anti-aliasing cost to a one-time precomputation, which is exactly the efficiency I want.

But I can't literally precompute a mipmap. A mipmap presupposes I *have* the scene — I'd build the pyramid once and read it many times. In my setting the scene is the unknown; I'm recovering it from images by optimization, and the geometry only emerges during training. So I can't prefilter ahead of time. What I can do is take the *spirit* — trace a cone per pixel, and make the representation answer queries at the scale the cone sets — and have the network *learn* the prefiltered, multiscale representation during optimization. And one more thing the mipmap gets wrong for me: a mipmap is discrete, a fixed ladder of scales. My cameras can sit at any distance, so I need a *continuous* notion of scale, not a handful of pyramid levels.

The replacement has to follow that geometry all the way through: cast a cone from each pixel instead of a ray; chop the cone into a sequence of conical frustums (a cone sliced perpendicular to its axis between t_0 and t_1); and instead of encoding a single point, build a feature that represents the *volume of each frustum* — its position, size, and shape — and hand that to the same MLP, render the same way. I need to pin down the cone's geometry, figure out how to featurize a frustum's volume in a way the MLP can use, and make it cheap and closed-form, because if featurizing a frustum costs anything like supersampling I've gained nothing.

The cone first. Apex at the camera center o, axis along the ray direction d through the pixel center. At the image plane (a unit step along d) the cone has some radius — call it ṙ, the cross-sectional radius per unit distance. What should ṙ be? It should make the cone's cross-section at the image plane match the pixel's footprint there. The pixel is a little square; if I'm going to approximate its footprint with a circular cone cross-section, I want their *spreads* to agree — same variance in the image plane. A pixel of width Δ (in world units at the image plane) — modeled as a uniform square — has variance Δ²/12 per axis. A disk of radius ṙ, taken as a uniform distribution on the disk, has variance ṙ²/4 per axis. Match them: ṙ²/4 = Δ²/12, so ṙ = Δ·2/√12. Good — that's the cone radius: the pixel width scaled by 2/√12 so the disk's variance equals the pixel's footprint variance. The set of points inside the frustum between t_0 and t_1 is the points whose projection onto the axis lies in (t_0, t_1) and whose angle off the axis is within the cone — a clean indicator function, but I won't need its exact form, just its first two moments.

Now the hard part: the feature. What I'd *like* is the expected positional encoding over the frustum — the average of γ(x) for x ranging uniformly over the frustum's volume:

  γ*(frustum) = [ ∫ γ(x) 1{x ∈ frustum} dx ] / [ ∫ 1{x ∈ frustum} dx ].

That's exactly the right object: it's γ averaged over what the pixel sees, so it depends on the size and shape of the region, and the averaging will naturally damp frequencies that oscillate across the frustum. But the numerator has no closed form — γ is a stack of sinusoids of x and the frustum is an awkward solid; I'm not going to integrate sin(B x) over a cone slice analytically. If I had to estimate it by sampling points in the frustum I'm back to supersampling. So this exact expectation is a wall.

Back up. The expectation only needs to be *approximately* right, and what makes it intractable is the frustum's shape. So approximate the frustum's uniform distribution by a *multivariate Gaussian* with the same mean and covariance. A Gaussian is the maximum-entropy match to those two moments, it's the natural "blob" of the right position and spread, and — this is the payoff — expectations of sines and cosines under a Gaussian have clean closed forms. If I can get the mean μ and covariance Σ of a point uniformly distributed in the frustum, I can replace "average γ over the frustum" with "average γ under N(μ, Σ)," which I *can* do in closed form.

So I need the moments of the uniform distribution over a conical frustum. By symmetry the frustum is rotationally symmetric about the axis, so the Gaussian is characterized by just three numbers: the mean distance along the ray μ_t, the variance along the ray σ_t², and the variance perpendicular to the ray σ_r². Let me actually derive them rather than guess. Put the cone axis along z and parameterize a point by (r, t, θ) with (x,y,z) = (r t cosθ, r t sinθ, t), so at distance t the disk has radius ṙt via 0 ≤ r ≤ ṙ; θ ∈ [0,2π); t from t_0 to t_1.

First the volume element. The Jacobian of (x,y,z) with respect to (r, t, θ): differentiate. ∂(x,y,z)/∂r = (t cosθ, t sinθ, 0); ∂/∂t = (r cosθ, r sinθ, 1); ∂/∂θ = (−r t sinθ, r t cosθ, 0). The determinant of those three rows works out to r t² (the z-component of the t-row, the "1", multiplies the 2×2 of the r- and θ-rows in x,y, giving t·(r t)(cos²+sin²) = r t²). So dx dy dz = r t² dr dt dθ.

Volume of the frustum, the normalizer: V = ∫₀^{2π} ∫_{t_0}^{t_1} ∫_0^{ṙ} r t² dr dt dθ = (ṙ²/2)·((t_1³−t_0³)/3)·2π = π ṙ² (t_1³ − t_0³)/3. The density of the uniform distribution is r t² / V.

Mean along the ray, E[t] = (1/V) ∫ t · r t² = (1/V) π ṙ² (t_1⁴−t_0⁴)/4 = 3(t_1⁴−t_0⁴) / [4(t_1³−t_0³)]. By symmetry E[x] = E[y] = 0.

Second moment along the ray, E[t²] = (1/V) ∫ t² · r t² = (1/V) π ṙ² (t_1⁵−t_0⁵)/5 = 3(t_1⁵−t_0⁵)/[5(t_1³−t_0³)]. So σ_t² = E[t²] − E[t]² = 3(t_1⁵−t_0⁵)/[5(t_1³−t_0³)] − μ_t².

Second moment perpendicular, E[x²] = (1/V) ∫ (r t cosθ)² r t² = (1/V) ∫_θ cos²θ dθ ∫_t t⁴ ∫_r r³ = (1/V)·π·(t_1⁵−t_0⁵)/5·(ṙ⁴/4). Plug V: this is ṙ²/4 · 3(t_1⁵−t_0⁵)/[5(t_1³−t_0³)]. Since E[x]=0, σ_r² = E[x²] = ṙ² · 3(t_1⁵−t_0⁵)/[20(t_1³−t_0³)].

Now a numerical landmine: these are ratios of differences of fifth and cubed powers of t_0 and t_1. During training t_0 and t_1 are often very close (samples bunch up near surfaces), and subtracting nearly equal large powers loses all precision — I'll get 0/0 or NaN and training dies. So reparameterize by the midpoint t_μ = (t_0 + t_1)/2 and half-width t_δ = (t_1 − t_0)/2, and expand each ratio as a leading term plus higher-order corrections in t_δ, which stays accurate as t_δ → 0. Carrying the algebra through:

  μ_t = t_μ + 2 t_μ t_δ² / (3 t_μ² + t_δ²),
  σ_t² = t_δ²/3 − 4 t_δ⁴ (12 t_μ² − t_δ²) / [15 (3 t_μ² + t_δ²)²],
  σ_r² = ṙ² ( t_μ²/4 + 5 t_δ²/12 − 4 t_δ⁴ / [15 (3 t_μ² + t_δ²)] ).

Sanity check σ_t² for a thin slab: as t_δ → 0 the correction vanishes and σ_t² → t_δ²/3, which is exactly the variance of a uniform distribution on an interval of half-width t_δ. Good. And σ_r² → ṙ² t_μ²/4 at small t_δ, the variance of a uniform disk of radius ṙ t_μ — the cone's cross-section at distance t_μ. Both limits are what they should be.

Lift this from the (along-ray, perpendicular) frame into world coordinates. Mean: μ = o + μ_t d. The t-coordinate variance becomes σ_t² d dᵀ because moving t by one moves the world point by d; the perpendicular spread lives in the plane orthogonal to d, so it uses the projector I − d dᵀ/‖d‖². Thus Σ = σ_t² (d dᵀ) + σ_r² (I − d dᵀ/‖d‖²). That's my Gaussian N(μ, Σ) approximating the frustum.

Now the feature: the expected positional encoding under this Gaussian. Rewrite γ in Fourier-feature form. Stack the per-axis frequencies into a matrix P whose rows are the standard basis times powers of two — so P x lays out, for each axis, x scaled by 1, 2, 4, …, 2^{L−1}. Then γ(x) = [sin(P x); cos(P x)]. This linearization is what makes the Gaussian expectation tractable: P x is a linear function of x, so under x ∼ N(μ, Σ), the vector P x is Gaussian with mean μ_γ = P μ and covariance Σ_γ = P Σ Pᵀ (the covariance of a linear map A x is A Σ Aᵀ).

So I need E[sin(v)] and E[cos(v)] for a scalar Gaussian v ∼ N(m, s²). These are textbook (the characteristic function of a Gaussian): E[exp(i v)] = exp(i m − s²/2), and taking imaginary/real parts, E[sin v] = sin(m) exp(−s²/2), E[cos v] = cos(m) exp(−s²/2). There it is — the expected sinusoid is just the sinusoid of the *mean* multiplied by exp(−½ variance). The mean tells you the phase; the variance tells you how much to shrink it.

Read what that exp(−½ s²) does. Each entry of P x is a particular frequency of a particular position coordinate; its variance under the Gaussian is the corresponding diagonal entry of Σ_γ. If that frequency's period is *large* compared to the frustum's spread in that direction, its variance s² is tiny, exp(−½ s²) ≈ 1, the feature passes through essentially unchanged. If the period is *small* compared to the spread — the sinusoid oscillates many times across the frustum, which is exactly the frequency that would alias — then s² is large, exp(−½ s²) → 0, and the feature is smoothly driven to zero. So the encoding *automatically* keeps the frequencies that are stable over the region and *erases* the ones that oscillate across it. This is anti-aliasing falling straight out of the math: it's a soft, per-frequency low-pass whose cutoff is set by the frustum's size. Call it an integrated positional encoding — the positional encoding integrated over the volume the pixel sees.

Since each encoding dimension depends only on its own marginal Gaussian, I only need the *diagonal* of Σ_γ, never the full matrix (which would be huge and pointless to form). And the diagonal of Σ_γ = P Σ Pᵀ for this block-of-powers P is just the diagonal of Σ replicated and scaled by 4^l per frequency level l: diag(Σ_γ) = [diag(Σ), 4 diag(Σ), …, 4^{L−1} diag(Σ)] (because the frequency 2^l scales the variance by (2^l)² = 4^l). And diag(Σ) itself, from the world-frame covariance, is σ_t² (d∘d) + σ_r² (1 − d∘d/‖d‖²), elementwise. So:

  IPE(μ, Σ) = [ sin(μ_γ) ∘ exp(−½ diag(Σ_γ)); cos(μ_γ) ∘ exp(−½ diag(Σ_γ)) ].

Computed this way it costs essentially the same as a plain positional encoding — no integral, no extra samples. The wall (no closed form for the exact frustum integral) is gone, paid for with a Gaussian approximation that is exact in its first two moments and whose error only shows up if the frustum's near and far radii differ wildly, which happens only for frustums right at the camera under a very wide field of view — irrelevant in practice.

There's a lovely free consequence: L stops being a tuning knob. In the point-sampled encoding, L is a hard truncation — keep all frequencies below 2^L, drop the rest, and you have to hand-pick where to cut. Here the cut is *soft* and *automatic*: exp(−½ s²) attenuates each frequency by how much it varies over the frustum, and that variation is dictated entirely by camera geometry. So I can set L to something absurdly large — large enough that the highest frequency's feature is already below numerical epsilon for any frustum I'll see — and never touch it again. A point sample is just the degenerate case Σ = 0, where exp(−½·0) = 1 for every frequency and IPE collapses back to plain PE. So this genuinely generalizes the old encoding; the old one is the zero-width limit.

Now reconsider the two-network hierarchical-sampling machinery. Why did the point-sampled pipeline need a *separate* coarse MLP and fine MLP? Because each MLP, fed point encodings, could only ever learn the scene at a *single* scale — it had no way to know which scale a query belonged to. The coarse network operates on a coarse set of samples, the fine network on a denser set; tangling them into one network confuses it. But now my features *carry scale explicitly* — a frustum's IPE encodes how big it is. So one MLP can answer queries at any scale; it can be queried coarsely and finely and it knows the difference from the feature itself. I can collapse the two networks into a single multiscale MLP, query it twice in a hierarchy. That halves the parameters and, since I no longer keep two distinct architectures or sort a union of coarse-plus-fine samples, it's actually a touch faster and simpler.

Mechanically: cast a cone, pick n+1 distances {t_k} along it, form the n frustums between adjacent pairs, IPE-featurize each, run the single MLP to get (σ_k, c_k), composite with the same alpha rule as before. For the hierarchy: draw a coarse set of distances by stratified sampling, render, read off the compositing weights w_k = T_k(1 − exp(−σ_k δ_k)) as a piecewise-constant PDF, draw a fine set from that PDF by inverse-transform sampling, render again with the *same* MLP. I'll use the same number of samples at each level so the total MLP evaluations match the old budget for a fair comparison.

Two small fixes on the resampling, both to avoid a known failure of greedily following the coarse weights. First, the coarse PDF can be spiky and can have empty regions with literally zero weight, so the fine samples would never probe there and the model could never discover content it initially missed. So before resampling I widen and smooth the weights: a 2-tap max filter followed by a 2-tap blur (a blurpool) to get a smooth upper envelope over w, then add a small constant α to that envelope before renormalizing — w'_k = ½(max(w_{k−1}, w_k) + max(w_k, w_{k+1})) + α — so every region keeps a floor of probability and some fine samples always land even in nominally empty space. I'll set α small, around 0.01.

Second, with one shared MLP I now have *two* reconstruction losses — coarse and fine — that have to be balanced, where the old two-network setup never needed balancing because the losses hit different networks. So the objective is the squared pixel error on both passes with a multiplier on the coarse term:

  min_Θ Σ_rays [ λ · ‖C_true − C(coarse; Θ)‖² + ‖C_true − C(fine; Θ)‖² ],

with λ small (≈ 0.1) so the fine reconstruction dominates while the coarse pass is still trained enough to place good samples. Net hyperparameter count is a wash: I add λ and the resampling floor α, and I delete L (no longer tuned) and the separate coarse/fine sample counts.

A couple of stability choices for the optimization, none of them central but worth getting right. For density I use a shifted softplus log(1 + exp(x − 1)) instead of a hard ReLU: ReLU can drive the whole density field negative, where its gradient is zero everywhere and training never recovers; softplus is smooth and the −1 shift biases initial densities small, which speeds early training because dense content doesn't immediately shadow everything behind it. For color I use a sigmoid widened slightly past [0,1] — (1+2ε)/(1+e^{−x}) − ε with ε≈0.001 — so the network never has to push activations into the dead tails of the sigmoid to hit pure black or white, which would zero the gradient. And when training on the multiscale data I scale each pixel's loss by its footprint area (the ¼-resolution pixels by 16, etc.) so the handful of low-resolution pixels carry weight comparable to the many high-resolution ones.

Let me write it down, mirroring the standard volumetric harness but with cones and IPE.

```python
import torch

def lift_gaussian(d, t_mean, t_var, r_var):
    # frustum-frame moments -> world-space mean and (diagonal) covariance
    mean = t_mean[..., None] * d[..., None, :]                     # mu = o + mu_t * d, o added by caller
    d_sq = d**2
    d_norm_sq = d_sq.sum(-1, keepdim=True).clamp_min(1e-10)
    # diag(Sigma) = sigma_t^2 (d . d) + sigma_r^2 (1 - d.d/||d||^2)
    diag = (t_var[..., None] * d_sq[..., None, :]
            + r_var[..., None] * (1.0 - d_sq[..., None, :] / d_norm_sq[..., None, :]))
    return mean, diag

def conical_frustum_to_gaussian(d, t0, t1, base_radius):
    # numerically stable moments via midpoint/half-width
    mu = (t0 + t1) / 2.0          # t_mu
    hw = (t1 - t0) / 2.0          # t_delta
    den = 3.0 * mu**2 + hw**2
    t_mean = mu + 2.0 * mu * hw**2 / den
    t_var  = hw**2 / 3.0 - (4.0/15.0) * hw**4 * (12.0*mu**2 - hw**2) / den**2
    r_var  = base_radius**2 * (mu**2/4.0 + 5.0*hw**2/12.0 - (4.0/15.0)*hw**4/den)
    return t_mean, t_var, r_var

def integrated_pos_enc(mean, diag, L):
    # P scales each axis by 2^l; means scale linearly, variances by 4^l
    scales = 2.0 ** torch.arange(L, device=mean.device)           # [L]
    mu_g  = (mean[..., None, :] * scales[:, None]).reshape(*mean.shape[:-1], -1)
    var_g = (diag[..., None, :] * (scales**2)[:, None]).reshape(*diag.shape[:-1], -1)
    damp = torch.exp(-0.5 * var_g)                                # soft per-frequency low-pass
    return torch.cat([torch.sin(mu_g) * damp, torch.cos(mu_g) * damp], dim=-1)   # IPE feature

def cast_cone(origins, directions, base_radius, t_vals):
    # t_vals: n+1 sorted distances -> n frustums; one IPE feature per frustum
    t0, t1 = t_vals[..., :-1], t_vals[..., 1:]
    t_mean, t_var, r_var = conical_frustum_to_gaussian(directions, t0, t1, base_radius)
    mean, diag = lift_gaussian(directions, t_mean, t_var, r_var)
    mean = mean + origins[..., None, :]                           # mu = o + mu_t d
    return mean, diag

def volume_render(sigma, rgb, t_vals):
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1)[..., :-1]
    w = alpha * T
    color = (w[..., None] * rgb).sum(-2)
    return color, w

def resample(t_vals, w):                  # widen + floor the coarse PDF, then inverse-transform sample
    w_pad = torch.cat([w[..., :1], w, w[..., -1:]], -1)
    w_max = torch.maximum(w_pad[..., :-1], w_pad[..., 1:])         # 2-tap max
    w_blur = 0.5 * (w_max[..., :-1] + w_max[..., 1:])              # 2-tap blur (blurpool)
    weights = w_blur + 0.01                                       # alpha floor so empty space is probed
    return inverse_transform_sample(t_vals, weights)

def render_pixel(mlp, origins, directions, base_radius, n=128, near=2.0, far=6.0):
    # single shared MLP, queried coarse then fine
    t_c = stratified(near, far, n + 1, origins.shape[:-1])
    mean, diag = cast_cone(origins, directions, base_radius, t_c)
    sigma, rgb = mlp(integrated_pos_enc(mean, diag, L=16), directions)
    C_coarse, w = volume_render(sigma, rgb, t_c)

    t_f = resample(t_c, w)
    mean, diag = cast_cone(origins, directions, base_radius, t_f)
    sigma, rgb = mlp(integrated_pos_enc(mean, diag, L=16), directions)
    C_fine, _ = volume_render(sigma, rgb, t_f)
    return C_coarse, C_fine

def loss_fn(C_coarse, C_fine, C_true, lam=0.1):
    return lam * ((C_coarse - C_true)**2).mean() + ((C_fine - C_true)**2).mean()
```

The causal chain, start to finish: a pixel integrates radiance over a cone, but point sampling features only its axis, so near and far cameras seeing the same point get identical inputs — that ambiguity is the close-up blur and the full-bandwidth encoding is the distant aliasing. Cure it by featurizing the *region* the pixel sees: cast a cone, cut it into frustums, and feed the MLP the expected positional encoding over each frustum. The exact frustum integral has no closed form, so approximate the frustum by a Gaussian matched to its first two moments (derived analytically and reparameterized by midpoint/half-width for numerical stability); under a Gaussian the expected sinusoid is the sinusoid of the mean times exp(−½ variance), which is a soft per-frequency low-pass that keeps frequencies stable across the frustum and erases the ones that would alias — anti-aliasing for free, and L ceases to be a hyperparameter. Because the feature now encodes scale, one MLP suffices for the whole hierarchy: a single multiscale network, queried coarse-then-fine with a smoothed, floored resampling PDF and a small coarse-loss multiplier — half the parameters, a touch faster, and correct at every scale.
