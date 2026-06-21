I have a point-sampled neural radiance field that works beautifully on the standard benchmark — every camera the same distance from the object, same focal length, same resolution — but collapses the moment the cameras observe scene content at different scales. Close-up renderings come out mushy; distant or downsampled renderings come out aliased, full of crawling jagged edges and shimmer. The root cause is geometric, not a tuning failure. A pixel is not a point: it is a little square on the sensor, and the radiance it records is the integral over the thin cone of directions its footprint subtends. The true pixel color is an *average* over that cone. But the pipeline casts one infinitesimally thin ray through the pixel center and encodes *points* $r(t_k) = o + t_k d$ along it; a point has no width, so the feature $\gamma(r(t_k))$ tells the MLP *where* the sample is but nothing about *how big a chunk of the world* this pixel is looking at. Run the thought experiment that makes this precise: two cameras image the same surface point $P$, one up close (footprint a square millimeter), one far away (footprint a square centimeter). Both cast a ray through $P$, both sample $P$, both compute the identical $\gamma(P)$ and get the identical MLP output — yet the two pixels *should* differ, one a sharp detail and the other a neighborhood average. During training the near and far images hand the network contradictory targets for the same input, and the best it can do is average them. That is the close-up blur, and it is an ambiguity baked into point sampling. The aliasing is the same coin's other face: the positional encoding carries frequencies up to $2^{L-1}$ regardless of footprint, and for a distant pixel those frequencies sit far above the rate at which it can sample the scene, so they fold down into spurious low-frequency garbage. No single encoding degree $L$ is right — too high for far pixels (alias), too low for near pixels (blur) if you lower it.

Graphics already solved aliasing twice, and the two cures decide the road. Supersampling casts many rays per pixel to raise the sampling rate toward Nyquist; it works but its cost is linear in the ray count, and each of my rays already costs hundreds of MLP evaluations, so a $16\times$ supersample is a $16\times$ tax — dead on arrival for training. Prefiltering instead lowpasses the content so the Nyquist rate needed drops; the mipmap precomputes the texture at a pyramid of scales and reads the level matching the pixel's footprint, equivalently tracing a *cone* per pixel and reading the scene at the scale the cone dictates. Prefiltering moves the cost to a precomputation, which is the efficiency I want — but I cannot literally precompute a mipmap, because in inverse rendering the scene is the *unknown*; the geometry only emerges during optimization. And a mipmap is discrete, a fixed ladder of scales, whereas my cameras can sit at any distance. So I take the spirit — trace a cone, answer queries at the scale the cone sets — and have the network *learn* the prefiltered, continuous, multiscale representation during optimization.

I propose Mip-NeRF. The single design move is to stop featurizing a *point* on the ray's axis and instead featurize the *region* the pixel sees over an interval, including its size and shape, so that a near and a far camera looking at $P$ receive *different* features and high frequencies are automatically suppressed rather than aliased. Concretely: cast a cone with apex at the camera center $o$ and axis $d$ through the pixel center, with cross-sectional radius per unit distance $\dot r$ chosen so the cone's footprint at the image plane matches the pixel's. Modeling the square pixel of width $\Delta$ as uniform gives per-axis variance $\Delta^2/12$, and a uniform disk of radius $\dot r$ has per-axis variance $\dot r^2/4$; matching them, $\dot r^2/4 = \Delta^2/12$, so $\dot r = \Delta\,\tfrac{2}{\sqrt{12}}$. Slice the cone perpendicular to its axis into conical frustums between successive sample distances, and for each frustum build the feature I actually want: the *expected* positional encoding over the frustum's volume,
$$\gamma^*(\text{frustum}) = \frac{\int \gamma(x)\,\mathbf{1}\{x \in \text{frustum}\}\,dx}{\int \mathbf{1}\{x \in \text{frustum}\}\,dx}.$$
This is exactly $\gamma$ averaged over what the pixel sees, so it depends on the region's size and shape, and the averaging naturally damps frequencies that oscillate across the frustum. The trouble is the numerator has no closed form — $\gamma$ is a stack of sinusoids and the frustum is an awkward solid — and estimating it by sampling points inside the frustum would put me right back at supersampling.

The resolution is to keep the moments and drop the shape. Approximate the frustum's uniform distribution by a multivariate Gaussian with the *same mean and covariance*: the Gaussian is the maximum-entropy match to those two moments, and expectations of sinusoids under a Gaussian are closed-form. I derive the moments directly. Parameterizing a point as $(x,y,z) = (r t\cos\theta,\, r t\sin\theta,\, t)$ with $0 \le r \le \dot r$, $\theta \in [0,2\pi)$, $t \in [t_0,t_1]$, the Jacobian gives volume element $r t^2\,dr\,dt\,d\theta$, so the normalizer is $V = \pi\dot r^2(t_1^3 - t_0^3)/3$. Integrating yields the along-ray mean $\mu_t = 3(t_1^4 - t_0^4)/[4(t_1^3 - t_0^3)]$, the along-ray variance $\sigma_t^2 = 3(t_1^5 - t_0^5)/[5(t_1^3 - t_0^3)] - \mu_t^2$, and the perpendicular variance $\sigma_r^2 = \dot r^2\,3(t_1^5 - t_0^5)/[20(t_1^3 - t_0^3)]$, with $\mathbb{E}[x] = \mathbb{E}[y] = 0$ by symmetry. Those raw expressions are a numerical landmine: during training $t_0$ and $t_1$ bunch tightly near surfaces, and subtracting nearly equal fifth and cubed powers loses all precision and produces NaNs. So I reparameterize by the midpoint $t_\mu = (t_0 + t_1)/2$ and half-width $t_\delta = (t_1 - t_0)/2$ and expand as a leading term plus corrections in $t_\delta$, which stays accurate as $t_\delta \to 0$:
$$\mu_t = t_\mu + \frac{2 t_\mu t_\delta^2}{3 t_\mu^2 + t_\delta^2}, \quad \sigma_t^2 = \frac{t_\delta^2}{3} - \frac{4 t_\delta^4(12 t_\mu^2 - t_\delta^2)}{15(3 t_\mu^2 + t_\delta^2)^2}, \quad \sigma_r^2 = \dot r^2\!\left(\frac{t_\mu^2}{4} + \frac{5 t_\delta^2}{12} - \frac{4 t_\delta^4}{15(3 t_\mu^2 + t_\delta^2)}\right).$$
The limits check out: as $t_\delta \to 0$, $\sigma_t^2 \to t_\delta^2/3$ (variance of a uniform interval of half-width $t_\delta$) and $\sigma_r^2 \to \dot r^2 t_\mu^2/4$ (variance of a uniform disk of radius $\dot r t_\mu$, the cone's cross-section at $t_\mu$). Lifting into world coordinates, the mean is $\mu = o + \mu_t d$; the along-ray variance acts in the $d$ direction and the perpendicular spread in the plane orthogonal to $d$, so $\Sigma = \sigma_t^2\,d d^\top + \sigma_r^2\,(I - d d^\top/\|d\|^2)$.

Now the payoff. Write $\gamma$ in Fourier-feature form $\gamma(x) = [\sin(Px);\,\cos(Px)]$, where $P$ scales each axis by $2^0, 2^1, \ldots, 2^{L-1}$. Because $Px$ is *linear* in $x$, under $x \sim \mathcal{N}(\mu, \Sigma)$ the vector $Px$ is Gaussian with mean $\mu_\gamma = P\mu$ and covariance $\Sigma_\gamma = P\Sigma P^\top$. For a scalar Gaussian $v \sim \mathcal{N}(m, s^2)$, the characteristic function $\mathbb{E}[e^{iv}] = e^{im - s^2/2}$ gives $\mathbb{E}[\sin v] = \sin(m)\,e^{-s^2/2}$ and $\mathbb{E}[\cos v] = \cos(m)\,e^{-s^2/2}$ — the expected sinusoid is the sinusoid of the mean times $e^{-\frac12 s^2}$. The mean sets the phase; the variance shrinks the amplitude. I call the result the integrated positional encoding:
$$\mathrm{IPE}(\mu, \Sigma) = \big[\,\sin(\mu_\gamma) \circ \exp(-\tfrac12\,\mathrm{diag}(\Sigma_\gamma));\ \cos(\mu_\gamma) \circ \exp(-\tfrac12\,\mathrm{diag}(\Sigma_\gamma))\,\big].$$
Reading the $\exp(-\frac12 s^2)$ factor is the whole story: each entry of $Px$ is a particular frequency of a particular coordinate, and its variance is the matching diagonal entry of $\Sigma_\gamma$. If that frequency's period is large compared with the frustum's spread, $s^2$ is tiny, the damping is $\approx 1$, and the feature passes through unchanged; if the period is small — the sinusoid oscillates many times across the frustum, exactly the frequency that would alias — then $s^2$ is large, the damping drives the feature smoothly to zero. Anti-aliasing falls straight out as a soft, per-frequency low-pass whose cutoff is set by the frustum's size. Because each encoding dimension depends only on its own marginal Gaussian, I need only the *diagonal* of $\Sigma_\gamma$, never the full matrix: the block-of-powers structure of $P$ makes $\mathrm{diag}(\Sigma_\gamma) = [\mathrm{diag}(\Sigma),\,4\,\mathrm{diag}(\Sigma),\,\ldots,\,4^{L-1}\,\mathrm{diag}(\Sigma)]$ (frequency $2^l$ scales variance by $4^l$), and $\mathrm{diag}(\Sigma) = \sigma_t^2\,(d \circ d) + \sigma_r^2\,(1 - d\circ d/\|d\|^2)$ elementwise. Computed this way the IPE costs essentially the same as a plain positional encoding — no integral, no extra samples. There is also a free consequence: $L$ stops being a tuning knob. In point-sampled encoding $L$ is a hard truncation you must hand-pick; here the cutoff is soft and automatic, dictated entirely by camera geometry, so I set $L$ absurdly large (say $16$, large enough that the top frequency's feature is already below numerical epsilon for any frustum) and never touch it. Plain PE is recovered exactly as the degenerate $\Sigma = 0$ case, where every $\exp(-\frac12 \cdot 0) = 1$, so the IPE genuinely generalizes the old encoding as its zero-width limit.

This unlocks one more simplification. The point-sampled pipeline needed *two* separate MLPs — a coarse and a fine network — because a point-fed MLP could only ever learn the scene at a single scale, having no way to know which scale a query belonged to. But the IPE *carries scale explicitly*: a frustum's feature encodes how big it is. So one MLP can answer queries at any scale, and I collapse the two networks into a single multiscale MLP queried twice in a hierarchy — half the parameters, and a touch faster and simpler. Mechanically: cast a cone, draw a coarse stratified set of distances, form frustums, IPE-featurize them, run the MLP to $(\sigma_k, c_k)$, and composite with the standard alpha rule $C = \sum_k T_k(1 - e^{-\sigma_k \delta_k}) c_k$ with $T_k = e^{-\sum_{k'<k}\sigma_{k'}\delta_{k'}}$; read off the compositing weights $w_k = T_k(1 - e^{-\sigma_k \delta_k})$ as a piecewise-constant PDF; draw a fine set from it by inverse-transform sampling; render again with the *same* MLP. Two small fixes on the resampling guard against greedily following a spiky coarse PDF that has empty zero-weight regions the fine pass would never probe: widen the weights with a 2-tap max filter followed by a 2-tap blur (a blurpool) and add a small floor $\alpha$, $w'_k = \tfrac12\big(\max(w_{k-1}, w_k) + \max(w_k, w_{k+1})\big) + \alpha$ with $\alpha \approx 0.01$, so every region keeps a probability floor and some fine samples always land even in nominally empty space. Because the one shared MLP now incurs two reconstruction losses that must be balanced, the objective puts a small multiplier on the coarse term,
$$\min_\Theta \sum_{\text{rays}} \Big[\lambda\,\|C_{\text{true}} - C_{\text{coarse}}\|^2 + \|C_{\text{true}} - C_{\text{fine}}\|^2\Big], \quad \lambda \approx 0.1,$$
so the fine reconstruction dominates while the coarse pass is still trained enough to place good samples. The net hyperparameter count is a wash: I add $\lambda$ and the floor $\alpha$, and I delete $L$ and the separate coarse/fine sample counts. A few stability choices round it out: density uses a shifted softplus $\log(1 + e^{x-1})$ rather than a ReLU (a hard ReLU can drive the whole density field negative into a zero-gradient dead zone, and the $-1$ shift biases initial densities small so early content does not shadow everything behind it); color uses a slightly widened sigmoid $(1+2\epsilon)/(1+e^{-x}) - \epsilon$ with $\epsilon \approx 0.001$ so the network need never push into the dead sigmoid tails to reach pure black or white; and for multiscale training each pixel's loss is scaled by its footprint area so the few low-resolution pixels are not drowned out. Training uses Adam, batch $4096$, learning rate annealed logarithmically from $5\cdot10^{-4}$ to $5\cdot10^{-6}$ over $1\mathrm{M}$ iterations, $128$ samples per level.

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
