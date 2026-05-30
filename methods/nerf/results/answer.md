# NeRF: Neural Radiance Fields for View Synthesis

## Problem

Given a sparse set of RGB images of a static scene with known camera poses and intrinsics,
synthesize photorealistic novel views. The representation should be continuous (no
resolution ceiling), compact, optimizable from 2D images alone, and able to capture both
high-frequency geometry/texture and view-dependent (specular) appearance.

## Key idea

Represent the scene as a continuous 5D function — a multilayer perceptron (MLP)

  F_Θ : (x, d) → (c, σ),   x = (x,y,z) position,  d = viewing direction (unit 3-vector),

returning an emitted RGB color c and a volume density σ. Render any pixel by tracing the
camera ray r(t) = o + t·d through this field and compositing color and density with the
classical emission–absorption volume-rendering integral, which is differentiable, so the
MLP is optimized directly against the input photos.

Three components make it work:

1. **Volume rendering by quadrature.** The expected color of a ray is
   C(r) = ∫_{t_n}^{t_f} T(t) σ(r(t)) c(r(t),d) dt,  with transmittance
   T(t) = exp(−∫_{t_n}^{t} σ(r(s)) ds). Sample N points by stratified sampling (one uniform
   draw per evenly-spaced bin, which keeps the queried positions continuous over training),
   and evaluate the integral as alpha compositing:

     Ĉ(r) = Σ_i T_i (1 − exp(−σ_i δ_i)) c_i,   T_i = exp(−Σ_{j<i} σ_j δ_j) = Π_{j<i}(1−α_j),

   where δ_i = t_{i+1} − t_i and α_i = 1 − exp(−σ_i δ_i). The per-sample weight is
   w_i = T_i α_i.

2. **Positional encoding.** Plain MLPs are biased toward low-frequency functions, so a
   network fed raw coordinates renders oversmoothed results. Lift each input coordinate
   through a Fourier feature map before the network:

     γ(p) = ( sin(2^0 π p), cos(2^0 π p), …, sin(2^{L−1} π p), cos(2^{L−1} π p) ),

   applied per coordinate of x (L = 10) and of d (L = 4). F_Θ = F'_Θ ∘ γ.

3. **Density-from-position, color-from-(position, direction).** σ = σ(x) depends on position
   only, which enforces multiview-consistent geometry; c = c(x, d) depends additionally on
   direction, capturing specular / non-Lambertian appearance.

And for efficiency, **hierarchical volume sampling**: a coarse network is evaluated on N_c
stratified samples; its normalized weights ŵ_i = w_i / Σ_j w_j form a piecewise-constant
PDF along the ray; N_f more points are drawn from it by inverse-transform sampling; a fine
network is evaluated on the union of all N_c + N_f points for the final color.

## Architecture

γ(x) → 8 fully-connected ReLU layers, width 256, with a skip connection re-injecting γ(x)
at the 5th layer → outputs σ (ReLU-rectified, nonnegative) and a 256-d feature vector. The
feature is concatenated with γ(d) and passed through one 128-wide ReLU layer → RGB color
(sigmoid into [0,1]).

## Loss

Squared error on both coarse and fine renders (the coarse term keeps the coarse weights
useful for guiding the fine sampling):

  L = Σ_{r ∈ R} [ ‖Ĉ_c(r) − C(r)‖² + ‖Ĉ_f(r) − C(r)‖² ].

One network pair per scene. Adam, learning rate decaying 5×10⁻⁴ → 5×10⁻⁵; batch of ~4096
rays sampled from all pixels; N_c = 64, N_f = 128. For unbounded forward-facing real scenes,
rays are remapped into normalized device coordinates (NDC), where the depth axis is linear
in disparity so the far plane at infinity becomes a finite bound; with far → ∞ the ray maps
to t' ∈ [0,1) via t' = 1 − o_z/(o_z + t d_z).

## Code

Faithful to the standard PyTorch implementation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Positional encoding (Sec. positional encoding) ----
class Embedder:
    def __init__(self, num_freqs, input_dims=3, include_input=True, log_sampling=True):
        self.embed_fns, self.out_dim = [], 0
        if include_input:
            self.embed_fns.append(lambda x: x); self.out_dim += input_dims
        if log_sampling:
            freqs = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            freqs = torch.linspace(2. ** 0., 2. ** (num_freqs - 1), num_freqs)
        for f in freqs:
            for p in (torch.sin, torch.cos):
                self.embed_fns.append(lambda x, p=p, f=f: p(x * f * torch.pi))
                self.out_dim += input_dims

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    e = Embedder(num_freqs=multires)
    return (lambda x: e(x)), e.out_dim


# ---- Model: sigma(x), rgb(x, d) ----
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, skips=(4,)):
        super().__init__()
        self.skips = skips
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W + input_ch, W) if i in skips else nn.Linear(W, W)
             for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.pts_linears[0].in_features,
                                                  self.views_linears[0].in_features
                                                  - self.feature_linear.out_features], -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)                  # density: position only
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)     # bring in viewing direction
        for l in self.views_linears:
            h = F.relu(l(h))
        rgb = self.rgb_linear(h)
        return torch.cat([rgb, alpha], -1)            # [..., 4], raw


# ---- Volume rendering quadrature: integral -> alpha compositing ----
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False):
    dists = z_vals[..., 1:] - z_vals[..., :-1]                          # delta_i
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)            # true world length
    rgb = torch.sigmoid(raw[..., :3])
    noise = torch.randn_like(raw[..., 3]) * raw_noise_std if raw_noise_std > 0 else 0.
    alpha = 1. - torch.exp(-F.relu(raw[..., 3] + noise) * dists)        # 1 - exp(-sigma*delta)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1. - alpha + 1e-10], -1), -1)[..., :-1]  # T_i = prod_{j<i}(1-a_j)
    weights = alpha * T                                                 # w_i = T_i * alpha_i
    rgb_map = torch.sum(weights[..., None] * rgb, -2)                   # sum_i w_i c_i
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map, weights


# ---- Hierarchical sampling: inverse-transform sampling of the weight PDF ----
def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0., 1., N_samples).expand(*cdf.shape[:-1], N_samples)
    else:
        u = torch.rand(*cdf.shape[:-1], N_samples)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)
    matched = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched), -1, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


# ---- Per-ray render: stratified coarse pass + weight-guided fine pass ----
def render_rays(rays_o, rays_d, viewdirs, near, far, embed_fn, embeddirs_fn,
                network_coarse, network_fine, N_samples=64, N_importance=128,
                perturb=True, raw_noise_std=0., white_bkgd=False):
    def query(net, pts):
        x = embed_fn(pts.reshape(-1, 3))
        d = embeddirs_fn(viewdirs[:, None].expand(pts.shape).reshape(-1, 3))
        out = net(torch.cat([x, d], -1))
        return out.reshape(*pts.shape[:-1], out.shape[-1])

    N_rays = rays_o.shape[0]
    t_vals = torch.linspace(0., 1., N_samples)
    z_vals = (near * (1. - t_vals) + far * t_vals).expand(N_rays, N_samples)
    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = query(network_coarse, pts)
    rgb_c, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    z_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samp = sample_pdf(z_mid, weights[..., 1:-1], N_importance, det=not perturb).detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samp], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = query(network_fine, pts)
    rgb_f, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
    return rgb_c, rgb_f


# ---- Training step: MSE on both coarse and fine ----
def train_step(rays_o, rays_d, viewdirs, near, far, target,
               embed_fn, embeddirs_fn, net_c, net_f, optimizer, **kw):
    rgb_c, rgb_f = render_rays(rays_o, rays_d, viewdirs, near, far,
                               embed_fn, embeddirs_fn, net_c, net_f, **kw)
    loss = F.mse_loss(rgb_c, target) + F.mse_loss(rgb_f, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss


# ---- NDC remap for unbounded forward-facing scenes ----
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]          # shift to near plane
    rays_o = rays_o + t[..., None] * rays_d
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]
    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    return torch.stack([o0, o1, o2], -1), torch.stack([d0, d1, d2], -1)


# ---- Wiring ----
def build():
    embed_fn, in_ch = get_embedder(10)        # gamma(x), L = 10
    embeddirs_fn, in_ch_views = get_embedder(4)  # gamma(d), L = 4
    net_c = NeRF(input_ch=in_ch, input_ch_views=in_ch_views)
    net_f = NeRF(input_ch=in_ch, input_ch_views=in_ch_views)
    params = list(net_c.parameters()) + list(net_f.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-4, betas=(0.9, 0.999), eps=1e-7)
    return embed_fn, embeddirs_fn, net_c, net_f, optimizer
```
