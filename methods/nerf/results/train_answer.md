I represent the scene as a continuous neural radiance field, the canonical name being NeRF, short for Neural Radiance Fields for View Synthesis. The goal is to take a sparse set of RGB photographs of a static scene, each with known camera pose and intrinsics, and synthesize photorealistic images from viewpoints that were never captured. Rather than committing to a discrete voxel grid or a template mesh, I encode the scene directly into the weights of a single multilayer perceptron. This makes the representation compact and continuous, free of any baked-in resolution ceiling, and because the rendering step is fully differentiable, I can optimize it from nothing more than the input images.

The core function is a mapping from a 3D position and a viewing direction to a volume density and an emitted RGB color. Concretely, I write F_Theta : (x, d) -> (c, sigma), where x is a point in space, d is a unit viewing-direction vector, sigma is a scalar density, and c is a three-channel color. To render a pixel, I trace the camera ray r(t) = o + t d from near bound t_n to far bound t_f and integrate the contributions of all points along the ray. The physical model is the classical emission-absorption optical model for participating media. The transmittance T(t) is the probability that light survives from the start of the ray to parameter t without being absorbed, and it satisfies the differential equation dT/dt = -sigma(r(t)) T(t). Solving gives T(t) = exp(-integral from t_n to t of sigma(r(s)) ds). The expected color reaching the camera is then the integral from t_n to t_f of T(t) sigma(r(t)) c(r(t), d) dt. This integral is differentiable in both sigma and c, so I can backpropagate a pixel-level reconstruction error all the way into the network weights.

Because I cannot evaluate the integral analytically, I discretize it by sampling points along the ray. I partition the interval into N evenly spaced bins and draw one sample uniformly at random inside each bin, a strategy called stratified sampling. This is crucial: any fixed set of depths would quietly turn the continuous network back into a discrete lookup table, but stratification means that over the course of training the network is queried at a dense continuum of positions and must actually learn a continuous function. Assuming sigma and c are constant on each small interval with value sigma_i and c_i and width delta_i, the contribution of interval i becomes T_i (1 - exp(-sigma_i delta_i)) c_i, where T_i = exp(-sum of sigma_j delta_j for j < i). Defining alpha_i = 1 - exp(-sigma_i delta_i), this reduces to exactly front-to-back alpha compositing: the rendered color is sum_i (product_{j < i} (1 - alpha_j)) alpha_i c_i. The bridge between density and opacity, alpha_i = 1 - exp(-sigma_i delta_i), correctly accounts for the physical distance the ray travels through each sample. The remaining transmittance, 1 minus the sum of weights, can be composited against a known background color when the dataset uses one.

A naïve MLP fed raw 3D coordinates and directions produces blurry, oversmoothed renderings. The reason is the spectral bias of deep ReLU networks: they fit low-frequency components first and struggle to represent rapid spatial variation such as sharp edges and fine texture. To fix this, I lift each input coordinate through a Fourier feature map before it enters the network. For a scalar coordinate p, the encoding is gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ..., sin(2^{L-1} pi p), cos(2^{L-1} pi p)). I apply this independently to each component of x with L = 10, giving the network access to spatial frequencies up to the finest detail resolvable in the images, and to each component of d with L = 4, since view-dependent effects such as specular highlights vary more smoothly with angle. With this positional encoding in place, the network can represent crisp geometry and texture while remaining a smooth function of its lifted input.

The architecture routes position and direction carefully to enforce a structural constraint. Density must depend on position only, because geometry is a property of the scene, not of the observer. If sigma could see the viewing direction, the network could place opaque content at different depths for different cameras and fit each photograph independently without learning any coherent 3D structure. Color, on the other hand, is allowed to depend on both position and direction, because real materials exhibit view-dependent specularities and non-Lambertian effects. The network therefore processes the encoded position gamma(x) through eight fully-connected ReLU layers of width 256, with a skip connection that re-injects gamma(x) at the fifth layer to preserve fine positional information. From this trunk I output sigma through a single linear head, rectified to be nonnegative. A 256-dimensional feature vector is then concatenated with the encoded viewing direction gamma(d) and passed through a 128-wide ReLU layer to produce the final RGB color, squashed into [0, 1] with a sigmoid.

Uniform sampling along every ray is wasteful: most of the interval is empty space or lies behind the first opaque surface and contributes almost nothing to the integral. I therefore use hierarchical volume sampling with two networks, a coarse one and a fine one. First, the coarse network is evaluated at N_c stratified samples along the ray. From the coarse pass I obtain per-sample weights w_i = T_i alpha_i, which form a piecewise-constant probability density describing where the ray's color actually comes from. I normalize these weights and draw N_f additional samples from this distribution by inverse-transform sampling, building the cumulative distribution function and inverting uniform random draws. The fine network is then evaluated on the union of the coarse and fine samples, and that composite gives the final pixel color. Both renders are supervised against the true pixel color with a mean-squared-error loss, so the coarse network remains a reliable guide for importance sampling. In practice I use N_c = 64 and N_f = 128.

For unbounded forward-facing real scenes, such as a fern on a table photographed from the front, the far range is effectively infinite and uniform depth sampling is poorly allocated. I remap rays into normalized device coordinates, where the depth axis becomes linear in disparity rather than metric depth. Under this remap, the point at infinity maps to a finite coordinate t' in [0, 1), and sampling uniformly in t' corresponds to sampling uniformly in inverse depth in the original scene. This naturally places more samples near the camera, where small depth changes have large image effects, and fewer samples far away. The remap preserves straight lines, so ray tracing remains straightforward.

Training optimizes one network pair per scene using Adam with a learning rate decaying from 5e-4 to 5e-5. Each training step samples a batch of rays from all pixels of all input images, renders both coarse and fine colors, and minimizes the sum of squared errors against the observed pixels. No depth maps, meshes, or 3D supervision are required. The result is a compact, continuous scene representation that can render novel views with high-frequency detail and realistic view-dependent appearance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder:
    def __init__(self, num_freqs, input_dims=3, include_input=True):
        self.embed_fns, self.out_dim = [], 0
        if include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dim += input_dims
        freqs = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        for f in freqs:
            for p in (torch.sin, torch.cos):
                self.embed_fns.append(lambda x, p=p, f=f: p(x * f * torch.pi))
                self.out_dim += input_dims

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    e = Embedder(num_freqs=multires)
    return (lambda x: e(x)), e.out_dim


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
        input_pts, input_views = torch.split(
            x, [self.pts_linears[0].in_features,
                self.views_linears[0].in_features - self.feature_linear.out_features], -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = F.relu(l(h))
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
        for l in self.views_linears:
            h = F.relu(l(h))
        rgb = self.rgb_linear(h)
        return torch.cat([rgb, alpha], -1)


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[..., :3])
    noise = torch.randn_like(raw[..., 3]) * raw_noise_std if raw_noise_std > 0 else 0.0
    alpha = 1.0 - torch.exp(-F.relu(raw[..., 3] + noise) * dists)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]),
                                 1.0 - alpha + 1e-10], -1), -1)[..., :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    return rgb_map, weights


def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0.0, 1.0, N_samples).expand(*cdf.shape[:-1], N_samples)
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


def render_rays(rays_o, rays_d, viewdirs, near, far,
                embed_fn, embeddirs_fn,
                network_coarse, network_fine,
                N_samples=64, N_importance=128,
                perturb=True, raw_noise_std=0.0, white_bkgd=False):
    def query(net, pts):
        x = embed_fn(pts.reshape(-1, 3))
        d = embeddirs_fn(viewdirs[:, None].expand(pts.shape).reshape(-1, 3))
        out = net(torch.cat([x, d], -1))
        return out.reshape(*pts.shape[:-1], out.shape[-1])

    N_rays = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, N_samples)
    z_vals = (near * (1.0 - t_vals) + far * t_vals).expand(N_rays, N_samples)
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = query(network_coarse, pts)
    rgb_c, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    z_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samp = sample_pdf(z_mid, weights[..., 1:-1], N_importance, det=not perturb).detach()
    z_vals, _ = torch.sort(torch.cat([z_vals, z_samp], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = query(network_fine, pts)
    rgb_f, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
    return rgb_c, rgb_f


def train_step(rays_o, rays_d, viewdirs, near, far, target,
               embed_fn, embeddirs_fn, net_c, net_f, optimizer, **kw):
    rgb_c, rgb_f = render_rays(rays_o, rays_d, viewdirs, near, far,
                               embed_fn, embeddirs_fn, net_c, net_f, **kw)
    loss = F.mse_loss(rgb_c, target) + F.mse_loss(rgb_f, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def build():
    embed_fn, in_ch = get_embedder(10)
    embeddirs_fn, in_ch_views = get_embedder(4)
    net_c = NeRF(input_ch=in_ch, input_ch_views=in_ch_views)
    net_f = NeRF(input_ch=in_ch, input_ch_views=in_ch_views)
    params = list(net_c.parameters()) + list(net_f.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-4, betas=(0.9, 0.999), eps=1e-7)
    return embed_fn, embeddirs_fn, net_c, net_f, optimizer


# Minimal sanity check: build and forward a small batch.
if __name__ == "__main__":
    embed_fn, embeddirs_fn, net_c, net_f, opt = build()
    rays_o = torch.randn(32, 3)
    rays_d = torch.randn(32, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    viewdirs = rays_d.clone()
    target = torch.rand(32, 3)
    loss = train_step(rays_o, rays_d, viewdirs, 2.0, 6.0, target,
                      embed_fn, embeddirs_fn, net_c, net_f, opt,
                      N_samples=32, N_importance=32)
    print("loss", loss.item())
```
