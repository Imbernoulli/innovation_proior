# 3D Gaussian Splatting, distilled

3D Gaussian Splatting (3DGS) represents a scene as an unstructured cloud of anisotropic 3D
Gaussians — each with a position, a covariance, an opacity, and spherical-harmonic color — and
optimizes them, per scene, against a photometric loss on the input photographs. It is both a
differentiable volumetric representation (so it optimizes like a radiance field) and an explicit
primitive that projects to an analytic 2D footprint (so a GPU can splat millions of them in real
time). A tile-based rasterizer with a single global depth sort renders it; adaptive density
control adds, splits, and prunes Gaussians during optimization. The supervision is the
photometric loss alone — there is no regularization on the Gaussians' parameters.

## Problem it solves

Novel-view synthesis from SfM-calibrated photos of a static scene, reaching state-of-the-art
quality on held-out views while training in minutes and rendering in real time (>=30 fps at
1080p) for complete, unbounded scenes — the combination no prior representation achieved at once.
Input is calibrated cameras plus a sparse SfM point cloud; no depth, normals, or MVS geometry.

## Key idea

Both continuous radiance fields and point-based renderers compute a pixel by the *same* front-to-
back alpha compositing `C = sum_i c_i alpha_i prod_{j<i}(1-alpha_j)`; the cost difference lives in
the *representation*. Choose a primitive that is volumetric-and-differentiable *and* explicit-and-
rasterizable: the anisotropic 3D Gaussian `G(x) = exp(-1/2 (x-mu)^T Sigma^{-1} (x-mu))`. It needs
no normals, has an analytic 2D footprint under projection, and lets the photometric gradient flow.

- **Valid covariance by construction.** Optimizing `Sigma` directly breaks positive-semi-
  definiteness under gradient descent. Factor `Sigma = R S S^T R^T` and store a scale vector `s`
  (via `exp`, always positive) and a unit quaternion `q` (normalized) — any optimizer step yields
  a legal `Sigma`. Opacity uses `sigmoid` (into `[0,1)`); these activations are what make an
  unconstrained optimizer respect each quantity's constraint.
- **Screen projection (EWA).** With viewing transform `W` and the Jacobian `J` of the local
  affine approximation of the perspective projection, the screen-space covariance is
  `Sigma' = J W Sigma W^T J^T`; the 2D footprint covariance is the **upper-left 2x2** of `Sigma'`
  after skipping the third row and column. Per-pixel `alpha` = (2D Gaussian
  footprint) x opacity. Gradients for scale and rotation are derived explicitly (no autodiff in
  the inner loop): `dSigma/dM = 2M^T` with `M = RS`, plus `dM/ds` and `dM/dq` from `R(q)`.
- **Color.** Spherical harmonics (degree 3) for view-dependent appearance; only the DC band is
  optimized at first, with one band added every 1000 iterations so angularly under-constrained
  high-order coefficients cannot corrupt the base color.
- **Real-time rasterizer.** Tile the screen into 16x16 blocks; frustum/tile cull (99% confidence
  + guard band); instance each Gaussian per overlapped tile; assign a 64-bit key
  `[tile_id | depth32]`; **one global radix sort** orders all splats by tile then depth at once
  (no per-pixel sort). Per tile, blend front-to-back, stopping when accumulated `alpha` saturates.
- **Backward with unlimited splats, constant memory.** Re-traverse the same sorted per-tile
  arrays back-to-front; recover each intermediate transmittance from the stored final
  `T_final = prod_i(1-alpha_i)` by dividing out one `(1-alpha_i)` factor at a time.
  No per-pixel blend lists, no cap on the number
  of Gaussians that receive gradients (a cap, as in order-limited sphere rasterization,
  destabilizes optimization). Stability: skip `alpha < 1/255`, clamp `alpha <= 0.99`.
- **Loss.** `L = (1-lambda) L1 + lambda (1 - SSIM)` with `lambda = 0.2` (i.e. `0.8*L1 + 0.2*(1-SSIM)`):
  L1 for pixel fidelity, D-SSIM for structure (L1 alone blurs). This is the *entire* objective.
- **Adaptive density control.** The photometric gradient only adjusts existing Gaussians. Both
  under- and over-reconstruction show large **view-space positional gradients**. Every 100 iters,
  for Gaussians whose averaged view-space position-gradient magnitude exceeds `tau_pos = 0.0002`:
  **clone** the small ones (copy at the same position and size; later steps separate the copy under
  the same gradient signal) and
  **split** the large ones into two (sample new means using the Gaussian as a PDF, divide each
  child scale by `0.8*N`, so `N=2` gives `/1.6`: add detail without keeping one broad footprint).
  **Prune** Gaussians with opacity below threshold or that are
  too large in world/screen space. Every 3000 iters **reset opacity** to near zero so the
  optimization must re-justify each Gaussian, dissolving floaters.
- **Initialization.** Means from SfM points; isotropic covariance with axis = mean distance to
  the 3 nearest neighbors; opacity 0.1; exponential-decay learning-rate schedule on positions.

## Final algorithm

```
Initialize Gaussians from sparse SfM points (mu, isotropic Sigma, alpha=0.1, SH DC = color)
for step in 1..30000:
    if step % 1000 == 0: activate one more SH band (up to degree 3)
    sample a training camera V and its photo I_gt
    I = Rasterize(Gaussians, V)                         # EWA project -> tile sort -> alpha-composite
    L = 0.8 * L1(I, I_gt) + 0.2 * (1 - SSIM(I, I_gt))    # photometric loss = the whole objective
    backprop L (explicit covariance gradients; back-to-front re-traversal recovers transmittance)
    track max image-space radius and average view-space position gradient
    if step in densification window:
        every 100 steps: clone (small & high-grad), split (large & high-grad),
                         prune (alpha<eps or too large)
        every 3000 steps: reset opacity ~ 0
    Adam step over (mu, scale, quat, opacity, SH); zero grad
```

## Working code

Grounded in the canonical optimization loop and Gaussian model (factored covariance, activations,
clone/split/prune, opacity reset). The rasterizer is a CUDA kernel (EWA projection + global sort +
front-to-back blend); shown here as its call. No regularization term appears in the loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_matrix(q):
    """Unit quaternion q=(real, i, j, k) to rotation matrix."""
    r, x, y, z = q.unbind(-1)
    rows = (
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z),     2 * (x * z + r * y),
        2 * (x * y + r * z),     1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y),     2 * (y * z + r * x),     1 - 2 * (x * x + y * y),
    )
    return torch.stack(rows, dim=-1).reshape(q.shape[:-1] + (3, 3))


def build_covariance(scaling, rotation):
    """Sigma = R S S^T R^T -- PSD for any (scaling > 0, unit quaternion)."""
    R = quaternion_to_matrix(F.normalize(rotation, dim=-1))   # [N,3,3]
    M = R @ torch.diag_embed(scaling)                         # M = R S, Sigma = M M^T
    return M @ M.transpose(1, 2)


class GaussianModel(nn.Module):
    def __init__(self, points, colors, sh_degree=3):
        super().__init__()
        self._xyz = nn.Parameter(points)                                       # means (SfM points)
        dist2 = torch.clamp_min(knn_mean_sq_dist(points, k=3), 1e-7)
        self._scaling = nn.Parameter(torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3))  # log-scale
        rots = torch.zeros((points.shape[0], 4), device=points.device); rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots)                                    # quaternion
        self._opacity = nn.Parameter(inverse_sigmoid(0.1 * torch.ones((points.shape[0], 1),
                                                               device=points.device)))
        features = torch.zeros((points.shape[0], 3, (sh_degree + 1) ** 2), device=points.device)
        features[:, :3, 0] = rgb_to_sh(colors)                                # DC color only at start
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
        self.max_radii2D = torch.zeros((points.shape[0],), device=points.device)
        self.max_sh_degree, self.active_sh_degree = sh_degree, 0

    @property
    def scaling(self):  return torch.exp(self._scaling)                        # > 0
    @property
    def opacity(self):  return torch.sigmoid(self._opacity)                    # [0,1)
    @property
    def rotation(self): return F.normalize(self._rotation, dim=-1)             # unit quaternion
    @property
    def features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)

    def covariance(self):
        return build_covariance(self.scaling, self.rotation)

    def render(self, camera):
        # EWA project (Sigma' = J W Sigma W^T J^T, 2x2 footprint) -> 16x16 tile, one global
        # radix sort on [tile|depth] -> front-to-back alpha-composite; CUDA rasterizer.
        return rasterize_gaussians(
            means=self._xyz, covariance=self.covariance(), opacity=self.opacity,
            sh=self.features, active_sh_degree=self.active_sh_degree,
            viewmat=camera.world_view_transform, K=camera.intrinsics,
            width=camera.image_width, height=camera.image_height,
        )  # -> image, viewspace_points, visibility, radii


def l1_loss(x, gt): return (x - gt).abs().mean()
# ssim(x, gt): standard 11x11 windowed SSIM


def photometric_loss(rendered, gt, lam=0.2):
    return (1.0 - lam) * l1_loss(rendered, gt) + lam * (1.0 - ssim(rendered, gt))


def train(model, cameras, images, num_steps=30000,
          densify_from=500, densify_until=15000, tau_pos=0.0002):
    opt = torch.optim.Adam([
        {"params": [model._xyz],           "lr": 1.6e-4, "name": "xyz"},   # exp-decayed
        {"params": [model._features_dc],   "lr": 2.5e-3, "name": "f_dc"},
        {"params": [model._features_rest], "lr": 2.5e-3 / 20.0, "name": "f_rest"},
        {"params": [model._opacity],       "lr": 2.5e-2, "name": "opacity"},
        {"params": [model._scaling],       "lr": 5e-3, "name": "scaling"},
        {"params": [model._rotation],      "lr": 1e-3, "name": "rotation"},
    ], eps=1e-15)

    accum = torch.zeros((model._xyz.shape[0], 1), device=model._xyz.device)
    denom = torch.zeros_like(accum)
    extent = cameras.extent()

    for step in range(num_steps):
        if step > 0 and step % 1000 == 0 and model.active_sh_degree < model.max_sh_degree:
            model.active_sh_degree += 1

        cam, gt = sample_view(cameras, images)
        image, viewspace_points, visibility, radii = model.render(cam)
        loss = photometric_loss(image, gt, lam=0.2)        # 0.8*L1 + 0.2*(1-SSIM); no regularizer
        loss.backward()

        with torch.no_grad():
            if step < densify_until:
                model.max_radii2D[visibility] = torch.maximum(model.max_radii2D[visibility], radii[visibility])
                accum[visibility] += viewspace_points.grad[visibility, :2].norm(dim=-1, keepdim=True)
                denom[visibility] += 1
                if step > densify_from and step % 100 == 0:
                    grads = (accum / denom.clamp_min(1)).squeeze()
                    densify_and_clone(model, opt, grads, tau_pos, extent)   # small  -> clone
                    densify_and_split(model, opt, grads, tau_pos, extent)   # large  -> split
                    max_screen = 20 if step > 3000 else None
                    prune(model, opt, min_opacity=0.005, extent=extent, max_screen=max_screen)
                    accum = torch.zeros((model._xyz.shape[0], 1), device=image.device)
                    denom = torch.zeros_like(accum)
                if step > 0 and step % 3000 == 0:
                    reset_opacity(model, opt, value=0.01)
            opt.step()
            opt.zero_grad(set_to_none=True)


def densify_and_clone(model, opt, grads, tau_pos, extent, percent_dense=0.01):
    sel = (grads >= tau_pos) & (model.scaling.max(dim=1).values <= percent_dense * extent)
    append_gaussians(model, opt, sel,                                      # copy, same position/size
        new_xyz=model._xyz[sel],
        new_features_dc=model._features_dc[sel], new_features_rest=model._features_rest[sel],
        new_opacity=model._opacity[sel], new_scaling=model._scaling[sel],
        new_rotation=model._rotation[sel])

def densify_and_split(model, opt, grads, tau_pos, extent, percent_dense=0.01, N=2):
    sel = (grads >= tau_pos) & (model.scaling.max(dim=1).values > percent_dense * extent)
    stds = model.scaling[sel].repeat(N, 1)
    samples = torch.normal(torch.zeros_like(stds), stds)                     # Gaussian as PDF
    R = quaternion_to_matrix(model.rotation[sel]).repeat(N, 1, 1)
    new_xyz = (R @ samples.unsqueeze(-1)).squeeze(-1) + model._xyz[sel].repeat(N, 1)
    new_scaling = torch.log(model.scaling[sel].repeat(N, 1) / (0.8 * N))     # divide scale by 1.6
    append_gaussians(model, opt, sel,
        new_xyz=new_xyz,
        new_features_dc=model._features_dc[sel].repeat(N, 1, 1),
        new_features_rest=model._features_rest[sel].repeat(N, 1, 1),
        new_opacity=model._opacity[sel].repeat(N, 1),
        new_scaling=new_scaling,
        new_rotation=model._rotation[sel].repeat(N, 1))
    prune_indices(model, opt, sel)                                           # remove parents

def prune(model, opt, min_opacity, extent, max_screen=20):
    mask = (model.opacity < min_opacity).squeeze()
    if max_screen is not None:
        mask |= model.max_radii2D > max_screen
        mask |= model.scaling.max(dim=1).values > 0.1 * extent
    prune_mask(model, opt, mask)

def reset_opacity(model, opt, value=0.01):
    new = inverse_sigmoid(torch.minimum(model.opacity, torch.full_like(model.opacity, value)))
    set_param(model, opt, "_opacity", new)
```

## Relation to prior methods

- **NeRF / Mip-NeRF 360:** same image-formation model (`C = sum T_i alpha_i c_i`), but implicit
  and ray-marched -> slow render/train; here the representation is explicit Gaussians and the
  renderer is a rasterizer, so render and each step are fast.
- **Plenoxels / InstantNGP:** fast via structured grids but still ray-march and are
  grid-resolution-limited; here the representation is unstructured anisotropic Gaussians, no grid.
- **2D surfel splatting (Kopanas 2021; Yifan 2019):** planar discs *with normals* and *per-pixel*
  sorting; here full 3D anisotropic covariance (no normals) and a *single global* sort.
- **Pulsar:** sorts once per frame but is order-*independent* and caps gradient-receiving splats;
  here blending respects visibility order and an unlimited number of splats receive gradients.
