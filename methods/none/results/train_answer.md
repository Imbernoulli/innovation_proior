I have photographs of a static scene and cameras calibrated by Structure-from-Motion, and I want to render new viewpoints — full, unbounded scenes with sky and far background, not a single object on a turntable. What I want is three things that no representation gives me together: quality matching the best method around, optimization that takes minutes rather than days, and rendering fast enough to fly through the scene in real time at 1080p. Every option on the table buys one or two and loses the third, and the reason is always structural — it lives in how the scene is stored and how that storage becomes pixels. The continuous radiance fields (NeRF, Mip-NeRF 360) reach the highest quality, but the representation is implicit: to evaluate a pixel I march the ray and query the field at many samples, including through empty space, which makes training take tens of hours and rendering take seconds per frame. The fast continuous methods (Plenoxels, InstantNGP) interpolate features from a structured grid to cut the query cost, but they still ray-march many samples per pixel, so they stay short of real-time at high resolution, and their quality is capped by the grid's resolution and its difficulty representing empty space and fine detail compactly. The explicit point-based and surfel-splatting renderers rasterize fast on a GPU, but they assume each primitive is a planar disc carrying a surface normal — and a sparse SfM cloud cannot supply reliable normals — while correct depth-ordered alpha blending forces them into expensive per-pixel sorting; the variants that avoid that (Pulsar) blend order-independently and cap how many front-most primitives receive gradients, an approximation that destabilizes optimization on scenes whose depth complexity varies.

The reframing that breaks the problem open is this: continuous fields and point renderers compute a pixel by *exactly the same* sum. The volumetric quadrature $C = \sum_i T_i(1 - e^{-\sigma_i \delta_i}) c_i$ with $T_i = e^{-\sum_{j<i}\sigma_j\delta_j}$, once I write $\alpha_i = 1 - e^{-\sigma_i\delta_i}$ and notice $T_i = \prod_{j<i}(1-\alpha_j)$, collapses to front-to-back alpha compositing
$$C = \sum_i c_i\,\alpha_i \prod_{j<i}(1-\alpha_j),$$
and a point-based renderer blending ordered overlapping primitives with color $c_i$ and opacity $\alpha_i$ computes the very same thing. The image-formation model is shared; the gulf between "tens of hours, seconds per frame" and "rasterize in a millisecond" is the *representation*, not the integrator. So I do not need a cleverer integrator. I need a primitive that optimizes as gently as a continuous field yet a GPU can splat almost for free.

I propose 3D Gaussian Splatting. The scene is an unstructured cloud of anisotropic 3D Gaussians, each $G(x) = \exp\!\big(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\big)$ with a position $\mu$, a covariance $\Sigma$, an opacity, and spherical-harmonic color, optimized per scene against a photometric loss and nothing else. The Gaussian is the right primitive precisely because it lives on both sides of the divide: it is a smooth volumetric kernel, so the photometric gradient flows through it like through a density field, and it stays a Gaussian under affine maps and integrates along a line to a 2D Gaussian, so it projects to an analytic 2D footprint a rasterizer can splat — and it needs no normal anywhere. I make it anisotropic, with a full $3\times3$ covariance rather than a scalar radius, because a surface is locally a flat, elongated thing: a squashed ellipsoid can lie flat against it and stretch along it, covering the same geometry with far fewer primitives and far less off-surface bleed than a pile of isotropic balls.

The first wall is that $\Sigma$ is meaningful only if it is positive semi-definite, and if I hand its six free entries to Adam a gradient step will produce a negative eigenvalue — the Gaussian inverts and the render is garbage. Gradient descent knows nothing of the PSD cone and projecting back every step is awkward. The resolution is to dissolve the constraint into the parameterization: any PSD matrix factors as $\Sigma = R S S^\top R^\top$, the eigendecomposition in disguise, with $R$ a rotation and $S$ a diagonal scaling holding the axis lengths. If I store a scaling vector and a rotation and *build* $\Sigma$ from them, then whatever the optimizer does, $\Sigma$ comes out PSD by construction. I represent the rotation by a quaternion (normalized to unit length to be a valid rotation), which optimizes far more smoothly than the nine entries of $R$. I keep the axis lengths positive not by constraint but by storing their logarithm and passing it through $\exp$, since $\exp(\cdot) > 0$ always and its multiplicative nature gives clean gradients across the orders of magnitude that separate tiny and huge Gaussians. Opacity, which must live in $[0,1)$ to be a blending weight, is stored as a raw logit and squashed through a sigmoid. None of these activations is decoration — each is exactly the device that lets an unconstrained optimizer respect a constraint the quantity must obey.

For color, a fixed RGB triple cannot show a specular highlight that moves as I orbit, so I attach spherical harmonics (degree 3, sixteen coefficients per channel) per Gaussian — an orthonormal basis on the sphere of directions, evaluated per view at negligible cost. But a capture that does not see a Gaussian from all around leaves the high-order coefficients badly under-constrained, and the optimizer can dump nonsense into them and even corrupt the zero-order (DC) term, which is the base diffuse color. So I do not turn on all bands at once: I optimize only the DC color first and introduce one more band every thousand iterations, letting the base color settle before angularly-starved coefficients are allowed to move.

Turning a 3D Gaussian into a screen footprint fast and differentiably is the EWA splatting step. A Gaussian's covariance transforms under an affine map as $\Sigma \to M\Sigma M^\top$, but the perspective projection is not affine — the divide makes it nonlinear. The fix is the local affine approximation: replace the projective map near the Gaussian's center by the first two terms of its Taylor expansion, with Jacobian $J = dm/du$ at the center, which for the pinhole projection $(x = u_0/u_2,\,y = u_1/u_2)$ is the familiar derivative of the perspective divide. With viewing transform $W$ taking world to camera, the ray-space covariance is
$$\Sigma' = J\,W\,\Sigma\,W^\top J^\top,$$
a $3\times3$ matrix; the EWA reduction to the 2D screen ellipse is simply its **upper-left $2\times2$ block** — skip the third row and column. The per-pixel contribution is this 2D Gaussian evaluated at the pixel offset times the Gaussian's own opacity, which is exactly the $\alpha_i$ that enters the compositing sum, the front-to-back blend justified by $e^{-x}\approx 1-x$. Because I will have millions of Gaussians and tens of thousands of steps, each of which renders, I cannot afford autodiff through $\Sigma' = JW\Sigma W^\top J^\top$ in the inner loop, so I derive the gradients by hand. Writing $M = RS$ so $\Sigma = MM^\top$, differentiating the bilinear form gives the partial of each screen entry with respect to $\Sigma_{ij}$ as products of the entries of $U = JW$; differentiating $MM^\top$ and using symmetry collapses the shared part to $d\Sigma/dM = 2M^\top$; and from there $dM/ds$ is a sparse pattern (since $M_{ij} = R_{ij}s_j$) while $dM/dq$ comes from differentiating $R(q)$ entrywise with each column scaled by its axis length, chained through the quaternion normalization $dq/dr = (I - qq^\top)/\|r\|$. Chaining these yields explicit scale and rotation gradients with no autodiff inside the loop, while the covariance stays valid by construction.

Rendering in real time is the property I refuse to give up, and the cost that historically killed alpha-blended splatting is the *per-pixel* depth sort. I sort once for the whole image instead. I tile the screen into $16\times16$ blocks; cull to Gaussians whose 99%-confidence ellipsoid meets the frustum and reject ones too near the camera plane where the projected covariance is numerically unstable; for each survivor I instantiate one copy per overlapped tile and give it a 64-bit key whose high bits are the tile index and low 32 bits are the quantized view-space depth. A single GPU radix sort over all instances then orders everything by tile and, within a tile, by depth at once — one sort per frame, with every tile's depth order falling out together. Each tile becomes a thread block; threads cooperatively load packets of Gaussians into shared memory, and each pixel walks its tile's list front-to-back accumulating color and alpha, stopping when its alpha saturates. This is approximate — the depth order is resolved once per tile rather than re-resolved at each pixel — but the error shrinks as splats approach pixel size, and the payoff is that every optimization step gets a fast visibility-aware render.

The backward pass has its own trap. Storing, per pixel, the explicit list of which splats blended into it forces dynamic memory I do not want; capping the number of front-most splats that get gradients is the Pulsar-style approximation I distrust, because a fixed cap either starves deep regions of gradient or misattributes error by computing each splat's gradient as if the splats behind it did not exist, which makes optimization thrash. So I cap nothing and store no lists. Instead I re-traverse the *same* sorted per-tile arrays from the forward pass, this time back-to-front. The one thing the gradient needs that is not free this way is the intermediate transmittance at each splat, and rather than store the whole shrinking sequence I store only the final $T_{\text{final}} = \prod_i (1-\alpha_i)$ per pixel and, walking back, undo one factor at a time via $T \leftarrow T/(1-\alpha_i)$. That reconstructs every intermediate coefficient from a single scalar, giving gradients for an *unlimited* number of blended Gaussians at constant per-pixel overhead. The division is ill-conditioned only when a factor is near zero, so I skip any splat with $\alpha < 1/255$ in both passes and clamp $\alpha \le 0.99$, keeping the recovered transmittance well-behaved.

The supervision is a light combination of L1 and D-SSIM,
$$L = (1-\lambda)\,L_1 + \lambda\,(1 - \mathrm{SSIM}),\qquad \lambda = 0.2,$$
i.e. $0.8\,L_1 + 0.2\,(1-\mathrm{SSIM})$. L1 alone drifts to a blurry optimum because it will average away structure that costs little per-pixel error; SSIM rewards getting local edges and texture right but does not pin absolute pixel values; the convex combination, weighted lightly toward structure, gets both. The photographs are the only signal — there is no regularization on the Gaussians' parameters of any kind; the data term is the whole objective.

The final ingredient is what turns a clever renderer into a reconstruction method, because the photometric gradient can only adjust the Gaussians that *exist* — it can nudge a Gaussian's position, covariance, color, and opacity, but it has no move to create one where a region is empty and no move to relieve one large Gaussian straining to cover detail it cannot. Starting from a sparse SfM cloud, vast parts of the scene have no primitive, and the loss leaves them blank. So I interleave adaptive density control with optimization, keyed on a single observation: both failure modes — an under-reconstructed region with too few small Gaussians, and an over-reconstructed region crammed into one large Gaussian — share the symptom that the optimizer is pulling hard on *where* the responsible Gaussian sits because reshaping it is not enough, which shows up as a large view-space positional gradient. Every hundred iterations, after a warm-up, I look at Gaussians whose averaged view-space position-gradient magnitude exceeds $\tau_{\text{pos}} = 0.0002$ and densify by size. If the Gaussian is small (scale below a fraction of the scene extent), the region needs more coverage, so I **clone** it — a copy at the same position and size, which later steps separate under the same gradient signal. If it is large, the region has support but not detail, so I **split** it into two smaller children, sampling their positions using the parent Gaussian itself as a probability density and dividing each child's scale by $0.8N$ (with $N=2$, by $1.6$). I prune as I go — removing Gaussians whose opacity has fallen below threshold (essentially transparent) and ones grown too large in world or screen space — and, to fight floaters that the optimizer conjures near the input cameras, every three thousand iterations I reset all opacities to near zero, forcing the optimization to re-justify each Gaussian: the genuinely needed ones have their opacity driven back up and the floaters stay low and get culled. Initialization sets the means at the SfM points, gives each an isotropic covariance whose axis length is the mean distance to its three nearest neighbors, sets opacity to $0.1$, and schedules the position learning rate with exponential decay so positions move freely early and settle late. The whole procedure runs for thirty thousand steps.

The implementation has two pieces: a `GaussianModel` that holds the parameters and builds the covariance from the activated scale and normalized quaternion, and a training step that renders, takes the photometric loss, backpropagates, records the radii and gradient statistics, runs the density-control bookkeeping, and steps the optimizer. The rasterizer is a CUDA kernel (EWA projection, single global sort, front-to-back blend), shown here as its call.

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
