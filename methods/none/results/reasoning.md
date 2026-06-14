Let me start from what actually hurts. I have photographs of a static scene and cameras calibrated by SfM, and I want to render new viewpoints. I want three things that nobody has together: quality as good as the best method around, training that takes minutes not days, and rendering fast enough to fly through the scene in real time at high resolution — full scenes, with sky and far background, not one clean object on a turntable. Every representation I know of buys me one or two of these and loses the third, and the reason is always the same: it's baked into how the scene is stored and how that storage becomes pixels. So before I pick anything, I want to understand exactly where the loss happens.

The continuous radiance fields render by walking a ray and integrating emission against absorption. Color of a pixel is C = sum_i T_i (1 - exp(-sigma_i delta_i)) c_i, with transmittance T_i = exp(-sum_{j<i} sigma_j delta_j) accumulated from the front. If I name alpha_i = 1 - exp(-sigma_i delta_i) and notice T_i = prod_{j<i}(1 - alpha_j), this collapses to C = sum_i c_i alpha_i prod_{j<i}(1 - alpha_j) — plain front-to-back alpha compositing. Now here's the thing that I keep coming back to: a point-based renderer that blends ordered overlapping primitives, each with a color c_i and an opacity alpha_i, computes the pixel by *exactly that same sum*. The continuous people and the point people are using one image-formation model. The difference between "tens of hours and seconds per frame" and "rasterize it on a GPU in a millisecond" is not the compositing math — it's the representation underneath and the algorithm that produces the ordered list of contributions. That reframes the whole problem for me. I don't need a cleverer integrator. I need a *primitive* such that (a) optimization flows through it as gently as it flows through a continuous field, and (b) a GPU can splat millions of them into an image almost for free. The continuous field is great at (a) and dies on (b) because it's implicit — to find the samples in that sum I have to query the field at many points along every ray, including through empty space, which is the entire cost. Explicit points are great at (b) and historically bad at (a): plain point splatting tears holes, aliases, and is discontinuous, and the differentiable versions that fix that lean on MVS geometry and inherit its over- and under-reconstruction, or bolt on a CNN that flickers in time.

So what primitive is differentiable-volumetric *and* explicit-rasterizable at once? Let me think about what I'd ask of it. It should have a finite, smooth footprint so I can blend it and get gradients, like the splatting people's elliptical discs. But the disc-with-a-normal model is a trap here: it assumes I can give each primitive a surface normal, and my input is a *sparse* SfM cloud — estimating normals from that is hopeless, and optimizing noisy normals is worse. I want something that needs no normals, that has a smooth analytic shape, and that projects cleanly to screen. A Gaussian. Specifically a 3D Gaussian, G(x) = exp(-1/2 (x - mu)^T Sigma^{-1} (x - mu)), centered at mu with a covariance Sigma. It's a smooth volumetric kernel — gradients flow through it like through a density field, so I get property (a). And it has a property I want to lean on hard: a Gaussian stays a Gaussian under affine maps, and it integrates along a line to a 2D Gaussian, so it projects to an analytic 2D footprint I can rasterize — property (b). No normal anywhere. The infinite-but-rapidly-decaying support means a primitive far away can still pass a little gradient to pull the optimization, while in practice it only touches a small screen region.

Should the Gaussian be round or stretched? If I force it isotropic — one scalar radius, the same in every direction — then to cover a flat wall or a thin twig I need a pile of little balls, and each ball still bleeds off the surface in the two directions I don't want. A surface is locally a flat, elongated thing; an anisotropic Gaussian, a squashed ellipsoid, can lie flat against it and stretch along it, representing the same geometry with far fewer primitives and far less off-surface bleed. So I want the full anisotropic covariance Sigma, a 3x3 matrix, not a scalar. That's the representation: a cloud of anisotropic 3D Gaussians, each with a position mu, a covariance Sigma, an opacity, and a color.

Now the trouble starts, and it's a real wall. I want to optimize Sigma by gradient descent against the photometric loss. But Sigma is a covariance — it has meaning only if it's positive semi-definite. The moment I treat its six free entries as parameters and let Adam push them around, a gradient step will hand me a matrix with a negative eigenvalue, which is not a covariance of anything; the "Gaussian" inverts, blows up, the render is garbage. Gradient descent has no idea about the PSD cone and I can't easily project it back every step. So directly optimizing the entries of Sigma is out. I need a parameterization where *every* point in parameter space maps to a valid PSD Sigma, so any step Adam takes is legal by construction. What is a covariance, geometrically? It's an ellipsoid: a set of axis lengths and an orientation. Any PSD matrix factors as Sigma = R S S^T R^T, where R is a rotation and S is a diagonal scaling — that's just the eigendecomposition wearing different clothes (S S^T holds the squared axis lengths, R holds the axes). If I store a scaling vector s (the three axis lengths) and a rotation, and *build* Sigma = R S S^T R^T from them, then no matter what the optimizer does to s and the rotation, Sigma comes out PSD. The constraint is dissolved into the parameterization instead of being fought every step. I'll store s as a 3-vector and represent the rotation by a quaternion q (four numbers, normalized to a unit quaternion to be a valid rotation), because quaternions optimize more smoothly than wrestling with the nine entries of R. To keep the axis lengths positive without another constraint, I don't optimize s itself but its logarithm and pass it through exp — exp(anything) > 0, so scales can never go negative or zero, and the multiplicative nature of exp gives smooth gradients across orders of magnitude, which I'll want because some Gaussians are tiny and some are huge. Same spirit for opacity: it must live in [0,1) to be a blending weight, so I store a raw logit and squash it through a sigmoid; the sigmoid is smooth and saturating, so the optimizer gets clean gradients and can never produce an illegal opacity. So the activations aren't decoration — each one is exactly the device that makes an unconstrained optimizer respect a constraint the quantity must obey.

Color next. A fixed RGB per Gaussian can't show a specular highlight that moves as I orbit — real surfaces change color with viewing direction. I need a cheap function on the sphere of directions per Gaussian. Spherical harmonics are the orthonormal basis for that: a handful of low-order coefficients per channel encode the directional color, and I can evaluate them per view at negligible cost. I'll use a few bands. But there's a subtlety I should anticipate: if a capture doesn't see a Gaussian from all around — an inside-out or corner capture — the high-order SH are badly under-constrained, and the optimizer can dump nonsense into them and even corrupt the zero-order term, which is the base diffuse color. So I won't turn on all the bands at once. I'll optimize only the zero-order (DC) color first, then introduce one more band every so often, letting the base color settle before the directional detail is allowed to move. That way the angularly-starved coefficients never get a chance to poison the part that matters most.

Now the half I haven't touched: turning the 3D Gaussian into a 2D footprint on screen, fast and differentiably. This is where I lean on the splatting theory for elliptical kernels. The world-space Gaussian has covariance Sigma. Under an *affine* map x -> M x + c, a Gaussian's covariance transforms as Sigma -> M Sigma M^T — clean. The trouble is that the camera projection isn't affine; perspective divide makes it nonlinear. The fix is the local affine approximation: take the projective map m(u) from camera coordinates to ray/screen coordinates and replace it, near the Gaussian's center, with the first two terms of its Taylor expansion, m(u) ≈ m(u_k) + J (u - u_k), where J = dm/du evaluated at the center is the Jacobian. For the standard pinhole projection (x = u0/u2, y = u1/u2) that Jacobian is the familiar

  J = [ 1/u2,    0,   -u0/u2^2 ;
        0,    1/u2,   -u1/u2^2 ; ... ],

the derivatives of the perspective divide. With the viewing transform W taking world to camera, the covariance in screen/ray space is then Sigma' = J W Sigma W^T J^T — affine map M = J W applied to the world covariance. That's a 3x3 matrix. But on screen I only need the 2D footprint of the splat, the shape of the ellipse the pixel grid sees. After the local ray-space transform, the EWA reduction is exactly the upper-left 2x2 block of Sigma': skip the third row and column of the 3x3, and the remaining 2x2 covariance defines the screen ellipse. So: build Sigma in world space from (exp-scale, unit-quat), push it through M = J W to get the 3x3 Sigma', drop the third row and column, and I have the 2x2 screen covariance that defines the elliptical footprint. The per-pixel opacity contribution of this splat is then this 2D Gaussian evaluated at the pixel offset, multiplied by the Gaussian's own opacity — exactly the alpha_i that goes into the compositing sum. The blending itself, front-to-back with prod (1 - alpha_j), is the same one I derived is shared with the continuous model; it's justified by the first-order Taylor exp(-x) ≈ 1 - x of the attenuation, the standard splatting approximation. So the renderer is: project each Gaussian to a 2D elliptical footprint via EWA, evaluate the footprint times opacity to get alpha per pixel, composite in depth order.

I could let autograd differentiate all of this. But I'm going to have millions of Gaussians and tens of thousands of steps, and every step renders; pushing Sigma' = J W Sigma W^T J^T and the 2x2 reduction through automatic differentiation per primitive per step is overhead I can't afford. I'll derive the gradients by hand. Let me actually do it, because it's the part the implementation has to get exactly right. Write U = J W, so Sigma' is the upper-left 2x2 of U Sigma U^T. By the chain rule dSigma'/ds = (dSigma'/dSigma)(dSigma/ds) and likewise for q. The first factor, dSigma'/dSigma, I get by differentiating the bilinear form U Sigma U^T entrywise: the (1,1) entry of Sigma' is sum_{i,j} U_{1i} Sigma_{ij} U_{1j}, so its derivative with respect to Sigma_{ij} is U_{1i} U_{1j}; collecting the four screen entries, the partial of Sigma' with respect to Sigma_{ij} is the little 2x2

  [ U_{1i}U_{1j}   U_{1i}U_{2j} ;
    U_{1j}U_{2i}   U_{2i}U_{2j} ].

For the second factor I introduce M = R S so that Sigma = R S S^T R^T = M M^T. Then dSigma/dM: differentiating M M^T, and using that Sigma and its gradient are symmetric, the shared part collapses to dSigma/dM = 2 M^T. From there it splits. For scaling, M_{ij} = sum_k R_{ik} S_{kj}, and S is diagonal, so M_{ij} = R_{ij} s_j; the derivative of M_{ij} with respect to s_k is R_{ik} if j = k and 0 otherwise — a clean sparse pattern. For rotation I need dM/dq, which means differentiating R(q). Writing the unit quaternion as real part q_r and imaginary q_i, q_j, q_k, the rotation matrix is

  R(q) = 2 [ 1/2 - (q_j^2 + q_k^2),   (q_i q_j - q_r q_k),     (q_i q_k + q_r q_j) ;
             (q_i q_j + q_r q_k),     1/2 - (q_i^2 + q_k^2),   (q_j q_k - q_r q_i) ;
             (q_i q_k - q_r q_j),     (q_j q_k + q_r q_i),     1/2 - (q_i^2 + q_j^2) ],

and since M = R S each column of R is scaled by the matching s, so dM/dq_r, dM/dq_i, dM/dq_j, dM/dq_k are the entrywise derivatives of R times the scales — e.g. dM/dq_r = 2 [ 0, -s_y q_k, s_z q_j ; s_x q_k, 0, -s_z q_i ; -s_x q_j, s_y q_i, 0 ], and similarly for the imaginary parts where the diagonal entries pick up the -2 s q_i, -2 s q_j, -2 s q_k terms from differentiating the squared imaginaries. If the optimizer stores an unconstrained raw quaternion r and I normalize q = r / ||r|| before building R, the last Jacobian is d q / d r = (I - q q^T) / ||r||, so the rotation gradient is the matrix-product chain through that normalization. Chaining dSigma'/dSigma -> dSigma/dM = 2M^T -> dM/ds and dM/dq gives me explicit gradients for scale and rotation with no autodiff in the inner loop, while the covariance remains valid by construction.

Now rendering *fast*, which is the property I refuse to give up. The naive correct way to alpha-composite splats is, for each pixel, gather the splats that cover it, sort them by depth, and blend. The per-pixel sort is the killer — it's what made earlier alpha-blended splatting slow. I want to sort once for the whole image. Tile the screen into 16x16 blocks. Cull first: keep only Gaussians whose 99%-confidence ellipsoid intersects the view frustum, and trivially reject ones at extreme positions near the camera plane where the projected 2D covariance would be numerically unstable (a guard band). For each surviving Gaussian, figure out which tiles its footprint touches and instantiate one copy per overlapped tile. Give each instance a 64-bit key whose high bits are the tile index and whose low 32 bits are the quantized view-space depth. Then a single GPU radix sort over all instances sorts them simultaneously by tile and, within a tile, by depth — one sort for the entire frame, with the depth ordering for every tile falling out at once. After the sort, the splats for any tile are a contiguous range I can find by scanning for where the high (tile) bits change. Launch one thread block per tile; the threads collaboratively load packets of Gaussians into shared memory and each pixel walks its tile's list front-to-back, accumulating color and alpha, stopping when its accumulated alpha saturates (I cut off near alpha = 1). When all pixels in a tile have saturated, the whole tile is done. This is approximate — I sort once per tile, not per pixel, so the depth order within a tile is shared across its pixels rather than re-resolved at each one — but the error shrinks as splats shrink toward pixel size, and the payoff is that every optimization step gets a fast visibility-aware render.

There's a real problem in the backward pass, though. Earlier renderers either store, per pixel, the explicit list of which splats blended into it (so the backward pass can replay them) or cap the number of front-most splats that get gradients to keep that list bounded. The cap is exactly the Pulsar-style approximation, and I distrust it: scenes have wildly varying depth complexity, so a fixed N either starves deep regions of gradient or wastes work, and a too-aggressive cap destabilizes optimization — when I imagine pushing it down to a handful of splats, the gradient each splat sees is computed as if the splats behind it didn't exist, which is a severe misattribution of error, and I'd expect the optimization to thrash. So I won't cap. But storing arbitrarily long per-pixel blend lists means dynamic memory management I also don't want. The way out: don't store the lists at all — re-traverse the *same* sorted per-tile arrays I already built for the forward pass, this time back-to-front. I keep the sorted instance array and the tile ranges from the forward pass; in the backward pass each pixel starts from the last splat that affected it and walks backward. The one thing the gradient needs that isn't free this way is the intermediate transmittance T at each splat during the forward blend. Storing the whole shrinking sequence per pixel is what I'm trying to avoid. But I don't have to: the forward pass stores only the final transmittance T_final = prod_i(1 - alpha_i), and as I walk back-to-front I undo one factor at a time with T = T / (1 - alpha_i). That reconstructs every intermediate coefficient from a single stored scalar per pixel, so I get gradients for an *unlimited* number of blended Gaussians with only constant overhead per pixel. Division has an obvious hazard — dividing by a factor too close to zero is ill-conditioned — so I skip any splat whose alpha is below 1/255 in both passes and clamp alpha from above at about 0.99 so the recovered transmittance stays well-conditioned; and I stop including a splat in the forward blend if doing so would push accumulated opacity past ~0.9999.

The loss. I'm comparing rendered image to captured image, and the obvious choice is L1 on pixels. But L1 alone tends to a blurry optimum — it's happy to average away structure that costs it little per-pixel error. SSIM, structural similarity, looks at local means, variances, and covariance over small windows, so it rewards getting edges and texture *structure* right, but on its own it doesn't pin absolute pixel values tightly. I want both, so I take a convex combination: L = (1 - lambda) * L1 + lambda * (1 - SSIM), the second term being the D-SSIM dissimilarity. I'll weight it lightly, lambda = 0.2 — enough structural pressure to kill the blur, with the L1 term carrying the pixel fidelity. So the supervision is 0.8 * L1 + 0.2 * (1 - SSIM), and there is nothing else in it: the photographs are the only signal.

But here's the gap that the gradient alone cannot close, and it's the thing that turns a clever renderer into an actual reconstruction method. Backpropagating the photometric loss adjusts the Gaussians that *exist* — it nudges their positions, reshapes their covariances, tunes color and opacity. It has no move that *creates* a Gaussian where a region is empty of them, and no move that *relieves* a region where one big Gaussian is straining to cover detail it can't. Starting from a sparse SfM cloud, vast parts of the scene have no primitive at all, and the loss will happily leave them blank because there's nothing there to push. So I need a procedure, interleaved with optimization, that adds and removes Gaussians. The question is where. Let me think about the symptom both failure modes share. An under-reconstructed region — geometry that needs covering but has too few small Gaussians — and an over-reconstructed region — geometry crammed into one large Gaussian that can't represent its detail — both have the optimizer straining to move the responsible Gaussian to reduce error it can't otherwise reduce. That strain shows up as a *large view-space positional gradient*: the loss is pulling hard on where the Gaussian sits because reshaping it isn't enough. So I'll use the magnitude of the view-space position gradient, averaged over recent iterations, as my detector. Every so often — every hundred iterations, after a warm-up — I look at Gaussians whose average view-space position gradient exceeds the small normalized threshold 0.0002, and I densify them. Which way I densify depends on the Gaussian's size. If it's small — its scale is below a fraction of the scene extent — it's an under-reconstruction case: the region needs *more* coverage, so I *clone* the Gaussian, making a copy at the same position and size; the next optimizer steps can separate the copy under the same view-space gradient signal, increasing the number of Gaussians and the total covered volume. If it's large, it's an over-reconstruction case: the region has enough spatial support but not enough *detail*, so I *split* it into two smaller Gaussians, sampling their new positions by using the original Gaussian itself as a probability density (so the children sit where the parent had mass) and shrinking their scale by the implementation factor 0.8 * N; with N = 2, each child scale is divided by 1.6. Clone grows volume and count; split replaces one broad primitive with smaller children that add resolution. And I prune as I go: remove Gaussians whose opacity has fallen below a small threshold (essentially transparent, contributing nothing), and remove Gaussians that have grown too large in world space or cover too big a footprint in screen space.

One more pathology to handle. Near the input cameras the optimizer can conjure floaters — Gaussians that sit close to a camera and happen to explain a training view, inflating the count without earning their keep, a local minimum the densification then feeds. To fight this, every few thousand iterations I reset all opacities to near zero. That forces the optimization to re-justify each Gaussian: the ones the scene genuinely needs have their opacity driven back up by the loss, and the ones that were only floaters stay near zero and get culled by the pruning threshold. It's a periodic amnesty that keeps the population honest and the count bounded.

Initialization matters more than it looks. I start the means at the SfM points; for each I set an isotropic initial covariance whose axis length is the average distance to its three nearest neighbors, so the initial Gaussians roughly tile the local point density; opacity starts low, around 0.1; and I schedule the position learning rate with an exponential decay so positions move freely early and settle late. SfM points carry the structure for free — they sit where there's actual scene content — and starting there rather than from random noise especially helps the background and the regions thinly seen in training, which a random start litters with floaters that the optimization then can't remove.

Let me assemble the whole thing as the loop I'd actually run. Initialize Gaussians from the SfM cloud. Then repeat: sample a training camera and its photo; render the image with the tile rasterizer (project each Gaussian's covariance to a 2D footprint via Sigma' = J W Sigma W^T J^T with the 2x2 reduction, evaluate footprint times opacity for alpha, composite front-to-back per tile after the single global sort); compute L = 0.8 L1 + 0.2 (1 - SSIM) against the photo; backpropagate (through the renderer with the explicit gradients, recovering transmittance in the back-to-front re-traversal); record the maximum image-space radii and view-space position-gradient statistics; on the refinement schedule, clone the small high-gradient Gaussians and split the large ones, prune the transparent and the oversized, and every few thousand iterations reset opacity; then step Adam over all parameters. Increase the active SH band every thousand iterations until all bands are on. Run for thirty thousand steps. That's it — and the only supervision in the entire procedure is the photometric comparison; there is no extra penalty on the Gaussians' shapes or opacities or spacing, nothing regularizing the parameters beyond what the photographs demand. The loss is the data term and nothing else.

The implementation has two pieces: a GaussianModel that holds the parameters and builds the covariance from the activated scale and normalized quaternion, and a training step that renders, takes the photometric loss, backpropagates, records the radii and gradient statistics, runs the density-control bookkeeping, and then steps the optimizer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_to_matrix(q):
    # q is unit length and ordered as (real, i, j, k).
    r, x, y, z = q.unbind(-1)
    rows = (
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z),     2 * (x * z + r * y),
        2 * (x * y + r * z),     1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y),     2 * (y * z + r * x),     1 - 2 * (x * x + y * y),
    )
    return torch.stack(rows, dim=-1).reshape(q.shape[:-1] + (3, 3))


def build_covariance(scaling, rotation):
    # Sigma = R S S^T R^T, built so it is PSD for any (scaling, rotation).
    # scaling is post-exp (positive axis lengths); rotation is a normalized quaternion.
    R = quaternion_to_matrix(F.normalize(rotation, dim=-1))   # [N,3,3]
    S = torch.diag_embed(scaling)                             # [N,3,3]
    M = R @ S                                                 # M = R S, so Sigma = M M^T
    return M @ M.transpose(1, 2)                              # [N,3,3] PSD covariance


class GaussianModel(nn.Module):
    """The scene: a cloud of anisotropic 3D Gaussians.
    Parameters are stored in unconstrained form and mapped through activations
    that guarantee the constraint each quantity must obey."""

    def __init__(self, points, colors, sh_degree=3):
        super().__init__()
        self._xyz = nn.Parameter(points)                              # means mu, from SfM points
        # isotropic init: axis = mean distance to the 3 nearest neighbors
        dist2 = torch.clamp_min(knn_mean_sq_dist(points, k=3), 1e-7)
        self._scaling = nn.Parameter(torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3))  # log-scale -> exp
        rots = torch.zeros((points.shape[0], 4), device=points.device); rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots)                           # quaternion (normalized at use)
        self._opacity = nn.Parameter(inverse_sigmoid(0.1 * torch.ones((points.shape[0], 1),
                                                                      device=points.device)))  # logit
        n_sh = (sh_degree + 1) ** 2
        features = torch.zeros((points.shape[0], 3, n_sh), device=points.device)
        features[:, :3, 0] = rgb_to_sh(colors)                        # only DC color set at start
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
        self.max_radii2D = torch.zeros((points.shape[0],), device=points.device)
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0                                     # grow one band / 1000 iters

    # activations turn unconstrained params into legal quantities (smooth gradients):
    @property
    def scaling(self):  return torch.exp(self._scaling)              # > 0 always
    @property
    def opacity(self):  return torch.sigmoid(self._opacity)          # in [0,1)
    @property
    def rotation(self): return F.normalize(self._rotation, dim=-1)    # unit quaternion
    @property
    def features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)

    def covariance(self):
        return build_covariance(self.scaling, self.rotation)

    def render(self, camera):
        # EWA project + tile-sort + front-to-back alpha-composite (CUDA rasterizer).
        # Per Gaussian: Sigma' = J W Sigma W^T J^T, keep upper-left 2x2 for the 2D footprint;
        # alpha = footprint(pixel) * opacity; C = sum_i c_i alpha_i prod_{j<i}(1 - alpha_j).
        return rasterize_gaussians(
            means=self._xyz, covariance=self.covariance(),
            opacity=self.opacity, sh=self.features, active_sh_degree=self.active_sh_degree,
            viewmat=camera.world_view_transform, K=camera.intrinsics,
            width=camera.image_width, height=camera.image_height,
        )


def l1_loss(x, gt):  return (x - gt).abs().mean()
# ssim(x, gt): standard 11x11 windowed SSIM


def photometric_loss(rendered, gt, lam=0.2):
    # the ENTIRE supervision: data term only, no parameter regularization
    return (1.0 - lam) * l1_loss(rendered, gt) + lam * (1.0 - ssim(rendered, gt))


def train(model, cameras, images, num_steps=30000):
    opt = torch.optim.Adam([
        {"params": [model._xyz],           "lr": 1.6e-4, "name": "xyz"},     # exp-decayed
        {"params": [model._features_dc],   "lr": 2.5e-3, "name": "f_dc"},
        {"params": [model._features_rest], "lr": 2.5e-3 / 20.0, "name": "f_rest"},
        {"params": [model._opacity],       "lr": 2.5e-2, "name": "opacity"},
        {"params": [model._scaling],       "lr": 5e-3, "name": "scaling"},
        {"params": [model._rotation],      "lr": 1e-3, "name": "rotation"},
    ], eps=1e-15)

    xyz_grad_accum = torch.zeros((model._xyz.shape[0], 1), device=model._xyz.device)
    denom = torch.zeros_like(xyz_grad_accum)
    scene_extent = cameras.extent()

    for step in range(num_steps):
        if step > 0 and step % 1000 == 0 and model.active_sh_degree < model.max_sh_degree:
            model.active_sh_degree += 1                              # introduce one SH band

        cam, gt = sample_view(cameras, images)
        out = model.render(cam)                                      # rendered image + view-space pts
        image, viewspace_points, visibility, radii = out

        loss = photometric_loss(image, gt, lam=0.2)                 # 0.8*L1 + 0.2*(1-SSIM)
        loss.backward()

        with torch.no_grad():
            # adaptive density control: the moves the photometric gradient cannot make
            if step < DENSIFY_UNTIL:
                # accumulate average view-space positional gradient magnitude per Gaussian
                model.max_radii2D[visibility] = torch.maximum(model.max_radii2D[visibility], radii[visibility])
                xyz_grad_accum[visibility] += viewspace_points.grad[visibility, :2].norm(dim=-1, keepdim=True)
                denom[visibility] += 1
                if step > DENSIFY_FROM and step % 100 == 0:          # densify every 100 iters
                    grads = (xyz_grad_accum / denom.clamp_min(1)).squeeze()
                    densify_and_clone(model, opt, grads, tau_pos=0.0002, extent=scene_extent)  # small -> clone
                    densify_and_split(model, opt, grads, tau_pos=0.0002, extent=scene_extent)  # large -> split
                    max_screen = 20 if step > 3000 else None
                    prune(model, opt, min_opacity=0.005, extent=scene_extent, max_screen=max_screen)
                    xyz_grad_accum = torch.zeros((model._xyz.shape[0], 1), device=image.device)
                    denom = torch.zeros_like(xyz_grad_accum)
                if step > 0 and step % 3000 == 0:                    # periodic opacity reset
                    reset_opacity(model, opt, value=0.01)

            opt.step()
            opt.zero_grad(set_to_none=True)


def densify_and_clone(model, opt, grads, tau_pos, extent, percent_dense=0.01):
    sel = (grads >= tau_pos) & (model.scaling.max(dim=1).values <= percent_dense * extent)  # small + high grad
    # clone: copy the selected Gaussians at the same size and position.
    append_gaussians(model, opt, sel,
        new_xyz=model._xyz[sel],
        new_features_dc=model._features_dc[sel], new_features_rest=model._features_rest[sel],
        new_opacity=model._opacity[sel], new_scaling=model._scaling[sel],
        new_rotation=model._rotation[sel])

def densify_and_split(model, opt, grads, tau_pos, extent, percent_dense=0.01, N=2):
    sel = (grads >= tau_pos) & (model.scaling.max(dim=1).values > percent_dense * extent)   # large + high grad
    stds = model.scaling[sel].repeat(N, 1)
    samples = torch.normal(torch.zeros_like(stds), stds)                                    # sample using the
    R = quaternion_to_matrix(model.rotation[sel]).repeat(N, 1, 1)                           #   Gaussian as a PDF
    new_xyz = (R @ samples.unsqueeze(-1)).squeeze(-1) + model._xyz[sel].repeat(N, 1)
    new_scaling = torch.log(model.scaling[sel].repeat(N, 1) / (0.8 * N))                    # divide scale by 1.6
    append_gaussians(model, opt, sel,
        new_xyz=new_xyz,
        new_features_dc=model._features_dc[sel].repeat(N, 1, 1),
        new_features_rest=model._features_rest[sel].repeat(N, 1, 1),
        new_opacity=model._opacity[sel].repeat(N, 1),
        new_scaling=new_scaling,
        new_rotation=model._rotation[sel].repeat(N, 1))
    prune_indices(model, opt, sel)                                                          # remove the parents

def prune(model, opt, min_opacity, extent, max_screen=20):
    mask = (model.opacity < min_opacity).squeeze()                                          # essentially transparent
    if max_screen is not None:
        mask |= model.max_radii2D > max_screen                                              # too large on screen
        mask |= model.scaling.max(dim=1).values > 0.1 * extent                              # too large in world space
    prune_mask(model, opt, mask)

def reset_opacity(model, opt, value=0.01):
    new = inverse_sigmoid(torch.minimum(model.opacity, torch.full_like(model.opacity, value)))
    set_param(model, opt, "_opacity", new)                                                  # force re-justification
```

The causal chain, start to finish. I wanted SOTA quality with fast training and real-time rendering of full scenes, and saw the obstruction is the representation, not the compositing math, since the continuous and point-based families share one image-formation model. I picked the 3D anisotropic Gaussian because it is simultaneously a smooth differentiable volumetric kernel (so the photometric loss flows through it like through a density field) and an explicit primitive that projects to an analytic 2D footprint a GPU can splat (so rendering, and therefore each training step, is fast) — and it needs no normals, which a sparse SfM cloud cannot provide. Optimizing the covariance directly broke positive-semi-definiteness under gradient descent, so I dissolved that constraint by building Sigma = R S S^T R^T from an exp-activated scale and a normalized quaternion, with sigmoid opacity, so every optimizer step yields a legal Gaussian; SH coefficients grown one band at a time give view-dependent color without letting angularly-starved high-order terms corrupt the base color. I projected each Gaussian via the EWA local-affine approximation, Sigma' = J W Sigma W^T J^T with the upper-left 2x2 taken as the screen footprint, and derived the scale and rotation gradients explicitly to keep the inner loop cheap. I made rendering real-time by tiling the screen and replacing per-pixel depth sorting with a single global radix sort over (tile, depth) keys, blending front-to-back per tile until alpha saturates; and I made the backward pass handle unlimited blended Gaussians with constant per-pixel memory by re-traversing the same sorted arrays back-to-front and recovering each intermediate transmittance from T_final by dividing out one (1 - alpha) factor at a time, with the 1/255 skip and 0.99 clamp keeping that division stable. I supervised everything with a light L1 + D-SSIM combination, lambda = 0.2, because L1 alone blurs and SSIM alone under-pins pixel values. And because the photometric gradient can only adjust existing Gaussians, I added adaptive density control keyed on the view-space positional gradient — clone the small straining Gaussians to grow coverage, split the large ones with scale divided by 0.8 * N (1.6 for two children) to add detail, prune the transparent and oversized, and periodically reset opacity to dissolve floaters — initialized from the SfM points with neighbor-distance covariances. The result optimizes a cloud of anisotropic Gaussians purely against the photographs, with no penalty on their parameters of any kind: the data term is the whole objective.
