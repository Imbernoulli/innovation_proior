# Context: regularizing 3D Gaussian Splatting for clean novel-view geometry (circa mid-2024)

## Research question

3D Gaussian Splatting represents a scene as millions of anisotropic 3D Gaussians and fits them
by gradient descent against a per-image photometric loss. It renders in real time and reconstructs
training views beautifully, but the per-scene optimization is badly under-constrained: nothing in
the photometric objective says what *shape* an individual Gaussian should take. The result is a
collection of pathological primitives — extremely elongated "needle" Gaussians with one variance
far larger than the other two, semi-transparent floaters, and primitives that look right from the
training cameras but fall apart from held-out viewpoints. The precise problem is to add a scalar
penalty, computed only from the Gaussian parameters themselves (no depth, normal, or feature
supervision), that pushes the optimizer toward primitives that generalize to novel views — higher
PSNR / SSIM and lower LPIPS on standard benchmarks — without degrading the visual fidelity the
photometric loss already buys. A solution has to (i) act on each Gaussian's geometry, (ii) be
differentiable so it can flow gradients through the same per-scene optimization, (iii) be cheap
enough to evaluate at every one of tens of thousands of steps over millions of primitives, and
(iv) not over-constrain — thin, legitimately slender structures still have to be representable.

## Background

**3D Gaussian Splatting.** A scene is a set of learnable primitives `{G_k}`. Each Gaussian has a
mean `mu_k in R^3`, a covariance `Sigma_k = R_k S_k S_k^T R_k^T` that is factored into a rotation
`R_k` (parameterized by a quaternion) and a diagonal scale `S_k = diag(s_k)` with
`s_k = (s_{k1}, s_{k2}, s_{k3})`, a scalar opacity `alpha_k in [0,1]`, and a view-dependent color in
spherical harmonics. The density is `G_k(x) = exp(-1/2 (x-mu_k)^T Sigma_k^{-1} (x-mu_k))`. Each
Gaussian is projected to screen space by the EWA-splatting affine approximation
`Sigma'_k = J W Sigma_k W^T J^T` (W world-to-camera, J the projection Jacobian), then the primitives
are alpha-blended front-to-back,
`c(u) = sum_k c_k alpha_k prod_{j<k} (1 - alpha_j G^{2D}_j(u))`, and the rendered image is supervised
by a photometric loss combining L1 and SSIM, e.g. `0.8 * L1 + 0.2 * (1 - SSIM)`. Because primitives
are initialized from sparse structure-from-motion points, Adaptive Density Control (ADC) clones and
splits Gaussians during optimization wherever the view-space positional gradient is large,
`|| sum_{i in P} (dL/dp_i)(dp_i/du) ||_2 > tau`: regions that are still poorly reconstructed produce
large positional gradients, so densifying there increases capacity.

**The shapes Gaussians actually take.** A covariance is fully described by its three scales (its
rotation only orients it). Looking at the diagonal `S_k S_k^T = diag(s_1^2, s_2^2, s_3^2)`, the
relative sizes of the three squared scales decide whether a Gaussian is a sphere (all comparable), a
disk (two comparable, one tiny), or a needle (one dominant, two tiny). It is well established in the
surface-reconstruction literature that for accurate geometry the density should be concentrated near
the surface (NeuS; Wang et al. 2021) — primitives that lie flat on the surface and cover area, rather
than sticking out of it. A planar, disk-like Gaussian covers a meaningful patch of surface; a needle
covers a negligible sliver and, blended over many views, shows up as a spike.

**The observed failure mode.** Across 3DGS and its surface-oriented
variants, the great majority of Gaussians converge to one dominant variance with the other two near
zero — they become needles. This is not unique to vanilla 3DGS: SuGaR (Guédon & Lepetit 2023), which
explicitly regularizes Gaussians to be flat, and 2DGS (Huang et al. 2024), which collapses the
primitive to a 2D disk and therefore *starts* every Gaussian in the planar configuration, both drift
back toward a large population of needle-like primitives. Tracking the scene over training, the count
of these degenerate primitives keeps rising while PSNR and Chamfer distance plateau — the optimizer is
overfitting the training views with spikes that buy no real reconstruction gain. There are
identifiable mechanical reasons the needles form: a screen-space dilation that adds a small constant
to the projected covariance to guarantee a minimum on-screen size, combined with an implicit
shrinkage bias, leads the optimizer to underestimate scale; and ADC, which splits along directions of
large positional gradient, gets little gradient signal along a Gaussian's *long* axis (moving along
it barely changes pixels) and a lot along the short axis, so it adjusts scales into needles rather
than splitting the long axis — and a split preserves the scales, so existing needles are never
shortened.

**A real-valued measure of "how many axes matter."** Integer matrix rank gives the right labels only
in exact degenerate limits — rank 3 for a sphere, rank 2 for a disk, rank 1 for a needle. It is the
wrong optimization signal because it is discrete and non-differentiable, and with small positive
scales both near-disks and near-needles remain full-rank covariances with no useful gradient. There
is a continuous, differentiable generalization from signal processing: the effective rank (Roy &
Vetterli, EUSIPCO 2007, "The effective rank: a measure of effective dimensionality"). For a matrix
with singular values `sigma_i`, normalize them into a distribution `q_i = sigma_i / ||sigma||_1` and
define `erank = exp(H(q))` where `H` is the Shannon entropy `-sum_i q_i log q_i`. It is real-valued,
differentiable in the singular values, and reads off how spread the spectral energy is across
directions — exactly the "how many axes carry the shape" question.

## Baselines

**3DGS (Kerbl et al., SIGGRAPH 2023).** The base method above: photometric loss only, ADC for
densification. *Limitation:* the loss constrains rendered pixels, not primitive shape, so the
optimizer is free to manufacture needles and floaters that fit the training cameras; held-out views
expose them as spikes and the geometry (normals, surface coverage) is poor.

**SuGaR (Guédon & Lepetit 2023).** Adds an SDF-based regularization to align Gaussians with a surface
and flatten them, plus Poisson mesh extraction. *Limitation:* flattening pushes one scale small, but
a Gaussian with one small scale can be either a disk *or* a needle — flatness alone does not
distinguish them, and in practice many primitives still collapse to needles.

**2DGS (Huang et al. 2024).** Replaces the 3D primitive with a 2D Gaussian disk, so each primitive is
planar by construction and rasterized with a view-consistent 2D homography. *Limitation:* even
starting every primitive at the planar configuration, the optimization drifts the disks toward
needle-like 2D Gaussians (one in-plane axis collapses), so the degeneracy reappears; it also changes
the primitive type rather than acting as a drop-in penalty on an existing model.

**Per-Gaussian compactness L1 (3DGS-MCMC; Kheradmand et al., NeurIPS 2024).** Recasts the Gaussians
as MCMC samples and, to encourage parsimonious use of primitives, adds an L1 penalty on the quantities
that define a Gaussian's extent — its opacity and its scale — to the training loss:
`... + lambda_o sum_i |o_i| + lambda_s sum_i sqrt(eig(Sigma_i))`. In an implementation that optimizes
log-scales and logit-opacities, this is L1 on `exp(scale)` and on `sigmoid(opacity)`, with a small
weight on each (`1e-2` in the compactness slot). It shrinks unneeded Gaussians toward disappearance,
keeping the set compact and prunable. *Limitation:* it controls *size and sparsity* (how big, how many), not
*shape* — it will happily shrink a Gaussian while leaving it a needle, because penalizing total extent
says nothing about the ratio of the three axes.

**Aspect-ratio / anisotropy penalties.** Bound `max(scale)/min(scale)` to keep primitives from
becoming too elongated. *Limitation:* a hard ratio cap is a blunt, two-axis comparison; it cannot tell
a disk (which legitimately has one small axis) apart from a needle, and tends to over-regularize thin
structures that should stay slender.

## Evaluation settings

The yardsticks already in use for this problem:

- **Mip-NeRF 360 (Barron et al. 2022):** 9 unbounded indoor/outdoor scenes at ~`1600x1050`,
  used for novel-view synthesis. Every 8th image is held out for testing; the rest train. Metrics:
  PSNR and SSIM (higher better), LPIPS (lower better), measured on the held-out views.
- **DTU (Jensen et al. 2014):** 15 forward-facing bounded scenes, images downsampled to `800x600`
  following common practice, used for both geometry reconstruction (Chamfer distance against the
  ground-truth scan, after TSDF-fusion mesh extraction with Open3D) and novel-view PSNR.
- **Protocol:** COLMAP structure-from-motion initializes the point cloud; each scene is trained
  per-scene for a fixed budget (on the order of 30k steps) under a fixed schedule with spherical
  harmonics degree raised gradually; renderer, optimizer (Adam/AdamW with per-parameter rates),
  photometric loss `0.8 L1 + 0.2 (1 - SSIM)`, and densification strategy are held fixed so that the
  penalty term is the only thing that varies. A regularizer is meant to drop in on top of any of the
  baseline pipelines as an add-on.

## Code framework

The penalty plugs into an existing per-scene splatting trainer. Everything around the slot already
exists: a CUDA rasterizer that renders the current Gaussians, the photometric loss, an AdamW
optimizer with per-parameter learning rates, and the ADC densification strategy — all fixed. The
Gaussians live in a parameter dictionary whose first dimension is the number of primitives `N`:
`scales [N,3]` are *log*-scales (apply `exp`), `opacities [N]` are *logits* (apply `sigmoid`),
`quats [N,4]` an unnormalized rotation quaternion, `means [N,3]` world positions, and `sh0`/`shN`
the spherical-harmonic colors. The trainer calls one function every step, adds its scalar return
value to the photometric loss with no extra scaling, and backpropagates through it. The single empty
slot is *what that scalar should be* — a per-Gaussian penalty, derived from these parameters,
costing at most `O(N)` (no all-pairs computation) and numerically safe (no `log(0)`, no `exp` of a
large number, no divide-by-zero in the backward pass). The `step` argument is available so the
penalty can be scheduled over training.

```python
import torch


# scales [N,3] are log-scales -> exp(...) for actual scale
# opacities [N] are logits     -> sigmoid(...) for [0,1] opacity
# means [N,3], quats [N,4], sh0 [N,1,3], shN [N,K,3]

def compute_regularizer(splats, step, scene_scale):
    """Scalar penalty added to the photometric loss every step.

    Computed only from the Gaussian parameters; O(N); numerically safe;
    pre-multiplies its own weights (added to the loss unscaled).

    TODO: the per-Gaussian penalty we will design.
    """
    # placeholder: no penalty
    return torch.zeros((), device=splats["means"].device)


# existing per-scene training loop the penalty plugs into (fixed):
def train(splats, optimizer, rasterizer, cameras, max_steps):
    for step in range(max_steps):
        cam = cameras.sample()
        rendered = rasterizer(splats, cam)              # fixed CUDA renderer
        gt = cam.image
        loss = 0.8 * l1(rendered, gt) + 0.2 * (1.0 - ssim(rendered, gt))   # fixed photometric loss
        loss = loss + compute_regularizer(splats, step, cam.scene_scale)   # add penalty, unscaled
        optimizer.zero_grad()
        loss.backward()                                 # flows through the penalty too
        optimizer.step()
        densify_and_prune(splats, step)                 # fixed ADC strategy
```
