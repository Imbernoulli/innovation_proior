## Research question

A static scene has been captured as photographs with cameras calibrated by Structure-from-Motion, and
the scene is fit as an unstructured cloud of `N` anisotropic 3D Gaussians — means, log-scales, rotation
quaternions, logit-opacities, spherical-harmonic colours — optimized per scene by gradient descent
against a photometric loss alone. That loss is a pure data term, so the fit is wildly under-determined:
enormously many Gaussian configurations render the *training* views identically, and the optimizer
settles into wasteful ones — semi-transparent floaters in regions no camera constrains, extremely
elongated "needle" Gaussians that read as spikes off the surface the moment the camera moves, and
oversized splats smeared over fine detail. None of these hurt the training views; that is exactly why
they survive, and exactly why they hurt the *held-out* novel views, which is what is actually scored.
The single thing being designed is a **scalar regularizer on the Gaussian parameters** — added directly
to the photometric loss at every step — that expresses the preference the data term lacks. Everything
else about the optimization is fixed. No depth, normal, or feature-level supervision is permitted: the
regularizer sees only the Gaussian parameters.

## Prior art before the first rung (what the data term alone gives, and where it fails)

The first rung is the bare optimizer the ladder reacts to; the lineage below is what it inherits and
where each piece leaves a gap that a regularizer is asked to close.

- **3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023).** Both continuous radiance fields and
  point-based renderers composite a pixel by the *same* front-to-back sum
  `C = Σ_i c_i α_i Π_{j<i}(1−α_j)`; the cost difference is the representation, not the math. Choosing
  the anisotropic 3D Gaussian — volumetric-and-differentiable yet projecting to an analytic 2D
  footprint a GPU can splat — gives state-of-the-art quality, fast per-scene optimization, and
  real-time rendering. Covariance is kept valid by construction via `Σ = R S Sᵀ Rᵀ` from an
  `exp`-activated scale and a normalized quaternion, opacity via `sigmoid`. **Gap:** the supervision is
  the photometric loss and nothing else — there is no penalty on the Gaussians' shapes, sizes, or
  opacities, so the under-determined fit drifts into floaters and needles that the gradient cannot
  remove on its own.
- **Adaptive density control (the fixed densification strategy here).** Interleaved with optimization,
  the strategy clones small high-gradient Gaussians, splits large ones, and prunes the transparent and
  oversized — the only mechanism that adds and removes primitives, since the photometric gradient can
  only reshape the Gaussians that already exist. **Gap:** densification reacts to view-space positional
  gradient, not to whether a primitive's *shape or opacity* is wasteful; it grows and prunes the
  population but does not, by itself, stop the optimizer from re-forming needles and faint floaters in
  the slack the photos leave.
- **Mip-NeRF 360 unbounded scenes (Barron et al., 2022).** The benchmark these scenes come from —
  large outdoor and full indoor captures with background at infinity, every 8th image held out. The
  unbounded geometry is precisely where the under-constraint bites: vast regions are seen by few
  cameras, so the data term is flat there and the optimizer is free to leave junk. **Gap:** it defines
  the held-out task and the failure surface; it is not itself a fix.

## The fixed substrate

The training loop is frozen and must not be touched. The renderer is the `gsplat` CUDA rasterizer; the
optimizer is per-parameter Adam; the photometric loss is `0.8·L1 + 0.2·(1−SSIM)`; densification is
`gsplat`'s `DefaultStrategy` (original 3DGS clone/split/prune); training is 30,000 steps per scene with
SH degree 3 introduced gradually. Each step the loop renders a sampled training view, forms the
photometric loss, **adds the regularizer's scalar return unscaled** (`loss = photo_loss + reg_loss`),
backpropagates, steps Adam, and runs the densification hooks. Gaussians are initialized from the SfM
points: scales from nearest-neighbour distances, identity quaternions, opacity 0.1, SH DC from point
colour. The regularizer is the only quantity that changes.

## The editable interface

Exactly one function is editable — `compute_regularizer(splats, step, scene_scale)` in
`gsplat/custom_regularizer.py` (lines 37–51 of the template). The contract:

- `splats` — a `torch.nn.ParameterDict`, first dim `N`: `means [N,3]` (world positions); `scales [N,3]`
  (**log-scales**; `torch.exp` for actual extent); `quats [N,4]` (unnormalized quaternion);
  `opacities [N]` (**logit**; `torch.sigmoid` for `[0,1]`); `sh0 [N,1,3]`, `shN [N,K,3]` (SH colour).
- `step` — current iteration `0 .. max_steps−1`; lets the regularizer schedule itself (warmup, cooldown,
  switch-over).
- `scene_scale` — approximate scene radius for distance normalization.
- Returns a **scalar `torch.Tensor`**, added to the photometric loss with no extra scaling — so the
  regularizer must pre-multiply its own weights. Backward flows through every operation; avoid
  `log(0)`, `exp(big)`, divide-by-zero. Keep the cost at most `O(N)` (no all-pairs `N×N` over `means`),
  since this runs at every one of 30k steps. Helpers and module-level constants may be added inside the
  editable region.

The starting point is the scaffold default: **no regularization** — `compute_regularizer` returns zero,
so only the photometric loss supervises. Each rung on the ladder replaces exactly this function.

```python
# EDITABLE region of gsplat/custom_regularizer.py — default fill (no regularization)
import torch
import torch.nn.functional as F


def compute_regularizer(splats, step, scene_scale):
    """Return a scalar tensor added to the photometric loss.

    TODO: design a regularizer that improves reconstruction quality.
    Hints:
      - Parameter-level penalties (scale, opacity, SH) are cheap and often
        effective. Tune the weights per scene scale.
      - Neighbor-based priors (e.g. kNN over splats["means"]) add a small
        amount of spatial structure.
      - Keep the cost bounded: this function is called every training step.
    """
    # Default: no regularization.
    return torch.zeros((), device=splats["means"].device)
```

## Evaluation settings

Evaluation runs on Mip-NeRF 360 scenes (Barron et al., 2022), every 8th image held out for testing.
Each scene is trained for 30k steps under the fixed schedule and evaluated on the held-out views.
Reported, higher-is-better except LPIPS: **PSNR** (peak signal-to-noise ratio, the primary metric),
**SSIM** (structural similarity), **LPIPS** (learned perceptual similarity, lower is better). The
leaderboard records best PSNR per scene — **garden**, **bicycle**, **bonsai**, **stump** — at the
single seed `42`. The first three are visible during development; **stump** is held out.
