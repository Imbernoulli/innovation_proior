# Scale and opacity L1 regularizer

This compactness regularizer for a 3D-Gaussian scene representation is an L1 penalty on each
Gaussian's *activated* scale and *activated* opacity, each with a small weight, added directly
to the per-scene photometric loss. It encodes a parsimony prior — push useless Gaussians to
vanish — by pressing on the two factors that jointly set a Gaussian's visible footprint.

## Problem it solves

Fitting `N` anisotropic 3D Gaussians to posed images with a pure photometric loss is
under-determined: many configurations render the training views equally well, and gradient
descent picks wasteful ones — faint floaters in unconstrained regions, anisotropic needles,
oversized over-reconstructing splats. These pass on training views, hurt held-out novel views,
and inflate the Gaussian count (hence memory and render time). The regularizer adds the
preference the data term lacks: *use Gaussians efficiently*, driving a Gaussian that is not
needed to disappear so the training framework's recycle/relocation loop can reclaim it.

## Key idea

A Gaussian's contribution to a pixel is a **product**,
`a_i(x) = o_i · exp(-1/2 · (x-proj(μ_i))^T proj_θ(Σ_i)^{-1} (x-proj(μ_i)))`: opacity `o_i`
times a spatial bump whose size is induced by the scales. So a Gaussian is "loud" iff it is
opaque **and** large.
To reliably remove one, you must be able to drive **either** factor to zero — hence penalize
**both** opacity and extent.

- **Both factors, not one.** Opacity-only pressure leaves a big splat held just above the dead
  threshold (loud, cheap to the penalty, never pruned) and needlessly penalizes a tiny opaque
  speck. The footprint is a product, so the penalty is a sum of two terms.
- **Extent = the scale vector.** With `Σ = R S S^T R^T` and `R` orthogonal, the eigenvalues of
  `Σ` are `s_j^2`, so `sqrt(eig_j(Σ)) = s_j`. Penalizing extent is penalizing `Σ_j s_j`.
  Penalize the three axes **additively** (not the volume `∏_j s_j`): an additive per-axis term
  squashes a needle's long axis at full strength, whereas a volume penalty is satisfied by
  collapsing one axis while keeping the others huge.
- **L1, not L2.** In the activated opacity/scale value, L1's sub-gradient is the constant
  `sign(·)`, so pressure toward zero does not weaken just because the value is already small.
  L2's gradient `2w` vanishes at zero, leaving small-but-nonzero clutter.
- **Penalize the activated quantities.** Act on real opacity `σ(opacity) ∈ (0,1)` and real
  scale `exp(log_scale) > 0` — the values that enter the blending equation — not the raw
  logit / log-scale. Both activations are nonnegative, so `|·|` is a no-op and the L1 collapses
  to a plain mean: cheap, smooth, and free of `log(0)` / NaN hazards.
- **Small weight.** The regularizer is a gentle prior that should only bite where the data term
  is flat (the slack the photos leave). Use weight `0.01` on each term.

## Final form

Per-Gaussian, with raw opacity logit `z_i` and raw log-scales `ℓ_{ij}` (axes `j = 1,2,3`),
added to the data term:

```
L_reg = 0.01 · mean_i |σ(z_i)|  +  0.01 · mean_{ij} |exp(ℓ_{ij})|
      = 0.01 · mean_i σ(z_i)    +  0.01 · mean_{ij} exp(ℓ_{ij}),
```

the second line because `σ(·) > 0` and `exp(·) > 0`. It is `O(N)` — an elementwise activation
plus a reduction, no `N × N` over the means.

## Working code

Filling the `compute_regularizer` slot of the training harness:

```python
import torch

SCALE_REG = 1e-2      # weight on the extent (scale) penalty
OPACITY_REG = 1e-2    # weight on the opacity penalty


def compute_regularizer(splats, step, scene_scale):
    """L1 parsimony prior on per-Gaussian scale and opacity.

    Footprint a(x) is a PRODUCT of opacity and a projected Gaussian bump, so press
    on opacity and extent. L1 gives steady pressure in the activated variables,
    unlike L2 near zero. Penalize the ACTIVATED quantities; both are nonnegative
    so |.| is just a mean (smooth, NaN-free). O(N).
    """
    # exp(log_scales) = actual per-axis extents s_j = sqrt(eig_j(Sigma)); shrink every axis.
    scale_loss = torch.exp(splats["scales"]).mean()
    # sigmoid(opacity_logits) = actual opacity in (0,1); fade faint useless Gaussians.
    opa_loss = torch.sigmoid(splats["opacities"]).mean()
    return SCALE_REG * scale_loss + OPACITY_REG * opa_loss
```

The same scalar can be written as the two reusable reductions called by the training loop with
both weights set to `0.01`:

```python
import torch
from torch import Tensor


def opacity_reg_loss(opacities: Tensor) -> Tensor:
    """Mean activated opacity. Penalizes high opacity to encourage transparency."""
    return torch.sigmoid(opacities).mean()


def scale_reg_loss(log_scales: Tensor) -> Tensor:
    """Mean activated scale. Penalizes large Gaussian extent."""
    return torch.exp(log_scales).mean()
```

## Why it is the right shape here

The regularizer is matched to the surrounding recycle loop: opacity pressure makes the
*primary* death channel (opacity below the prune threshold) the common case, while scale
pressure cleans up the cases opacity cannot reach — big over-reconstructing splats — and
prevents needle shapes from forming. Because both terms act on the *activated* footprint
factors and use L1, the prior produces genuinely dead Gaussians that the framework relocates to
where they help, rather than a haze of small-but-present clutter; and because the weight is
small, useful Gaussians — held up by the photometric loss they are actively reducing — are not
erased by the prior alone.
