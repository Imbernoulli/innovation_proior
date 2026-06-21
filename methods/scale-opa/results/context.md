# Context: regularizing a 3D-Gaussian scene representation (circa 2023–2024)

## Research question

A scene is represented by a large set of anisotropic 3D Gaussians and fit to a handful of
posed photographs by minimizing a per-scene photometric loss. The loss is purely a *data*
term: it rewards any configuration of Gaussians that reproduces the training images, and is
otherwise blind to how that configuration is built. The open question is how to add a scalar
regularization term to this photometric loss that encodes a preference for parsimonious,
efficient use of Gaussians — computationally affordable at every step of a 30k-step
optimization and numerically safe under autodiff.

## Background

**The 3D-Gaussian scene representation (Kerbl et al. 2023).** A scene is a set of `N`
anisotropic 3D Gaussians. Each Gaussian `i` carries a world-space mean `μ_i`, an opacity, a
covariance `Σ_i`, and view-dependent color stored as spherical-harmonic (SH) coefficients.
The covariance is not optimized directly — a covariance must stay positive semi-definite, and
unconstrained gradient steps would break that — so it is factored as

```
Σ = R S S^T R^T ,
```

with `S = diag(s_1, s_2, s_3)` a per-axis scale and `R` a rotation, stored as a 3-vector `s`
and a quaternion `q`. Because `R` is orthogonal, `Σ = R diag(s_1^2, s_2^2, s_3^2) R^T` is
already the eigendecomposition of `Σ`: its eigenvalues are `s_j^2`, the variances along the
principal axes, and `sqrt(eig_j(Σ)) = s_j` is the standard deviation — the physical extent —
along axis `j`. Two activations keep the parameters in their valid ranges with smooth
gradients: opacity is `sigmoid` of a stored logit, so it lives in `(0, 1)`; scale is `exp` of
a stored log-scale, so it stays strictly positive. To render a pixel `x` from camera pose `θ`,
the Gaussians are depth-sorted and `α`-blended front to back,

```
C(x) = Σ_i c_i · a_i(x) · Π_{j<i} (1 - a_j(x)) ,
a_i(x) = o_i · exp( -1/2 · (x - proj(μ_i))^T proj_θ(Σ_i)^{-1} (x - proj(μ_i)) ) ,
```

where `o_i` is the opacity and `c_i` the SH color evaluated at `θ`. The whole scene is fit by
gradient descent on a photometric loss against the training images,

```
L_photo = (1 - λ) · L1(render, gt) + λ · (1 - SSIM(render, gt)) ,   λ = 0.2 .
```

Read off the blending equation what determines how much a single Gaussian *shows up* in an
image: the contribution
`a_i(x) = o_i · exp(-1/2 · (x-proj(μ_i))^T proj_θ(Σ_i)^(-1) (x-proj(μ_i)))` is the product of
two factors — the opacity `o_i ∈ (0,1)` and a spatial bump whose width is induced by the
scales `s_j`. The integrated mass of the underlying 3D bump grows with the Gaussian volume,
which is proportional to `sqrt(det Σ) = ∏_j s_j`. So a Gaussian's visible footprint is governed
jointly by its opacity and its scale: it is loud if it is opaque *and* large.

**Observed configurations in unregularized photometric fits.** The data term alone leaves
the configuration under-determined, and the observed configurations of plain 3D-Gaussian fits
include: (a) *Floaters*: faint, low-opacity Gaussians left hovering in regions the training
cameras barely constrain (e.g. just outside the view frustum). (b) *Needles*: extremely
anisotropic Gaussians, one axis scale far larger than the others. (c) *Over-reconstruction*:
a single oversized splat covering small-scale geometry. The number of Gaussians directly sets
the memory and the per-frame render time.

**Sparsity-inducing penalties.** A standard tool for pushing many values to zero is the L1
penalty `Σ |w|`. Its defining property versus the L2 penalty `Σ w^2` is the behavior of the
gradient in the penalized value `w` near zero: L1's subgradient is the constant `sign(w)`, so
the pressure toward zero does *not* weaken as `w` shrinks. L2's gradient `2w` instead vanishes
as `w → 0`, so it only shrinks values softly and leaves a cloud of small-but-nonzero
parameters. When the goal is to *eliminate* components rather than merely shrink them, L1 is
the standard choice; if `w` is produced by a positive activation, this statement applies to the
activated value, while the stored unconstrained parameter receives the corresponding chain-rule
gradient.

**Recycling dead capacity.** The training framework these Gaussians live in periodically
identifies "dead" Gaussians — those whose opacity has fallen below a small pruning threshold
(`o < 0.005`) — and recycles them: it relocates each dead Gaussian to where a useful "live"
Gaussian already sits, sampling targets with probability proportional to opacity, and lets the
count of live Gaussians grow gradually toward a budget.

## Baselines

These are the prior approaches a new regularizer is measured against and reacts to.

**Adaptive density control via hand-engineered heuristics (Kerbl et al. 2023).** The original
way to manage *how many* Gaussians exist and *where* they go is a set of interleaved rules run
every 100 iterations. *Clone*: a Gaussian with large view-space positional gradient in an
under-reconstructed region is duplicated (a same-size copy offset along the gradient). *Split*:
an oversized Gaussian in a high-variance ("over-reconstructed") region is replaced by two
Gaussians with scale divided by an experimentally chosen factor `φ = 1.6`, positioned by
sampling the original Gaussian as a PDF. *Prune*: any Gaussian with opacity below a threshold
`ε_α` is deleted. *Opacity reset*: every `N = 3000` iterations every Gaussian's opacity is
forced near zero, and the optimizer then re-raises opacity only where the data term demands
it. Core idea: grow and shape the Gaussian set by local geometric rules keyed off the
positional gradient, using hand-tuned thresholds — the gradient threshold `τ_pos = 0.0002`,
the split factor `1.6`, the prune threshold, and the 3000-iteration reset cadence.

**Anisotropy / aspect-ratio penalties.** A different line bounds `max(s)/min(s)` per Gaussian
to keep shapes close to isotropic. Core idea: penalize the *ratio* of the largest to smallest
scale.

**Neighbour-consistency / blob-prior penalties.** Encourage spatially adjacent Gaussians to
have similar parameters, smoothing the field.

**Opacity-only pruning pressure.** One could simply push opacity down (or rely on the reset)
so low-opacity Gaussians fall below the prune threshold. Core idea: a Gaussian that is faint
enough is removed.

## Evaluation settings

The natural yardsticks already in use for novel-view synthesis from posed images:

- **Mip-NeRF 360 (Barron et al. 2022)**: unbounded real indoor/outdoor scenes; every 8th image
  held out for testing; the standard benchmark for this regime. Indoor scenes downsampled by a
  factor of two, outdoor by four, following common practice. (Other standard scene sets in the
  field — NeRF Synthetic / Blender, Tanks & Temples, Deep Blending — share the same protocol.)
- **Metrics**: PSNR (higher is better; the primary metric), SSIM (higher is better), LPIPS
  (lower is better, a learned perceptual distance). All computed on the held-out views.
- **Protocol**: each scene trained for 30,000 steps under a fixed schedule; SH color degree
  increased gradually (one band added roughly every 1000 iterations until full degree); the
  renderer (a CUDA `α`-blending rasterizer), the optimizer (Adam-family with per-parameter
  learning rates), the photometric loss `0.8 · L1 + 0.2 · (1 - SSIM)`, and the
  density-management strategy are held fixed across all comparisons. Runs are repeated and
  averaged to account for randomness.

## Code framework

The regularizer plugs into an existing per-scene training loop. The renderer, the optimizer,
the photometric loss, and the Gaussian parameter container all already exist; what does not
yet exist is the scalar term that encodes the parsimony preference. So the substrate is just
the generic harness, with one empty slot: a function that, given the current Gaussian
parameters, returns a scalar to add to the photometric loss every step.

```python
import torch

# Per-Gaussian parameters (first dim is N, the number of Gaussians). These already exist:
#   splats["means"]     [N, 3]    world-space positions
#   splats["scales"]    [N, 3]    LOG-scales; torch.exp(...) gives the actual per-axis scales
#   splats["quats"]     [N, 4]    rotation quaternion (unnormalized)
#   splats["opacities"] [N]       LOGITS; torch.sigmoid(...) gives opacity in (0, 1)
#   splats["sh0"], splats["shN"]  spherical-harmonic color coefficients


def compute_regularizer(splats, step, scene_scale):
    """Scalar term added directly to the photometric loss at every training step.

    splats      : ParameterDict of the per-Gaussian tensors described above.
    step        : current iteration (0 .. max_steps-1); available for scheduling.
    scene_scale : approximate scene radius, for distance normalization if needed.

    Must return a scalar torch.Tensor. It is added to the loss with NO extra scaling,
    so it must carry its own weight(s). Keep it O(N) (no N x N over means) and
    numerically safe (no log(0), no exp overflow, no divide-by-zero).
    """
    # TODO: the scalar penalty we will design.
    pass


# existing per-scene training loop the regularizer plugs into
def train(splats, render_fn, optimizer, train_views, max_steps, scene_scale):
    for step in range(max_steps):
        view, gt = next(train_views)
        render = render_fn(splats, view)                       # CUDA alpha-blending rasterizer
        loss_photo = 0.8 * l1(render, gt) + 0.2 * (1 - ssim(render, gt))   # fixed data term
        loss = loss_photo + compute_regularizer(splats, step, scene_scale)
        optimizer.zero_grad()
        loss.backward()                                        # grads flow through every op
        optimizer.step()                                       # per-parameter update
```

The data term, renderer, optimizer, and parameter layout are fixed; the single empty slot is
the body of `compute_regularizer`.
