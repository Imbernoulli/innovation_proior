## Research question

A 3D Gaussian Splatting (3DGS) scene is a cloud of anisotropic 3D Gaussians, each carrying a
position, a covariance (factored into scale and rotation), an opacity, and view-dependent
color, all optimized by differentiable rasterization against multi-view photographs. The
representation does not start dense enough to capture fine geometry: it is seeded from a
sparse Structure-from-Motion point cloud and must *grow* during training. That growth is the
job of **adaptive density control (ADC)** — the rule that decides, every hundred or so
iterations, which Gaussians to add (clone, split), which to remove (prune), and when to
reset opacities. Crucially, ADC is *interleaved* with the gradient-descent optimization of
the Gaussian parameters: every densification step perturbs a representation that the
optimizer has already partly fitted, and then optimization has to recover from the
perturbation before it can make further progress.

The precise problem is to make that growth rule do its job without fighting the optimizer.
Concretely, a good density-control rule must decide when a Gaussian needs more local
capacity, add that capacity without handing gradient descent a large discontinuity, avoid
near-duplicate Gaussians that never separate, keep the total count practical for memory and
rendering speed, and remove Gaussians whose contribution is not reliable across views. The
standard ADC achieves the coarse goal of growing the scene, but its edit operations and its
static opacity-and-size pruning leave these sub-goals only partly handled.

## Background

By 2024, 3DGS (Kerbl, Kopanas, Leimkühler & Drettakis, SIGGRAPH 2023) is the dominant
real-time novel-view-synthesis representation, having displaced the slow volumetric MLPs of
NeRF (Mildenhall et al. 2020) for interactive rendering. It is explicit and differentiable:
a scene is N Gaussians

```
G(x) = exp( -1/2 (x - mu)^T Sigma^{-1} (x - mu) ),   Sigma = R S S^T R^T,
```

with S a diagonal scale (a 3-vector s) and R a rotation (from a quaternion q); each Gaussian
also has an opacity (stored as a logit, activated by sigmoid) and spherical-harmonic color.
An image is formed by projecting the Gaussians to 2D and alpha-compositing them front to back,

```
C = sum_i  c_i alpha_i  prod_{j<i} (1 - alpha_j),
```

so a near Gaussian with high alpha attenuates everything behind it. All parameters are
trained by Adam against a photometric loss (an L1 + D-SSIM mix) with differentiable
rasterization.

The load-bearing background concepts and the diagnostic findings about this system, all
knowable before any new rule is designed:

- **The view-space positional gradient is the densification signal.** 3DGS decides where to
  densify from the gradient of the loss with respect to each Gaussian's *projected 2D mean*.
  Per view k it sums the per-pixel sub-gradients, `dL/dmu_{k,x} = sum_j dL_j/dmu_{i,x}` (and
  likewise y), forms the per-view magnitude, and averages over the views a Gaussian appears
  in: `grad_i = (1/M^i) sum_k sqrt((dL/dmu_{k,x})^2 + (dL/dmu_{k,y})^2)`. A large
  `grad_i` means "this Gaussian's footprint wants to move inconsistently to fit the image" —
  a signal that one Gaussian is being asked to cover detail it cannot represent. Gaussians
  above a threshold `tau_pos = 0.0002` are densified, every 100 iterations between iteration
  500 and 15000.

- **Densification is interleaved with optimization, so each edit is a perturbation.** The
  Gaussians' shapes are continuously driven by backprop to align with scene geometry. Any
  densification op that changes the *rendered output* at the moment it fires hands the
  optimizer a discontinuity it must spend iterations undoing. The cheaper an edit is for the
  renderer to absorb — the smaller the change in the density/color it produces along every ray
  — the faster optimization resumes.

- **Gradient collision starves the large Gaussians that most need splitting** (diagnostic
  finding, Ye et al. 2024; Yu et al. 2024). A big Gaussian over a textured region covers many
  pixels whose positional sub-gradients point in *opposing* directions; summing the signed
  sub-gradients per view, `sum_j dL_j/dmu`, lets them cancel, so the aggregated magnitude
  falls *below* the densification threshold and the blurry Gaussian is never split. The very
  Gaussians responsible for over-reconstruction are the ones the signal hides.

- **Opacity reset and its purpose** (diagnostic, from 3DGS and follow-ups). Every 3000
  iterations 3DGS clamps every opacity down to a small value (alpha <- min(alpha, 0.01)).
  The stated reason: floaters that accumulate near the cameras saturate ray transmittance
  early, so gradients cannot flow to the occluded parts of the scene; lowering all opacities
  lets gradients flow again and lets Gaussians that never re-earn opacity be pruned. After a
  reset the optimizer *re-grows* the opacities of the Gaussians it still needs.

- **Pruning is opacity-and-size based.** 3DGS removes Gaussians with opacity below ~0.005 and
  Gaussians that have grown too large in screen space or world space (> 0.1 * scene extent).

- **Size converges to a single scale under optimization** (diagnostic observation). With the
  shape parameters under continuous gradient descent, a Gaussian's size tends to settle at one
  value that trades off under- vs over-reconstruction to minimize the loss, whether the region
  was initially over- or under-fit.

- **Overfit Gaussians are real and survive opacity pruning** (diagnostic observation). Some
  Gaussians improve a few training views while harming others; at steady state they hold a
  moderate opacity, so an opacity-threshold prune cannot distinguish them from genuinely
  useful Gaussians. Any pruning rule that only looks at final opacity therefore misses a
  class of view-inconsistent primitives.

## Baselines

The prior art a new density-control rule would be measured against and reacts to.

**3DGS adaptive density control (Kerbl et al. 2023).** The reference rule. Gaussians above
the gradient threshold are densified by one of two operations chosen by size. *Clone* (for
small Gaussians, max scale below `percent_dense * extent`): copy the Gaussian at the same
position with identical parameters; the copy receives no gradient during the cloning
iteration, so the two separate only through the parent's parameter update that step. *Split*
(for large Gaussians): replace the parent with two children, each scale divided by 1.6, each
keeping the parent's shape, opacity, and color, and each placed at the parent's center plus a
sample drawn from the parent's own covariance (a draw from `N(0, Sigma)` rotated into world
frame). **Gaps:** clone's two copies only separate if the parent happens to take a large step
that iteration; when the step is small the copies stay nearly coincident, receive almost
identical gradients thereafter, and can never be individually optimized — the
under-reconstruction persists. Split changes the rendered scene at the moment it fires: two
children that keep the *full* parent shape cannot tile the parent's footprint without
overshooting it, so the covered shape differs before and after the edit; the probabilistic
sampling of child positions injects run-to-run variance; and two children at the parent's
opacity overcompound the density along a ray that passes through both.

**Homodirectional / absolute-value gradient (Ye et al. 2024, AbsGS; Yu et al. 2024, GOF).**
Cure gradient collision by taking the magnitude of each per-pixel sub-gradient *before*
summing over pixels: `ghat_{i,x} = sum_j |dL_j/dmu_{i,x}|`, then combine x and y. Now opposing
per-pixel pulls add instead of cancel, so over-reconstructed Gaussians clear the threshold and
get split. Because the aggregated magnitudes are larger, the densification threshold is raised
(e.g. 0.0002 -> 0.0004). **Gap:** this fixes *which* Gaussians are selected for densification;
it does nothing about *how* the selected Gaussian is subdivided — the clone/split operations,
with all their perturbation problems, are untouched.

**Budgeted score-based densification (Mallick et al. 2024, TamingGS).** Replace the fixed
threshold with an explicit Gaussian-count budget and a per-step growth schedule that
interpolates the count from the SfM seed toward a target, and choose which Gaussians to
densify from a weighted multi-feature saliency score (gradient blended with coverage, depth,
opacity, scale, blending weight, and per-pixel saliency) rather than the gradient alone. A
common reduced form blends the *average* positional gradient with the per-Gaussian *maximum*
gradient over the interval, which keeps a Gaussian that spikes strongly in even one view.
**Gap:** again this improves *when and which* to densify and how many; the underlying clone
and split *operations* are inherited unchanged.

**Revised split opacity (Rota Bulo et al. 2024, RevisingGS).** Observes that turning one
Gaussian of opacity alpha into two of the same alpha biases compositing: two stacked children
transmit `(1 - alpha)^2`, not the parent's `(1 - alpha)`. Solving `(1 - alpha) = (1 -
alphahat)^2` gives the corrected child opacity `alphahat = 1 - sqrt(1 - alpha)`, which keeps
the through-ray transmittance invariant when both children lie on a ray. **Gap:** corrects
the opacity for the *transmittance* of two stacked children, but is derived for the standard
covariance-sampled split and leaves the child *positions and shapes* — and hence the change in
the spatial density profile — unaddressed.

**Deterministic principal-axis placement (Chen et al. 2024, VCR-GauS).** Observes that
children sampled from the same parent covariance stay clustered (same distribution, similar
positions), causing surface protrusions. Replaces the random sample with deterministic
placement along the *longest* principal axis, the two children evenly dividing the parent's
maximum scale. **Gap:** deterministic placement removes the clustering and the sampling
variance, but the children keep the parent's full shape and opacity, so the covered shape and
the density mass still jump at the edit.

Across all of these, one fault line stays open: prior work improves *which* Gaussians to
densify, *how many*, and one-off corrections such as opacity adjustment or deterministic
placement, but the edit itself still changes the optimized density/color field in ways the
optimizer must repair. The pruning side has a parallel gap: static opacity, size, and
importance scores are weak signals for Gaussians whose contribution is inconsistent across
views.

## Evaluation settings

The natural yardsticks already in use for 3DGS-family density control, all predating any new
rule:

- **Mip-NeRF 360** (Barron et al. 2022): nine real unbounded scenes, five outdoor and four
  indoor, the standard hard benchmark for detail and far-field reconstruction.
- **Tanks & Temples** (Knapitsch et al. 2017): the `train` and `truck` scenes.
- **Deep Blending** (Hedman et al. 2018): the `drjohnson` and `playroom` scenes.
- **Protocol.** Every 8th image is held out for testing (as in 3DGS). Training is 30,000
  iterations per scene; densification runs in an early window and stops well before the end.
  Renderer (CUDA rasterizer), optimizer (Adam with per-parameter learning rates), and the
  L1 + D-SSIM photometric loss are held fixed; only the density-control rule varies.
- **Metrics.** PSNR (primary), SSIM, and LPIPS (Zhang et al. 2018), plus the final Gaussian
  count, training time, and rendering FPS as efficiency measures. Count budgets can be fixed
  (via a TamingGS-style cap) to isolate the effect of the *operation* from the effect of how
  many Gaussians are grown.

## Code framework

The rule plugs into an existing 3DGS training harness as a density-control strategy object.
The renderer, the Adam optimizer, the photometric loss, and the training loop already exist
and are fixed; the strategy owns its own running statistics and is called by the loop through
two hooks — one before the backward pass, to retain the screen-space gradient it needs, and
one after the backward pass, where it may edit the Gaussian set. The primitive edit
operations on the Gaussian set — duplicate a masked subset, split a masked subset, remove a
masked subset, clamp all opacities to a value — already exist as library calls that correctly
resize the parameters *and* their Adam state in lockstep. What does not exist is the policy
inside the post-backward hook. That is the single empty slot.

```python
from dataclasses import dataclass
from typing import Any, Dict
import torch

# Existing library edit primitives (resize params AND their optimizer/Adam state together):
#   duplicate(params, optimizers, state, mask)            # copy the masked Gaussians
#   split(params, optimizers, state, mask)                # covariance-sampled split (baseline)
#   remove(params, optimizers, state, mask)               # delete the masked Gaussians
#   reset_opa(params, optimizers, state, value)           # clamp all opacities down to `value`
from gsplat.strategy.ops import duplicate, split, remove, reset_opa


@dataclass
class Strategy:
    """Base class: a density-control rule called by the fixed training loop."""
    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        ...
    def step_pre_backward(self, params, optimizers, state, step, info):
        ...
    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        ...


@dataclass
class CustomStrategy(Strategy):
    """The density-control rule to design.

    Gaussian parameters in `params`:
      means [N,3], scales [N,3] (log-scale; exp for true scale),
      quats [N,4], opacities [N] (logit; sigmoid for true opacity), sh0/shN colors.
    Per-step `info` from the rasterizer: means2d (with .grad / .absgrad after backward),
      width, height, n_cameras, radii, gaussian_ids.
    """

    # densification cadence / thresholds the loop already understands
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_opa: float = 0.005

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        # running per-Gaussian gradient statistics used to decide densification
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info):
        info["means2d"].retain_grad()  # we need the screen-space mean gradient after backward

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # accumulate the densification signal from this step's screen-space gradient,
        # then, on a refine step, apply the density-control policy.
        # TODO: density-control policy.
        pass
```

The post-backward hook is the policy slot; everything around it — the renderer, the
optimizer, the loss, the loop, and the four edit primitives — is already in place.
