**Problem.** The no-regularizer floor (mean 28.925 PSNR; bicycle lowest at 26.641) confirms a
*missing-preference* problem: with a pure photometric loss the Gaussian fit is under-determined, and the
optimizer leaves floaters, needles, and oversized splats in the weakly-observed regions of the unbounded
outdoor scenes. The fixed `DefaultStrategy` prunes Gaussians whose activated opacity drops below ~0.005
or whose scale grows too large, but the data term never pushes anything across those thresholds.

**Key idea.** A Gaussian's pixel contribution is a **product**, `α_i(x) = o_i · exp(-½ ‖x-μ_i‖²_Σ)`:
opacity times a spatial bump whose size is set by the scales. So a Gaussian is "loud" iff it is opaque
**and** large. To reliably remove one, drive **either** factor toward zero — hence penalize **both**
opacity and extent, feeding exactly the cases the prune loop is waiting for.

- **Both factors, not one.** Opacity-only pressure leaves a big splat held just above the prune
  threshold (loud, cheap to the penalty, never pruned) and needlessly penalizes a tiny opaque speck. The
  footprint is a product, so the penalty is a sum of two terms.
- **Extent = the scale vector.** With `Σ = R S Sᵀ Rᵀ` and `R` orthogonal, the eigenvalues of `Σ` are
  `sⱼ²`, so `√eigⱼ(Σ) = sⱼ`. Penalize the three axes **additively** (not the volume `∏ⱼ sⱼ`): an
  additive per-axis term squashes a needle's long axis at full strength; a volume penalty is satisfied
  by collapsing one axis while the others stay huge.
- **L1, not L2.** In the activated value, L1's sub-gradient is the constant `sign(·)`, so pressure toward
  zero (and across the prune threshold) does not weaken as the value shrinks. L2's gradient `2w` vanishes
  at zero, leaving small-but-present clutter the strategy never prunes.
- **Penalize the activated quantities.** Act on real opacity `sigmoid(opacity) ∈ (0,1)` and real scale
  `exp(log_scale) > 0` — the values that enter the blending equation and that the prune threshold checks
  — not the raw logit/log-scale. Both activations are nonnegative, so `|·|` is a no-op and the L1
  collapses to a plain mean: cheap, smooth, `O(N)`, free of `log(0)`/NaN.

**Hyperparameters.** `SCALE_REG = OPACITY_REG = 1e-2`. Small, equal weights keep the prior gentle (the
photometric loss is ~0.03–0.1) so it bites only in the slack the photos leave; the means make it
count-independent. No schedule — on from step 0.

**Why it is the right shape here.** Opacity pressure makes the strategy's primary death channel (opacity
below the prune threshold) the common case, while scale pressure cleans up the cases opacity cannot reach
— big over-reconstructing splats — and prevents needle shapes from forming. Useful Gaussians, held up by
the photometric loss they are reducing, survive the gentle pressure.

```python
# EDITABLE region of gsplat/custom_regularizer.py — step 2: scale + opacity L1 (3DGS-MCMC compactness)
import torch
import torch.nn.functional as F

SCALE_REG = 1e-2
OPACITY_REG = 1e-2

def compute_regularizer(splats, step, scene_scale):
    """L1 penalty on per-Gaussian scale and opacity."""
    scale_loss = torch.abs(torch.exp(splats["scales"])).mean()
    opa_loss = torch.abs(torch.sigmoid(splats["opacities"])).mean()
    return SCALE_REG * scale_loss + OPACITY_REG * opa_loss
```
