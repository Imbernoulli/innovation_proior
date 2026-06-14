# Context: choosing a discrepancy for matching predicted to target latent feature maps

## Research question

A temporal Joint Embedding Predictive Architecture trains an encoder and a predictor
together. The encoder turns each frame of a Moving-MNIST sequence into a spatial latent
feature map; the predictor rolls forward autoregressively in time and emits a *predicted*
feature map for each future step. Training needs a scalar cost that measures how far the
predicted latent tensor is from the encoder's target latent tensor, both of shape
`[B, C, T, H, W]`, and is added to a variance–covariance anti-collapse regularizer before
backprop. The precise problem: **pick the per-coordinate discrepancy function applied to the
residual `r = target - predicted` in latent space.** The function must produce a useful
gradient signal everywhere, settle cleanly when the prediction is already close, and — the
load-bearing requirement — not be wrecked by the residuals that are *not* close. In an
autoregressive latent roll-out the residual distribution is not clean: early in training, on
hard frames, and at later roll-out steps where errors compound, a fraction of the
coordinate-wise residuals are large. The cost on those large residuals decides whether
training is stable or has to be babysat. What a good solution must achieve: behave well for
the bulk of small residuals (smooth, statistically sensible near the optimum) *and* keep the
contribution of the occasional large residual under control, with a single interpretable knob
for "how large is large."

## Background

The discrepancy between a prediction and a target is, formally, the negative log-likelihood
of a noise model on the residual. Two classical choices anchor the field, and they sit at
opposite ends of a robustness–efficiency trade-off.

**Squared error and the mean.** If the residual is assumed Gaussian, the maximum-likelihood
fit minimizes `Σ ρ(r)` with `ρ(r) = ½ r²`. Its derivative — the quantity that actually drives
the gradient — is `ψ(r) = ρ'(r) = r`, which is *unbounded*: a residual twice as large pulls
twice as hard, a residual ten times as large pulls ten times as hard, with no ceiling. For a
location estimate this is exactly the sample mean, and a single arbitrarily large observation
drags the mean arbitrarily far. The same unboundedness is the practical failure mode when the
targets/residuals can be large: the gradient produced by one big residual can dominate the
batch and, with unbounded targets, destabilize training unless the learning rate is tuned
down to compensate.

**Absolute error and the median.** If the residual is assumed Laplace (heavier tails), the
MLE minimizes `Σ |r|`, whose derivative `ψ(r) = sign(r)` is *bounded* at ±1 — every residual,
large or small, contributes a unit-magnitude pull. For a location estimate this is the sample
median, which a single gross outlier cannot move arbitrarily: it is robust. But `ψ` is
constant in magnitude all the way down to `r → 0` and has a kink there (the loss is
non-differentiable at the optimum), so near a good fit the cost keeps applying full-strength,
sign-only gradients with no quadratic basin to settle into — statistically inefficient at a
clean Gaussian center and numerically jittery near the minimum.

**The M-estimation frame.** Both are instances of an M-estimator: estimate the location by
`μ̂ = argmin_t Σ ρ(X_i − t)`, with first-order optimality `Σ ψ(X_i − t) = 0`, `ψ = ρ'`. The
asymptotic variance of such an estimator is `E[ψ²] / (E[ψ'])²`; minimizing it over `ψ`
recovers the maximum-likelihood score `ψ ∝ −f'/f` for whatever density `f` actually generated
the data (a Cauchy–Schwarz argument makes the MLE score the variance-minimizer). So the choice
of `ρ` *is* an implicit choice of noise model, and `ψ` — the **influence function** — is what
sets how much any single residual can move the answer. A bounded `ψ` is the formal statement
of robustness; an unbounded `ψ` is the formal statement of fragility.

**The motivating phenomenon: contaminated data.** Real residual distributions are rarely a
clean Gaussian. The standard model for "mostly clean, occasionally gross" is the
ε-contamination class `F = (1 − ε)Φ + εH`: a fraction `1 − ε` of residuals come from a
nominal Gaussian `Φ`, and a fraction `ε` come from an arbitrary, possibly heavy-tailed
contaminating distribution `H`. Under squared error the contaminating `ε` fraction, through
the unbounded `ψ`, controls the estimate; under absolute error the bulk `1 − ε` Gaussian
fraction is fit inefficiently. Neither pure choice is good across this class — squared error is
optimal only at `ε = 0`, absolute error only as `ε → 1`. This is the observed gap that frames
the problem: for `0 < ε < 1` the right `ρ` should be neither pure quadratic nor pure
absolute. In the latent-prediction setting the "contamination" is concrete — outlier residual
coordinates from compounding roll-out error, hard frames, and a noisy encoder target — so the
same trade-off is live, and the question is which `ρ` to put on the residual.

## Baselines

**Squared / `L2` prediction loss (current default).** `ρ(r) = ½ r²`, summed or averaged over
all coordinates of the `[B, C, T, H, W]` residual; in this harness it is `F.mse_loss(state,
predicted)`, treating every channel and spatial/temporal location identically. Core idea: MLE
under Gaussian residuals, gives the smooth quadratic basin near the optimum and the efficient
mean-like estimate when residuals are well-behaved. **Gap (observed limitation):** its
influence `ψ(r) = r` is unbounded, so a large latent residual produces a proportionally large
gradient; when the residuals can be large — early training, compounding autoregressive
roll-out, an occasional hard frame — those gradients can dominate and destabilize, forcing the
learning rate down. It does not cap the cost of the residuals that are not close.

**Absolute / `L1` prediction loss.** `ρ(r) = |r|`, `ψ(r) = sign(r)`. Core idea: MLE under
Laplace residuals, the median-like estimate, robust because its influence is bounded — one
huge residual cannot pull the fit arbitrarily. **Gap (observed limitation):** `ψ` has constant
magnitude down to zero and a kink at `r = 0`, so there is no quadratic region near the
optimum; the fit keeps receiving full-magnitude, sign-only gradients even when it is already
good, which is statistically inefficient on the Gaussian bulk and leaves the loss
non-differentiable exactly where the predictor should be settling.

These two leave a gap on opposite sides: one is smooth-and-efficient in the middle but
fragile in the tail, the other is robust in the tail but kinked-and-inefficient in the middle.
Within the M-estimation frame, where `ρ` is free to be chosen and the data live in the
ε-contamination class for some `0 < ε < 1`, neither endpoint is the right `ρ`.

## Evaluation settings

- **Dataset / task:** Moving MNIST; the JEPA encoder produces spatial latent feature maps per
  frame, the predictor unrolls autoregressively, and a detection head reads the predicted
  latents. The prediction loss is one term of the total training loss (added to a VICReg-style
  variance–covariance regularizer that prevents representation collapse).
- **Metric:** mean detection Average Precision across the predicted timesteps on Moving MNIST
  (higher is better); reported as the final mean detection AP after training.
- **Protocol:** 50 epochs, Adam optimizer at `lr = 1e-3`, fixed seed. Evaluated across three
  model sizes (small/base/large encoder, state, predictor widths) to check that the choice of
  discrepancy generalizes across capacity.
- **Interface:** the cost receives `state` (target latents) and `predicted`, both
  `[B, C, T, H, W]`, and returns a scalar that is backpropagated. The latent features carry
  spatial and temporal structure, and the residual distribution over coordinates is the object
  the discrepancy acts on.

## Code framework

The discrepancy plugs into a fixed `nn.Module` slot. Everything around it already exists: the
encoder, the autoregressive predictor's `unroll()`, the variance–covariance regularizer, and
the Adam training loop that sums the prediction term with the regularizer and backprops. The
only undecided piece is the per-coordinate function applied to the residual between the
predicted and target latent tensors. Standard tensor primitives are available
(`F.mse_loss`, `F.l1_loss`, elementwise arithmetic, reductions). The scaffold is a single
empty `forward`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Prediction cost for temporal JEPA. Maps the residual between predicted and
    target latent feature maps to a scalar that is added to the anti-collapse
    regularizer and backpropagated. Both inputs are [B, C, T, H, W]."""

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        """
        Args:
            state:     [B, C, T, H, W] target encoded representations.
            predicted: [B, C, T, H, W] predicted representations.
        Returns:
            scalar loss (lower = predicted closer to state).
        """
        # residual the cost will act on:
        #     r = state - predicted
        # TODO: the per-coordinate discrepancy we will design, reduced to a scalar.
        pass


# existing JEPA training step the cost plugs into (fixed)
def jepa_train_step(jepa, batch, predcost, regularizer, optimizer):
    optimizer.zero_grad()
    state, predicted = jepa.unroll(batch)        # encoder targets + predictor roll-out
    pred_loss = predcost(state, predicted)       # the slot above
    reg_loss = regularizer(state)                # variance-covariance anti-collapse term
    total = pred_loss + reg_loss
    total.backward()
    optimizer.step()
    return total
```

The body of `forward` is the one slot to fill: a scalar function of the residual `r = state −
predicted`, reduced over the `[B, C, T, H, W]` coordinates.
