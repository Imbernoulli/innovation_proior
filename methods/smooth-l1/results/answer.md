# Smooth L1 (Huber) prediction loss, distilled

Smooth L1 loss is the per-coordinate discrepancy that is quadratic for small residuals and
linear for large ones — squared-error in the middle, absolute-error in the tail, spliced `C¹`
at a crossover threshold. At `beta = 1` it coincides with the Huber robust M-estimator loss:
the minimax-optimal cost for data that are mostly Gaussian but contaminated by an arbitrary
fraction of gross outliers. For the temporal-JEPA prediction term it is applied to the
residual `r = state − predicted` between predicted and target latent feature maps and averaged
over all coordinates.

## Problem it solves

Pick the per-coordinate function applied to the residual between predicted and target latent
tensors. Squared error (`½r²`) is smooth and Gaussian-efficient near the optimum but its
influence `ψ(r) = r` is unbounded, so a few large residuals — from compounding autoregressive
roll-out, hard frames, or a noisy moving target — produce gradients that dominate the batch
and destabilize training. Absolute error (`|r|`) has bounded influence `ψ(r) = sign(r)` (robust)
but is kinked and non-differentiable at the optimum with no quadratic basin (inefficient on the
Gaussian bulk). Need one loss with both: quadratic-and-efficient in the middle, bounded-influence
in the tail.

## Key idea

Model the residual density as an ε-contaminated Gaussian `F = (1 − ε)Φ + εH`, `H` arbitrary.
Choose the influence function `ψ = ρ'` to minimize worst-case asymptotic variance over this
class (a minimax / two-player game). The least-favorable density is the smallest-Fisher-
information member: Gaussian in the center, exponential tails outside. Its MLE score is

  `ψ(t) = clamp(t, −k, k)`  — linear (=`t`) where data are Gaussian (efficient),
  saturating (=`±k`) in the tails (bounded influence ⇒ robust).

Integrating `ψ` (with value-continuity forcing the additive constant) gives the **Huber loss**:

  `ρ(t) = ½ t²`        for `|t| ≤ k`
  `ρ(t) = k|t| − ½k²`  for `|t| > k`.

Squared error is the `k → ∞` limit (the mean); absolute error is the `k → 0` limit (the
median). So Smooth L1 is the one-parameter family interpolating the two baselines, sitting at
the contaminated interior `0 < ε < 1`. The maximum per-coordinate gradient is bounded by `k`
(in the unit-saturating parameterization, by `1`), which is the stability that squared error
lacks: no single large residual can pull harder than the cap.

## Two parameterizations (same family)

- **Huber, threshold `δ`** (quadratic fixed, tail slope `= δ`):
  `ρ_δ(t) = ½t²` if `|t| < δ`, else `δ(|t| − ½δ)`; gradient saturates at `±δ`.
- **Smooth L1, threshold `β`** (unit-saturating; quadratic divided by `β`):
  `ρ_β(t) = ½t²/β` if `|t| < β`, else `|t| − ½β`; gradient `clamp(t/β, −1, 1)` (capped at `±1`).

They are related by `ρ_δ(t) = δ · ρ_β(t)` when `δ = β`, and coincide at `δ = β = 1`. The
unit-saturating form decouples *where* the crossover is from the gradient ceiling (always `1`),
which is why it is the standard library default. JEPA uses `β = 1`.

## Properties

- **`C¹` at the join:** in the Smooth-L1 parameterization, the values `½β` and `β − ½β`
  match at `|t| = β`, and the slopes `sign(t)` match as well; the `−½β` constant enforces
  value-continuity. In the Huber-`δ` parameterization the corresponding tail constant is
  `−½δ²`.
- **Bounded gradient:** `|ρ_β'(t)| ≤ 1` — robust to outlier residuals; no learning-rate
  babysitting to contain exploding gradients on large/unbounded targets.
- **Convex**, with a strongly-convex quadratic basin near `0` (efficient settling) and linear
  growth in the tail (so loss/gradient grow only linearly in the residual magnitude).

## Final form for the JEPA prediction loss

Apply `ρ_β` (β = 1) elementwise to `r = state − predicted` and average over the `[B, C, T, H,
W]` coordinates (mean reduction keeps the term comparable across model sizes and batch/
resolution). This is exactly PyTorch's `F.smooth_l1_loss` at its default `beta = 1`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Smooth L1 (Huber, beta=1) prediction loss for temporal JEPA.

    Per coordinate of the residual r = state - predicted:
        rho(r) = 0.5 * r^2 / beta   if |r| <  beta   (Gaussian-efficient quadratic basin)
               = |r| - 0.5 * beta   if |r| >= beta   (bounded-influence linear tail)
    gradient with respect to r = clamp(r / beta, -1, 1): squared-error pull capped
    at unit magnitude, robust to the outlier residuals of compounding roll-out.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return F.smooth_l1_loss(state, predicted)  # beta=1, mean over [B,C,T,H,W]
```

Explicit elementwise equivalent (same values; the derived `ρ` written out, with a general
threshold), which `F.smooth_l1_loss` reproduces exactly:

```python
    def forward_explicit(self, state, predicted, beta: float = 1.0):
        r = state - predicted
        z = r.abs()
        quadratic = 0.5 * z * z / beta      # |r| <  beta : quadratic basin
        linear = z - 0.5 * beta             # |r| >= beta : bounded-gradient tail
        return torch.where(z < beta, quadratic, linear).mean()
```

The Huber-parameterized sibling (threshold `δ`, tail slope `δ`) is `F.huber_loss(state,
predicted, delta=δ)`, which equals `δ · F.smooth_l1_loss(state, predicted, beta=δ)`; at `δ = 1`
it is identical to the smooth-L1 form above.
