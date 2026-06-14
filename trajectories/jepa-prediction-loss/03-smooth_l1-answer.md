**Problem.** MSE fixed cosine's magnitude failure — the large width recovered to 0.6409 and the
monotone-with-width decline vanished — but its per-coordinate influence `ψ(r) = 2r` is unbounded, so a
few large residuals dominate the gradient. The small model produces the largest, most erratic
autoregressive roll-out residuals, and it sagged to 0.6019, the lowest MSE number and below base/large —
the signature of heavy-tailed residuals the Gaussian-default loss cannot defend against. The prediction
loss must stay efficient on the Gaussian bulk *and* bound the influence of the large residuals.

**Key idea.** Model the residuals as an ε-contaminated Gaussian `F = (1 − ε)Φ + εH` with `H` arbitrary,
and pick the per-coordinate influence `ψ = ρ'` that minimizes worst-case asymptotic variance over that
class (a minimax game). The least-favorable density is the smallest-Fisher-information member — Gaussian
in the center, exponential tails outside — and its MLE score is `ψ(r) = clamp(r, −k, k)`: the
squared-error pull where data look Gaussian (efficient), saturating at `±k` in the tails (bounded
influence ⇒ robust). Integrating that score, with value-continuity forcing the additive constant, gives
the **Huber / Smooth-L1** loss:

```
ρ(r) = ½ r²        for |r| ≤ k     (Gaussian-efficient quadratic basin)
ρ(r) = k|r| − ½k²  for |r| > k     (bounded-gradient linear tail)
```

Squared error is the `k → ∞` limit (MSE, the mean); absolute error the `k → 0` limit (the median). So
Smooth-L1 is the one-parameter family that *contains* the MSE baseline as a limit and sits at the
contaminated interior where the roll-out residuals actually live — not a third option but MSE's robust
generalization.

**Why it should beat MSE where MSE fell.** Smooth-L1 is identical to MSE for every residual below the
threshold, so where outliers are rare (base, large) it changes little — I expect those to hold. Where
outliers are common (the small model's large roll-out residuals) the capped gradient `clamp(r, −1, 1)`
stops any single coordinate from swamping the batch, which is the protection MSE lacked and the reason
the small width should recover. The `C¹` splice keeps the smooth quadratic basin at the optimum (no kink,
unlike absolute error or cosine).

**Hyperparameters.** Crossover `k = β = 1` — the natural outlier scale, since VC-normalized latents are
of order one. Mean reduction over `[B, C, T, H, W]`, so the term is per-coordinate and scale-matched
across the three widths (and the per-coordinate influence is capped by `1/N`). This is exactly
`F.smooth_l1_loss` at its default `beta = 1`.

**Step-3 edit.** Replace `CustomPredictionLoss.forward` with Smooth-L1. This is the literal scaffold fill
of the editable region (`custom_prediction_loss.py`, lines 36–54).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Smooth L1 (Huber, beta=1) prediction loss for temporal JEPA.

    Per coordinate of the residual r = state - predicted:
        rho(r) = 0.5 * r^2          if |r| <  1   (Gaussian-efficient quadratic basin)
               = |r| - 0.5          if |r| >= 1   (bounded-influence linear tail)
    gradient clamp(r, -1, 1): the squared-error pull capped at unit magnitude, robust to
    the large compounding roll-out residuals MSE could not contain. Averaged over
    [B, C, T, H, W] so the term is scale-matched across the three model widths.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return F.smooth_l1_loss(state, predicted)    # beta=1, mean over [B, C, T, H, W]
```
