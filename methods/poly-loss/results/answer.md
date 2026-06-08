# PolyLoss (Poly-1): a polynomial-expansion view of classification losses

**Problem.** Cross-entropy and focal loss are fixed curves chosen by habit; there is no argument that the cross-entropy coefficient profile is optimal for a given architecture, dataset, or class distribution. We want a low-dimensional, interpretable family of losses that contains cross-entropy and focal loss as special cases and can be tuned with a tiny task-specific search.

**Key idea.** Expand cross-entropy in the Mercator series of the logarithm around the target-class probability `P_t`:

  `L_CE = -log(P_t) = Σ_{j≥1} (1/j)(1-P_t)^j = (1-P_t) + ½(1-P_t)² + ⅓(1-P_t)³ + …`

This exposes cross-entropy as a weighted sum of polynomial bases `(1-P_t)^j` with coefficients `α_j = 1/j`. The general family is `L = Σ_j α_j (1-P_t)^j`; with nonnegative coefficients it is monotone decreasing in `P_t`. In this view:

- **Cross-entropy** is `α_j = 1/j`. Its gradient is the geometric series `-dL_CE/dP_t = Σ_j (1-P_t)^{j-1} = 1 + (1-P_t) + …`, whose leading term is the constant `1` (independent of `P_t`).
- **Focal loss** `-(1-P_t)^γ log(P_t) = Σ_j (1/j)(1-P_t)^{j+γ}` is a *horizontal* shift of the `1/j` profile by `γ`; its training push starts with a term proportional to `(1-P_t)^γ`, so the `P_t`-independent leading constant is gone and easy examples are suppressed.

**Why Poly-1.** Truncating the tail (`α_{j>N}=0`) fails on many-class data: when `P_t ≈ 0` early in training the high-order terms remain non-negligible (at `P_t=0.001` the 500th gradient coefficient is `0.999^499 ≈ 0.6`), so hundreds of terms are needed. Tuning all coefficients is infeasible. But the leading polynomial `(1-P_t)` carries more than half of the cross-entropy gradient for most of training, so it is the highest-leverage single coefficient to adjust. Keep the entire cross-entropy profile and *vertically* perturb the leading coefficient:

  `L_Poly-1 = (1 + ε₁)(1-P_t) + ½(1-P_t)² + … = -log(P_t) + ε₁(1-P_t)`.

A positive `ε₁` strengthens the surviving confidence-pressure that plain cross-entropy lets decay too soon — the opposite of focal loss's move — and helps on balanced data; the optimal sign/magnitude is task-dependent, with negative values reducing the leading push in over-confident or heavily imbalanced settings.

**Hyperparameters.** One knob: `ε₁`, swept by 1-D grid search; `ε₁ = 2.0` is the image-classification default. `ε₁ = 0` recovers cross-entropy; `ε₁ ≥ -1` keeps the leading coefficient nonnegative. `P_t` is the softmax probability at the ground-truth class.

```python
import torch
import torch.nn.functional as F


def poly1_cross_entropy(logits, targets, epsilon=2.0):
    """PolyLoss Poly-1: cross-entropy with the leading polynomial coefficient
    perturbed by epsilon.

        L = -log(P_t) + epsilon * (1 - P_t)

    Recovers cross-entropy at epsilon = 0. Keeping epsilon >= -1 leaves the
    leading coefficient nonnegative. P_t is the softmax probability the model
    assigns to the ground-truth class.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")          # -log(P_t), [B]
    p_t = torch.softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1)                          # P_t, [B]
    poly1 = ce + epsilon * (1.0 - p_t)                               # leading-coeff perturbation
    return poly1.mean()
```
