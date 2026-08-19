# PolyLoss (Poly-1): a polynomial-expansion view of classification losses

**Problem.** Cross-entropy and focal loss are fixed curves chosen by habit; there is no argument that the cross-entropy coefficient profile is optimal for a given architecture, dataset, or class distribution. We want a low-dimensional, interpretable family of losses that contains cross-entropy and focal loss as special cases and can be tuned with a tiny task-specific search.

**Key idea.** Expand cross-entropy in the Mercator series of the logarithm around the target-class probability `P_t`:

  `L_CE = -log(P_t) = Σ_{j≥1} (1/j)(1-P_t)^j = (1-P_t) + ½(1-P_t)² + ⅓(1-P_t)³ + …`

This exposes cross-entropy as a weighted sum of polynomial bases `(1-P_t)^j` with coefficients `α_j = 1/j`. The general family is `L = Σ_j α_j (1-P_t)^j`; with nonnegative coefficients it is monotone decreasing in `P_t`. In this view:

- **Cross-entropy** is `α_j = 1/j`. Its gradient is the geometric series `-dL_CE/dP_t = Σ_j (1-P_t)^{j-1} = 1 + (1-P_t) + …`, whose leading term is the constant `1` (independent of `P_t`).
- **Focal loss** `-(1-P_t)^γ log(P_t) = Σ_j (1/j)(1-P_t)^{j+γ}` is a *horizontal* shift of the `1/j` profile by `γ`; its training push starts with a term proportional to `(1-P_t)^γ`, so the `P_t`-independent leading constant is gone and easy examples are suppressed.

**Why Poly-1.** Truncating the tail (`α_{j>N}=0`) fails on many-class data: when `P_t ≈ 0` early in training the high-order terms remain non-negligible (at `P_t=0.001` the 500th gradient coefficient is `0.999^499 ≈ 0.6`), so hundreds of terms are needed. Tuning all coefficients is infeasible. But the leading polynomial `(1-P_t)` carries more than half of the cross-entropy gradient for most of training, so it is the highest-leverage single coefficient to adjust. Keep the entire cross-entropy profile and *vertically* perturb the leading coefficient:

  `L_Poly-1 = (1 + ε₁)(1-P_t) + ½(1-P_t)² + … = -log(P_t) + ε₁(1-P_t)`.

A positive `ε₁` strengthens the surviving confidence-pressure that plain cross-entropy lets decay too soon — the opposite of focal loss's move. Class imbalance by itself does not decide the sign: a 21,841-way classification head and an 80-class detection head are both imbalanced, so imbalance alone can't tell them apart. What should decide it instead is the model's own confidence trajectory, logged as mean `P_t` over training: whichever task's curve sits chronically low calls for `ε₁ > 0` to lift it, and whichever task's curve saturates near 1 well before training ends calls for `ε₁ < 0` to pull it back down, a softer version of focal loss's suppression. My guess is the 21K-way classification head is the chronically-low case — with that many target classes, near-certain softmax mass on the right one should be rare even when the prediction is correct — and the detector head is the saturating case, since an 80-way decision is comparatively easy to be sure of; logging each curve is what would confirm that before committing a sign.

**Hyperparameters.** One knob: `ε₁`, swept by 1-D grid search against each task's logged confidence trajectory rather than fixed in advance — expect classification and detection to land on different signs, given their different predicted confidence regimes. `ε₁ = 0` recovers cross-entropy exactly; `ε₁ ≥ -1` keeps the leading coefficient nonnegative, so that is the floor the search is bounded by. `P_t` is the softmax probability at the ground-truth class.

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
