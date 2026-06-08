**Problem.** Focal loss came in lowest on the easy-pool pair (MobileNetV2/FashionMNIST, 94.14) and near cross-entropy on the two CIFAR-100 pairs: on balanced data its move — down-weighting easy examples — is the wrong direction, starving the late-training gradient rather than rebalancing anything. The loss is still the only lever.

**Key idea.** Expand cross-entropy in the Mercator series: `-log(P_t) = Σ_j (1/j)(1-P_t)^j`, so cross-entropy is a polynomial in `(1-P_t)` with coefficients `1/j`. Focal loss `-(1-P_t)^γ log(P_t) = Σ_j (1/j)(1-P_t)^{j+γ}` is a *horizontal* slide of that profile, which deletes the constant leading gradient term. Its gradient is the geometric series `1 + (1-P_t) + (1-P_t)² + …`; the leading `1` is the non-decaying push that keeps lifting confidence. PolyLoss Poly-1 keeps the whole profile and *raises the leading coefficient* instead: `L = -log(P_t) + ε₁(1-P_t)`.

**Why.** A positive `ε₁` adds to the non-decaying leading gradient term, so the model keeps being pushed toward higher true-class confidence even on examples it already mostly has — the literal opposite of focal loss on the same axis. The tail cannot be touched (early in training, at `P_t ≈ 0`, the high-order terms carry most of the gradient), but the leading polynomial contributes more than half the gradient over the bulk of training, so it is the single highest-leverage coefficient.

**Hyperparameters.** One knob, `ε₁ = 2.0` (image-classification default, applied to all three balanced pairs); `ε₁ = 0` recovers cross-entropy. `P_t` is the softmax probability at the true class.

**What to watch.** Expect MobileNetV2/FashionMNIST to recover above focal's 94.14 (the crushed gradient is now amplified); expect the two CIFAR-100 pairs roughly level with focal (a modest nudge on a 100-class profile with a long intact tail).

```python
# EDITABLE region of pytorch-vision/custom_loss.py — step 2: PolyLoss Poly-1 (eps=2.0)
def compute_loss(logits, targets, config):
    """PolyLoss Poly-1: cross-entropy plus a leading-polynomial correction,
    L = -log(p_t) + eps * (1 - p_t), with eps = 2.0. p_t is the softmax
    probability at the true class."""
    ce = F.cross_entropy(logits, targets)                                   # -log(p_t)
    pt = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze()  # p_t
    return ce + 2.0 * (1 - pt).mean()                                       # + eps*(1 - p_t)
```
