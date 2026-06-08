**Problem.** Label smoothing is the strongest baseline (mean 80.283) but it went *backwards* on the hardest pair, ResNet-56/CIFAR-100 (71.36, below focal's 71.67 and Poly-1's 71.56). It softens the target by a static, uniform, example-blind `ε/C` floor, so it cannot tell a clean example from a mislabeled or ambiguous one — and the loss is still convex+unbounded (wrong-confident outliers dominate) and the softmax still short-tailed (boundary-ambiguous examples are chased). The lever is still only the loss.

**Key idea.** Attack the loss and the softmax at the source. Cross-entropy is the *matching loss* of the softmax (softmax = dual-gradient of the negative-entropy `F`; CE = Bregman `Δ_F`), so deform `F` with a temperature and both halves re-derive together. Tempered functions: `log_t(x)=(x^{1-t}-1)/(1-t)` (bounded below by `-1/(1-t)` for `t<1`) and its inverse `exp_t(x)=[1+(1-t)x]_+^{1/(1-t)}` (heavy-tailed for `t>1`), both → log/exp at `t→1`.

- **Tempered softmax (`t₂>1`):** `ŷ_i = exp_{t₂}(â_i - λ_{t₂}(â))`, normalizer `λ_{t₂}` with no closed form, solved by a differentiable fixed-point iteration. Heavy tail ⇒ does not chase boundary-ambiguous examples.
- **Tempered Bregman loss (`t₁<1`):** for one-hot class `c`, `L = -log_{t₁}(ŷ_c) - (1/(2-t₁))(1 - Σ_i ŷ_i^{2-t₁})`. Bounded ⇒ wrong-confident outliers cannot dominate.

**Why.** Mismatching `t₁<1<t₂` makes the loss bounded *and* heavy-tailed at once — per example, hardest on exactly the wrong-confident and ambiguous examples label smoothing's uniform floor cannot distinguish. Non-convex in the activations, but that convexity was the source of outlier sensitivity. Stays a **proper** loss (built on the Bregman, not the Tsallis, divergence) and **Bayes-risk consistent**. It also subsumes the strongest baseline (optional uniform label-smoothing of the target lives inside the family), so this generalizes the rung rather than jumping sideways.

**Hyperparameters.** Two temperatures, swept per pair over `t₁ ∈ [0.5,1)`, `t₂ ∈ (1,4]`; mild default `(t₁,t₂)=(0.8,1.2)` for a residual net on moderately-noisy data; `(1,1)` recovers softmax CE. Fixed-point iterations: 5.

**The bar (no feedback for the finale).** Label smoothing's marks are 71.36 / 74.67 / 94.82 (mean 80.283). The falsifiable claim: bound the loss and heavy-tail the softmax, and ResNet-56/CIFAR-100 — the hard residual pair every static reshaping left behind — should be the one that finally rises above 71.36, while VGG-16-BN holds ≥74.67 and MobileNetV2/FashionMNIST stays near 94.82. Validate by sweeping `(t₁,t₂)` per pair and checking the fixed-point normalization stays stable.

```python
# EDITABLE region of pytorch-vision/custom_loss.py — finale: robust bi-tempered logistic loss
def compute_loss(logits, targets, config):
    """Robust bi-tempered logistic loss.

        L = Delta_{F_t1}( one_hot(y), tempered_softmax(logits, t2) )

    t1 < 1 bounds the per-example loss (resists wrong-confident outliers);
    t2 > 1 heavy-tails the softmax (resists boundary-ambiguous examples);
    t1 = t2 = 1 recovers softmax cross-entropy.
    """
    t1, t2, num_iters = 0.8, 1.2, 5

    def log_t(u, t):
        return torch.log(u) if t == 1.0 else (u.pow(1.0 - t) - 1.0) / (1.0 - t)

    def exp_t(u, t):
        if t == 1.0:
            return torch.exp(u)
        return torch.clamp(1.0 + (1.0 - t) * u, min=0.0).pow(1.0 / (1.0 - t))

    def tempered_softmax(a, t):
        if t == 1.0:
            return F.softmax(a, dim=-1)
        mu = a.max(dim=-1, keepdim=True).values            # shift for stability
        shifted = a - mu
        z = shifted
        for _ in range(num_iters):                          # fixed point for the normalizer
            partition = exp_t(z, t).sum(dim=-1, keepdim=True)
            z = shifted * partition.pow(1.0 - t)
        partition = exp_t(z, t).sum(dim=-1, keepdim=True)
        norm = -log_t(1.0 / partition, t) + mu
        return exp_t(a - norm, t)

    y = F.one_hot(targets, config['num_classes']).to(logits.dtype)  # one-hot target
    p = tempered_softmax(logits, t2)                               # heavy-tailed probs
    term1 = (y * (log_t(y + 1e-10, t1) - log_t(p, t1))).sum(dim=-1)        # bounded log-loss
    term2 = (1.0 / (2.0 - t1)) * (y.pow(2.0 - t1) - p.pow(2.0 - t1)).sum(dim=-1)  # normalizer
    return (term1 - term2).mean()
```
