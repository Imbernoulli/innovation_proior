Softmax cross-entropy is the default classifier loss because it is a clean matching loss: softmax is the dual-gradient of the negative entropy convex function F, and cross-entropy is the corresponding Bregman divergence Δ_F. That pairing gives convexity in the last-layer activations and an unbiased estimator whose minimizer recovers the true posterior. But the same construction is fragile under label noise in two separate ways. A mislabeled example that sits far on the correct side of the boundary becomes a large-margin outlier: as the model grows confident, the probability assigned to the wrong label heads toward zero, and −log of that probability grows without bound, so a handful of such points can dominate the empirical risk and bend the boundary. A mislabeled example near the decision boundary becomes small-margin noise: the exponentially light tail of the softmax saturates probabilities to near 0 or 1, forcing the classifier to distort itself to fit an ambiguous point. Existing fixes touch only one symptom or stay inside the softmax family. Label smoothing softens the target, but it is a static, example-independent floor and leaves the loss convex, unbounded, and the softmax light-tailed. Focal loss reweights by confidence but still builds on standard softmax and an unbounded log. MAE-like losses are bounded but train slowly and do not heavy-tail the probabilities. The right move is to deform the single convex object that links the transfer function and the divergence, so both halves change together and the good properties survive.

The method is the Bi-Tempered Logistic Loss. It replaces the ordinary logarithm and exponential with their one-parameter tempered versions. The tempered logarithm is log_t(x) = (x^{1−t} − 1)/(1 − t), which recovers log as t → 1 and is bounded below by −1/(1−t) for 0 ≤ t < 1. That boundedness caps the per-example loss, curing the large-margin outlier problem. Its inverse is the tempered exponential exp_t(x) = [1 + (1−t)x]_+^{1/(1−t)}, which recovers exp as t → 1 and has a heavy, polynomial tail for t > 1. A softmax built from exp_t therefore does not saturate as aggressively, curing the small-margin boundary-noise problem. Two temperatures are used, one for each half. The transfer function is the tempered softmax ŷ_i = exp_{t₂}(â_i − λ_{t₂}(â)), where λ_{t₂} is the per-example normalizer chosen so the probabilities sum to one. Because exp_t does not turn sums into products, this normalizer has no closed form when t₂ ≠ 1; it is found by a stable fixed-point iteration, shifted by max(â) for numerical safety. The divergence is the Bregman divergence induced by the convex function F_t whose gradient is log_t, evaluated at t₁. For a one-hot target with true class c, the loss becomes L = −log_{t₁}(ŷ_c) − (1/(2 − t₁))(1 − Σ_i ŷ_i^{2−t₁}). The first term is the bounded tempered log-loss on the correct class; the second term is the normalization that keeps the whole object a proper Bregman divergence rather than a clipped heuristic.

The key design choice is the temperature mismatch: t₁ < 1 bounds the loss, t₂ > 1 heavy-tails the probabilities, and the pair is deliberately not equal, so the loss is no longer convex in the activations. That loss of convexity is intentional, because last-layer convexity was the structural source of unbounded outlier sensitivity. What is preserved is properness: up to constants independent of the model, the empirical objective is an unbiased estimator of the expected Bregman divergence, so minimizing it still recovers the true class posterior. It is also Bayes-risk consistent, because at the conditional minimizer the predicted class argmax matches the true posterior argmax. The implementation uses the closed-form escort-distribution gradient of the normalizer rather than back-propagating through the fixed-point loop. Typical values are found by a small grid search over t₁ ∈ [0.5, 1) and t₂ ∈ (1, 4]; a mild, stable default for moderately noisy image classification is (t₁, t₂) = (0.8, 1.2), and (1, 1) recovers ordinary softmax cross-entropy.

```python
import torch
import torch.nn.functional as F


def log_t(u, t):
    """Tempered logarithm; bounded below by -1/(1-t) for 0 <= t < 1."""
    if t == 1.0:
        return torch.log(u)
    return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Tempered exponential; heavy-tailed for t > 1 on negative inputs."""
    if t == 1.0:
        return torch.exp(u)
    return torch.clamp(1.0 + (1.0 - t) * u, min=0.0).pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Solve sum_i exp_t(a_i - lambda) = 1 for t > 1 by fixed point."""
    mu = activations.max(dim=-1, keepdim=True).values
    normalized_step_0 = activations - mu
    normalized = normalized_step_0
    for _ in range(num_iters):
        Z = exp_t(normalized, t).sum(dim=-1, keepdim=True)
        normalized = normalized_step_0 * Z.pow(1.0 - t)
    Z = exp_t(normalized, t).sum(dim=-1, keepdim=True)
    return -log_t(1.0 / Z, t) + mu


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax p_i = exp_t(a_i - lambda_t(a))."""
    if t == 1.0:
        return F.softmax(activations, dim=-1)
    norm = compute_normalization_fixed_point(activations, t, num_iters)
    return exp_t(activations - norm, t)


class _BiTemperedLogisticLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, t1, t2, num_iters):
        p = tempered_softmax(logits, t2, num_iters)
        y = F.one_hot(targets, logits.shape[-1]).to(dtype=logits.dtype)
        if t1 == 1.0:
            loss = y * (torch.log(y + 1e-10) - torch.log(p))
        else:
            loss = y * (log_t(y, t1) - log_t(p, t1))
            loss = loss - (1.0 / (2.0 - t1)) * (y.pow(2.0 - t1) - p.pow(2.0 - t1))
        loss = loss.sum(dim=-1)
        ctx.save_for_backward(p, y)
        ctx.t1 = t1
        ctx.t2 = t2
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        p, y = ctx.saved_tensors
        t1, t2 = ctx.t1, ctx.t2
        delta = (p - y) * p.pow(t2 - t1)
        delta_sum = delta.sum(dim=-1, keepdim=True)
        escorts = p.pow(t2)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_logits = delta - escorts * delta_sum
        return grad_output.unsqueeze(-1) * grad_logits, None, None, None, None


def bi_tempered_logistic_loss(logits, targets, t1=0.8, t2=1.2, num_iters=5):
    """Sparse robust loss. t1 = t2 = 1 recovers ordinary cross-entropy."""
    if t1 == 1.0 and t2 == 1.0:
        return F.cross_entropy(logits, targets)
    losses = _BiTemperedLogisticLoss.apply(
        logits, targets, float(t1), float(t2), int(num_iters)
    )
    return losses.mean()
```
