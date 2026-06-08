# Robust Bi-Tempered Logistic Loss

**Problem.** Softmax cross-entropy is fragile under label noise in two distinct ways: it is convex and *unbounded* in the activations, so a large-margin mislabeled example can dominate the objective; and the softmax is *light-tailed* (exponential decay), so a small-margin mislabeled example near the boundary is chased and forced toward an extreme probability. We want a loss that is bounded *and* heavy-tailed, while remaining a proper, Bayes-risk-consistent loss.

**Key idea.** Cross-entropy is the *matching loss* for the softmax: the softmax is the dual-gradient of the negative-entropy convex function `F`, and cross-entropy is the Bregman divergence `Δ_F`. Generalize `F` with a temperature so both halves deform together. Use the tempered logarithm and exponential

  `log_t(x) = (x^{1−t} − 1)/(1 − t)`  (bounded below by `−1/(1−t)` for `0 ≤ t < 1`),
  `exp_t(x) = [1 + (1−t)x]_+^{1/(1−t)}`  (heavy, polynomial tail for `t > 1`),

both recovering `log`/`exp` as `t → 1`. Build `F_t` with `∇F_t = log_t`, giving `F_t(y) = Σ_i (y_i log_t y_i + (1/(2−t))(1 − y_i^{2−t}))` and `∇²F_t(y) = diag(y^{−t}) ⪰ 0` (a β-divergence with `β = 2−t`; KL at `t=1`).

- **Tempered softmax (transfer function, temperature `t₂`):** `ŷ_i = exp_{t₂}(â_i − λ_{t₂}(â))`, with `λ_{t₂}` the per-example normalizer enforcing `Σ_i ŷ_i = 1`. No closed form for `t₂ ≠ 1` (because `exp_t` does not factor sums into products); for `t₂ > 1`, solve it by the fixed-point iteration with `μ=max(â)`, `ã_0=â−μ`, `Z=Σ_i exp_{t₂}(ã_i)`, `ã=ã_0 Z^{1−t₂}`, and `λ=−log_{t₂}(1/Z)+μ`. `t₂ > 1` ⇒ heavy-tailed ⇒ resists small-margin boundary noise.
- **Tempered Bregman divergence (loss, temperature `t₁`):** for a one-hot label class `c`, `L = −log_{t₁}(ŷ_c) − (1/(2−t₁))(1 − Σ_i ŷ_i^{2−t₁})`. `t₁ < 1` ⇒ bounded loss ⇒ resists large-margin outliers.

**Why it works.** When `t₁ = t₂` the matching duality holds and the loss is convex. *Mismatching* `t₁ < 1 < t₂` makes the loss bounded and heavy-tailed simultaneously (non-convex, which is fine — last-layer convexity was the source of outlier sensitivity). It stays a **proper** loss: up to constants independent of the model, the empirical mean `−log_{t₁} p_y + (1/(2−t₁))Σ_i p_i^{2−t₁}` is an unbiased estimator of the expected Bregman divergence. The Tsallis-divergence construction is biased because `log_t(a/b) ≠ log_t a − log_t b` and would require the unknown conditional label distribution. It is also **Bayes-risk consistent**: at the conditional-risk minimizer, `argmin_i l_i = argmax_i p_i = argmax_i â_i = argmax_i η_i`.

**Hyperparameters.** Two temperatures, grid-searched over `t₁ ∈ [0.5, 1)`, `t₂ ∈ (1, 4]`. Pure small-margin noise ⇒ tail-only `(1, 4)`; pure large-margin outliers ⇒ bounded-only `(0.2, 1)`; realistic mixed noise ⇒ both. A mild, stable image-classification default is `(t₁, t₂) = (0.8, 1.2)`; `(1, 1)` recovers softmax cross-entropy. Normalization iterations: 5.

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
    return torch.relu(1.0 + (1.0 - t) * u).pow(1.0 / (1.0 - t))


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
    if t == 1.0:
        return F.softmax(activations, dim=-1)
    norm = compute_normalization_fixed_point(activations, t, num_iters)
    return exp_t(activations - norm, t)


class _BiTemperedLogisticLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, t1, t2, num_iters):
        p = tempered_softmax(logits, t2, num_iters)
        y = F.one_hot(targets, logits.shape[-1]).to(dtype=logits.dtype)
        loss = -log_t(p.gather(-1, targets[..., None]).squeeze(-1), t1)
        loss = loss - (1.0 / (2.0 - t1)) * (1.0 - p.pow(2.0 - t1).sum(dim=-1))
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
    losses = _BiTemperedLogisticLoss.apply(logits, targets, float(t1), float(t2), int(num_iters))
    return losses.mean()
```
