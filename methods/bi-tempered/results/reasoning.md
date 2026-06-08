Let me start from a nagging fragility in the loss I train classifiers with every day. Softmax on the last-layer activations, then cross-entropy against the one-hot label. It is convex in the activations, which everyone treats as a feature. But watch what convexity does when a training label is simply wrong. Take an example the model has correctly learned to be confident about, and suppose its label got flipped — a large-margin mislabeled point, sitting far on the correct side of the boundary but now told it belongs to the other class. Cross-entropy on it is `-log` of a probability heading to zero, so the loss grows without bound as the activation gets more confident, and the example remains a persistent source of pressure on the empirical risk. A handful of such points can dominate the objective and bend the decision boundary toward them. That is not a tuning problem; it is structural — convex potential losses are provably not robust to this kind of noise.

There is a second, separate fragility, and I want to keep it distinct from the first. Consider instead a mislabeled point sitting *near* the boundary — small margin. The softmax tail decays exponentially, so probabilities saturate to 0 or 1 very fast as activations move. A near-boundary point therefore gets assigned a near-extreme probability, and to satisfy `-log(p)` on it the classifier has to distort itself locally to fit the noise. So I actually have two different diseases: large-margin outliers (the loss is unbounded, so they dominate) and small-margin boundary noise (the softmax is light-tailed, so it chases them). If I picture the synthetic two-dimensional case in my head — scatter a clean two-class problem, then inject the two kinds of noise separately — the logistic loss stretches its boundary toward small-margin noise and lurches toward large-margin noise, and these are clearly two different motions that I will need two different cures for.

What would cure each? For the large-margin outliers I want the per-example loss to be *bounded* — past some point, telling the model "you are very wrong about this one" should stop adding loss, so one mislabeled point cannot contribute unboundedly. For the small-margin boundary noise I want the probability assignment to be *heavy-tailed* — a point far on the wrong side should still be allowed a non-negligible probability rather than being forced to 0/1, so the classifier is not compelled to fit it exactly. Two properties: bounded loss, heavy-tailed probabilities. The naive move is to bolt on an ad-hoc clipped loss and an ad-hoc fat-tailed normalization, but I do not want two unrelated hacks. I want to understand *why* softmax and cross-entropy go together in the first place, and then deform that pairing so the two cures fall out of the same construction and the good properties survive.

So why *do* softmax and the log-loss belong together? Cross-entropy is the KL divergence from the one-hot label to the softmax output, and KL is the Bregman divergence induced by the negative entropy `F(y) = Σ_i (y_i log y_i − y_i)`. The Bregman divergence induced by a strictly convex `F` is `Δ_F(y, ŷ) = F(y) − F(ŷ) − (y − ŷ)·∇F(ŷ)` — nonnegative, zero iff `y = ŷ`, and convex in its first argument. And the softmax is exactly the gradient of the convex *dual* of that same `F`. When you pair a transfer function `ŷ = τ(â)` with the Bregman divergence of the convex function whose dual gradient is `τ`, you get a "matching loss" `Δ_F(y, τ(â))`, and matching losses are *convex in the activations* `â`. The logistic loss is the matching loss for the softmax. That is the real reason the two pieces fit: they are dual halves of one convex object. If I am going to generalize, I should generalize `F` — the convex function — and let both the transfer function and the divergence be re-derived from it, so the duality is preserved by construction rather than broken by two independent hacks.

Now I need raw material for the deformation. The two cures point at the two operations I want to bend: the `log` in the loss (to bound it) and the `exp` in the softmax (to fatten its tail). There is a one-parameter deformation that does exactly this. Define a tempered logarithm

  `log_t(x) = (x^{1−t} − 1)/(1 − t)`,

monotone increasing and concave, recovering ordinary `log` as `t → 1` (take the limit: `(x^{1−t}−1)/(1−t) → log x`). The property I care about: for `0 ≤ t < 1`, `log_t` is *bounded below* by `−1/(1−t)`. Ordinary `log` runs to `−∞` as its argument goes to 0 — that is precisely the unboundedness that lets a large-margin outlier blow up the loss. A `log_t` with `t < 1` floors at a finite value, so a loss built from it caps each example's contribution. There is the bounded-loss cure, with `t_1 < 1` the dial.

Its inverse is the tempered exponential

  `exp_t(x) = [1 + (1−t)x]_+^{1/(1−t)}`, with `[·]_+ = max(·, 0)`,

again recovering ordinary `exp` as `t → 1`. The property I care about here is the opposite end: for `t > 1`, `exp_t` has a *heavier* tail than `exp` — it decays polynomially rather than exponentially for negative arguments. A softmax built on `exp_t` with `t > 1` does not saturate as fast, so it can assign a non-negligible probability to a far-wrong point instead of crushing it to zero. There is the heavy-tail cure, with `t_2 > 1` the dial. Two temperatures, two cures, and they are inverse functions of each other — which is the hint that they will sit naturally on the two dual halves of the matching loss.

Let me build the convex function so its gradient is the tempered log, since `∇F` being the link is what made the original construction work. I want `∇F_t(y) = log_t(y)` (elementwise). Integrate `log_t`: `∫ (y^{1−t} − 1)/(1−t) dy = (1/(1−t)) [ y^{2−t}/(2−t) − y ]`, and collecting constants so it is clean,

  `F_t(y) = Σ_i ( y_i log_t(y_i) + (1/(2−t))(1 − y_i^{2−t}) )`.

Check it is convex: `∇²F_t(y) = diag(y^{−t}) ⪰ 0` for `y` in the positive orthant. Good — `F_t` is strictly convex (in fact strongly convex on bounded sets for `0 ≤ t ≤ 1`), so it induces a legitimate Bregman divergence, and at `t = 1` it collapses back to the negative entropy and I recover ordinary KL. Now write that divergence by plugging `F_t` into the Bregman definition. The gradient term is `(y − ŷ)·log_t(ŷ)`, and after the algebra,

  `Δ_{F_t}(y, ŷ) = Σ_i ( y_i log_t(y_i) − y_i log_t(ŷ_i) − (1/(2−t)) y_i^{2−t} + (1/(2−t)) ŷ_i^{2−t} )`.

Equivalently,

  `Δ_{F_t}(y, ŷ) = Σ_i ( y_i^{2−t}/((1−t)(2−t)) − y_i ŷ_i^{1−t}/(1−t) + ŷ_i^{2−t}/(2−t) )`.

This is the β-divergence axis with `β = 2 − t`; at `t = 1` it is KL and at `t = 0` it is squared Euclidean, so this deformation is still anchored in a familiar Bregman family. The bound also comes out cleanly: for `0 ≤ t < 1`, the strong-convexity/smoothness corollary gives `Δ_{F_t}(y, ŷ) ≤ 2B^{2−t}/(1−t)^2` whenever both arguments lie in the `L_{2−t}` ball of radius `B`; on the simplex I can take `B = 1`. So choosing the divergence's temperature `t_1 < 1` gives me the bounded loss I wanted, with the bound coming straight from the floor on `log_{t_1}`.

Now the transfer function — the heavy-tailed softmax. In the original construction the softmax is the dual-gradient of `F` restricted to the simplex. Repeat that with `F_t` restricted to the simplex: I want the `ŷ` that, given activations `â`, satisfies the link `log_t(ŷ) = â − λ·1`, where `λ` is a Lagrange multiplier enforcing `Σ_i ŷ_i = 1`. Inverting with `exp_t`,

  `ŷ_i = exp_t(â_i − λ_t(â))`, with `λ_t(â)` chosen so that `Σ_i exp_t(â_i − λ_t(â)) = 1`.

This is the tempered softmax. Pick its temperature `t_2 > 1` and it is heavy-tailed — the second cure. There is a catch I have to confront, and it is the price of leaving `t = 1`: when `t ≠ 1` there is *no closed form* for `λ_t(â)`. With ordinary softmax, `λ = log Σ exp(â_i)` falls out for free; the tempered normalizer does not, because `exp_t(a + b) ≠ exp_t(a) exp_t(b)` — the tempered exponential does not turn sums into products, so the partition function does not factor. I have to solve `Σ_i exp_t(â_i − λ) = 1` for `λ` numerically, per example. For `t_2 > 1` there is a clean fixed-point iteration: set `μ = max(â)`, keep the original shifted vector `ã_0 = â − μ`, initialize `ã = ã_0`, then repeatedly compute `Z = Σ_i exp_{t_2}(ã_i)` and reset `ã = ã_0 Z^{1−t_2}`. After the fixed point settles, `λ_t(â) = −log_t(1/Z) + μ`. Five iterations is the standard implementation choice.

Putting the two halves together: take the tempered softmax with temperature `t_2` as the transfer function, and score it with the tempered Bregman divergence at temperature `t_1`:

  `L(â | y) = Δ_{F_{t_1}}( y, exp_{t_2}(â − λ_{t_2}(â)) )`.

When `t_1 = t_2` the matching duality holds and the loss is convex (it is just the matching loss for one temperature). But I deliberately *mismatch* — use a smaller `t_1 < 1` for the divergence and a larger `t_2 > 1` for the transfer function — so the loss is bounded (from `t_1 < 1`) and the probabilities are heavy-tailed (from `t_2 > 1`) at the same time. The mismatch makes the loss non-convex in the activations, and that is fine: convexity in the last layer was never convexity in the network's actual parameters, and it was the very thing buying me the outlier-sensitivity I am trying to remove. For a one-hot label with true class `c`, the divergence simplifies — only the `i = c` term has `y_i = 1`, the rest have `y_i = 0`, so

  `L = −log_{t_1}(ŷ_c) − (1/(2 − t_1))(1 − Σ_i ŷ_i^{2−t_1})`,

with `ŷ = exp_{t_2}(â − λ_{t_2}(â))`. The first term is the bounded tempered log-loss on the true class; the second is a normalization term over all classes that is what makes the whole thing a real Bregman divergence and not just a clipped log.

Let me make sure I did not break the two properties a classification loss must have. First, *properness*: I want the minimizer of the expected loss to be the true posterior, so that fitting the loss actually recovers `P(y|x)`. For a fixed `x`, the expected Bregman divergence differs from

  `Σ_i [ −η_i log_{t_1} p_i + (1/(2−t_1)) p_i^{2−t_1} ]`

only by terms that depend on the true posterior `η` and not on the model probabilities `p`. Sampling `y_n ~ η(·|x_n)` gives the unbiased empirical objective `(1/N) Σ_n [ −log_{t_1} p_{y_n}(x_n) + (1/(2−t_1)) Σ_i p_i(x_n)^{2−t_1} ]`, again up to those model-independent constants. Every term is a function of the model's probabilities and the observed label, with no hidden dependence on the unknown conditional. Contrast the Tsallis-divergence route, which looks similar but uses `log_t(ŷ/y)`: because `log_t(a/b) ≠ log_t(a) − log_t(b)`, its unbiased estimator would need the true `P(y|x)` inside the logarithm, and approximating that conditional by 1 makes the estimator biased. So building on the *Bregman* divergence rather than the Tsallis one is exactly what keeps the loss proper, and that distinction is load-bearing, not cosmetic. Second, *Bayes-risk consistency*: at the conditional-risk minimizer the predicted probabilities equal the class posteriors `η_i`; the repeated normalization term in each class loss does not affect the classwise argmin, and since `−log_{t_1}` is monotone decreasing for `0 ≤ t_1 < 1`, `argmin_i l_i = argmax_i p_i = argmax_i â_i = argmax_i η_i`. The loss picks the Bayes-optimal class even in the non-convex regime. Both properties survive the mismatch.

What do I set the temperatures to? Each dial cures one disease, so the choice should track which noise I expect. Pure small-margin boundary noise wants tail-heaviness only — `t_1 = 1`, `t_2` large (e.g. 4). Pure large-margin outliers want boundedness only — `t_1` small (e.g. 0.2), `t_2 = 1`. Realistic noise has both, so I want both dials engaged, but not so aggressively that the loss becomes hard to optimize — very extreme temperatures can fail to converge. For a moderate-scale, moderately-noisy image-classification problem with a residual network, a mild pair like `t_1 = 0.8`, `t_2 = 1.2` sits in the safe, effective range: enough boundedness and tail-heaviness to resist the noise, gentle enough that training is stable. Higher noise or smaller, cleaner-architecture datasets push toward more extreme values; large clean datasets want the dials near 1. The pair is a two-dimensional grid search, `t_1 ∈ [0.5, 1)` and `t_2 ∈ (1, 4]`.

Now the implementation, grounded in the tempered primitives. I need `log_t` and `exp_t` (with the `relu` clamp realizing `[·]_+`), the fixed-point normalizer for `t_2 > 1`, the tempered softmax, and then the one-hot Bregman divergence. I also want the backward pass to use the closed-form derivative rather than whatever gradient happens to come from a finite number of fixed-point iterations. Differentiate the normalization equation `Σ_j p_j = 1`, where `p_j = exp_{t_2}(a_j − λ)`: since `d exp_t(u)/du = exp_t(u)^t`, I get `∂λ/∂a_i = p_i^{t_2}/Σ_j p_j^{t_2}`, the escort distribution. The loss derivative with respect to `p_j` is `(p_j − y_j)p_j^{−t_1}`, so the chain rule gives

  `∂L/∂a_i = (p_i − y_i)p_i^{t_2−t_1} − q_i Σ_j (p_j − y_j)p_j^{t_2−t_1}`,

where `q_i = p_i^{t_2}/Σ_j p_j^{t_2}`. That is the gradient I should put into the PyTorch port; when `t_1 = t_2 = 1`, I skip all of this and call ordinary cross-entropy.

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

The chain in one breath: softmax cross-entropy is fragile in two distinct ways — its convex, unbounded log lets large-margin mislabeled points dominate the objective, and its light-tailed softmax forces the classifier to chase small-margin boundary noise. Both pieces of the loss are dual halves of one convex object (the matching loss: softmax is the dual-gradient of the negative entropy `F`, cross-entropy is the Bregman divergence of `F`), so I generalize `F` rather than hack the pieces separately. A tempered logarithm `log_t`, bounded below for `t_1 < 1`, gives a bounded divergence (curing large-margin outliers); its inverse, the tempered exponential `exp_t`, heavy-tailed for `t_2 > 1`, gives a non-saturating tempered softmax (curing small-margin noise) — the price being a per-example normalizer with no closed form, solved by a fixed-point iteration. Pairing the divergence at `t_1 < 1` with the transfer function at `t_2 > 1` deliberately *mismatches* the temperatures, producing a bounded, heavy-tailed, non-convex loss that is still a *proper* loss (because it is built on the Bregman, not the Tsallis, divergence) and Bayes-risk consistent; the implementation uses the corresponding escort-distribution gradient and the ordinary cross-entropy branch at `(t_1, t_2) = (1, 1)`.
