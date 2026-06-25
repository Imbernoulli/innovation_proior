Let me start from a nagging fragility in the loss I train classifiers with every day. Softmax on the last-layer activations, then cross-entropy against the one-hot label. It is convex in the activations, which everyone treats as a feature. But watch what convexity does when a training label is simply wrong. Take an example the model has correctly learned to be confident about, and suppose its label got flipped — a large-margin mislabeled point, sitting far on the correct side of the boundary but now told it belongs to the other class. Cross-entropy on it is `-log` of a probability heading to zero, so the loss grows without bound as the activation gets more confident, and the example remains a persistent source of pressure on the empirical risk. Let me actually put a number on "without bound": true class 0, but the wrong class gets activation `w`, so `p_0 = 1/(1+e^w)` and the loss is `-log p_0 = log(1+e^w) ≈ w` for large `w`. At `w = 5, 20, 100` that is `5.01, 20.0, 100.0` — it just tracks the margin linearly, forever. A handful of such points can dominate the objective and bend the decision boundary toward them. That is not a tuning problem; it is structural — convex potential losses are provably not robust to this kind of noise.

There is a second, separate fragility, and I want to keep it distinct from the first. Consider instead a mislabeled point sitting *near* the boundary — small margin. The softmax tail decays exponentially, so probabilities saturate to 0 or 1 very fast as activations move. A near-boundary point therefore gets assigned a near-extreme probability, and to satisfy `-log(p)` on it the classifier has to distort itself locally to fit the noise. So I actually have two different diseases: large-margin outliers (the loss is unbounded, so they dominate) and small-margin boundary noise (the softmax is light-tailed, so it chases them). If I picture the synthetic two-dimensional case in my head — scatter a clean two-class problem, then inject the two kinds of noise separately — the logistic loss stretches its boundary toward small-margin noise and lurches toward large-margin noise, and these are clearly two different motions that I will need two different cures for.

What would cure each? For the large-margin outliers I want the per-example loss to be *bounded* — past some point, telling the model "you are very wrong about this one" should stop adding loss, so one mislabeled point cannot contribute unboundedly. For the small-margin boundary noise I want the probability assignment to be *heavy-tailed* — a point far on the wrong side should still be allowed a non-negligible probability rather than being forced to 0/1, so the classifier is not compelled to fit it exactly. Two properties: bounded loss, heavy-tailed probabilities. The naive move is to bolt on an ad-hoc clipped loss and an ad-hoc fat-tailed normalization, but I do not want two unrelated hacks. The reason I distrust that route is concrete: if I clip the loss and separately swap the normalization, I lose any guarantee that the pair still defines a *proper* loss whose minimizer is the posterior — the two pieces of softmax cross-entropy are not independent, and I would be cutting a joint object in two arbitrary places. So before deforming anything I want to understand *why* softmax and cross-entropy are a single object, and deform that object as a whole.

So why do softmax and the log-loss belong together? Cross-entropy is the KL divergence from the one-hot label to the softmax output, and KL is the Bregman divergence induced by the negative entropy `F(y) = Σ_i (y_i log y_i − y_i)`. The Bregman divergence induced by a strictly convex `F` is `Δ_F(y, ŷ) = F(y) − F(ŷ) − (y − ŷ)·∇F(ŷ)` — nonnegative, zero iff `y = ŷ`, and convex in its first argument. And the softmax is exactly the gradient of the convex *dual* of that same `F`. When you pair a transfer function `ŷ = τ(â)` with the Bregman divergence of the convex function whose dual gradient is `τ`, you get a "matching loss" `Δ_F(y, τ(â))`, and matching losses are convex in the activations `â`. So the logistic loss is the matching loss for the softmax — the two pieces are not a coincidence, they are dual halves of one convex object. That reframes the deformation: I should generalize `F` — the single convex function — and let both the transfer function and the divergence be re-derived from it, so the duality is preserved by construction rather than broken by two independent hacks.

Now I need raw material for the deformation. The two cures point at the two operations I want to bend: the `log` in the loss (to bound it) and the `exp` in the softmax (to fatten its tail). There is a one-parameter deformation that bends exactly these. Define a tempered logarithm

  `log_t(x) = (x^{1−t} − 1)/(1 − t)`,

monotone increasing and concave, which should recover ordinary `log` as `t → 1`. Let me check that limit rather than take it on faith — numerically, `log_t(2)` at `t = 0.99, 0.999, 0.9999` gives `0.6956, 0.6934, 0.69317`, converging onto `log 2 = 0.69315`. Good. The property I am hoping for: for `0 ≤ t < 1`, as `x → 0` the term `x^{1−t} → 0`, so `log_t(x) → (0 − 1)/(1 − t) = −1/(1 − t)`, a finite floor. Numerically at `t = 0.5`, `log_t(10⁻⁸) = −1.9998`, sitting just above the predicted `−1/(1−0.5) = −2`. So `log_t` with `t < 1` is *bounded below*, where ordinary `log` runs to `−∞` as its argument goes to 0 — and that `−∞` is precisely the unboundedness I watched the large-margin outlier exploit. A loss built from a floored `log` would cap each example's contribution. That is encouraging for the first disease, with `t < 1` as the dial.

Its inverse should be a tempered exponential

  `exp_t(x) = [1 + (1−t)x]_+^{1/(1−t)}`, with `[·]_+ = max(·, 0)`,

recovering ordinary `exp` as `t → 1`. Let me confirm it really is the inverse: composing, `exp_t(log_t(x))` at `t = 1.5` and `x = 0.3, 1.0` returns `0.3, 1.0` exactly. The property I want at this end is the opposite of the floor: for `t > 1`, `1/(1−t)` is a negative exponent, so for large negative `x` the base `1 + (1−t)x` grows linearly and `exp_t` decays only *polynomially*, not exponentially. So a softmax built on `exp_t` with `t > 1` should not saturate as fast, and could leave a non-negligible probability on a far-wrong class instead of crushing it to zero. That is encouraging for the second disease, with `t > 1` as the dial. Two temperatures, two diseases, and the two functions are inverses of each other — which is suggestive, because the two dual halves of the matching loss are also linked by inversion (`τ` is the inverse of `∇F` on the simplex). It would be clean if `t < 1` and `t > 1` landed naturally on the divergence half and the transfer half respectively. Let me build the construction and see whether they do.

I'll build the convex function so its gradient is the tempered log, since `∇F` being the link is what made the original construction work. I want `∇F_t(y) = log_t(y)` elementwise. Integrate `log_t`: `∫ (y^{1−t} − 1)/(1−t) dy = (1/(1−t)) [ y^{2−t}/(2−t) − y ]`, and collecting constants so it is clean,

  `F_t(y) = Σ_i ( y_i log_t(y_i) + (1/(2−t))(1 − y_i^{2−t}) )`.

Before trusting that, let me check the gradient by finite differences — it is easy to drop a factor when integrating. At `t = 0.7`, `y = (0.3, 0.5, 0.2)`, central differences of `F_t` give `(−1.0105, −0.6258, −1.2766)`, and `log_t(y)` evaluates to `(−1.0105, −0.6258, −1.2766)`. They agree, so `∇F_t = log_t` as intended. Is `F_t` convex? `∇²F_t(y) = diag(d log_t(y)/dy) = diag(y^{−t})`, which is `⪰ 0` on the positive orthant, so `F_t` is strictly convex there (strongly convex on bounded sets for `0 ≤ t ≤ 1`) and induces a legitimate Bregman divergence. And at `t = 1` it should collapse to negative entropy and KL — I'll verify the divergence does, in a moment, rather than assert it.

Now the divergence, from the Bregman definition with this `F_t`. The gradient term is `(y − ŷ)·log_t(ŷ)`, and after the algebra,

  `Δ_{F_t}(y, ŷ) = Σ_i ( y_i log_t(y_i) − y_i log_t(ŷ_i) − (1/(2−t)) y_i^{2−t} + (1/(2−t)) ŷ_i^{2−t} )`,

equivalently

  `Δ_{F_t}(y, ŷ) = Σ_i ( y_i^{2−t}/((1−t)(2−t)) − y_i ŷ_i^{1−t}/(1−t) + ŷ_i^{2−t}/(2−t) )`.

Two derivations of the same object — let me make sure they actually coincide and that I trust them. At `t = 0.7`, `y = (0.6, 0.3, 0.1)`, `ŷ = (0.4, 0.4, 0.2)`: the raw Bregman definition `F_t(y) − F_t(ŷ) − (y−ŷ)·log_t(ŷ)` gives `0.062176`, and the second closed form gives `0.062176`. They match, so the algebra is right. Now the limiting cases that should anchor this in a family I already trust: at `t → 1` the divergence should be KL, and at `t → 0` it should be squared Euclidean. Numerically, at `t = 1 − 10⁻⁷` the Bregman value is `0.087660` against `KL(y‖ŷ) = Σ(y log(y/ŷ) − y + ŷ) = 0.087660`; at `t = 10⁻⁹` it is `0.030000` against `½‖y − ŷ‖² = 0.030000`. So this deformation is the β-divergence axis with `β = 2 − t`, KL at `t = 1` and squared Euclidean at `t = 0` — it never leaves the Bregman family I started in, it just slides along it.

Does choosing `t < 1` for this divergence actually bound the loss, the way the floor on `log_t` suggested? I'll defer the precise number until I have the transfer function, because the loss on a one-hot label couples the divergence to the predicted probabilities, and I want to compute the saturation on the *whole* loss, not just the `−log_{t}` term.

Now the transfer function. In the original construction the softmax is the dual-gradient of `F` restricted to the simplex, i.e. the `ŷ` solving the link `∇F(ŷ) = â − λ·1` under `Σ ŷ_i = 1`. Repeat that with `F_t`: I want `log_t(ŷ) = â − λ_t(â)·1`, and inverting with `exp_t`,

  `ŷ_i = exp_t(â_i − λ_t(â))`, with `λ_t(â)` chosen so that `Σ_i exp_t(â_i − λ_t(â)) = 1`.

This is the tempered softmax — and notice the temperature that lands here is the one I will set `> 1` for heavy tails, while the divergence above carries the temperature I will set `< 1` for boundedness. So the two dials did fall onto the two dual halves, as the inversion symmetry hinted. There is a catch, and it is the price of leaving `t = 1`: when `t ≠ 1` there is *no closed form* for `λ_t(â)`. With ordinary softmax, `λ = log Σ exp(â_i)` falls out for free because `exp(a − λ) = exp(a)/exp(λ)` factors the constraint. The tempered case does not factor — `exp_t(a + b) ≠ exp_t(a) exp_t(b)` — so the partition function does not separate from the activations, and `λ_t(â)` has to be solved numerically, per example. For `t > 1` there is a fixed-point iteration: set `μ = max(â)`, keep `ã_0 = â − μ`, initialize `ã = ã_0`, then repeatedly compute `Z = Σ_i exp_t(ã_i)` and reset `ã = ã_0 Z^{1−t}`; after it settles, `λ_t(â) = −log_t(1/Z) + μ`.

Let me actually run that iteration and check two things — that it converges and that its output is a valid probability vector. With `t = 1.2` and `â = (2.0, 0.5, −1.0, 3.0)`, five iterations give `p = (0.2630, 0.0935, 0.0397, 0.6037)`, summing to `1.0000`. So the normalizer does its job. And the heavy tail — does it really leave more mass on the far-wrong class? Ordinary softmax on the same `â` gives `(0.2506, 0.0559, 0.0125, 0.6811)`; on the lowest-activation class the tempered softmax holds `0.0397` versus the ordinary `0.0125`, three times the mass. So `t > 1` does fatten the lower tail, exactly the behavior the second disease needs.

Putting the two halves together: take the tempered softmax with temperature `t_2` as the transfer function, and score it with the tempered Bregman divergence at temperature `t_1`,

  `L(â | y) = Δ_{F_{t_1}}( y, exp_{t_2}(â − λ_{t_2}(â)) )`.

When `t_1 = t_2` this is just the matching loss at one temperature, so the duality holds and the loss is convex. But I want both cures at once, which means *mismatching*: a smaller `t_1 < 1` on the divergence for boundedness, a larger `t_2 > 1` on the transfer function for heavy tails. For a one-hot label with true class `c`, only the `i = c` term has `y_i = 1` and the rest have `y_i = 0`, so the divergence collapses to

  `L = −log_{t_1}(ŷ_c) − (1/(2 − t_1))(1 − Σ_i ŷ_i^{2−t_1})`,

with `ŷ = exp_{t_2}(â − λ_{t_2}(â))`. The first term is the tempered log-loss on the true class; the second is a normalization term over all classes — it is what makes the whole thing a genuine Bregman divergence and not just a clipped log.

Now I can finally compute the boundedness I deferred, on the whole loss, against the same outlier setup that defeated cross-entropy. Set `t_1 = 0.5`, `t_2 = 1.2`, true class 0, and drive the wrong class's activation `w` up. Cross-entropy gave `5.01, 20.0, 100.0` at `w = 5, 20, 100`. The bi-tempered loss gives `1.622, 1.964, 1.999`, and at `w = 10⁶` it is `2.000000` — it saturates. So the loss really is bounded; one mislabeled point can contribute at most a fixed amount. And the saturation value is not mysterious: as `ŷ_c → 0`, `−log_{t_1}(ŷ_c) → −(0 − 1)/(1 − t_1) = 1/(1 − t_1) = 2`, while `ŷ → (0,1)` sends the normalization term `−(1/(2−t_1))(1 − Σ ŷ_i^{2−t_1}) → 0`, so the ceiling is exactly `1/(1 − t_1) = 2`, matching the numeric `2.000000`. The bound is the floor on `log_{t_1}` showing up directly as the loss ceiling. (More generally on the `L_{2−t}` ball of radius `B`, the strong-convexity/smoothness corollary gives `Δ ≤ 2B^{2−t}/(1−t)^2`; on the simplex `B = 1` and the per-example loss is capped.)

The mismatch makes the loss non-convex in the activations. I should check that rather than assume it, since "non-convex" is doing real work in the argument — it is what lets the loss stop chasing outliers, and it had better not also be hiding spurious flatness. Trace the loss along the line `â = (0, x)` as `x` sweeps `[−3, 6]`, with the mismatched pair `t_1 = 0.5, t_2 = 2.0`, and look at the discrete second difference: its minimum is `−1.9 × 10⁻⁵`, genuinely negative, so the loss is non-convex. Doing the same sweep with a *matched* pair `t_1 = t_2 = 1.2`, the minimum second difference is `+4.5 × 10⁻⁶`, nonnegative — convex, as the matching-loss theory says it must be. So the convexity is exactly controlled by the match/mismatch, and that is fine: convexity in the last layer was never convexity in the network's actual parameters, and it was the very thing buying me the outlier-sensitivity I am trying to remove.

Now the two properties a classification loss must keep. First, *properness*: I want the minimizer of the expected loss to be the true posterior, so that fitting the loss recovers `P(y|x)`. For a fixed `x`, the expected Bregman divergence differs from

  `Σ_i [ −η_i log_{t_1} p_i + (1/(2−t_1)) p_i^{2−t_1} ]`

only by terms that depend on the true posterior `η` and not on the model probabilities `p`. Sampling `y_n ~ η(·|x_n)` then gives the empirical objective `(1/N) Σ_n [ −log_{t_1} p_{y_n}(x_n) + (1/(2−t_1)) Σ_i p_i(x_n)^{2−t_1} ]`, again up to those model-independent constants — every term is a function of the model's probabilities and the *observed* label, with no hidden dependence on the unknown conditional. This is the place where building on the Bregman divergence rather than the Tsallis one earns its keep. The Tsallis route looks almost identical but uses `log_t(ŷ/y)`, and because `log_t(a/b) ≠ log_t(a) − log_t(b)` the ratio does not split into a model term and a label term; its unbiased estimator would need the true `P(y|x)` inside the logarithm, and approximating that conditional by 1 makes the estimator biased. So the choice of Bregman over Tsallis is exactly what keeps the empirical loss an unbiased, proper objective — load-bearing, not cosmetic. Second, *Bayes-risk consistency*: at the conditional-risk minimizer the predicted probabilities equal the posteriors `η_i`; the per-class normalization term does not depend on which class I am scoring, so it does not move the classwise argmin, and since `−log_{t_1}` is monotone decreasing for `0 ≤ t_1 < 1`, `argmin_i l_i = argmax_i p_i = argmax_i â_i = argmax_i η_i`. The loss still picks the Bayes-optimal class in the non-convex regime. So both properties survive the mismatch; the only thing the mismatch sacrifices is the convexity I just confirmed I wanted to lose.

What do I set the temperatures to? Each dial cures one disease, so the choice should track which noise I expect. Pure small-margin boundary noise wants tail-heaviness only — `t_1 = 1`, `t_2` large (e.g. 4). Pure large-margin outliers want boundedness only — `t_1` small (e.g. 0.2), `t_2 = 1`. Realistic noise has both, so I want both dials engaged, but not so aggressively that optimization suffers — very extreme temperatures flatten the loss surface and slow convergence. For a moderate-scale, moderately-noisy image-classification problem with a residual network, a mild pair like `t_1 = 0.8`, `t_2 = 1.2` should sit in the safe, effective range: enough boundedness and tail-heaviness to resist noise, gentle enough that training stays stable. Higher noise or smaller, cleaner-architecture datasets push toward more extreme values; large clean datasets want the dials near 1. In practice this is a two-dimensional grid search, `t_1 ∈ [0.5, 1)` and `t_2 ∈ (1, 4]`.

Now the implementation, grounded in the tempered primitives. I need `log_t` and `exp_t` (with a `relu`/`clamp` realizing `[·]_+`), the fixed-point normalizer for `t_2 > 1`, the tempered softmax, and the one-hot Bregman divergence. I also want the backward pass to use the closed-form derivative rather than whatever gradient leaks out of a finite number of fixed-point iterations. Differentiate the constraint `Σ_j p_j = 1` with `p_j = exp_{t_2}(a_j − λ)`: using `d exp_t(u)/du = exp_t(u)^t`, the implicit-function theorem gives `∂λ/∂a_i = p_i^{t_2}/Σ_j p_j^{t_2}`, the escort distribution. The loss derivative with respect to `p_j` is `(p_j − y_j)p_j^{−t_1}`, so the chain rule through `p_j = exp_{t_2}(a_j − λ(a))` gives

  `∂L/∂a_i = (p_i − y_i)p_i^{t_2−t_1} − q_i Σ_j (p_j − y_j)p_j^{t_2−t_1}`,

where `q_i = p_i^{t_2}/Σ_j p_j^{t_2}`. Before I commit this to a custom backward, let me check it against finite differences of the forward loss — including the implicit `λ(a)` dependence, which is exactly where a hand-derived gradient tends to be wrong. At `t_1 = 0.7, t_2 = 1.4`, true class 1, `â = (0.3, −0.7, 1.1, 0.0)`, the closed form gives `(0.0558, −0.2393, 0.1426, 0.0408)` and central differences of the full forward loss give `(0.0558, −0.2393, 0.1426, 0.0408)`, agreeing to `3 × 10⁻¹⁰`. So the escort-distribution gradient is correct, implicit `λ` and all, and it is safe to drop it into the PyTorch port; when `t_1 = t_2 = 1`, I skip all of this and call ordinary cross-entropy.

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

The chain in one breath: softmax cross-entropy fails in two distinct ways under label noise — its unbounded `log` lets large-margin mislabeled points add loss linearly in their margin (I watched `−log p_0` track `5, 20, 100`), and its light-tailed softmax forces near-boundary noise toward 0/1. Both pieces are dual halves of one convex object — softmax is the dual-gradient of the negative entropy `F`, cross-entropy is its Bregman divergence — so I deformed `F` as a whole rather than hacking the pieces. A tempered logarithm floored at `−1/(1−t)` for `t_1 < 1` makes the loss bounded (numerically saturating at `1/(1−t_1) = 2`); its inverse, the tempered exponential heavy-tailed for `t_2 > 1`, makes the softmax non-saturating (three times the mass on a far-wrong class) at the price of a per-example normalizer with no closed form, which a fixed-point iteration solves to a valid probability vector in five steps. Mismatching `t_1 < 1 < t_2` gives a bounded, heavy-tailed loss whose second difference goes negative (non-convex, confirmed) while the matched case stays convex, and the loss remains proper — because it is built on the Bregman, not the Tsallis, divergence — and Bayes-risk consistent. The PyTorch port uses the escort-distribution gradient, checked against finite differences to `10⁻¹⁰`, with an ordinary cross-entropy branch at `(t_1, t_2) = (1, 1)`.
