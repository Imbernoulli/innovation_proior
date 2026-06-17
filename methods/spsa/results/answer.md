# SPSA: a gradient-free L_inf adversarial attack

## Problem

Evaluate adversarial robustness when the model's gradient is unavailable (non-differentiable
preprocessing) or untrustworthy (gradient masking). A gradient-based attack like PGD can be
defeated by withholding a usable gradient rather than by removing adversarial examples, so a
PGD "pass" is weak evidence of robustness. We want a strong `L_inf` attack that uses **only
forward queries** to the model (logits), works in high input dimension `D = C·H·W`, and
tolerates a stochastic/noisy objective.

## Key idea

Estimate the gradient of the logit-margin objective with **simultaneous-perturbation stochastic
approximation (SPSA)**: perturb *all* input coordinates at once with a single random
Rademacher vector `v ∈ {+1, -1}^D`, take a two-sided finite difference along it, and read each
coordinate's partial derivative back out — using only **two function evaluations regardless of
`D`**. Average several such estimates to cut variance, descend with Adam on the perturbation,
and project back into the `L_inf` ball and `[0, 1]`.

## The estimator

Objective (untargeted, minimized): the correct-class advantage
`J(x) = m(x)_{y0} - max_{j ≠ y0} m(x)_j`, where `m(x)_j` is the logit of class `j`; `J < 0`
iff misclassified. The code-facing margin is `M(x) = -J(x) = max_{j ≠ y0} m(x)_j - m(x)_{y0}`,
which is positive after a successful untargeted attack. The margin form is preferred over
cross-entropy because logits stay roughly linear where the softmax has saturated.

For a random `v` with independent, mean-zero, **finite-inverse-moment** components and probe
radius `delta`, the two-sided directional difference is

```
df = J(x + delta·v) - J(x - delta·v) = 2·delta·(v · ∇J) + O(delta^3),
```

so the per-coordinate estimate is

```
ghat_i = df / (2·delta·v_i) = ∂_i J + Σ_{j≠i} (v_j/v_i)·∂_j J + O(delta^2).
```

Taking expectations, `E[v_j/v_i] = E[v_j]·E[1/v_i] = 0` for `j ≠ i` (independence + mean zero),
so `E[ghat_i] = ∂_i J + O(delta^2)` — almost unbiased — from just `J(x±delta·v)`, two queries,
**independent of `D`**. (Compare coordinatewise finite differences, which cost `2D` queries.)
Asymptotically, SPSA attains the same per-iteration mean-squared error as the `2D`-query
finite-difference scheme while using a factor of `D` fewer evaluations: the random per-step
"misdirections" average out across iterations.

**Why Rademacher, not Gaussian.** The cancellation needs `E[1/v_i]` to exist as a finite
expectation, and the usual regularity condition requires finite inverse moments such as
`E[|1/v_i|]` and `E[1/v_i^2]`. For `v_i ~ N(0,1)` (or uniform around 0), those inverse moments
diverge because the distribution puts mass arbitrarily close to zero, so the estimator produces
wild terms when some `v_i ≈ 0`. The symmetric Bernoulli `v_i ∈ {+1, -1}` has bounded inverse
moments of every order. As a bonus, since `v_i = ±1`, `1/v_i = v_i`, so dividing by `v_i` is
just multiplying by `v_i`: `ghat = (df / 2·delta)·v`.

**Variance reduction.** Average `n` independent estimates per step, `ghat_bar = (1/n) Σ ghat^{(i)}`,
each with its own `v^{(i)}`; variance falls like `1/n`, and the `2n` forward passes batch on a
GPU. Cost per step is `2n` queries; `n` is the strength/budget dial.

**Descent and projection.** Run Adam on the perturbation `dx` with `ghat_bar` as the gradient
(Adam's per-coordinate scaling and smoothing handle the noisy, unevenly-scaled estimate, and
converge faster than a raw step). The torchattacks/advertorch loop initializes `dx = 0`; after
each step, it clamps `dx` to `[-eps, eps]` (Euclidean projection onto the `L_inf` box) and
`x0 + dx` to `[0, 1]`. The probe `delta` trades Taylor bias `O(delta^2)` against the model's
noise floor (`df` scales with `delta`, noise is fixed), so a small but nonzero value (~0.01 on
`[0,1]` pixels) is best.

## Algorithm

```
Input: model m, image x0, label y0, eps, probe delta, lr, iterations T, samples n
dx <- 0;   opt <- Adam([dx], lr)
for t = 1..T:
    # SPSA gradient of the attack objective, two queries per sample, batched:
    g <- 0
    repeat n times (in GPU batches):
        v   <- Rademacher(+-1), one value per pixel, shared across channels
        df  <- loss(m(x0+dx + delta*v), y0) - loss(m(x0+dx - delta*v), y0)
        g   <- g + (df / (2*delta)) * v          # = df/(2*delta*v), since v in {+-1}
    g <- g / n
    dx.grad <- g;  opt.step()                     # Adam descent on the perturbation
    dx <- clamp(dx, -eps, eps);  dx <- clamp(x0+dx, 0, 1) - x0    # project to box & [0,1]
return x0 + dx
```

For an untargeted attack the implementation's `MarginalLoss` is `M = max_wrong - true`; Adam
minimizes `-M`, which is the correct-class advantage `J` and pushes the best wrong logit above
the true logit. For a targeted attack, the sign is reversed by the attack wrapper.

## Reference implementation

Faithful to the canonical implementation (torchattacks / advertorch):

```python
import torch
from torch.nn.modules.loss import _Loss


class MarginalLoss(_Loss):
    def forward(self, logits, targets):
        top_logits, top_classes = torch.topk(logits, 2, dim=-1)
        target_logits = logits[torch.arange(logits.shape[0]), targets]
        max_nontarget_logits = torch.where(
            top_classes[..., 0] == targets, top_logits[..., 1], top_logits[..., 0],
        )
        loss = max_nontarget_logits - target_logits
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError("unknown reduction: '%s'" % (self.reduction,))


class SPSA:
    def __init__(self, model, eps=0.3, delta=0.01, lr=0.01,
                 nb_iter=1, nb_sample=128, max_batch_size=64):
        self.model = model
        self.eps, self.delta, self.lr = eps, delta, lr
        self.nb_iter, self.nb_sample = nb_iter, nb_sample
        self.max_batch_size = max_batch_size
        self.loss_fn = MarginalLoss(reduction="none")
        self.targeted = False

    def loss(self, logits, y):
        m = self.loss_fn(logits, y)
        return m if self.targeted else -m

    def linf_clamp_(self, dx, x, eps):
        dx_clamped = torch.clamp(dx, min=-eps, max=eps)
        x_adv = torch.clamp(x + dx_clamped, min=0.0, max=1.0)
        dx += x_adv - x - dx
        return dx

    def _batch_sizes(self, n, mbs):
        b = [mbs] * (n // mbs)
        if n % mbs:
            b.append(n % mbs)
        return b

    @torch.no_grad()
    def spsa_grad(self, images, labels, delta, nb_sample, max_batch_size):
        grad = torch.zeros_like(images)
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)

        def f(xv, yv):
            return self.loss(self.model(xv), yv)

        images = images.expand(max_batch_size, *images.shape[1:]).contiguous()
        labels = labels.expand(max_batch_size, *labels.shape[1:]).contiguous()
        v = torch.empty_like(images[:, :1, ...])               # one v per pixel
        for bs in self._batch_sizes(nb_sample, max_batch_size):
            x_, y_ = images[:bs], labels[:bs]
            vb = v[:bs].bernoulli_().mul_(2.0).sub_(1.0)        # Rademacher +-1
            v_ = vb.expand_as(x_).contiguous()                 # shared across channels
            x_shape = x_.shape
            x_ = x_.reshape(-1, *images.shape[2:])
            y_ = y_.reshape(-1, *labels.shape[2:])
            v_ = v_.reshape(-1, *v.shape[2:])
            df = f(x_ + delta * v_, y_) - f(x_ - delta * v_, y_)
            df = df.view(-1, *([1] * (v_.dim() - 1)))
            grad_ = (df / (2.0 * delta * v_)).view(x_shape).sum(dim=0)
            grad += grad_
        grad /= nb_sample
        return grad

    def perturb(self, x, y):
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(x + dx, y, self.delta,
                                     self.nb_sample, self.max_batch_size)
            optimizer.step()
            dx = self.linf_clamp_(dx, x, self.eps)
        return x + dx
```

With a per-sample query budget `B`, set `nb_iter = B // (2·nb_sample)` since each iteration
spends `2·nb_sample` forward queries. Typical settings: `delta = 0.01`, `lr = 0.01`,
`nb_sample` from 128 (query-efficient) up to several thousand (maximum attack strength).
