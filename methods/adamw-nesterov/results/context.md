# Context

## Research question

When training a deep network with a stochastic first-order method, the optimizer reshapes the raw
gradient using accumulated statistics — a momentum term that smooths the descent direction, and (for
adaptive methods) a per-coordinate scale. The widely used adaptive method maintains a decaying average
of the gradient (its first moment) and steps along that smoothed direction, rescaled by a decaying
average of the squared gradient. Momentum itself comes in more than one form: the classical running
combination of past gradients, and the accelerated variant that uses a look-ahead gradient. The
question is how to form the per-step parameter update from these accumulated gradient statistics, and
how the choice of momentum form interacts with per-coordinate adaptive rescaling in terms of
convergence speed and final model quality.

## Background

The template every method here specializes is plain stochastic gradient descent: at step *t*, compute
the gradient gₜ = ∇fₜ(θₜ₋₁) and step θₜ = θₜ₋₁ − α gₜ. Two enhancements sit on top of it.

**Classical momentum (Polyak 1964).** Accumulate a decaying sum of past gradients into a velocity
vector mₜ = μ mₜ₋₁ + gₜ and step θₜ = θₜ₋₁ − α mₜ. Reading the update as
θₜ = θₜ₋₁ − α(μ mₜ₋₁ + gₜ), it is a step in the *previous* velocity direction plus a step along the
*current* gradient. It moves faster along low-curvature directions where the gradient is small but
consistent, and damps oscillation along high-curvature directions where the gradient keeps flipping
sign (Sutskever et al. 2013).

**Nesterov's accelerated gradient (Nesterov 1983; Sutskever et al. 2013).** The momentum step
μ mₜ₋₁ does not depend on the current gradient; this variant computes the gradient *after* taking the
momentum step, at the look-ahead point θₜ₋₁ − μ mₜ₋₁:
gₜ = ∇fₜ(θₜ₋₁ − μ mₜ₋₁); mₜ = μ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α mₜ. On convex, smooth problems this attains
an O(1/k²) rate, compared with gradient descent's O(1/k); Sutskever et al. 2013 give empirical evidence
it beats classical momentum, plain SGD, and Hessian-free methods on hard deep-learning objectives. As
written, the iteration differentiates the objective at the look-ahead point θₜ₋₁ − μ mₜ₋₁, while an
ordinary training loop runs its forward and backward pass at the current parameters θₜ₋₁.

**Adaptive moment estimation (Adam, Kingma & Ba 2014).** Track an exponential moving *mean* of the
gradient mₜ = μ mₜ₋₁ + (1−μ) gₜ and of the squared gradient nₜ = ν nₜ₋₁ + (1−ν) gₜ², correct each for
its initialization-to-zero bias (m̂ₜ = mₜ/(1−μᵗ), n̂ₜ = nₜ/(1−νᵗ)), and step
θₜ = θₜ₋₁ − α m̂ₜ/(√n̂ₜ + ε). Adam's momentum is the classical kind (a decaying combination of past
gradients), expressed as a mean rather than a sum; the adaptive component is the per-coordinate
division by √n̂ₜ. Using past *gradients* (a mean) rather than past *updates* (a sum) lets the direction
keep adapting even after the learning rate has annealed near the end of training. Kingma & Ba report
Adam outperforming several alternatives, including NAG, on a handful of benchmarks.

Adam carries classical momentum; the accelerated momentum is used in non-adaptive SGD.

**Decoupled weight decay (AdamW, Loshchilov & Hutter 2019).** A separate but composable choice for the
regularizer. Folding an L2 penalty into the gradient gₜ ← gₜ + λθₜ is *not* equivalent to weight decay
under an adaptive method, because the penalty term then gets divided by √n̂ₜ and is rescaled
inconsistently across coordinates. Decoupling it — shrinking the weights directly,
θ ← θ − α(update + λθ), i.e. θ ← (1 − αλ)θ before the adaptive step — restores the intended uniform
shrinkage.

## Baselines

**SGD with classical momentum (Polyak 1964).** mₜ = μ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α mₜ. Smooths the
direction and accelerates low-curvature progress; a single α serves all coordinates.

**NAG (Nesterov 1983; Sutskever et al. 2013).** Evaluates the gradient at the look-ahead point,
attaining the accelerated convex rate; a single α serves all coordinates, and the look-ahead
evaluation sits off the current parameters θₜ₋₁.

**Adam (Kingma & Ba 2014).** mₜ = μ mₜ₋₁ + (1−μ)gₜ; nₜ = ν nₜ₋₁ + (1−ν)gₜ²; bias-correct;
θₜ = θₜ₋₁ − α m̂ₜ/(√n̂ₜ + ε). Adaptive and robust default; its momentum is the classical kind.

**AdamW (Loshchilov & Hutter 2019).** Adam with the weight decay decoupled from the gradient,
θ ← (1 − αλ)θ then the adaptive step; the momentum is the classical kind.

## Evaluation settings

The natural yardsticks are training and validation loss / error versus epochs (and versus wall-clock)
on the standard supervised objectives where Adam is the default — image
reconstruction / classification and language modeling — comparing optimizers under matched learning
rates and schedules, with the learning rate swept per optimizer and the remaining hyperparameters
(momentum coefficient, second-moment decay, ε) at their usual values. The metric is convergence speed
to a given loss and the final loss reached; no single value is fixed in advance.

## Code framework

The substrate is a standard PyTorch optimizer plugged into an ordinary training loop. The first- and
second-moment buffers, a step counter, and the weight-decay branch already exist as primitives;
the open slot is the *form of the parameter update*.

```python
import torch
from torch.optim.optimizer import Optimizer


class AdaptiveMomentOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)       # first moment m
                    state["exp_avg_sq"] = torch.zeros_like(p)    # second moment n
                state["step"].add_(1)
                m, n = state["exp_avg"], state["exp_avg_sq"]

                # weight decay either shrinks the weights directly or enters the gradient
                if wd != 0:
                    if group["decoupled_weight_decay"]:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g.add(p, alpha=wd)

                # moment EMAs
                m.lerp_(g, 1 - beta1)
                n.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # TODO: form the bias-corrected parameter update from the moment
                #       buffers and apply it.
                pass
        return None
```
