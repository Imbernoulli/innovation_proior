# Context

## Research question

When training a deep network with a stochastic first-order method, the optimizer reshapes the raw
gradient using accumulated statistics — a momentum term that smooths the descent direction, and (for
adaptive methods) a per-coordinate scale. The widely used adaptive method maintains a decaying average
of the gradient (its first moment) and steps along that smoothed direction, rescaled by a decaying
average of the squared gradient. The momentum it uses is *classical* momentum — a running combination
of past gradients applied as-is. But there is long-standing evidence that a different momentum, the
accelerated kind, gives a better descent direction and a provably better convergence rate. The
question is whether the adaptive method's momentum can be upgraded from classical to accelerated
without disturbing its adaptive per-coordinate rescaling — and whether that upgrade actually speeds
convergence and improves the final model in practice.

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
μ mₜ₋₁ does not depend on the current gradient, so a higher-quality direction is obtained by computing
the gradient *after* taking the momentum step, at the look-ahead point θₜ₋₁ − μ mₜ₋₁:
gₜ = ∇fₜ(θₜ₋₁ − μ mₜ₋₁); mₜ = μ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α mₜ. On convex, smooth problems this attains
an O(1/k²) rate, better than gradient descent's O(1/k); Sutskever et al. 2013 give empirical evidence
it beats classical momentum, plain SGD, and Hessian-free methods on hard deep-learning objectives.
Sutskever et al. also supply a reformulation that avoids stepping to the look-ahead point and back:
applying the *next* timestep's momentum once during the current update,
gₜ = ∇fₜ(θₜ₋₁); mₜ = μₜ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α(μₜ₊₁ mₜ + gₜ) — now both the momentum step and the
gradient step depend on the current gradient.

**Adaptive moment estimation (Adam, Kingma & Ba 2014).** Track an exponential moving *mean* of the
gradient mₜ = μ mₜ₋₁ + (1−μ) gₜ and of the squared gradient nₜ = ν nₜ₋₁ + (1−ν) gₜ², correct each for
its initialization-to-zero bias (m̂ₜ = mₜ/(1−μᵗ), n̂ₜ = nₜ/(1−νᵗ)), and step
θₜ = θₜ₋₁ − α m̂ₜ/(√n̂ₜ + ε). Adam's momentum is the classical kind (a decaying combination of past
gradients), expressed as a mean rather than a sum; the adaptive component is the per-coordinate
division by √n̂ₜ. Using past *gradients* (a mean) rather than past *updates* (a sum) lets the direction
keep adapting even after the learning rate has annealed near the end of training. Kingma & Ba report
Adam outperforming several alternatives, including NAG, on a handful of benchmarks.

The two enhancements have not been combined: Adam carries classical momentum, and the accelerated
momentum lives in non-adaptive SGD. The diagnostic comparison from the prior work gives the relevant
gap — accelerated momentum gives a better direction than classical momentum on the same objective —
which sets up the substitution.

**Decoupled weight decay (AdamW, Loshchilov & Hutter 2019).** A separate but composable choice for the
regularizer. Folding an L2 penalty into the gradient gₜ ← gₜ + λθₜ is *not* equivalent to weight decay
under an adaptive method, because the penalty term then gets divided by √n̂ₜ and is rescaled
inconsistently across coordinates. Decoupling it — shrinking the weights directly,
θ ← θ − α(update + λθ), i.e. θ ← (1 − αλ)θ before the adaptive step — restores the intended uniform
shrinkage.

## Baselines

**SGD with classical momentum (Polyak 1964).** mₜ = μ mₜ₋₁ + gₜ; θₜ = θₜ₋₁ − α mₜ. Smooths the
direction and accelerates low-curvature progress. Gap: the momentum step ignores the current gradient,
so the direction it commits to can be improved; no per-coordinate adaptivity, so a single α must serve
coordinates of very different gradient scale.

**NAG / simplified NAG (Nesterov 1983; Sutskever et al. 2013).** Evaluates the gradient at the
look-ahead point (or, in the reformulation, applies the next momentum step using the current
gradient), yielding a better direction and the accelerated convex rate. Gap: still no adaptive
per-coordinate rescaling; the accelerated momentum has never been carried inside an adaptive method.

**Adam (Kingma & Ba 2014).** mₜ = μ mₜ₋₁ + (1−μ)gₜ; nₜ = ν nₜ₋₁ + (1−ν)gₜ²; bias-correct;
θₜ = θₜ₋₁ − α m̂ₜ/(√n̂ₜ + ε). Adaptive and robust default. Gap: its momentum is classical — the step
commits to μ mₜ₋₁ without consulting the current gradient — leaving the accelerated-momentum
improvement on the table.

**AdamW (Loshchilov & Hutter 2019).** Adam with the weight decay decoupled from the gradient,
θ ← (1 − αλ)θ then the adaptive step. Fixes the L2-under-Adam distortion; the momentum is still
classical.

## Evaluation settings

The natural yardsticks are training and validation loss / error versus epochs (and versus wall-clock)
on the standard supervised objectives where Adam is the default — image
reconstruction / classification and language modeling — comparing optimizers under matched learning
rates and schedules, with the learning rate swept per optimizer and the remaining hyperparameters
(momentum coefficient, second-moment decay, ε) at their usual values. The metric is convergence speed
to a given loss and the final loss reached; no single value is fixed in advance.

## Code framework

The substrate is a standard PyTorch optimizer plugged into an ordinary training loop. The first- and
second-moment buffers, a scalar product for any scheduled first-moment bias correction, and the
weight-decay branch already exist as primitives; the open slot is the *form of the parameter update* —
specifically how the first moment is combined with the current gradient when forming the step.

```python
import torch
from torch.optim.optimizer import Optimizer


class AdaptiveMomentOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, first_moment_schedule_rate=None,
                 decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        first_moment_schedule_rate=first_moment_schedule_rate,
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
                    state["first_moment_product"] = torch.tensor(1.0)
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

                # TODO: update any first-moment product required by the chosen scheduled
                #       momentum rule, then form the bias-corrected step combining m with
                #       the current gradient g, divided by √n_hat + eps, and apply it.
                pass
        return None
```
