# Context

## Research question

Gradient descent updates parameters by x_{t+1} = x_t − η·g_t, and the single most consequential knob is the global learning rate η. Set it too high and the objective diverges; set it too low and learning crawls. Picking a good η is "more art than science" — it requires a tuning sweep, and the right value depends on the model, the data, and even the layer (gradients in different layers of a deep network can differ by orders of magnitude). The goal is a method that removes this burden: a *per-dimension* learning rate computed automatically from first-order information only, with negligible overhead over plain SGD, no manually-set global learning rate, and enough robustness that its few remaining hyperparameters barely matter across architectures, data modalities, and noise levels (including the noisy gradients of a distributed parameter-server setup).

## Background

The simplest update is SGD: Δx_t = −η·g_t, with one global η shared across all dimensions and held fixed or annealed by a hand-tuned schedule. The most powerful correction is Newton's method, Δx_t = H_t^{-1} g_t, which gives the optimal step on a quadratic by dividing the gradient by the curvature — but the Hessian H is far too large to form or invert for big models. Everything practical lives between these two: improve the use of first-order information, or cheaply approximate the second-order information.

Two refinements of first-order SGD are central. **Momentum** keeps an exponentially-decaying accumulation of past updates, Δx_t = ρ·Δx_{t-1} − η·g_t. On a long narrow valley, the gradients *along* the valley are small but consistently signed, so they accumulate and speed progress; the gradients *across* the valley are large but flip sign, so they cancel in the accumulation and the oscillation is damped. This happens per-dimension, so progress along the valley is not slowed by damping across it — something a single global η cannot achieve. **AdaGrad** gives each dimension its own dynamic rate: Δx_t = −η / √(Σ_{τ=1}^{t} g_τ²) · g_t. The per-dimension denominator is the ℓ₂ norm of *all* past gradients in that coordinate, so dimensions with large gradients get small rates and dimensions with small gradients get large rates — this "evens out" progress across dimensions (valuable when layer gradients span orders of magnitude) and acts like automatic annealing (the denominator only grows). AdaGrad has the second-order-flavored benefit of per-dimension scaling using only first-order quantities.

But AdaGrad has two diagnosable flaws, both stemming from accumulating squared gradients from the very first iteration. (1) Every term is positive, so the denominator grows without bound and the effective learning rate decays monotonically toward zero — eventually training stops making progress entirely. (2) The accumulation is dominated by early, possibly large, gradients, so AdaGrad is sensitive to the initial conditions (large initial gradients permanently depress the rate) and to the global η that is supposed to counteract this — reintroducing exactly the tuning problem one wanted to avoid.

On the second-order-approximation side, **Becker & LeCun (1988)** use a diagonal approximation to the Hessian, Δx_t = −1/(|diag(H_t)| + μ)·g_t, where the absolute value keeps the step in the descent direction and μ conditions regions of small curvature; computing diag(H) costs an extra forward/backward pass (roughly doubling SGD's cost). **Schaul et al. (2012)** combine a diagonal Hessian with AdaGrad-like windowed gradient statistics, Δx_t = −(1/|diag(H_t)|)·(E[g_{t-w:t}]² / E[g²_{t-w:t}])·g_t, again requiring the diagonal Hessian (extra cost) plus a window-size heuristic.

A unit-analysis observation underlies the eventual fix and is worth stating as a pre-existing fact about all these rules. A parameter update Δx ought to carry the *same units* as the parameter x. For SGD/momentum, units(Δx) ∝ units(g) = units(∂f/∂x) ∝ 1/units(x) (taking f unitless) — wrong. For AdaGrad (and any rule whose update is a ratio of gradient quantities), the update is *unitless* — also wrong. Only second-order rules get it right: Δx ∝ H^{-1}g ∝ (∂f/∂x)/(∂²f/∂x²) ∝ units(x). Whatever fixes AdaGrad's two flaws should also, if it is to be principled, repair this unit mismatch.

## Baselines

**SGD (Robbins & Monro 1951).** Δx_t = −η·g_t. One global, hand-tuned η. Gap: no per-dimension adaptation; brittle to η.

**Momentum.** Δx_t = ρ·Δx_{t-1} − η·g_t. Accelerates consistent directions, damps oscillating ones, per-dimension. Gap: still has a global η to tune; can overshoot/oscillate near minima without annealing.

**AdaGrad (Duchi et al. 2011).** Δx_t = −η/√(Σ_{τ≤t} g_τ²)·g_t. Per-dimension rate, evens out progress across dimensions, anneals automatically. Gaps: the all-time denominator decays the rate to zero and stalls training; sensitive to initial gradients and still needs a global η.

**Diagonal-Hessian methods (Becker & LeCun 1988; Schaul et al. 2012).** Δx_t = −g_t/(|diag(H)| + μ), and the windowed variant. Correct units, curvature-aware. Gap: require the diagonal Hessian, i.e. an extra forward/backward pass per step, plus (for Schaul) a window heuristic.

## Evaluation settings

Pre-existing yardsticks: handwritten-digit classification on MNIST with a fully-connected network (500 then 300 hidden units, tanh or rectified-linear activations, softmax output, cross-entropy loss, minibatches of 100), measuring test error vs epochs and the *sensitivity* of test error to hyperparameters (sweeping the learning rate for the baselines and the decay ρ and ε for the new method); and large-vocabulary acoustic modeling on several hundred hours of US-English speech (26 frames × 40 log-energy filterbank inputs, 4 hidden layers of 2560 units, logistic or rectified-linear, 8000 senone output labels from a GMM-HMM forced alignment), trained in a distributed parameter-server setup with 100 or 200 asynchronous replicas — chosen specifically because asynchronous replicas inject substantial gradient noise, testing robustness. Metric: frame classification accuracy on a held-out set vs training progress.

## Code framework

The substrate is the ordinary per-parameter optimizer loop: for each parameter, read its gradient, maintain a small amount of per-dimension running state, compute an update, and apply it. SGD needs no state; AdaGrad needs one accumulator per dimension. A per-dimension adaptive method slots in as the body that maintains its running statistics and forms the update.

```python
import torch
from torch.optim.optimizer import Optimizer


class PerDimAdaptive(Optimizer):
    def __init__(self, params, rho=0.9, eps=1e-6):
        defaults = dict(rho=rho, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            rho, eps = group['rho'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    # TODO: per-dimension running accumulators the method needs,
                    #       all initialized to zero
                    pass
                # TODO: update the running statistic(s) of the gradient,
                #       form the per-dimension update `delta` (no global learning rate),
                #       apply x <- x + delta, and update any running statistic of the update
                pass
        return None
```
