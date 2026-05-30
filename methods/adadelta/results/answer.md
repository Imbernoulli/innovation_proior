# ADADELTA

**Problem.** Plain SGD needs a hand-tuned global learning rate η that is brittle (too high diverges, too low crawls) and cannot be right for every layer at once. ADADELTA computes a per-dimension learning rate automatically from first-order information only, with negligible overhead over SGD, and — crucially — *no manually-set global learning rate*.

**Key idea.** It is derived from AdaGrad by fixing two flaws and then enforcing dimensional consistency:
1. **Window the gradient accumulation.** AdaGrad's denominator sums g² from the first step, so the effective rate decays monotonically to zero and stalls. Replace the all-time sum with an exponentially-decaying average E[g²]_t = ρ·E[g²]_{t-1} + (1−ρ)·g_t², a *local* estimate that never runs to zero. Define RMS[g]_t = √(E[g²]_t + ε).
2. **Correct the units.** A parameter update should have the same units as the parameter. SGD/momentum updates have units of the gradient (∝ 1/x); AdaGrad's are unitless — both wrong. Only second-order updates Δx ∝ H⁻¹g ∝ (∂f/∂x)/(∂²f/∂x²) ∝ x are right. Rearranging Newton (diagonal H), 1/(∂²f/∂x²) = Δx/(∂f/∂x), shows the numerator needs a quantity with units of Δx. Since the current Δx_t is unknown, approximate it (locally-smooth curvature) by the RMS of *past* updates, RMS[Δx]_{t-1} = √(E[Δx²]_{t-1} + ε).

The two ideas give a learning-rate-free, per-dimension update equal to a first-order inverse-diagonal-Hessian approximation:

  Δx_t = − ( RMS[Δx]_{t-1} / RMS[g]_t ) · g_t.

**Algorithm.**

```
given decay ρ, constant ε
init E[g²]_0 = 0,  E[Δx²]_0 = 0
for t = 1, 2, ...:
    g_t      = ∇f(x_t)
    E[g²]_t  = ρ·E[g²]_{t-1} + (1−ρ)·g_t²
    Δx_t     = − ( RMS[Δx]_{t-1} / RMS[g]_t ) · g_t        # RMS[u] = √(u + ε)
    E[Δx²]_t = ρ·E[Δx²]_{t-1} + (1−ρ)·Δx_t²
    x_{t+1}  = x_t + Δx_t
```

Defaults: ρ = 0.9–0.95, ε ≈ 1e-6 (shown insensitive). Both accumulators are one number per dimension; total cost is one extra running average over SGD.

**Why it works.**
- Always follows the descent direction −g_t (like SGD); the RMS ratio is always positive.
- Numerator RMS[Δx] accumulates past updates over a window (an acceleration term, like momentum).
- Denominator RMS[g] uses per-dimension squared-gradient info to even out progress across dimensions (like AdaGrad) but is *windowed*, so it never decays to zero.
- The ratio approximates an inverse diagonal Hessian (like Becker–LeCun / Schaul) at the cost of a single gradient evaluation, no Hessian, no extra backward pass.
- The numerator lags the denominator by one step (Δx_t is unknown when forming the step), so a large sudden gradient spikes the denominator and shrinks the effective rate *this* step before the numerator reacts — robustness to gradient spikes and noise.
- The ε in the numerator bootstraps step 1 (where Δx_0 = 0) and keeps updates alive when past updates shrink. Late in training, ε dominates both RMS terms, the effective rate drifts toward 1, and updates smoothly tend to zero — an implicit annealing with no schedule.

**Code (PyTorch).** `square_avg` = E[g²], `acc_delta` = E[Δx²]; `lr` is an optional outer scale (lr = 1.0 is the pure method).

```python
import torch
from torch.optim.optimizer import Optimizer


class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rho, eps = group['rho'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)   # E[g^2]
                    state['acc_delta'] = torch.zeros_like(p.data)    # E[dx^2]
                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)   # E[g^2]_t
                std = square_avg.add(eps).sqrt_()                          # RMS[g]_t
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)    # RMS[dx]_{t-1}/RMS[g]_t * g
                p.data.add_(delta, alpha=-group['lr'])                     # x <- x - lr*delta
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)  # E[dx^2]_t

        return loss
```
