# Lookahead

**Problem.** SGD-style optimizers are fragile to the learning rate: larger useful steps can contract the bias of the iterates quickly but inflate their steady-state variance and can overshoot sharp directions, so getting good performance needs expensive learning-rate tuning. Adaptive (Adam, AdaGrad) and accelerated (momentum) methods reshape the gradient but still need tuning and do not target iterate variance directly.

**Key idea.** Run *any* inner optimizer for `k` steps on a set of **fast weights**, then take **slow weights** one step toward the final fast weights by linear interpolation, and restart the fast weights from the new slow weights. The slow weights are thus an exponential moving average of the *final* fast weight of each window — recent-weighted, with no "when to start averaging" decision — and, unlike a passive Polyak/SWA average, the average feeds back into the trajectory because the next window starts from it. It wraps the existing optimizer in one extra line, costs one extra parameter copy and `O((k+1)/k)` work, and is orthogonal to adaptive/accelerated schemes.

**Algorithm.** Slow weights φ, fast weights θ, inner optimizer A, window k, step size α:

```
for t = 1, 2, ...:
    θ ← φ                              # sync
    for i = 1..k:
        sample minibatch d ~ D
        θ ← θ + A(L, θ, d)             # k steps forward
    φ ← φ + α (θ − φ)                  # 1 step back: interpolate toward fast weights
return φ
```

Equivalently, after T completed windows the slow weights are the EMA φ_T = (1−α)^T φ₀ + α Σ_{j=0}^{T−1}(1−α)^{T−1−j}θ_{j,k}. Defaults: k = 5, α = 0.8 (α = 1 recovers the inner optimizer). Inner-optimizer state (momentum) is maintained by default; it may alternatively be reset or interpolated.

**Why it works.** On the noisy quadratic model L̂(x) = ½(x−c)ᵀA(x−c), c ~ N(0,Σ), with SGD as the inner optimizer at learning rate γ, the slow weights converge to a strictly smaller steady-state variance than SGD at the *same* γ:

V*_LA = α²(I−(I−γA)^{2k}) / [α²(I−(I−γA)^{2k}) + 2α(1−α)(I−(I−γA)^k)] · V*_SGD,

and the leading factor is < 1 per coordinate for α ∈ (0,1) whenever the inner SGD dynamics are stable, |1−γaᵢ| < 1. In non-oscillatory coordinates with 0 ≤ (1−γaᵢ)^k < 1, the cost is a slower bias contraction (factor 1−α+α(1−γaᵢ)^k vs (1−γaᵢ)^k); in sign-flipping coordinates the interpolation can also damp oscillation. Deterministically (momentum on a quadratic), the start→end interpolation cuts across oscillations and improves the convergence rate in the under-damped regime.

**Choosing α.** For a quadratic with optimum x*, the loss-minimizing interpolation weight is α* = (θ₀−x*)ᵀA(θ₀−θ_k) / [(θ₀−θ_k)ᵀA(θ₀−θ_k)], with the degenerate θ₀ = θ_k case leaving α irrelevant. Approximating x* by a Newton step with a diagonal empirical-Fisher curvature Â gives an adaptive, clipped estimate, but a fixed α avoids maintaining Â and keeps the wrapper optimizer-agnostic.

**Code.** A PyTorch wrapper around `torch.optim.Optimizer`; the fast weights are the live parameters the inner optimizer mutates, the slow weights are cached per parameter.

```python
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        # optimizer: any inner optimizer A (SGD, Adam, ...). We only ever call its .step().
        self.optimizer = optimizer
        self._la_step = 0                 # counts inner steps within the current window
        self.la_alpha = la_alpha          # interpolation weight alpha; alpha=1 -> just the inner optimizer
        self._total_la_steps = la_steps   # window length k
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)
        # slow weights phi: a cached copy of the parameters, taken at sync time
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        # swap in the slow weights phi for evaluation
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        # k steps forward: let the inner optimizer take one fast-weight step
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    # 1 step back: phi <- phi + alpha (theta_k - phi), in place on p.data.
                    # p = alpha*theta_k + (1-alpha)*phi  is exactly that interpolation.
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)
                    # commit: the new phi becomes the sync point and the next window's start
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        # interpolate the inner momentum buffer the same way as the params
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
        return loss
```
