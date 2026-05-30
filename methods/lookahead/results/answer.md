# Lookahead

**Problem.** SGD-style optimizers are fragile to the learning rate: a large step contracts the bias of the iterates quickly but inflates their steady-state variance (the noisy jitter that holds the loss above its floor), so getting good performance needs expensive learning-rate tuning. Adaptive (Adam, AdaGrad) and accelerated (momentum) methods reshape the gradient but still need tuning and do not target iterate variance directly.

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

Equivalently the outer update is the EMA φ_{t+1} = α θ_{t,k} + α(1−α)θ_{t−1,k} + … + (1−α)^t φ₀. Defaults: k = 5, α = 0.8 (α = 1 recovers the inner optimizer). Inner-optimizer state (momentum) is maintained by default; it may alternatively be reset or interpolated.

**Why it works.** On the noisy quadratic model L̂(x) = ½(x−c)ᵀA(x−c), c ~ N(0,Σ), with SGD as the inner optimizer at learning rate γ, the slow weights converge to a strictly smaller steady-state variance than SGD at the *same* γ:

V*_LA = α²(I−(I−γA)^{2k}) / [α²(I−(I−γA)^{2k}) + 2α(1−α)(I−(I−γA)^k)] · V*_SGD,

and the leading factor is < 1 for α ∈ (0,1). The cost is a slightly slower bias contraction (factor 1−α+α(I−γA)^k vs (I−γA)^k), which is cheap in the large-learning-rate, variance-limited regime where networks are usually trained. Deterministically (momentum on a quadratic), the start→end interpolation cuts across oscillations and improves the convergence rate in the under-damped regime.

**Optimal α (used only as justification, not shipped).** For a quadratic with optimum x*, the loss-minimizing interpolation weight is α* = (θ₀−x*)ᵀA(θ₀−θ_k) / [(θ₀−θ_k)ᵀA(θ₀−θ_k)]; approximating x* by a Newton step with a diagonal empirical-Fisher curvature Â gives an adaptive, clipped estimate. A fixed α converges comparably, generalizes better, and avoids maintaining Â — so a fixed α is used.

**Code.** A PyTorch wrapper around `torch.optim.Optimizer`; the fast weights are the live parameters the inner optimizer mutates, the slow weights are cached per parameter.

```python
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        self.optimizer = optimizer
        self._la_step = 0
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        # swap in the slow weights for evaluation
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
        loss = self.optimizer.step(closure)          # k steps forward (one fast-weight step)
        self._la_step += 1
        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    # 1 step back: phi <- alpha*theta_k + (1-alpha)*phi
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
        return loss
```
