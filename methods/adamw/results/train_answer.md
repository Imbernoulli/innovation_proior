The problem starts with a convention that most practitioners treat as a harmless equivalence: weight decay and L2 regularization are implemented the same way, by adding a λ′θ term to the gradient. For plain SGD this is fine, because the updates are identical after rescaling. But adaptive gradient methods such as Adam do not take raw-gradient steps; they divide each coordinate by a per-coordinate estimate of its historical gradient magnitude. When the L2 penalty is folded into the gradient, it gets swept into the same preconditioner as the loss gradient, so the amount of shrinkage applied to each weight inherits the adaptive scaling. Parameters with large gradients or large magnitudes end up regularized far less than parameters with small gradients. This is a concrete mechanical reason why Adam with L2 often generalizes worse than well-tuned SGD with momentum, even when it reaches lower training loss. Existing fixes such as AMSGrad or sharper-minima theories do not address this specific entanglement, and simply sweeping the L2 coefficient helps Adam far less than it helps SGD because the regularizer is not doing uniform weight decay at all.

The right fix is to return to the literal definition of weight decay: multiply every weight by (1 − λ) each step, separately from the adaptive gradient step. This gives AdamW. The algorithm is otherwise ordinary Adam. We maintain the first and second moment estimates on the raw loss gradient, bias-correct them, and take the usual preconditioned step. The only difference is that the weight-decay shrink is applied directly to the parameter tensor before the moment computation, so it is never divided by √v̂ and never pollutes the moment buffers. This restores true uniform weight decay, makes the regularization strength λ independent of the learning rate α, and explains the empirical observation that Adam's best run with L2 is often no better than Adam with no regularization at all. In the fixed-preconditioner limit, the direct-decay update is equivalent to L2 regularization on a scale-adjusted penalty that weights coordinates by their typical gradient magnitude, which means it actually regularizes the brittle high-gradient directions more heavily instead of letting them off the hook. A normalized weight-decay rule, λ = λ_norm · √(b/(B·T)), makes the effective cumulative shrinkage budget-invariant across batch sizes, dataset sizes, and training epochs. When combined with cosine annealing and warm restarts, the result is AdamWR, which finally lets Adam benefit from the SGDR schedule because the underlying regularizer is sound.

```python
import math
import torch


class AdamW:
    """Adam with decoupled weight decay.

    The decay is applied as a direct (1 - lr * wd) multiply on the
    parameters, outside the Adam moment update. The gradient feeding the
    moments is the raw loss gradient (no wd * param added to it).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, schedule=None):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0,
                                  m=torch.zeros_like(p),
                                  v=torch.zeros_like(p))
                      for p in self.params}

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])

            # Decoupled weight decay: direct shrink, not routed through grad.
            p.mul_(1 - eta * self.lr * self.wd)

            # Standard Adam moment accumulation on the raw gradient.
            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)

            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            denom = (st["v"] / bc2).sqrt().add_(self.eps)
            step_size = eta * self.lr / bc1
            p.addcdiv_(st["m"], denom, value=-step_size)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


class SGDW:
    """SGD with momentum and decoupled weight decay."""

    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-4,
                 schedule=None):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.schedule = schedule or (lambda step: 1.0)
        self.state = {id(p): dict(step=0, m=torch.zeros_like(p))
                      for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            eta = self.schedule(st["step"])

            p.mul_(1 - eta * self.lr * self.wd)
            st["m"].mul_(self.momentum).add_(g, alpha=eta * self.lr)
            p.add_(st["m"], alpha=-1)

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


def normalized_weight_decay(lambda_norm, batch_size, dataset_size, epochs):
    """Per-step decay that keeps cumulative shrinkage budget-invariant.

    lambda = lambda_norm * sqrt(b / (B * T)).
    """
    return lambda_norm * math.sqrt(batch_size / (dataset_size * epochs))


def pytorch_weight_decay(lambda_norm, batch_size, dataset_size, epochs, lr):
    """PyTorch AdamW expects weight_decay = lambda / lr."""
    return normalized_weight_decay(lambda_norm, batch_size,
                                   dataset_size, epochs) / lr


def cosine_eta(T_cur, T_i, eta_min=0.0, eta_max=1.0):
    """SGDR/AdamWR schedule multiplier; restart by resetting T_cur to 0."""
    return (eta_min + 0.5 * (eta_max - eta_min) *
            (1 + math.cos(math.pi * T_cur / T_i)))
```
