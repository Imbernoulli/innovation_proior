The problem is that ordinary SGD depends on a single hand-tuned global learning rate. Set it too high and optimization diverges; set it too low and progress is impractically slow. Worse, the right scale differs across dimensions and layers, so one global number cannot be optimal everywhere. AdaGrad addresses part of this by giving each dimension its own rate based on the running sum of squared gradients, which evens out progress across coordinates and anneals automatically. But its denominator accumulates every squared gradient from the start of training, so it grows without bound and eventually forces the effective rate toward zero, stalling training before convergence. It is also sensitive to the initial gradients and still relies on a global learning-rate constant.

To fix AdaGrad's decay, I replace the all-time sum with a moving, exponentially-decaying estimate of recent squared gradients. This keeps the denominator local and bounded, so learning does not grind to a halt. However, that still leaves a learning-rate constant in the numerator. A unit-consistency argument shows what should replace it. A parameter update should carry the same units as the parameter itself. SGD and momentum violate this because their updates are proportional to the gradient, whose units are inverse parameter units. AdaGrad's ratio of gradients is unitless, which is also wrong. Newton's method, by contrast, divides the gradient by curvature, yielding an update with the correct parameter units. Rearranging the one-dimensional Newton step shows that the missing scale in the numerator should be a quantity with units of the parameter update. Since the current update is unknown when forming the step, I approximate it by the root-mean-square of past updates.

The resulting method is Adadelta. It maintains two running, per-dimension exponential averages: one over squared gradients and one over squared parameter updates. At each step the update is the negative gradient scaled by the ratio of the RMS of past updates to the RMS of recent gradients. No global learning rate appears in the derived update. The ratio is always positive, so the step always points along the descent direction. The denominator plays the same per-dimension evening-out role as AdaGrad, but because it is windowed it does not collapse to zero. The numerator supplies the correct units and behaves like a curvature-dependent acceleration term in the spirit of momentum. Typical choices for the decay are rho around 0.9 or 0.95 and a small epsilon such as 1e-6 for conditioning.

The lag between numerator and denominator also provides robustness. A sudden large gradient immediately raises the denominator and shrinks the effective step before the numerator can react, which helps in noisy or distributed settings. The epsilon term bootstraps the first step, when no past updates exist, and keeps updates from vanishing when gradients become very small. As training proceeds and both gradients and updates shrink, epsilon dominates both RMS quantities and the effective rate drifts toward one, producing an implicit annealing without any hand-designed schedule.

```python
import torch
from torch.optim.optimizer import Optimizer


class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

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
                if grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['acc_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                p.add_(delta, alpha=-group['lr'])
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        return loss
```
