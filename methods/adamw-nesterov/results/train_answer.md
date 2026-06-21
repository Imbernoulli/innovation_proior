The default adaptive optimizer, AdamW, keeps two moving averages: a first moment that smooths the gradient and a second moment that rescales each coordinate. The second moment is the source of its robustness — one learning rate works across coordinates of wildly different gradient scale — but the first moment is only classical momentum, a decaying average of past gradients. Classical momentum commits to the previous velocity direction before seeing the current gradient, so the step can overshoot or miss a better correction. Nesterov's accelerated gradient fixes this by evaluating the gradient after taking the momentum step, yielding a better direction and a provably faster rate on smooth convex problems. The obstacle is that the look-ahead point sits off the current parameters, which is awkward in a standard training loop that differentiates only at the current point. The right upgrade is to keep AdamW's adaptive rescaling but replace its classical momentum with Nesterov-accelerated momentum, folding the look-ahead into the update rather than the gradient-evaluation point.

The method is NAdam, short for Nesterov-accelerated adaptive moment estimation. It keeps the adaptive denominator exactly as Adam does and only changes how the first moment is used. The stored buffer remains the ordinary exponential moving average m_t = beta1 * m_{t-1} + (1 - beta1) * g_t. The parameter step is then written as a bias-corrected blend of a fresh-gradient piece and an advanced-momentum piece. The fresh-gradient piece is weighted by (1 - mu_t) and normalized by (1 - mu_product), where mu_product is the running product of scheduled momentum coefficients. The advanced-momentum piece is weighted by mu_{t+1} and normalized by (1 - mu_product * mu_{t+1}), because it has been advanced by one step and therefore looks one timestep further into the future. The two pieces share the same adaptive denominator sqrt(n_t / (1 - beta2^t)) + eps, so the per-coordinate rescaling is unchanged. A scheduled mu_t starts near beta1/2 and rises smoothly toward beta1; this prevents the optimizer from leaning too heavily on the accelerated term while the moment estimates are still noisy early in training. Decoupled weight decay shrinks the parameters directly before the adaptive step, which restores uniform L2 shrinkage and avoids the coordinate-dependent distortion that occurs when weight decay is folded into the gradient and then divided by the adaptive denominator.

NAdam is therefore the smallest principled change to AdamW: the same second-moment geometry, the same weight-decay semantics, but a first-moment step that reacts to the current gradient twice — once directly and once through the advanced momentum. On language-model pretraining this is a direct replacement for AdamW; in the ladder it serves as the diagonal-momentum refinement before moving on to geometry changes such as sign steps or non-diagonal preconditioners.

```python
import torch
from torch.optim.optimizer import Optimizer


class NAdam(Optimizer):
    """Adam with Nesterov-accelerated momentum, with optional AdamW-style
    decoupled weight decay. The second-moment adaptive rescaling is unchanged."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=4e-3, decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        momentum_decay=momentum_decay,
                        decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            psi = group["momentum_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["mu_product"] = torch.tensor(1.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"].add_(1)
                t = int(state["step"].item())
                m, n = state["exp_avg"], state["exp_avg_sq"]

                mu_t = beta1 * (1.0 - 0.5 * (0.96 ** (t * psi)))
                mu_t1 = beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * psi)))
                state["mu_product"].mul_(mu_t)
                mu_product = float(state["mu_product"].item())

                if wd != 0:
                    if group["decoupled_weight_decay"]:
                        p.mul_(1 - lr * wd)
                    else:
                        g = g.add(p, alpha=wd)

                m.lerp_(g, 1 - beta1)
                n.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (n / (1 - beta2 ** t)).sqrt_().add_(eps)

                p.addcdiv_(g, denom, value=-lr * (1 - mu_t) / (1 - mu_product))
                p.addcdiv_(m, denom, value=-lr * mu_t1 / (1 - mu_product * mu_t1))
        return loss
```
