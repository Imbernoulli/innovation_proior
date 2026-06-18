# Rectified Adam (RAdam)

RAdam keeps Adam's first- and second-moment EMAs, but multiplies the adaptive step by a scalar rectifier derived from the effective sample size of the second-moment estimate. Let

```
m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2
mhat  = m_t / (1 - beta1^t)
rho_inf = 2 / (1 - beta2) - 1
rho_t   = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
```

For the active branch, use

```
r_t = sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
           ((rho_inf - 4) * (rho_inf - 2) * rho_t))

theta_t = theta_{t-1} - lr * r_t * mhat * sqrt(1 - beta2^t) / (sqrt(v_t) + eps)
```

The exact scaled-inverse-chi-square calculation gives

```
Var[sqrt(x)] =
tau^2 * (rho/(rho - 2)
         - rho * 2^(2*rho - 5) / pi * B((rho - 1)/2, (rho - 1)/2)^2)
```

with finite exact variance for `rho > 2`. The implementation threshold `rho_t >= 5` comes from the stable first-order approximation

```
Var[sqrt(x)] ~= rho / (2 * (rho - 2) * (rho - 4) * sigma^2)
```

which uses `Var[x]` and therefore needs `rho > 4`; the reference code is conservative and activates at `N_sma >= 5`.

```python
import math

import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            for group in params:
                if "betas" in group and tuple(group["betas"]) != tuple(betas):
                    group["buffer"] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state["step"] += 1
                step = state["step"]

                buffered = group["buffer"][int(step % 10)]
                if step == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    beta2_t = beta2 ** step
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * step * beta2_t / (1 - beta2_t)

                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4) / (n_sma_max - 4)
                            * (n_sma - 2) / n_sma
                            * n_sma_max / (n_sma_max - 2)
                        ) / (1 - beta1 ** step)
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** step)
                    else:
                        step_size = -1

                    buffered[0] = step
                    buffered[1] = n_sma
                    buffered[2] = step_size

                if n_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)

        return loss
```

The public reference implementation defaults `degenerated_to_sgd=False`, so `N_sma < 5` steps are skipped. Setting `degenerated_to_sgd=True` gives the unadapted momentum fallback branch.
