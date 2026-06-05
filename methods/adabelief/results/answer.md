# AdaBelief

**Problem.** Adam converges quickly but can generalize worse than SGD; SGD often generalizes well but is slow and fragile in volatile settings. AdaBelief keeps Adam's first-moment machinery and changes the denominator so the step size depends on how reliable the observed gradient is, not just how large it is.

**Key idea.** Adam divides by an EMA of the raw squared gradient, v_t = EMA(g_t^2). AdaBelief divides by an EMA of the squared prediction error, s_t = EMA((g_t - m_t)^2), where m_t is the EMA of the gradient. If g_t agrees with its prediction m_t, the residual is small, the denominator is small, and the step is large. If g_t surprises the prediction, the denominator is large and the step is cautious. Since m_t approximates E[g_t], s_t approximates Var(g_t), a centered second moment rather than an uncentered second moment.

**Algorithm.** The core update introduces no new mathematical hyperparameters beyond Adam's alpha, beta_1, beta_2, and epsilon. The practical implementation may expose existing Adam-family switches such as AMSGrad, RAdam-style rectification, and decoupled weight decay.

```text
m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
s_t = beta_2 s_{t-1} + (1 - beta_2) (g_t - m_t)^2 + epsilon
mhat_t = m_t / (1 - beta_1^t)
shat_t = s_t / (1 - beta_2^t)
theta_t = theta_{t-1} - alpha * mhat_t / (sqrt(shat_t) + epsilon)
```

The centered denominator matches the desired curvature behavior in the motivating cases: large steps in flat regions, small steps in steep oscillating valleys, and large steps on long steady slopes where the gradient is large but barely changing. On f(x,y) = |x| + |y|, Adam sees v_x = v_y = 1 and steps equally in the steady and oscillating coordinates; AdaBelief gets s_x near 0 and s_y near 1, so it advances in the consistent direction and damps the oscillation.

**Code.**

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True,
                 fixed_decay=False, rectify=True, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param_group in params:
                if 'betas' in param_group and param_group['betas'] != betas:
                    param_group['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify
        self.degenerated_to_sgd = degenerated_to_sgd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                elif group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4) / (N_sma_max - 4)
                                * (N_sma - 2) / N_sma
                                * N_sma_max / (N_sma_max - 2)
                            ) / bias_correction1
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / bias_correction1
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

        return loss
```
