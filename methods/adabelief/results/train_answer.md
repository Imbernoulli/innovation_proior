The central frustration with first-order optimizers is that Adam and SGD occupy opposite corners of a trade-off. Adam adapts per-coordinate steps through an exponential moving average of squared gradients, which makes it fast early and stable in volatile landscapes such as GAN training, but on standard supervised tasks it often generalizes worse than well-tuned SGD with momentum. SGD with momentum generalizes better at the end because it simply follows the gradient direction, yet it lacks per-coordinate adaptation and can be slow in flat regions or unstable in steep, narrow valleys. The goal is a single optimizer that converges as fast as Adam, generalizes as well as SGD, and remains stable, ideally as a small modification to Adam rather than a wholesale redesign.

The weakness is not Adam's moving-average machinery but its denominator. Adam divides the update by the square root of an EMA of raw squared gradients, v_t = EMA(g_t^2). This tracks gradient magnitude, not curvature or reliability. Magnitude happens to correlate with curvature in two classic cases: in flat regions both are small and Adam takes a large step, while in steep oscillating valleys both are large and Adam takes a small step. But in a long steady slope where the gradient is large and barely changing, magnitude is large while curvature is small. Adam shrinks the step there, even though the direction is reliable, whereas SGD's step is proportional to the gradient and moves boldly. The raw squared gradient also discards sign information: on f(x,y) = |x| + |y|, Adam sees v_x and v_y both near one and takes equal steps in the steady x direction and the oscillating y direction, which is exactly wrong.

The method I propose is AdaBelief. It keeps Adam's first-moment buffer m_t = beta_1 m_{t-1} + (1 - beta_1) g_t and replaces the denominator statistic with an exponential moving average of the squared prediction residual, s_t = beta_2 s_{t-1} + (1 - beta_2) (g_t - m_t)^2. Here m_t is the running prediction of the gradient, so g_t - m_t measures how much the observed gradient surprises that prediction. When the gradient is steady and predictable the residual is small, s_t is small, and the optimizer takes a large step; when the gradient is noisy or oscillating the residual is large, s_t is large, and the step is cautious. After bias correction, m_t approximates E[g_t], so s_t approximates Var(g_t), the centered second moment rather than the uncentered second moment that Adam uses. This distinction directly fixes the long-slope case: a large mean gradient with small variance no longer inflates the denominator, so the optimizer can move confidently along a reliable direction.

AdaBelief therefore recovers the right behavior in all three motivating cases. In flat regions the gradient is small and stable, the residual is small, and the step is large. In steep oscillating valleys the gradient changes quickly, the residual is large, and the step is small. On long steady slopes the gradient is large but predictable, the residual is small, and the step is again large. On f(x,y) = |x| + |y|, m_x approaches one while m_y approaches zero, so s_x approaches zero and s_y approaches one, producing a large step in the consistent direction and a small step in the oscillating one. The core update introduces no new hyperparameters beyond Adam's learning rate, beta_1, beta_2, and epsilon. Engineering branches inherited from the Adam family, such as AMSGrad-style running maxima, RAdam-style rectification for early training, and decoupled weight decay, remain orthogonal switches rather than part of the denominator design.

The following implementation stores the centered second-moment statistic with epsilon baked in for numerical stability, supports AMSGrad and rectified updates, and follows PyTorch's Optimizer interface:

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
