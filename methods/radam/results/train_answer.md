The method I am presenting is Rectified Adam, abbreviated RAdam. It is an adaptive stochastic gradient optimizer that keeps the first- and second-moment exponential moving averages of Adam, but modifies the adaptive step size by a scalar rectifier that depends only on the current step count and the second-moment decay hyperparameter. The purpose of the rectifier is to suppress the high variance of the adaptive learning rate during the very beginning of training, the phase where second-moment estimates are based on almost no samples and where Adam often requires an ad-hoc learning-rate warmup to avoid divergence or bad local minima.

The starting observation is that Adam's adaptive rate is essentially the reciprocal square root of a variance-like estimate. At initialization the gradients are roughly zero-mean Gaussian in each coordinate, so the first squared gradient in a coordinate is a draw from a chi-square distribution with one degree of freedom. The reciprocal of its square root has an infinite mean and an infinite variance because a Gaussian has positive density arbitrarily close to zero. In practice this means that any coordinate whose first sampled gradient happens to be small receives an enormous adaptive rate, and a handful of such coordinates can push the parameters into a region from which the optimizer never recovers. This is exactly the distortion seen in gradient histograms during the first ten updates of Transformer training without warmup. Warmup suppresses the problem by shrinking the global learning rate during that critical window, but it offers no principled rule for choosing its length or shape.

RAdam replaces that hand-tuned warmup with a variance rectifier derived from the effective sample size of the second-moment moving average. The EMA with decay beta2 has a center of mass, and matching that center of mass to a simple moving average gives an effective length. In the limit this length is rho_infinity = 2 / (1 - beta2) - 1, and at finite step t it is rho_t = rho_infinity - 2 * t * beta2^t / (1 - beta2^t). For beta2 = 0.999 this starts near one and grows to about 1999. The adaptive rate is modeled as the square root of a scaled inverse chi-square variable with rho_t degrees of freedom. Its exact variance is finite for rho_t larger than two, but a stable first-order approximation for the variance uses the variance of the underlying scaled inverse chi-square, which is finite only for rho_t larger than four. The reference implementation is conservative and activates the adaptive branch only when the effective sample size is at least five.

When the branch is active the update is theta_t = theta_{t-1} - lr * r_t * mhat_t / (sqrt(vhat_t) + eps), where mhat_t and vhat_t are Adam's usual bias-corrected moments. The scalar rectifier r_t is chosen so that the variance of the effective adaptive rate matches the late-time variance reached at rho_infinity. Substituting the stable approximation and canceling the unknown gradient variance sigma^2 gives r_t = sqrt(((rho_t - 4) * (rho_t - 2) * rho_infinity) / ((rho_infinity - 4) * (rho_infinity - 2) * rho_t)). Because early variance is larger than late variance, r_t is below one at first and rises toward one, producing the same rising step-size shape as warmup but tied automatically to the optimizer's own second-moment averaging. When the branch is inactive the canonical implementation skips the adaptive update entirely, while an optional degenerated_to_sgd flag instead applies an unadapted momentum step.

The practical effect is that RAdam provides the fast early progress of adaptive methods without requiring a separate warmup schedule. On Transformer translation and language-model pretraining tasks this removes the need to search over warmup lengths. On image classification it behaves comparably to a well-tuned Adam with warmup. The method also clarifies why warmup works: both warmup and the rectifier reduce the variance of the adaptive rate while the denominator is still a poor statistic.

The code below implements a clean version of RAdam in PyTorch. It maintains the two EMA buffers, computes the effective sample size, caches the per-step rectifier in a small circular buffer to avoid redundant exponentiations, and applies either the full adaptive update or a momentum-only fallback depending on the branch condition. The implementation defaults to skipping steps when the effective sample size is below five, matching the public reference, but exposes degenerated_to_sgd for users who prefer a momentum fallback.

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, degenerated_to_sgd=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
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

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                state["step"] += 1
                step = state["step"]

                buffered = group["buffer"][int(step % 10)]
                if buffered[0] == step:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    beta2_t = beta2 ** step
                    n_sma_max = 2.0 / (1.0 - beta2) - 1.0
                    n_sma = n_sma_max - 2.0 * step * beta2_t / (1.0 - beta2_t)

                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1.0 - beta2_t)
                            * (n_sma - 4.0) / (n_sma_max - 4.0)
                            * (n_sma - 2.0) / n_sma
                            * n_sma_max / (n_sma_max - 2.0)
                        ) / (1.0 - beta1 ** step)
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1.0 - beta1 ** step)
                    else:
                        step_size = -1.0

                    buffered[0] = step
                    buffered[1] = n_sma
                    buffered[2] = step_size

                if n_sma >= 5:
                    if group["weight_decay"] != 0.0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)
                elif step_size > 0.0:
                    if group["weight_decay"] != 0.0:
                        p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])
                    p.data.copy_(p_data_fp32)

        return loss
```
