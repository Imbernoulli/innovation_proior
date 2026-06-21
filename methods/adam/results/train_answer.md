We need a first-order optimizer for high-dimensional stochastic objectives where gradients are noisy from minibatching and dropout, parameters number in the millions, and anything quadratic in the parameter dimension is impossible. The existing options each cover only part of the problem. SGD with momentum smooths gradients and damps oscillation, but it uses a single global learning rate, which is structurally wrong when convolutional layers and dense layers produce gradients of very different magnitudes. AdaGrad gives every parameter its own scale by dividing by the accumulated sum of squared gradients, which is excellent for sparse features, but the sum never forgets, so the effective learning rate decays monotonically toward zero and eventually stalls on non-stationary objectives. RMSProp fixes this by replacing the cumulative sum with an exponential moving average of squared gradients, but its zero-initialized denominator is biased toward zero when the decay rate is pushed close to one, which is exactly the regime sparse gradients need, causing huge early steps and instability.

The method is Adam. It keeps two exponential moving averages per parameter: a first-moment estimate of the gradient and a second raw-moment estimate of the squared gradient. The first moment supplies a smoothed, momentum-style direction; the second moment supplies a per-parameter scale. Both are bias-corrected for their zero initialization, and the update steps along their ratio. Because the numerator scales linearly with the gradient and the denominator scales quadratically, the ratio is scale-invariant while numerical regularization is negligible, which means the global learning rate acts as a trust region in parameter space rather than a knob tied to the loss magnitude. The ratio also behaves like a signal-to-noise ratio, so steps automatically shrink near an optimum where the true gradient vanishes but minibatch variance remains.

The bias correction is the critical addition. Unrolling the second-moment EMA shows that, in expectation, it equals the true second moment multiplied by (1 - beta_2^t). Dividing by this factor removes the early-transient bias, and the same correction applies to the first moment with (1 - beta_1^t). Without correcting both, the initial ratio is wrong in either direction: an uncorrected denominator gives enormous first steps, while correcting only the denominator makes the first step ten times too small. With both corrected, the method safely uses a slow-decaying second moment, which is what makes it reliable on sparse gradients. The default hyperparameters follow directly from these roles: beta_1 = 0.9 averages over roughly ten gradients for a responsive direction, beta_2 = 0.999 averages over roughly a thousand gradients for a smooth denominator, alpha = 0.001 is a conservative cap on per-step parameter movement, and eps = 1e-8 is a small floor to prevent division by zero on dead coordinates.

The update rule is m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t, v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2, then m_hat_t = m_t / (1 - beta_1^t) and v_hat_t = v_t / (1 - beta_2^t), and finally theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + eps). In practice both bias corrections are folded into a single scalar step size, alpha_t = alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t), and the update is written with the raw buffers as theta_t = theta_{t-1} - alpha_t * m_t / (sqrt(v_t) + eps). AdaGrad and RMSProp emerge as limiting cases, confirming that Adam is the right unification rather than an unrelated fourth method.

```python
import math
import torch


class Adam:
    """Adam optimizer with bias-corrected first- and second-moment estimates."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameters: {betas}")
        self.params = list(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients")
            state = self.state[id(p)]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]

            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)
            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            p.addcdiv_(exp_avg, denom, value=-step_size)
```
