# Adam, distilled

Adam is a first-order optimizer that maintains per-parameter exponential moving averages of
the gradient and the squared gradient, bias-corrects both for their zero initialization, and
steps along their ratio. It unifies momentum (a smoothed gradient direction, the first
moment) with RMSProp/AdaGrad-style per-parameter scaling (the second moment in the
denominator), and adds the de-biasing step that neither predecessor had. It needs only
first-order gradients and two extra vectors of memory.

## Problem it solves

First-order optimization of a noisy stochastic objective `f(theta)` — noise from
minibatching and/or stochastic regularization such as dropout — over high-dimensional
parameters, where you want per-parameter adaptive step sizes, robustness to sparse and
non-stationary gradients, low memory, and little hyperparameter tuning, without any
second-order or curvature information.

## Key idea

Maintain exponential moving averages (EMAs) of the gradient (first moment `m`) and the
squared gradient (second raw moment `v`), bias-correct both, and step along the ratio
`m_hat / sqrt(v_hat)`:

- `m` is a momentum-style smoothed gradient — the *direction*, with minibatch noise averaged
  out and ravine oscillation damped.
- `sqrt(v)` is an RMSProp-style per-parameter scale — the *magnitude / uncertainty*; a
  windowed (not cumulative) second-moment estimate, so the effective learning rate does not
  decay to zero the way AdaGrad's growing sum forces, and it adapts to non-stationary
  objectives. It is also a cheap diagonal approximation to the square root of the Fisher
  information — a more conservative cousin of natural-gradient preconditioning.
- The ratio is **scale-invariant** while `eps` is negligible: rescaling gradients by `c` scales
  `m` by `c` and `v` by `c^2`, which cancel. So the step magnitude obeys
  `|Delta_t| <~ alpha`, making `alpha` a
  **trust region** in parameter space (a cap on per-step parameter movement, decoupled from
  the loss scale). The ratio also behaves like a signal-to-noise ratio, giving **automatic
  annealing** near an optimum: the true gradient shrinks while its variance does not, so the
  SNR and hence the step shrink on their own.

The piece neither momentum nor RMSProp had: **bias correction**. With `v_0 = 0`, unrolling
the EMA gives, in expectation, `E[v_t] = E[g_t^2]·(1 - beta_2^t) + small`, so dividing by
`(1 - beta_2^t)` de-biases it (and `(1 - beta_1^t)` for `m`). This is what makes `beta_2`
near 1 — needed for reliable second-moment estimates on sparse gradients, which require a
long averaging window — usable instead of divergent. The correction must hit *both* moments:
de-biasing only `v` would make the `t=1` ratio `0.1·sign(g)`, ten times too small.

## Defaults and why

`alpha = 0.001`, `beta_1 = 0.9`, `beta_2 = 0.999`, `eps = 1e-8`.

- The decay `beta` sets an EMA memory window of about `1/(1-beta)` samples. `beta_1 = 0.9`
  (~10 samples) keeps the direction responsive; `beta_2 = 0.999` (~1000 samples) keeps the
  denominator smooth, because a variance estimate is noisier than a mean estimate and a
  jittery denominator jitters every step — estimate the thing you divide by more carefully.
- `alpha = 0.001` is a conservative trust-region cap; SNR annealing handles the fine end.
- `eps = 1e-8` floors the denominator against divide-by-zero on dead coordinates, sized well
  below any healthy gradient RMS so it does not perturb the scale-invariant regime.

## Final algorithm

```
m_0, v_0, t  <-  0, 0, 0
while not converged:
    t   <- t + 1
    g_t <- grad_theta f_t(theta_{t-1})
    m_t <- beta_1 * m_{t-1} + (1 - beta_1) * g_t            # 1st moment EMA
    v_t <- beta_2 * v_{t-1} + (1 - beta_2) * g_t^2          # 2nd raw moment EMA (elementwise)
    m_hat <- m_t / (1 - beta_1^t)                           # bias correction
    v_hat <- v_t / (1 - beta_2^t)
    theta_t <- theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + eps)
```

Reference-code efficient form (fold both corrections into one scalar):
`alpha_t = alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t)`, then
`theta_t = theta_{t-1} - alpha_t * m_t / (sqrt(v_t) + eps)`.

This is algebraically exact for the bias corrections when `eps = 0`. With the algorithmic
denominator `sqrt(v_hat) + eps`, exact folding would use `sqrt(v_t) + eps * sqrt(1 - beta_2^t)`;
the reference implementation keeps `eps` after `sqrt(v_t)` as a fixed numerical floor.

## Theory (online convex optimization)

Regret `R(T) = sum_t [f_t(theta_t) - f_t(theta*)]`. Starting from the convexity hyperplane
bound `f_t(theta_t) - f_t(theta*) <= g_t^T(theta_t - theta*)`, with `alpha_t = alpha/sqrt(t)`,
a *decaying* first-moment coefficient `beta_{1,t} = beta_1·lambda^{t-1}` (lambda just below
1), the sharper gradient-summation control used in the proof, and the monotone-preconditioner
condition `sqrt(v_hat_{t,i}) / alpha_t` nondecreasing in `t`, the clean bound is

```
R(T) <= D^2/(2 alpha (1-beta_1)) * sum_i sqrt(T * v_hat_{T,i})
      + alpha(1+beta_1) G_inf / ((1-beta_1) sqrt(1-beta_2) (1-gamma)^2) * sum_i ||g_{1:T,i}||_2
      + sum_i D_inf^2 G_inf sqrt(1-beta_2) / (2 alpha (1-beta_1) (1-lambda)^2),
```

with `gamma = beta_1^2 / sqrt(beta_2) < 1`. Hence `R(T) = O(sqrt(T))` and average regret
`R(T)/T = O(1/sqrt(T)) -> 0` under those proof conditions. If only the summation step is
replaced by the unconditional Cauchy bound, the middle term picks up a `sqrt(1 + log T)`
factor but average regret still vanishes. The third term is constant in `T` only because
`beta_1` decays geometrically (`lambda < 1`); a constant momentum coefficient would make it
`Theta(T^{3/2})` and break the bound. The clean bound shrinks further when gradients are
sparse, since it is written in per-coordinate gradient norms rather than `d·G_inf·sqrt(T)`.

## Relation to prior methods

- **RMSProp** = the same EMA-of-squared-gradient denominator idea, but without the zero-init
  bias correction. The momentum variant of RMSProp puts momentum on the already-rescaled
  gradient; here the first and second moments are estimated separately and then combined.
- **AdaGrad** = this method with `beta_1 = 0`, infinitesimal `(1 - beta_2)`, and `alpha`
  replaced by the annealed `alpha·t^{-1/2}` (then `v_hat_t -> t^{-1} sum g^2`). The
  correspondence only holds *with* bias correction.
- **AdaMax** (infinity-norm variant): generalize the `L2` denominator to `Lp`, let `p ->
  infinity`, and the power-EMA collapses to a decayed running max
  `u_t = max(beta_2 · u_{t-1}, |g_t|)`; step `theta_t = theta_{t-1} - (alpha/(1-beta_1^t)) ·
  m_t / u_t`. No bias correction is needed for `u` because a max does not average in the zero
  init. For `beta_1 < beta_2`, the exact all-sequence envelope is
  `|Delta_t|/alpha <= ((1-beta_1)/(1-beta_1^t)) * (1-(beta_1/beta_2)^t)/(1-beta_1/beta_2)`,
  which is alpha-scale and essentially one for `beta_2` near 1. The looser cap `|Delta_t| <=
  alpha` also holds but is slightly conservative; the exact envelope above is the tight bound.
  The implementation adds `eps` inside the max as the zero-denominator floor. Good default
  `alpha = 0.002`.

## Working code

Filling the `step()` slot of the first-order harness, with per-parameter buffers
`exp_avg = m`, `exp_avg_sq = v` and the folded bias-correction `step_size`. This follows the
clean PyTorch v0.3.1 reference update; it rejects framework sparse tensors even though the math
is designed to work well when feature gradients are sparse:

```python
import math
import torch


class Adam:
    """Adam optimizer. Maintains per-parameter EMAs of the gradient (exp_avg = m)
    and squared gradient (exp_avg_sq = v), with bias correction."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
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
                raise RuntimeError('Adam does not support sparse gradients')
            state = self.state[id(p)]

            # State init: m_0 = v_0 = 0, t = 0
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)       # m
                state['exp_avg_sq'] = torch.zeros_like(p)    # v

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            t = state['step']

            if self.weight_decay != 0:
                grad = grad.add(p, alpha=self.weight_decay)

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)         # sqrt(v_t) + eps

            bias_correction1 = 1 - beta1 ** t                # (1 - beta1^t)
            bias_correction2 = 1 - beta2 ** t                # (1 - beta2^t)
            # Folded bias correction; exact for eps=0, with eps kept as a fixed floor.
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            # theta_t = theta_{t-1} - alpha_t * m_t / (sqrt(v_t) + eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)
```

AdaMax (infinity-norm sibling) — replace the squared-gradient EMA with an elementwise decayed
max `exp_inf`, and bias-correct only the first moment:

```python
import torch


class Adamax:
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas
        eps, lr, wd = self.eps, self.lr, self.weight_decay
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[id(p)]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)    # m
                state['exp_inf'] = torch.zeros_like(p)    # u (decayed running max)
            exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
            state['step'] += 1
            t = state['step']

            if wd != 0:
                grad = grad.add(p, alpha=wd)

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # u_t = max(beta2 * u_{t-1}, |g_t| + eps)   -- no bias correction needed
            norm_buf = torch.cat(
                [exp_inf.mul_(beta2).unsqueeze(0),
                 grad.abs().add_(eps).unsqueeze_(0)], 0)
            torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

            bias_correction1 = 1 - beta1 ** t              # only the first moment is corrected
            clr = lr / bias_correction1
            # theta_t = theta_{t-1} - (alpha/(1-beta1^t)) * m_t / u_t
            p.addcdiv_(exp_avg, exp_inf, value=-clr)
```
