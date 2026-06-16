# AdamW (Decoupled Weight Decay), distilled

AdamW is Adam with the weight decay **decoupled** from the gradient-based update. Standard
deep-learning libraries regularize Adam with L2 — they add `lambda' * theta` to the gradient
and run it through Adam's adaptive normalization. AdamW instead shrinks the weights directly,
as a separate step kept *outside* the `sqrt(v_hat)` normalization, so every weight decays by
the same factor `(1 - eta_t * lambda)` per step regardless of its gradient history. The Adam
adaptive step is otherwise untouched; setting `lambda = 0` recovers plain Adam exactly.

## Problem it solves

Tuned adaptive methods (Adam) generalize worse than tuned SGD-with-momentum on
regularization-sensitive tasks (image classification on CIFAR/ImageNet), even at equal
training loss, forcing a per-task choice between the two optimizers. AdamW closes most of that
gap with a one-line change that leaves Adam's per-parameter step intact and makes the
regularization hyperparameter easier to set.

## Key idea: L2 regularization ≠ weight decay for adaptive methods

- **Weight decay** (Hanson & Pratt 1988): `theta_{t+1} = (1 - lambda) theta_t - alpha grad f_t`,
  a separate multiplicative shrink.
- **L2 regularization**: `f^reg_t = f_t + (lambda'/2)||theta||^2`, i.e. add `lambda' theta` to
  the gradient.
- For **SGD** they are identical with `lambda' = lambda/alpha` (Proposition 1). The
  reparameterization ties `lambda'` to `alpha`, so their best settings are coupled (lie on a
  diagonal).
- For an **adaptive** optimizer with update `theta_{t+1} = theta_t - alpha M_t grad f_t`
  (`M_t` a diagonal preconditioner, `M_t != k I`), matching the shrink terms requires
  `lambda I = alpha lambda' M_t`, which holds for all `theta` only if `M_t = k I` — false for
  adaptive methods. So **no L2 coefficient reproduces weight decay** (Proposition 2).
- Mechanistic consequence: with L2, the regularizer `lambda' theta` is divided by the same
  per-coordinate `sqrt(v_hat)` as the loss gradient. Weights with large historic gradients sit
  on a large denominator, so they get decayed *least* — exactly the weights a regularizer most
  wants to shrink. The adaptive normalization defeats the regularizer.

## The fix

Keep `lambda * theta` out of the preconditioner. Decouple the decay (Adam update, decoupled
term boxed in the original):

```
theta_t = theta_{t-1} - eta_t ( alpha * m_hat_t / (sqrt(v_hat_t) + eps) + lambda * theta_{t-1} )
```

with `m_t, v_t` the usual Adam moment EMAs and `m_hat, v_hat` their bias corrections. `eta_t`
is a global schedule multiplier applied to **both** the adaptive step and the decay — the
decay must be re-scheduled with `eta_t`, since decoupling it from the gradient removes the
implicit `eta_t`-scaling L2 enjoyed. Equivalently, apply the decay multiplicatively *before*
the Adam step: `theta <- (1 - eta_t*lambda) theta`, then subtract the Adam step. This is
exactly the additive old-parameter form for the same already-computed gradient and moment
state; applying the multiplicative decay after the Adam step would add a small cross-term.

Defaults are inherited from Adam unchanged: `alpha = 0.001`, `beta_1 = 0.9`, `beta_2 = 0.999`,
`eps = 1e-8`; the new knob is `lambda`. The same decoupling applied to SGD with momentum gives
SGDW, which (correctly) leaves SGD's behavior unchanged up to `lambda' = lambda/alpha`.

## Why it generalizes better (two justifications)

**Scale-adjusted L2 (fixed-preconditioner case, Proposition 3).** For a fixed preconditioner
`M = diag(s)^{-1}` (`s_i > 0`), running the method with decoupled weight decay `lambda` and base
rate `alpha` executes exactly the same steps as running it without decay on the scale-adjusted
loss

```
f^sreg_t(theta) = f_t(theta) + (lambda'/2) || theta (elementwise*) sqrt(s) ||_2^2,
                                 with lambda' = lambda/alpha.

Equivalently, in terms of the per-step decay lambda:
f^sreg_t(theta) = f_t(theta) + (lambda/(2 alpha)) || theta (elementwise*) sqrt(s) ||_2^2.
```

Proof: the scale-adjusted penalty's gradient is `lambda'(theta ⊙ s)`; dividing by the
preconditioner `/s` cancels one factor of `s`; the step contributes `alpha lambda' theta`,
which matches the decoupled shrink `lambda theta` iff `lambda' = lambda/alpha`.
Interpretation: decoupled decay
penalizes `theta_i` in proportion to `sqrt(s_i)`, so coordinates with historically large
gradients are regularized *more* than under plain L2 — the opposite of the L2-in-Adam
pathology. (Caveat: practical adaptive methods change `M_t` every step, so this is intuition,
not an exact statement for real Adam.)

**Bayesian filtering.** Viewing adaptive optimization as Bayesian filtering, the preconditioner
is the posterior covariance (`mu_post = mu_prior + Sigma_post g`), and the state-transition
prior `P(theta_{t+1}|theta_t) = N((I - A) theta_t, Q)` with `A = lambda I` multiplies the mean
by `(1 - lambda)` per step — i.e. decoupled weight decay emerges directly, applied to the prior
and *independent* of per-parameter uncertainty. L2 (a Gaussian prior contributing a gradient
term) instead gets divided by the uncertainty and, like any prior, is overwhelmed by data, so
its effective decay vanishes over long runs while a positive decay stays empirically useful.

## Practical additions (AdamWR)

**Normalized weight decay.** The optimal `lambda` shrinks as the number of batch passes grows.
Reparameterize `lambda = lambda_norm * sqrt( b / (B*T) )` (`b` batch size, `B` training-set
size, `T` epochs), so `lambda_norm` is the decay for a single batch pass. The best `lambda_norm`
then transfers across budgets and datasets (CIFAR-10 → ImageNet32x32). The durable claim is
that *some* normalization helps; `sqrt` is the best simple choice found.

**Cosine annealing + warm restarts (from SGDR).** Schedule the global multiplier
`eta_t = eta_min^(i) + 0.5(eta_max^(i) - eta_min^(i))(1 + cos(pi T_cur / T_i))`; for the `[0,1]`
range, `eta_t = 0.5 + 0.5 cos(pi T_cur / T_i)`. A warm restart resets `T_cur = 0` (keeping the
weights) and sets `T_{i+1} = T_i * T_mult`. **AdamWR** = AdamW + this schedule + normalized
decay; it gives strong anytime performance. (Restarts had failed for L2-Adam precisely because
its regularization was the weak link.)

## Working code

Filling the `step()` slot of the adaptive-optimizer harness — plain Adam plus one decoupled
decay line, grounded in the canonical PyTorch `torch.optim.AdamW` (`param.mul_(1 - lr*wd)`
before the Adam step) and the Lua reference implementation (the old parameter is subtracted
separately after the `addcdiv` Adam step):

```python
import math
import torch


class AdamW:
    """Adam with decoupled weight decay. The adaptive step is plain Adam; the decay
    acts directly on the weights, outside the sqrt(v) normalization."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        self.params = list(params)
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.state = {id(p): {} for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @staticmethod
    def schedule_multiplier(t):
        return 1.0  # fixed; or cosine-with-restarts: 0.5 + 0.5*cos(pi*T_cur/T_i)

    @torch.no_grad()
    def step(self, t):
        beta1, beta2 = self.betas
        eta_t = self.schedule_multiplier(t)            # global step-size multiplier
        lr_t = eta_t * self.lr
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad                              # loss gradient only (no wd*theta added in)
            if grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            state = self.state[id(p)]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)     # m
                state["exp_avg_sq"] = torch.zeros_like(p)  # v
            state["step"] += 1

            # Decoupled weight decay: theta <- (1 - lr_t*weight_decay) * theta, kept OUT of
            # the adaptive normalization. PyTorch applies lr_t * weight_decay.
            if self.weight_decay != 0:
                p.mul_(1 - lr_t * self.weight_decay)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)            # m_t
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
            denom = exp_avg_sq.sqrt().add_(self.eps)                   # sqrt(v_t) + eps

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = lr_t * math.sqrt(bias_correction2) / bias_correction1
            p.addcdiv_(exp_avg, denom, value=-step_size)              # plain Adam step
```

In this PyTorch-style API, the effective per-step shrink is `lr_t * weight_decay`. In the
derivation notation, normalized weight decay sets that effective `lambda` to
`lambda_norm * (b / (B * T)) ** 0.5`, with `b` the batch size, `B` the number of training
points, and `T` the number of epochs in the current run.
