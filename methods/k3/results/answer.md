# k3 — a low-variance unbiased KL-divergence estimator, distilled

`k3` is a single-sample Monte-Carlo estimator of `KL[q, p]` built from per-sample
log-probabilities. Starting from the unbiased-but-noisy naive estimator
`k1 = log(q/p)`, it adds the zero-mean control variate `r - 1` (with `r = p/q`,
`E_{x~q}[r] = 1`) at coefficient `1`, giving

```
k3 = (r - 1) - log r,     x ~ q,   r = p(x)/q(x).
```

It is **unbiased** (the control variate has zero mean, so it inherits `k1`'s exact
expectation), **non-negative on every sample** (it is the Bregman divergence of
`-log` between `r` and `1` — the gap between `log` and its tangent at `1`, and
`log r <= r - 1`), and **low variance** (`r - 1` is negatively correlated with
`-log r`, cancelling fluctuation; near `r = 1` it equals `k2 = 0.5(log r)^2` to
leading order, inheriting `k2`'s calmness while keeping `k1`'s exact mean).

## Problem it solves

Estimate `KL[q, p]` when you can evaluate `log p(x)`, `log q(x)` pointwise and sample
`x ~ q`, but cannot sum/integrate over the support (e.g. an LLM's vocabulary of
continuations). You want the estimate to be simultaneously unbiased, low-variance,
and well-behaved in sign — properties the existing estimators only achieve one or two
at a time.

## Key idea

`KL[q, p] = E_{x~q}[log(q/p)] = E_{x~q}[-log r]`, so the naive `k1 = -log r` is
exactly unbiased — but it is negative whenever `r > 1` (about half the samples near
coincidence), so it has high variance. The squared form `k2 = 0.5(log r)^2` is
non-negative and low-variance, but biased: its expectation is the f-divergence
with `f(x) = 0.5(log x)^2`, which matches KL only to second order (both KL and this
`f` have `f''(1) = 1`), so it over-estimates as `p`, `q` separate.

To keep `k1`'s exact unbiasedness while removing its variance, apply a **control
variate**: add `lambda(r-1)`, where `E_{x~q}[r-1] = 0` (since `E_q[r] = sum_x p(x)
= 1`), so the mean is unchanged for any `lambda`. The variance-optimal `lambda*` =
`-Cov(-log r, r-1)/Var(r-1)` depends on the unknown `p`, `q`, so instead pick
`lambda = 1`, which yields

```
-log r + (r - 1) = (r - 1) - log r = k3.
```

This choice is special: because `log` is concave, `log r <= r - 1` with equality at
`r = 1`, so `k3 >= 0` on **every** sample — the non-negativity that squaring chased,
obtained without sacrificing unbiasedness. Geometrically `k3` is the **Bregman
divergence** `B_{-log}(r, 1) = (-log r) - (-log 1) - (-1)(r - 1)`.

**Generalization.** For any convex `f` (f-divergence `D_f = E_q[f(r)]`), the always
non-negative unbiased estimator is `f(r) - f'(1)(r - 1)`. For `KL[q,p]`, `f = -log`,
`f'(1) = -1`, recovering `k3`. For the reverse `KL[p,q]`, `f = x log x`, `f'(1) = 1`,
giving `r log r - (r - 1)`.

## Local behaviour

Writing `s = log r`, the estimator is

```
k3 = exp(s) - s - 1 = s^2/2 + s^3/6 + ...
```

so near `p = q` it matches `k2 = 0.5(log r)^2` to leading order. That is the
variance story: locally it behaves like the calm squared-log estimator, while its
mean remains the exact KL because the added term has zero expectation.

## A subtlety for differentiable use

When `k3` is differentiated through a trainable `log q` (RL fine-tuning), its
**value** is unbiased but its fixed-sample **gradient** is not the score-function
gradient of KL. With score `s_theta = grad_theta log q_theta(x)`,

```
grad_theta k3 = (1 - r) s_theta
grad_theta KL[q_theta,p] = E_q[-(log r) s_theta]
grad_theta k2 = -(log r) s_theta
```

The `+1` in `grad E_q[log(q/p)] = E_q[(log(q/p)+1)s_theta]` drops out because
`E_q[s_theta] = 0`. Thus value-unbiasedness does not imply gradient-unbiasedness:
`(1-r)` and `-(log r)` agree only to first order near `r=1`.

## Final estimator and code

In RL fine-tuning the current policy is `q` (sampled, trainable) and the frozen
reference is `p`. With `logprob = log q(x)`, `ref_logprob = log p(x)`, the log-ratio
is `s = log r = log(p/q) = ref_logprob - logprob`, and

```
k3 = exp(s) - s - 1,     s = ref_logprob - logprob,
```

which is `>= 0` since `exp(s) - s - 1 >= 0` (exp above its tangent at 0). Two clamps
mirror the deployed bounded branch: clamp `s` to `[-20, 20]` before `exp` (so a rare
large log-ratio cannot overflow — `exp(20) ~ 4.85e8` is finite), and clamp the output
to `[-10, 10]` so one large positive outlier token cannot dominate the aggregated KL
loss. The lower output bound is not a non-negativity projection; the analytic
non-negativity is from `exp(s) - s - 1 >= 0`.

```python
import torch


def estimate_kl(
    logprob: torch.Tensor,      # (bs, seq_len) log q(x): current policy, carries grad
    ref_logprob: torch.Tensor,  # (bs, seq_len) log p(x): frozen reference, detached
) -> torch.Tensor:              # (bs, seq_len) per-token k3 estimate of KL[policy || ref]
    """k3 (low-variance, unbiased): exp(s) - s - 1, with s = log(p/q)."""
    kl = ref_logprob - logprob              # s = log r = log(p/q)
    kl = torch.clamp(kl, min=-20, max=20)   # bound exp argument (no overflow)
    ratio = torch.exp(kl)                   # r = exp(s)
    kld = (ratio - kl - 1).contiguous()     # k3 = r - s - 1 >= 0  (Bregman of -log)
    return torch.clamp(kld, min=-10, max=10)  # bound any single token's penalty
```

The per-token tensor is masked, aggregated to a scalar, scaled by a small
coefficient, and added to the policy-gradient loss; gradients flow back through
`logprob` to the policy (do not wrap in `torch.no_grad`).
