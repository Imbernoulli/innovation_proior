# k2 (MSE) KL-divergence estimator, distilled

`k2` is a per-token Monte-Carlo estimator of the KL penalty used to tether an RL-fine-tuned
policy `π_θ` to a frozen reference `π_ref`. From the two sampled log-probs it returns
`½(log π_θ(x) − log π_ref(x))²`. As a *value* it is biased (its expectation is locally
KL-calibrated but is not the exact KL), but as a *loss* its gradient is an unbiased estimator
of the true KL gradient — which is the property that matters when the penalty is
backpropagated into the policy.

## Problem it solves

A reference-KL penalty in policy-gradient LLM fine-tuning. The exact per-token KL is a sum over
the whole vocabulary — too expensive and awkward when only the sampled tokens' log-probs are
retained. Estimate the penalty from `logprob = log π_θ(x)` and `ref_logprob = log π_ref(x)`
(two scalars per token), cheaply, with no overflow, and — on the KL-loss path — with a gradient
that actually pulls `π_θ` back toward `π_ref`.

## Key idea

Let `Δ = log π_θ(x) − log π_ref(x)`. The penalty is `k2 = ½Δ²`.

- **Value (biased, low variance, nonnegative).** `E_{x∼π_θ}[½Δ²]` has the same local generator
  calculation used for f-divergence expansions. Near coincidence,
  `D_f(p_0,p_θ) = (f''(1)/2)·θ^T F θ + O(θ³)`; KL uses `f(x)=−log x`. Both `f=−log x` and
  `f=½(log x)²` have `f''(1)=1` (`f=−log x`: `f''=1/x²`; `f=½(log x)²`:
  `f''=(1−log x)/x²`), so `k2` matches KL to second order when `π_θ≈π_ref` — small bias exactly
  where a penalty operates. Bias grows as the distributions separate. Squaring makes it
  one-signed and concentrates the variance (the naive log-ratio is two-signed with std ≫ mean
  on small KL).

- **Gradient (unbiased for the KL gradient — the decisive property).** With `π_ref` frozen,
  `∇_θ k2 = Δ·∇_θ log π_θ`. The true gradient is
  `∇_θ KL(π_θ‖π_ref) = ∫(∇_θπ_θ)log(π_θ/π_ref) + ∫π_θ∇_θlog(π_θ/π_ref)
   = E[Δ·∇_θ log π_θ] + E[∇_θ log π_θ] = E[Δ·∇_θ log π_θ]`
  (the second term vanishes by the score identity `E[∇_θ log π_θ]=0`). So
  `E[∇_θ k2] = ∇_θ KL` exactly.

- **Numerics.** No `exp`/`log` beyond the input log-probs, so nothing overflows; the square is
  a subtraction and a multiply. The factor `½` does double duty: it sets `f''(1)=1` (value
  calibration) and makes the gradient coefficient on `Δ` exactly `1` (no stray factor of two).

## How it compares to the other cheap estimators

With `r = π_ref/π_θ = e^{−Δ}`, all per-token, samples `x∼π_θ`:

| estimator | formula | autograd gradient `∇_θ` | value behavior | gradient behavior |
|---|---|---|---|---|
| `k1` (`kl`) | `Δ` | `∇log π_θ` | unbiased, high var | `E[·]=0` (vanishes) |
| `abs` | `|Δ|` | `sign(Δ)·∇log π_θ` | biased, robust | sign-only, drops magnitude |
| `k2` (`mse`) | `½Δ²` | `Δ·∇log π_θ` | **biased**, low var, `≥0` | **= ∇KL** |
| `k3` (`low_var_kl`) | `(r−1)−log r = e^{−Δ}−1+Δ` | `(1−r)·∇log π_θ` | unbiased, low var, `≥0` | biased; needs `exp` clamp |

`k1` and `k3` get the *value* right but the *gradient* wrong; `k2` gets the *gradient* right but
the *value* (somewhat) wrong. On the KL-loss path the gradient is what is backpropagated, so
`k2` is the one to use. `k3`'s control-variate value (`λ=1` via `log x ≤ x−1`, the Bregman gap)
is the best *logged* number; combining `k3`'s value with `k2`'s gradient via a straight-through
(`s − s.detach() + k3.detach()` with `s = ½Δ²`) gives both — this is the framework's `k3+`
variant. `k2` itself never needs the `+` because its own gradient is already correct.

## Final form

```python
import torch


def compute_kl_penalty(
    logprob: torch.Tensor,      # (batch, response_length) — log π_θ(x), current policy (requires grad)
    ref_logprob: torch.Tensor,  # (batch, response_length) — log π_ref(x), frozen reference
) -> torch.Tensor:              # (batch, response_length) — per-token KL penalty
    """k2 (MSE) estimator: ½ (logprob - ref_logprob)².

    Value: biased local generator that matches KL to 2nd order near π_θ≈π_ref (f''(1)=1),
           always ≥ 0, low variance.
    Gradient: ∇(½Δ²) = Δ·∇log π_θ, whose on-policy expectation IS ∇KL(π_θ‖π_ref).
    No exp/log → nothing to overflow. No torch.no_grad(): the gradient must reach logprob.
    """
    return 0.5 * (logprob - ref_logprob).square()
```

The surrounding RL loop masks this per-token tensor by the response mask, aggregates it to a
scalar, multiplies by `kl_loss_coef`, and adds it to the policy-gradient loss.

Straight-through hybrid (k3 value, k2 gradient) when a faithful logged KL value is also wanted:

```python
def compute_kl_penalty_k3plus(logprob, ref_logprob):
    log_ratio = logprob - ref_logprob                      # Δ
    backward_score = 0.5 * log_ratio.square()               # k2 gradient
    kl = torch.clamp(ref_logprob - logprob, min=-20, max=20)  # -Δ, clamped before exp
    forward_score = torch.clamp(kl.exp() - kl - 1, min=-10, max=10)  # k3 value
    return backward_score - backward_score.detach() + forward_score.detach()
```
