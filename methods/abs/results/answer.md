# The `abs` KL estimator

`abs` is the per-token absolute log-ratio used as an actor-side KL-loss estimator:

```
l = log π_θ(x) − log π_ref(x)
r = exp(l) = π_θ(x) / π_ref(x)
s = exp(−l) = π_ref(x) / π_θ(x)
abs(l) = |l|
```

It is intended for the setting where tokens are sampled from `π_θ`, the trainer only has
`log π_θ(x)` and `log π_ref(x)` for those sampled tokens, and the returned tensor is added to a
differentiated actor loss.

## Estimator comparison

| estimator | per-token value | sampled-loss gradient | value behavior |
| --- | --- | --- | --- |
| `k1` | `l` | `∇ log π_θ`, whose expectation is `0` | unbiased, signed, high variance |
| `k2` | `½ l²` | `l ∇ log π_θ`, whose expectation is `∇ KL(π_θ ‖ π_ref)` | biased, non-negative, locally matches KL because `f''(1)=1`, tail-sensitive |
| `k3` | `(s−1) − log s` | `(1−s) ∇ log π_θ`, whose expectation is `∇ KL(π_ref ‖ π_θ)` | unbiased, non-negative, low variance, needs `exp(−l)` |
| `abs` | `|l|` | `sign(l) ∇ log π_θ` | non-negative, biased upward, robust to large `|l|` |

## Key properties

- **Upward biased value.** Since `E[l] = KL(π_θ ‖ π_ref) ≥ 0`, Jensen's inequality gives
  `E[|l|] ≥ |E[l]| = KL(π_θ ‖ π_ref)`.
- **Not an `f`-divergence.** Written as a function of `s`, `|l| = |log s|`. This is convex on
  `(0,1]` because it equals `−log s`, but concave on `[1,∞)` because it equals `log s`; it is not
  globally convex.
- **Bounded actor-gradient weight.** Differentiating the sampled value gives
  `∇_θ |l| = sign(l) ∇_θ log π_θ(x)`, so the scalar multiplier on the sampled-token score has
  magnitude at most one.
- **Autograd implementation.** The implementation is just subtraction plus `.abs()`. It is not
  wrapped in `torch.no_grad()`, because the actor-side KL loss relies on the subgradient.

## Working code

```python
import torch


def compute_kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
) -> torch.Tensor:
    return (logprob - ref_logprob).abs()
```

The trainer applies the response mask, aggregates valid response tokens, scales by the KL
coefficient, and adds the result to the policy-gradient loss.
