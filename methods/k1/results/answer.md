# k1 naive KL estimator

`k1` is the sampled log-ratio estimator for the forward KL from the current policy to the
reference:

```text
k1(x_t) = log p_theta(x_t) - log p_ref(x_t) = logprob - ref_logprob
```

It is the simplest per-token value that can be produced from aligned sampled log-probabilities
in the actor KL hook.

## Problem

The exact token-level KL,

```text
KL[p_theta || p_ref] = sum_x p_theta(x) log(p_theta(x) / p_ref(x)),
```

requires a sum over the whole vocabulary. In the rollout/update path, the available tensors are
usually the current-policy and reference log-probabilities for the sampled token at each
response position. The estimator must return a same-shaped per-token penalty that the caller can
mask, aggregate, scale, and add to the actor loss.

## Estimator

Because the response token is sampled from `p_theta`,

```text
KL[p_theta || p_ref]
  = E_{x ~ p_theta}[log p_theta(x) - log p_ref(x)].
```

The one-sample Monte-Carlo integrand is therefore:

```text
k1 = logprob - ref_logprob.
```

So the response-masked average of `k1` is an unbiased Monte-Carlo estimate of
`KL[p_theta || p_ref]`.

## Properties

- **Value:** exactly unbiased,
  `E_{x ~ p_theta}[k1] = KL[p_theta || p_ref]`.
- **Variance:** high relative variance in the small-KL regime; in a small Gaussian example with
  true KL `0.005`, `stdev(k1) / true KL = 20`.
- **Direct loss gradient:** not the true KL gradient. Backpropagating through the sampled
  tensor gives `grad_theta k1 = s_theta`, so `E[s_theta] = 0`.
- **True KL gradient:**

  ```text
  grad_theta KL[p_theta || p_ref]
    = E_{x ~ p_theta}[s_theta * (k1 + 1)]
    = E_{x ~ p_theta}[s_theta * k1],
  ```

  because `E[s_theta] = 0`.
- **Squared estimator gradient:** `grad_theta(0.5 * k1^2) = k1 * s_theta`, whose expectation is
  the true KL gradient. This is why straight-through variants use the squared estimator on the
  backward pass.

## Use

Use `k1` when the priority is the exact sampled KL value with the least machinery: one
subtraction, no `exp`, no clamping, and no straight-through split. Do not treat its direct
autodiff gradient as an unbiased gradient of the expected KL in a differentiable actor-loss
path; it is the naive value baseline, not the gradient-correct penalty.

## Working code

Faithful to the `verl.trainer.ppo.core_algos.kl_penalty_forward` dispatcher:

```python
import torch


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        raise NotImplementedError

    raise NotImplementedError
```
