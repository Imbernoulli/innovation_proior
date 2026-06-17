## Research question

I want a single-sample Monte-Carlo estimator of the Kullback–Leibler divergence
`KL[q, p]` between two distributions `q` and `p` over the same support, in the
regime where I **cannot** sum or integrate over that support. The support may be
enormous (the vocabulary of a language model, every possible continuation of a
prompt) or the integral may have no closed form. What I *can* do, cheaply, is
evaluate the probabilities — or, more usefully, the log-probabilities — `log p(x)`
and `log q(x)` at any point `x`, and draw samples `x ~ q`.

This setting is the everyday one in reinforcement-learning fine-tuning of large
models, where a policy `q` is regularized to stay near a frozen reference `p`. The
KL between them is used as a penalty (or a diagnostic), and it is computed not from
the full categorical distributions but from the per-token log-probs that the
training loop already has on hand — the policy's `log q(x)` and the reference's
`log p(x)` for the sampled tokens. So the practical question is: given a stream of
samples `x ~ q` and the two scalars `log p(x)`, `log q(x)` at each, what function
of those scalars should I average to get a *good* estimate of `KL[q, p]`?

What makes one estimator "good" has three components, and they pull against each
other:

1. **Unbiasedness** — its expectation over `x ~ q` should equal `KL[q, p]` exactly,
   not merely approximately.
2. **Low variance** — with a finite batch of samples, the sample average should not
   swing wildly from batch to batch. A noisy KL estimate feeds a noisy penalty,
   which feeds a noisy gradient, which destabilizes training.
3. **The right sign behaviour** — the true KL is always `>= 0`. An estimator that is
   negative on many individual samples is awkward to interpret and, empirically,
   tends to be the high-variance ones.

The pain point is that the obvious estimator is unbiased but extremely noisy, and
the obvious fix for the noise introduces bias. Getting all three at once — unbiased,
low-variance, and well-behaved in sign — is the open problem.

## Background

**The divergence itself.** By definition,
```
KL[q, p] = sum_x q(x) log( q(x)/p(x) ) = E_{x ~ q}[ log( q(x)/p(x) ) ].
```
It is already an expectation under `q`, which is what makes a Monte-Carlo estimate
natural: draw `x ~ q`, evaluate the integrand, average. The quantity that recurs is
the likelihood ratio `r = p(x)/q(x)`; note `log(q/p) = -log r`. Two facts about `r`
under sampling from `q` are load-bearing:
- `E_{x ~ q}[ r ] = sum_x q(x) · p(x)/q(x) = sum_x p(x) = 1`. The ratio averages to
  exactly one because `p` is a normalized distribution.
- The true KL is `E_{x ~ q}[ -log r ]`, and Jensen's inequality on the convex
  `-log` gives `E[-log r] >= -log E[r] = -log 1 = 0`, recovering `KL >= 0`.

**Convexity facts that will matter for sign and magnitude.** The logarithm is
concave, so its graph lies below its tangent lines; a convex function lies above its
tangent lines. Tangent-gap quantities are therefore natural non-negative objects,
and they vanish to second order at the tangent point — a Taylor expansion about that
point makes the linear term cancel and leaves a leading quadratic term.

**f-divergences.** A broad family of "distances" between distributions has the form
`D_f(p, q) = E_{x ~ q}[ f(r) ]` for a convex `f` with `f(1) = 0`. KL is a member:
`KL[q, p]` is the case `f(x) = -log x`, and the *reverse* `KL[p, q]` is the case
`f(x) = x log x`. A standard local fact about this family (Amari; standard
information geometry) is that **every** differentiable f-divergence looks the same to
second order when `p` is close to `q`: for a parametric family `p_theta` near
`p_0 = q`,
```
D_f(p_0, p_theta) = (f''(1)/2) · theta^T F theta + O(theta^3),
```
with `F` the Fisher information. So two f-divergences with the same `f''(1)` are
indistinguishable to leading order near coincidence.

**Variance reduction by control variates.** The general tool for reducing the
variance of an unbiased Monte-Carlo estimator `X` without changing its mean: find a
quantity `Y` with *known* zero expectation and form `X + lambda·Y`. For any
constant `lambda` the mean is unchanged; if `Y` is negatively correlated with `X`,
a suitable `lambda > 0` cancels part of `X`'s fluctuation and lowers the variance.
The optimal `lambda` is `-Cov(X, Y)/Var(Y)`. Likelihood-ratio problems are a
natural place to look for such terms because normalized densities often imply
identities with known expectations, but choosing a zero-mean term and a coefficient
is part of the estimator design rather than part of the setup.

**Where this is used.** In RL fine-tuning of language models, a per-token KL between
the current policy and a frozen reference (typically the supervised-fine-tuned
initialization) keeps the policy from drifting too far while it chases reward. Two
wiring choices exist: subtract the per-token KL from the reward before advantage
estimation, or aggregate the per-token KL and add it to the policy-gradient loss with
a coefficient. Either way the per-token KL is built from the two log-probs the
forward passes already produce; the exact categorical KL over the whole vocabulary
is avoidable and usually avoided. Crucially, the policy's log-prob `log q(x)` carries
the trainable parameters, so whatever estimator goes into the loss is differentiated
through, and its *gradient* (not just its value) feeds back to the policy.

## Baselines

These are the estimators already in use for this exact problem; a new estimator is
measured against them.

**The naive log-ratio estimator (`k1`).** Read the definition literally and average
the integrand:
```
k1 = log( q(x)/p(x) ) = -log r,   x ~ q.
```
Core idea: `KL[q, p] = E_q[-log r]` is exactly this, so `E_q[k1] = KL[q, p]`
identically — it is unbiased, with no approximation. **Gap:** it has high variance.
Although `KL >= 0`, the per-sample value `-log r` is negative whenever `r > 1`
(whenever the reference assigns more mass to `x` than the policy), which happens on a
large fraction of samples. The estimator therefore spends much of its mass on the
wrong side of zero and cancels back to a small positive mean only after many samples;
on small batches it is extremely jumpy.

**The squared-log-ratio estimator (`k2`).** Average half the square of the log-ratio:
```
k2 = 0.5 · ( log( p(x)/q(x) ) )^2 = 0.5 (log r)^2,   x ~ q.
```
Core idea: each sample reports *how far apart* `p` and `q` are at `x`, and the square
makes every term non-negative, so there is no destructive cancellation and the
variance is much lower than `k1`. Its expectation `E_q[0.5 (log r)^2]` is the
f-divergence with `f(x) = 0.5 (log x)^2`, and since `f''(1) = 1` — the same `f''(1)`
as the KL's `f(x) = -log x` — it agrees with KL to second order, so near `p ~ q` its
bias is tiny. **Gap:** it is *biased*. `E_q[k2] != KL[q, p]` in general; the
agreement is only the leading Taylor term, and as `p` and `q` separate the higher-
order mismatch grows and `k2` systematically over-estimates the divergence. It buys
low variance at the cost of correctness whenever the two distributions are not close.

**The absolute-difference estimator (`abs`).** Average `|log p(x) - log q(x)| =
|log r|`. Always non-negative and robust to outliers, with moderate variance.
**Gap:** biased — its expectation is not `KL[q, p]` (it is an L1-type quantity, not
the KL integrand) — so it trades correctness for robustness in much the same way `k2`
trades it for variance.

The shared situation across these: the only *unbiased* one (`k1`) is the noisy one,
and the *low-variance* ones (`k2`, `abs`) are biased. None of them is simultaneously
unbiased, low-variance, and non-negative on every sample.

A second, subtler axis the baselines differ on — relevant because in RL the estimator
is differentiated through the policy's log-prob — is whether the estimator's
*gradient* with respect to the policy parameters matches the gradient of the true KL,
which is a distinct property from whether its *value* matches the true KL.

## Evaluation settings

The natural way to characterize a candidate estimator, using only facts knowable
before committing to one:

- **Synthetic Gaussians with known KL.** Take `q = N(0, 1)` and `p = N(mu, 1)` so the
  true `KL[q, p]` is available in closed form (`mu^2/2`). Sweep `mu` from a tiny
  offset (true KL ~ 0.005, the near-coincident regime) to a large one (true KL ~ 0.5
  and beyond, the well-separated regime). Draw a large sample (e.g. 10^7 points)
  from `q`, evaluate each candidate, and report **relative bias** `(E[estimate] −
  trueKL)/trueKL` and **relative standard deviation** `std[estimate]/trueKL`. This
  isolates the bias/variance tradeoff cleanly across the close-vs-far regimes.
- **The downstream training setting.** In LLM RL fine-tuning, a policy initialized
  from an SFT model is updated with a PPO-style policy gradient on
  math-reasoning prompts, with a per-token KL-to-reference term entering the loss at a
  small coefficient. The estimator under test supplies that per-token KL from the
  policy and reference log-probs; the yardstick is downstream task accuracy and
  training stability. Datasets and metrics (math word problems and competition-math
  benchmarks, accuracy of the final answer) pre-exist the choice of estimator.

## Code framework

The estimator plugs into a fixed training harness. Everything around it already
exists: the forward passes that produce the current-policy and reference log-probs,
the response mask, the aggregation that turns a per-token tensor into a scalar loss
term, the coefficient that scales it, and the optimizer. What is *not* settled is the
function that maps the two per-token log-prob tensors to a per-token divergence — that
mapping is exactly what is to be designed. So the substrate is just the generic
plumbing, with one empty slot.

```python
import torch


def estimate_kl(
    logprob: torch.Tensor,      # (batch, seq_len) log q(x): current policy, carries grad
    ref_logprob: torch.Tensor,  # (batch, seq_len) log p(x): frozen reference, detached
) -> torch.Tensor:              # (batch, seq_len) per-token divergence estimate
    """Map the two per-token log-probabilities to a per-token estimate of the
    KL divergence between the current policy and the reference. Gradients must
    flow back through `logprob` to the policy parameters (do not detach / no
    torch.no_grad), and the output must have the same shape as the inputs."""
    # TODO: the per-token estimator we will design.
    #       Given only log q(x) and log p(x) at the sampled token x ~ q,
    #       return the per-token quantity to average.
    pass


# existing training loop the estimator plugs into
def add_kl_loss(pg_loss, logprob, ref_logprob, response_mask, kl_coef):
    per_token_kl = estimate_kl(logprob, ref_logprob)   # the slot above
    per_token_kl = per_token_kl * response_mask        # mask out padding
    kl_term = agg_loss(per_token_kl, response_mask)     # existing aggregation to a scalar
    return pg_loss + kl_coef * kl_term                  # add to the policy-gradient loss
```

The harness hands the slot two log-prob tensors and consumes a same-shaped tensor of
per-token divergences; the body of `estimate_kl` is the entire contribution.
