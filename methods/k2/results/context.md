# Context: estimating a KL penalty from sampled log-probs (RL fine-tuning of LLMs)

## Research question

When a policy is fine-tuned with reinforcement learning against a reward, it must be held
close to a trusted reference distribution so it does not drift into degenerate, high-reward
but off-distribution behavior. The standard tether is a Kullback–Leibler penalty: a term that
grows as the current policy `π_θ` moves away from a frozen reference `π_ref` (typically the
supervised-fine-tuned initialization). On a language model the distributions live over the
entire vocabulary at every position, so the exact per-token KL,
`KL(π_θ‖π_ref) = Σ_v π_θ(v) log(π_θ(v)/π_ref(v))`, is a sum over tens of thousands of tokens
at every step — expensive to form and to differentiate through, and awkward when the training
machinery only retains the log-probabilities of the tokens that were actually sampled, not the
full categorical distribution.

So the concrete problem is: from nothing but the per-token log-probabilities of the sampled
tokens under the current policy and under the reference — two scalars per token,
`logprob = log π_θ(x)` and `ref_logprob = log π_ref(x)` — produce a per-token penalty that
(1) is cheap, using only those two numbers; (2) behaves like a KL when aggregated, so the
coefficient `kl_loss_coef` has a stable, interpretable meaning across runs; (3) is numerically
safe (no overflow on rare tokens); and crucially (4) is a *loss term that will be
backpropagated* — gradients must flow back into `π_θ` — so the quantity that matters is not
only what the penalty *is* but what its *gradient* pushes the policy toward. A penalty that
reports a sensible KL value but whose gradient drives the policy in the wrong direction (or
nowhere) is useless here. Closing the gap between "good KL value" and "good KL gradient", at
the cost of two log-probs per token, is the problem.

## Background

The field state: large language models are aligned and improved with policy-gradient RL —
trust-region methods and their clipped successors (Schulman et al. 2015, 2017), RLHF
(Ziegler et al. 2019; Ouyang et al. 2022), and group-relative variants (Shao et al. 2024,
DeepSeekMath) that drop the value network and normalize advantages within a group of rollouts
per prompt. Across all of them a reference-KL penalty is ubiquitous: it implements the
"stay near a policy you trust" prior, sharing the spirit of trust regions and of pessimism in
offline RL. In an LLM RL framework the penalty can enter in two places — subtracted from the
reward before advantage estimation (KL-in-reward), or aggregated into a per-token loss added to
the policy-gradient loss with a coefficient (the KL-loss path). The KL-loss path is the one
where the penalty's gradient flows directly into the policy, so its gradient behavior is what
governs training.

The load-bearing pieces of theory underneath:

- **Monte-Carlo estimation under sample access.** We can evaluate `π_θ(x)` and `π_ref(x)` for
  any sampled `x` but cannot afford the full sum, so the penalty is a sample average over
  tokens drawn on-policy from `π_θ`. A good estimator is unbiased and low-variance. Define the
  per-token log-ratio `Δ = log π_θ(x) − log π_ref(x) = log(π_θ/π_ref)`; the quantity to
  estimate is `KL(π_θ‖π_ref) = E_{x∼π_θ}[Δ]`.

- **The score-function identity (Williams 1992).** `E_{x∼π_θ}[∇_θ log π_θ(x)] = ∫ ∇_θ π_θ =
  ∇_θ ∫ π_θ = ∇_θ 1 = 0`. The mean of the score is zero. This is the identity that makes
  policy gradients work — and, as it turns out, the identity that can make a KL estimator's
  gradient silently vanish.

- **f-divergence generators and the local Fisher geometry (Csiszár; Amari).** Quantities of
  the form `E_{x∼q}[f(p(x)/q(x))]` are controlled near `p=q` by the generator's second
  derivative at `1`. For a smooth generator with `f(1)=0`, the local expansion is
  `D_f(p_0, p_θ) = (f''(1)/2)·θ^T F θ + O(θ³)`, with `F` the Fisher information at `p_0`.
  KL in the `KL(q‖p)` direction uses `f(x)=−log x`, whose `f''(1)` is `1`. The practical
  lesson is that local KL calibration is a second-derivative question, not only an exact-value
  question.

- **Control variates.** The textbook way to cut variance without adding bias: take an
  estimator and add a term with zero expectation that is negatively correlated with it. The
  one quantity here guaranteed to have zero mean is `r − 1` with `r = π_ref/π_θ`, since
  `E_{x∼π_θ}[r] = ∫ π_θ (π_ref/π_θ) = ∫ π_ref = 1`.

The motivating empirical fact about the existing sample-access estimator is visible in a toy
where the true KL is computable — e.g. two equal-variance Gaussians with a swept mean offset,
drawing many samples from the distribution that defines the expectation and comparing the
sample average to the analytic KL. The naive log-ratio estimator is exactly unbiased but has
very high variance: on a small true KL its standard deviation can be an order of magnitude or
more above the true value because the per-sample log-ratio is negative for roughly half the
samples even though KL is nonnegative. That leaves a sharp design tension: value unbiasedness,
low variance, nonnegativity, numerical safety, and the gradient seen by autograd are different
requirements, not one requirement.

## Baselines

The prior estimators a new per-token penalty would be measured against, with the gap each
leaves open.

**Naive log-ratio penalty (`k1`).** Use the Monte-Carlo-unbiased quantity directly,
`k1 = log π_θ(x) − log π_ref(x) = Δ`, whose mean over `x∼π_θ` is exactly `KL(π_θ‖π_ref)`.
Core idea: the most honest single-sample estimate of the KL value. Limitation: as a *value*
it has high variance — it takes both signs while KL is one-signed, so its standard deviation
dwarfs its mean on small divergences. And as a *loss to differentiate*, its autograd gradient
is `∇_θ Δ = ∇_θ log π_θ` (the reference term is constant in `θ`); averaged over on-policy
samples this is `E[∇_θ log π_θ] = 0` by the score-function identity. The penalty reports a
correct KL value yet, in expectation, pushes the policy nowhere — the regularizing signal is
washed out at any reasonable batch size.

**Absolute-value penalty (`abs`).** Take `|Δ|`. Core idea: force nonnegativity and bound the
penalty's magnitude robustly. Limitation: the kink at `Δ=0` makes its gradient depend only on
the *sign* of the log-ratio, `sign(Δ)·∇_θ log π_θ`, discarding the magnitude — the size of the
correction no longer scales with how far the policy has drifted, so it is a coarse,
sign-only tether rather than a faithful KL gradient.

**Control-variate / Bregman penalty (`k3`, `low_var_kl`).** Add the zero-mean control variate
`r − 1` (with `r = π_ref/π_θ`) to the log-ratio. Choosing the multiplier `λ=1` — justified by
the concavity bound `log x ≤ x − 1`, which makes the result the always-nonnegative vertical
gap between `log` and its tangent (a Bregman divergence) — gives `k3 = (r − 1) − log r =
e^{−Δ} − 1 + Δ`. Core idea: keep `k1`'s unbiased KL *value* but slash its variance and force
nonnegativity. This is the value-best estimator: in diagnostics it keeps the unbiased mean
while greatly reducing the raw log-ratio's spread, and it is the default in several frameworks.
Limitations: it requires an exponential `e^{−Δ}`, which overflows when a sampled token is far
less probable under the policy than under the reference, so it needs defensive clamping; and —
the deeper one for the KL-loss path — its autograd gradient is
`∇_θ k3 = (1 − r)·∇_θ log π_θ`, whose expectation is *not* the true KL gradient in general and
is observed to drift substantially once the policy moves away from the reference. It is built
to get the value right, and the value is not what is backpropagated.

## Evaluation settings

Two natural yardsticks, both pre-existing.

- **Estimator diagnostic on a known KL.** Pick distributions where `KL` is analytic — e.g.
  `q = N(0,1)`, `p = N(μ,1)` for a swept `μ`, so `KL` ranges from tiny (`μ=0.1`) to moderate
  (`μ=1`). Draw a large sample (order `10^7`) `x∼q`, form the per-sample log-ratio
  `log r = log p(x) − log q(x)`, build each candidate estimator from it, and report
  `(mean − trueKL)/trueKL` (relative bias) and `std/trueKL` (relative spread).

- **Downstream RL fine-tuning.** A small instruction policy (Qwen2.5-0.5B, full-parameter) in
  a verl-style framework with a group-relative advantage estimator; the KL penalty enters on
  the KL-loss path (`use_kl_loss=True`, a small `kl_loss_coef≈10⁻³`); math-reasoning prompts.
  The penalty receives `(batch, response_length)` current and reference log-probs and returns a
  per-token `(batch, response_length)` tensor; the loop masks it by the response mask,
  aggregates, scales by the coefficient, and adds it to the policy-gradient loss. Metric:
  mean@1 accuracy on held-out math benchmarks (e.g. GSM8K, MATH-500, AMC23).

## Code framework

The penalty plugs into a fixed RL training loop. Everything around it already exists — the
rollout that samples sequences from the current policy, the forward passes that produce
per-token log-probs under the current and the frozen reference model, the response mask, the
aggregation into a scalar, the multiplication by `kl_loss_coef`, and the addition to the
policy-gradient loss. The one empty slot is the per-token map from the two log-prob tensors to
the penalty tensor. It must keep the gradient connected to `logprob` (no `no_grad`) and return
the same shape as its inputs.

```python
import torch


def compute_kl_penalty(
    logprob: torch.Tensor,      # (batch, response_length) — log π_θ(x), current policy (requires grad)
    ref_logprob: torch.Tensor,  # (batch, response_length) — log π_ref(x), frozen reference
) -> torch.Tensor:              # (batch, response_length) — per-token KL penalty
    # TODO: the per-token estimator we will design.
    #       Map the two log-probs to a per-token penalty whose aggregate behaves like the
    #       reference KL and whose gradient flows usefully back into the policy. Cheap
    #       (these two scalars only), numerically safe, gradient-connected.
    pass


# existing RL training step the estimator plugs into
def kl_loss_step(logprob, ref_logprob, response_mask, kl_loss_coef, pg_loss):
    kl = compute_kl_penalty(logprob, ref_logprob)   # per-token penalty (the slot above)
    kl = kl * response_mask                          # ignore padding / prompt positions
    kl = aggregate(kl, response_mask)                # existing reduction to a scalar
    return pg_loss + kl_loss_coef * kl               # added to the policy-gradient loss
```

The rollout, the two log-prob forward passes, the mask, `aggregate`, and the coefficient are
all in place; only the body of `compute_kl_penalty` is to be designed.
