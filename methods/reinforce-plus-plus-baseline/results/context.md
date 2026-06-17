# Context: Advantage Estimation for Critic-Free LLM Policy Optimization

## Research question

We are fine-tuning a large language model with online reinforcement learning on a verifiable-reward
task (math reasoning). For each prompt we sample one or more responses, score each response with a
(usually sparse, end-of-sequence) reward, and then update the policy with a PPO/GRPO-style clipped
actor loss. The single object that decides *how much each response and each token pushes the policy*
is the **advantage estimate**. The pain point is concrete: the cleanest way to get accurate advantages
in PPO is a learned value function (critic), but the critic is a second network of comparable size to
the policy, doubling memory and compute and adding its own training instability. The critic-free
alternatives that replace it estimate the advantage from the rewards of a small group of responses to
the *same* prompt — and that small-group estimate is exactly where instability creeps in. We want an
advantage estimator that (a) needs no critic, (b) gives a *stable* per-token advantage even when a
prompt's few sampled rewards are nearly identical, and (c) does not push the policy to merely "beat
its own group" at the expense of generalization. The estimator must consume per-token rewards, a
response mask, and group identifiers, and emit per-token advantages and returns for the actor loss.

## Background

**The RLHF / RLVR loop.** A prompt `q` is drawn, the old policy `π_old` samples one or more responses
`o`, a reward model or verifier assigns a scalar reward (for verifiable tasks, a 0/1 or -1/1 correctness
signal at the final token), and the policy `π_θ` is updated to raise the probability of high-advantage
tokens. To keep `π_θ` from collapsing onto the reward model, a KL penalty against a frozen reference
`π_ref` is applied, either folded into the per-token reward or added as a separate loss.

**PPO and the critic.** The dominant algorithm, PPO (Schulman et al. 2017), maximizes a clipped
surrogate
```
L_PPO(θ) = E_{q, o~π_old}[ (1/|o|) Σ_t min( s_t(θ) A_t, clip(s_t(θ), 1-ε, 1+ε) A_t ) ],
            s_t(θ) = π_θ(o_t | q, o_<t) / π_old(o_t | q, o_<t).
```
The advantage `A_t` is computed with Generalized Advantage Estimation, `A_t = Σ_l (γλ)^l δ_{t+l}`,
`δ_t = r_t + γ V(o_{t+1}) - V(o_t)`, where `V` is a learned critic. In the LLM setting only the EOS
token usually carries a reward, which makes an accurate per-token critic hard to train, and the critic
is itself a large model — substantial memory and compute overhead.

**The bandit view of LLM RL.** Ahmadian et al. (2024) and the REINFORCE line (Williams 1992) argue
that modeling each token as a separate timestep with bootstrapped intermediate values is unnecessary
when the reward only lands at the end: model the *entire generation as one action*, with the prompt as
the initial state, and optimize the sequence-level objective `E_{y~π_θ}[ R(y,x) ∇_θ log π_θ(y|x) ]`
directly. With reward only at EOS, the natural discount is `γ = 1`. The variance of this estimator is
reduced — while keeping it unbiased — by subtracting a *baseline* `b` that is independent of the
sampled action.

**Critic-free baselines from group statistics.** Once the critic is gone, the baseline must come from
somewhere. The family of methods that emerged estimates it from multiple responses to the same prompt:

- **ReMax** (Li et al. 2023): baseline = the reward of a single greedy-decoded response,
  `A = r(o) - r(ô)`.
- **RLOO** (Ahmadian et al. 2024): sample `k` responses, baseline each one with the *mean of the
  others*, `A_i = r(o^(i)) - (1/(k-1)) Σ_{j≠i} r(o^(j))`. The leave-one-out baseline is independent
  of `o^(i)`, so the estimator stays unbiased.
- **GRPO** (Shao et al. 2024): sample `k` responses and standardize within the group,
  `A_i = (r(o^(i)) - mean(r)) / (std(r) + ε)`, broadcasting the scalar to every token. Its KL term is
  added to the *loss* (not the reward) through Schulman's sample estimator
  `k3 = π_ref/π_θ - log(π_ref/π_θ) - 1`.

**Advantage standardization as a PPO implementation knob.** Standardizing advantages — subtracting the mean and
dividing by the standard deviation over a minibatch (whitening) before they enter the policy loss — is
a long-standing PPO implementation detail. The large-scale empirical study of Andrychowicz et al.
(2020) catalogs per-minibatch advantage normalization among its >50 design choices and finds it a
common, broadly-used standardization step. The useful prior fact is that centering and scaling
advantages over a pool is already an established PPO implementation choice; what remains open is how
to choose that pool when the critic is gone.

**Diagnostic observations about local group statistics.** Three facts about the small-group estimators
matter.

*First, the local-std estimator is statistically biased.* Take the GRPO estimator and write the group
rewards as `r_i = θ + ε_i` with `ε_i ~ N(0, σ²)` i.i.d., `i = 1..N`. With `ε̄ = (1/N)Σ ε_j`,
`D = sqrt((1/N) Σ (ε_j - ε̄)²)`, and `A_i = (ε_i - ε̄)/D`, the conditional numerator is
`E[ε_i - ε̄ | ε_i] = (1 - 1/N) ε_i`, but the denominator `D` is itself a function of `ε_i`:
`E[D² | ε_i] = ((N-1)²/N²) σ² + ((N-1)/N²) ε_i²`, which grows with `ε_i²`. Because the centered reward
in the numerator and the standard deviation in the denominator are computed from the *same small
sample*, they are not independent. The Taylor expansion of `f(x)=x^{-1/2}` around
`μ=E[D²|ε_i]` is
`f(x)=μ^{-1/2}-(1/2)μ^{-3/2}(x-μ)+(3/8)μ^{-5/2}(x-μ)²+O((x-μ)³)`, so the leading denominator scale
already depends on `ε_i²`. A rigorous quick check is boundedness: for any realized centered residuals,
`|A_i| ≤ sqrt(N-1)` under this `1/N` variance convention, while the Gaussian `ε_i` is unbounded; hence
`E[A_i | ε_i]` cannot equal `ε_i` as a function for any finite `N ≥ 2`. The coupling weakens as `N`
grows, but prompt-level groups in this setting are deliberately small.

*Second, the local std is a brittle small-sample scale.* In practice the group size `k` is small (4 or
8). If all sampled responses to a prompt receive exactly the same reward, the centered numerator is
also zero, so the exact standardized residual is not literally unbounded. The problem is instead that a
few rewards determine the whole prompt's scale: a single discrete reward flip can make an order-one
standardized residual, exact ties give no ranking signal, and every token from that prompt inherits a
scale set by only those few samples.

*Third, optimizing relative to a prompt's own group rewards a response only for being "better than its
siblings," not for being globally good*, which encourages overfitting to whichever prompts produce
diverse within-group rewards.

**The KL estimators as a separate loss.** With `ℓ(y) = log π_θ(y) - log π_ref(y)` and
`δ(y) = π_ref(y)/π_θ(y) = exp(-ℓ)`, three sample forms are in use:
`k1 = ℓ`, `k2 = (1/2)ℓ²`, and `k3 = δ - 1 + ℓ`. As scalar Monte Carlo values under samples from
`π_θ`, `k1` and `k3` estimate the reverse KL `D_KL(π_θ || π_ref)`. As ordinary fixed-sample loss terms,
however, their autodiff gradients are different objects:
`∇k1 = ∇log π_θ`, `∇k2 = ℓ ∇log π_θ`, and `∇k3 = (1 - exp(-ℓ)) ∇log π_θ`. The reverse-KL gradient is
`E_{y~π_θ}[ ℓ(y) ∇_θ log π_θ(y) ]` (the `+1` score-function term integrates to zero). Thus a standalone
KL loss has to be judged by its gradient, not only by scalar unbiasedness; after the score-term part
vanishes in expectation, the `k3` gradient leaves the importance-weighted term
`-E_{y~π_θ}[(π_ref/π_θ)∇log π_θ]`, which is the forward-KL gradient and can explode when `π_θ(y)` is
tiny.

## Baselines

**PPO with a critic (Schulman et al. 2017).** Clipped surrogate with GAE advantages from a learned
value function `V`. Accurate when `V` is good, but the critic is a second model the size of the policy:
heavy memory/compute, and hard to fit accurately when reward is only at EOS. *Gap:* the critic is
expensive and, in the sparse-reward LLM regime, an unreliable source of per-token advantages.

**RLOO (Ahmadian et al. 2024).** Full-sequence REINFORCE with a leave-one-out baseline,
`A_i = r(o^(i)) - (1/(k-1)) Σ_{j≠i} r(o^(j))`. Critic-free and the baseline is genuinely independent of
the scored response. *Gap:* the baseline is built from only the `k-1` other responses to the *same
prompt*; with small `k` it is a high-variance estimate, and the advantages across different prompts are
never put on a common scale.

**GRPO (Shao et al. 2024).** Critic-free; standardizes within the group,
`A_i = (r(o^(i)) - mean(r)) / (std(r) + ε)`, broadcast to all tokens; KL added to the loss via the `k3`
estimator. *Gap:* the within-group standardization couples a centered numerator to a denominator
computed from the same few samples (biased for finite `k`), the local std is a noisy scale set by only
one prompt's rewards, and the relative objective rewards beating one's own group rather than global
quality.

**ReMax (Li et al. 2023).** Greedy-response baseline, `A = r(o) - r(ô)`. *Gap:* a single greedy sample
is a noisy baseline and requires an extra greedy decode per update.

## Evaluation settings

- **Policy:** a small instruction-capable model (e.g. Qwen2.5-0.5B), full-parameter training.
- **Training data:** a math-reasoning training set of competition-style problems (e.g. MATH levels
  3-5), several thousand problems, with verifiable correctness rewards.
- **RL protocol:** on-policy rollouts, multiple sampled responses per prompt (group sampling, e.g. 16),
  a fixed global batch size, a fixed number of update steps, a fixed reward manager and KL setting,
  one GPU per experiment.
- **Metrics:** math-reasoning accuracy (`mean@1`) on held-out benchmarks — grade-school math
  (GSM8K), competition-math subsets (MATH-500), and contest problems (AMC). Higher is better.

## Code framework

The training loop already exists: it samples responses, computes per-token rewards (with the
reference-KL term handled upstream), and calls a registered advantage estimator to turn rewards into
the advantages and returns the actor loss consumes. The estimator is registered by name and receives
the per-token rewards, the response mask, and per-sample group identifiers. The numerical primitives
that already exist are masked reductions over the valid response tokens.

```python
import torch
from collections import defaultdict
from typing import Optional


def masked_mean(values, mask, axis=None):
    """Mean of `values` over the elements selected by `mask` (already provided)."""
    s = (values * mask).sum(axis=axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Variance of `values` over masked elements, Bessel-corrected (already provided)."""
    mean = masked_mean(values, mask)
    centered = values - mean
    var = masked_mean(centered ** 2, mask)
    if unbiased:
        n = mask.sum()
        var = var * (n / (n - 1))
    return var


def masked_whiten(values, mask, shift_mean=True):
    """Standardize `values` to zero mean / unit variance over masked elements (already provided)."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def register_adv_est(name):
    """The training loop's advantage-estimator registry (already provided)."""
    def deco(fn):
        return fn
    return deco


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: torch.Tensor = None,         # (bs,) group id per sample
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: turn per-token rewards + group ids into per-token (advantages, returns)
    pass
```
