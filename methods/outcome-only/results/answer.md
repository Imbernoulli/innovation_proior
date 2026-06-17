# Outcome supervision RL with GRPO, distilled

Outcome supervision is the reward/credit-assignment design used by GRPO (Group Relative
Policy Optimization) when the only feedback is a verifiable correctness check of the final
answer. A single scalar reward is attached to each complete response at its last token; the
reward space is left **raw** (no shaping, no upstream normalization); and the GRPO advantage
estimator alone normalizes it — subtract the per-prompt group mean, divide by the per-prompt
group std, and broadcast that one number uniformly to every token of the response. There is
no value function: the group mean is a free Monte-Carlo baseline. The KL-to-reference
regularizer is kept out of the reward and added to the loss with the k3 estimator.

"Outcome only" is exactly this: the reward-side normalization is the identity, so the
per-response correctness scalar passes through untouched and the group-relative estimator
downstream does all of the normalization.

## Problem it solves

On-policy RL fine-tuning of an SFT language reasoner where the reward is a rule check of the
final answer: one scalar per complete response (correct/incorrect, optionally a format
bonus), realized at the last token, with no per-token ground truth. PPO would need a learned,
policy-sized value function `V_ψ` to turn that into per-token advantages via GAE — costly, and
ill-posed when the reward appears only at the end. The goal is critic-free, with comparable
per-prompt contributions and a non-entangled regularizer.

## Key idea

For each question `q`, sample a group `{o_1, ..., o_G}` from `π_θold`, score each with the
checker to get `{r_1, ..., r_G}`, and form the group-relative advantage

```
A_hat_{i,t} = r_tilde_i = ( r_i - mean(r) ) / ( std(r) + eps )    for all tokens t of o_i.
```

- **Raw reward, single last-token scalar.** The correctness bit is only defined for a
  complete response, so it is encoded as one scalar at the terminal token. Reward space is
  left untransformed; the per-response scalar is recovered by summing the token-level reward
  tensor along the token axis (all other positions are zero).
- **Group mean = free baseline-shaped signal (no critic).** `mean(r)` over `G`
  same-prompt samples is a Monte-Carlo estimate of the prompt's expected return under the
  rollout policy. A leave-one-out mean would be the exact action-independent baseline; the
  implemented full-group mean includes the scored response itself, so the estimator has a
  finite-`G` self-dependence (without std scaling, a `(G−1)/G` shrinkage of the reward-gradient
  term). This is the deliberate group-relative estimator that replaces PPO's value network.
- **Divide by group std = comparable footing.** The centered signal's magnitude would
  otherwise be set by each prompt's reward spread (large for frontier prompts, ~0 for
  near-unanimous ones), so high-spread prompts would dominate the batch. The per-prompt
  z-score equalizes their contributions. A unanimous group has zero numerator, so the
  `eps`-floored denominator makes it contribute nothing.
- **Normalize in exactly one place.** Leaving reward space raw and letting the group z-score
  be the *sole* normalizer avoids double-normalization: an affine upstream normalization is
  silently undone by the group z-score, and a non-affine one is compounded with it
  unpredictably. The reward-side transform is therefore the identity.
- **Uniform broadcast = critic-free credit assignment.** With a trajectory-level return and
  no per-token value function, the same trajectory-level advantage is attached identically to
  every token. Differentiating credit across tokens would require exactly the critic GRPO
  discards. (Finer credit needs a per-step / process reward — extra supervision deliberately
  avoided here.)

## KL regularizer in the loss, via k3

Folding a per-token KL into the reward (RLHF style) would entangle the regularizer with the
correctness signal and corrupt the group statistics. Instead the advantage stays a clean
normalized reward and the KL is a separate loss term. With `r = π_ref/π_θ`, the estimator
family `−log r + λ(r−1)` is unbiased for `KL[π_θ||π_ref]` for any `λ` (since `E_θ[r−1] = 0` is
a zero-mean control variate). The k3 choice `λ = 1`,

```
D_KL_hat = (π_ref/π_θ) − log(π_ref/π_θ) − 1 = (r − 1) − log r,
```

is **non-negative** (`r=1` is its global minimum; it is the Bregman divergence of `−log`,
equivalently the gap in `log r ≤ r − 1`) and cancels the leading small-step variation:
for `r = 1 + δ`, `(r−1)−log r = δ²/2 + O(δ³)`. The exact variance-minimizing `λ` can differ
away from `r≈1`, but `λ=1` is the local low-variance choice that also preserves global
non-negativity.

## Final objective

```
J_GRPO(θ) = E[ q, {o_i}~π_θold ]  (1/G) Σ_i (1/|o_i|) Σ_t {
    min( ρ_{i,t} A_hat_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) A_hat_{i,t} )
    − β [ (π_ref(o_{i,t}|·)/π_θ(o_{i,t}|·)) − log(π_ref(o_{i,t}|·)/π_θ(o_{i,t}|·)) − 1 ] }
```

with `ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_θold(o_{i,t}|q,o_{i,<t})`, clip from PPO
(`ε ≈ 0.2`), and `A_hat_{i,t} = r_tilde_i` broadcast over the response.

## Unified-paradigm view (why it is RL, not filtered SFT)

Every recipe is `∇J = E[ (1/|o|) Σ_t GC · ∇_θ log π_θ(o_t|q,o_{<t}) ]` for some gradient
coefficient `GC`. Assuming `π_θold = π_θ` (single update per exploration, so `ρ=1` and the
clip is inactive), this objective's coefficient is

```
GC = A_hat_{i,t} + β( π_ref/π_θ − 1 ) = r_tilde_i + β( π_ref/π_θ − 1 ).
```

Compare: SFT has `GC ≡ 1`; rejection-sampling fine-tuning (RFT / online RFT) has
`GC = I(o) ∈ {0,1}` — reinforce correct responses uniformly, never penalize wrong ones.
Outcome-supervised GRPO's `r_tilde_i` is a **signed, magnitude-graded** coefficient:
strongly positive above the group mean, strongly negative below it, near zero for average —
differential reinforcement *and* penalization, which the 0/1 gate cannot express.

## Working code

The reward-side normalization (the "outcome only" slot) is the identity; all normalization
lives in the group-relative advantage estimator.

```python
import torch
import numpy as np
from collections import defaultdict
from typing import Optional


@torch.no_grad()
def normalize_rewards(token_level_scores, response_mask, index=None,
                      epsilon=1e-6, config=None, **kwargs):
    """outcome_only: no reward-space normalization. The per-response correctness
    scalar stays raw at its last valid token; the group-relative advantage stage
    is the sole normalizer."""
    return token_level_scores * response_mask


@torch.no_grad()
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6,
                                   norm_adv_by_std_in_grpo: bool = True,
                                   config: Optional[object] = None):
    """GRPO outcome-supervision advantage: per-prompt group z-score of the
    trajectory reward, broadcast to every token. No value function."""
    scores = token_level_rewards.sum(dim=-1)                # per-response scalar (last token)

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])                # group siblings by prompt id
    for idx in id2score:
        if len(id2score[idx]) == 1:                         # canonical singleton fallback
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        else:
            g = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(g)                    # free MC baseline
            id2std[idx] = torch.std(g)                      # per-prompt spread
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]       # center-only variant
    scores = scores.unsqueeze(-1) * response_mask           # broadcast to all tokens
    return scores, scores                                   # advantages == returns


def kl_penalty_forward(logprob, ref_logprob, kl_penalty="k3"):
    """k3 / low_var_kl: r = pi_ref / pi_theta = exp(ref_logprob - logprob)."""
    if kl_penalty not in ("low_var_kl", "k3"):
        raise NotImplementedError(f"unsupported KL penalty: {kl_penalty}")
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()                     # (r - 1) - log r
    return torch.clamp(kld, min=-10, max=10)
```

## Relation to alternatives

- **PPO** keeps a learned value `V_ψ` for `A_t` via GAE; outcome-supervised GRPO replaces it
  with the group mean (free) and broadcasts a single response advantage — no critic.
- **RFT / online RFT** uses `GC = I(o) ∈ {0,1}` (filtered imitation); the group z-score gives
  a signed, graded coefficient instead.
- **Process supervision** scores each reasoning step and sets `A_hat_{i,t} = Σ_{step j ≥ t}
  r_tilde_i^{(j)}` for finer credit, at the cost of a per-step reward source; outcome
  supervision needs only a final-answer check and therefore broadcasts one scalar uniformly.
