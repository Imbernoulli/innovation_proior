# Group-relative (group-std) advantage, distilled

The group-relative advantage — the core of **GRPO (Group Relative Policy Optimization)** — is a
critic-free way to produce the per-token advantage for on-policy RL fine-tuning of language
models. For each prompt, sample a group of responses, score each with a reward model (or
correctness rule), and turn each response's scalar reward into an advantage by **subtracting the
group mean and dividing by the group standard deviation** — a within-prompt z-score — then
broadcast that scalar across the response's tokens. The group mean is a same-prompt sample
estimate of the value baseline, so no learned value/critic network is needed; the group std
makes every prompt contribute on a comparable scale regardless of the reward model's arbitrary
per-prompt range. The reward-stage `group_std` variant applies the same per-prompt centering and
rescaling idea before advantage estimation.

## Problem it solves

On-policy RL fine-tuning of an SFT LLM against a reward model that emits one scalar per response
(at the last token). PPO needs a separate, policy-sized value network `V_psi` to make per-token
advantages via GAE; that critic is a large memory/compute cost and is the hardest part to fit
when the reward arrives only at the end of the sequence. The group-relative advantage keeps
PPO's stable clipped update and its variance-reducing baseline while removing the critic.

## Key idea

Any baseline `b(s)` that depends only on the state leaves the policy gradient unbiased:
`E_{a~pi}[grad log pi(a|s) b(s)] = b(s) sum_a grad pi(a|s) = b(s) grad 1 = 0`. That proves why a
state-value baseline is allowed. For a prompt `q`, the response-level state value is the expected
reward `E_{o~pi}[r_phi(q,o)]`, and the `G` rollouts already drawn for that prompt are Monte-Carlo
samples of it. GRPO uses their in-group mean as the practical baseline and rescales by the group
std for per-prompt scale invariance:

```
A_{i,t} = ( r_i - mean(r_1..r_G) ) / (std(r_1..r_G) + epsilon)
          for every valid token t of response o_i
```

(`r_i = r_phi(q, o_i)`, one scalar per response; broadcast across the response's tokens for
outcome supervision.) A leave-one-out mean would be exactly action-independent for response `i`.
With the in-group mean, the mean-only estimator has expectation `(G-1)/G` times the raw
response-level policy-gradient term; after std normalization it is best read as the normalized
group-relative coefficient used by the GRPO objective, not as an exactly unbiased estimator of
the raw-return gradient.

## Full objective

Keep PPO's clipped surrogate (so several inner steps per batch stay stable), with importance
ratio `rho_{i,t} = pi_theta(o_{i,t}|q,o_{i,<t}) / pi_{theta_old}(o_{i,t}|q,o_{i,<t})`, and move
the KL-to-reference regularizer out of the reward and into the loss so the advantage stays a
clean group-normalized reward:

```
J(theta) = E_{q, {o_i}~pi_old} (1/G) sum_i (1/|o_i|) sum_t
             { min[ rho_{i,t} A_{i,t}, clip(rho_{i,t}, 1-eps, 1+eps) A_{i,t} ]
               - beta * D_KL[ pi_theta || pi_ref ] }
```

with the per-token k3 KL estimator. For `u = pi_ref(o_t)/pi_theta(o_t)`, `u - 1` has zero
expectation under `o_t ~ pi_theta`, so it is a control variate for the naive estimator
`-log u`; and `log x <= x - 1` makes each sampled term non-negative:

```
k3(o_t) = pi_ref(o_t)/pi_theta(o_t) - log(pi_ref(o_t)/pi_theta(o_t)) - 1
```

In the single-inner-step case (`pi_old = pi_theta`, `rho = 1`) the effective per-token gradient
coefficient is `A_{i,t} + beta*(pi_ref/pi_theta - 1)`: the group-normalized advantage plus a
pull-to-reference term that vanishes when the policy equals the reference. Unlike keeping only
correct samples (coefficient `1[correct] in {0,1}`), this coefficient is signed and
magnitude-aware — it penalizes wrong answers and reinforces by how far a response beat its peers.

## Code — reward-stage group-std normalization

This is the per-prompt mean-and-std normalization applied to the reward tensor (the `group_std`
form). It recovers the per-response scalar from the last-token sum, z-scores it within each
prompt group, and re-places it at the last valid token, masked to the response.

```python
import torch
from collections import defaultdict


def normalize_rewards(token_level_scores, response_mask, index=None, epsilon=1e-6, **kwargs):
    """group-std: per-prompt group mean + std normalization of the response reward."""
    with torch.no_grad():
        bsz, seq_len = token_level_scores.shape
        scores = token_level_scores.sum(dim=-1)              # (bs,): per-response scalar

        if index is None:
            # no grouping info -> fall back to batch-level normalization
            mean = scores.mean()
            std = scores.std(unbiased=False)
            scores = (scores - mean) / (std + epsilon)
        else:
            id2score = defaultdict(list)
            id2mean, id2std = {}, {}
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx, vs in id2score.items():
                if len(vs) == 1:
                    id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                    id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
                else:
                    stacked = torch.stack(vs)
                    id2mean[idx] = stacked.mean()
                    id2std[idx] = stacked.std(unbiased=False)
            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        # re-place the normalized scalar at the last valid token (outcome-reward semantics)
        out = torch.zeros_like(token_level_scores)
        last_idx = (response_mask.long().sum(dim=-1) - 1).clamp(min=0)   # (bs,)
        out[torch.arange(bsz, device=out.device), last_idx] = scores
        return out * response_mask
```

## Code — group-relative advantage (outcome supervision)

The downstream advantage-stage counterpart: z-score per prompt, broadcast across all valid tokens.

```python
import torch
from collections import defaultdict


@torch.no_grad()
def compute_group_relative_advantage(
    token_level_rewards, response_mask, index, epsilon=1e-6, norm_adv_by_std_in_grpo=True
):
    scores = token_level_rewards.sum(dim=-1)                 # (bs,): per-response scalar
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            # Verl fallback: no group statistic exists, so keep the scalar finite.
            id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        else:
            v = torch.stack(id2score[idx])
            id2mean[idx] = v.mean()
            id2std[idx] = v.std()                          # torch.std default: sample std
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            scores[i] = scores[i] - id2mean[index[i]]
    advantages = scores.unsqueeze(-1) * response_mask       # broadcast to valid tokens
    return advantages, advantages
```

Process-supervision variant (when a per-step reward model exists): normalize all step rewards
across the group the same way, `r~_i^{j} = (r_i^{j} - mean(R)) / std(R)`, and set
`A_{i,t} = sum_{index(j) >= t} r~_i^{index(j)}` — the sum of normalized rewards of steps at or
after token `t`. Outcome supervision is the special case of a single terminal step.

## Why each choice

- **Group mean instead of a learned critic:** the only structural job of `V` is a state
  baseline; the group of same-prompt rollouts already drawn gives a sample estimate of that
  response-level baseline, removing the policy-sized critic and the per-token-value-fitting
  problem under a last-token-only reward. The exact unbiased-baseline theorem applies to a
  state-only or leave-one-out baseline; the in-group mean/std rule is the normalized coefficient
  used by the objective.
- **Divide by group std:** without it, the reward model's arbitrary per-prompt range sets the
  step size; the z-score makes easy and hard prompts contribute on a comparable scale, keyed to
  within-prompt ordering only.
- **`epsilon` in the denominator + 1-sample fallback:** identical multi-sample groups produce
  clean zero coefficients instead of `0/0`; the one-sample fallback used by verl/group_std sets
  mean=0,std=1, which is a finite pass-through rather than a meaningful relative advantage.
- **KL in the loss, not the reward:** folding KL into the reward before grouping would
  contaminate the group mean/std; as a separate sampled-loss term its gradient is derived
  independently and the advantage stays the clean normalized reward.
- **k3 KL estimator:** unbiased under samples from the policy because
  `pi_ref/pi_theta - 1` has zero mean, always non-negative by `log x <= x-1`, and with the
  first-order log-ratio fluctuation canceled near `pi_theta = pi_ref`.
- **Clipped surrogate kept from PPO:** a pessimistic lower bound that keeps multiple inner
  gradient steps per batch of rollouts stable.
