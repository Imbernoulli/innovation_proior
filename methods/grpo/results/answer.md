# GRPO, distilled

Group Relative Policy Optimization (GRPO) is a critic-free variant of PPO for RL fine-tuning of
language models. It keeps PPO's clipped importance-sampling surrogate but replaces the learned
value-function baseline with the **empirical mean reward of a group of outputs sampled for the
same prompt**, normalized within the group to an advantage. The KL-to-reference regularizer is
moved out of the reward and added directly to the loss with the k3 per-token estimator, whose
unclamped expression is unbiased and always non-negative. The result removes the policy-sized
critic network and the hard problem of fitting a per-token value function under a reward that
arrives only at the last token.

## Problem it solves

On-policy RL fine-tuning of an SFT language model with a reward model `r_phi(q, o)` that emits a
single scalar per response (effectively at the last token). PPO needs a separate, policy-sized
value network `V_psi` to produce the per-token advantages (via GAE); that network is a large
memory/compute cost and is hard to fit accurately per token when the reward is last-token-only.
GRPO keeps PPO's stable update and its variance-reducing baseline while eliminating the critic.

## Key idea

For each question `q`, sample a group of `G` outputs `{o_1, ..., o_G}` from the old policy and
score each with the reward model, getting `{r_1, ..., r_G}`. The group mean is a free Monte-Carlo
estimate of the question's response-level value under the rollout policy, so use it instead of a
learned `V` and accept the usual self-normalized, same-batch baseline approximation. Normalize
within the group (z-score) and broadcast the scalar advantage to every token of the response:

```
A_hat_{i,t} = ( r_i - mean(r_1..r_G) ) / std(r_1..r_G)     for all tokens t of o_i   (outcome supervision)
```

Two choices distinguish GRPO from PPO:

1. **Group-mean baseline instead of a learned critic.** No value network: the baseline is the
   group's own mean reward, which costs nothing (the samples are already drawn) and matches how
   reward models are trained (on comparisons among same-prompt outputs). Dividing by the group
   std puts easy and hard questions on a comparable footing.
2. **KL in the loss, not the reward.** PPO/RLHF folds a per-token KL penalty into the reward
   (`r_t = r_phi - beta*log(pi_theta/pi_ref)`); GRPO leaves the advantage a clean normalized
   reward and adds the KL as a separate loss term, estimated with the k3 form
   `D_KL[pi_theta||pi_ref] = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1`.

## Final objective

With ratio `rho_{i,t} = pi_theta(o_{i,t}|q,o_{i,<t}) / pi_theta_old(o_{i,t}|q,o_{i,<t})`:

```
J_GRPO(theta) = E_{q, {o_i} ~ pi_old} (1/G) sum_i (1/|o_i|) sum_t {
    min[ rho_{i,t} A_hat_{i,t}, clip(rho_{i,t}, 1-eps, 1+eps) A_hat_{i,t} ]
    - beta ( pi_ref(o_{i,t}|.)/pi_theta(o_{i,t}|.) - log(pi_ref(o_{i,t}|.)/pi_theta(o_{i,t}|.)) - 1 )
}
```

**Process-supervision variant** (when a step-level reward model exists): normalize the per-step
rewards `r_tilde_i^{index(j)} = (r_i^{index(j)} - mean(R)) / std(R)` and set a token's advantage
to the sum of normalized rewards from that step onward, `A_hat_{i,t} = sum_{index(j) >= t}
r_tilde_i^{index(j)}`.

**Iterative variant**: periodically set `pi_ref <- pi_theta` and continue-train the reward model
on fresh policy samples while retaining 10% historical data in replay.

## The KL estimator (why this form)

For a single sample `x ~ pi_theta`, let `u = pi_ref(x)/pi_theta(x)`. The naive KL estimator is
`-log u = log(pi_theta/pi_ref)`, unbiased but signed (it can be negative per token) and high
variance. Adding the zero-mean control variate `u - 1` (since `E_{pi_theta}[u-1] =
sum_x pi_ref(x) - 1 = 0`) gives `(u - 1) - log u`: same expectation (still unbiased for the KL),
but always `>= 0` because `log u <= u - 1`. So the per-token KL penalty never swings negative.

## Unified gradient-coefficient view

Every fine-tuning method's gradient can be written
`grad J = E_{(q,o)~D} (1/|o|) sum_t GC(q,o,t) grad_theta log pi_theta(o_t|q,o_<t)`, with a
per-token gradient coefficient `GC`:

| Method | Data source | Gradient coefficient `GC` |
| --- | --- | --- |
| SFT | curated `(q,o)` | `1` |
| RFT / Online RFT | SFT model / live policy | `I(o has correct answer) in {0,1}` |
| PPO | live policy | `A_t` (GAE advantage, needs learned `V`) |
| GRPO | live policy, group of `G` | `A_hat_{i,t} + beta ( pi_ref/pi_theta - 1 )` |

Derivation of `GC_GRPO` (single update per rollout, so `pi_old = pi_theta`, `rho = 1`): the
surrogate term contributes `A_hat_{i,t}`; the KL term contributes, using
`grad_theta(pi_ref/pi_theta) = -(pi_ref/pi_theta) grad log pi_theta` and
`grad_theta(-log(pi_ref/pi_theta)) = grad log pi_theta`,
`grad_theta[-beta(pi_ref/pi_theta - log(pi_ref/pi_theta) - 1)] = beta(pi_ref/pi_theta - 1) grad log pi_theta`.
GRPO's coefficient is graded and signed (it reinforces correct answers by margin and pushes down
wrong ones), where RFT's is a binary gate; it recovers PPO's expressiveness with the group
z-score replacing the GAE advantage and no value network.

## Defaults

Policy learning rate `1e-6`; KL coefficient `beta = 0.04`; group size `G = 64` outputs per
question; max response length `1024`; training batch size `1024`; a single policy update per
exploration (sampling) stage. PPO clip `eps = 0.2`; dual-clip cap `clip_ratio_c = 3.0` for
negative-advantage tokens; advantage-normalization epsilon `1e-6`.

## Working code

Filling the on-policy RL fine-tuning harness: the group-relative advantage estimator, the clipped
actor loss, and the k3 reference-policy penalty.

```python
from collections import defaultdict

import numpy as np
import torch


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1)
    return (values * mask).sum() / denom


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_len); nonzero only at the last token
    response_mask: torch.Tensor,         # (bs, response_len); 1 on response tokens
    index: np.ndarray,                   # (bs,); prompt-group id for each row
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-relative advantage: group mean as baseline, group std z-score, broadcast to tokens."""
    scores = token_level_rewards.sum(dim=-1)         # r_i, one scalar per response

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                g = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(g)         # baseline = group mean reward
                id2std[idx] = torch.std(g)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]   # centered-only variant
        scores = scores.unsqueeze(-1) * response_mask       # broadcast to every token

    return scores, scores                            # (advantages, returns)


def kl_penalty_forward(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    kl_penalty: str = "k3",
) -> torch.Tensor:
    """KL primitive; for k3 the unclamped form is (u - 1) - log u, u = pi_ref/pi_theta."""
    if kl_penalty in ("kl", "k1"):
        return log_prob - ref_log_prob
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (log_prob - ref_log_prob).square()
    if kl_penalty not in ("low_var_kl", "k3"):
        raise NotImplementedError(f"unsupported KL penalty: {kl_penalty}")

    kl = ref_log_prob - log_prob                     # log u
    kl = torch.clamp(kl, min=-20.0, max=20.0)
    u = torch.exp(kl)                                # pi_ref / pi_theta
    kld = u - kl - 1.0                                # (u - 1) - log u  >= 0
    return torch.clamp(kld, min=-10.0, max=10.0)


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,     # under pi_theta_old (the sampler)
    log_prob: torch.Tensor,         # under current pi_theta
    advantages: torch.Tensor,       # from compute_grpo_outcome_advantage
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,        # PPO clip eps
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,      # dual-clip cap for negative-advantage tokens
) -> torch.Tensor:
    """PPO-style clipped actor loss after the group-relative advantages are computed."""
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)            # rho_t = pi_theta / pi_theta_old

    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)      # = -min(rho*A, clip(rho)*A)

    pg_losses3 = -advantages * clip_ratio_c          # dual-clip cap (advantages < 0)
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    return masked_token_mean(pg_losses, response_mask)


def grpo_actor_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    kl_coef: float = 0.04,          # beta
    kl_penalty: str = "k3",
) -> torch.Tensor:
    """Clipped surrogate with the reference-policy penalty kept outside the reward."""
    pg_loss = compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask)
    if ref_log_prob is not None and kl_coef != 0.0:
        kld = kl_penalty_forward(log_prob, ref_log_prob, kl_penalty=kl_penalty)
        kl_loss = masked_token_mean(kld, response_mask)
        pg_loss = pg_loss + kl_coef * kl_loss        # KL as its own loss term

    return pg_loss
```
