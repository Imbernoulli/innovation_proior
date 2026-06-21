We are fine-tuning an instruction-tuned language model with reinforcement learning on reasoning tasks. A reward model gives one scalar for an entire generated solution, effectively attached only to the last token. The standard tool is PPO, which trains a separate value network the same size as the policy to serve as a per-token baseline. That is expensive in memory, but the deeper issue is that the value network must estimate every token's value from a single terminal reward, which is exactly the regime where per-token value fitting is hardest. A bad value function feeds biased, noisy advantages into the policy update. We want the stable clipped update and variance reduction that make PPO work, but without carrying a learned critic.

Existing alternatives each miss a piece. SFT and rejection-sampling fine-tuning can only push up selected outputs with a uniform or binary weight; they cannot penalize bad outputs or reward margin. Direct preference optimization is offline and pairwise, so it never learns from the current policy's fresh mistakes. PPO itself has the right structure, but it pays the value-network cost. Process-reward variants add denser credit but still ride on the actor-critic stack.

The method is Group Relative Policy Optimization, GRPO. For each question it samples a group of outputs from the old policy, scores each with the reward model, and turns the group into a baseline. The group mean reward is a free Monte-Carlo estimate of the question's response-level value, so it replaces the learned value function. The advantage is the per-question z-score, broadcast to every token because the terminal reward gives no finer within-sequence signal:

A_hat_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps),

applied as A_hat_{i,t} for every token t of response o_i. Normalizing by the group std puts easy and hard questions on a comparable scale. The actor update is PPO's clipped surrogate on these advantages, and the KL regularizer is kept as a separate loss term rather than folded into the reward, so it is not entangled with the group normalization.

The KL term uses the k3 estimator. For a sampled token with u = pi_ref / pi_theta, the naive log-ratio is unbiased but can be negative per sample. Adding the zero-mean control variate (u - 1) gives (u - 1) - log u, which has the same expectation but is always non-negative because log u <= u - 1. The resulting gradient coefficient is A_hat_{i,t} + beta * (pi_ref/pi_theta - 1): a graded, signed reinforcement where rejection sampling has only a binary gate, with no value network anywhere.

Typical defaults are a policy learning rate of 1e-6, KL coefficient beta = 0.04, group size G = 64, PPO clip epsilon 0.2, and one policy update per sampling stage. The whole procedure plugs into the standard generate-score-update loop as a critic-free replacement for PPO.

```python
from collections import defaultdict

import numpy as np
import torch


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1)
    return (values * mask).sum() / denom


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)

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
                id2mean[idx] = torch.mean(g)
                id2std[idx] = torch.std(g)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def kl_penalty_forward(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    kl_penalty: str = "k3",
) -> torch.Tensor:
    if kl_penalty in ("kl", "k1"):
        return log_prob - ref_log_prob
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (log_prob - ref_log_prob).square()
    if kl_penalty not in ("low_var_kl", "k3"):
        raise NotImplementedError(f"unsupported KL penalty: {kl_penalty}")

    kl = ref_log_prob - log_prob
    kl = torch.clamp(kl, min=-20.0, max=20.0)
    u = torch.exp(kl)
    kld = u - kl - 1.0
    return torch.clamp(kld, min=-10.0, max=10.0)


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
    clip_ratio_low: float | None = None,
    clip_ratio_high: float | None = None,
    clip_ratio_c: float = 3.0,
) -> torch.Tensor:
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    if clip_ratio_low is None:
        clip_ratio_low = clip_ratio
    if clip_ratio_high is None:
        clip_ratio_high = clip_ratio

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    return masked_token_mean(pg_losses, response_mask)


def grpo_actor_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    kl_coef: float = 0.04,
    kl_penalty: str = "k3",
) -> torch.Tensor:
    pg_loss = compute_policy_loss_vanilla(old_log_prob, log_prob, advantages, response_mask)
    if ref_log_prob is not None and kl_coef != 0.0:
        kld = kl_penalty_forward(log_prob, ref_log_prob, kl_penalty=kl_penalty)
        kl_loss = masked_token_mean(kld, response_mask)
        pg_loss = pg_loss + kl_coef * kl_loss
    return pg_loss
```
