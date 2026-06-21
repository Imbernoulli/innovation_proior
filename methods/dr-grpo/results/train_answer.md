I am fine-tuning a base language model with reinforcement learning on math-reasoning problems where the reward is a simple verifiable outcome: each sampled response gets a 1 if its final answer is correct and a 0 otherwise. The policy is the autoregressive model itself, so the only thing shaping how each response is weighted in the gradient is the advantage estimator. The estimator takes a group of scored responses to the same question and emits a per-token advantage that the PPO actor loss multiplies against the log-probability ratio.

The observed problem is that response length climbs steadily during training even after reward has flattened, and the growth is concentrated in incorrect responses. This is not the emergence of deeper reasoning; it is tokens being spent with no accuracy payoff. The culprit is GRPO, the dominant critic-free estimator for this setting. GRPO centers each response's return by the group mean and then divides by the group's standard deviation, and the PPO loss is usually implemented with a per-response length average. Both choices look like harmless normalization, but both reweight the gradient in unintended ways. The per-response 1/|o_i| term makes long incorrect responses penalized less per token than short incorrect ones, giving wrong answers an incentive to sprawl. The per-group std rescales each prompt by the spread of its rewards, inflating nearly-unanimous easy or hard prompts and suppressing genuinely informative mixed-outcome prompts. Neither term appears in the derivation of the policy-gradient objective.

The fix is Dr. GRPO, short for GRPO Done Right. It keeps the group-mean baseline that makes GRPO critic-free, but removes the per-group standard-deviation scaling and pairs the estimator with a loss aggregation that uses a constant divisor instead of per-response length. The result is the plain Monte-Carlo policy-gradient direction induced by a leave-one-out baseline, written in its group-mean-centered form.

The derivation starts from the standard policy-gradient identity. Because the reward is nonzero only at the final token, the reward-to-go from every token equals the whole-trajectory return, so the advantage is a single scalar per response broadcast over its tokens. With a multi-sample baseline, the leave-one-out advantage for response i is R_i minus the mean of the other G-1 responses. The algebra shows that (G/(G-1)) times the group-mean-centered score R_i - mean(R) equals exactly that leave-one-out advantage. So R_i - mean(R) is the unbiased RLOO estimator up to a global constant that folds into the learning rate. No standard deviation appears, and no per-response length factor appears. For a group whose responses are all scored the same, the centered advantage is simply zero, which is the correct signal. For a singleton group, the mean is set to zero so the raw return passes through.

Dr. GRPO therefore computes per-response returns by summing the per-token outcome reward, subtracts the group mean, and broadcasts that scalar over the valid tokens of each response. The PPO actor loss then sums the per-token losses and divides by a fixed constant such as the generation budget, rather than by each response's own length. This removes both the length bias and the difficulty bias while keeping the setup fully critic-free.

```python
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length); outcome scalar at last valid token
    response_mask: torch.Tensor,         # (bs, response_length); 1 = valid response token
    index: np.ndarray = None,            # (bs,) group/prompt id; same id = same question
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dr. GRPO: mean-only group-relative advantages for outcome rewards."""
    scores = token_level_rewards.sum(dim=-1)  # (bs,) per-sequence return R_i

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)  # single sample: no group baseline
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))  # cheap baseline
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # R_i - mean(R); no division by std
            scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask  # broadcast over tokens, mask pad

    return scores, scores  # advantage == return for outcome reward


def masked_loss_aggregate(per_token_loss, response_mask, max_tokens):
    # Dr. GRPO: constant divisor preserves the PPO token-sum objective.
    # GRPO's typical masked_mean(..., dim=-1) divides by each response's own
    # length and introduces the length bias we are removing.
    return (per_token_loss * response_mask).sum(-1) / max_tokens
```
