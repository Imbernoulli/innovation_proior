The task is to post-train a pretrained language model on verifiable reasoning problems where every prompt comes with both a rule-based verifier and a teacher demonstration. The two natural signals point in opposite directions. On-policy reinforcement learning, in the form of GRPO, samples a group of rollouts, scores them, and standardizes the rewards within the group to form an advantage. When every rollout in the group is wrong, every reward is the same, the centered advantage is identically zero, and the prompt contributes no gradient. That is exactly the case where the model needs help most, so pure RL can sharpen existing competence but cannot bootstrap a new capability the model never samples. Supervised fine-tuning on the demonstration fixes the bootstrap problem, but it is blind to the model's own rollouts: it pulls every teacher token upward regardless of whether the model already solves the prompt, which narrows the policy, encourages memorization, and hurts out-of-distribution generalization. Existing combinations of the two usually rely on a fixed coefficient, a hand-tuned schedule, or a multi-stage pipeline, all of which commit to a static balance even though the right balance varies from prompt to prompt and changes as the model improves.

A cleaner way to think about the problem is to notice that SFT and RL are not two unrelated objectives. If we write a single objective that maximizes expected verifier reward while keeping the model close to the demonstration distribution, its gradient splits cleanly into an on-policy reward term and a behavior-cloning term. After a change of measure to a common reference policy, both terms take the same form: a stabilized gradient estimator whose components are a trust-region mask, a reference-policy denominator, an advantage, and the likelihood gradient. SFT, REINFORCE, PPO, GRPO, LUFFY, and SRFT are all instances of this Unified Policy Gradient Estimator with different choices of reference and advantage. A trust-region penalty simply shifts the advantage by a log-ratio term, and clipping is just a stop-gradient mask on unsafe samples. Because these methods are estimators of the same underlying gradient, the right way to combine them is not a fixed global blend but a per-prompt choice of which estimator is currently most reliable, since the bias and variance of each estimator depend on how well the model already solves the prompt.

I propose Hybrid Post-Training, or HPT. HPT routes each prompt independently between on-policy GRPO and off-policy SFT based on the model's live rollout accuracy on that prompt, all inside a single training pass. For a prompt q, sample n rollouts from the current policy, verify them, and compute the pass-rate P as the fraction that are correct. Then apply a hard switch at a gate gamma: if P is greater than gamma, keep the on-policy rollouts and optimize the standard GRPO clipped surrogate with group-normalized advantages computed over the on-policy group only; if P is less than or equal to gamma, drop the rollout group and take a single supervised fine-tuning step on the teacher demonstration. With gamma set to 0, SFT is used only when every rollout is wrong, which is precisely the degenerate case where GRPO's advantage collapses to zero. A model like Llama that is more fragile at the start can use a higher gate such as two correct rollouts out of eight. The mixture is therefore self-adjusting: early in training more prompts fail and receive SFT updates, while later more prompts cross the gate and continue with RL.

The reason to use plain SFT rather than off-policy RL for the stuck prompts is that the teacher trace was produced by an unknown behavior policy. Off-policy RL would need to set the reference policy to 1, which turns importance sampling into rejection sampling and injects a heavy bias unless the offline data uniformly covers the trajectory space, which it does not. Plain SFT avoids that ill-posed ratio entirely. Keeping the GRPO advantage normalized over on-policy samples only prevents the injected demonstration from contaminating the RL measurement. The result is the minimal demonstration intervention: the teacher is used exactly where the on-policy signal dies, and exploration is preserved everywhere else.

Concretely, the switch is a batch edit inside the actor rather than a separate code path. A controller looks at how many of the on-policy rollouts for a prompt solved it and returns how many rollouts to drop and how many demonstration rows to splice in: when the solve count is at or below the gate it removes the eight-response on-policy group and adds one demonstration row; otherwise it leaves the on-policy batch untouched. The advantage computation only ever aggregates over rollouts flagged as on-policy, so a spliced-in demonstration row never enters the group mean or standard deviation that GRPO standardizes against. The actor's loss then reads a prefix mask to tell which rows are demonstrations: those get the plain token negative log-likelihood, the remaining on-policy rows get the clipped GRPO surrogate, and the two are combined as `pg_loss = sft_loss * sft_loss_coef + pg_loss`, guarded so a batch with no demonstration rows just falls back to the RL loss untouched.

```python
from collections import defaultdict
import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo import core_algos


def select_on_off_ada_balance(config, on_solve_num):
    """Return (on_remove_num, on_add_num, off_add_num), as in the mix_src trainer."""
    if config.trainer.unify_strategy == "switch":
        on_add_num = 0
        if on_solve_num <= config.trainer.switch_gate:
            return 8, on_add_num, 1          # remove on-policy group, add SFT target sample
        if on_solve_num <= config.trainer.switch_gate_off:
            return 8, on_add_num, -1         # optional off-policy-RL arm in the shared path
        return 0, on_add_num, 0              # keep on-policy GRPO samples

    if config.trainer.unify_strategy == "soft":
        return 0, 0, 1

    raise NotImplementedError


def compute_grpo_outcome_advantage_split(token_level_rewards, eos_mask, index,
                                         on_policy_mask, epsilon=1e-6, use_std=True):
    """Compute group-normalized advantages using only non-prefix (on-policy) samples."""
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    id2score, id2mean, id2std = defaultdict(list), {}, {}

    with torch.no_grad():
        for i in range(scores.shape[0]):
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for uid, values in id2score.items():
            if len(values) == 1:
                id2mean[uid] = torch.tensor(0.0)
                id2std[uid] = torch.tensor(1.0)
            else:
                id2mean[uid] = torch.mean(torch.tensor(values))
                id2std[uid] = torch.std(torch.tensor([values]))
                if id2std[uid].item() == 0:
                    id2std[uid] = torch.tensor(1.0)
        for i in range(scores.shape[0]):
            centered = scores[i] - id2mean[index[i]]
            scores[i] = centered / (id2std[index[i]] + epsilon) if use_std else centered

    advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return advantages, advantages


def compute_sft_pure_loss(log_prob, eos_mask):
    return verl_F.masked_mean(-log_prob, eos_mask)


def actor_mixed_loss(log_prob, old_log_prob, advantages, response_mask, prefix_mask, config):
    off_policy_mask = prefix_mask.any(-1)
    sft_loss = compute_sft_pure_loss(
        log_prob=log_prob[off_policy_mask],
        eos_mask=response_mask[off_policy_mask],
    )

    on_policy_mask = ~off_policy_mask
    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
        old_log_prob=old_log_prob[on_policy_mask],
        log_prob=log_prob[on_policy_mask],
        advantages=advantages[on_policy_mask],
        eos_mask=response_mask[on_policy_mask],
        cliprange=config.clip_ratio,
        loss_remove_token_mean=config.loss_remove_token_mean,
        loss_remove_clip=config.loss_remove_clip,
    )

    if not torch.isnan(sft_loss):
        pg_loss = sft_loss * config.sft_loss_coef + pg_loss

    return pg_loss, pg_clipfrac, ppo_kl
```
