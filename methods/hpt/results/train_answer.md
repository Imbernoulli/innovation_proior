The task is to post-train a pretrained language model on verifiable reasoning problems where every prompt comes with both a rule-based verifier and a teacher demonstration. The two natural signals point in opposite directions. On-policy reinforcement learning, in the form of GRPO, samples a group of rollouts, scores them, and standardizes the rewards within the group to form an advantage. When every rollout in the group is wrong, every reward is the same, the centered advantage is identically zero, and the prompt contributes no gradient. That is exactly the case where the model needs help most, so pure RL can sharpen existing competence but cannot bootstrap a new capability the model never samples. Supervised fine-tuning on the demonstration fixes the bootstrap problem, but it is blind to the model's own rollouts: it pulls every teacher token upward regardless of whether the model already solves the prompt, which narrows the policy, encourages memorization, and hurts out-of-distribution generalization. Existing combinations of the two usually rely on a fixed coefficient, a hand-tuned schedule, or a multi-stage pipeline, all of which commit to a static balance even though the right balance varies from prompt to prompt and changes as the model improves.

A cleaner way to think about the problem is to notice that SFT and RL are not two unrelated objectives. If we write a single objective that maximizes expected verifier reward while keeping the model close to the demonstration distribution, its gradient splits cleanly into an on-policy reward term and a behavior-cloning term. After a change of measure to a common reference policy, both terms take the same form: a stabilized gradient estimator whose components are a trust-region mask, a reference-policy denominator, an advantage, and the likelihood gradient. SFT, REINFORCE, PPO, GRPO, LUFFY, and SRFT are all instances of this Unified Policy Gradient Estimator with different choices of reference and advantage. A trust-region penalty simply shifts the advantage by a log-ratio term, and clipping is just a stop-gradient mask on unsafe samples. Because these methods are estimators of the same underlying gradient, the right way to combine them is not a fixed global blend but a per-prompt choice of which estimator is currently most reliable, since the bias and variance of each estimator depend on how well the model already solves the prompt.

I propose Hybrid Post-Training, or HPT. HPT routes each prompt independently between on-policy GRPO and off-policy SFT based on the model's live rollout accuracy on that prompt, all inside a single training pass. For a prompt q, sample n rollouts from the current policy, verify them, and compute the pass-rate P as the fraction that are correct. Then apply a hard switch at a gate gamma: if P is greater than gamma, keep the on-policy rollouts and optimize the standard GRPO clipped surrogate with group-normalized advantages computed over the on-policy group only; if P is less than or equal to gamma, drop the rollout group and take a single supervised fine-tuning step on the teacher demonstration. With gamma set to 0, SFT is used only when every rollout is wrong, which is precisely the degenerate case where GRPO's advantage collapses to zero. A model like Llama that is more fragile at the start can use a higher gate such as two correct rollouts out of eight. The mixture is therefore self-adjusting: early in training more prompts fail and receive SFT updates, while later more prompts cross the gate and continue with RL.

The reason to use plain SFT rather than off-policy RL for the stuck prompts is that the teacher trace was produced by an unknown behavior policy. Off-policy RL would need to set the reference policy to 1, which turns importance sampling into rejection sampling and injects a heavy bias unless the offline data uniformly covers the trajectory space, which it does not. Plain SFT avoids that ill-posed ratio entirely. Keeping the GRPO advantage normalized over on-policy samples only prevents the injected demonstration from contaminating the RL measurement. The result is the minimal demonstration intervention: the teacher is used exactly where the on-policy signal dies, and exploration is preserved everywhere else.

```python
import torch
import torch.nn.functional as F


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def grpo_group_advantage(rewards, eps=1e-6):
    mean = rewards.mean()
    std = rewards.std()
    if std == 0:
        std = 1.0
    return (rewards - mean) / (std + eps)


def policy_loss(log_prob, old_log_prob, advantages, eos_mask, clip=0.2):
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -masked_mean(torch.min(unclipped, clipped), eos_mask)


def sft_loss(log_prob, eos_mask):
    return -masked_mean(log_prob, eos_mask)


def hpt_step(model, old_model, prompt, demo, optimizer, n_rollouts=8, gate=0.0):
    # On-policy rollouts: returns log-probabilities and padding masks.
    roll_logp, roll_mask = sample_rollouts(model, prompt, n_rollouts)
    with torch.no_grad():
        rewards = verify(roll_logp)               # binary scores, shape (n_rollouts,)
        pass_rate = rewards.float().mean().item()

    if pass_rate > gate:
        # RL branch: GRPO with group-normalized advantage over on-policy samples only.
        old_logp, _ = sample_rollouts(old_model, prompt, n_rollouts)
        advantages = grpo_group_advantage(rewards)
        advantages = advantages.unsqueeze(-1).expand_as(roll_logp)
        loss = policy_loss(roll_logp, old_logp, advantages, roll_mask)
    else:
        # SFT branch: plain negative log-likelihood on the demonstration.
        demo_logp, demo_mask = demo_log_probs(model, prompt, demo)
        loss = sft_loss(demo_logp, demo_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
