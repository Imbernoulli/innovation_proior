# InstructGPT: aligning a language model to user intent with RLHF

## The problem

A language model pretrained by next-token prediction optimizes a proxy ("continue internet text") that diverges from what users want ("follow my instruction, helpfully, truthfully, harmlessly"), and scaling parameters does not close the gap. The goal is to fine-tune such a model so that intent-following is its default behavior, without destroying the broad capabilities acquired in pretraining (i.e. with minimal "alignment tax").

## The key idea

Reinforcement learning from human feedback (RLHF) in three stages, each removing the previous stage's ceiling:

1. **Supervised fine-tuning (SFT).** Fine-tune the pretrained model on human-written demonstrations with the ordinary next-token loss. This puts the model into instruction-following mode and provides the anchor policy $\pi^{\text{SFT}}$ for stage 3. Ceiling: imitation cannot tell a better response from a worse one, nor exceed the demonstrator.

2. **Reward modeling (RM).** Optimizing human judgment directly is impossible — humans are far too slow to sit inside an RL loop — so fit a model $r_\theta(x,y)$ of human judgment from *offline* comparisons. Humans rank $K$ responses ($K\in[4,9]$) per prompt, yielding $\binom{K}{2}$ pairwise comparisons cheaply. Under the Bradley–Terry model the probability a human prefers $y_w$ over $y_l$ is $\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)$, so the reward difference is the log-odds of preference. The RM is the SFT model with its unembedding replaced by a scalar head; a single mid-sized (6B) RM is used for policies of all sizes (larger RMs are unstable as the value function and far costlier).

3. **RL optimization (PPO + KL penalty + pretraining mix).** Treat the model as a bandit policy (state = prompt, action = full response, one-step episode) and maximize the learned reward with PPO. Because $r_\theta$ is only accurate near the SFT distribution and is gamed under hard optimization (reward over-optimization), add a per-token KL penalty to $\pi^{\text{SFT}}$ that leashes the policy to the region where the reward is trustworthy. Because RL fine-tuning regresses standard NLP capabilities (alignment tax) and tightening the KL penalty does *not* repair it, mix the pretraining log-likelihood gradient back in to anchor the capabilities. The result is **PPO-ptx = InstructGPT**.

## The objectives, stated cleanly

Reward-model loss for a ranked prompt $x$, with pair set $\mathcal P(x)$:

$$\ell_x(\theta)=-\frac{1}{\binom{K}{2}}\sum_{(y_w,y_l)\in\mathcal P(x)}\log \sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big),\qquad \mathrm{loss}(\theta)=\mathbb E_{x\sim D_{\mathrm{rank}}}[\ell_x(\theta)].$$

(Reward is identified only up to an additive constant; normalize so demonstrations have mean reward 0 before RL.)

RL objective maximized over the policy $\pi^{\text{RL}}_\phi$:

$$\mathrm{objective}(\phi)=\mathbb{E}_{(x,y)\sim \pi^{\text{RL}}_\phi}\Big[r_\theta(x,y)-\beta\log\frac{\pi^{\text{RL}}_\phi(y\mid x)}{\pi^{\text{SFT}}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}\big[\log\pi^{\text{RL}}_\phi(x)\big].$$

$\beta$ is the KL coefficient ($\approx 0.02$), $\gamma$ the pretraining-mix coefficient ($\approx 27.8$; $\gamma=0$ gives plain "PPO"). In PPO the bracketed term is realized as a per-token reward (KL penalty per token, RM score added at the final token), then optimized with the clipped surrogate.

## Algorithm

```
1. SFT:  fine-tune pretrained LM on demonstrations (cross-entropy) -> pi_SFT
2. RM:   init from SFT (unembedding -> scalar head); train one epoch on
         comparisons with the Bradley-Terry pairwise loss; normalize bias.
3. PPO:  init policy from SFT, value function from RM. Repeat:
           sample responses from the policy on a batch of prompts;
           per-token reward = -beta*(logp_policy - logp_SFT), plus RM score
             at the final token;
           advantages/returns via GAE (no discount);
           clipped-surrogate policy update + clipped value loss (single inner epoch);
           minimize PPO loss plus gamma * pretraining NLL (ptx);
           step.
```

Key hyperparameters: Adam $(\beta_1,\beta_2)=(0.9,0.95)$; RM lr 9e-6, batch 64 prompts, 1 epoch; PPO clip $\epsilon=0.2$, GAE no discount, $\lambda=0.95$, batch 512 / minibatch 64, 1 inner epoch, KL coef $\beta=0.02$, EMA decay 0.992; ptx coef $\gamma=27.8$ with 8× pretraining examples per RL episode.

## Working code

The code mirrors the relevant TRL conventions: `-logsigmoid(chosen - rejected)`, token reward `-kl_coef * (logp - ref_logp)`, GAE, and the negative PPO surrogate with `max()`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)

def gather_token_logprobs(logits, input_ids):
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    labels = input_ids[:, 1:]
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def response_nll(model, input_ids, loss_mask):
    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    token_logp = gather_token_logprobs(logits, input_ids)
    return -masked_mean(token_logp, loss_mask[:, 1:])

def train_demonstration_model(model, demo_loader, optimizer):
    for input_ids, loss_mask in demo_loader:
        loss = response_nll(model, input_ids, loss_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class PreferenceScorer(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        last = attention_mask.long().sum(dim=1) - 1
        row = torch.arange(h.size(0), device=h.device)
        return self.score(h[row, last]).squeeze(-1)

def preference_loss(scores_chosen, scores_rejected, margin=None, pair_mask=None):
    diff = scores_chosen - scores_rejected
    if margin is not None:
        diff = diff - margin
    losses = -F.logsigmoid(diff)
    if pair_mask is not None:
        pair_mask = pair_mask.to(losses.dtype)
        pair_counts = pair_mask.sum(dim=-1).clamp_min(1)
        per_prompt = (losses * pair_mask).sum(dim=-1) / pair_counts
        valid_prompts = (pair_mask.sum(dim=-1) > 0).to(losses.dtype)
        return (per_prompt * valid_prompts).sum() / valid_prompts.sum().clamp_min(1)
    if losses.ndim > 1:
        return losses.mean(dim=-1).mean()
    return losses.mean()

def train_preference_scorer(scorer, comparison_loader, optimizer):
    # The loader can group all pairs from ranked prompts as [batch, pairs, tokens].
    def score_pair_batch(input_ids, attention_mask):
        if input_ids.dim() == 3:
            bsz, pairs, seqlen = input_ids.shape
            flat_scores = scorer(input_ids.reshape(bsz * pairs, seqlen),
                                 attention_mask.reshape(bsz * pairs, seqlen))
            return flat_scores.view(bsz, pairs)
        return scorer(input_ids, attention_mask)

    for batch in comparison_loader:
        chosen = score_pair_batch(batch.input_ids_chosen, batch.attention_mask_chosen)
        rejected = score_pair_batch(batch.input_ids_rejected, batch.attention_mask_rejected)
        loss = preference_loss(chosen, rejected,
                               margin=getattr(batch, "margin", None),
                               pair_mask=getattr(batch, "pair_mask", None))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def token_rewards(scores, logprobs, ref_logprobs, masks, kl_coef):
    # Response-only logprobs/ref_logprobs/masks (TRL convention).
    kls = logprobs - ref_logprobs
    non_score_rewards = -kl_coef * kls
    rewards = non_score_rewards.clone()
    for i, (score, mask) in enumerate(zip(scores, masks)):
        last = mask.nonzero(as_tuple=False)[-1].item()
        rewards[i, last] += score
    return rewards, non_score_rewards, kls

def masked_whiten(values, mask, shift_mean=True):
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    return whitened if shift_mean else whitened + mean

def estimate_advantages(values, rewards, mask, gamma=1.0, lam=0.95):
    values = values * mask
    rewards = rewards * mask
    lastgaelam = 0
    reversed_advantages = []
    for t in reversed(range(rewards.shape[-1])):
        nextvalues = values[:, t + 1] if t < rewards.shape[-1] - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        reversed_advantages.append(lastgaelam)
    advantages = torch.stack(reversed_advantages[::-1]).transpose(0, 1)
    returns = advantages + values
    advantages = masked_whiten(advantages, mask).detach()
    return values, advantages, returns

def clipped_policy_loss(old_logprobs, values, logits, vpreds, logprobs,
                        mask, advantages, returns, cliprange=0.2,
                        cliprange_value=0.2, vf_coef=0.1):
    vpred_clipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpred_clipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

    ratio = torch.exp(logprobs - old_logprobs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mask)
    return pg_loss + vf_coef * vf_loss

def train_policy(policy, value_model, scorer, reference_policy,
                 prompt_loader, retention_loader, optimizer,
                 kl_coef, retention_coef):
    for prompts, retention_batch in zip(prompt_loader, retention_loader):
        responses, old_logprobs = policy.generate_with_logprobs(prompts)
        model_inputs = pack(prompts, responses)
        response_mask = model_inputs.response_mask

        with torch.no_grad():
            ref_logprobs = reference_policy.logprobs(model_inputs)
            scores = scorer(model_inputs.input_ids, model_inputs.attention_mask)
            old_values = value_model(model_inputs)

        rewards, _, _ = token_rewards(scores, old_logprobs, ref_logprobs, response_mask, kl_coef)
        old_values, advantages, returns = estimate_advantages(old_values, rewards, response_mask)

        logits, logprobs, vpreds = policy.forward_with_values(model_inputs)
        ppo = clipped_policy_loss(old_logprobs.detach(), old_values.detach(), logits, vpreds,
                                  logprobs, response_mask, advantages, returns.detach())
        ptx = response_nll(policy, retention_batch.input_ids, retention_batch.loss_mask)

        optimizer.zero_grad()
        (ppo + retention_coef * ptx).backward()
        optimizer.step()
```
