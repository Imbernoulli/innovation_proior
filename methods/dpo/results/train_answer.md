The alignment problem is to take a large language model that already knows how to speak and steer it toward the responses humans actually prefer, using only pairwise comparisons {x, y_w, y_l} where y_w is judged better than y_l. Existing practice does this with a long pipeline: fit a reward model by Bradley-Terry maximum likelihood, then optimize that reward with PPO under a KL penalty that keeps the policy near a supervised-fine-tuned reference. This works, but it is painful to run at scale. It requires three coupled models, on-policy sampling from a multi-billion-parameter language model inside the training loop, a learned value baseline, careful reward normalization, and a KL coefficient that must be babysat. It is unstable and expensive. Simpler alternatives also fail: fine-tuning only on the preferred completion ignores the contrast that the preferences encode; unlikelihood training that directly raises p(y_w) and lowers p(y_l) degenerates into repetitive nonsense because nothing stops the likelihood minimization from running away.

I propose Direct Preference Optimization, or DPO. The starting point is the same KL-constrained reward-maximization objective that RLHF optimizes: maximize expected reward minus β times the KL divergence from the reference policy. This objective has a well-known closed-form optimum, an exponential tilt of the reference distribution by the reward, but nobody uses it directly because its partition function Z(x) is a sum over all possible completions and is intractable for language. The usual response is to approximate that optimum with RL. DPO takes a different route: it solves the closed-form relation for the reward instead of for the policy. Writing the optimum as π_r(y|x) ∝ π_ref(y|x) exp(r(x,y)/β) and taking logs gives r(x,y) = β log(π_r(y|x)/π_ref(y|x)) + β log Z(x). The partition-function term depends only on the prompt x, not on the completion y. Bradley-Terry preferences, however, depend only on reward differences for the same prompt, so that x-only term cancels. The optimal policy therefore satisfies a Bradley-Terry model whose reward is the implicit reward r̂(x,y) = β log(π_θ(y|x)/π_ref(y|x)). Fitting the policy directly by maximum likelihood on the preference data becomes a single supervised classification loss, with no separate reward model, no value function, and no on-policy sampling.

The DPO loss is L_DPO(π_θ; π_ref) = −E_{(x,y_w,y_l)~D}[ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]. Its gradient raises the log-probability of the preferred completion and lowers that of the dispreferred one, but the update is weighted by σ(r̂(x,y_l) − r̂(x,y_w)). That weight is large when the policy currently ranks the pair in the wrong order and nearly zero when the preferred completion is already comfortably ahead. This self-pacing prevents the degeneration that unweighted unlikelihood causes. Because the loss is just Bradley-Terry maximum likelihood in a reparameterization where the partition function has been eliminated, it inherits the usual consistency properties. It also generalizes naturally to K-way rankings through Plackett-Luce, since the same x-only cancellation happens inside every softmax factor.

In practice, the reference π_ref is the supervised-fine-tuned model; if none exists, it can be built by fine-tuning on the chosen completions. A default β of 0.1 controls the KL leash. The implementation needs only sequence-level log-probabilities for chosen and rejected completions under both the trainable policy and the frozen reference.

```python
import torch
import torch.nn.functional as F


def get_batch_logps(logits, labels, average_log_prob=False):
    """Sum (or average) per-token log-probs over completion tokens.

    logits: (B, T, V) language-model logits.
    labels: (B, T) target token ids, with -100 on prompt / padding positions.
    """
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    summed = (per_token_logps * loss_mask).sum(-1)
    if average_log_prob:
        return summed / loss_mask.sum(-1)
    return summed


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
    label_smoothing=0.0,
    reference_free=False,
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)
    logits = pi_logratios - ref_logratios

    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def concatenated_forward(model, batch):
    """One forward pass over chosen+rejected concatenated inputs."""
    all_logits = model(
        batch["concat_input_ids"], attention_mask=batch["concat_attention_mask"]
    ).logits
    all_logps = get_batch_logps(all_logits, batch["concat_labels"])
    n_chosen = batch["chosen_input_ids"].shape[0]
    return all_logps[:n_chosen], all_logps[n_chosen:]


def train_step(policy_model, reference_model, batch, beta, optimizer):
    policy_chosen_logps, policy_rejected_logps = concatenated_forward(policy_model, batch)
    with torch.no_grad():
        ref_chosen_logps, ref_rejected_logps = concatenated_forward(reference_model, batch)

    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
    )
    loss = losses.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss.item(), accuracy.item()
```
