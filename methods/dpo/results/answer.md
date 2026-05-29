# Direct Preference Optimization (DPO)

## The problem

Align a language model to human preferences given a fixed dataset of pairwise
comparisons {x, y_w, y_l} (preferred ≻ dispreferred). The standard RLHF route
fits a reward model, then maximizes it with PPO under a KL penalty to a
reference policy — a multi-model, sampling-in-the-loop, unstable, expensive
procedure. DPO targets the same KL-regularized solution with one supervised
classification loss, no reward model, no value function, no sampling, no RL.

## Key idea

Optimize the standard RLHF objective

  max_π  E_{x~D, y~π}[r(x,y)] − β·KL(π(y|x) ‖ π_ref(y|x)).

Its closed-form optimum is an exponential tilt of the reference,

  π_r(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β),   Z(x)=Σ_{y'} π_ref(y'|x)exp(r(x,y')/β),

but Z(x) is intractable. Invert the relation to express the reward through its
own optimal policy:

  r(x,y) = β·log( π_r(y|x)/π_ref(y|x) ) + β·log Z(x).

The Bradley-Terry preference model p*(y1≻y2|x) = σ(r(x,y1)−r(x,y2)) depends only
on reward *differences*, and β·log Z(x) is a function of x alone, so it cancels.
The optimal policy therefore satisfies a Bradley-Terry model whose reward is the
*implicit reward* r̂(x,y) = β·log(π(y|x)/π_ref(y|x)). Fitting the policy by
maximum likelihood on the preferences is then a single binary-cross-entropy
loss.

## Final objective

  L_DPO(π_θ; π_ref) =
    − E_{(x,y_w,y_l)~D} [ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x))
                                  − β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ].

Gradient (mechanism):

  ∇_θ L_DPO = − β·E[ σ( r̂(x,y_l) − r̂(x,y_w) )·( ∇log π_θ(y_w|x) − ∇log π_θ(y_l|x) ) ],

with r̂ = β·log(π_θ/π_ref). It raises the preferred completion and lowers the
dispreferred one, weighted by how wrongly the implicit reward currently orders
the pair (weight → 1 when wrong, → 0 when already correct). This dynamic weight
prevents the degeneration that an unweighted likelihood/unlikelihood objective
causes.

This generalizes to K-way rankings via Plackett-Luce (the same β·log Z(x)
cancellation holds inside each softmax factor), and it is a faithful
reparameterization (under full support, each reward equivalence class has a
unique representative of the form β·log(π/π_ref), so no generality is lost),
inheriting Bradley-Terry MLE consistency.

## Algorithm

1. Get a reference π_ref (= the SFT model). If none exists, set π_ref by
   fine-tuning on the preferred completions only: argmax_π E[log π(y_w|x)].
2. For each batch of pairs, compute sequence log-probs (sum of per-token
   log-probs over completion tokens) for chosen and rejected under both π_θ and
   the frozen π_ref.
3. Minimize L_DPO. Defaults: β=0.1 (0.5 for TL;DR summarization), RMSprop,
   lr 1e-6, linear warmup over ~150 steps, batch 64.

## Working code

```python
import torch
import torch.nn.functional as F


def get_batch_logps(logits, labels, average_log_prob=False):
    """Per-sequence log-prob: sum of log p(token) over completion tokens.

    logits: (B, T, V) language-model logits.
    labels: (B, T) target token ids, with -100 on prompt / padding positions.
    """
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()      # token t is predicted from positions < t
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)        # score only completion tokens
    labels[labels == -100] = 0          # gather needs valid indices
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    return (per_token_logps * loss_mask).sum(-1)   # (B,) — sum, per the BT derivation


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta, label_smoothing=0.0, reference_free=False):
    """DPO loss for a batch of preference pairs.

    *_logps: sequence log-probs, shape (B,). beta: KL strength.
    """
    pi_logratios  = policy_chosen_logps   - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:                  # drop the reference anchor (ablation)
        ref_logratios = torch.zeros_like(pi_logratios)
    logits = pi_logratios - ref_logratios          # β log Z cancels here

    # binary cross-entropy that y_w outscores y_l (label_smoothing optional)
    losses = (-F.logsigmoid(beta * logits) * (1 - label_smoothing)
              - F.logsigmoid(-beta * logits) * label_smoothing)

    # implicit rewards r̂ = β log(π_θ/π_ref); detached, for logging the margin
    chosen_rewards   = beta * (policy_chosen_logps   - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def concatenated_forward(model, batch):
    """One forward pass over chosen+rejected concatenated, returns (chosen, rejected) logps."""
    all_logits = model(batch["concat_input_ids"],
                       attention_mask=batch["concat_attention_mask"]).logits
    all_logps = get_batch_logps(all_logits, batch["concat_labels"])
    n_chosen = batch["chosen_input_ids"].shape[0]
    return all_logps[:n_chosen], all_logps[n_chosen:]


def train_step(policy_model, reference_model, batch, beta, optimizer):
    policy_chosen_logps, policy_rejected_logps = concatenated_forward(policy_model, batch)
    with torch.no_grad():               # reference frozen — no gradient, no RL
        ref_chosen_logps, ref_rejected_logps = concatenated_forward(reference_model, batch)
    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps, beta)
    loss = losses.mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()   # implicit-reward acc
    return loss.item(), accuracy.item()
```

The code is faithful to the canonical implementation: `get_batch_logps` and the
log-ratio/logsigmoid loss mirror the standard DPO trainer, with the reference
model frozen and run under `no_grad`, and the policy/reference forward passes
batched over the chosen+rejected concatenation for efficiency.
