# SimPO, distilled

SimPO (Simple Preference Optimization) is an offline preference-optimization objective for language
models that needs **no reference model**. It replaces DPO's reference-based log-ratio reward with the
policy's own **length-normalized average log-likelihood** — which is the metric the model is ranked by
at generation — and adds a **target reward margin** `gamma` to the Bradley-Terry objective.

## Problem it solves

Tune an SFT policy `pi_theta(y|x)` from offline preference pairs `(x, y_w, y_l)` so that (1) no
reward model and no frozen reference model are needed (light memory/compute), (2) the training reward
is the same quantity used to rank/generate at inference (average log-likelihood), (3) the objective
does not reward verbosity, and (4) winners are separated from losers by a controllable margin. DPO
keeps a reference model, optimizes a log-ratio reward that differs from the generation metric (so its
training ordering need not imply the average-log-likelihood ordering), and has a length-unnormalized
gradient that lets long responses dominate.

## Key idea

Two components.

1. **Length-normalized, reference-free reward** — the policy's average per-token log-probability, i.e.
   the generation-ranking metric:

   ```
   r_SimPO(x, y) = (beta / |y|) log pi_theta(y|x) = (beta / |y|) sum_i log pi_theta(y_i | x, y_<i)
   ```

   The `1/|y|` removes the structural length penalty of summed log-probs (so no incentive to inflate
   long sequences) and makes the reward identical in kind to the inference metric; dropping `pi_ref`
   makes it reference-free. Regularization is left implicit in the training regime (small lr, few
   epochs, diverse data, LLM robustness) rather than an explicit KL term.

2. **Target reward margin** `gamma > 0` — an additive bias in the Bradley-Terry model so the winner
   must outscore the loser by at least `gamma`, not merely by an infinitesimal amount:

   ```
   p(y_w > y_l | x) = sigma( r(x, y_w) - r(x, y_l) - gamma )
   ```

   A margin improves generalization (the max-margin / Bradley-Terry "home-advantage" principle).
   Too small a `gamma` gives little separation; too large a `gamma` can over-suppress fluent losing
   responses, so `gamma` is tuned.

## Final objective

```
L_SimPO(pi_theta) = - E_{(x,y_w,y_l)~D} [
    log sigma( (beta/|y_w|) log pi_theta(y_w|x) - (beta/|y_l|) log pi_theta(y_l|x) - gamma )
]
```

**Gradient** (reference-free weight, length-normalized updates):

```
nabla L_SimPO = - beta * E[ s_theta * ( (1/|y_w|) nabla log pi(y_w|x) - (1/|y_l|) nabla log pi(y_l|x) ) ]
s_theta = sigma( (beta/|y_l|) log pi(y_l|x) - (beta/|y_w|) log pi(y_w|x) + gamma )
```

`s_theta` is large exactly when the policy wrongly ranks `y_l` above `y_w` (plus the margin), and uses
no reference model; each response's gradient is divided by its own length, so long responses no longer
dominate (unlike DPO's un-normalized `nabla log pi(y_w) - nabla log pi(y_l)`).

## Defaults and tuning

- `beta` scales the reward difference (sigmoid sharpness); chat settings use roughly `beta in [2, 2.5]`.
- Tune `gamma` directly over `{0.3, 0.5, 1.0, 1.2, 1.4, 1.6}`; the official trainer stores the
  equivalent normalized code parameter `gamma_beta_ratio = gamma / beta`.
- Small learning rate (~`5e-7`–`1e-6`), cosine schedule with ~10% warmup, batch size ~128, a few
  epochs, max sequence length 2048, Adam.
- For a math task with a 1.5B math-instruct base and response-level chosen/rejected pairs, a typical
  setting is `beta = 2.0`, `gamma = 1.0` (so `gamma/beta = 0.5`), `lr = 5e-7`, cosine + 10% warmup.

## Relation to prior objectives

- **DPO** = Bradley-Terry on the reference-based reward `beta log[pi/pi_ref]`. SimPO replaces that
  reward with the length-normalized reference-free `(beta/|y|) log pi` and adds the margin.
- **IPO** also enforces a target gap, but through a squared regression of a reference-based log-ratio;
  SimPO keeps the logistic loss, drops the reference, and length-normalizes.
- **ORPO** is reference-free too, but via an odds-ratio penalty plus an SFT term, not the
  average-log-likelihood reward with a margin.
- **SimPO without length normalization** reduces to a contrastive reward-maximization objective close
  to CPO (without its SFT loss) and reintroduces length bias.

## Note on math-reasoning preferences

When chosen and rejected solutions differ in very few tokens (`2+2=4` vs `2+2=5`), a contrastive
objective can grow the reward margin while the absolute likelihood of the *chosen* sequence falls
(pushing the near-identical rejected sequence down drags the chosen down too). The margin term does
not fix this on its own. An optional supervised anchor on the winner mitigates it:

```
L_SimPO+SFT = - E[ log sigma( (beta/|y_w|) log pi(y_w|x) - (beta/|y_l|) log pi(y_l|x) - gamma )
                   + lambda log pi(y_w|x) ]
```

## Working code

The objective fills the two slots of the paired-preference harness: a `sequence_score` that returns
the **average** log-likelihood (length normalization), and a `compute_preference_loss` that forms the
margin-shifted Bradley-Terry loss. This mirrors the canonical implementation: LLaMA-Factory divides
per-response log-probs by valid length for the SimPO branch and computes
`logits = (chosen_logps - rejected_logps) - simpo_gamma / beta`; the official trainer computes the
same loss with its stored `gamma_beta_ratio = gamma / beta`; both minimize
`-logsigmoid(beta * logits)`.

```python
import torch
import torch.nn.functional as F


def per_token_logps(logits, labels, ignore_index=-100):
    """Return (summed log-prob over response tokens, response length |y|)."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()
    mask = labels != ignore_index
    labels[~mask] = 0
    token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    summed = (token_logps * mask).sum(-1)      # sum_i log pi(y_i | x, y_<i)
    length = mask.sum(-1)                       # |y|
    return summed, length


def sequence_score(summed_logp, length):
    """Length normalization: average log-likelihood = generation metric, reference-free."""
    return summed_logp / length                # (1/|y|) log pi(y|x)


def simpo_loss(policy_chosen_logps, policy_rejected_logps, beta=2.0, gamma=1.0):
    """SimPO loss from per-sequence AVERAGE log-probs of chosen/rejected responses.

    L = -log sigma( beta*(avg_w - avg_l) - gamma )
      = -log sigma( beta*((avg_w - avg_l) - gamma/beta) )
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps   # avg_w - avg_l
    gamma_logratios = gamma / beta                               # full margin -> code-space ratio
    logits = pi_logratios - gamma_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = (beta * policy_chosen_logps).detach()       # length-normalized implicit reward
    rejected_rewards = (beta * policy_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def train_step(model, batch, optimizer, beta=2.0, gamma=1.0):
    optimizer.zero_grad()
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    summed, length = per_token_logps(logits, batch["labels"])
    n = summed.shape[0] // 2
    chosen_avg = sequence_score(summed[:n], length[:n])
    rejected_avg = sequence_score(summed[n:], length[n:])
    losses, _, _ = simpo_loss(chosen_avg, rejected_avg, beta=beta, gamma=gamma)
    loss = losses.mean()
    loss.backward()
    optimizer.step()
    return loss
```
