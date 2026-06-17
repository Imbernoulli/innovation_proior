# ORPO, distilled

ORPO (Odds Ratio Preference Optimization) is a **reference-free, single-stage** preference
alignment loss for language models. It appends a log-odds-ratio penalty to the ordinary SFT
negative-log-likelihood, so domain adaptation and chosen-vs-rejected preference learning
happen in one training run, with no frozen reference model and no separate SFT warm-up.

## Problem it solves

Standard alignment (RLHF, DPO) is two-stage and reference-based: an SFT warm-up produces a
checkpoint, then a preference phase optimizes the policy *against a frozen copy of it*,
costing a second model in memory and extra forward passes (DPO: four per batch). Worse,
plain SFT on chosen responses has no penalty term — its cross-entropy only rewards label
tokens — so it raises the likelihood of rejected responses alongside chosen ones. ORPO folds
the preference contrast directly into SFT, reference-free and single-stage.

## Key idea

For a length-`m` response, use the length-normalized sequence log-probability
`log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t|x, y_<t)` (so `P ∈ (0,1)`), its **odds**
`odds_θ(y|x) = P_θ(y|x)/(1−P_θ(y|x))`, and the **odds ratio** of chosen over rejected
`OR_θ(y_w,y_l) = odds_θ(y_w|x)/odds_θ(y_l|x)`. The objective is

```
L_ORPO = E_{(x,y_w,y_l)} [ L_SFT + λ · L_OR ],
  L_SFT = −log P_θ(y_w|x),                                  # NLL on the chosen response
  L_OR  = −log σ( log[ odds_θ(y_w|x) / odds_θ(y_l|x) ] ).   # odds-ratio preference penalty
```

The **odds** ratio (not the probability ratio) is the load-bearing choice. For the same input
probabilities, `log OR` is far more spread out than `log PR` (the logit explodes near `P=1`),
so minimizing `−log σ(log OR)` requires only a *mild* per-example margin. The probability
ratio's `log PR` is tightly concentrated, so `−log σ(log PR)` forces an *extreme* margin that
over-suppresses rejected-token logits and degenerates a model still adapting during SFT.
`λ` (small, e.g. 0.1) weights the penalty; if it is too large, the contrast can dominate the
SFT adaptation signal and recreate the over-suppression problem the odds ratio is meant to avoid.

## Gradient

With `g = odds_θ(y_w|x)/odds_θ(y_l|x)` and `L_OR = −log σ(log g)`,

```
∇_θ log σ(log g) = δ(d) · h(d),
∇_θ L_OR = −δ(d) · h(d),
  δ(d) = [ 1 + odds_θ(y_w|x)/odds_θ(y_l|x) ]^{−1},
  h(d) = ∇_θ log P_θ(y_w|x) / (1−P_θ(y_w|x)) − ∇_θ log P_θ(y_l|x) / (1−P_θ(y_l|x)).
```

Derivation sketch: `∇ log σ(log g) = σ(−log g)·∇ log g`, and `σ(−log g) = 1/(1+g) = δ(d)`;
then `∇ log odds(y) = ∇ log P(y) − ∇ log(1−P(y))` with
`∇ log(1−P(y)) = −odds(y)·∇ log P(y)`, giving `∇ log odds(y) = ∇ log P(y)/(1−P(y))`.
Gradient descent therefore moves in the `+δ(d)h(d)` direction. `δ(d) → 0` when the model
already prefers the chosen response and approaches `1` when the rejected response dominates;
`1/(1−P)` is the log-odds sensitivity factor, growing as the corresponding response becomes
more probable. No second model appears: the contrast is computed from the current policy's
chosen/rejected probabilities and their complements.

## Code (faithful to the LLaMA-Factory ORPO branch)

```python
import torch
import torch.nn.functional as F


def get_batch_logps(logits, labels):
    """Per-response SUMMED label log-probs and valid (non-pad) lengths."""
    # gather log P(label_t | x, y_<t) over valid positions, sum per response
    return summed_logps, valid_length  # both shape (batch,)


class PreferenceTrainer:
    def __init__(self, model, beta=0.1):
        self.model = model
        self.beta = beta  # lambda: weight on the odds-ratio penalty

    def concatenated_forward(self, batch):
        labels = batch.pop("labels")
        logits = self.model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        summed_logps, valid_length = get_batch_logps(logits, labels)
        # ORPO uses average log-probs: log of the geometric-mean token probability.
        all_logps = summed_logps / valid_length
        bsz = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(bsz, dim=0)
        chosen_logits, rejected_logits = logits.split(bsz, dim=0)
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps,
        }

    def odds_ratio_loss(self, chosen_logps, rejected_logps):
        # log OR = (c - r) - [log(1-P_w) - log(1-P_l)].
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        return sft_loss + self.beta * odds_ratio_loss

    def compute_preference_loss(self, policy_chosen_logps, policy_rejected_logps):
        losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()
        return losses, chosen_rewards, rejected_rewards

    def compute_loss(self, batch):
        output = self.concatenated_forward(batch)
        losses, _, _ = self.compute_preference_loss(output["chosen_logps"], output["rejected_logps"])
        return losses.mean()
```

## Why it works / properties

- **Single stage, reference-free**: one model in memory, two forward passes per batch
  (chosen, rejected) vs DPO's four; no SFT warm-up — `L_SFT` adapts from the pre-trained
  checkpoint directly.
- **Self-pacing**: `δ(d)` shuts the update direction off on solved examples and fires it on
  examples where the rejected response dominates.
- **Mild, SFT-compatible discrimination**: the odds ratio's wide log-range avoids the
  extreme rejected-logit suppression a probability-ratio contrast would force during SFT.
