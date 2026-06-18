**Problem (from step 2).** IPO defends the correct chain's likelihood against a frozen reference but is
conservative — it regresses the gap onto a fixed target and goes silent once met, so it never *grows* the
correct chain, and it pays for a second resident model (four forwards/step) to provide the anchor. I want
active likelihood growth on the correct chain *and* a contrast against the rejected one, with no reference.

**Key idea.** Fuse SFT and the preference contrast into one reference-free stage. The NLL on the chosen
chain (`L_SFT = −log P_θ(y_w)`) actively grows the correct chain's likelihood; a penalty term contrasts it
against the rejected chain. Use the **odds ratio**, not the probability ratio, for the contrast: the log of
the odds ratio is far more spread out (the logit explodes near P=1) than the log probability ratio, so a
log-sigmoid on it reaches its target with a *mild* per-example margin instead of crushing the rejected
chain's tokens — which matters because the rejected math chain shares almost all its tokens with the
correct one.

```
L_ORPO = E[ L_SFT + λ·L_OR ],   L_SFT = -log P_θ(y_w|x),
L_OR = -log σ( log[ odds_θ(y_w|x) / odds_θ(y_l|x) ] ),   odds_θ(y) = P_θ(y) / (1 - P_θ(y)),
log P_θ(y|x) = length-normalized (geometric-mean) sequence log-prob.
```

**Why it works here.** The SFT term is the active growth IPO lacked; the odds-ratio penalty is mild
(spread-out log-odds), so it trims the wrong chain without dragging the near-identical correct chain down.
The gradient is `+δ·h` with self-pacing weight `δ = [1 + odds_w/odds_l]^{−1}` and contrast direction
`h = ∇log P(y_w)/(1−P(y_w)) − ∇log P(y_l)/(1−P(y_l))` — no second model anywhere, one stage, two
forwards/batch.

**Substrate wiring.** `orpo` is in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` hands the
loss **average** per-token log-probs (so `P_θ ∈ (0,1)` and the odds is finite). `orpo` *is* in the
reference-free set, so `use_ref_model` is False, no reference is loaded, and the loss lands in the
reference-free branch dispatched to `odds_ratio_loss`. `self.beta` plays the role of λ. The SFT term is
added by `get_batch_loss_metrics` via `−chosen_logps_avg` inside `odds_ratio_loss`.

**Scaffold edit / hyperparameters.** `pref_loss=orpo`, `pref_beta=0.1` (= λ), `lr=5e-7`, cosine + 10%
warmup, 4 epochs. Numerically stable via `log1p(-exp(·))` and `logsigmoid`.

```python
def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
    r"""Compute ORPO's odds ratio (OR) loss for batched (length-averaged) log probabilities."""
    log_odds = (chosen_logps - rejected_logps) - (
        torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
    )                                                   # log[odds_w / odds_l], stable
    sft_loss = -chosen_logps                            # active NLL growth on the correct chain
    odds_ratio_loss = -F.logsigmoid(log_odds)           # mild contrast penalty
    orpo_loss = sft_loss + self.beta * odds_ratio_loss  # self.beta == lambda
    return orpo_loss


def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""Compute loss for preference learning."""
    if not self.finetuning_args.use_ref_model:                       # orpo: reference-free branch
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    else:
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
    return losses, chosen_rewards, rejected_rewards
```
