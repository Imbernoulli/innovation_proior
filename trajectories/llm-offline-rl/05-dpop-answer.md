**Problem (from step 4).** DPO is the ladder's strongest baseline because it has both the reference anchor
and a self-pacing growth term — but its logit cares only about the chosen-minus-rejected *gap*, blind to
whether that gap was won by raising the chosen chain or lowering the rejected one. On near-identical math
pairs the optimizer can win the margin by collapsing the rejected chain, which drags the near-duplicate
correct chain down — so the correct chain's *absolute* likelihood (what greedy accuracy depends on) can fall
even while DPO is "winning."

**Key idea (DPOP, DPO-Positive).** Add a one-sided barrier inside the DPO logit that forbids the correct
chain's reference-relative log-ratio `ρ(y_w) = log π_θ(y_w) − log π_ref(y_w)` from going negative. Penalize
`max(0, −ρ(y_w)) = max(0, log[π_ref(y_w)/π_θ(y_w)])`, scaled by λ, subtracted inside the β-scaled logit:

```
L_DPOP = -E[ log σ( β·( [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))]
                        - λ·max(0, log(π_ref(y_w)/π_θ(y_w))) ) ) ]
```

**Why it works here.** The barrier is exactly zero when `ρ(y_w) ≥ 0` (correct chain at least as likely as
the reference) — so DPOP *is* DPO wherever the chain is already anchored, losing nothing that made DPO the
top baseline — and active only when `ρ(y_w) < 0`, where it pushes `+∇log π_θ(y_w)` to restore the correct
chain above the reference floor. One-sided keeps all of DPO's upside growth (which cracked AIME) and clips
only the downside slip (which nicked MATH-500). λ = 0 recovers DPO exactly, so it cannot regress below the
strongest baseline by construction.

**Substrate wiring.** `custom` is *not* in the `["ipo","orpo","simpo"]` set, so the loss receives **summed**
sequence log-probs (correct: `ρ` lives in log-likelihoods). `custom` is *not* in the reference-free set in
`finetuning_args.py`, so `use_ref_model` stays True for `pref_loss=custom`, the frozen reference is loaded,
and all four log-probs reach `compute_preference_loss` — `reference_chosen_logps` supplies the floor. The
`finetuning_args.py` line is left untouched; only an `elif self.loss_type == "custom"` branch is added.

**What this must clear / what I would validate.** Bar: DPO's 85.9 / 74.2 / 13.33, average 57.81. Expect
GSM8K flat, MATH-500 edging above 74.2 and AIME holding/improving 13.33 (gains concentrated on the
near-identical pairs DPO let slip). Validation diagnostic: log mean `ρ(y_w)` — negative early under DPO (the
pathology), pinned ≥ 0 under DPOP. Smallest λ that pins `ρ(y_w) ≥ 0` is the target; too-large λ degrades to
SFT-on-chosen.

**Scaffold edit / hyperparameters.** `pref_loss=custom`, `pref_beta=0.1`, `lr=5e-7`, cosine + 10% warmup, 4
epochs; new knob `lambda_dpop` (`λ`), start ≈ 5 and sweep.

```python
def custom_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: "torch.Tensor",
    reference_rejected_logps: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""DPO-Positive (DPOP): DPO plus a one-sided barrier that forbids the chosen chain's
    reference-relative log-ratio from going below zero."""
    lambda_dpop = getattr(self.finetuning_args, "lambda_dpop", 5.0)
    pi_logratios = policy_chosen_logps - policy_rejected_logps        # log pi(y_w) - log pi(y_l)  (SUMMED)
    ref_logratios = reference_chosen_logps - reference_rejected_logps # same for pi_ref
    dpo_logits = pi_logratios - ref_logratios                        # (r_hat_w - r_hat_l)/beta
    # rho(y_w) = log pi(y_w) - log pi_ref(y_w);  penalty = relu(-rho(y_w)) = relu(log[pi_ref(y_w)/pi(y_w)])
    chosen_logratio = policy_chosen_logps - reference_chosen_logps
    penalty = torch.relu(-chosen_logratio)                           # 0 when chosen >= reference; else > 0
    logits = dpo_logits - lambda_dpop * penalty                      # subtract the barrier inside the logit
    losses = -F.logsigmoid(self.beta * logits)
    chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).to(
        self.accelerator.device).detach()
    rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).to(
        self.accelerator.device).detach()
    return losses, chosen_rewards, rejected_rewards


def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""Compute loss for preference learning."""
    if not self.finetuning_args.use_ref_model:
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    elif self.loss_type == "custom":                                 # DPOP: DPO + one-sided chosen barrier
        losses, chosen_rewards, rejected_rewards = self.custom_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
    else:
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
    return losses, chosen_rewards, rejected_rewards
```
