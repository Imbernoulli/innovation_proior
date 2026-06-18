**Problem (from step 1).** SimPO's reference-free, purely relative objective has no anchor on the
*absolute* likelihood of the correct chain, and its saturating sigmoid keeps paying out as the margin
grows — so on near-identical math pairs it drags the correct chain down to win a margin, and AIME (longest,
most fragile correct chains) collapsed.

**Key idea.** Two fixes, both from one move: bring back the reference *and* replace the saturating sigmoid
with a finite regression target. The Bradley-Terry logit objective is unbounded (a deterministic
preference sends the implied reward gap to `+∞`, so β stops regularizing for any β). Choosing the bounded
identity transform `Ψ(q) = q` and folding the analytic optimum into a squared residual collapses, after
partner-averaging the label and completing the square, to a single regression of the reference-corrected
winner-over-loser log-ratio gap onto one fixed target:

```
L_IPO(π) = E_D[ ( h_π(y_w, y_l) - 1/(2β) )^2 ],
h_π(y_w, y_l) = [log π(y_w) - log π_ref(y_w)] - [log π(y_l) - log π_ref(y_l)]
```

**Why it works here.** The reference term penalizes letting the correct chain fall below `π_ref`; the
finite target `1/(2β)` makes the loss zero (gradient vanishes) once the gap is met, so there is no
unbounded push to drag the chosen down. The knob β now genuinely controls distance to `π_ref` even on
deterministic preferences (`π* = σ(±1/(2β))` on the minimal instance), unlike the logit objective which
sat at `π(loser)=0` for all β.

**Substrate wiring.** `ipo` is in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` hands the
loss **length-averaged** per-token log-probs (the sequence-length normalization `h_π` needs). `ipo` is
*not* in the reference-free set, so `use_ref_model` is True, the frozen reference is loaded, and the loss
routes through `self.dpo_loss`, which receives all four log-probs and forms `h_π` as the IPO branch.

**Scaffold edit / hyperparameters.** `pref_loss=ipo`, `pref_beta=0.1`, `lr=5e-7`, cosine + 10% warmup, 4
epochs. No edit to `compute_preference_loss` beyond the default dispatch; the IPO branch lives inside the
harness `dpo_loss`.

```python
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
    else:                                                  # ipo: reference-based branch
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
    return losses, chosen_rewards, rejected_rewards


# Inside the harness dpo_loss (TRL), the loss_type == "ipo" branch the dispatch routes to:
#   pi_logratios  = policy_chosen_logps    - policy_rejected_logps          # avg log pi(y_w) - avg log pi(y_l)
#   ref_logratios = reference_chosen_logps - reference_rejected_logps       # same for pi_ref
#   logits = pi_logratios - ref_logratios                                  # h_pi(y_w, y_l)
#   losses = (logits - 1.0 / (2.0 * self.beta)) ** 2                       # regress onto 1/(2*beta)
#   chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps).detach()
#   rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```
