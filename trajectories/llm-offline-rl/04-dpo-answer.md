**Problem (from step 3).** The ladder revealed two ingredients that are both necessary and never appeared
together: a reference *anchor* (IPO had it, ORPO dropping it regressed MATH-500) and a self-pacing *growth*
term on the correct chain (ORPO had it, IPO lacked it). I want both in one loss.

**Key idea.** DPO is exactly that synthesis. The KL-constrained reward optimum is the exponential tilt
`π* ∝ π_ref exp(r/β)`, whose partition function `Z(x)` is intractable — but solving the relation for the
reward gives `r(x,y) = β log(π*/π_ref) + β log Z(x)`, and the x-only `β log Z(x)` term *cancels* in the
Bradley-Terry difference (preferences see only reward differences). So the implicit reward is
`r̂(x,y) = β log(π_θ(y|x)/π_ref(y|x))`, and the policy is fit directly by the preference MLE:

```
L_DPO = -E[ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

**Why it works here.** The reference is in every term (the **anchor** that stops the chains it is not
actively pushing from eroding — ORPO's missing piece), and the gradient
`∇L_DPO = −β·E[σ(r̂_l − r̂_w)·(∇log π(y_w) − ∇log π(y_l))]` raises the correct chain and lowers the wrong
one with a **self-pacing** weight `σ(r̂_l − r̂_w)` that is large only on wrongly-ordered pairs and vanishes
once a pair is correctly separated — the growth IPO lacked, with a brake the naive unlikelihood objective
lacked (which is what kept SimPO-style collapse at bay).

**Substrate wiring.** `sigmoid` is *not* in the `["ipo","orpo","simpo"]` set, so `concatenated_forward`
hands **summed** sequence log-probs (correct: the BT derivation and the `β log Z` cancellation live in
log-likelihoods, not per-token averages). `sigmoid` is *not* reference-free, so `use_ref_model` is True, the
frozen reference is loaded, and the loss routes through `self.dpo_loss` with all four log-probs.

**Scaffold edit / hyperparameters.** `pref_loss=sigmoid`, `pref_beta=0.1`, `dpo_label_smoothing=0`,
`lr=5e-7`, cosine + 10% warmup, 4 epochs. No edit beyond the default dispatch; the sigmoid branch lives in
the harness `dpo_loss`.

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
    else:                                                  # sigmoid (DPO): reference-based branch
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )
    return losses, chosen_rewards, rejected_rewards


# Inside the harness dpo_loss (TRL), the loss_type == "sigmoid" (DPO) branch the dispatch routes to:
#   pi_logratios  = policy_chosen_logps    - policy_rejected_logps          # log pi(y_w) - log pi(y_l)  (SUMMED)
#   ref_logratios = reference_chosen_logps - reference_rejected_logps       # same for pi_ref
#   logits = pi_logratios - ref_logratios                                  # (r_hat_w - r_hat_l)/beta; beta log Z cancelled
#   losses = -F.logsigmoid(self.beta * logits)                             # label_smoothing == 0 here
#   chosen_rewards   = self.beta * (policy_chosen_logps   - reference_chosen_logps).detach()
#   rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
```
