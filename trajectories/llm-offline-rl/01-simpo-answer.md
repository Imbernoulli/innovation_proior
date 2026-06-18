**Problem.** Tune the math-SFT policy from offline preference pairs with the lightest possible loss — no
reward model, no frozen reference — and have the training reward *be* the quantity the model is graded by
at generation (per-token average log-likelihood), without rewarding verbosity.

**Key idea.** Two components. (1) A length-normalized, reference-free reward: the policy's average
per-token log-probability `r_SimPO(x,y) = (β/|y|) log π_θ(y|x)`, which is the generation-ranking metric
and removes the structural length penalty of summed log-probs. (2) A target reward margin `γ > 0`, an
additive bias in the Bradley-Terry sigmoid so the winner must outscore the loser by at least `γ`, not by
an infinitesimal amount.

**Why it works here.** Reference-free halves the memory/compute of the DPO stage; length normalization
stops long correct chains from carrying a handicap; the margin gives a generalization cushion. The
substrate does the length normalization for me — `simpo` is in the `["ipo","orpo","simpo"]` set, so
`concatenated_forward` hands `compute_preference_loss` the *average* per-token log-probs, and
`use_ref_model` is False, so my loss lands in the reference-free branch.

**Why it is the weakest rung (what to watch).** Purely relative and reference-free, it has no anchor on
the *absolute* likelihood of the correct chain. On math, where chosen and rejected solutions are
near-identical, widening the margin by pushing the rejected down drags the near-duplicate chosen down too,
so the correct chain's likelihood — the thing greedy accuracy depends on — can erode. Expect stable
training and healthy reward margins, GSM8K flat (near-saturated), and the erosion to surface on
MATH-500/AIME.

**Scaffold edit / hyperparameters.** `pref_loss=simpo`, `pref_beta=2.0`, `simpo_gamma=1.0` (so
`γ/β = 0.5`), `lr=5e-7`, cosine + 10% warmup, 4 epochs. The loss is the harness `simpo_loss` helper,
dispatched in the reference-free branch of `compute_preference_loss`.

```python
def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
    r"""Compute SimPO loss for batched (length-averaged) log probabilities of the policy model."""
    pi_logratios = chosen_logps - rejected_logps        # avg_w - avg_l (already length-normalized)
    gamma_logratios = self.simpo_gamma / self.beta      # full margin gamma -> code-space ratio
    logits = pi_logratios - gamma_logratios
    simpo_loss = -F.logsigmoid(self.beta * logits)      # -log sigma( beta*(avg_w - avg_l) - gamma )
    return simpo_loss


def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""Compute loss for preference learning."""
    if not self.finetuning_args.use_ref_model:                       # simpo: reference-free branch
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
