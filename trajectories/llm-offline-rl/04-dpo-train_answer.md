ORPO's number is the most interesting yet, because it broke the pattern in a way that tells me what actually matters. Seed 42: GSM8K 85.37 (flat), MATH-500 72.4, AIME 13.33, average 57.03 — the biggest move so far, but a *trade*, not a clean win. AIME leapt to four correct problems where IPO had two and SimPO had one — the active SFT growth on the correct long chains paying off — but MATH-500 *fell*, 74.4 → 72.4, the first time a benchmark went backward on the ladder. Reference-free active growth cracks the hardest chains and is slightly corrosive on the broad middle, because nothing anchors the chains it is *not* actively growing. So the diagnosis sharpens into a clean synthesis problem. SimPO: reference-free, no anchor, unbounded — collapsed AIME. IPO: reference anchor, finite target — defended everything but grew nothing. ORPO: reference-free active growth — cracked AIME but regressed MATH-500. The two ingredients I have been toggling are both necessary and I have never had them together: an **anchor against a reference** (so the chains I am not actively pushing do not erode — ORPO's missing piece) *and* a **self-pacing growth term** on the correct chain (IPO's missing piece). The cleanest object with both is the one I have been circling the whole ladder without running — DPO itself, the reference-based Bradley-Terry MLE every other rung was a reaction to.

I propose **DPO**, and I derive it so I understand exactly why it should top the ladder. The objective is $\max_\pi \mathbb{E}_{y\sim\pi}[r(x,y)] - \beta\,\text{KL}(\pi\|\pi_{\text{ref}})$ with the reward fit from preferences by Bradley-Terry, but the painful stage is maximizing $\mathbb{E}_\pi[r]$ over a discrete autoregressive LM, which needs RL — a reward model, a critic, on-policy sampling. I collapse the two stages into one supervised loss. Writing the KL out, pulling $-\beta$, and flipping max to min gives $\min_\pi \mathbb{E}_{y\sim\pi}[\log(\pi/\pi_{\text{ref}}) - (1/\beta)r]$; manufacturing $\pi^*(y|x) = (1/Z(x))\,\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$ so that $-(1/\beta)r = -\log(Z\pi^*/\pi_{\text{ref}})$ turns the objective into $\min_\pi \mathbb{E}_x[\text{KL}(\pi\|\pi^*) - \log Z(x)]$, so the optimum is the exponential tilt

$$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\!\left(\frac{r(x,y)}{\beta}\right),\qquad Z(x) = \sum_{y'}\pi_{\text{ref}}(y'|x)\exp(r(x,y')/\beta).$$

This is famous and famously unusable directly, because $Z(x)$ sums over all sequences — the intractable wall the reward-weighted-regression rung hit. The move is to read the relation right-to-left. It is one equation relating $r, \pi^*, \pi_{\text{ref}}, Z$; solve it for $r$: $r(x,y) = \beta\log(\pi^*(y|x)/\pi_{\text{ref}}(y|x)) + \beta\log Z(x)$. Any reward is $\beta$ times the log-ratio of its own optimal policy to the reference plus an $x$-only term, so I parameterize the *policy* directly and read off its implicit reward $\hat r(x,y) = \beta\log(\pi_\theta/\pi_{\text{ref}}) + \beta\log Z(x)$. The reward enters the data only through Bradley-Terry, which depends on the *difference* $r(y_w) - r(y_l)$ — and $\beta\log Z(x)$ is a function of $x$ only, identical for $y_w$ and $y_l$, so it **cancels** in the difference. The thing that made the closed form unusable evaporates the moment I express preferences through it, because preferences see only reward *differences* and $Z$ was a reward *offset*. So the implicit reward is just $\hat r(x,y) = \beta\log(\pi_\theta(y|x)/\pi_{\text{ref}}(y|x))$ — the policy is secretly a reward model — and fitting it by the preference NLL gives one stage, a supervised classification loss, no reward model, no critic, no sampling:

$$L_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right].$$

This has *both* ingredients the ladder told me I needed. The reference is in every term — $\beta\log(\pi_\theta/\pi_{\text{ref}})$ is measured against $\pi_{\text{ref}}$, the **anchor** ORPO's MATH-500 regression showed was missing. And the gradient is the **self-pacing growth** IPO lacked: with $s = \hat r(y_w) - \hat r(y_l)$, $\nabla L_{\text{DPO}} = -\beta\,\mathbb{E}[\sigma(\hat r(y_l) - \hat r(y_w))\,(\nabla\log\pi_\theta(y_w) - \nabla\log\pi_\theta(y_l))]$. The bracket *raises* the correct chain's log-probability and *lowers* the wrong one — active growth on $y_w$ — and the scalar weight $\sigma(\hat r(y_l) - \hat r(y_w))$ is large precisely when the implicit reward orders the pair *wrong* and $\to 0$ once it is correctly ordered with margin, pouring gradient into the examples the model gets wrong and stopping on the ones it has right. That weight is the crux separating DPO from the naive "just raise $\log p(y_w)$, lower $\log p(y_l)$" unweighted unlikelihood objective, which has no brake on the minimization of $y_l$ and degenerates; the $\sigma$ weight, scaled by $\beta$ and anchored to $\pi_{\text{ref}}$, pushes only until the pair is ordered, and "lowering $p(y_l)$" is measured relative to the reference rather than absolutely — which is what prevents the collapse that ate SimPO. So DPO is exactly the synthesis, IPO's reference anchor plus ORPO's self-pacing growth, in one loss with a finite implicit-reward margin the saturating $\sigma$ stops paying out on once met. The only cost is the one ORPO escaped — the frozen reference, four forwards per step — and on this ladder that cost buys precisely what ORPO's MATH-500 regression said was worth buying.

The substrate wiring differs from the reference-free rungs in one place that the derivation demands. The `sigmoid` loss is *not* in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` does **not** length-average — DPO sees the **summed** sequence log-probs, which is correct: the Bradley-Terry derivation is written in log-likelihoods and the $\beta\log Z$ cancellation is a property of the summed log-prob, not a per-token average. And `sigmoid` is *not* reference-free, so `use_ref_model` is True, the frozen reference is loaded, and the loss routes through `self.dpo_loss`, which receives all four log-probs and forms `logits = (chosen − rejected) − (ref_chosen − ref_rejected)` — that is $(\hat r(y_w) - \hat r(y_l))/\beta$ — returning $-\log\sigma(\beta\cdot\text{logits})$ with `label_smoothing = 0` here. My edit is the default dispatch into `dpo_loss` with `loss_type="sigmoid"`, $\beta = 0.1$.

DPO restores ORPO's dropped anchor while keeping a self-pacing growth term, so my prediction is specific: it *repairs MATH-500* back toward the ~74 SimPO and IPO held (the anchor stops the middle-benchmark erosion) while *holding AIME* near ORPO's strong 13.33 (the self-pacing growth still cracks the long correct chains), GSM8K saturated ~85–86. If both hold, DPO clears ORPO's 57.03 — roughly +1.5 on MATH-500 with AIME flat nets an average around 57.8, the best on the ladder, which would explain why plain DPO is the strongest baseline despite being the one the others reacted to. The residual risk: DPO's summed-log-prob reward is *not* length-normalized, so on long math chains the un-normalized gradient lets long responses dominate a batch — the very length sensitivity SimPO removed. If that bites, the signature is a length-correlated error that the finale must fix by re-normalizing or by directly protecting the long correct chain's likelihood.

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
