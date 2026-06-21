DPO topped the ladder, and the way it topped it is precisely the opening for the next move. Seed 42: GSM8K 85.9, MATH-500 74.2, AIME 13.33, average 57.81 — the best result, and exactly the synthesis I predicted: it repaired MATH-500 to ~74 where ORPO had regressed to 72.4 (the reference anchor doing its job) while holding AIME at ORPO's strong 13.33 (the self-pacing growth still cracking the long correct chains). So having both ingredients together beats every rung that had only one. But re-read the risk I flagged at the close of the DPO derivation, because the number is consistent with it. DPO's implicit reward $\beta\log(\pi_\theta/\pi_{\text{ref}})$ is a function of *summed* log-probs, and the loss is satisfied by making the *difference* $\log(\pi_\theta(y_w)/\pi_\theta(y_l))$ exceed the reference's difference. On near-identical math pairs that condition can be met two ways — by raising the correct chain's likelihood, or by *lowering the rejected chain's* — and lowering the rejected chain, which shares almost every token with the correct one, drags the correct chain down too. DPO's $\sigma$ self-pacing brake stops the push *once the pair is ordered*, milder than SimPO's unbounded version, which is why DPO does not collapse; but "stop once ordered" is not "never let the correct chain's absolute probability fall." The DPO logit cares only that the winner-minus-loser *gap* clears the reference's; it is blind to whether that gap was won by pushing the chosen up or the rejected down. On a benchmark scored by greedy correctness — which depends on the *absolute* likelihood of the correct chain, not the gap — that blindness is the residual failure mode sitting underneath 57.81, and it is the same disease, viewed through DPO's reference-anchored lens, that ate SimPO's AIME and nicked ORPO's MATH-500. So the natural move past the strongest baseline is not a new family: take DPO, which already has the right anchor and the right self-pacing, and add a term that *forbids the correct chain's likelihood from falling below the reference's*. Keep everything that made DPO the top; close the one hole the leaderboard's shape implies is still open.

I propose **DPOP (DPO-Positive)**, and I derive the term from the failure rather than bolt it on. Define the per-response log-ratio $\rho(y) = \log\pi_\theta(y|x) - \log\pi_{\text{ref}}(y|x)$ — how much more (or less) likely the policy makes a response than the reference does — so the DPO logit is $\beta(\rho(y_w) - \rho(y_l))$ and the loss is happy whenever this is large and positive. The pathology is $\rho(y_w) < 0$: the policy has made the *correct* chain *less* likely than the reference did, while $\rho(y_w) - \rho(y_l)$ stays positive only because $\rho(y_l)$ fell even further. The fix must make the loss *unhappy* when $\rho(y_w) < 0$ regardless of the gap, and the cleanest such term is a one-sided penalty on the correct chain's reference-relative log-ratio, $\max(0, -\rho(y_w)) = \max(0, \log\pi_{\text{ref}}(y_w) - \log\pi_\theta(y_w))$: exactly zero when the policy keeps the correct chain at least as likely as the reference ($\rho(y_w) \ge 0$, nothing to fix), growing linearly as it falls below. One-sided is the whole point — I do not want to *force* the correct chain above the reference everywhere (that is just SFT and would fight the contrast), only to *forbid* it from dropping below: a floor, not a target. The penalty goes inside the DPO logit, subtracted and scaled by the same $\beta$ so it lives in the reward-gap units and the $\sigma$ self-pacing applies to it too:

$$L_{\text{DPOP}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\Big(\underbrace{\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}}_{\rho(y_w)-\rho(y_l),\ \text{the DPO logit}} - \lambda\,\underbrace{\max\!\big(0,\ \log\tfrac{\pi_{\text{ref}}(y_w)}{\pi_\theta(y_w)}\big)}_{\max(0,\,-\rho(y_w))}\Big)\right)\right].$$

Reading the gradient: when $\rho(y_w) \ge 0$ the $\max(0,\cdot)$ is zero and so is its gradient — DPOP is *exactly DPO*, so on every pair where the correct chain is already at least reference-likely I keep DPO's behavior untouched and lose nothing that made it the top baseline. When $\rho(y_w) < 0$ the penalty $-\lambda(-\rho(y_w)) = \lambda\rho(y_w)$ is active and negative, which shrinks the logit, raises the loss, and through the same $\sigma$ weight pours gradient into $+\nabla\log\pi_\theta(y_w)$ — actively pushing the correct chain back up above the reference. So the penalty is a *barrier*: invisible while the correct chain stays anchored, sharply restoring once it slips — precisely the "never let the correct chain's absolute probability fall" guarantee the DPO logit was blind to, expressed in the reference-relative coordinate DPO already uses so it composes with the anchor and the self-pacing rather than fighting them. Two checks make me trust it. Setting $\lambda = 0$ vanishes the penalty identically, so DPOP is a strict generalization of DPO with one extra non-negative knob — it can never do worse than DPO on pairs that do not trigger the barrier, and the ladder's strongest result is recoverable by construction. And one-sided rather than a symmetric pull toward $\rho(y_w) = 0$ is deliberate: a symmetric term would penalize $\rho(y_w) > 0$ too, punishing the model for making the correct chain *more* likely than the reference — exactly the active growth that cracked AIME. The $\max(0,\cdot)$ keeps all of that upside and clips only the downside; that asymmetry is the difference between anchoring to a floor and regressing to a fixed value, and I want the floor.

The finale's edit is a real fill of `compute_preference_loss` — the `custom` slot the task reserves. `custom` is *not* in the `["ipo","orpo","simpo"]` set, so the loss receives **summed** sequence log-probs, correct because $\rho(y) = \log\pi_\theta(y) - \log\pi_{\text{ref}}(y)$ lives in log-likelihoods, the same coordinate in which the $\beta\log Z$ cancellation holds. And `custom` is *not* in the reference-free set in `finetuning_args.py` (`use_ref_model = stage=="dpo" and pref_loss not in ["orpo","simpo"]`), so for `pref_loss=custom`, `use_ref_model` stays True, the frozen reference is loaded, and all four log-probs reach `compute_preference_loss` — `reference_chosen_logps` supplies the floor. I leave the `finetuning_args.py` line untouched and add an `elif self.loss_type == "custom"` branch on the reference-based side that forms the standard DPO log-ratios `pi_logratios − ref_logratios`, the chosen-side $\rho(y_w) = \text{policy\_chosen\_logps} - \text{reference\_chosen\_logps}$, the one-sided penalty $\text{relu}(-\rho(y_w))$, subtracts $\lambda\cdot\text{penalty}$ inside the logits, and returns $-\log\sigma(\beta\cdot\text{logits})$ with the same detached implicit rewards DPO logs. $\lambda$ is a new scalar; large-model runs used $\lambda$ around 50 with a larger $\beta$, but those are 34B/72B numbers — for this $\beta = 0.1$, 1.5B math setting I start $\lambda$ small (on the order of a few, e.g. $\lambda = 5$) so the barrier corrects the slip without overwhelming the contrast, and treat it as the one knob to sweep.

The bar is DPO's 85.9 / 74.2 / 13.33, average 57.81. My falsifiable claim is that DPOP holds DPO's gains everywhere the barrier is inactive (so GSM8K stays ~85–86 and the result can never regress below DPO by construction at $\lambda = 0$) and *adds* on the pairs where DPO was silently letting the correct chain slip — the near-identical math pairs, densest on MATH-500 and AIME — so I expect MATH-500 to edge above 74.2 and AIME to hold or improve on 13.33, lifting the average past 57.81. The cleanest validation is the diagnostic the penalty was built from: log the mean of $\rho(y_w)$ over training. Under DPO it should trend negative early (the chosen log-ratio decreasing — the pathology); under DPOP it should be pinned at or above zero, and the benchmark gain should track the fraction of pairs where DPO's $\rho(y_w)$ would have gone negative. If $\rho(y_w)$ never goes negative under DPO on this data, the barrier is inactive and DPOP collapses to DPO with no gain — that is the falsifier, and the log of $\rho(y_w)$ tells me immediately which world I am in. The risk in the other direction is $\lambda$ too large: the barrier then dominates and the loss becomes SFT-on-chosen with a vestigial contrast, pushing GSM8K and MATH-500 around unpredictably and blunting the AIME growth — so the smallest $\lambda$ that pins $\rho(y_w) \ge 0$ is the target.

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
