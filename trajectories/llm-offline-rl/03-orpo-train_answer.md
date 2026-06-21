IPO did roughly what I predicted, and the size of the move tells me the diagnosis was right but the remedy was timid. Seed 42: GSM8K 85.9 (flat, saturated), MATH-500 74.4 (a hair up from SimPO's 74.0), AIME 6.67 — *two* correct problems where SimPO had one. So bringing back the reference and replacing the saturating sigmoid with a finite target stopped the unbounded push that was dragging the correct chains down, and the most fragile benchmark recovered; the average climbed 54.46 → 55.66. But the AIME recovery is one extra problem out of thirty, and MATH-500 barely moved, because IPO is conservative by construction: it regresses the reference-corrected log-ratio gap onto a fixed target and goes silent the instant the gap is met. It *defends* the correct chain's likelihood against the reference; it never *grows* it. And it pays for that anchor with a frozen reference — a second resident model and four forwards per step. SimPO was reference-free and cheap but had no anchor; IPO has an anchor but rents a whole extra model to provide it. I want both: active growth on the correct chain's absolute likelihood *and* a contrast against the rejected one, with no reference.

Start from SFT, because if supervised fine-tuning on the chosen chains already grew their likelihood I would be most of the way there. The causal-LM NLL on a chosen response, $L = -(1/m)\sum_k\sum_i y_i^{(k)}\log p_i^{(k)}$ with $y_i$ one-hot, only survives at the single label token per position; for every other token — including every token that would build a *rejected* chain — the term vanishes. So cross-entropy *rewards* the correct chain's label tokens and is silent about the wrong ones: one-sided, all reward, no penalty. And I know what that one-sidedness does in this domain — raising the probability of chosen responses raises the probability of the *whole neighborhood*, and the rejected response (same problem, near-identical chain branching at one step) lives in exactly that neighborhood, so its log-prob climbs right alongside. Plain SFT gives me the active growth IPO lacked but also grows the wrong chain, the math-reasoning trap. So I need SFT's growth on the chosen chain *plus* a penalty running alongside that pushes the rejected chain down — one stage, no reference.

I propose **ORPO**. The penalty that appends to NLL is a contrast wrapped in a log-sigmoid, but the *kind* of ratio inside it is the load-bearing choice, because I am running this *during* SFT on a model still adapting. For any ratio $R$, minimizing $-\log\sigma(\log R)$ does not just want $R > 1$, it wants $R$ large into the saturating tail, and how hard it pushes depends on how spread out $\log R$ is across the data: if $\log R$ is tightly concentrated near zero, the only way to move the loss is to force each example to an *extreme* margin. With the **probability ratio** $P(y_w)/P(y_l)$, that means crushing $P(y_l)$ toward zero — slamming rejected-token logits down — and on a model still learning those tokens overlap heavily with good tokens it still needs (the rejected math chain is a near-duplicate), degenerating generation: SimPO's failure reappearing from the optimization side. I make this quantitative with $X_1, X_2 \sim \text{Unif}(0,1)$ as stand-ins for $P(y_w), P(y_l)$: $\log X_1 - \log X_2$ is symmetric and *concentrated* near 0, but the **odds ratio** uses the logit $\log(P/(1-P))$, which blows up toward $\pm\infty$ as $P\to 1$ or $0$, so $\log\text{OR} = \text{logit}(X_1) - \text{logit}(X_2)$ is far more *spread out* than $\log\text{PR}$ — the $\log(1-X)$ piece explodes near $X=1$. That flips the design: with the wide-ranging odds ratio a given target sigmoid output is reached by a *modest* per-example margin, no overshoot, so it gives a *mild* discrimination of the rejected chain — exactly the right intensity for penalizing during SFT without degenerating, and on near-identical math pairs, mild is what protects the correct chain that shares those tokens.

Pinning the objects down: for a response of length $m$ I take the length-normalized sequence log-prob $\log P_\theta(y|x) = (1/m)\sum_t\log P_\theta(y_t|x,y_{<t})$ — the geometric mean of per-token probs, so $P_\theta \in (0,1)$, the odds $P/(1-P)$ is finite, and different-length chains are comparable. The penalty wraps the log odds ratio in a negative log-sigmoid, and the full single-stage objective is

$$L_{\text{ORPO}} = \mathbb{E}[L_{\text{SFT}} + \lambda\,L_{\text{OR}}],\quad L_{\text{SFT}} = -\log P_\theta(y_w|x),\quad L_{\text{OR}} = -\log\sigma\!\left(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\right),\quad \text{odds}_\theta(y) = \frac{P_\theta(y)}{1 - P_\theta(y)}.$$

$L_{\text{SFT}}$ is the **active likelihood growth** IPO never had; $L_{\text{OR}}$ is the new contrast; and crucially there is **no $\pi_{\text{ref}}$** — the contrast is between $y_w$ and $y_l$ under the *same current* parameters, and the "don't generate the wrong chain" pressure comes from comparing each chain's probability with its own complement $1-P$. One model, one stage; the anchor on the correct chain is the SFT term, not a frozen reference — the trade I wanted off IPO. The gradient confirms it does the two things I claimed. With $u = \log(\text{odds}_w/\text{odds}_l)$, descent moves in $+\delta\cdot h$, where the example weight $\delta = [1 + \text{odds}_w/\text{odds}_l]^{-1}$ is large when the model wrongly prefers the rejected chain and $\to 0$ once the example is solved (automatic difficulty weighting, self-pacing), and the direction $h = \nabla\log P(y_w)/(1-P(y_w)) - \nabla\log P(y_l)/(1-P(y_l))$ is a contrast — a $+\nabla\log P(y_w)$ on the chosen, a $-\nabla\log P(y_l)$ on the rejected (the discrimination SFT alone could not do), each scaled by $1/(1-P)$, which grows as $P\to 1$. So a rejected chain that has become *too plausible* — the math trap — gets a sharper negative push, and nothing in $\delta$ or $h$ touches a second model: one model in memory, two forwards per batch versus IPO's four, and no SFT warm-up since $L_{\text{SFT}}$ is in the objective.

The substrate drops this straight in. The harness hands me per-response summed token log-probs and a valid length; dividing gives $c = \log P_\theta(y_w)$ and $r = \log P_\theta(y_l)$, each a mean-per-token log-prob $\le 0$ with $P = e^c$, so the log odds ratio is $(c - r) - [\log(1 - e^c) - \log(1 - e^r)]$. The term $\log(1 - e^c)$ underflows naively, so the stable primitive is $\text{log1p}(-\exp(c))$ for $c \le 0$. Because `orpo` is in the `["ipo","orpo","simpo"]` set, `concatenated_forward` hands me the **average** per-token log-probs the odds needs, and because `orpo` *is* in the reference-free set, `use_ref_model` is False, no reference is loaded, and the loss lands in the top branch dispatched to `odds_ratio_loss` — which computes $(c-r) - (\text{log1p}(-\exp(c)) - \text{log1p}(-\exp(r)))$, then $-c + \beta\cdot(-\log\sigma(\text{log\_odds}))$, with `self.beta` playing the role of $\lambda$. I set $\beta = \lambda = 0.1$.

The new ingredient relative to IPO is the active SFT term growing the correct chain paired with a *mild* odds-ratio penalty instead of IPO's defend-only target, so I predict ORPO improves where IPO left headroom — MATH-500 most plausibly, where the SFT term can push the correct chains above IPO's conservative 74.4 — with GSM8K saturated near 85–86 and AIME a wild card (it moves in 3.33-point quanta on thirty problems, as likely to tick to 13.33 as to hold; I will not over-read one problem). The risk is the mirror of SimPO's: ORPO is also reference-free, so too-large $\lambda$ stops being mild and the near-identical rejected chains drag the chosen down again — but at the default small $\lambda$ I expect the SFT term to dominate and the net to be a likelihood *gain*. If MATH-500 fails to move, the SFT term is not buying the growth I claimed, and the next rung would have to grow the correct chain's likelihood more directly, anchored against a reference so the growth cannot be undone by the contrast.

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
