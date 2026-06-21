I have a math-specialized SFT policy (`Qwen2.5-Math-1.5B-Instruct`) and a pile of offline preference pairs over math solutions, and the one thing I get to design is the loss that turns a `(prompt, chosen, rejected)` triple into a gradient on the policy — no reward model, no on-policy sampling. The honest place to start a ladder is the lightest loss that could conceivably work, run it, and let its number tell me what the math-reasoning setting actually punishes. The obvious baseline, DPO, bothers me on two counts before I have even run it. The cost: it keeps a frozen copy of the SFT model resident and runs a second forward pass every batch to compute reference log-probs `log \pi_{\text{ref}}(y|x)`, a whole extra model's footprint and a doubled forward on a memory-tight 4×GPU box, purely to subtract off a baseline. And the mismatch that actually itches: at generation — which is exactly how this task is scored, greedy decoding graded by sympy — there is no reference model anywhere; the model is judged on the per-token likelihood it itself assigns to a correct chain. But DPO's reward $r(x,y) = \beta \log[\pi_\theta(y)/\pi_{\text{ref}}(y)]$ is a log-*ratio* against the reference, a different function of the same response. Satisfying $r(x,y_w) > r(x,y_l)$ rearranges to a condition on summed log-probs *offset by whatever the reference assigns*, and there is no reason that implies the thing I am graded on. The reward I optimize and the metric I am judged by are not the same object.

So I propose **SimPO**: make the training reward *be* the generation-ranking metric, and build it reference-free out of the policy alone. The metric the model is decoded and ranked by is the average per-token log-likelihood, $(1/|y|)\log\pi_\theta(y|x)$, so I make that the reward directly. The naive reference-free instinct — just use $r = \beta\log\pi_\theta(y)$ — kills the memory problem but is the *summed* log-prob, and summed log-prob has a structural length defect: every extra token contributes another $\log\pi \le 0$, so longer sequences score systematically lower. In math, where correct full derivations are often longer than truncated wrong ones, the model would then have to overcome a length handicap, and its only lever is to crank token probabilities on the long winning sequence — baking in a "long = good" verbosity artifact. The fix is already in what I wanted: the generation metric is the *average*, not the sum, and the $1/|y|$ is exactly the length normalization that cancels the handicap, putting winner and loser on the same per-token footing. The two problems collapse into one choice — use the average log-likelihood as the reward:

$$r_{\text{SimPO}}(x,y) = \frac{\beta}{|y|}\log\pi_\theta(y|x),$$

reference-free, generation-aligned, and length-debiased at once. I drop $\pi_{\text{ref}}$ entirely; the KL leash it provided is replaced by a practical leash — a strong math-SFT start, a tiny learning rate ($5\times10^{-7}$), four passes over 10K diverse problems — not a theorem, but enough reason to try the cheaper policy-only reward.

Plugging $r_{\text{SimPO}}$ into Bradley-Terry the way DPO plugs in its reward gives $L = -\mathbb{E}[\log\sigma(r_{\text{SimPO}}(x,y_w) - r_{\text{SimPO}}(x,y_l))]$, and the gradient confirms it: with $u$ the reward difference, the per-example weight is $\sigma(-u)$, large exactly when the policy *wrongly* scores the loser above the winner, and each log-prob gradient is divided by its own length so a long and a short response push with comparable magnitude — unlike DPO's un-normalized $\nabla\log\pi(y_w) - \nabla\log\pi(y_l)$, where twice the tokens means roughly twice the gradient. But the bare Bradley-Terry skeleton only asks $r(y_w) > r(y_l)$, satisfied the instant the winner outscores the loser by an infinitesimal amount; getting the sign right is a weak requirement, and the lesson of margins (max-margin SVMs, the home-advantage offset in Bradley-Terry ranking models) is that a comfortable gap generalizes better than a barely-separating one. So I put a **target margin** $\gamma > 0$ into the preference model itself, $p(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l) - \gamma)$, shifting the sigmoid so the loss is not near-minimal until the reward gap has crossed $\gamma$ and the model keeps pulling the winner up until there is a real cushion:

$$L_{\text{SimPO}} = -\mathbb{E}\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi(y_w) - \frac{\beta}{|y_l|}\log\pi(y_l) - \gamma\right)\right].$$

Too-small $\gamma$ is back to asking only for the right sign; too-large $\gamma$ demands an unrealistic per-token gap and over-suppresses fluent losing responses, so $\gamma$ is a knob to tune, not derive — for this 1.5B math base I use $\beta = 2.0$, $\gamma = 1.0$.

The frozen loop does the rest for me. Because `pref_loss=simpo` puts `simpo` in the `["ipo","orpo","simpo"]` set, `concatenated_forward` divides each per-response log-prob by `valid_length` before my loss sees it — so the `policy_chosen_logps`/`policy_rejected_logps` I receive are already the *average* per-token log-probs, the length normalization done. And because `simpo` is in the reference-free set in `finetuning_args.py`, `use_ref_model` is False, no reference is loaded, and my loss lands in the top branch. My entire contribution is the `simpo_loss` helper: form the average-log-prob difference, subtract the code-space margin $\gamma/\beta$ (so the single $\beta$-multiply reproduces $\beta\Delta - \gamma$ exactly, since $\beta(\Delta - \gamma/\beta) = \beta\Delta - \gamma$), and return $-\log\sigma(\beta\cdot\text{logits})$; the logged implicit rewards are $\beta\cdot\text{policy\_chosen\_logps}$ and $\beta\cdot\text{policy\_rejected\_logps}$.

I will flag the one regime where I expect this to be fragile, because it is precisely this task's regime: preferences over *math* solutions whose winning and losing chains are nearly identical, branching at one wrong step. A contrastive objective there can widen the *reward margin* by pushing the loser's probability down, but because the chosen sequence shares almost every token with the rejected one, dragging the rejected down drags the chosen down too — the absolute likelihood of the *correct* answer can fall even as the margin grows. SimPO's margin term only asks for a *bigger* gap, so it will not rescue this, and SimPO has no anchor on the chosen chain's absolute likelihood at all. Greedy correctness lives on that absolute likelihood, so my falsifiable expectation for the floor is concrete: stable training, healthy reward margins, GSM8K flat (near-saturated), and the erosion biting on MATH-500/AIME — most visibly on AIME, the hardest and highest-variance benchmark. If AIME comes in low while the easy benchmarks hold, that is the signature, and it tells the next rung to stop letting the relative objective drag the correct chain down — by re-anchoring to a reference, or by trading the saturating sigmoid for a target the model can sit at.

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
