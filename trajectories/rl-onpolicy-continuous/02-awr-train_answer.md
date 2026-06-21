The adaptive-KL rung learned everywhere, exactly as I expected, but it left precisely the weakness I was worried about, and the numbers say where. On HalfCheetah it landed a mean of 1676.6 but with a huge spread — 1194.9 / 2695.0 / 1139.9 across the three seeds, a $2.4\times$ gap between best and worst — so the dense-reward environment is *learnable* under the penalty servo but wildly inconsistent: one seed got lucky with its KL overshoots and ran to 2695, two paid for the overshoots and stalled near 1150. Swimmer came in at 101.4 (81.6 / 113.0 / 109.5), again with seed 42 far below the others — the same one-seed-drags-it-down signature. The diagnosis is sharp: the soft KL penalty does not *forbid* a bad move, it *prices* one, and on a noisy minibatch the optimizer will sometimes pay the price and take a step it should not, which is why seed variance is the dominant failure rather than a flat ceiling. The whole penalty-vs-clip family is fighting the same enemy — the importance ratio $r_t$, the thing that fans out and that the KL penalty only softly reins back in — so the move I want is to stop using a ratio surrogate entirely.

I propose **AWR**, advantage-weighted regression: frame policy improvement as plain *supervised regression* onto the actions the agent took, each weighted by how good it was. Step back to what I actually want from an update — make the actions that turned out better (high advantage) more likely and the worse ones less likely, without moving so far that the data stops describing the policy. The penalty form does this through $r_t\hat A_t$ minus a KL term. AWR does it with no ratio, no clipping, no KL — just a weighted maximum-likelihood fit. The weight is an exponentiated advantage $w_t=\exp(\hat A_t/\beta)$ and the policy loss is the negative weighted log-likelihood

$$\mathcal L_\pi=-\hat{\mathbb E}_t\big[\,w_t\,\log\pi_\theta(a_t\,|\,s_t)\,\big],\qquad w_t=\exp(\hat A_t/\beta).$$

If there is no ratio in the loss, there is no ratio to fan out, and the reactive KL servo I was unhappy with disappears entirely.

What makes the exponential weight the *right* shape, and not just a convenient knob, is that it is the closed-form solution of the same trust region the penalty rung was approximating — solved exactly instead of penalized. Frame the update as: find the new policy that maximizes expected advantage subject to staying KL-close to the data-generating policy. The solution to "maximize $\mathbb E_\pi[A]$ subject to $\mathrm{KL}[\pi\,\|\,\pi_{old}]\le\epsilon$" is the exponentially-tilted policy $\pi^*(a|s)\propto\pi_{old}(a|s)\exp(A(s,a)/\beta)$, where $\beta$ is the Lagrange multiplier for the KL constraint. I cannot represent that tilted distribution directly, but I can *project* it onto my Gaussian policy class by minimizing $\mathrm{KL}[\pi^*\|\pi_\theta]$ — and that projection is exactly weighted maximum likelihood, $\min_\theta-\mathbb E_{s,a\sim\pi_{old}}[\exp(A/\beta)\log\pi_\theta(a|s)]$. So the exponential advantage weight is the trust region I wanted at rung 1, achieved by *construction* — the weights bake "stay close" in — rather than by a soft penalty the optimizer can trade away. That is the conceptual reason to expect it steadier on the dense-reward environment where the penalty form swung between 1150 and 2695.

I have to be careful here, because the canonical AWR is *off-policy* and importing that story would be the wrong method. The original algorithm keeps a large replay buffer, recomputes TD($\lambda$) returns over stored paths for advantages, fits a separate critic and actor with their own momentum optimizers and step counts, and uses a temperature near $1.0$ — the whole point there is reusing *old* data across many iterations, which is exactly where a ratio surrogate degrades and a weighted regression onto whatever is in the buffer does not. None of that machinery exists in this scaffold. The loop here is strictly on-policy: it collects 2048 fresh transitions, computes *GAE* advantages (not buffer TD($\lambda$)) in the frozen reverse scan, hands me one shared 2×64 actor-critic, and asks me to fill one `compute_losses` that Adam steps on for ten epochs over that one fresh batch. So the faithful realization is *on-policy AWR*: the advantage-weighted-regression objective dropped into the PPO loop's plumbing, keeping the idea — supervised weighted regression instead of a ratio surrogate — and discarding the off-policy apparatus the idea was born with.

That harness shape forces three concrete choices, each a real departure from the canonical recipe. First, the temperature. The canonical $\beta\approx1.0$ is calibrated for raw, separately-normalized advantages over a buffer. But the loop here has already normalized `mb_advantages` to roughly unit scale per minibatch (`norm_adv=True`), so unit-scale advantages divided by $\beta=1.0$ give weights $\exp(\pm1)$ that barely separate good actions from bad — the regression would be nearly uniform and learn almost nothing. To get real selectivity I need a *small* temperature, $\beta=0.05$, which turns a $+1\sigma$ advantage into a weight $\approx e^{20}$ before clipping and a $-1\sigma$ advantage into $\approx e^{-20}\approx0$ — it sharply concentrates the regression on the better-than-average actions. This is the single most important deviation, and it is *because* the loop pre-normalizes the advantages: the temperature has to be read against the scale of the input it actually receives, not the scale the original method assumed.

Second, the weight clip and stabilization. With $\beta=0.05$ the exponential weights have an enormous dynamic range, and a single outlier advantage would produce a weight that dominates the entire minibatch gradient — the AWR analogue of the ratio blowing up. So I clamp the weights at `_awr_max_weight = 20.0` (the canonical clip value, which carries over cleanly), then do something the off-policy version does not need: I *self-normalize* the clipped weights to mean one across the minibatch, `weights = weights / (weights.sum() + 1e-8) * weights.numel()`. The reason is specific to this loop. The off-policy version runs the actor for a fixed number of gradient steps with a momentum optimizer at its own step size, so the absolute weight scale just rolls into that step size. Here the regression shares one Adam and the global gradient-norm clip with the value loss, and Adam's update is sensitive to the *scale* of the gradient relative to its running second moment — a minibatch whose weights are mostly tiny (every action below average) would produce a vanishing policy gradient and waste the step, while one with a single huge surviving weight would saturate the gradient-norm clip. Renormalizing to mean one keeps the effective regression step size constant from minibatch to minibatch, the same robustness the loop's input-side advantage normalization buys, and it is machinery the buffer-based version simply does not have.

Third, what carries the gradient. The advantage weights are a *target*, not a path: the regression should push $\log\pi_\theta$ up on high-weight actions, but the weights themselves must not receive gradient, or the optimizer could cheat by reshaping the advantage estimate. So the entire weight computation — exp, clamp, renormalization — sits under `torch.no_grad()`, and only `newlogprob` carries gradient into the policy loss $-\hat{\mathbb E}[w\,\log\pi_\theta]$. The value head gets the same plain MSE toward the GAE returns that rung 1 used — I am still introducing no clipping discipline, and the regression framing does not change the critic's job. What survives, and what makes this AWR, is the core: replace the importance-ratio surrogate with an exponentiated-advantage-weighted supervised regression onto the taken actions.

My central bet is *reliability through construction*: because the trust region is baked into the regression weights rather than enforced by a soft, reactive penalty, I expect AWR to *win* where the advantage signal is clean and the better-than-average actions are well separated — HalfCheetah most of all, where I expect to clear the penalty rung's 1676.6, and InvertedDoublePendulum, where the large achievable return rewards decisive exploitation of high-advantage actions, so I expect to beat 6877.1 there too. But the sharp temperature is a genuine risk on the *low-signal* environment. Swimmer has long-horizon credit assignment and noisy advantages; a temperature that concentrates so aggressively on the top-weighted actions will, on noisy advantages, concentrate on *noise* — it confidently regresses toward whichever actions happened to draw a high (possibly spurious) advantage, with no ratio and no KL to pull it back. So my falsifiable prediction is asymmetric: AWR beats the penalty rung on HalfCheetah and InvertedDoublePendulum but is *worse* on Swimmer than 101.4, possibly the worst Swimmer of any rung. Because the task aggregates by geometric mean, a Swimmer collapse would cap AWR even if it wins the other two outright — which would be the exact signature that the next move is not "abandon the ratio" but "keep a ratio surrogate and make its trust region *hard and built into the loss*."

```python
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    """AWR: advantage-weighted regression loss."""
    _awr_beta = 0.05
    _awr_max_weight = 20.0

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Compute advantage weights: exp(advantage / beta), clamped for stability
    with torch.no_grad():
        weights = torch.exp(mb_advantages / _awr_beta)
        weights = torch.clamp(weights, max=_awr_max_weight)
        weights = weights / (weights.sum() + 1e-8) * weights.numel()

    # Policy loss — advantage-weighted regression (supervised)
    pg_loss = -(newlogprob * weights).mean()

    # Value loss — simple MSE
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
