AWR confirmed both halves of my prediction, and the second half is the one that matters. It won where I expected: HalfCheetah rose to a mean of 1996.7 (up from the penalty rung's 1676.6) and InvertedDoublePendulum to 7299.2 (up from 6877.1), so abandoning the importance ratio for a weighted regression did buy real gains where the advantage signal is clean. But Swimmer collapsed to 90.2, *below* the penalty rung's 101.4, and the per-seed numbers are the tell: 124.4 / 46.0 / 100.1, with seed 123 cratering to 46. That is exactly the failure I flagged — the sharp $\beta=0.05$ temperature concentrates the regression on whichever actions drew the top advantages on a minibatch, and on Swimmer's long-horizon noisy advantages those are frequently *noise*; with no ratio and no KL to pull it back, the policy confidently regresses toward spurious actions on some seeds and never recovers. And even HalfCheetah's win is a one-seed story (1358 / 1331 / 3301), so the gain is fragile. AWR is the *least balanced* update rule, and because the task scores by geometric mean, that Swimmer 90.2 caps it hard. The next move is therefore neither "abandon the ratio" (AWR tried that and bought instability on the low-signal environment) nor "soft-penalize the ratio" (the penalty rung tried that and bought seed variance on the dense one). It is: *keep* a ratio surrogate, but replace both the soft penalty and the closed-form regression with a trust region that is *hard and built directly into the loss*, non-negotiable on every environment at once.

I propose **PPO** with the clipped surrogate. Go back to what governs whether the surrogate is honest: $L(\theta)=\hat{\mathbb E}_t[r_t(\theta)\hat A_t]$ with $r_t=\pi_\theta/\pi_{old}$ is the Kakade–Langford lower bound on true improvement, and its honesty degrades with KL distance from $\pi_{old}$. The thing both prior rungs fought is the same: the optimizer is *rewarded* for pushing $r_t$ far from 1 on a positive-advantage sample, because inflating the ratio is the cheapest way to raise the objective. So instead of penalizing the distance after the fact, I *remove the reward for moving the ratio outside a band at all*. Pick a small $\epsilon=0.2$ and clip the ratio inside the objective, $\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat A_t$. With $\hat A_t>0$, once $r_t$ exceeds $1+\epsilon$ the term flattens at $(1+\epsilon)\hat A_t$ and its gradient w.r.t. $r_t$ dies — no more incentive to crank that action's probability up; with $\hat A_t<0$, once $r_t$ drops below $1-\epsilon$ it flattens at $(1-\epsilon)\hat A_t$ and again the gradient dies. A flat clip kills the incentive to leave the band — the trust region I wanted at rung 1, expressed as a *flat spot in the loss*, with no KL term, no Lagrange coefficient to servo, no closed-form projection. It constrains the *unit-free* ratio directly, which is why a single $\epsilon=0.2$ can be reliable across all three environments where the penalty rung's $\beta$ had to chase three advantage scales and AWR's over-concentrated on the noisy one.

But a plain clip is not safe, and the unsafe case is exactly the overshoot that wrecked AWR. Imagine an action that was actually bad, $\hat A_t<0$, whose ratio a noisy earlier minibatch has *already* pushed up to $r_t=3$. The *unclipped* term $r_t\hat A_t=3\hat A_t$ is very negative, so its gradient wants to pull $r_t$ back down — it corrects the overshoot. But the *clipped* term $(1+\epsilon)\hat A_t$ sits in the flat region with zero gradient, so a pure clip would *freeze in* the overshoot it should be undoing — precisely AWR's disease, commit to a move and never correct it. So I want the clip to remove the *incentive to overshoot* but keep the *gradient that corrects an overshoot*. That means keeping both the unclipped and the clipped term and taking the more pessimistic one — the one that gives the objective *less* credit, the minimum:

$$L^{CLIP}(\theta)=\hat{\mathbb E}_t\big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\big].$$

The min does exactly what I want, case by case. With $\hat A_t>0$: inside the band the two terms are equal; above $1+\epsilon$ the clipped $(1+\epsilon)\hat A_t$ is smaller so `min` picks it — capped, gradient zero, stop paying for going higher; below $1-\epsilon$ (a good action made *less* likely, wrong direction) the clipped $(1-\epsilon)\hat A_t$ is *larger* so `min` keeps the unclipped term — gradient alive, pulling the probability back up. With $\hat A_t<0$: inside, equal; below $1-\epsilon$ (bad action suppressed past the band, favored direction) the clipped term is more negative so `min` picks it — flat, no gradient; above $1+\epsilon$ (bad action made *more* likely, the overshoot) the unclipped term is more negative so `min` keeps it — gradient alive, *pulling the overshoot back down*. So the min clips away the incentive to push the ratio further in the direction the advantage already favors, but never clips away the gradient that corrects a move in the wrong direction. It is a pessimistic lower bound on the unclipped surrogate: ignore the full ratio change only when including it would make the objective look *better*; keep it when it makes the objective look worse. This is the exact property AWR lacked. And to first order at $\theta_{old}$, where every $r_t=1$, the clip is inactive and $L^{CLIP}$ equals the plain policy-gradient surrogate, so the first epoch is ordinary ascent and the brake engages only as the policy moves through the K epochs.

Why $\epsilon=0.2$? It is the half-width of the band the policy may move within per update, in ratio space — a $\pm20\%$ change in any action's probability before the brake fully engages. Too small and the brake bites immediately, every update is microscopic, and I waste the ten epochs of reuse — the penalty rung's microscopic-step failure in a different dress. Too large and the band rarely engages across the K epochs, the ratios fan out, and I am back to the placeholder blow-up. Around $0.2$ the policy moves a useful amount while the realized KL after a full update stays $\sim0.01$–$0.02$ — the same small-move regime the penalty rung was *targeting* with its servo, but reached by a fixed unit-free constant instead of a feedback loop that lagged behind every overshoot. That is the structural reason I expect the clip to fix the HalfCheetah seed variance: the penalty rung hit $0.01$ KL only on average and reactively, so individual minibatches overshot and the seeds diverged; the clip enforces the band on *every* minibatch *before* the step, so the seeds should track each other.

In the harness the clipped surrogate is naturally three lines — `pg_loss1 = -mb_advantages * ratio`, `pg_loss2 = -mb_advantages * clamp(ratio, 1-clip_coef, 1+clip_coef)`, `pg_loss = max(pg_loss1, pg_loss2).mean()` — where the `max` of the two *negatives* is the pessimistic `min` of the two products, because the loop minimizes the loss; `clip_coef` is the loop's $0.2$. The `get_action_and_value` head stays the standard Gaussian, identical to both prior rungs; the entire contribution lives in the loss.

There is one more piece the harness exposes that neither prior rung used, and the symmetry argument says I should take it: the value clip. The whole reason the policy surrogate is clipped is that across ten epochs the network drifts and large moves are destructive — but the value head is on the *same* drifting network getting the *same* ten epochs, so a single minibatch could yank $V_\theta(s)$ far from the `mb_values` the rollout recorded, an analogous overshoot that hurt the plain-MSE value loss the prior rungs used. So I clip the value update too, by the same pessimism logic: `max((newvalue - mb_returns)**2, (clip(newvalue, mb_values±clip_coef) - mb_returns)**2)`, never letting the clipped form *reduce* the loss, only capping how far the value prediction is rewarded for moving per update. The loop gates this on `args.clip_vloss` (default `True`), so I honor that flag. A more accurate, more stable critic feeds back into better advantages, which feeds back into a better policy update — so the value clip is the critic-side half of the same trust region, the one structural ingredient the two prior rungs left entirely on the table. There is no entropy bonus in play (`ent_coef=0` on MuJoCo, where the Gaussian's learned log-std supplies exploration), so I include the entropy term for contract-completeness but expect it to contribute nothing; the GAE advantages, per-minibatch normalization, LR anneal, and global gradient-norm clip are all the loop's. My contribution is exactly the clipped surrogate plus the clipped value loss — PPO-clip as the scaffold realizes it.

My central bet is *balance*: the hard, per-minibatch, unit-free band should give up a little of AWR's HalfCheetah peak — I would not be shocked if PPO's HalfCheetah mean lands *below* AWR's 1996.7, because AWR's 3301 seed inflated that mean and PPO trades that lucky tail for consistency — but should *not* collapse on Swimmer the way AWR did. I expect PPO's Swimmer to clear AWR's 90.2 and the penalty rung's 101.4 with much tighter seed spread, because the min-clip corrects the spurious-action overshoots that cratered AWR's seed-123 to 46, and to be at least competitive with AWR's 7299.2 on InvertedDoublePendulum. The falsifiable claim is about the *geometric mean*: PPO should be the most balanced of the three, with no environment where it is worst by a wide margin, so its three-environment geometric mean should top the baselines. If that holds, the only way past PPO is to keep its loss exactly as-is and improve the *exploration* the fixed Gaussian head supplies.

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
    """PPO clipped surrogate objective + clipped value loss."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — clipped surrogate
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss — clipped
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
