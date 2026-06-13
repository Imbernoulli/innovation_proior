**Problem (from rung 3).** PPO topped the ladder on the geometric mean by being the most balanced
(tightest Swimmer, steady everywhere), but it leaves return on the table where exploration is the limit:
HalfCheetah only reaches its 2441 tail on one seed, and the fixed Gaussian's learned `actor_logstd`
collapses as the clip and GAE reward sharpening, so the policy commits to a basin before finding the
better one. The clip bounds the *step*, not the *entropy*. Fixing it must not touch the earned loss and
cannot add capacity (frozen parameter count).

**Key idea (RPO — Rahman & Xue, 2022).** Keep PPO's clipped surrogate + clipped value loss *unchanged*
and perturb the action-mean with uniform noise $z\sim\mathcal U(-\alpha,\alpha)$ **only during the
update** (when `action is not None`), never during data collection. The rollout stays honest; every
re-scoring of $\log\pi_\theta(a|s)$ over the K epochs sees the mean wiggled by $\pm\alpha$.

**Why it works.** Re-scoring under a moving mean penalizes a near-deterministic (tiny-$\sigma$) policy —
its log-prob swings wildly when the mean shifts — and rewards a wider $\sigma$ that is robust to the
shift. So it is an *implicit, scale-free entropy regularizer* that arrests the std collapse and smooths
the loss landscape, fighting the over-confident sharpening that caps PPO's HalfCheetah, with one $\alpha$
that transfers across environments far better than a per-environment `ent_coef`. Uniform (bounded) noise
on the *mean* at *re-scoring time* never breaks the clip's trust region and leaves the behavior policy
untouched — the `if action is None: sample else: perturb` gate is load-bearing.

**Verified against the canonical reference.** Perturbation in the `else` branch only, `rpo_alpha=0.5`
default, `uniform_(-rpo_alpha, rpo_alpha)`, Normal rebuilt before evaluation; `compute_losses` is
verbatim PPO. Matches `rpo_continuous_action.py` (edit hardcodes $\alpha=0.5$ locally and uses
`obs.device` — functionally identical; the noise is sampled, not a parameter, so it passes the
parameter-count guard).

**The bar it must clear (no feedback — endpoint).** PPO's real numbers: HalfCheetah 1757.6, Swimmer
113.2, InvertedDoublePendulum 7048.4. Validate: HalfCheetah mean up *and* the per-seed minimum rises
(more seeds find the better gait); Swimmer stays in PPO's tight band (do no harm to long-horizon credit);
InvertedDoublePendulum's worst seed does not drop below PPO (the unstable-balancing risk of over-perturbing).

**Hyperparameters.** $\alpha=$ `rpo_alpha` $=0.5$, sampled per re-scoring call on `obs.device`. All else
identical to PPO: `clip_coef=0.2`, `clip_vloss=True`, `vf_coef=0.5`, `ent_coef=0`, $\gamma=0.99$,
$\lambda=0.95$, Adam $3\mathrm{e}{-4}$, 10 epochs, 32 minibatches.

```python
    def get_action_and_value(self, obs, action=None):
        rpo_alpha = 0.5
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            # RPO: add uniform noise to action mean during update
            z = torch.FloatTensor(action_mean.shape).uniform_(-rpo_alpha, rpo_alpha).to(obs.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    """PPO clipped surrogate objective + clipped value loss (same as PPO)."""
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
