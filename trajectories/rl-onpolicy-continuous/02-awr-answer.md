**Problem (from rung 1).** The adaptive-KL penalty learns but is *unbalanced*: HalfCheetah swung
1150–2695 across seeds because a soft KL penalty *prices* a bad move instead of forbidding it, and the
reactive servo only shrinks $\beta$ after a minibatch has already overshot. The enemy is the importance
ratio itself.

**Key idea.** Drop the ratio surrogate entirely and frame policy improvement as *supervised weighted
regression*: minimize $-\hat{\mathbb E}_t[\,w_t\log\pi_\theta(a_t|s_t)\,]$ with
$w_t=\exp(\hat A_t/\beta)$. The exponential weight is the closed-form solution of "maximize expected
advantage subject to a KL trust region" projected onto the Gaussian policy class — so the trust region
is *baked into the target* rather than enforced by a penalty the optimizer can trade away.

**Why it works.** No ratio means nothing fans out; the weights concentrate the regression on
better-than-average actions while the implicit KL constraint keeps it near $\pi_{old}$ by construction.

**Same-named ≠ paper (what the harness forces).** The canonical AWR is *off-policy* (replay buffer,
buffer TD($\lambda$), separate critic/actor momentum optimizers, $\beta\approx1.0$). This loop is
strictly on-policy, so I drop all of that and keep only the objective: GAE advantages from the frozen
scan, one shared Adam over the fresh batch. Because the loop pre-normalizes advantages to unit scale, I
use a *small* $\beta=0.05$ (not $1.0$) to get real selectivity, clamp weights at $20$, and
**self-normalize** the clipped weights to mean one — machinery the buffer-based version does not need but
the shared-optimizer K-epoch loop does.

**Hyperparameters.** Temperature $\beta=0.05$, weight clip $20.0$, weights renormalized to mean one,
weight computation under `no_grad` (target, not path), plain MSE value loss. Loop defaults:
$\gamma=0.99$, $\lambda=0.95$, Adam $3\mathrm{e}{-4}$, 10 epochs, 32 minibatches, `norm_adv=True`.

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
