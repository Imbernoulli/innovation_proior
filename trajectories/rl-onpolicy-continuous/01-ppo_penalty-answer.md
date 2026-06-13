**Problem.** Reusing one on-policy batch for ten epochs of SGD is the data-efficiency the loop is built
for, but naive ascent on $-\hat{\mathbb E}[\hat A\,r]$ blows up as the policy drifts off $\pi_{old}$.
The first update rule needs a first-order leash on the policy move per batch.

**Key idea.** Take the penalty form the trust-region theory suggests directly — maximize $\hat{\mathbb
E}_t[r_t\hat A_t-\beta\,\mathrm{KL}[\pi_{old},\pi_\theta]]$ with the *un-clipped* ratio
$r_t=\pi_\theta/\pi_{old}$ — and make the one hopeless part, the coefficient $\beta$, tunable by
servoing it onto a target KL: after each update, if the realized KL exceeds $1.5\,d_{targ}$ double
$\beta$, if it drops below $d_{targ}/1.5$ halve it. The penalty is the sole brake; there is no ratio
clipping.

**Why it works.** A fixed $\beta$ cannot balance $r\hat A$ against $\beta\,\mathrm{KL}$ because the
advantage scale and KL sensitivity change across environments and across a run; closing a feedback loop
on the *measured* KL removes the guess. The lazily-initialized `agent._kl_beta` / `agent._target_kl`
persist the servo state across the free-function calls the harness makes.

**Why this rung is the floor.** The adaptation is reactive (it shrinks $\beta$ only *after* a minibatch
overshoots) and the penalty is *soft* (the optimizer can pay the KL cost to chase a noisy advantage), so
this is the least balanced update rule — expect a weak environment to drag its geometric-mean score.

**Hyperparameters.** Initial $\beta=0.5$, target KL $d_{targ}=0.01$, adaptation band $1.5\times$,
multiply/divide by $2$, $\beta$ clamped to $[10^{-4},100]$. KL estimator $\hat{\mathbb E}[(r-1)-\log r]$
**with gradient** for the penalty, detached for the adaptation. Plain MSE value loss (no value clipping).
Loop defaults: $\gamma=0.99$, $\lambda=0.95$, Adam $3\mathrm{e}{-4}$, 10 epochs, 32 minibatches.

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
    """PPO-Penalty: adaptive KL penalty instead of clipped surrogate."""
    if not hasattr(agent, '_kl_beta'):
        agent._kl_beta = 0.5
        agent._target_kl = 0.01

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # KL divergence — WITH gradient for the penalty term
    kl = ((ratio - 1) - logratio).mean()

    with torch.no_grad():
        approx_kl = kl.detach()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss — KL-penalized (no clipping)
    pg_loss = -(mb_advantages * ratio).mean() + agent._kl_beta * kl

    # Adapt KL penalty coefficient
    with torch.no_grad():
        if approx_kl > 1.5 * agent._target_kl:
            agent._kl_beta = min(agent._kl_beta * 2.0, 100.0)
        elif approx_kl < agent._target_kl / 1.5:
            agent._kl_beta = max(agent._kl_beta / 2.0, 1e-4)

    # Value loss — simple MSE (no clipping)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
