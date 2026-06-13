**Problem (from rung 2).** AWR is the *least balanced* rung: it won HalfCheetah (1996.7) and
InvertedDoublePendulum (7299.2) but collapsed on Swimmer (90.2, seed 123 at 46) because the sharp
weighted regression over-concentrates on noisy advantages and has no way to correct a commitment. The
penalty rung was unbalanced the other way (HalfCheetah seed swing 1150–2695). The fix: keep a ratio
surrogate but make its trust region *hard and built into the loss*, non-negotiable on every environment.

**Key idea.** Clip the importance ratio $r_t=\pi_\theta/\pi_{old}$ to $[1-\epsilon,1+\epsilon]$ inside
the objective and take the pessimistic minimum of the clipped and unclipped terms,
$L^{CLIP}=\hat{\mathbb E}_t[\min(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$
($\epsilon=0.2$). The clip removes the incentive to push $r_t$ past the band *in the advantage-favored
direction*; the `min` keeps the gradient that *corrects* a wrong-direction overshoot — the exact
mechanism AWR lacked.

**Why it works.** A trust region as a flat spot in the loss: no KL term, no Lagrange coefficient to
servo, no closed-form projection. It constrains the unit-free ratio directly, so one $\epsilon=0.2$ is
reliable across all three environments where the penalty rung's $\beta$ chased three advantage scales.
Enforced per-minibatch *before* the step, so seeds track each other instead of diverging.

**Value clip (the rung-2-and-1 omission).** The value head drifts over the same 10 epochs, so clip it by
symmetry: `max((V-ret)^2, (clip(V, V_old±ε)-ret)^2)`, gated on `clip_vloss=True`. A steadier critic →
better advantages → better policy update.

**Hyperparameters.** $\epsilon=$ `clip_coef` $=0.2$, `clip_vloss=True`, `vf_coef=0.5`, `ent_coef=0`
(Gaussian log-std handles exploration). Loop defaults: $\gamma=0.99$, $\lambda=0.95$, Adam $3\mathrm{e}{-4}$
(eps $1\mathrm{e}{-5}$), 10 epochs, 32 minibatches, per-minibatch advantage normalization, global
gradient-norm clip $0.5$.

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
