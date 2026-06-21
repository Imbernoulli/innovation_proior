## Research question

On-policy actor-critic for continuous control. I collect a batch of trajectories with the current Gaussian policy, turn the rewards into advantages with Generalized Advantage Estimation, and then run several epochs of mini-batch optimization over that one freshly-collected batch before throwing it away and collecting the next. The single thing I get to design is the **policy-update rule** — how a mini-batch of `(obs, action, old-logprob, advantage, return, old-value)` becomes a scalar loss whose gradient moves the actor and the critic. Everything else — the rollout, the GAE scan, the optimizer, the network — is frozen. The question is which update rule produces reliable improvement per batch across MuJoCo environments with very different dynamics.

## Prior art / Background / Baselines

- **REINFORCE / vanilla policy gradient (Williams 1992).** Estimate the policy gradient by Monte Carlo sampling of log-probability-weighted advantages.
- **Conservative policy iteration / surrogate bound (Kakade & Langford 2002).** Write policy improvement as maximizing an importance-weighted surrogate advantage plus a term that keeps the new policy close to the old one.
- **TRPO (Schulman et al. 2015b).** Approximately maximize the surrogate advantage subject to a hard KL trust region, using conjugate gradients and Fisher-vector products.
- **GAE (Schulman et al. 2015a).** Estimate advantages by a weighted sum of n-step TD residuals, $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$, with $\lambda$ trading bias for variance. This is fixed substrate, not a policy-update rule.
- **Advantage-weighted regression (Peters & Schaal 2007; Peng et al. 2019).** Treat policy improvement as supervised regression onto past actions, weighted by exponentiated advantages.

## Fixed substrate / Code framework

A single-actor CleanRL-style on-policy actor-critic loop is frozen and must not be touched. It builds the MuJoCo env with standard wrappers, runs one environment for `num_steps=2048` steps per iteration, then computes truncated GAE in a single reverse scan ($\gamma=0.99$, $\lambda=0.95$) with a $(1-\text{done})$ mask, and value targets `returns = advantages + values`. It then runs `update_epochs=10` epochs of shuffled mini-batch SGD (`num_minibatches=32`) over that one batch, with per-minibatch advantage normalization (`norm_adv=True`), Adam (`lr=3e-4`, `eps=1e-5`, linear LR anneal to 0), and a global gradient-norm clip at `max_grad_norm=0.5`. The optimizer steps on whatever scalar `loss` my code returns.

The network capacity is **fixed and enforced at runtime** by a parameter-count assertion: a 2×64 `tanh` MLP critic ($s\to64\to64\to1$), a 2×64 `tanh` MLP actor mean ($s\to64\to64\to a$), and a state-independent learned log-std vector `actor_logstd`. I cannot add capacity; the contribution must be algorithmic.

## Editable interface

Exactly one region of `custom_onpolicy_continuous.py` is editable (lines 175–221): the `get_action_and_value` method of `Agent` (the action distribution and what it returns) and the free-function `compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args)` (the per-minibatch loss). The contract is fixed: `get_action_and_value(obs, action)` returns `(action, logprob, entropy, value)` summed over action dims; `compute_losses(...)` returns `(loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac)`. The loop has already normalized `mb_advantages` per minibatch before handing them in.

The starting point is the scaffold default: a Gaussian policy and a **placeholder** un-clipped policy gradient $-\hat{\mathbb E}[\hat A\,r]$ with plain MSE value loss — the slot each method fills.

```python
    # EDITABLE region of custom_onpolicy_continuous.py (lines 175-221) — default fill
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    """Compute policy and value losses for a minibatch."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Policy loss -- placeholder (un-clipped policy gradient)
    pg_loss = (-mb_advantages * ratio).mean()

    # Value loss -- placeholder (plain MSE)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```

## Evaluation settings

Trained for `total_timesteps=1_000_000` and evaluated on three Gymnasium MuJoCo continuous-control environments spanning very different dynamics — **HalfCheetah-v4** (dense reward, forward locomotion), **Swimmer-v4** (low-dimensional, long-horizon credit assignment where the discount/GAE settings bite), and **InvertedDoublePendulum-v4** (unstable balancing, large achievable return) — each over three seeds {42, 123, 456}. The metric is mean episodic return over `eval_episodes=10` evaluation episodes at the fixed budget (higher is better), reported per environment; the task score is the geometric mean across the three, so a method that is strong on two environments but collapses on the third is penalized hard. Strong methods stay reliable across all three rather than tuning to one.
