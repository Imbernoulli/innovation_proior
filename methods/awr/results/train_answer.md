Reinforcement learning for continuous control is usually built from two incompatible pieces. Policy-gradient methods are conceptually direct, but they treat the gradient as an expectation under the current policy, so every batch is thrown away after one update and the estimator is notoriously high-variance. Off-policy actor-critics reuse a replay buffer, yet they depend on Bellman backups that evaluate a learned critic at actions the current policy would take, and when those actions lie outside the training data the value extrapolates, the error bootstraps, and the policy chases phantom returns. What is missing is an algorithm whose inner loop looks like ordinary supervised regression — a squared-error fit and a weighted log-likelihood fit — but that can still reuse old data and avoid out-of-distribution action queries entirely.

Reward-weighted regression already has the right skeleton: cast policy improvement as expectation-maximization, weight observed actions by exponentiated return, and fit the policy by weighted maximum likelihood. The trouble is that the weight uses the raw return with no baseline, so it is sensitive to reward offset and scale, and the sampling policy is the current policy, so the method is on-policy and discards data. The fix is to replace the return by the advantage and to derive the exponential weight from a KL-constrained improvement objective rather than from a heuristic.

The method is Advantage-Weighted Regression, or AWR. It starts from the performance-difference identity: the improvement of a new policy π over a sampling policy μ equals the advantage of μ accumulated under π’s visitation. Because π’s own state distribution is not available during optimization, AWR swaps it for μ’s state distribution and adds a KL trust-region constraint that keeps π close to μ. Solving this constrained variational problem with a Lagrangian yields a closed-form Gibbs policy: the data policy μ reweighted by the exponential of the advantage divided by the KL multiplier β. Projecting this non-parametric optimum onto a neural network via the forward KL turns the update into a sample-based weighted maximum-likelihood regression over the actions actually observed in the data. The trust region is therefore baked into the target distribution rather than enforced by a penalty the optimizer can trade away, and the policy is never evaluated at an action it has not seen.

With a replay buffer that holds trajectories from many past policies, the sampling policy becomes a mixture. The right baseline is then a state-density-weighted average of the value functions of all those past policies, and a single squared-error regression of one value network onto returns over the whole buffer computes exactly that average automatically. The return inside the exponential is approximated by the single observed return target for each buffer sample, which is a biased but tractable estimate. In practice each iteration therefore reduces to two stable regressions over the buffer: fit a value function by mean squared error, then fit the policy by weighted maximum likelihood with weights exp((R − V(s))/β). The temperature β controls how aggressively the update sharpens toward high-advantage actions; weight clipping prevents a handful of outliers from dominating the regression; and TD(λ) returns trade a small bias for substantially lower variance.

```python
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        h = 64
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    _awr_beta = 0.05
    _awr_max_weight = 20.0

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    with torch.no_grad():
        weights = torch.exp(mb_advantages / _awr_beta)
        weights = torch.clamp(weights, max=_awr_max_weight)
        weights = weights / (weights.sum() + 1e-8) * weights.numel()

    pg_loss = -(newlogprob * weights).mean()
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
