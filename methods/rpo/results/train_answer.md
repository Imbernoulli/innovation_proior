The failure I want to kill is quiet but ruinous: when I train a Gaussian policy $\pi_\theta(a\mid s)=\mathcal{N}(\mu_\theta(s),\sigma_\theta)$ on continuous control with the proximal clipped surrogate — the reliable first-order base learner — the stochasticity collapses too fast. The learned standard deviation $\sigma$ shrinks, the mean $\mu$ sharpens onto whatever action currently looks best, and from then on the agent only samples tiny jitters around that one action. If that action were truly optimal this would be fine, but it usually is not — it is merely whatever had the highest advantage in the first few batches, before the agent had seen enough of the action space to know better — and once $\sigma$ is small the gradient only reinforces the basin the policy is already in. Exploration dies and does not return. The trouble is not stability; the clip already gives me that. The trouble is that *the stable update is also the update that drives the policy deterministic*, and it does so before the agent has earned the right to be deterministic. The cause is visible in the score-function gradient of a Gaussian. From $\log\mathcal{N}(a;\mu,\sigma)=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma-\text{const}$, the derivative with respect to the mean is proportional to $\frac{a-\mu}{\sigma^2}A$ — push $\mu$ toward high-advantage actions, as it should — but the derivative with respect to $\log\sigma$ is proportional to $\left(\frac{(a-\mu)^2}{\sigma^2}-1\right)A$. For a good action ($A>0$) that landed near the mean ($(a-\mu)^2<\sigma^2$) this bracket is negative, so the update *decreases* $\log\sigma$. As the policy improves, the good actions are increasingly the near-the-mean ones, so on average the variance is pushed down. The collapse is not an implementation bug; it is what the maximum-likelihood-style policy gradient does to a parametric Gaussian, and the clip bounds only the size of each such step, never its direction.

The standard remedies each target the right symptom but with the wrong instrument. An entropy bonus $c_2\,H[\pi_\theta]$ — with $H$ of a diagonal Gaussian equal to $\sum_i\frac12\log(2\pi e\,\sigma_i^2)$, monotone in $\log\sigma$ — directly counter-pushes the $\log\sigma$ gradient, but the coefficient that balances it on HalfCheetah is wrong on Swimmer, and the value that is right early in a run is wrong late, because the entropy push is fixed-scale while the variance-shrinking push has the scale of the *advantages*, which vary by orders of magnitude across environments and across training. Set $c_2$ too small and the policy still collapses; too large and $\sigma$ is pinned high forever and the policy never commits. That is one tuning problem traded for another, and the new knob needs a per-task sweep. Changing the distribution family — a Beta, a heavier tail, a normalizing flow — adds parameters and assumptions, changes the network I am told to leave fixed, and helps unevenly across tasks. Observation augmentation perturbs the *input* side, while the entropy collapse I diagnosed lives on the *output* side, in the action distribution's spread, so it aims at the wrong end of the network. What none of them does is the most direct thing: keep the *update itself* from being able to lock onto a single sharp mean.

I propose Robust Policy Optimization (RPO): PPO, entirely unchanged, except that during the policy update the Gaussian mean is perturbed by bounded uniform noise before the stored action's log-probability is re-evaluated. At update time I am re-evaluating, for each stored transition $(s,a)$, the log-probability $\log\pi_\theta(a\mid s)$ under the current parameters to form the clipped surrogate. Instead of scoring under $\mathcal{N}(\mu_\theta(s),\sigma_\theta)$, I draw a small random vector $z$ and score under $\mathcal{N}(\mu_\theta(s)+z,\sigma_\theta)$, so the gradient that flows back to $\mu_\theta(s)$ becomes "move the mean to where this good action was, *given that the mean is jittered by $z$*." Over the minibatch and the epochs the mean is asked to be good on average across a cloud of perturbed positions, so it cannot collapse to a needle-sharp peak that is optimal only at one precise $\mu$. That is exactly "keep entropy up," but enforced through the geometry of the update rather than through a coefficient in the loss. Three design choices make it work, and each beats its obvious alternative. First, the *shape* of the noise: I want it zero-mean (so $\mathbb{E}[z]=0$ and it spreads the mean without biasing it in any direction) and strictly bounded (so one rare draw cannot yank the mean far and destabilize the clip). A zero-mean Gaussian $z$ is symmetric but unbounded — a fat tail can occasionally throw the mean wildly — whereas $z\sim U(-\alpha,\alpha)$ per dimension is symmetric, zero-mean, and hard-capped by $\alpha$ in the same action units, which is precisely what lets one $\alpha$ mean roughly the same thing across environments after the harness's observation and action normalization. Second, and this is the crux, *where* the perturbation applies: at update time only, never at rollout. During the rollout the agent acts in the environment, and I want that data to come from the actual policy $\mathcal{N}(\mu_\theta(s),\sigma_\theta)$, clean, because the stored action and its log-prob $\log\pi_{\text{old}}(a\mid s)$ are the reference that the importance ratio $r_t=\pi_\theta/\pi_{\text{old}}$ and the advantages are built against; jittering the mean at sample time would collect data from a different policy and evaluate ratios against it inconsistently, and exploration at action-selection time is already $\sigma$'s job. So sampling stays clean (when no stored action is supplied), and the perturbation enters purely as a regularizer on the gradient (when a stored action is supplied). Third, the *magnitude* $\alpha$: at $\alpha=0$ the method is exactly PPO and the entropy collapses as before, while a very large $\alpha$ jitters the mean so hard that the stored action's log-prob becomes noise and the gradient is uninformative; the sweet spot is large enough to keep the mean from collapsing to a needle yet small enough that the gradient still points at the right action. After the standard observation normalization and action clipping, actions live on a roughly unit scale, so a half-width of about $0.5$ — jittering the mean by up to half an action-unit — keeps the policy honestly spread without drowning the signal, and because $\alpha$ is bounded and measured in action units rather than balanced against the drifting advantage scale, a single default carries across environments far better than a single entropy coefficient could.

Two checks make the claim rigorous. The first is that the trust region survives untouched: the perturbation enters only through $\texttt{newlogprob}$, the numerator of $r_t=\exp(\texttt{newlogprob}-\texttt{mb\_logprobs})$, while $\texttt{mb\_logprobs}=\log\pi_{\text{old}}$ is the clean rollout value, so the ratio, the clip with the same $\varepsilon=0.2$, the min, the GAE advantages with $\lambda=0.95$, the clipped value loss, and the $K$ epochs are all structurally identical to PPO — I have added one line inside the action readout, not replaced the base learner. The second is the entropy argument itself, on which the whole method rests. With the mean jittered by $z\sim U(-\alpha,\alpha)$ each update, the action distribution the surrogate effectively trains the policy to be good under is a mixture over $z$ of $\mathcal{N}(\mu+z,\sigma)$, which is exactly the law of $(\mu+z)+\epsilon$ with $\epsilon\sim\mathcal{N}(0,\sigma^2)$ and $z\sim U(-\alpha,\alpha)$ independent — the unperturbed Gaussian *convolved* with independent, non-degenerate uniform noise. Adding an independent, non-deterministic random variable can only smear a density out, so $H[X+Y]\ge H[X]$ with strict inequality when $Y$ is non-degenerate, and the perturbed distribution has strictly higher differential entropy than $\mathcal{N}(\mu,\sigma)$. At every update the policy is therefore fit to do well under a higher-entropy version of itself, which keeps $\sigma$ from collapsing and $\mu$ from sharpening onto a single point: entropy is raised early when exploration matters most and *maintained* throughout, because the jitter is applied every update and never annealed, so the policy can still escape a bad basin late in training. One last consistency note: because $z$ is sampled fresh and is independent of $\theta$, it adds no spurious gradient path — $\mu+z$ differentiates to $\nabla\mu$ and $z$ is an additive constant in that backward pass — so the perturbation is a stochastic regularizer that costs zero new parameters, leaving the parameter count identical to PPO's and the contribution purely algorithmic. The result is one bounded knob $\alpha$ (default about $0.5$, in normalized action units, hence portable), a single added line in the action readout, and a method that is PPO wherever PPO already works and strictly more exploratory wherever PPO collapses too soon.

```python
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal


class Agent(nn.Module):
    """RPO = PPO + a uniform perturbation of the policy mean applied only during
    the update. Same network, same parameter count, same clipped surrogate."""

    def __init__(self, obs_dim, action_dim, rpo_alpha=0.5):
        super().__init__()
        h = 64
        self.rpo_alpha = rpo_alpha          # half-width of the uniform mean perturbation
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
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()                       # ROLLOUT: clean, unperturbed
        else:
            # UPDATE: jitter the mean by z ~ U(-alpha, alpha) before scoring the
            # stored action; keeps the policy from collapsing to a sharp mean.
            z = torch.empty_like(action_mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """Identical to PPO: clipped surrogate + clipped value loss. The only change
    from PPO is the mean perturbation in get_action_and_value above."""
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -args.clip_coef, args.clip_coef)
        # torch.max of two tensors is the elementwise max (torch.maximum); .mean() is taken after.
        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb_returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```
