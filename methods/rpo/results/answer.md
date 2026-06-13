# Robust Policy Optimization (RPO), distilled

RPO is PPO with one change: during the policy update, the Gaussian policy's mean is perturbed by
bounded uniform noise before the stored action's log-probability is re-evaluated. Sampling at rollout
stays clean. This keeps the policy's action-space entropy from collapsing as training sharpens it —
without an entropy coefficient to tune, without changing the action-distribution family, and without
touching the network, the parameter count, or the clipped surrogate that make PPO stable.

## Problem it solves

On-policy actor-critic updates that are stable (PPO's clipped surrogate) are also the updates that
drive a parametric Gaussian policy deterministic: the score-function gradient shrinks the learned
`sigma` and sharpens the mean `mu`, so the policy commits to whatever looked best early — before it has
explored enough — and a small `sigma` can no longer escape that basin. Entropy regularization needs a
per-task coefficient that fights the advantage scale; alternative distributions change the network and
help unevenly; observation augmentation perturbs the wrong end. The goal is to *maintain* useful
action-space entropy across training with one portable, bounded knob.

## Key idea

At update time only — when re-evaluating a stored action `a` to form the importance ratio — perturb
the network's mean: draw `z ~ U(-alpha, alpha)` per action dimension and evaluate
`log N(a; mu_theta(s) + z, sigma_theta)` instead of `log N(a; mu_theta(s), sigma_theta)`. The mean is
then fit to be good across a *cloud* of perturbed positions, so it cannot collapse to a needle-sharp
peak; equivalently the policy is trained against a higher-entropy mixture of jittered Gaussians.
Rollout sampling uses the clean `N(mu, sigma)`, so the collected data and the ratio `r_t =
pi_theta/pi_old` stay consistent with the policy that produced them.

## Why it works

The perturbation enters only through the re-evaluated log-prob, so the clipped surrogate, the clipped
value loss, GAE, the K-epoch loop, the network, and the parameter count are exactly PPO's — the trust
region is intact. Uniform noise is symmetric (zero-mean, so no systematic bias of the mean) and
strictly bounded by `alpha` (a hard cap, so one rare draw cannot destabilize the clip), and `alpha` is
in normalized action units, so a single value is portable across environments — unlike an entropy
coefficient measured against the drifting advantage scale. The perturbed distribution is the
unperturbed Gaussian *convolved* with independent, non-degenerate uniform noise, so it has strictly
higher differential entropy than `N(mu, sigma)` (adding an independent non-deterministic variable can
only raise entropy); entropy is raised early and maintained throughout, with no schedule. The cost is
one added line and zero new parameters.

## Hyperparameters

- `rpo_alpha = 0.5` — half-width of the uniform mean perturbation, in action units. `alpha = 0`
  recovers PPO exactly; very large `alpha` drowns the gradient signal. `0.5` is the robust default; a
  few environments do better with a much smaller `alpha` (near-PPO behavior).
- Everything else is PPO: clip `eps = 0.2`, GAE `lambda = 0.95`, `gamma = 0.99`, K update epochs,
  per-minibatch advantage normalization, clipped value loss, Adam `eps = 1e-5`, LR anneal,
  grad-norm clip `0.5`.

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
