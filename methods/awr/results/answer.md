# Advantage-Weighted Regression (AWR), distilled

AWR is a simple off-policy reinforcement-learning algorithm whose learning subroutines are
ordinary supervised regression. Each iteration does two regressions over a replay buffer: fit a
value function to observed returns by squared error, then fit the policy by *weighted maximum
likelihood* on the observed actions, where each action's weight is the exponential of its
advantage. It accommodates continuous and discrete actions, reuses off-policy data (a replay
buffer of past policies, or a static dataset) with no structural change, and never bootstraps a
value at an action the data does not contain.

## Problem it solves

A stable, easy-to-implement RL algorithm for continuous (and discrete) control that (1) uses
only simple convergent losses (squared error, weighted log-likelihood), (2) can reuse
off-policy data instead of discarding it, and (3) avoids the out-of-distribution-action
extrapolation that makes value-bootstrapping off-policy methods fragile — while staying as
simple as a regression loop.

## Key idea

Maximize the expected *improvement* over the data-collecting (sampling) policy `mu`, under a KL
trust region that keeps the new policy close to `mu`:

```
maximize_pi   E_{s ~ d_mu(s)} E_{a ~ pi(a|s)} [ A^mu(s, a) ]
subject to    E_{s ~ d_mu(s)} [ D_KL( pi(.|s) || mu(.|s) ) ] <= epsilon,
```

with the advantage `A^mu(s,a) = R^mu_{s,a} - V^mu(s)`. Phrasing the goal as improvement (not
raw return) is what *introduces the advantage/baseline*, via the performance-difference
identity `J(pi) - J(mu) = E_{s~d_pi, a~pi}[A^mu(s,a)]`; the surrogate swaps the
unsamplable `d_pi` for `d_mu`, valid near `mu`, which is what forces the KL constraint.

Solving the KL-constrained problem with a Lagrangian (multiplier `beta`) gives a closed-form
Gibbs/Boltzmann optimal policy:

```
pi*(a|s) = (1/Z(s)) * mu(a|s) * exp( (1/beta) ( R^mu_{s,a} - V^mu(s) ) ),
Z(s) = integral mu(a'|s) exp( (1/beta)( R^mu_{s,a'} - V^mu(s) ) ) da'.
```

Projecting `pi*` onto a parametric policy by the **forward** KL `argmin_pi
E_{s~d_mu}[D_KL(pi* || pi)]` turns the update into a sample-based *weighted regression*:

```
argmax_pi  E_{s ~ d_mu, a ~ mu} [ log pi(a|s) * exp( (1/beta) ( R^mu_{s,a} - V^mu(s) ) ) ].
```

Forward KL (not reverse) is what makes this a weighted MLE over observed actions that needs
only samples from the buffer — the trust-region constraint is enforced *implicitly*, with no
explicit model of `mu`'s density. The per-state normalizer `Z(s)` is exact to ignore for an
independent per-state projection; with a shared neural policy, dropping it is the practical
state-reweighting approximation that keeps the loss sample-based. The policy is trained only
on actions present in the data, so no out-of-distribution value is ever queried.

## Experience replay (off-policy over many past policies)

With the buffer modeled as a trajectory-level mixture of past policies, improvement over the
mixture is `eta(pi) = sum_i w_i (J(pi) - J(pi_i))`. The tractable surrogate replaces each term's
new-policy state distribution by that term's own data distribution `d_{pi_i}`, which is what
produces the state-density-weighted average of per-policy advantages in the exponent. Two
simplifications make it practical:

- **One mean value function.** Instead of fitting a separate `V^{pi_i}` per policy (data-starved
  and inaccurate), fit a single `Vbar` by squared-error regression onto returns over the whole
  buffer. The minimizer of the squared loss is exactly the needed weighted-average baseline
  `Vbar(s) = [ sum_i w_i d_{pi_i}(s) V^{pi_i}(s) ] / [ sum_j w_j d_{pi_j}(s) ]`.
- **Single-sample return target.** The density-weighted expected return in the exponent is
  replaced by the single observed return target `R^D_{s,a}` of the buffer sample. Because the
  expectation sits inside `exp`, this is a *biased* estimate (Jensen), accepted for tractability.

Practical update over the buffer `D`:

```
Vbar  = argmin_V   E_{(s,a) ~ D} [ ( R^D_{s,a} - V(s) )^2 ]
pi    = argmax_pi  E_{(s,a) ~ D} [ log pi(a|s) * exp( (1/beta)( R^D_{s,a} - Vbar(s) ) ) ].
```

## Algorithm

```
pi_1 <- random policy;  D <- empty
for iteration k = 1, ..., k_max:
    add trajectories {tau_i} sampled with pi_k to the FIFO replay buffer D
    V_k  <- argmin_V   E_{s,a ~ D} [ || R^D_{s,a} - V(s) ||^2 ]            # value regression
    pi_{k+1} <- argmax_pi E_{s,a ~ D} [ log pi(a|s)
                          * exp( (1/beta) ( R^D_{s,a} - V_k(s) ) ) ]        # weighted regression
```

## Defaults and why

- **Fixed temperature.** `beta` is the KL multiplier doubling as the temperature: small `beta`
  sharpens the exponential (aggressive, larger step from `mu`), large `beta` flattens it toward
  `mu`. Fixing it avoids the dual optimization that prior methods use to adapt it. The released
  replay code standardizes advantages and uses `temp = 1.0` in its configs, so the exponent sees a
  normalized advantage at unit temperature. The CleanRL loss slot below is the on-policy benchmark
  port: already-normalized minibatch advantages are divided by `_awr_beta = 0.05`, then clipped
  weights are renormalized to mean one.
- **Weight clipping `omega_max = 20`.** `exp((1/beta)A)` can be astronomically large for a high
  advantage, letting one sample dominate and the gradient explode; clip the weight to a
  ceiling. The full replay implementation standardizes advantages before the exponential. The
  CleanRL slot receives normalized advantages and then normalizes clipped weights to mean one
  so the shared optimizer sees a stable policy-loss scale.
- **`TD(lambda)` returns, `lambda = 0.95`.** Monte-Carlo returns are high-variance; `TD(lambda)`
  bootstraps with the previous iteration's value to trade some bias for lower variance. Setting
  `lambda = 1` recovers the Monte-Carlo target.
- **FIFO replay buffer (e.g. 50k samples).** Reuses off-policy data for sample efficiency; size
  trades stability vs. speed — a larger buffer slows `mu`'s drift, and via the KL constraint
  slows (stabilizes) `pi`; a smaller buffer is faster but overfits a small dataset.
- **Uniform state sampling from `D`** instead of the discounted `d_mu` — a standard, simpler
  approximation.

## Working code

Filling the policy/value loss slot of the actor-critic harness — a Gaussian policy and a
squared-error value loss, with the advantage-weighted regression policy loss:

```python
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    """On-policy actor-critic backbone: Gaussian policy (mean MLP + learned log-std)
    and a separate value MLP. AWR reuses this and changes only the loss."""

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
    """AWR loss for the CleanRL-style slot: weighted policy MLE + MSE value loss."""
    _awr_beta = 0.05
    _awr_max_weight = 20.0

    # log pi(a|s) of the OBSERVED actions, plus V(s); entropy for monitoring.
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)

    # Diagnostics only.
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

    # Gibbs weight from the KL-constrained improvement problem. mb_advantages has
    # already been normalized by the surrounding minibatch loop; the mean-one
    # rescale keeps this CleanRL loss slot on a stable gradient scale.
    with torch.no_grad():
        weights = torch.exp(mb_advantages / _awr_beta)
        weights = torch.clamp(weights, max=_awr_max_weight)
        weights = weights / (weights.sum() + 1e-8) * weights.numel()

    # Policy loss = -E[ log pi(a|s) * weight ]  (weighted maximum likelihood)
    pg_loss = -(newlogprob * weights).mean()

    # Value loss = 0.5 * E[ (V(s) - return)^2 ]  (single mean value fn by MSE regression)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```

## Relation to prior methods

- **RWR** (reward-weighted regression) = the same weighted-regression skeleton, but the weight
  is `exp((1/beta) R)` of the *raw return* (no baseline), and it is on-policy with an adaptive
  temperature. Replacing the return by the advantage — which the *improvement* objective
  produces automatically — and pooling off-policy data is the delta.
- **REPS** = the same KL-constrained policy search whose dual yields the identical Gibbs form,
  but it optimizes raw return, weights by a one-step Bellman error tied to a feature-matching
  linear value function, requires minimizing an intricate dual for the temperature, and models
  the sampling policy as only the latest policy (no experience replay).
- **MPO** = a deep REPS variant that fits a *bootstrapped* Q-critic with Retrace(λ) and solves a
  dual for the temperature; AWR replaces all of that with a value function fit by simple
  regression and a single weight clip.
- **Trust-region / clipped on-policy methods** optimize the same advantage surrogate under a KL
  trust region, but are fundamentally on-policy (the ratio/surrogate is defined against the
  data-collecting policy) and cannot pool a buffer of many past policies.
