# PPO with Adaptive KL Penalty (PPO-Penalty), distilled

PPO-Penalty is an on-policy actor-critic algorithm that reuses each batch of freshly
collected data for several epochs of first-order minibatch SGD, kept stable by penalizing
the KL divergence between the new and old policy and **adapting the penalty coefficient**
each update to hit a target KL. It is the KL-penalty variant of proximal policy optimization:
same data flow and surrogate as the clipped variant, but the leash is an adaptive penalty
rather than a clip.

## Problem it solves

Vanilla policy gradient gets one gradient step per expensive batch of environment
interaction; doing several steps on the same batch makes the policy drift off the data
distribution and the updates blow up. TRPO fixes the stability with a hard KL trust-region
but needs second-order machinery (Fisher-vector products, conjugate gradient, line search)
and fights parameter sharing / noisy architectures. PPO-Penalty wants TRPO-grade stability
and data efficiency using only first-order optimization and a single differentiable loss.

## Key idea

Optimize the importance-sampled surrogate `L^CPI(θ) = Ê_t[ r_t(θ) Â_t ]`, with
`r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)`, but subtract a KL penalty and tune its coefficient:

```
L^KLPEN(θ) = Ê_t[ r_t(θ) Â_t − β · KL[π_old(·|s_t), π_θ(·|s_t)] ].
```

The surrogate is the local (state-visitation-frozen) approximation to the return; it is
faithful only near `θ_old`, so the KL penalty is the leash. The penalty form comes straight
from the policy-improvement lower bound `η(π̃) ≥ L_π(π̃) − C·max_s KL[π,π̃]`,
`C = 4εγ/(1−γ)²` — but that `C` is far too large for practical steps, so β is not fixed:

**Adaptive β (control the step, not the coefficient).** Measure the realized KL
`d = Ê_t[KL[π_old, π_θ]]` from the current policy evaluation and adjust β multiplicatively
toward a target `d_targ`; the updated coefficient is used on subsequent optimization calls:

```
if  d > 1.5 · d_targ:   β ← β · 2     # moved too far → tighten the leash
if  d < d_targ / 1.5:   β ← β / 2     # moved too little → loosen it
                                       # otherwise leave β unchanged (deadband)
```

This makes the controlled quantity the *policy-space step size* `d_targ` (TRPO's `δ`),
decoupled from the advantage scale and from the changing curvature of the policy across
training and across environments — exactly what no single fixed β can achieve. The factors
`1.5`/`2` are heuristic and the algorithm is insensitive to them; the initial β is unimportant
because the controller reaches the right level within a few updates. β is multiplicative
because KL responds to β on a multiplicative scale.

## Components and why

- **Probability-ratio surrogate** `r_t Â_t`: importance sampling lets one batch of old-policy
  data drive several SGD epochs; `r_t = 1` at `θ_old`, so it matches the true return to first
  order there, and only there.
- **KL estimator** `(r − 1) − log r` (Schulman's k3): unbiased (the control variate `r − 1`
  has mean 0 under `π_old`), nonnegative on every sample (since `log x ≤ x − 1`), and
  low-variance — unlike the naive `−log r`, which swings sign and can come out negative on a
  batch. Used **with gradient**: it is the penalty term, not just a diagnostic; the same value
  detached is read off to adapt β.
- **GAE(γ,λ) advantages** `Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}`, `δ_t = reward_t + γV(s_{t+1}) − V(s_t)`:
  the exponentially-weighted blend of k-step TD estimators, with `λ` the bias-variance dial
  (`λ=0` → low-variance/biased `δ_t`, `λ=1` → unbiased/high-variance Monte-Carlo). Defaults
  `γ=0.99`, `λ=0.95`. Value target is the GAE return `R_t = Â_t + V(s_t)`.
- **Value loss + entropy bonus**: squared-error critic fit `½(V_θ(s_t) − R_t)²` and an
  entropy bonus for exploration, combined as `loss = pg_loss − c_ent·H + c_vf·v_loss`.
- **Outer loop**: `N` parallel actors × `T` steps per batch; snapshot `π_old`; `K` epochs of
  Adam minibatch SGD on the freshly collected batch; adapt β from detached KL readings for
  subsequent calls; repeat. Safe to do `K` epochs only because the self-tuned `β·KL` keeps
  every pass proximal.

## Theory it rests on

The penalty is the TRPO/CPI minorization bound. With `L_π(π̃) = η(π) + Σ_s ρ_π(s) Σ_a π̃(a|s)
A_π(s,a)` and `ε = max_{s,a}|A_π|`:

```
η(π̃) ≥ L_π(π̃) − C · max_s KL[π(·|s), π̃(·|s)],     C = 4εγ/(1−γ)².
```

`L` and the bound are tight at `π̃ = π` and match `η` to first order there, so maximizing
`M(π̃) = L_π(π̃) − C·KL` gives monotonic improvement (a minorize-maximize step;
equivalently a proximal / KL-mirror-descent update over policies). PPO-Penalty keeps this
penalty structure but replaces the conservative fixed `C` with an adaptive β that targets a
KL budget.

## Algorithm

```
initialize policy/value params θ, penalty coefficient β, target KL d_targ
for iteration = 1, 2, ...:
    for each of N actors: run π_θ for T steps, store (s, a, log π_old(a|s), reward, done, V)
    compute Â_t = Σ_l (γλ)^l δ_{t+l},  δ_t = reward_t + γV(s_{t+1})(1−done) − V(s_t)
    R_t = Â_t + V(s_t);   standardize Â over the batch
    θ_old ← θ
    for epoch = 1..K, for each minibatch:
        r_t = exp(log π_θ(a|s) − log π_old(a|s))
        kl  = mean( (r_t − 1) − (log π_θ − log π_old) )          # k3, with gradient
        approx_kl = stop_gradient(kl)
        pg  = − mean(r_t · Â_t) + β · kl
        v   = ½ mean( (V_θ(s) − R_t)² )
        loss = pg − c_ent · entropy + c_vf · v
        if approx_kl > 1.5 d_targ:  β ← min(2β, 100)             # affects next call
        elif approx_kl < d_targ/1.5: β ← max(β/2, 1e-4)
        θ ← Adam step on ∇_θ loss
```

## Working code

Fills the `compute_losses` slot of the on-policy actor-critic harness (Gaussian-MLP policy,
critic, GAE, K-epoch Adam loop). β is carried as adaptive state across minibatches and updates.

```python
import torch
from torch.distributions import Normal


def get_action_and_value(self, obs, action=None):
    action_mean = self.actor_mean(obs)
    action_logstd = self.actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    probs = Normal(action_mean, action_std)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


def compute_losses(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages,
                   mb_returns, mb_values, args):
    """PPO-Penalty: adaptive KL penalty instead of a clipped surrogate."""
    # adaptive penalty state, created once and carried across updates
    if not hasattr(agent, "_kl_beta"):
        agent._kl_beta = 0.5        # initial β (controller quickly finds the right level)
        agent._target_kl = 0.01     # d_targ: targeted per-update policy-space step

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
    logratio = newlogprob - mb_logprobs         # log r_t
    ratio = logratio.exp()                       # r_t = π_θ(a)/π_old(a)

    # KL[π_old, π_θ] via the (r−1)−log r estimator: unbiased, ≥0 pointwise, low variance.
    # WITH gradient — this is the penalty term itself.
    kl = ((ratio - 1) - logratio).mean()

    with torch.no_grad():
        approx_kl = kl.detach()                  # realized KL, read off to adapt β
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()  # diagnostic

    # penalized surrogate (no clipping)
    beta = agent._kl_beta
    pg_loss = -(mb_advantages * ratio).mean() + beta * kl

    # adapt β toward d_targ (multiplicative, ×1.5 deadband, clamped);
    # this mutation affects subsequent minibatch calls, not the scalar pg_loss above.
    with torch.no_grad():
        if approx_kl > 1.5 * agent._target_kl:
            agent._kl_beta = min(agent._kl_beta * 2.0, 100.0)
        elif approx_kl < agent._target_kl / 1.5:
            agent._kl_beta = max(agent._kl_beta / 2.0, 1e-4)

    # squared-error critic fit to GAE returns (no value clipping)
    newvalue = newvalue.view(-1)
    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, pg_loss, v_loss, entropy_loss, approx_kl, clipfrac
```

Relation to the clip variant: identical data flow, GAE, and surrogate `r_t Â_t`; PPO-Clip
replaces the `β·KL` penalty with `min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)` and drops the β
controller. The adaptive-KL penalty is the variant that keeps the explicit, principled link
back to TRPO's trust-region / minorization theory.
