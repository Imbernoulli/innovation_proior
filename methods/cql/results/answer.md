# Conservative Q-Learning (CQL)

## Problem

Offline (batch) RL learns a policy from a fixed dataset `D ~ d^β(s) π_β(a|s)` with no further
environment interaction. Standard off-policy value methods fail: the Bellman target
`r + γ E_{a'~π}[Q(s',a')]` queries `Q` at actions `a'` chosen by the policy being improved, which
drifts to out-of-distribution (OOD) actions whose `Q`-values are erroneously high and — with no
online feedback to correct them — get bootstrapped and amplified, so values diverge. Prior methods
constrain the policy toward `π_β` (BCQ, BEAR, BRAC) but leave the Q-function unregularized, so
function-approximation coupling still inflates `Q` at OOD actions; they also need an explicit
behavior-policy estimate.

## Key idea

Don't fence in the policy — make the Q-function *conservative*: learn a `Q` whose induced policy
value provably **lower-bounds** the true value, by penalizing `Q` at OOD actions while supporting
it at in-data actions. Optimizing a policy against a lower bound is safe.

## Objective

**Basic conservative evaluation (pointwise lower bound).** Add a push-down term under a chosen
`μ(a|s)`:

  `Q̂^{k+1} = argmin_Q  α E_{s~D, a~μ}[Q(s,a)]  +  ½ E_{s,a,s'~D}[(Q − B̂^π Q̂^k)²]`.

Tabular stationarity gives `Q̂^{k+1} = B̂^π Q̂^k − α μ/π_β ≤ B̂^π Q̂^k`, so at the fixed point
`Q̂^π(s,a) ≤ Q^π(s,a)` pointwise for `α` large enough to cover the `C_{r,T,δ}R_max/((1−γ)√|D(s,a)|)`
sampling error.

**Tighter bound (value, not pointwise).** Only the policy *value* `E_π[Q]` needs bounding, so add a
push-up term under the data distribution `π_β`:

  `Q̂^{k+1} = argmin_Q  α(E_{s~D, a~μ}[Q] − E_{s~D, a~π_β}[Q])  +  ½ E_D[(Q − B̂^π Q̂^k)²]`.

Now `Q̂^{k+1} = B̂^π Q̂^k − α(μ/π_β − 1)` (not a pointwise bound), but with `μ = π`,
`V̂^π(s) ≤ V^π(s)` because `D_CQL(s) = Σ_a π(π/π_β − 1) = Σ_a (π − π_β)²/π_β ≥ 0`. This is tighter
(the `−1`). Pushing up under `π_β` is *necessary*: `ν = π_β` is the only distribution for which the
bound holds for every target `π` (worst-case `max_ν min_π` analysis).

**CQL(H) — the practical objective.** Make `μ` adaptive by maximizing over it with an entropy
regularizer `R(μ) = H(μ)`. The inner `max_μ E_μ[Q] + H(μ)` has solution `μ* ∝ exp Q`, and the term
collapses to a soft-maximum:

  `min_Q  α E_{s~D}[ log Σ_a exp Q(s,a) − E_{a~π_β}[Q(s,a)] ]  +  ½ E_D[(Q − B̂^π Q̂^k)²]`.

(`R = −D_KL(μ‖ρ)` gives `μ* ∝ ρ exp Q`; `ρ = Unif` recovers CQL(H), `ρ = π̂^{k-1}` gives CQL(ρ),
preferable in high-dim action spaces where the log-sum-exp sample estimate is high-variance.)

## Guarantees

- **Gap-expanding:** the backup widens `E_{π_β}[Q] − E_μ[Q]` beyond the true gap (push-up under
  `π_β` adds zero expected shift; push-down under `μ` subtracts `α Δ̂^k ≥ 0`), so OOD actions are
  pushed below in-data actions — implicitly constraining the policy without a behavior model.
- **Across-iteration lower bound:** holds when the policy changes slowly,
  `D_TV(π̂^{k+1}, π_{Q̂^k}) ≤ ε` — hence a small actor learning rate.
- **Well-defined objective / safe improvement:** CQL solves the empirical MDP with reward
  `r − α(π/π_β − 1)`, equivalently `max_π J(π, M̂) − α/(1−γ) E[D_CQL(π, π_β)]`, and gives
  `J(π*, M) ≥ J(π_β, M) − 2 error + α/(1−γ)E[D_CQL]`, so the slack shrinks as `|D|` grows.
- Extends to linear and NTK non-linear function approximation with `α_k` chosen to absorb
  representation error.

## Algorithm

```
Initialize Q-function Q_θ (and policy π_φ for actor-critic).
for t in 1..N:
    Train Q_θ for G_Q steps on the CQL(H) objective (use B* for Q-learning, B^{π_φ} for actor-critic).
    (actor-critic only) Improve π_φ for G_π steps on  E_{s~D, a~π_φ}[Q_θ(s,a) − log π_φ(a|s)]
                        at a small learning rate.
    Soft-update target networks.
```

Continuous control: built on SAC, with the log-sum-exp estimated by importance sampling over
`N=10` actions from `Unif`, the current policy at `s`, and the current policy at `s'`; constants
from proposal weights are absorbed into the gap threshold. Discrete control: built on QR-DQN, with
the log-sum-exp computed exactly. `α` is either fixed or auto-tuned by a Lagrange dual against a
budget `τ`. No behavior-policy estimator is needed.

## Code (continuous control, on top of SAC)

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self):
        return self.value

class ContinuousCQL:
    """SAC with the CQL(H) regularizer added to the critic loss."""

    def __init__(self, actor, critic_1, critic_2, target_critic_1, target_critic_2,
                 discount=0.99, cql_n_actions=10, cql_temp=1.0, cql_alpha=5.0,
                 cql_lagrange=False, cql_target_action_gap=-1.0,
                 cql_importance_sample=True, cql_clip_diff_min=-np.inf,
                 cql_clip_diff_max=np.inf, backup_entropy=False, qf_lr=3e-4):
        self.actor = actor
        self.critic_1, self.critic_2 = critic_1, critic_2
        self.target_critic_1, self.target_critic_2 = target_critic_1, target_critic_2
        self.discount, self.backup_entropy = discount, backup_entropy
        self.cql_n_actions, self.cql_temp, self.cql_alpha = cql_n_actions, cql_temp, cql_alpha
        self.cql_lagrange, self.cql_target_action_gap = cql_lagrange, cql_target_action_gap
        self.cql_importance_sample = cql_importance_sample
        self.cql_clip_diff_min, self.cql_clip_diff_max = cql_clip_diff_min, cql_clip_diff_max
        if cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(self.log_alpha_prime.parameters(), lr=qf_lr)

    def _critic_regularizer(self, obs, actions, next_obs, q1_data, q2_data):
        B, action_dim = actions.shape[0], actions.shape[-1]
        rand_actions = actions.new_empty((B, self.cql_n_actions, action_dim)).uniform_(-1, 1)
        current_actions, current_logp = self.actor(obs, repeat=self.cql_n_actions)
        next_actions, next_logp = self.actor(next_obs, repeat=self.cql_n_actions)
        current_actions, current_logp = current_actions.detach(), current_logp.detach()
        next_actions, next_logp = next_actions.detach(), next_logp.detach()

        q1_rand = self.critic_1(obs, rand_actions)
        q2_rand = self.critic_2(obs, rand_actions)
        q1_current = self.critic_1(obs, current_actions)
        q2_current = self.critic_2(obs, current_actions)
        q1_next = self.critic_1(obs, next_actions)
        q2_next = self.critic_2(obs, next_actions)

        cat_q1 = torch.cat([q1_rand, q1_next, q1_current], dim=1)
        cat_q2 = torch.cat([q2_rand, q2_next, q2_current], dim=1)
        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cat_q1 = torch.cat([q1_rand - random_density,
                                q1_next - next_logp,
                                q1_current - current_logp], dim=1)
            cat_q2 = torch.cat([q2_rand - random_density,
                                q2_next - next_logp,
                                q2_current - current_logp], dim=1)

        q1_ood = torch.logsumexp(cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        q2_ood = torch.logsumexp(cat_q2 / self.cql_temp, dim=1) * self.cql_temp
        q1_diff = torch.clamp(q1_ood - q1_data, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        q2_diff = torch.clamp(q2_ood - q2_data, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime().exp(), min=0.0, max=1e6)
            min_q1 = alpha_prime * self.cql_alpha * (q1_diff - self.cql_target_action_gap)
            min_q2 = alpha_prime * self.cql_alpha * (q2_diff - self.cql_target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            (-(min_q1 + min_q2) * 0.5).backward(retain_graph=True)  # dual ascent
            self.alpha_prime_optimizer.step()
        else:
            min_q1 = self.cql_alpha * q1_diff
            min_q2 = self.cql_alpha * q2_diff
        return min_q1 + min_q2

    def _critic_loss(self, obs, actions, next_obs, rewards, dones, ent_alpha):
        # --- standard twin-Q TD loss (Bellman fit) ---
        q1 = self.critic_1(obs, actions)
        q2 = self.critic_2(obs, actions)
        next_a, next_logp = self.actor(next_obs)
        target_q = torch.min(self.target_critic_1(next_obs, next_a),
                             self.target_critic_2(next_obs, next_a))
        if self.backup_entropy:
            target_q = target_q - ent_alpha * next_logp
        td_target = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.discount * target_q.detach()
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        return td_loss + self._critic_regularizer(obs, actions, next_obs, q1, q2)

    def _actor_loss(self, obs, ent_alpha):        # unchanged SAC; use a small policy learning rate
        a, logp = self.actor(obs)
        q = torch.min(self.critic_1(obs, a), self.critic_2(obs, a))
        return (ent_alpha * logp - q).mean()
```

Discrete control (QR-DQN): identical except `log Σ_a exp Q(s,a)` is computed exactly with
`torch.logsumexp` over the action dimension and the backup uses `B*` (the max).
