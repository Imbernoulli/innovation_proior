# Dual goal representations, distilled

A dual goal representation encodes a goal by its optimal temporal relations to the state space:
`phi^vee(g) = (s |-> d*(s, g))`. The practical bilinear version learns a transformed optimal
goal-conditioned value

```text
V_rep(s, g) = psi(s)^T phi(g) / sqrt(d)
```

with goal-conditioned IQL, then uses the goal-side embedding `phi(g)` as the representation consumed
by the downstream policy and value networks.

## Guarantees

With sparse absorbing reward `r(s, g) = I(s = g)`,

```text
V*(s, g) = gamma^{d*(s, g)}.
```

**Sufficiency.** Define

```text
pi^vee(s, f) = argmax_a E_{s' ~ p(. | s, a)}[gamma^{f(s')}].
```

For `f = phi^vee(g)`,

```text
pi^vee(s, phi^vee(g))
  = argmax_a E[gamma^{d*(s', g)}]
  = argmax_a E[V*(s', g)]
  = argmax_a Q*(s, a, g).
```

The resulting policy is greedy with respect to `Q*`, so it attains `V*`. The raw goal is not needed
once the functional supplies the successor-state optimal values.

**Noise invariance.** In a block CMP with latent map `p^ell` and latent reward
`r^ell(s, g) = I(p^ell(s) = p^ell(g))`, if `p^ell(g_1) = p^ell(g_2)`, then for every trajectory
and every time step

```text
I(p^ell(s_t) = p^ell(g_1)) = I(p^ell(s_t) = p^ell(g_2)).
```

Therefore `V*(s, g_1) = V*(s, g_2)` for every `s`, and
`phi^vee(g_1) = phi^vee(g_2)`. Observation noise that does not change the latent task is discarded.

## Practical objective

Use a finite state embedding `psi(s)` and goal embedding `phi(g)`. The inner product is chosen over a
metric aggregator because a bilinear form with learned feature maps is universal for continuous
two-variable functions on compact domains and can represent directed values; a shared Euclidean
metric is symmetric and not universal. The implementation divides by `sqrt(d)` so the dot-product
scale does not grow with representation width.

Goal-conditioned IQL trains the bilinear value toward an upper expectile of a target critic. With
residual `u = bar Q(s, a, g) - V_rep(s, g)`,

```text
ell_kappa^2(u) = |kappa - I(u < 0)| u^2
L_value = E[ell_kappa^2(bar Q(s, a, g) - V_rep(s, g))]
L_Q     = E[(Q(s, a, g) - r(s, g) - gamma V_rep(s', g))^2].
```

For `kappa > 0.5`, the larger weight is applied when the target Q exceeds the current value, giving
an in-support implicit maximum without querying out-of-distribution actions. With OGBench's shifted
reward `r(s, g) = I(s = g) - 1`, the optimum is
`V*(s, g) = -(1 - gamma^{d*(s, g)}) / (1 - gamma)`, still a monotone transform of temporal distance.

The bilinear representation value is not the downstream control value. After learning `phi(g)`, a
separate monolithic offline GCRL learner trains `pi(a | s, phi(g))` and its own value functions,
because the structured bilinear head is useful for representation learning but can be too constrained
for policy extraction.

## Implementation

```python
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.networks import GCBilinearValue, GCValue


class BilinearGoalValue(nn.Module):
    """Bilinear value whose goal branch is the exported goal code."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        self.network = GCBilinearValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            layer_norm=self.layer_norm,
            ensemble=self.ensemble,
        )

    def __call__(self, observations, goals=None):
        if goals is not None:
            return self.network(observations, goals, actions=None, info=False)

        dummy_observations = jnp.zeros_like(observations)
        _, _, goal_branch = self.network(
            dummy_observations, observations, actions=None, info=True)
        return goal_branch.mean(axis=0) if self.ensemble else goal_branch


class GoalRepresentation(nn.Module):
    """Goal-side embedding from a bilinear temporal-value learner."""

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True
    rep_expectile: float = 0.7
    discount: float = 0.99

    def setup(self):
        self.rep_value = BilinearGoalValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.rep_dim,
            layer_norm=self.layer_norm,
            ensemble=True,
        )
        self.rep_critic = GCValue(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            ensemble=True,
        )
        self.target_rep_critic = GCValue(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            ensemble=True,
        )

    def encode_goal(self, goals):
        return self.rep_value(goals)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * (diff ** 2)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        q1_t, q2_t = self.target_rep_critic(observations, goals, actions)
        q_t = jax.lax.stop_gradient(jnp.minimum(q1_t, q2_t))

        v = self.rep_value(observations, goals).mean(axis=0)
        adv = q_t - v
        value_loss = self.expectile_loss(
            adv, adv, self.rep_expectile).mean()

        next_v = self.rep_value(next_observations, goals).mean(axis=0)
        td_target = jax.lax.stop_gradient(
            rewards + self.discount * masks * next_v)
        q1, q2 = self.rep_critic(observations, goals, actions)
        critic_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()

        info = {
            'rep/value_loss': value_loss,
            'rep/critic_loss': critic_loss,
            'rep/v_mean': v.mean(),
        }
        return value_loss + critic_loss, info

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(observations, goals, next_observations,
                                         rewards, masks, actions)
        return self.encode_goal(goals)
```

Implementation notes: `GCBilinearValue` computes the square-root-scaled inner product internally and
returns an ensemble in axis `0`. Its internal names are `phi` for the observation branch and `psi` for
the goal branch; the wrapper exports that goal branch, averaged across the two ensemble members, as
the downstream goal representation. The training loop should EMA-update `target_rep_critic` with
`tau = 0.005`, matching the rest of the OGBench-style value code.
