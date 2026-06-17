# Hilbert Goal Representation

The Hilbert representation learns a single shared encoder `phi: S -> R^D` that maps both
states and goals into a Euclidean (Hilbert) space such that the latent distance
`||phi(s) - phi(g)||` approximates the *optimal temporal distance* `d*(s, g)` — the minimum
number of steps to reach `g` from `s`. It is trained by parameterizing a goal-conditioned
value function as `V(s, g) = -||phi(s) - phi(g)||` and fitting it offline with expectile
(IQL/HIQL-style) goal-conditioned value regression. The goal code handed to a downstream
goal-conditioned agent is `phi(g)`, which replaces raw goals in `V(s, phi(g))`,
`pi(a | s, phi(g))`, and the target networks.

## Problem it solves

In offline goal-conditioned RL, feeding raw goal observations to the agent forces it to
learn the environment's reachability geometry from scratch, entangled with control, while
carrying exogenous noise and irrelevant features. A good goal representation should place
goals so that geometric proximity tracks *temporal reachability*, be learnable purely from
suboptimal offline trajectories, and hand the downstream value/actor a coordinate system in
which distance-to-reach is already laid out.

## Key idea

Goal-conditioned value and temporal distance are the same object. For the goal-reaching
reward `r(s, g) = -1(s != g)` with termination at `s = g`,

```
V*(s, g) = -d*(s, g).
```

So instead of regressing distances (no labels exist), learn a value function and *force its
form to be a distance between embeddings*:

```
V(s, g) = -||phi(s) - phi(g)||,        Z = R^D with the l2 norm (a Hilbert space).
```

Training `V` to satisfy the goal-conditioned Bellman equations drives
`||phi(s) - phi(g)|| -> d*(s, g)`: `phi` becomes an approximate isometry of the MDP's
temporal structure. This parameterization is symmetric (one shared `phi` for states and
goals) and gives `V(s, s) = 0` and `V <= 0` for free.

- **Why a Hilbert (l2) space, not a bare metric:** the `l2` norm is induced by an inner
  product (`l1`/`l∞` are not), so the space carries an algebra — directions, projections,
  midpoints — usable at test time. The goal-reaching optimality proof runs through
  Cauchy–Schwarz, which needs the inner product.
- **Why expectile regression:** offline, the optimal Bellman backup's `max` cannot query
  out-of-distribution transitions. The expectile loss `L_2^tau(u) = |tau - 1(u<0)| u^2`
  fits an in-support backup over dataset next states; upper settings such as `0.7`-`0.95`
  emphasize the best in-support targets for goal-reaching, while neutral settings can be
  used when the representation objective is not strongly optimistic.
- **Approximation, not exact isometry:** `d*` is asymmetric (a quasimetric) while
  `||·||` is symmetric; not every metric embeds isometrically into a Hilbert space; and a
  discount `gamma` is used for TD stability though `d*` is undiscounted. So the objective
  finds the best *discounted symmetric Hilbert approximation*.

## Objective

With a target representation `phi_bar` and discount `gamma`, the representation loss is the
goal-conditioned value loss under the distance parameterization:

```
E[ L_2^tau( -1(s != g) - gamma ||phi_bar(s') - phi_bar(g)|| + ||phi(s) - phi(g)|| ) ].
```

In the twin-head (HIQL) form used in practice, with goal-reaching shaping `r = -1` unless at
the goal and `mask = 0` at terminal/goal: bootstrap
`next_v = min(V_1_bar(s',g), V_2_bar(s',g))`, shared conservative target
`q = r + gamma * mask * next_v`, shared target advantage
`adv = q - (V_1_bar(s,g) + V_2_bar(s,g))/2` keying the expectile direction, and each online
head regressed to its own target `q_i = r + gamma * mask * V_i_bar(s',g)`:

```
L = E[ |tau - 1(adv<0)| (q_1 - V_1)^2 + |tau - 1(adv<0)| (q_2 - V_2)^2 ].
```

The norm is floored, `||phi(s)-phi(g)|| = sqrt(max(squared_dist, 1e-6))`, to keep the
square-root gradient finite. Goals are sampled by hindsight relabeling (future-geometric
~0.625, random ~0.375; the `g=s` case is dropped since `V(s,s)=0` is guaranteed).

## Why the representation is good for goal reaching

**Theorem.** Fix `s != g`. If the local embedding error is bounded,
`sup_{s' in N(s) ∪ {s}} |d*(s', g) - ||phi(s') - phi(g)||| <= eps_e`, the directional
movement error `||z'*(s,g) - phi(s_hat')|| <= eps_d` (where
`z'*(s,g) = phi(s) + (phi(g)-phi(s))/||phi(g)-phi(s)||` is the ideal unit step toward the
goal, and `s_hat'` is the neighbor maximizing the directional inner product
`< phi(s')-phi(s), (phi(g)-phi(s))/||phi(g)-phi(s)|| >` under `||phi(s)-phi(s')|| <= 1`),
and `4 eps_e + eps_d < 1`, then stepping to `s_hat'` is optimal: `d*(s_hat', g) = d*(s,g)-1`.

*Proof.* Insert the embedding twice and use the triangle inequality:
`|d*(s_hat',g) - (d*(s,g)-1)| <= 2 eps_e + |||phi(s_hat')-phi(g)|| - (||phi(s)-phi(g)||-1)|`.
The triangle inequality (`||phi(s_hat')-phi(g)|| + 1 >= ||phi(s)-phi(g)||`) opens the bars
with a fixed sign; inserting `z'*` bounds it by `eps_d + [||z'*-phi(g)|| - (||phi(s)-phi(g)||-1)]`.
A two-case computation (`||phi(g)-phi(s)|| >= 1` gives bracket `=0`; `< 1` gives `<= 2 eps_e`)
yields `|d*(s_hat',g) - (d*(s,g)-1)| <= 4 eps_e + eps_d < 1`. Since the left side is a
difference of integers strictly below 1, it is exactly 0. Applied along the path, the
direction-following policy is optimal. The directional objective attains `z'*` by
Cauchy–Schwarz: `< phi(s')-phi(s), u > <= 1` for unit `u`, with equality at `z'* = phi(s)+u`. ∎

## Core implementation (JAX/Flax)

The value module owns `phi`; the agent keeps a separate target copy for bootstrapping and
updates it by Polyak EMA (coefficient about `0.005`). The exposed goal code is the first
ensemble member, exactly as in `get_phi`.

```python
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence

from jaxrl_m.networks import MLP, ensemblize, default_init
from jaxrl_m.typing import PRNGKey, Shape, Dtype, Array


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable = nn.gelu
    activate_final: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(observations)


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final,
                      activations=nn.gelu)(observations)


class GoalConditionedPhiValue(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    skill_dim: int = 2
    use_layer_norm: bool = True
    ensemble: bool = True
    encoder: nn.Module = None

    def setup(self):
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        phi = repr_class((*self.hidden_dims, self.skill_dim),
                         activate_final=False, ensemble=self.ensemble)
        if self.encoder is not None:
            phi = nn.Sequential([self.encoder(), phi])
        self.phi = phi

    def get_phi(self, observations):
        return self.phi(observations)[0]

    def __call__(self, observations, goals=None, info=False):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        return -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))


def expectile_loss(adv, diff, expectile):
    weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * (diff ** 2)


def compute_value_loss(agent, batch, network_params):
    batch['masks'] = 1.0 - batch['rewards']
    batch['rewards'] = batch['rewards'] - 1.0

    next_v1, next_v2 = agent.network(
        batch['next_observations'], batch['goals'], method='target_value')
    next_v = jnp.minimum(next_v1, next_v2)
    q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

    v1_t, v2_t = agent.network(
        batch['observations'], batch['goals'], method='target_value')
    v_t = (v1_t + v2_t) / 2.0
    adv = q - v_t

    q1 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v1
    q2 = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v2
    v1, v2 = agent.network(
        batch['observations'], batch['goals'], method='value', params=network_params)
    v = (v1 + v2) / 2.0

    value_loss1 = expectile_loss(adv, q1 - v1, agent.config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2 - v2, agent.config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    return value_loss, {
        'value_loss': value_loss,
        'v max': v.max(),
        'v min': v.min(),
        'v mean': v.mean(),
        'abs adv mean': jnp.abs(adv).mean(),
        'adv mean': adv.mean(),
        'adv max': adv.max(),
        'adv min': adv.min(),
        'accept prob': (adv >= 0).mean(),
    }
```

## Relation to prior work

- **Goal value = temporal distance** (Kaelbling, 1993; quasimetric RL, Wang et al., 2023):
  supplies `V* = -d*`; the quasimetric view flags `d*`'s asymmetry.
- **IQL** (Kostrikov et al., 2022) / **HIQL** (Park et al., 2023): the expectile
  in-sample-max value learning, the action-free goal-conditioned variant, twin heads +
  LayerNorm + target net, and the hindsight goal relabeling — all reused, with the value
  *parameterization* swapped to `-||phi(s)-phi(g)||`.
- **l2 temporal feature embeddings** (TCN, R3M, VIP): same `-||phi(s)-phi(g)||` value form,
  but used `phi` only as a frozen feature extractor; here the metric/inner-product geometry
  is itself the control object.
- **METRA** (Park et al., 2024): online temporal-distance abstraction + directional
  inner-product reward; this learns the analogous representation *offline* by decoupling
  representation learning from policy learning.
