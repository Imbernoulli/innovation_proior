# Universal Value Function Approximators (UVFA)

A UVFA is a single function approximator `V(s, g; theta)` (or `Q(s, a, g; theta)`) that estimates
value as a function of **both** the state `s` and the goal `g`, so one trained model answers value
queries for many goals at once and generalizes to goals never seen in training — exactly as ordinary
function approximation generalizes value over unvisited states. The **state-identity** member is the
most direct realization: with goals being states (`G ⊂ S`), the goal enters the network as the raw
state observation (identity goal representation), concatenated with the state.

## Problem it solves

A value function caches the utility of each state, but utility is defined only relative to the goal:
change the goal and the cached `V(s)` is wrong and must be relearned, even though related goals share
their entire value skeleton (the environment's reachability structure) and differ only in
goal-dependent detail near the reward. Prior approaches either list goals without generalizing across
them (Horde's enumerated demons) or factor value with a low-capacity per-fragment constant model
(Foster & Dayan, 2002). UVFA generalizes value over the goal space with a flexible approximator,
learnable from supplied targets or directly from experience by bootstrapping.

## Key idea

Make the goal a second input to one shared approximator. Specify "reach goal `g`" as a general value
function via a goal-as-state pseudo-reward and a pseudo-discount that is zero at the goal and otherwise
follows the environment's continuation discount. Approximate the resulting optimal value/Q over a whole
goal set with a single `V(s, g; theta)` / `Q(s, a, g; theta)`.

## Definitions

```
R_g(s,a,s') = 1   if s' = g and gamma_ext(s) != 0,   else 0
gamma_g(x)  = 0   if x = g,                            else gamma_ext(x)

V_{g,pi}(s) := E[ sum_{t=0}^inf (prod_{k=1}^{t} gamma_g(s_k))
                         R_g(s_t,a_t,s_{t+1}) | s_0 = s ]
Q_{g,pi}(s,a) := E_{s'}[ R_g(s,a,s') + gamma_g(s') * V_{g,pi}(s') ]

pi_g*(s) = argmax_a Q_{g,pi}(s,a),   V_g* = V_{g,pi_g*},   Q_g* = Q_{g,pi_g*}
Approximators:  V(s,g;theta) ~= V_g*(s),   Q(s,a,g;theta) ~= Q_g*(s,a).
```

## Architectures

- **Concatenated** `F: S x G -> R`: one MLP on `[s ; g]`. Most flexible, no factorization assumption.
  With `G ⊂ S` the goal is a raw state observation, so the goal encoder is the **identity** — this is
  the state-identity baseline.
- **Two-stream**: `phi: S -> R^n`, `psi: G -> R^n`, combined by `h`: `V(s,g) = h(phi(s), psi(g))`.
  With `G ⊂ S`, `phi`/`psi` may share first layers. Builds in the goal-independent / goal-dependent
  factorization.
- **Symmetric / distance-based** (reversible envs, where `V_g*(s) = V_s*(g)`): `phi = psi` and
  `h(phi(s), psi(g)) = -||phi(s) - psi(g)||`, yielding a metric embedding of the environment.
- With a single goal, the goal stream is a constant multiplier and the two-stream dot product collapses
  to conventional single-goal value-function approximation, so UVFA generalizes ordinary FA.

## Training

- **Supervised end-to-end**: minimize `E[(V_g*(s) - V(s,g;theta))^2]` by SGD; works for any
  architecture.
- **Supervised two-stage (two-stream, dot-product `h`)**: lay observed values into a table `M`
  (rows = states, cols = goals, `M_{s,g} = V_g*(s)`); (1) find row/column factors
  `M_{s,g} ~= phi_hat_s^T psi_hat_g`
  (OptSpace for sparse/noisy, SVD when dense) to get target embeddings `phi_hat_s`, `psi_hat_g`;
  (2) train `phi`, `psi` by separate regression toward those targets. Optional stage 3 fine-tunes
  `phi, psi, h` jointly.
- **RL, Horde-seeded**: a finite Horde learns `Q_g` for training goals off-policy; build
  `M_{t,g} = Q_g(s_t, a_t)`; rank-`n` factorize; train `phi, psi` by regression. Generalizes to goals
  no demon was trained on.
- **RL, direct bootstrapping**: goal-conditioned Q-learning, sampling a transition AND a goal:
  ```
  target_g = r_g + gamma_g(s_{t+1}) * max_{a'} Q(s_{t+1},a',g)
  Q(s_t,a_t,g) <- Q(s_t,a_t,g) + alpha*(target_g - Q(s_t,a_t,g)).
  ```
  Bootstrapping while generalizing over goals is prone to instability (deadly triad); use smaller
  learning rates and/or a bounded, well-behaved combiner — e.g. `h(a,b) = gamma^{||a-b||_2}` in
  `(0, 1]`, matching the `[0,1]` pseudo-rewards (cost: less generality).

## State-identity code (drop-in goal encoder)

The plainest, most-robust member: raw state IS the goal vector, no learned encoder, no auxiliary loss.
The agent's own value/actor losses train `V(s, g)` and `pi(a | s, g)` end-to-end. The downstream goal
width must match the raw observation/goal width; in the state-identity edit, `rep_dim=0` triggers the
runner to auto-set `rep_dim = obs_dim`.

```python
from typing import Sequence

import flax.linen as nn


class GoalRepresentation(nn.Module):
    """Identity goal representation (raw state as goal; UVFA concatenated form)."""

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        pass                                            # no parameters: phi(g) = g

    def encode_goal(self, goals):
        return goals                                    # phi(g) = g

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        return 0.0, {}                                  # no auxiliary loss

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations, rewards, masks, actions)
        return self.encode_goal(goals)
```

## Canonical computational core (concatenated UVFA + direct bootstrapping)

```python
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class UVFAConcat(nn.Module):
    """V(s,g;theta) or Q(s,.,g;theta) = MLP([s ; g])."""

    hidden_dims: Sequence[int] = (256, 256)
    act_dim: Optional[int] = None

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        out_dim = 1 if self.act_dim is None else self.act_dim
        return nn.Dense(out_dim)(x)


def goal_as_state_terms(next_observations, goals, gamma_ext_t, gamma_ext_next):
    reached = jnp.all(next_observations == goals, axis=-1)
    continuing = gamma_ext_t != 0
    rewards = (reached & continuing).astype(jnp.float32)
    gamma_next = jnp.where(reached, 0.0, gamma_ext_next)
    return rewards, gamma_next


def tabular_uvfa_q_update(q_sa, next_q_values, rewards, gamma_next, alpha):
    td_target = rewards + gamma_next * jnp.max(next_q_values, axis=-1)
    return q_sa + alpha * (td_target - q_sa)


def uvfa_td_loss(q_values, next_q_values, actions, rewards, gamma_next):
    q_sa = jnp.take_along_axis(q_values, actions[..., None], axis=-1).squeeze(-1)
    target = rewards + gamma_next * jnp.max(next_q_values, axis=-1)
    target = jax.lax.stop_gradient(target)
    return jnp.mean((target - q_sa) ** 2)
```
