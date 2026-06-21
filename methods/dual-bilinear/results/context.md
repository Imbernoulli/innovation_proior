# Context: goal representations for offline goal-conditioned RL

## Research question

Offline goal-conditioned reinforcement learning trains one policy `pi(a | s, g)` to reach many
goals from many starts using only a fixed, reward-free dataset of trajectories. The goal usually
enters the value and policy networks as the raw goal observation `g`, which carries whatever the
observation channel contains: background pixels, sensor noise, lighting, distractor objects, and
other details. Two observations can describe the same underlying task while looking different, and
two observations that look close can require very different control.

The question is how to learn a goal code `phi(g)` that replaces the raw goal in the downstream
agent. The setting is offline: the learner works from a fixed dataset and cannot probe the
environment.

## Background

**Goal-conditioned values.** A controlled Markov process is `M = (S, A, p)` with transition kernel
`p(s' | s, a)`. Given a goal-conditioned reward `r(s, g)` and policy `pi(a | s, g)`, the value is
`V^pi(s, g) = E[sum_t gamma^t r(s_t, g)]` and the optimal value is
`V*(s, g) = max_pi V^pi(s, g)`. With the sparse indicator reward `r(s, g) = I(s = g)` and an
absorbing goal, the only positive reward is collected on first hitting the goal. If `d*(s, g)` is
the optimal reaching time, then `V*(s, g) = gamma^{d*(s, g)}`; in deterministic environments,
`d*` is the shortest-path length. OGBench-style GCRL often uses the shifted reward
`r(s, g) = I(s = g) - 1`, giving `V*(s, g) = -(1 - gamma^{d*(s, g)}) / (1 - gamma)`, a monotone
transform of the same reaching time.

**Offline optimal-value learning.** An explicit `max_a Q(s, a, g)` backup queries actions outside
the data. Implicit Q-learning instead uses expectile regression to approximate an in-support
maximum. With residual `u = q - v`, the expectile loss is

```text
ell_kappa^2(u) = |kappa - I(u < 0)| * u^2.
```

For `kappa > 0.5`, targets above the current prediction receive the larger weight `kappa`, so the
fit moves toward an upper expectile of the dataset-action target distribution without evaluating
out-of-distribution actions. OGBench's goal-conditioned IQL/IVL implementations use target networks,
EMA updates, twin value or critic heads, layer normalization on value-style networks, and relabeled
goal batches containing `observations, actions, next_observations, rewards, masks, goals`.

**Block-structured observations.** A block CMP models the observation as a latent state plus
exogenous rendering noise. It has latent states `Z`, observations `S`, latent dynamics, an emission
distribution, and a latent map `p^ell : S -> Z`, with emissions from different latents having
disjoint supports. In this model the natural goal reward is latent,
`r^ell(s, g) = I(p^ell(s) = p^ell(g))`; reaching a goal means reaching the same latent state, not
matching every noisy observation coordinate.

**Representation forms.** A finite goal code is paired with the current state inside a value or
policy network. Shared metric embeddings model the goal target as symmetric. Goal reaching is
directed: going from `s` to `g` can have a different cost than going from `g` to `s`. Behavioral
temporal representations capture where the dataset tends to go.

## Baselines

**Raw goals.** The simplest baseline is `phi(g) = g`: pass the raw goal observation directly to the
downstream value and policy. It keeps all information in the goal observation.

**Variational information bottleneck.** A stochastic goal encoder
`phi(g) ~ N(mu(g), Sigma(g))` is trained with the downstream objective plus a KL penalty
`beta * D_KL(N(mu(g), Sigma(g)) || N(0, I))`, where `beta` controls the amount of compression.

**VIP.** A value-style goal representation is trained with a metric parameterization such as
`V(s, g) = -||phi(s) - phi(g)||_2`, tying representation geometry to goal reaching through a
symmetric norm.

**HILP.** Hilbert-style representations learn geometry useful for skills and planning, sharing the
temporal structure of metric methods.

**TRA.** Temporal representation alignment uses an inner-product representation trained by
behavioral temporal contrastive learning. Its target reflects the behavior policy or dataset
occupancy.

**BYOL-gamma.** A self-predictive temporal bootstrap loss trains a goal representation from temporal
consistency, capturing predictable temporal structure.

## Evaluation settings

The natural benchmark is OGBench, which provides offline goal-conditioned datasets and standard
offline GCRL algorithms.

- **State-based tasks:** point, ant, and humanoid mazes; antsoccer; and manipulation tasks such as
  cube, scene, and puzzle.
- **Pixel-based tasks:** visual variants with `64 x 64 x 3` observations.
- **Datasets:** task-agnostic navigate/play trajectories with hindsight goal relabeling; no reward
  labels are required.
- **Downstream algorithms:** GCIVL, CRL, and GCFBC can score a representation by training
  `pi(a | s, phi(g))`.
- **Metric:** goal-reaching success rate over held-out evaluation goals.
- **Common implementation settings:** Adam with learning rate `3e-4`, batch size `1024` for
  state-based tasks, hidden dimensions `(512, 512, 512)`, target EMA rate `0.005`, discount `0.99`
  by default with larger values on the longest mazes, and layer normalization on value-style MLPs.

## Code framework

A representation module sits between raw goals and the downstream offline GCRL agent. The downstream
agent already owns the policy, value functions, relabeled batches, optimizer, and target-network
update machinery. The representation slot only needs to expose an encoder and, when the
representation is learned by an auxiliary objective, a loss term that is added to the agent loss.

Already available as JAX/Flax primitives: a standard `MLP`, an `ensemblize` wrapper that vmaps a
module into an ensemble, a goal-conditioned value/critic module `GCValue` (concatenates its inputs
and outputs a scalar), and a bilinear inner-product value module `GCBilinearValue` that computes
`phi(s)^T psi(g) / sqrt(d)` from two embedding networks (returning the two embeddings on request).
What goes inside the encoder and the auxiliary objective is what is to be designed.

```python
from typing import Sequence

import jax.numpy as jnp
import flax.linen as nn


class GoalRepresentation(nn.Module):
    """Goal encoder slot used by the downstream GCRL harness."""

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        # TODO: build the representation's trainable modules.
        pass

    def encode_goal(self, goals):
        # TODO: map raw goal observations to the vectors consumed downstream.
        pass

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        # TODO: return an auxiliary representation loss and diagnostics.
        return jnp.array(0.0), {}

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(observations, goals, next_observations,
                                         rewards, masks, actions)
        return self.encode_goal(goals)


def downstream_train_step(agent, rep, batch):
    rep_goals = rep.encode_goal(batch['goals'])
    agent_loss, agent_info = agent.total_loss(batch, rep_goals)
    rep_loss, rep_info = rep.compute_rep_loss(
        batch['observations'], batch['goals'], batch['next_observations'],
        batch['rewards'], batch['masks'], batch.get('actions'))
    return agent_loss + rep_loss, {**agent_info, **rep_info}
```
