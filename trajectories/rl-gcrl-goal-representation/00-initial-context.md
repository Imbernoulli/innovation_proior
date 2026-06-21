## Research question

Offline goal-conditioned RL trains a single policy `pi(a | s, g)` to reach arbitrary goals from arbitrary states using only a fixed, reward-free trajectory dataset. The goal is usually fed to the value and actor as the raw observation. That raw observation is a sensing format, not a task format: it contains walls, textures, distractor objects, sensor jitter, and coordinates irrelevant to whether `g` is reachable. Two observations can encode the same reachable task while looking different; two that look similar can require a long detour.

The design object is the **goal-representation module**: a learned map `phi(g)` that replaces the raw goal everywhere the agent sees it — in `V(s, phi(g))`, `pi(a | s, phi(g))`, and the target networks. Everything else in the learner, training loop, evaluation, and data loading is fixed.

## Prior art / Background / Baselines

Existing designs:

- **Identity / raw-state goals.** Use `phi(g) = g`, concatenating `[s, g]` into value and actor. Convenient and assumption-free.

- **Variational information-bottleneck goal codes.** Compress `g` through a stochastic bottleneck with a KL penalty toward a prior.

- **Temporal metric embeddings.** Learn `phi` so that an embedding distance tracks temporal distance, then use `phi` as a frozen feature extractor.

- **Behavioral contrastive representations.** Use a flexible inner-product contrastive objective built from dataset behavior.

- **Temporal self-prediction.** Train `phi` with self-predictive temporal structure.

## Fixed substrate / Code framework

The substrate is a GCIVL (Goal-Conditioned Implicit V-Learning) agent that must not be modified. It keeps a twin value head `V(s, phi(g))` with a target copy (EMA `tau=0.005`), an AWR actor `pi(a | s, phi(g))`, and an IVL value loss that uses encoded goals throughout. Each training step sums three losses: the IVL value loss (expectile `0.9`), the AWR actor loss (`alpha=10`), and the representation module's own `rep_loss`. The downstream IVL value loss is computed so it does **not** backpropagate into `phi`; only the module's `rep_loss` trains the representation, while the value and actor consume encoded goals as inputs.

Goals are drawn by hindsight relabeling (`value_p_curgoal=0.2`, `value_p_trajgoal=0.5`, `value_p_randomgoal=0.3`, geometric future sampling). Rewards are the OGBench shifted indicator `r = I(s=g) - 1`, with `discount=0.99`. The loop also EMA-updates a `target_rep_critic` if the module exposes one, and supplies the batch fields the module may use: `observations, goals, next_observations, rewards, masks, actions`.

## Editable interface

Only one region is editable: the `GoalRepresentation(nn.Module)` class in `dual-goal-representations/custom_train.py` (lines 41-120). Every module fills the same contract:

- `setup()` — build any networks or parameters.
- `encode_goal(goals)` — map raw goal observations `(batch, obs_dim)` to `(batch, rep_dim)`. This is consumed by the value, actor, and target networks.
- `compute_rep_loss(observations, goals, next_observations, rewards, masks, actions)` — auxiliary loss summed into the agent objective; return `(0.0, {})` if none.
- `__call__(..., mode)` — dispatch `'encode'` to `encode_goal`, `'rep_loss'` to `compute_rep_loss`.

Available imports: `MLP`, `ensemblize`, `GCBilinearValue` from `utils.networks`, plus `jax`, `jax.numpy as jnp`, `flax.linen as nn`. Fixed attributes passed in: `obs_dim`, `rep_dim` (default 256), `hidden_dims=(512,512,512)`, `layer_norm=True`, `rep_expectile=0.7`, `discount=0.99`.

The default fill is **identity**: `encode_goal` returns the raw goal padded or truncated to `rep_dim`, and `compute_rep_loss` returns zero. Each candidate module replaces exactly this class and nothing else.

```python
# EDITABLE region of dual-goal-representations/custom_train.py — default fill (identity goal)
class GoalRepresentation(nn.Module):
    """Raw-state goal representation with no learned parameters."""

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True
    rep_expectile: float = 0.7
    discount: float = 0.99

    def setup(self):
        pass

    def encode_goal(self, goals):
        if self.obs_dim == self.rep_dim:
            return goals
        if self.obs_dim < self.rep_dim:
            pad_width = self.rep_dim - self.obs_dim
            pw = [(0, 0)] * (goals.ndim - 1) + [(0, pad_width)]
            return jnp.pad(goals, pw)                                 # pad last axis to rep_dim
        return goals[..., :self.rep_dim]                             # or truncate

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        return 0.0, {}                                               # no auxiliary loss

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations, rewards, masks, actions)
        return self.encode_goal(goals)
```

## Evaluation settings

Three OGBench environments are used: **antmaze-large-navigate-v0** (ant robot, large maze, long-horizon navigation), **cube-single-play-v0** (single-cube manipulation to a goal configuration), and **pointmaze-large-navigate-v0** (point robot, large maze). The downstream learner is GCIVL throughout; only the goal-representation module changes between candidates. The metric is **success rate** averaged across goal-reaching tasks in each environment. Each run trains for 1M steps with periodic evaluation at temperature 0.
