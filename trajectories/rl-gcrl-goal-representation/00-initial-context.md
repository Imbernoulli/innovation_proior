## Research question

Offline goal-conditioned RL trains one policy `pi(a | s, g)` to reach arbitrary goals `g` from
arbitrary states `s`, using only a fixed, reward-free dataset of trajectories. The goal usually enters
the value and actor as its **raw observation**. That observation is a sensing format, not a task
format: it carries walls, textures, distractor objects, sensor jitter, and nuisance coordinates that
have nothing to do with whether `g` is reachable. Two observations can mean the same task while looking
different; two that look close can be a long detour apart. The single thing being designed is the
**goal-representation module** — a learned map `phi(g)` that replaces the raw goal everywhere the agent
used to see `g`: in the value `V(s, phi(g))`, the actor `pi(a | s, phi(g))`, and the target networks.
Everything else about the agent (the GCIVL learner, the training loop, evaluation, dataset loading) is
fixed. A good `phi(g)` must keep exactly the information an optimal reaching policy needs and discard
the rest, learned purely offline.

## Prior art before the first rung (goal-representation lineage)

The ladder reacts to a line of goal-representation methods, each of which the first rung will fall
short of. These precede the ladder; the fixed substrate below is the harness they plug into.

- **Identity / raw-state goals (the default in most GCRL).** Set `phi(g) = g` and concatenate `[s, g]`
  into the value and actor. Convenient and assumption-free, but it forces the networks to learn the
  whole reachability geometry of the environment from scratch, entangled with control, while also
  learning to ignore exogenous junk. Gap: keeps all the noise; no structure handed to the agent.
- **Variational information bottleneck (VIB) goal codes (Alemi et al. 2017; goal-rep use Park
  et al. 2023).** Compress `g` through a stochastic bottleneck with a KL penalty toward a prior. The
  bottleneck shrinks the code, but the KL has no idea that the robot pose matters and a texture patch
  does not — it compresses generically, not toward reachability. Gap: compression objective is not
  control-aware.
- **Temporal metric embeddings (TCN/R3M/VIP, Ma et al. 2023).** Learn `phi` so that `-||phi(s) -
  phi(g)||` tracks a temporal value, then use `phi` as a *frozen* feature extractor. The geometry is
  temporal, which is the right instinct, but it is read off a behavioral/value head and the metric
  form is symmetric. Gap: symmetric metric + frozen features, not an actionable goal code.
- **Behavioral contrastive representations (TRA, Myers et al. 2025).** A more flexible inner-product
  aggregator, but the contrastive target is *behavioral* — it describes how the dataset's behavior
  policy moves, not how an *optimal* reaching policy would move. Gap: behavior occupancy, not optimal
  reachability.
- **Temporal self-prediction (BYOL-gamma, Lawson et al. 2025).** Self-predictive temporal structure,
  again without an optimal-reaching target. Gap: predicts dataset dynamics, not optimal reaching.

The common thread the ladder will chase: the *target* that defines "close in representation space"
should be the **optimal temporal distance** `d*(s, g)`, learned offline without out-of-distribution
action queries — and the *aggregator* (how a state code and a goal code combine into that distance)
is the design knob the rungs differ on.

## The fixed substrate

A GCIVL (Goal-Conditioned Implicit V-Learning) agent is frozen and must not be touched. It keeps a
twin value head `V(s, phi(g))` with a target copy (EMA `tau=0.005`), an AWR actor
`pi(a | s, phi(g))`, and an IVL value loss that uses **encoded** goals throughout. Three losses are
summed each step: the IVL `value_loss` (expectile `0.9`), the AWR `actor_loss` (`alpha=10`), and the
representation module's own `rep_loss` (added to the objective). Crucially, the downstream IVL value
loss is computed so it does **not** backpropagate into `phi`/`psi` — only the module's `rep_loss`
trains the representation; the value/actor see encoded goals as inputs. Goals are drawn by hindsight
relabeling (`value_p_curgoal=0.2`, `value_p_trajgoal=0.5`, `value_p_randomgoal=0.3`, geometric
future sampling), rewards are the OGBench shifted indicator `r = I(s=g) - 1`, `discount=0.99`. The
loop also EMA-updates a `target_rep_critic` inside the module when it exposes one, and supplies the
batch fields a module may use: `observations, goals, next_observations, rewards, masks, actions`.

## The editable interface

Exactly one region is editable — the `GoalRepresentation(nn.Module)` class in
`dual-goal-representations/custom_train.py` (lines 41-120). Every rung is a fill of this same
contract:

- `setup()` — build any networks/parameters (or none).
- `encode_goal(goals)` — map raw goal observations `(batch, obs_dim)` to a code `(batch, rep_dim)`.
  This is what the value, actor, and target networks consume.
- `compute_rep_loss(observations, goals, next_observations, rewards, masks, actions)` — the module's
  auxiliary training loss, summed into the agent objective; return `(0.0, {})` if none.
- `__call__(..., mode)` — dispatch `'encode'` to `encode_goal`, `'rep_loss'` to `compute_rep_loss`.

Available imports from the codebase: `MLP`, `ensemblize`, `GCBilinearValue` from `utils.networks`,
plus `jax`, `jax.numpy as jnp`, `flax.linen as nn`. The attributes the harness passes in are fixed:
`obs_dim`, `rep_dim` (default 256), `hidden_dims=(512,512,512)`, `layer_norm=True`,
`rep_expectile=0.7`, `discount=0.99`.

The starting point is the scaffold default: **identity** — `encode_goal` returns the raw goal (padded
or truncated to `rep_dim` so the fixed downstream nets, which are built for `rep_dim` features, still
type-check), and `compute_rep_loss` returns zero. Each later rung replaces exactly this class and
nothing else.

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

Three OGBench environments spanning navigation and manipulation: **antmaze-large-navigate-v0** (ant
robot, large maze — long-horizon navigation), **cube-single-play-v0** (robotic single-cube
manipulation to a goal configuration), and **pointmaze-large-navigate-v0** (point robot, large maze).
The downstream learner is GCIVL throughout; only the goal-representation module changes between rungs.
The metric is **success rate** averaged across the goal-reaching tasks in each environment, higher is
better; each run trains for 1M steps with periodic evaluation at temperature 0.
