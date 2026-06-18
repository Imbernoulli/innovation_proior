**Problem.** Offline goal-conditioned RL hands the goal to the value and actor as its raw observation.
That observation mixes the task with exogenous junk (nuisance pose, jitter, scene detail), and its
geometry has nothing to do with reachability. The floor is to do nothing in the goal channel and let
the downstream GCIVL agent learn everything itself.

**Key idea (the trivial fill).** `phi(g) = g`: no learned representation, no auxiliary loss. The only
non-obvious detail is plumbing — the downstream nets are built for a fixed `rep_dim` (256) feature
width, so the raw goal (width `obs_dim`) is deterministically reshaped to `rep_dim` without parameters:
return as-is if `obs_dim == rep_dim`, zero-pad the last axis if smaller, truncate if larger. Padding
adds no information, so this is still raw-state-as-goal.

**Why it is the floor.** The value and actor get a coordinate system where proximity does not track
reachability and every exogenous coordinate rides in as if relevant. The networks must learn the
reachability geometry, control on top of it, and noise suppression all at once from finite offline
data — the exact burden a learned `phi(g)` exists to lift, lifted here not at all.

**Hyperparameters.** None learned. Inherits the harness defaults: `rep_dim=256`, downstream IVL
expectile `0.9`, AWR `alpha=10`, `discount=0.99`, `tau=0.005`. `compute_rep_loss` contributes `0.0`.

**What to watch.** The gap to a learned representation should be smallest on pointmaze-large (raw
position is already a decent code), clear on antmaze-large (goal-relevant x-y buried in exogenous
pose), and largest on cube-single (raw geometry furthest from reachability). That gap is what a
reachability-aware encoder must close at step 2.

```python
# EDITABLE region of dual-goal-representations/custom_train.py — step 1: identity goal (no rep)
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
