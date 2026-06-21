Offline goal-conditioned RL trains one policy $\pi(a \mid s, g)$ to reach arbitrary goals from arbitrary states out of a fixed, reward-free dataset, and almost universally it hands the goal to the value and actor as its *raw observation*. That is the thing I want to fix with a learned goal-representation module, but a representation only ever bolts onto a downstream agent, and the honest way to measure what it buys is to first measure the agent with no representation at all. So before anything is learned in the goal channel, I want the floor: the trivial fill that proves the agent is competent and isolates exactly the burden a real $\phi(g)$ has to lift.

I propose the **identity goal representation** — $\phi(g) = g$, no learned parameters and no auxiliary loss. The scaffold is already built for it: the value $V(s, \phi(g))$, the actor $\pi(a \mid s, \phi(g))$, and the target networks all consume whatever `encode_goal` returns, so making `encode_goal` the identity feeds them the raw goal and reduces the whole system to the standard "concatenate $[s, g]$ and learn" recipe that most goal-conditioned RL defaults to. With nothing to train, `compute_rep_loss` returns $0.0$ and the agent's objective is just its own IVL value loss plus the AWR actor loss; the GCIVL learner is left to sort out everything itself.

There is exactly one non-obvious line, and it is plumbing rather than representation, so it is worth stating precisely. The downstream value and actor are initialized once, up front, for a fixed feature width `rep_dim` (default $256$); they do not adapt their input dimension to whatever `encode_goal` returns. But the raw goal observation has width `obs_dim`, which is not $256$ in general — on the order of $29$ in antmaze-large, larger in cube-single, tiny in pointmaze — so a literal `return goals` would feed an `obs_dim`-wide vector into nets expecting `rep_dim`, and the shapes would not match. The honest identity is therefore "raw goal, deterministically reshaped to `rep_dim` with no learned parameters": return the goal as-is when `obs_dim == rep_dim`; zero-pad the last axis up to `rep_dim` when it is smaller; truncate to the first `rep_dim` coordinates when it is larger. Padding with zeros adds no information and no parameters — the appended coordinates are constant, so the net simply learns to ignore them — and truncation only ever drops coordinates, which is the worst case I am deliberately exposing anyway. None of this is a $\phi$; it is reshaping to satisfy the fixed network width.

What makes this *the* floor, and not merely *a* rung, is the diagnosis it sets up. With $\phi(g) = g$ the value and actor are handed a coordinate system in which proximity has nothing to do with reachability: two goal observations that look almost identical in raw coordinates can be a long detour apart in the maze, and two that look very different can be one step apart. On top of that, every exogenous coordinate — nuisance pose terms, sensor jitter, in manipulation the parts of the scene the agent does not control — rides straight into the value and actor as if it were task-relevant. So the networks must do three jobs at once from finite offline data: learn the environment's reachability geometry from scratch, learn control on top of it, and learn to suppress the irrelevant coordinates. Nothing in the input pre-sorts any of this. That is exactly the burden a learned goal representation exists to lift, and the identity lifts none of it — by construction it carries the most noise and the least structure of anything on the ladder.

The expectation I will hold this floor to is asymmetric across environments, and that asymmetry is what points at the next step. The split falls on how much exogenous structure the raw goal carries and how far the reachability geometry diverges from raw-coordinate proximity. In **pointmaze-large** the observation is essentially the point's position — low-dimensional and already close to the quantity that matters — so raw goals are a relatively benign coordinate system and the identity should be least disadvantaged. In **antmaze-large** the goal observation mixes the x-y target the policy actually needs with a pile of joint angles and velocities exogenous to *which* goal this is, a noisy high-detour coordinate system, so the identity should leave real success on the table. In **cube-single** the goal configuration is entangled with arm and scene detail and the reachability structure is highly non-Euclidean in raw coordinates, so this is where the identity should be weakest relative to what a learned code can do. The conclusion is already the diagnosis that drives step 2: this is a *coordinate-system* problem, not a learning-capacity problem. The agent is competent; it is being handed the wrong axes, and the fix is to learn a $\phi(g)$ whose geometry already encodes how hard a goal is to reach.

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
