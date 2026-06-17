## Research question

A value function `V(s)` caches the single most useful piece of knowledge an agent has: starting
from state `s` and behaving well, how much total (discounted) reward will I collect? Function
approximation — a linear model or a neural network `V(s; theta)` — lets that knowledge generalize
across states: nearby states get similar values without being visited, and the estimate can be
learned from partial, off-policy trajectories by bootstrapping from later estimates. But the value
of a state is only defined *relative to what the agent is trying to do*. Change the goal — a new
waypoint to reach, a different reward — and the cached `V(s)` is simply wrong; it has to be relearned
from scratch for the new goal, throwing away everything that was learned for the old one.

This is wasteful precisely because goals are not unrelated. In a navigation problem the only thing
that changes between "reach the kitchen" and "reach the door next to the kitchen" is where the reward
sits; the walls, the corridors, the reachability structure are identical, so the two value landscapes
agree almost everywhere and differ only near the goal. There is just as much exploitable structure in
the space of goals as there is in the space of states. Standard function approximation captures the
first kind of structure (generalization over `s` for a fixed goal) and ignores the second entirely.

The precise goal: a model-free value-learning mechanism that reuses structure across related goals,
answers queries for goals that were not individually solved in advance, and still learns from partial,
off-policy experience by bootstrapping rather than from complete episodes for every goal. Closing the
gap between "one value function per solved goal" and useful value estimates for a whole related goal
class is the problem.

## Background

By the mid-2010s, value-function approximation is the backbone of reinforcement learning (Sutton &
Barto, 1998). The action-value function `Q^pi(s,a) = E[ G_t | S_t=s, A_t=a, A_{t+1:} ~ pi ]` is the
expected return from taking `a` in `s` and following `pi`; with a parametric approximator
`Q(s,a; theta)` and gradient temporal-difference updates it generalizes over states, learns
off-policy from incomplete trajectories, and underlies value-based control via greedy improvement.
What it does *not* do is carry the goal as an argument: each reward function induces its own value
function, and the approximator is fit to exactly one of them.

Several pieces of the surrounding theory matter here.

- **Goals as states, and the pseudo-reward / pseudo-discount machinery.** A clean way to talk about
  "reaching a goal `g`" inside an MDP is to attach to each goal a *pseudo-reward* `R_g(s,a,s')` and a
  *pseudo-discount* `gamma_g(s)` that acts as both state-dependent discounting and soft termination:
  it is zero at the goal, and otherwise follows the environment's own continuation discount. In the
  common case where goals *are* states -- `G ⊂ S`, and entering the goal state is what is
  rewarded -- these specialize to
  ```
  R_g(s,a,s') = 1   if s' = g and gamma_ext(s) != 0,   else 0
  gamma_g(x)  = 0   if x = g,                            else gamma_ext(x)
  ```
  with `gamma_ext` carrying ordinary environment discounting and external termination. Under such a
  pseudo-reward the value of `s` for goal `g` becomes the expected discounted first-hit reward for
  reaching `g` from `s`. The return is naturally written with the immediate transition reward first
  and the successor-state pseudo-discounts controlling all later terms:
  ```
  V_{g,pi}(s) = E[ sum_{t>=0} (prod_{k=1}^t gamma_g(s_k))
                         R_g(s_t,a_t,s_{t+1}) | s_0 = s ],
  Q_{g,pi}(s,a) = E_{s'}[ R_g(s,a,s') + gamma_g(s') V_{g,pi}(s') ].
  ```
  The empty product is `1`, so an immediate hit contributes its reward, while `gamma_g(s') = 0` cuts
  off bootstrapping after entering the goal. This framing -- value
  functions defined by arbitrary `(pi, gamma, r)` rather than the base task's reward -- is the
  *general value function* (GVF) viewpoint.

- **A goal's value landscape mirrors the environment's structure.** Plotting the optimal value
  function `V^g(x)` for a fixed goal in a maze lays bare the maze: values are smooth within a room and
  drop sharply across a wall, and for two *different* goals the within-room values differ but the
  across-wall discontinuities sit in the same places (Foster & Dayan, 2002). The set of optimal value
  functions for many goals therefore shares a goal-independent skeleton (the environment's
  reachability structure) plus goal-dependent detail (where the reward sits). This shared skeleton is
  why one should expect transfer across goals to be possible at all.

## Baselines

The prior methods a goal-generalizing value function would be measured against and reacts to.

**Single-goal value-function approximation (Sutton & Barto, 1998; Precup et al., 2001).** Fit one
parametric `V(s; theta)` (or `Q(s,a; theta)`) to the value function of one reward, learning even from
partial trajectories and off-policy data by bootstrapping. It exploits structure in state space and
is the workhorse of RL. **Gap:** the goal is baked into the reward the approximator was fit to; it
takes no goal argument, so a different goal requires a different model trained from scratch, and the
similarity between the value landscapes of related goals is left entirely unused.

**Horde / general value functions (Sutton et al., 2011).** Represent knowledge as a large collection
of independent value functions — "demons" — each a GVF `q(s,a; pi, gamma, r, z)` answering one
predictive or goal-directed question, with its own target policy `pi`, pseudo-discount `gamma`,
pseudo-reward `r` and pseudo-terminal-reward `z`, all learned in parallel and off-policy with
gradient-TD from a shared stream of experience:
```
q(s,a; pi,gamma,r,z) = E[ G_t | S_t=s, A_t=a, A_{t+1:T-1} ~ pi, T ~ gamma ],
q_hat(s,a; theta) = theta^T phi(s,a).
```
This is the source of the pseudo-reward/pseudo-discount vocabulary above and shows that many
goal-directed value functions can be learned at once off-policy. **Gap:** the demons are *enumerated
and learned separately* — knowledge is a discrete set of functions, one per question, with no sharing
of what is learned unless demons happen to share parameters. Adding a new goal means adding a new
demon and learning it; the architecture cannot answer a value query for a goal that was not put in
the set in advance, and its cost scales with the *number of demons* rather than with the inherent
complexity of the domain.

**Mixture-model structure in value-function space (Foster & Dayan, 2002).** The closest prior work on
sharing across goals. Treat the *set* of optimal value functions `D = {V^g(.)}` for many goals as
data and fit an unsupervised mixture model to it. Each state's value under a goal is modeled as a
mixture of constants,
```
P[V^g(x); theta] = sum_i pi_i(x) * Normal( V^g(x); e_i^g, omega ),
pi_i(x) = softmax_i( a_i(x) ),     V_bar^g(x) = sum_i pi_i(x) * e_i^g,
```
where the fragment-membership weights `pi_i(x)` are **goal-independent** (the shared skeleton, fit by
EM / online gradient ascent across goals) and the per-fragment values `e_i^g` are **goal-dependent**.
The fragments recovered correspond to rooms/regions of the maze, and an actor-critic over multiple
goals trained on a state representation *augmented* with fragment membership learns faster. **Gap:**
the value model is a mixture of *constants* — within a fragment, for a goal, the value is a single
number -- which is a low-capacity representation of how values vary within a fragment. In their
experiments it has to be augmented by a tabular state-and-goal representation, so it does not, on its
own, scale beyond small solved tables and hand-recovered fragments.

**Closest-policy / parameterized-skill methods (Da Silva et al., 2012; Kober et al., 2012;
Deisenroth et al., 2014).** Generalize across *tasks* by parameterizing a policy or combining local
policies into a situation-sensitive controller, largely in the policy-search setting. **Gap:** these
work at the policy level and often need complete episodes per task/goal; they do not exploit the
specific shared structure of *value* functions across goals, and (being policy-search) cannot in
general learn every goal off-policy by bootstrapping from any behavior stream.

## Evaluation settings

The natural yardsticks that already exist for a goal-generalizing value function:

- **Grid-world / maze domains with a goal that can be any state.** A 4-rooms grid-world with the four
  cardinal actions, and "LavaWorld" — a multi-room grid with deadly lava cells and doors that
  teleport between rooms, where the observation shows only the current room. Goals are states
  (`G ⊂ S`), entering the goal is rewarded. Supervised protocol: ground-truth optimal values
  `V^g*(s)` are supplied for a *training* subset of `(s, g)` pairs; held out is a *test* set of unseen
  `(s, g)` pairs (and, for transfer, whole unseen goals). Metrics: value prediction error (MSE on
  held-out pairs) and *policy quality* — the true discounted return of acting greedily/soft-greedily
  w.r.t. the learned values, normalized so optimal behavior scores 1 and the uniform random policy
  scores 0 -- measured as a function of training samples and of the fraction of `(s,g)` pairs held
  out.
- **Atari Ms. Pac-Man (Arcade Learning Environment; Bellemare et al., 2012).** A pixel-input domain
  with a hand-crafted goal space `G ⊂ R^2`: each pellet on screen is a goal, identified by its `(x,y)`
  coordinate; eating it is the per-goal reward. Used to test scaling to visual inputs and
  generalization to goals (pellet locations) with no dedicated training signal.
- **Protocol pieces shared across settings.** Train on a training set of goals `G_T`, evaluate on a
  disjoint test set `G_V`; sweep the proportion of `(s,g)` values unobserved; measure both prediction
  error and policy quality versus training samples or learning updates; for transfer, post-train on
  goals outside the training set and measure learning speed against training from scratch.

## Code framework

The goal-conditioned value/policy harness already has the generic pieces: an observation/goal data
pipeline that samples transitions and goals; standard neural primitives such as an MLP, ensemble
wrappers, and value/actor heads; an optimizer; and a learner that bootstraps from a target value
network. The open slot is a small module that turns the supplied goal descriptor into the vector the
downstream value and policy networks consume. The slot may have parameters or auxiliary losses, but
the harness only requires two operations: encode the goal, and optionally report a representation loss
to add to the ordinary value/actor objective.

```python
from typing import Sequence

import flax.linen as nn

# Pre-existing neural primitives in the codebase.
from utils.networks import MLP, GCValue, GCActor


class GoalRepresentation(nn.Module):
    """Maps a goal descriptor to the vector the actor/value nets consume.

    obs_dim : observation/goal descriptor dimension
    rep_dim : width the downstream actor/value nets expect for the encoded goal
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        # TODO: the goal-to-vector mapping and any parameters it needs.
        pass

    def encode_goal(self, goals):
        # TODO: return an encoded goal array of shape (batch, rep_dim).
        pass

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        # TODO: return any auxiliary loss for learning the mapping, or (0.0, {}).
        pass

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations, rewards, masks, actions)
        return self.encode_goal(goals)


class GoalConditionedAgent:
    def value_loss(self, batch, params):
        goal_reps = self.encode(batch['value_goals'])               # uses the slot above
        next_v = self.target_value(batch['next_observations'], goal_reps)
        target = batch['rewards'] + self.discount * batch['masks'] * next_v
        v = self.value(batch['observations'], goal_reps, params)
        return value_objective(v, target)                            # existing loss

    def actor_loss(self, batch, params):
        goal_reps = self.encode(batch['actor_goals'])               # uses the slot above
        # existing advantage-weighted / policy-gradient actor objective on (s, goal_rep)
        ...

    def total_loss(self, batch, params):
        v_loss, _ = self.value_loss(batch, params)
        a_loss, _ = self.actor_loss(batch, params)
        rep_loss, _ = self.rep_loss(batch, params)                  # the slot's aux loss
        return v_loss + a_loss + rep_loss
```

The `encode_goal` / `compute_rep_loss` bodies are the slot the method fills; every downstream value,
target-value, and actor call already consumes whatever `encode_goal` returns.
