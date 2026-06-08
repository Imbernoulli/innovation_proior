# Context: representation for temporal-difference value learning

## Research question

I want to predict, for each state of a Markovian environment, the long-run discounted
return an agent will accumulate from that state under a fixed policy — the value
function — and to *learn* it from experienced transitions without being handed the
environment's dynamics. Temporal-difference (TD) methods already do this: they bootstrap
each state's value estimate off the estimates of the states that follow it. But TD is
slow, and how slow depends entirely on how I *represent* states to the learner. The
precise question is: for a value learner that is linear in some state features, what is
the *right* representation — the set of features such that nearby features mean nearby
values, generalization helps rather than hurts, and learning is fast — when "nearby" has
to mean something temporal, not spatial? A solution has to (i) make a state's features
resemble those of the states it tends to lead to, (ii) be learnable from experience when
the transition structure is unknown, and (iii) ideally let value computed for one task be
reused when the goal changes but the world's dynamics do not.

## Background

The setting is a finite Markov chain (here, an absorbing chain). An agent in non-absorbing
state *i* moves to *j* with probability given by the *ij*th entry of a transition matrix,
or terminates in an absorbing state with some reward. The object of interest is the
expected discounted return from each state, collected into a value vector. Two facts about
this object are load-bearing and known before any new representation is invented.

First, the value of a state is a sum over the *futures* reachable from it: the
overall expected return of a state is a (biased) sum of the overall expected returns of its
potential successors. Written out for the absorbing-chain setup, the expected-return vector
is the immediate-return vector, plus the transition matrix applied to that vector, plus the
squared transition matrix applied to that vector, and so on: h + Qh + Q²h + ... =
(I − Q)⁻¹h. This is just the matrix form of the Bellman consistency the learner is trying
to enforce.

Second, **temporal-difference learning** (Sutton 1988, building on Samuel 1959 and Sutton
1984) estimates this value not by comparing each prediction to the eventual outcome, but by
comparing temporally successive predictions. TD(0) changes the estimate of the current
state to reduce the discrepancy between it and the (reward-plus-discounted) estimate of the
next state; TD(λ) blends in estimates from states further ahead, weighted geometrically.
For a linear approximator — value of a state = weight vector dotted with that state's
feature vector — the update nudges the weights along the current state's feature vector in
proportion to that one-step prediction error. Sutton proved, for batch updates with a small
learning rate and linearly independent state features, that the expected estimates converge
to the true values; this was later extended to TD(λ) for intermediate λ.

The known pain points of this framework, around its time:

- **Speed.** The convergence is slow, and the speed is dominated by the choice of features.
  Choosing good features *is* choosing a good approximation scheme in the linear case.
- **Representation is everything, and it is hard to guess.** For static (non-temporal)
  problems the received wisdom is that distributed representations work best *provided* the
  distribution matches the task — nearby points should have nearby solutions. The open
  difficulty is what "nearby" should mean when the task is a prediction over time.
- **The coupling problem.** Because a state's value is bound up with all of its successors'
  values, a representation tied directly to states couples them: a local change to the
  reward or the dynamics forces value to be re-estimated over much of the space, because the
  estimates are bootstrapped off one another.

A diagnostic phenomenon that sets up the whole problem comes from how existing distributed
representations behave in a simple maze. A CMAC / coarse-coding scheme (Albus 1975), as used
by Watkins (1989) for a grid navigation task, tiles the grid with overlapping receptive
fields so that grid-adjacent locations share many active features and therefore generalize
to similar values. Over most of an open maze this is excellent — locations close in the
Manhattan grid metric are indeed similar distances from a goal. But place a barrier in the
maze: two locations on opposite sides of the barrier are close in the grid and so share CMAC
features, yet they are very far apart in terms of the actual task (you must go around). The
fixed, a-priori spatial metric of the representation *hinders* learning exactly where the
task geometry departs from the grid geometry. This is a known failure mode of metric-fixed
distributed codes, and it is the concrete observation a temporally-appropriate
representation has to fix.

## Baselines

The prior approaches a temporally-appropriate representation would be measured against, and
reacts to:

- **Punctate (tabular / one-hot) representation.** One feature dimension per state, active
  only in that state. With a linear value learner this is just a lookup table. Core
  limitation: *no generalization whatsoever* between states — every state's value must be
  learned from its own visits, so learning is as slow as possible and a barrier or a goal
  move teaches nothing about neighboring states.

- **CMAC / coarse coding (Albus 1975; Watkins 1989).** Overlapping receptive fields tiled
  over the input space; a location activates the fields covering it, modulated like a radial
  basis function by distance to each field's center. Generalizes between spatially-near
  states. Core limitation: the generalization is governed by a *fixed a-priori metric*
  (here, grid distance). Where the task metric and the input metric disagree — across a
  barrier — it generalizes between states that should have very different values, and
  actively slows learning.

- **General hidden-feature learning for TD (Anderson 1986).** A multilayer
  backpropagation-through-a-TD-network that learns its own hidden representation tuned to
  predict TD targets. Core limitation: it is a *completely general* technique that makes no
  explicit reference to what actually makes a representation good for a temporal task — it
  has to discover the structure blindly, with all the slowness and opacity that implies.

- **Tree / memory-based function approximators (Moore 1990, kd-trees; Chapman & Kaelbling
  1991, decision trees over binary input predicates).** Sharper function approximation that
  preserves stored values and, for kd-trees, the convergence guarantees of Q-learning. Core
  limitation: like CMAC, their quality rests on an a-priori metric or on the input
  predicates being apt; the representation is not malleable to the task's own structure.

- **Learning a full world model (Sutton 1990, Dyna; Thrun, Möller & Linden 1991).** Learn
  the complete transition matrix (or the state-action-to-next-state map) and use it to plan,
  to learn while disconnected from the world, or to evaluate projected action sequences.
  This is maximally flexible — a veridical, goal-independent map; a local change to the
  dynamics is a local change to the model. Core limitation: a full model is the most
  expensive thing to learn and to compute with, and the value still has to be derived from
  it by iteration over the whole state space each time something changes.

- **Recurrent world-model net (Sutton & Pinette 1985).** A recurrent network whose
  activations, driven by a one-hot current state through the transition matrix, settle to
  exactly the discounted future-occupancy vector of that state; they augment the matrix so
  that the recurrence directly predicts future return. Core limitation: the predictions are
  produced by an iterated recurrence and so are *very sensitive to errors* in the estimated
  transition matrix, and the recurrence complicates any convergence analysis.

The gap all of these leave open: a representation whose generalization is *temporal* and
*task-derived* (so it bends around a barrier on its own), that is *learnable from
experience* by the same TD machinery already in hand (no full model, no opaque hidden
layer), and that *separates* the part of value that depends on dynamics from the part that
depends on reward.

## Evaluation settings

The natural yardstick is the grid-navigation task that had already become the canonical demo
of TD-based control (Watkins 1989; Barto, Sutton & Watkins 1989): a 2-D grid with walls and
an internal barrier, four-directional moves, a goal cell, a step penalty of −1 per move that
does not reach the goal, and future steps discounted geometrically. A policy maps grid cells
to move directions; a TD algorithm estimates each cell's distance-to-goal under that policy;
the policy is improved by an asynchronous, policy-iteration-style update that makes
better-than-expected actions more likely. Representations are compared on speed of learning —
how quickly value estimates (and hence the induced policy) become good — across the grid,
including in regimes where the agent is first allowed to wander the maze with no goal present
(a latent-learning phase) before the goal is switched on. The relevant probes are: how does a
representation behave near the barrier; does it support latent learning; and what happens when
the goal location is moved after learning.

## Code framework

A bare linear-TD value learner over an abstract finite environment needs a feature matrix
before it can generalize across states.

```python
import numpy as np

class Environment:
    """A finite Markov environment."""
    n_states: int
    def reset(self): ...
    def step(self, s):
        # returns (s_next, reward, done)
        ...

def build_features(env, gamma, alpha, n_steps):
    """Return a matrix F whose row s is the feature vector for state s."""
    # TODO: choose or learn the state features
    raise NotImplementedError

def features(F, s):
    return F[s]

def td0_value_update(w, x_t, x_tp1, reward, gamma, alpha, done):
    """Standard linear TD(0) on value: nudge w along x_t by the one-step error."""
    v_t = w @ x_t
    target = reward if done else reward + gamma * (w @ x_tp1)
    w += alpha * (target - v_t) * x_t
    return w

def train(env, gamma, alpha, n_steps):
    F = build_features(env, gamma, alpha, n_steps)
    w = np.zeros(F.shape[1])     # weights for the linear value approximator
    s = env.reset()
    x_t = features(F, s)
    for _ in range(n_steps):
        s_next, reward, done = env.step(s)
        x_tp1 = np.zeros_like(x_t) if done else features(F, s_next)
        w = td0_value_update(w, x_t, x_tp1, reward, gamma, alpha, done)
        if done:
            s = env.reset()
            x_t = features(F, s)
        else:
            s, x_t = s_next, x_tp1
    return w
```
