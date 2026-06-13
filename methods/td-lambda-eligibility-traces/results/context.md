# Context: learning long-horizon predictions online, between TD and Monte Carlo

## Research question

We have an agent moving through a sequence of states under a fixed policy π, receiving
rewards along the way, and we want to learn the value function v_π(s) — the expected
discounted return from each state. The two ways to do this that we already have sit at
opposite extremes, and each is unsatisfying.

One-step temporal-difference learning updates a state's value from the very next reward
plus the (discounted) estimated value of the next state. It is online and cheap, but it
only ever moves information one step at a time: a reward observed far down a trajectory
takes many episodes to seep back to the states that led to it.

Monte Carlo learning waits for the whole episode, then updates each visited state toward
the actual return. It propagates credit all the way back in one shot and makes no
bootstrapping error, but it cannot update until the episode ends, has high variance, and
is usable as stated only when episodes terminate.

The precise problem: find a single mechanism, controlled by one knob, that interpolates
smoothly between these extremes — so we can pick how far into the future an update reaches
— and that does so **online, causally, and with memory and per-step compute that do not
grow with how far back credit must travel**. The pain is concrete: any scheme that updates
a state from the next n rewards must wait n steps and store the last n feature vectors, and
choosing the single right n is a task-specific guess. We want the interpolation without the
storage, the delay, or the brittle choice of horizon.

## Background

**The prediction setting and the return.** Under policy π we see S_0, R_1, S_1, R_2, …,
R_T, S_T. The return is G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + … + γ^{T−t−1}R_T, with
discount γ ∈ [0,1]. The value function v_π(s) = E_π[G_t | S_t = s] satisfies the Bellman
relation v_π(s) = E_π[R_{t+1} + γv_π(S_{t+1}) | S_t = s]. We approximate it with a
parameterized v̂(s,w); the linear case v̂(s,w) = w^T x(s), with feature vector x(s) and
∇v̂ = x(s), includes the tabular case (x(s) a one-hot indicator) as a special case.

**One-step TD (TD(0)).** Given a transition S_t → R_{t+1}, S_{t+1}, form the TD error
δ_t = R_{t+1} + γv̂(S_{t+1},w) − v̂(S_t,w) and step w ← w + αδ_t ∇v̂(S_t,w). The next
estimate stands in for the unseen rest of the return (bootstrapping). Cheap and online;
low variance; but biased by the current estimate, and information crawls back one step
per visit.

**Monte Carlo / Widrow-Hoff (LMS).** Wait for G_t, then w ← w + α[G_t − v̂(S_t,w)]∇v̂.
For the linear case with a single terminal reward this is exactly the Least-Mean-Square /
Widrow-Hoff delta rule. Unbiased target, but available only at episode end, high variance,
usable as stated only when episodes terminate.

**n-step returns and their error-reduction property.** Between these lies the n-step
return, which bootstraps after n real rewards instead of one:

  G_{t:t+n} = R_{t+1} + γR_{t+2} + … + γ^{n−1}R_{t+n} + γ^n v̂(S_{t+n}),  with G_{t:t+n}=G_t if t+n≥T.

Its expectation is a strictly better estimate of v_π than the current value, in a
worst-state sense — the **error-reduction property**:

  max_s |E_π[G_{t:t+n} | S_t=s] − v_π(s)| ≤ γ^n · max_s |v̂(s) − v_π(s)|.

Because the bound shrinks by γ^n, every n-step return is a sound (contraction) target, and
n-step TD methods converge under the usual conditions. This is the load-bearing fact that
licenses combining returns: any weighted average of n-step returns, with non-negative
weights summing to one, is again a valid contraction target — a "compound" target. That
opens a whole design space of mixtures between one-step TD and Monte Carlo. The cost of the
n-step method is structural: no update can be made for the first n−1 steps, an update for
S_t is delayed until time t+n, and the implementation must hold the last n feature vectors
in memory. And there is no principled way to pick n.

**The eligibility idea (Klopf).** A separate thread, biological in origin, exists in the
literature. Klopf's heterostatic theory
(A. H. Klopf, *Brain Function and Adaptive Systems — A Heterostatic Theory*, AFCRL-72-0164,
1972; and his 1982 book) introduced the word *eligibility*: when a synapse is active it
becomes transiently **eligible** for modification, leaving a decaying trace of its recent
activity; a later reinforcing signal then modifies every synapse that is currently
eligible, in proportion to its trace. Credit for an outcome is spread back over the
recently active connections, fading with how long ago each was active. This is a local,
online memory of "what was recently responsible," developed in a biological setting and not
connected to the prediction-error methods above.

**Prior work on a TD–Monte-Carlo family (Sutton 1988).** Sutton's *Learning to Predict
by the Methods of Temporal Differences* (1988) studied a parameterized family of prediction
rules spanning the range between one-step bootstrapping and the Widrow-Hoff/Monte-Carlo
rule, controlled by a single recency parameter, and analyzed it in the undiscounted
prediction setting. What remained open is the discounted formulation with γ, a precise
characterization of the forward-view target such a family is implementing, and whether an
online incremental mechanism can reproduce that target.

**Empirical lay of the land.** On the standard random-walk prediction tasks it is well
observed that intermediate amounts of bootstrapping — neither pure one-step TD nor pure
Monte Carlo — give the lowest error, whether the interpolation is controlled by n in
n-step TD or by an exponential recency factor. Pure Monte Carlo (the large-n end) tends to do
poorly; an intermediate setting is best. The pattern motivates wanting a clean, cheap
interpolation knob.

## Baselines

- **One-step TD(0)** — w ← w + α[R_{t+1} + γv̂(S_{t+1}) − v̂(S_t)]∇v̂(S_t). Online, O(d),
  low variance. Gap: propagates credit one step per visit; slow when reward is delayed.

- **Constant-α Monte Carlo / Widrow-Hoff (LMS)** — w ← w + α[G_t − v̂(S_t)]∇v̂(S_t). No
  bootstrap bias, full back-propagation of credit in one episode. Gap: end-of-episode only,
  high variance, only usable as stated when episodes terminate, must retain per-step data
  for the end-of-episode sweep.

- **n-step TD** — target G_{t:t+n}; update w ← w + α[G_{t:t+n} − v̂(S_t)]∇v̂(S_t). Spans
  the continuum by choosing n; backed by the error-reduction property (γ^n bound). Gap:
  updates delayed n steps, last n feature vectors must be stored, and n is a single
  brittle horizon choice rather than a smooth mixture.

- **Compound / averaged-return targets** — any convex combination of n-step returns; valid
  by error-reduction. Gap: a generic average can only be formed once its longest component
  return is available (still acausal), and there is no obvious incremental implementation —
  these are a design space, not yet an algorithm.

## Evaluation settings

The natural yardstick is on-policy prediction of v_π on small Markov-chain tasks where the
true values are known, so error can be measured directly. The canonical case is the
multi-state random walk (a chain of states with terminal absorbing ends, e.g. a 19-state
walk with a reward of +1 at one end), evaluated by the root-mean-square error between the
learned and true state values, averaged over states and over the first several episodes,
swept across the step size α and across the interpolation parameter (n, or the exponential
factor). Small continuous-control tasks with linear features (mountain car with tile
coding, pole balancing, puddle world) provide the control-side yardstick, where the same
interpolation knob is swept. These tasks, metrics, and the α-sweep protocol predate any
particular interpolation mechanism.

## Code framework

```python
import numpy as np

# Linear value function over features x(s); tabular is the one-hot special case.
def v_hat(w, x):           # state value
    return float(w @ x)

def grad_v(w, x):          # gradient wrt w of the linear value is just x
    return x

# Existing prediction updates.
def td0_update(w, x_t, r, x_next, gamma, alpha):
    delta = r + gamma * v_hat(w, x_next) - v_hat(w, x_t)
    return w + alpha * delta * grad_v(w, x_t)

def mc_update(w, x_t, G, alpha):                 # Widrow-Hoff / constant-alpha MC
    return w + alpha * (G - v_hat(w, x_t)) * grad_v(w, x_t)

# The missing online interpolation slot.
# We want ONE mechanism, with one knob in [0,1], that:
#   (a) interpolates between td0_update (knob=0) and mc_update (knob=1),
#   (b) runs online and causally (update at step t using only data through t),
#   (c) uses memory and per-step compute that do NOT grow with the lookahead.
class IncrementalPredictor:
    def __init__(self, d, gamma, alpha, knob):
        self.w = np.zeros(d)
        self.gamma, self.alpha, self.knob = gamma, alpha, knob
        # TODO: any per-step auxiliary state the mechanism needs (must be O(d))
        self.reset_episode()

    def reset_episode(self):
        # TODO: re-initialize any per-episode auxiliary state
        pass

    def step(self, x_t, r, x_next):
        # TODO: the online credit-assignment update satisfying (a)-(c) above.
        #       Returns nothing; mutates self.w.
        raise NotImplementedError

# Training loop is the usual on-policy sweep; only step() is unknown.
def run_episode(agent, episode):                 # episode: list of (x_t, r, x_next)
    agent.reset_episode()
    for (x_t, r, x_next) in episode:
        agent.step(x_t, r, x_next)
```
