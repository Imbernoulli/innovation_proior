# Context: gradient-free credit assignment in stochastic connectionist networks

## Research question

A connectionist network is built from individual units, each of which fires stochastically: a
unit reads a vector of inputs, computes a parameter (a firing probability, a mean, …) from its
weights, and then *draws* its output from a distribution governed by that parameter. The whole
network propagates these random activities forward and produces an output. The environment then
returns a single scalar **reinforcement** `r` — a payoff judging the most recent input–output
pair — and broadcasts it to every unit. That is all the feedback there is: no target output, no
per-unit error, no labelled "correct answer", and no knowledge of how `r` depends on what the
network did.

The problem is to find a weight-update rule that makes the network's parameters climb the
**expected reinforcement** `E{r|W}` (the average payoff under the network's own randomness),
subject to three demands that the prevailing supervised-learning machinery cannot meet at once:

1. **Locality.** Each weight must be updated from quantities the owning unit already has — its
   own input, its own output, its own parameters — plus the one broadcast scalar `r`. No unit
   may need to know the rest of the network's wiring.
2. **No model of the environment.** The map from (input, output) to `r` is unknown and may be
   non-differentiable, noisy, or even adversarial. The rule may not require differentiating
   through it or storing an estimate of it.
3. **A correctness guarantee.** The rule should provably move `W` in an ascent direction for
   `E{r|W}` — ideally its *average* step should be the true gradient — so that whatever it does,
   it is doing hill-climbing on the right surface, not merely a plausible-looking heuristic.

The pain point is that the gradient of `E{r|W}` involves the environment's response, which is
exactly what is unavailable. A rule has to climb a gradient it can never write down.

## Background

**Backpropagation and why it does not apply here.** By the mid-1980s the dominant learning
engine for connectionist networks was error backpropagation (Werbos 1974; Parker 1985; leCun
1985; Rumelhart, Hinton & Williams 1986): given a differentiable error between the network's
output and a *target*, propagate partial derivatives backward through the deterministic units by
the chain rule, and descend. Backprop is a computational implementation of the chain rule and
nothing more. It needs (a) a differentiable error signal and (b) deterministic, differentiable
units so the chain of effects is traceable. In a reinforcement task neither holds: there is no
target to form an error from, the search-driving randomness sits *inside* the units, and the
payoff comes from an unknown environment. The chain of effects from a weight to `r` runs through
a coin flip and then through a black box.

**Stochastic units as the search mechanism.** Exploration in these networks is supplied by
randomness in the units themselves rather than by an external schedule. The canonical building
block is the *Bernoulli semilinear unit*: it computes `s = Σ_j w_ij x_j`, squashes it to a
probability `p_i = f(s_i)` (commonly the logistic `f(s)=1/(1+e^{-s})`), and emits `y_i ∈ {0,1}`
with `Pr{y_i=1}=p_i`. Such units appear in the Boltzmann machine (Hinton & Sejnowski 1986) and
throughout the reinforcement networks of Barto and colleagues (Barto & Anderson 1985; Barto,
Sutton & Anderson 1983; Barto, Sutton & Brouwer 1981). An equivalent picture is a linear
threshold with additive noise `η`: `y_i = 1` iff `Σ_j w_ij x_j + η > 0`, which gives
`Pr{y_i=1} = 1 − Ψ(−s_i)` for noise CDF `Ψ`, i.e. the same thing as a semilinear unit with
squashing function `f = 1 − Ψ(−·)`.

**Continuous stochastic units.** For real-valued actions the natural unit draws its output from a
multiparameter distribution — most usefully a *Gaussian unit* that computes a mean `μ` and a
standard deviation `σ` from its weights and then samples `y ~ N(μ, σ²)`. The appeal is that `μ`
says *where* to explore and `σ` says *how widely*; the two are separately controllable, unlike a
single-parameter Bernoulli unit where one number fixes both the action and the spread.

**Estimating gradients of expectations in simulation.** Outside connectionism, an independent line
in operations research had taken up the problem of computing the gradient of an expectation with
respect to parameters *of the sampling distribution itself* — as opposed to parameters sitting in
an integrand over a fixed distribution (Aleksandrov, Sysoyev & Shemeneva 1968; later the "score
function method" of Rubinstein and the "likelihood ratio method", Glynn 1990). The standing
technical requirements in that literature are differentiability of the density in the parameter and
the validity of interchanging differentiation and integration.

**The temporal credit-assignment problem.** When the network has loops, or the environment
delays its payoff, a single `r` arrives at the *end* of a stretch of activity and must be
apportioned among many earlier decisions. The standard device for reasoning about a recurrent
net over a fixed horizon is *unfolding in time*: duplicate the net once per time step into an
acyclic network whose tied copies `w_ij^t` all equal the shared `w_ij`. This converts a temporal
problem into a (larger) static one, and was the route by which gradient methods reached recurrent
networks.

## Baselines

These are the prior methods a new rule would be measured against and reacts to.

- **Two-action stochastic learning automaton / linear reward-inaction `L_{R-I}`** (Narendra &
  Thathatchar 1989). A single Bernoulli "unit" with no input maintains `p = Pr{y=1}` and, on a
  binary reward, nudges `p` toward the action just taken — but *only when rewarded* (inaction on
  penalty). It provably converges to a single deterministic action with probability 1, but —
  tellingly — with nonzero probability to the *inferior* action: even though the expected motion
  is toward the better action, the absorbing-barrier random walk can be absorbed at the wrong
  end. It is also nonassociative: one global `p`, no dependence on an input pattern.

- **Associative reward-penalty `A_{R-P}`** (Barto & Anandan 1985; Barto 1985; Barto & Jordan
  1987). The associative generalization: a network of Bernoulli-logistic units learning an
  input→output map from reinforcement. For `r ∈ [0,1]` its rule is
  `Δw_ij = α [ r (y_i − p_i) + λ (1 − r)(1 − y_i − p_i) ] x_j`, `0 < λ ≤ 1`. The first term
  reinforces the taken action in proportion to reward; the second, weighted by `λ`, pushes toward
  the *opposite* action in proportion to penalty `(1−r)`. Setting `λ = 0` gives associative
  reward-inaction `A_{R-I}`. Barto & Anandan proved a form of optimal convergence for the
  associative task, and the rule simultaneously generalizes a class of learning automata and
  pattern-classification methods tied to Robbins–Monro stochastic approximation. Its gap: it is a
  specific, somewhat hand-built rule for Bernoulli-logistic units with a penalty term whose role
  and whose extension to other unit types are not derived from a general principle — there is no
  statement that it *follows a gradient*, nor a recipe to produce the analogous rule for, say, a
  Gaussian unit.

- **Reinforcement comparison** (Sutton 1984). Rather than reacting to `r` directly, maintain a
  running prediction `r̄` of upcoming reinforcement (an exponential average
  `r̄(t) = γ r(t−1) + (1−γ) r̄(t−1)`) and drive learning by the *centered* reinforcement
  `(r − r̄)`. Empirically this converges faster and more reliably than uncentered rules,
  especially when `r` is always positive — comparing against the baseline lets the rule tell "good
  for here" from "merely positive". Its gap: `r̄` is introduced as an effective-reinforcement
  heuristic; *why* an arbitrary running prediction may be subtracted without corrupting what is
  being optimized is not established, and the analysis offers no principled way to choose it.

- **Stochastic approximation / Robbins–Monro.** The broad framework for driving a parameter to a
  root or optimum of an unknown function from noisy samples, with vanishing or small step sizes.
  It supplies the asymptotic-convergence language and underlies the automata and `A_{R-P}` lines,
  but on its own it does not say which *direction* a single update should take in a multi-weight
  associative network.

- **Backprop through a learned model** (Munro 1987; Jordan & Rumelhart 1990; Werbos). Learn an
  explicit differentiable model of the reinforcement-as-function-of-(input,output), then backprop
  through that surrogate to get a gradient. This is *model-based / indirect*: it works only as
  well as the learned model, and it re-introduces exactly the modelling and differentiation
  burden the locality/no-model demands were trying to avoid.

## Evaluation settings

The natural yardsticks at the time were small, controlled reinforcement tasks where the correct
behavior is known so that learning can be judged:

- **Nonassociative immediate-reinforcement (bandit-like) tasks** with a single Bernoulli unit or
  a single Gaussian unit — choose an action / a real value to maximize a fixed payoff function of
  the output. Sutton (1984) used single-Bernoulli "networks" on exactly these to compare automata
  and reinforcement-comparison rules.
- **Associative immediate-reinforcement tasks** — a network must learn an input→output mapping
  from a payoff that depends on the current input–output pair only; the setting for `A_{R-P}`.
- **Function optimization** posed as nonassociative reinforcement — networks of Bernoulli units
  searching a discrete space (Williams & Peng 1991).
- **Episodic / delayed-reinforcement tasks** — recurrent or multilayer networks that must learn a
  trajectory or solve a control problem (cart-pole-style) while receiving reinforcement only at
  the end of an episode.

Metrics are convergence speed and reliability (probability of reaching the better action /
optimum), and whether the average parameter motion lies in an ascent direction for expected
reinforcement.

## Code framework

The primitives that already exist are: a stochastic unit that turns weights and input into a
distribution parameter and *samples* an output; a forward pass that runs the whole network and
collects an action; an environment that returns a scalar `r`; and a trial/episode loop. What does
*not* yet exist is the learning rule — how a unit turns `(its own input, its own sampled output,
the broadcast r)` into a weight change. That is the single empty slot below.

```python
import numpy as np

class StochasticUnit:
    """A unit that computes a distribution parameter from w·x and samples its output."""
    def __init__(self, n_in):
        self.w = np.zeros(n_in)

    def param(self, x):
        # distribution parameter (e.g. a firing probability, or a mean) from weights and input
        raise NotImplementedError

    def sample(self, x):
        # draw the stochastic output y given the parameter
        raise NotImplementedError

    def learning_rule(self, x, y, r):
        # TODO: turn (own input x, own sampled output y, broadcast scalar reward r)
        #       into a weight increment Δw — using only local quantities and r.
        #       This is the slot the method will fill.
        raise NotImplementedError


def run_trial(units, x_env, environment):
    # forward pass: every unit samples; assemble the network output
    ys = [u.sample(x_env) for u in units]
    r = environment.reward(x_env, ys)        # scalar reinforcement, broadcast to all units
    for u, y in zip(units, ys):
        u.w += u.learning_rule(x_env, y, r)  # local update from r
    return r


def run_episode(units, environment, k):
    # temporal case: k steps of activity, one reward at the end; per-weight accumulator
    local_data = [np.zeros_like(u.w) for u in units]
    r = None
    for t in range(k):
        x_t = environment.observe()
        ys = [u.sample(x_t) for u in units]
        for i, (u, y) in enumerate(zip(units, ys)):
            local_data[i] += np.zeros_like(u.w)  # TODO: accumulate whatever local per-step data is needed
        r = environment.step_reward(ys)
    for i, u in enumerate(units):
        u.w += np.zeros_like(u.w)            # TODO: turn final reward and local data into Δw
    return r
```
