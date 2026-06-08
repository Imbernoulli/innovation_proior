# Context

## Research question

A decision maker faces the *same* decision problem over and over, but the payoff
rule changes every round and is revealed only *after* the decision is committed.
Concretely: a feasible set of decisions is fixed and known in advance; on each
round one must commit to a point in that set, and only then is the round's cost
function disclosed. The cost functions may have no statistical regularity — they
may be chosen by an adversary, may be unrelated from round to round, and may
drift arbitrarily over time. This models factory or farm production where the
value of goods is unknown until after they are produced, network routing where
flows are committed before demands are known, and — most importantly for theory —
repeated games and online prediction, where the "environment" can react to the
learner.

Because the future cost functions are unknown when each decision is made, asking
to *minimize each round's cost* is hopeless: the round's optimum is unknowable in
advance. The achievable goal is comparative: do nearly as well, in total, as the
single best fixed decision one *could* have committed to with hindsight of the
whole cost sequence. The gap between the learner's cumulative cost and that of
the best fixed decision in hindsight is the **regret**; the precise demand is
that *average* regret vanish as the horizon grows, with **no distributional
assumption** on the costs and **no assumption** relating one round's cost to the
next. A solution must say *which* point to play each round, *prove* the regret is
sublinear in the number of rounds, and do so for an *arbitrary* convex feasible
region and an *arbitrary* convex cost — not one fixed loss, not a finite menu of
actions.

## Background

**Convexity is the workable structure.** A set is convex if it contains the
segment between any two of its points; a function is convex if it lies below its
chords, equivalently (when differentiable) above each of its tangent planes:
`f(y) ≥ f(x) + ∇f(x)·(y − x)`. Two facts about convexity are load-bearing here.
First, a convex differentiable function is globally lower-bounded by its
first-order (linear) approximation at any point — the tangent-plane inequality
above. Second, a local minimum of a convex function over a convex set is a global
minimum, and at a constrained minimum `x*` the negative gradient points "out of"
the set: `∇f(x*)·(y − x*) ≥ 0` for every feasible `y` (the constrained optimality
/ KKT condition). A function whose gradients are bounded in norm by `G`
(`‖∇f(x)‖ ≤ G`) is `G`-Lipschitz: `|f(x) − f(y)| ≤ G‖x − y‖`. A bounded convex
set has a finite diameter `D` with `‖x − y‖ ≤ D` for all members.

**Euclidean projection onto a convex set.** For a closed convex set `F` and any
point `y`, there is a unique closest feasible point `P(y) = argmin_{x∈F} ‖x − y‖`.
The single property that makes projection useful in proofs is that it is
*non-expansive* toward any feasible point — the Pythagorean fact that for `x = P(y)`
and any `z ∈ F`, `‖y − z‖ ≥ ‖x − z‖`. Projecting a point back into the set can
only move it *closer* to any point already inside the set, never farther.

**Offline gradient descent.** The oldest method for minimizing a differentiable
convex function: repeatedly step opposite the gradient, `x ← x − η∇f(x)`; with a
constraint, step then project, `x ← P(x − η∇f(x))`. For a `G`-Lipschitz convex
objective over a set of diameter `D`, a suitable diminishing step size yields a
`1/√T` convergence rate in function value. This is a statement about *one fixed*
objective; the gradient and projection primitives, and the habit of tracking
`‖x_t − x*‖²` as a potential, are the pieces lying around.

**No-regret learning and game theory.** In a repeated game with vector-valued
payoffs, Blackwell (1956) asked what a player can guarantee against an
adversary, and his *approachability* theorem — an extension of the minimax
theorem to vector payoffs — gives an adaptive strategy whose average payoff
approaches a target convex set. Hannan (1957) gave a strategy whose average
payoff approaches that of the best fixed action in hindsight ("Hannan
consistency"). These established that one can guarantee good *average* behavior
against an arbitrary adversary, with regret as the currency — but through
problem-specific constructions, not a single optimization principle.

**The experts / weighted-majority line.** In the experts problem one has `n`
experts; each round one picks a distribution over them, then a cost vector
`c^t ∈ ℝⁿ` arrives and the incurred cost is the expectation under the chosen
distribution. The weighted-majority algorithm (Littlestone & Warmuth 1989) and
its randomized generalization Hedge / multiplicative weights (Freund & Schapire
1999) keep one weight per expert, multiply each weight by an exponential of that
expert's cost, and renormalize. Against an *adversarial* cost sequence this
attains regret `O(√(T log n))` versus the best single expert. The decision set
here is the probability simplex — a convex set — and the cost is linear in the
chosen distribution.

**Online regression / gradient and exponentiated-gradient updates.**
Cesa-Bianchi, Long & Warmuth (1994) and Kivinen & Warmuth (1997) analyzed online
linear prediction where each round a gradient-style update is made to a weight
vector and a *relative-loss* bound is proven against the best fixed predictor.
These bounds are derived for a *fixed* loss (e.g. squared error) and are stated
through a fixed Bregman divergence as the potential.

**Gradient dynamics in games.** Singh, Kearns & Mansour (2000) studied players in
a two-player, two-action repeated matrix game who each adjust their mixed
strategy by gradient ascent on expected payoff, projecting the gradient at the
boundary of the unit square, in the limit of infinitesimal step size
("infinitesimal gradient ascent", IGA). They proved that even when the strategies
themselves cycle, the *average* payoffs converge to those of a Nash equilibrium.
The analysis is special to the `2×2` case: it traces the eigenstructure of the
linear gradient dynamics in the unit square and does not extend to more actions.

## Baselines

**Weighted majority / Hedge (multiplicative weights).** Core idea: maintain
weights `w_i` over `n` experts, update `w_i ← w_i · exp(−η c^t_i)`, play the
normalized distribution. Analysis uses the *entropic* potential
`Φ = Σ_i w_i` (a KL / relative-entropy argument). Regret versus the best expert
is `O(√(T log n))`, with no statistical assumption on the costs. **Gap:** it is
tied to a *finite menu of experts* and to the *simplex* with *linear* costs; its
bound scales with `log n` (the number of experts), and there is no version that
handles an arbitrary convex decision region or an arbitrary convex cost. The
potential is problem-specific (entropic), so the proof does not obviously
transfer.

**IGA — infinitesimal gradient ascent (Singh–Kearns–Mansour 2000).** Core idea:
each player plays a mixed strategy `(α, β)` in the unit square; the expected
payoff `V_r(α,β)` is bilinear, so each player ascends its own gradient
`α ← α + η ∂V_r/∂α`, clipping/projecting the gradient at the boundary, with
`η → 0`. **Math/algorithm:** the joint dynamics are linear in `(α,β)`; the
behavior (convergent point, or cycling with convergent averages) is read off the
eigenvalues of the update matrix. **Gap:** proven only for **two players and two
actions**. The argument is the dynamical-systems analysis of a `2×2` linear
system; it gives no algorithm or guarantee for more than two actions, for an
arbitrary convex strategy set, or against an adaptive adversary in general.

**Online gradient / exponentiated-gradient regression (Cesa-Bianchi et al. 1994;
Kivinen & Warmuth 1997).** Core idea: for online prediction with a *fixed* loss,
update the predictor by a (possibly exponentiated) gradient step and bound
cumulative loss relative to the best fixed predictor via a fixed Bregman
divergence. **Gap:** the bounds are stated for a particular loss (squared error
and relatives) and a particular divergence; there is no single statement covering
an *arbitrary* convex loss over an *arbitrary* convex feasible set, and the games
and experts settings are not unified with it.

**Blackwell approachability / Hannan consistency (1956–57).** Core idea: in a
repeated vector-payoff game, an adaptive strategy can steer the average payoff
into a target convex set (approachability) or to the best-fixed-action payoff
(Hannan consistency). **Gap:** these are existence results obtained by
game-specific geometric constructions; they do not present a *single, simple
optimization algorithm* (a gradient method) with an explicit regret rate for
general convex decision problems.

## Evaluation settings

The natural yardstick is **regret as a function of the horizon `T`**: the
learner's cumulative cost `Σ_{t=1}^T c^t(x^t)` minus that of the best fixed
feasible point in hindsight `min_{x∈F} Σ_{t=1}^T c^t(x)`, with the demand that
*average* regret `R(T)/T → 0` (sublinear regret). Performance is also measured by
**per-round running time** (how expensive it is to produce each `x^t`, separating
the cheap gradient step from the possibly-costly projection). The relevant
problem instances are the canonical online settings these baselines live in: the
**experts problem** (decision set = probability simplex, linear costs bounded in
`ℓ∞`), **online linear / convex programming** over a convex polytope or general
convex body (e.g. the online-oblivious network-routing instance, where each round
a flow is committed and an adversary then chooses demands, and the cost is
congestion relative to the best routing in hindsight), and **repeated matrix
games**, where the criterion is *universal consistency* — that average regret
against the best fixed action tend to zero uniformly over *every* environment,
adversarial or not. A second, more demanding comparator is a **moving** target:
the best sequence of feasible points whose total step-to-step movement
(**path length**) is bounded, against which one asks for sublinear *dynamic*
regret.

## Code framework

```python
import numpy as np

# Pre-existing primitives.

def project(y, feasible_set):
    """Euclidean projection P(y) = argmin_{x in F} ||x - y||.
    Closest feasible point; non-expansive toward any point already in F."""
    return feasible_set.project(y)

def gradient(cost_fn, x):
    """Gradient (or a subgradient) of the round's convex cost at the played x."""
    return cost_fn.grad(x)

class FeasibleSet:
    """A fixed, known, closed convex set F: membership + projection + diameter."""
    def project(self, y):       # closest point in F to y
        raise NotImplementedError
    @property
    def diameter(self):         # D = max_{x,y in F} ||x - y||
        raise NotImplementedError


def online_decision_loop(feasible_set, cost_stream, T):
    """Commit a feasible point each round; the round's convex cost is revealed
    only afterward. Adversarial costs: no statistical or inter-round assumption.
    Goal: cumulative cost competitive with the best fixed point in hindsight."""
    x = feasible_set.some_point()          # arbitrary feasible start x^1
    history = []
    for t in range(1, T + 1):
        play(x)                            # commit x^t BEFORE seeing the cost
        c_t = cost_stream.reveal(t, x)     # adversary discloses convex cost c^t
        loss = c_t(x)
        history.append((x, loss))

        # TODO: choose the step size for this round.
        eta_t = choose_step_size(t, feasible_set, T)     # pass

        # TODO: the update rule that turns the revealed cost into the next
        #       feasible point x^{t+1}. This is the slot the method fills.
        x = update(x, c_t, eta_t, feasible_set)          # pass

    return history


def choose_step_size(t, feasible_set, T):
    # TODO
    pass

def update(x, c_t, eta_t, feasible_set):
    # TODO: produce the next feasible point from the current one and c^t.
    pass


def regret(history, feasible_set, cost_stream, T):
    """R(T) = sum_t c^t(x^t) - min_{x in F} sum_t c^t(x).  Want R(T)/T -> 0."""
    online_cost = sum(loss for (_, loss) in history)
    best_fixed = min_over_F(feasible_set, cost_stream, T)   # offline optimum
    return online_cost - best_fixed
```
