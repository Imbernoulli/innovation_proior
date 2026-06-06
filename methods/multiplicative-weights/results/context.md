## Research question

We face a sequence of decisions made under adversarial uncertainty. There are `n` available
decisions (call them "experts," "actions," or "constraints," depending on the application). The
process runs for `T` rounds. In each round `t` we must commit to one decision — or, more generally,
to a probability distribution `p^(t)` over the `n` decisions from which we draw — *before* seeing what
it will cost. Only after we commit does nature, possibly an adversary with full knowledge of our
distribution, reveal a cost vector `m^(t)`, where `m_i^(t)` is the cost the `i`-th decision would have
incurred this round. We pay the cost of the decision we drew, in expectation `m^(t) · p^(t)`.

The adversary is unconstrained except that costs are bounded; we assume `m_i^(t) ∈ [-1, 1]`. It may
choose the costs adaptively, even with knowledge of our distribution. We cannot hope to compete with
the best decision *per round* (that requires clairvoyance). The achievable goal is to compete with the
best *single fixed* decision in hindsight: keep our total cost `Σ_t m^(t) · p^(t)` close to
`min_i Σ_t m_i^(t)`, the cost of the one decision that, looking back, was best all along. The gap
between these two quantities is the *regret*. We want the regret to grow sublinearly in `T`, so that
the per-round regret tends to zero — we learn to do as well as the best fixed decision without knowing
in advance which one it is.

The defining difficulty is committing before seeing costs against an adversary, with only a bounded
cost assumption to lean on. A second, larger goal sits behind the first: many seemingly unrelated
computational problems — approximately solving a linear program, finding the value of a zero-sum game,
combining weak classifiers into a strong one, covering a universe with few sets — can be *cast* as such
a repeated decision problem, so a good regret-minimizing rule becomes a general-purpose solver.

## Background

**The prediction-with-expert-advice setting.** The cleanest instance is binary prediction. Each day we
predict a binary event (a stock goes up or down); a wrong prediction costs us, a correct one costs
nothing. The event sequence may be arbitrary and adversarial. To make the problem tractable we are
allowed to watch `n` "experts," each of whom also predicts each day; the experts may be arbitrarily
correlated and may know nothing. The goal is to make nearly as few mistakes as the best expert in
hindsight. Two naive strategies fail. Going with the *majority* opinion of the experts fails because a
majority can be wrong every single day. Picking an expert uniformly at random pays the *average*
expert's cost and never improves. The lesson is that we must *reweight* the experts over time, raising
the influence of those who have done well — but reweighting alone is not enough; *how* we reweight and
whether we randomize turn out to control the achievable regret.

**Bounded costs and the value of randomization.** Against an adversary, any deterministic rule that
outputs a single prediction can be made to err every round (the adversary sets the cost opposite to our
commitment). This is why the achievable guarantees split sharply between deterministic rules (which can
only stay within a *factor of two* of the best expert) and randomized rules (which can approach a
factor of one). Randomization, plus the linearity of expected cost in the played distribution
`p`, is what makes a clean potential-function analysis possible.

**The exponential / potential-function style of analysis.** A recurring technical motif across these
fields is to track a single scalar "potential" — a sum or a relative-entropy — and show it both
*shrinks fast* when the algorithm performs well relative to the best decision and *cannot shrink below*
a floor set by the best decision's cumulative cost. Sandwiching the potential between these two bounds
yields the regret guarantee. The key elementary facts used are `1 + x ≤ e^x`, the convexity
inequalities `(1-η)^x ≤ 1 - ηx` for `x ∈ [0,1]` and `(1+η)^{-x} ≤ 1 - ηx` for `x ∈ [-1,0]`, and the
logarithm estimates `ln(1/(1-η)) ≤ η + η²`, `ln(1+η) ≥ η - η²` valid for `η ≤ 1/2`.

**The meta-structure of the applications.** Each application instantiates the repeated game with a
problem-specific "decision," a problem-specific cost vector, and a problem-specific subroutine — an
*oracle* — that, given the current distribution `p^(t)` over decisions, returns a single response (a
best-response strategy, a feasible point, a chosen set, a weak classifier). The regret-minimizing rule
on top of the oracle is generic; the cleverness in each application is the encoding of the problem as
costs and the design of the oracle. The width of the costs — the largest absolute value `ρ` any cost
can take before normalization into `[-1,1]` — governs how many rounds are needed.

## Baselines

**Weighted Majority (Littlestone & Warmuth, 1994).** The first clean answer to the experts problem in
the deterministic mistake-bound model. Maintain a weight `w_i` per expert, initialized to 1. Predict
the weighted-majority opinion. Whenever an expert errs, scale its weight by a factor `(1 - η)` with
`0 < η ≤ 1/2`. The analysis tracks `Φ = Σ_i w_i`: every time the algorithm errs, at least half the
total weight sat on the wrong side, so `Φ` drops by a factor of at least `(1 - η/2)`; meanwhile the best
expert's own weight `(1-η)^{m_i}` is a lower bound on `Φ`. Combining gives a mistake bound of the form
`M ≤ 2(1+η) m_i + 2 ln(n)/η`. The factor of 2 is intrinsic to *any* deterministic rule and is the gap
this baseline leaves open: a deterministic predictor can be forced to make twice the best expert's
mistakes.

**Fictitious play and best-response dynamics (Brown, 1951; and the 1950s game-theory line).** In a
repeated zero-sum game each player tracks the empirical frequency of the opponent's past plays and plays
a pure best response to that empirical distribution. This converges to the value of the game in the
limit in various cases, but it is a *hard* best response — it commits fully to one action — and its
convergence is slow and delicate. The gap it leaves: no quantitative, polynomial iteration bound, and
brittleness from the all-or-nothing best response.

**The Winnow algorithm (Littlestone, 1988).** An early multiplicative-update rule for learning a linear
classifier / threshold function, tolerant of many irrelevant attributes. It demonstrated that
multiplicatively scaling per-feature weights yields strong mistake bounds, but it was presented as a
special-purpose learner rather than as an instance of a general regret-minimizing meta-algorithm.

**Lagrangian relaxation for packing/covering LPs (Plotkin, Shmoys & Tardos, 1995).** To approximately
solve a feasibility LP `Ax ≥ b` over an easy convex set `P`, dualize the hard constraints into a single
weighted constraint `p^T A x ≥ p^T b` and iterate, adjusting the multipliers `p` toward the violated
constraints. This is a quantitative Lagrangian-multiplier method; its running time depends on a *width*
parameter `ρ`, the largest constraint violation. The gap: the iteration count scales with `ρ²`, which
can be large, motivating later width-reduction ideas.

**Hannan's "follow the perturbed leader" (Hannan, 1957).** Pick the decision minimizing the total cost
so far *plus a random perturbation* `r_i` per decision. With carefully chosen perturbations this attains
regret comparable to reweighting schemes; it is an alternative route to low regret, related but distinct
in form (it leads, rather than maintaining an explicit distribution).

## Evaluation settings

The natural yardsticks are the problems the meta-algorithm would be instantiated on, together with the
quantities one would measure. *Prediction from expert advice:* `n` experts, an arbitrary/adversarial
event sequence of length `T`, measured by cumulative mistakes (or cumulative `[-1,1]` cost) of the
algorithm versus the best expert in hindsight — i.e. the regret. *Zero-sum games:* a payoff matrix `A`
with entries in `[0,1]` and `n` rows; one measures the additive gap between the algorithm's value and
the game value `λ*` as a function of the number of oracle (best-response) calls. *Packing/covering and
general feasibility LPs:* a matrix `A`, a vector `b`, an easy convex domain `P`, an error parameter `ε`,
and a width `ρ`; one measures the number of oracle calls to reach an `ε`-approximately feasible point,
as a function of `m` (constraints), `ρ`, and `ε`. *Set cover:* a universe of `n` elements and a family
of sets; the measure is the size of the cover returned relative to the optimum `OPT`. *Boosting:* a
training set of `N` samples and access to a `γ`-weak learner; the measure is the training error of the
combined hypothesis as a function of the number of boosting rounds. In every case the controlled
parameters are `n` (or `m`), `T` (or the round count), the cost range `[-ρ,ρ]`, and the target accuracy
`ε`.

## Code framework

The primitives that already exist: arrays of nonnegative floats, sampling an index proportional to a
list of weights, and numerical linear algebra (`numpy`) for the LP application. What does not yet exist
is the update rule that turns observed costs into a new weighting, and the encodings that cast each
target problem as a repeated decision game. The scaffold below lays out the generic harness and leaves
empty slots for exactly those pieces.

```python
import random
import numpy

# draw an index proportional to the given list of nonnegative weights
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1

# The repeated-decision harness. Each round: sample a decision from the current
# weighting, observe an outcome, collect a reward/cost vector, and revise the weights.
def repeated_decisions(objects, observeOutcome, reward, learningRate, numRounds):
    weights = [1] * len(objects)
    for t in range(numRounds):
        chosenObjectIndex = draw(weights)
        chosenObject = objects[chosenObjectIndex]
        outcome = observeOutcome(t, weights, chosenObject)
        for i in range(len(weights)):
            pass  # TODO: the weight-revision rule — the contribution to be designed
    return weights

# --- Casting a linear program as a repeated decision game ---
# Solve  min c.x  s.t.  Ax >= b, x >= 0.
# Need: an ORACLE that, given a weighting over the constraints, returns a single
# feasible point satisfying the weighted-average constraint; and a reward encoding
# of "how violated is constraint i".

def makeOracle(c, optimalValue):
    pass  # TODO: best single point against an averaged constraint

def solveGivenOptimalValue(A, b, linearObjective, optimalValue, learningRate):
    pass  # TODO: encode constraints as decisions, run the harness, average the points

def solve(A, b, linearObjective, maxRange=1000):
    pass  # TODO: binary search over the optimal value of c.x
```
