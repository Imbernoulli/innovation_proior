# Multiplicative Weights Update (MWU)

## Problem

A decision maker faces `n` decisions and must act for `T` rounds. Each round it commits to a
distribution `p^(t)` over the decisions and draws one *before* seeing the costs. Then an adversary —
who may know `p^(t)` — reveals a cost vector `m^(t) ∈ [-1,1]^n`, and the decision maker pays
`m^(t) · p^(t)` in expectation. It cannot beat the best decision *per round* (that needs
clairvoyance), so it aims for the best *single fixed* decision in hindsight, `min_i Σ_t m_i^(t)`. The
**regret** is the gap between its cumulative cost and that benchmark; the goal is regret sublinear in
`T`, so per-round regret vanishes.

## Key idea

Keep one nonnegative weight `w_i` per decision (all initialized to 1). Each round, play
proportional to weights (`p_i^(t) = w_i^(t)/Φ^(t)`, with `Φ^(t) = Σ_i w_i^(t)`), and after seeing
the costs, **multiply** each weight by a factor that shrinks with that decision's cost:
`w_i^(t+1) = w_i^(t)(1 − η m_i^(t))`. Multiplicative (not additive) updates make a chronically bad
decision's weight decay geometrically — `w_i = Π_τ (1 − η m_i^(τ))`, which in the pure-mistakes case
is `(1−η)^{#mistakes}` — so the best decision's weight quickly dominates the sum. Playing
*randomly* in proportion to weights (rather than deterministically) makes the expected cost linear in
`p`, which is what makes the analysis clean and removes the factor-2 penalty intrinsic to any
deterministic rule.

This one rule is a **meta-algorithm**: by choosing a problem-specific cost encoding and a
best-response *oracle*, it approximately solves linear programs, computes the value of a zero-sum
game (a constructive minimax proof), reduces to greedy set cover at `η = 1`, and is exactly AdaBoost
when the "decisions" are training examples.

## The algorithm

```
Initialize η ≤ 1/2 ;  w_i^(1) = 1 for all i.
For t = 1, …, T:
  1. p^(t) = w^(t) / Φ^(t),  Φ^(t) = Σ_i w_i^(t);  sample decision i ~ p^(t).
  2. Observe cost vector m^(t) ∈ [-1,1]^n.
  3. w_i^(t+1) = w_i^(t) (1 − η m_i^(t))  for all i.
```

## Regret guarantee

For all costs in `[-1,1]`, `η ≤ 1/2`, and **any** decision `i`:

    Σ_t m^(t)·p^(t)  ≤  Σ_t m_i^(t)  +  η Σ_t |m_i^(t)|  +  (ln n)/η.

*Proof.* Track the potential `Φ^(t) = Σ_i w_i^(t)`.
Upper bound: `Φ^(t+1) = Σ_i w_i^(t)(1 − η m_i^(t)) = Φ^(t)(1 − η m^(t)·p^(t)) ≤ Φ^(t) e^{−η m^(t)·p^(t)}`
(using `p_i = w_i/Φ` and `1 + x ≤ e^x`), so `Φ^(T+1) ≤ n·e^{−η Σ_t m^(t)·p^(t)}`.
Lower bound: `Φ^(T+1) ≥ w_i^(T+1) = Π_τ (1 − η m_i^(τ)) ≥ (1−η)^{Σ_{≥0} m_i} (1+η)^{−Σ_{<0} m_i}`
(using `(1−η)^x ≤ 1−ηx` on `[0,1]` and `(1+η)^{−x} ≤ 1−ηx` on `[-1,0]`). Taking logs and isolating
the algorithm's cost gives
`Σ_t m^(t)·p^(t) ≤ (ln n)/η + Σ_{≥0} m_i^(t) ln(1/(1−η))/η + Σ_{<0} m_i^(t) ln(1+η)/η`. Use
`ln(1/(1−η)) ≤ η + η²` and `ln(1+η) ≥ η − η²`; the second estimate multiplies the negative sum
`Σ_{<0} m_i`, giving the displayed regret bound. ∎

With `|m_i| ≤ 1`, `Σ_t|m_i^(t)| ≤ T`, so regret `≤ ηT + (ln n)/η`, minimized at
`η = √((ln n)/T)`, giving regret `≤ 2√(T ln n)` — sublinear, so average regret `→ 0`.

**Hedge variant** (exponential update `w_i^(t+1) = w_i^(t) e^{−η m_i^(t)}`, valid `η ≤ 1`):
`Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t (m^(t))²·p^(t) + (ln n)/η` — here the `η`-penalty depends on
the algorithm's own second moment rather than the best decision's loss.

**Gains form** (run on `−m`, so `w_i ← w_i (1 + η m_i)` with `m_i` a gain in `[-1,1]`):
`Σ_t m^(t)·p^(t) ≥ Σ_t m_i^(t) − η Σ_t |m_i^(t)| − (ln n)/η`.

**Deterministic weighted majority.** Predicting the weighted-majority label and shrinking wrong
experts by `(1−η)` gives `M ≤ 2(1+η)m_i + 2 ln(n)/η`; the factor 2 is the deterministic penalty that
randomized proportional play removes.

## Applications (instantiations of the same rule)

- **Zero-sum games / minimax.** Rows are decisions; each round an oracle returns the column player's
  best response `j^(t) = argmax_j A(p^(t), j)`, and `m^(t) = A(·, j^(t)) ∈ [0,1]`. The regret bound
  squeezes the time-averaged value between `λ*` and `λ* + η + (ln n)/(ηT)`; with `η = ε/2`,
  `T = ⌈4 ln(n)/ε²⌉` it is within `ε` of the game value in `O(log(n)/ε²)` oracle calls. The average
  strategies are `ε`-optimal — a constructive proof of von Neumann's `min max = max min`.
- **Packing/covering LPs (Lagrangian framework).** The `m` constraints are decisions; cost of
  constraint `i` at point `x` is `(1/ρ)(A_i x − b_i)` (width `ρ` normalizes into `[-1,1]`). An oracle
  finds `x ∈ P` meeting the single averaged constraint `p^T A x ≥ p^T b` (or certifies infeasibility).
  With an `(ℓ,ρ)` bounded oracle, averaging the oracle's points gives an `ε`-feasible solution in
  `O(ℓρ log(m)/ε²)` calls; when `ℓ = ρ`, the width appears quadratically.
- **Set cover (η = 1).** Elements are decisions; covered elements drop to weight 0, so `p^(t)` is
  uniform over uncovered elements and "maximize cost" = greedy "cover the most uncovered." Yields the
  `ln n`-approximation.
- **Boosting / AdaBoost.** Training examples are decisions; cost `m_x = 1` if the round's weak
  hypothesis labels `x` correctly, else `0`. The weak-learning guarantee gives `m^(t)·p^(t) ≥ 1/2+γ`;
  weight drifts onto misclassified examples; the majority vote of the weak hypotheses has training
  error `≤ ε` after `T = ⌈(2/γ²) ln(1/ε)⌉` rounds.

## Code

A runnable implementation. The generic harness collects *gains* (penalizing cost is the same rule on
`−gain`); the LP solver casts `min c·x s.t. Ax ≥ b, x ≥ 0` as a constraints-as-experts game with an
oracle that satisfies one averaged constraint, then binary-searches the optimal objective value.

```python
import random
import numpy


# draw an index proportional to nonnegative weights: realizes p_i = w_i / sum(w).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1


# Multiplicative Weights Update, gains form.
# Each round: sample i ~ p^(t) = w^(t)/Phi^(t), observe the outcome the world reveals,
# then multiply every weight by (1 + eta * gain_i) -- geometric reward/decay, not additive.
def MWUA(objects, observeOutcome, reward, learningRate, numRounds):
    weights = [1] * len(objects)            # Phi^(1) = n: maximum-entropy start
    cumulativeReward = 0
    outcomes = []
    for t in range(numRounds):
        assert all(w >= 0 for w in weights)
        chosenObjectIndex = draw(weights)
        chosenObject = objects[chosenObjectIndex]
        outcome = observeOutcome(t, weights, chosenObject)
        outcomes.append(outcome)
        cumulativeReward += reward(chosenObject, outcome)
        for i in range(len(weights)):
            weights[i] *= (1 + learningRate * reward(objects[i], outcome))
    return weights, cumulativeReward, outcomes


class InfeasibleException(Exception):
    pass


# Oracle for  min c.x  s.t. Ax >= b, x >= 0, given a guessed value of c.x.
# Given a weighting over constraints, return one point x in {x>=0, c.x = optimalValue}
# satisfying the averaged constraint (A^T w).x >= w.b; if none exists, w certifies
# infeasibility for this guessed value.
def makeOracle(c, optimalValue):
    n = len(c)

    def oracle(weightedVector, weightedThreshold):
        def quantity(i):
            return weightedVector[i] * optimalValue / c[i] if c[i] > 0 else -1
        biggest = max(range(n), key=quantity)            # put all budget on the best coordinate
        if quantity(biggest) < weightedThreshold:
            raise InfeasibleException
        return numpy.array(
            [optimalValue / c[i] if i == biggest else 0 for i in range(n)]
        )
    return oracle


def solveGivenOptimalValue(A, b, linearObjective, optimalValue, learningRate=0.1):
    m, n = A.shape
    oracle = makeOracle(linearObjective, optimalValue)

    # A violated constraint has positive reward, so weight concentrates on hard constraints.
    def reward(i, x):
        return b[i] - numpy.dot(A[i], x)

    def observeOutcome(_, weights, __):
        weights = numpy.array(weights)
        return oracle(A.transpose().dot(weights), weights.dot(b))  # solve the averaged constraint

    numRounds = 1000
    _, _, outcomes = MWUA(range(m), observeOutcome, reward, learningRate, numRounds)
    return sum(outcomes) / numRounds        # x_bar = (1/T) sum_t x^(t), in the convex domain


# Reduce optimization to feasibility by binary-searching the optimal value of c.x.
def solve(A, b, linearObjective, maxRange=1000):
    optRange = [0, maxRange]
    result = None
    while optRange[1] - optRange[0] > 1e-8:
        proposedOpt = sum(optRange) / 2
        learningRate = min(1 / max(2 * proposedOpt * ci for ci in linearObjective), 0.1)
        try:
            result = solveGivenOptimalValue(A, b, linearObjective, proposedOpt, learningRate)
            optRange[1] = proposedOpt        # feasible: lower the target
        except InfeasibleException:
            optRange[0] = proposedOpt        # infeasible: raise the target
    return result


if __name__ == "__main__":
    A = numpy.array([[1, 2, 3], [0, 4, 2]])
    b = numpy.array([5, 6])
    c = numpy.array([1, 2, 1])
    x = solve(A, b, c)
    print(x)             # eps-approximately optimal x
    print(c.dot(x))      # objective value
    print(A.dot(x) - b)  # constraint slack (>= ~0)
```
