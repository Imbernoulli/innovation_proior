A learner faces $n$ possible decisions over $T$ rounds, and the order of play is what gives the problem its teeth: on each round it must commit to a decision, or to a distribution over decisions, *before* an adversary reveals the full cost vector $m(t)$, where $m_i(t)$ is the cost decision $i$ would have paid that round. Costs are bounded but otherwise unconstrained — they can be correlated, adaptive, and chosen after the adversary sees the learner's current distribution. There is no stochastic model to lean on, so no proof can invoke independence or concentration around a hidden mean; everything has to come from boundedness and from the algebra of the learner's own state. Matching the best decision on every round is clairvoyance and is off the table, so the honest benchmark is the best single fixed decision in hindsight, $\min_i \sum_t m_i(t)$, and the goal is to keep the learner's cumulative expected cost within a gap that grows sublinearly in $T$.

The existing baselines all stall at the same place. Predicting with the majority of experts fails outright: a majority can be wrong every round while one quiet minority expert is always right, and uniform random play is no better in the long run because it pays the average expert forever. So memory is needed, and that memory must reshape the distribution over decisions. The deterministic weighted-majority rule adds that memory — it keeps weights on experts and discounts those that have erred — and a total-weight argument proves a mistake bound within roughly a factor of two of the best expert plus a logarithmic term. But the factor two is intrinsic: committing deterministically to the heavier side hands the adversary the majority boundary to exploit. Hedge-style randomized allocation removes the binary-prediction restriction and proves loss bounds through the same total-weight potential, but its standard guarantee charges a second-order term tied to the learner's *own* mixed losses, which leaves the comparator-side error term entangled — exactly the term that reductions needing a clean per-comparator certificate cannot afford to lose.

What I propose is the Multiplicative Weights Update method. The memory is a weight $w_i$ on each decision, initialized to $w_i(1)=1$, and the first design choice is that the weights move *multiplicatively* rather than additively. Additive penalties only ever separate a decision that has failed a thousand times from one that failed once by a difference, so a chronically bad decision stays visible too long; what I want is a ratio effect where repeated failure compounds, and that forces each weight to be a product of per-round factors. The cost-form update is
$$ w_i(t+1) = w_i(t)\,\bigl(1 - \eta\, m_i(t)\bigr), \qquad m_i(t)\in[-1,1],\ \ \eta \le \tfrac12, $$
and the learner plays in proportion to the current weights, $p_i(t) = w_i(t)/\Phi(t)$ with $\Phi(t)=\sum_i w_i(t)$. The proportional, randomized play is not cosmetic — it is the second load-bearing choice. With $\Phi=\sum_i w_i$ and $p_i = w_i/\Phi$, the weighted cost is exactly $\sum_i w_i m_i = \Phi\,(m\cdot p)$, so the learner's expected round cost $m(t)\cdot p(t)$ becomes a *linear* quantity that drops straight into the potential calculation. Committing deterministically to the weighted-majority side destroys this identity, and that destruction is precisely where the factor of two comes from.

The proof tells me the idea is real by squeezing the potential $\Phi(t)$ between two handles. The upper handle uses one step of the update together with $1+x\le e^x$:
$$ \Phi(t+1) = \Phi(t)\bigl(1 - \eta\, m(t)\cdot p(t)\bigr) \le \Phi(t)\,\exp\!\bigl(-\eta\, m(t)\cdot p(t)\bigr), $$
and multiplying over all rounds from $\Phi(1)=n$ gives $\Phi(T+1) \le n\,\exp\!\bigl(-\eta\sum_t m(t)\cdot p(t)\bigr)$. So if the learner's expected cumulative cost is large, the surviving mass must have been driven down. The lower handle knows about a fixed comparator: for any decision $i$, $\Phi(T+1) \ge w_i(T+1) = \prod_t (1 - \eta\, m_i(t))$. Here the sign of the cost matters, because a negative cost should *raise* a weight, so I split the rounds and use, on the nonnegative side, the convexity bound $(1-\eta)^x \le 1-\eta x$ for $x\in[0,1]$, and on the negative side the matching $(1+\eta)^{-x}\le 1-\eta x$ for $x\in[-1,0]$. Taking logs of the two handles, rearranging, and applying $\ln\frac{1}{1-\eta}\le \eta+\eta^2$ together with $\ln(1+\eta)\ge \eta-\eta^2$, the sign split recombines into the comparator's own total cost plus an $\eta$-weighted absolute-cost penalty:
$$ \sum_t m(t)\cdot p(t) \;\le\; \sum_t m_i(t) \;+\; \eta\sum_t |m_i(t)| \;+\; \frac{\ln n}{\eta}. $$
This holds for every fixed $i$, hence for the best one in hindsight. Under unit-bounded costs the regret is at most $\eta T + \frac{\ln n}{\eta}$; when $\sqrt{\ln n / T}\le \tfrac12$, balancing the two terms with $\eta=\sqrt{\ln n / T}$ yields regret at most $2\sqrt{T\ln n}$, and for shorter horizons I keep $\eta\le\tfrac12$ capped and use $\eta T + \frac{\ln n}{\eta}$ directly. The average regret goes to zero. The same calculation also explains away the deterministic factor of two: there a mistaken round only certifies that at least half the weight was wrong, so the potential drops by a factor like $1-\eta/2$ and the learner's mistakes enter only through that half-weight event, whereas here the expected loss $m\cdot p$ enters exactly, and linearity of expectation removes the scar.

There is a clean geometric reading that comes for free. Comparing the learner's distribution to any comparator distribution $p$, the update reduces the relative entropy $\mathrm{RE}(p\,\|\,p(t))$ whenever the learner's current expected cost is too high relative to $p$; on the full simplex this is the very same algorithm written in normalized coordinates, and under a convex restriction on allowed distributions one can update first and then project back with a relative-entropy (Bregman) projection, which only helps. The method is not guessing the best decision — it is repeatedly moving closer to any comparator that is proving cheaper. The variants then fall into place along the same proof. Exponential factors $w_i(t+1)=w_i(t)\exp(-\eta\,m_i(t))$ give Hedge (with $\eta\le 1$), whose second-order term depends on the learner's own squared losses $\sum_t (m(t))^2\cdot p(t)$ rather than on the comparator's absolute cost — fine sometimes, but the linear factor's clean comparator-side penalty is what the LP and set-cover reductions need. When maximizing gains instead of minimizing costs, the rule runs on $-m$, becoming $w_i(t+1)=w_i(t)(1+\eta\,m_i(t))$ with $\eta\le 1$ to keep factors nonnegative, and the inequality reverses to a lower bound.

This single update absorbs a surprising range of algorithmic reductions because they are all "experts with adversarial costs." In a zero-sum game the rows are decisions, each round the opponent returns a best-response column as the cost vector, and the regret bound puts the average play near the game value, constructing $\epsilon$-optimal mixed strategies. For a feasibility LP the constraints are the decisions: given weights over constraints, an oracle need only satisfy the single weighted-average constraint $p^\top A x \ge p^\top b$, and failure to do so is itself an infeasibility certificate; the cost fed back for constraint $i$ is its satisfaction amount $A_i x - b_i$ (scaled by the width), so a well-satisfied constraint earns positive cost and is down-weighted while a violated one earns negative cost and is up-weighted, and averaging the oracle's returned points yields an $\epsilon$-feasible solution. In the gains-form implementation this same signal appears with the sign flipped as reward $b_i - A_i x$, because the generic update multiplies by $1+\eta\cdot\text{reward}$. Set cover is the extreme $\eta=1$ case where covered elements drop to weight zero, so maximizing current expected coverage *is* greedy set cover; and boosting is the same pressure with training examples as the weighted objects — correctly classified examples lose relative emphasis, misclassified ones gain it, and the final majority vote succeeds because any example the vote misclassifies had low cumulative correctness while every weak round stayed above one half under the current distribution. The final insight is therefore a proof design as much as an update rule: make weights multiplicative so history is stored as a product, randomize proportionally so the one-step change in the potential is controlled exactly by expected cost, and sandwich that potential between the algorithm's cumulative cost and any comparator's surviving weight — equivalently, watch relative entropy fall toward the comparator.

The implementation uses the gains form for the generic update, `weights[i] *= 1 + learningRate * reward(...)`, and instantiates it as an LP solver whose experts are the constraints, with `reward(i, x) = b_i - A_i x`, a weighted-average oracle queried through `A.T @ weights` and `weights.dot(b)`, a binary search over the objective value, and a final average of the returned points.

```python
import random


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


# MWUA: the multiplicative weights update algorithm
def MWUA(objects, observeOutcome, reward, learningRate, numRounds):
    weights = [1] * len(objects)
    cumulativeReward = 0

    for t in range(numRounds):
        chosenObjectIndex = draw(weights)
        chosenObject = objects[chosenObjectIndex]

        outcome = observeOutcome(t, weights, chosenObject)
        thisRoundReward = reward(chosenObject, outcome)
        cumulativeReward += thisRoundReward

        for i in range(len(weights)):
            weights[i] *= (1 + learningRate * reward(objects[i], outcome))

    return weights
```

```python
import random
import numpy


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


# MWUA: the multiplicative weights update algorithm
def MWUA(objects, observeOutcome, reward, learningRate, numRounds):
    weights = [1] * len(objects)
    cumulativeReward = 0
    outcomes = []

    for t in range(numRounds):
        assert all(w >= 0 for w in weights)
        chosenObjectIndex = draw(weights)
        chosenObject = objects[chosenObjectIndex]

        outcome = observeOutcome(t, weights, chosenObject)
        outcomes.append(outcome)
        thisRoundReward = reward(chosenObject, outcome)
        cumulativeReward += thisRoundReward

        for i in range(len(weights)):
            weights[i] *= (1 + learningRate * reward(objects[i], outcome))

    return weights, cumulativeReward, outcomes


class InfeasibleException(Exception):
    pass


# create an oracle to solve the one-constraint optimization problem:
# (vector, scalar) -> find nonngeative x in { x : c.dot(x) = optimalValue }
#                     such that vector.dot(x) >= scalar
def makeOracle(c, optimalValue):
    n = len(c)

    def oracle(weightedVector, weightedThreshold):
        def quantity(i):
            return weightedVector[i] * optimalValue / c[i] if c[i] > 0 else -1

        biggest = max(range(n), key=quantity)
        if quantity(biggest) < weightedThreshold:
            raise InfeasibleException

        output = numpy.array([optimalValue / c[i] if i == biggest else 0 for i in range(n)])
        return output

    return oracle


# Solve a linear program of the form
#       min  c.dot(x)
#       s.t. Ax >= b, x >= 0
# given an optimal value for c.dot(x)
def solveGivenOptimalValue(A, b, linearObjective, optimalValue, learningRate=0.1):
    m, n = A.shape  # m equations, n variables
    oracle = makeOracle(linearObjective, optimalValue)

    # the reward function for the LP solver
    def reward(i, specialVector):
        constraint = A[i]
        threshold = b[i]
        return threshold - numpy.dot(constraint, specialVector)

    def observeOutcome(_, weights, __):
        weights = numpy.array(weights)
        weightedVector = A.transpose().dot(weights)
        weightedThreshold = weights.dot(b)
        return oracle(weightedVector, weightedThreshold)

    numRounds = 1000
    weights, cumulativeReward, outcomes = MWUA(
        range(m), observeOutcome, reward, learningRate, numRounds
    )
    averageVector = sum(outcomes) / numRounds

    return averageVector


def example():
    A = numpy.array([[1, 2, 3], [0, 4, 2]])
    b = numpy.array([5, 6])
    c = numpy.array([1, 2, 1])
    z = 3

    x = solveGivenOptimalValue(A, b, c, z)
    print(x)
    print(c.dot(x))
    print(A.dot(x) - b)


# Solve a linear program of the form
#       min  c.dot(x)
#       s.t. Ax >= b, x >= 0
def solve(A, b, linearObjective, maxRange=1000):
    optRange = [0, maxRange]

    while optRange[1] - optRange[0] > 1e-8:
        proposedOpt = sum(optRange) / 2
        print("Attempting to solve with proposedOpt=%G" % proposedOpt)

        # Because the binary search starts so high, it results in extreme
        # reward values that must be tempered by a slow learning rate. Exercise
        # to the reader: determine absolute bounds for the rewards, and set
        # this learning rate in a more principled fashion.
        learningRate = 1 / max(2 * proposedOpt * c for c in linearObjective)
        learningRate = min(learningRate, 0.1)

        try:
            result = solveGivenOptimalValue(A, b, linearObjective, proposedOpt, learningRate)
            optRange[1] = proposedOpt
        except InfeasibleException:
            optRange[0] = proposedOpt

    return result


if __name__ == "__main__":
    A = numpy.array([[1, 2, 3], [0, 4, 2]])
    b = numpy.array([5, 6])
    c = numpy.array([1, 2, 1])

    x = solve(A, b, c)
    print(x)
    print(c.dot(x))
    print(A.dot(x) - b)
```
