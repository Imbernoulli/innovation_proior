OK, let me think this through from scratch. I have `n` decisions, and I have to make one every round
for `T` rounds. The catch is the order of events: I commit to a decision *before* I learn what it
costs, and the costs are set by an adversary who may know exactly how I'm choosing. After I commit, a
cost vector `m^(t) ∈ [-1,1]^n` is revealed and I pay `m_i^(t)` for whichever decision `i` I picked.
I can't possibly match the best decision *each round* — that would mean predicting the adversary. So
what's the right target? The best *single fixed* decision in hindsight: `min_i Σ_t m_i^(t)`. If I can
keep my total cost close to that, I've effectively learned which decision was good without being told.
Call the gap regret. I want it sublinear in `T` so that per round it vanishes.

First, what definitely doesn't work. Suppose I think of this as binary prediction — each day predict
up or down, lose a dollar if wrong — and I have `n` experts whose predictions I can watch. The obvious
move is to go with the majority of the experts each day. But the adversary just arranges for the
majority to be wrong every single day while one quiet minority expert is always right; I lose every
round and the best expert loses nothing. Majority is hopeless. The opposite extreme — pick an expert
uniformly at random and follow them — pays the *average* expert's cost forever and never improves, no
matter how clearly one expert pulls ahead. So I need something in between: trust the experts unequally,
and let the trust *shift* toward the ones who've done well. Keep a weight `w_i` on each decision, draw
in proportion to weights, and push weight toward good decisions over time.

Now the real question is the *update*. How do I change `w_i` after seeing `m^(t)`? Let me try the most
natural thing: subtract the cost. `w_i ← w_i - η m_i^(t)`. Additive. Stare at this for a second — it's
already wrong. A decision that has blundered for a thousand rounds and one that blundered once differ
only by an additive amount; the bad one never becomes *negligible*, it just sits at a slightly lower
weight and keeps getting drawn. I want a bad decision's weight to *collapse* relative to a good one,
not merely trail it. Collapse relative to — that's a ratio, not a difference. So I should *multiply*.
`w_i ← w_i · (1 - η m_i^(t))`. Now the weight is a *product* of per-round factors:
`w_i^(t+1) = Π_{τ≤t} (1 - η m_i^(τ))`. A decision that keeps paying cost gets multiplied down round
after round; its weight decays geometrically while a consistently-good decision's weight holds up. In
the pure-mistakes case where each `m_i ∈ {0,1}`, this is literally `(1-η)^{#mistakes}` — exponential
in the number of mistakes. *That's* the separation I wanted: after a while the best decision's weight
dominates the sum, so drawing in proportion to weight means mostly drawing the best decision. Good.
This is the same multiplicative instinct behind the Winnow algorithm (Littlestone, 1988) for learning
linear classifiers, and behind the weighted-majority rule of Littlestone & Warmuth (1994) — scale a
weight down when its owner errs. Let me see exactly how much it buys.

Let me set up the bookkeeping. Each round I hold weights `w^(t)`, form the distribution
`p_i^(t) = w_i^(t) / Φ^(t)` where `Φ^(t) = Σ_i w_i^(t)`, and draw decision `i ∼ p^(t)`. My expected cost
that round is `m^(t) · p^(t)`. So my total expected cost is `Σ_t m^(t) · p^(t)`, and I want to bound
that against `Σ_t m_i^(t)` for the best `i`. I need a single quantity that ties the two together. The
sum of weights `Φ^(t)` is the obvious candidate: it's exactly the normalizer I'm already computing, and
the update touches every weight, so `Φ` should move in a way that records how the *whole ensemble* is
doing. Let me just compute how `Φ` evolves.

`Φ^(t+1) = Σ_i w_i^(t+1) = Σ_i w_i^(t)(1 - η m_i^(t)) = Σ_i w_i^(t) - η Σ_i w_i^(t) m_i^(t)`. Divide the
second sum by `Φ^(t)` and multiply back: `Σ_i w_i^(t) m_i^(t) = Φ^(t) Σ_i p_i^(t) m_i^(t) = Φ^(t) (m^(t)·p^(t))`.
So `Φ^(t+1) = Φ^(t)(1 - η m^(t)·p^(t))`. There it is — the per-round shrinkage of the potential is
governed by *my own expected cost that round*. The factor `(1 - η m^(t)·p^(t))` is awkward to compound,
but `1 + x ≤ e^x` for all `x`, so `Φ^(t+1) ≤ Φ^(t) exp(-η m^(t)·p^(t))`. Compounding from `Φ^(1) = n`
(all weights start at 1):

`Φ^(T+1) ≤ n · exp(-η Σ_t m^(t)·p^(t))`.

So *low expected cost for me means a large surviving potential, and high expected cost means the
potential has been driven down*. That's the upper handle on `Φ`. Now I need a lower handle that knows
about the best decision, because `Φ` is a sum and any single term lower-bounds it: `Φ^(T+1) ≥ w_i^(T+1)`
for every `i`, in particular the best one. And `w_i^(T+1) = Π_{τ} (1 - η m_i^(τ))`. If all the `m_i`
were in `[0,1]` I'd just take logs, but the adversary's costs live in `[-1,1]`, so I have to handle the
negative (gain) rounds too. Split by sign. For a round with `m_i ≥ 0`, I want a lower bound on
`(1 - η m_i)`; the convexity of `t ↦ (1-η)^t` gives `(1-η)^{m_i} ≤ 1 - η m_i` on `[0,1]`, i.e.
`1 - η m_i ≥ (1-η)^{m_i}`. For a round with `m_i < 0`, write `x = m_i ∈ [-1,0]`; the analogous convexity
fact `(1+η)^{-x} ≤ 1 - ηx` gives `1 - η m_i ≥ (1+η)^{-m_i}`. Multiplying across rounds:

`Φ^(T+1) ≥ w_i^(T+1) = Π_τ (1 - η m_i^(τ)) ≥ (1-η)^{Σ_{≥0} m_i^(τ)} · (1+η)^{-Σ_{<0} m_i^(τ)}`,

where the two sums run over the rounds where `m_i^(τ)` is `≥ 0` and `< 0` respectively. Now I have `Φ`
sandwiched — an upper bound that knows my cost, a lower bound that knows the best decision's cost. Take
logarithms of both. From the upper bound, `ln Φ^(T+1) ≤ ln n - η Σ_t m^(t)·p^(t)`. From the lower bound,
`ln Φ^(T+1) ≥ Σ_{≥0} m_i ln(1-η) - Σ_{<0} m_i ln(1+η)`. Chain them:

`ln n - η Σ_t m^(t)·p^(t) ≥ Σ_{≥0} m_i^(t) ln(1-η) - Σ_{<0} m_i^(t) ln(1+η)`.

Negate, and divide through by `η` to isolate my cost:

`Σ_t m^(t)·p^(t) ≤ (ln n)/η + (1/η) Σ_{≥0} m_i^(t) ln(1/(1-η)) + (1/η) Σ_{<0} m_i^(t) ln(1+η)`.

Those logs need to become clean linear terms. For `η ≤ 1/2` the estimates `ln(1/(1-η)) ≤ η + η²` and
`ln(1+η) ≥ η - η²` hold (these are the Taylor tails kept to second order). Substitute:

`Σ_t m^(t)·p^(t) ≤ (ln n)/η + (1/η) Σ_{≥0} m_i (η+η²) + (1/η) Σ_{<0} m_i (η-η²)`.

Distribute the `1/η`: `= (ln n)/η + Σ_{≥0} m_i + η Σ_{≥0} m_i + Σ_{<0} m_i - η Σ_{<0} m_i`. The two
sign-split sums of `m_i` recombine into `Σ_t m_i^(t)`, and the two `η`-terms recombine: `Σ_{≥0} m_i`
minus `Σ_{<0} m_i` is exactly `Σ_t |m_i^(t)|`. So everything collapses to

`Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t |m_i^(t)| + (ln n)/η`.

That's the whole guarantee, for *any* decision `i`, in particular the best one. Read it: my total cost
is at most the best decision's total cost, plus an error `η Σ|m_i| + (ln n)/η`. The `(ln n)/η` is the
price of not knowing the best decision a priori — it's how much I pay for spreading initial weight over
all `n`. The `η Σ|m_i|` is the price of over-reacting — each round my multiplicative nudge can overshoot
by `O(η)`. With `|m_i| ≤ 1` the second term is at most `ηT`. So regret `≤ ηT + (ln n)/η`, and these two
costs trade off in `η`: small `η` learns slowly (big `ln(n)/η`), large `η` is jumpy (big `ηT`). They're
equal when `η = √(ln n / T)`, giving regret `≤ 2√(T ln n)`. Sublinear in `T`. Per-round regret
`2√(ln n / T) → 0`. I've matched the best fixed decision in the limit, against an adversary, knowing
nothing in advance. The whole thing rode on one potential, `Φ = Σ w_i`, squeezed between an exponential
upper bound and a product lower bound.

Notice the analysis never assumed anything about the costs except the `[-1,1]` range — they can be
adversarial, correlated, even depend on my current distribution. That robustness is suspicious in a
good way: a rule this general should solve more than just prediction. Hold that thought.

Two quick variations fall out. First, suppose I'd rather use an *exponential* update,
`w_i ← w_i · exp(-η m_i)` — the Hedge form (this is what Freund & Schapire's 1997 Hedge does). Then
`Φ^(t+1) = Σ_i w_i exp(-η m_i)`, and the bookkeeping is the same except I use `exp(-ηx) ≤ 1 - ηx + η²x²`
for `|ηx| ≤ 1` in place of the linear factor. Carrying it through gives
`Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t (m^(t))²·p^(t) + (ln n)/η`. Almost the same, but the middle term
now depends on my *own* distribution's second moment, not on the best decision's loss. For some
applications I'll want the first form (penalty tied to the best decision) and for others the second;
worth keeping both. Second, sometimes the vector specifies *gains* to maximize rather than costs. Just
run the same rule on `-m`: the update becomes `w_i ← w_i (1 + η m_i)` and the bound flips to
`Σ_t m^(t)·p^(t) ≥ Σ_t m_i^(t) - η Σ_t |m_i^(t)| - (ln n)/η`. I now collect nearly as much as the best
decision.

Before I leave the experts setting, why did I insist on *randomizing* — drawing `i ∼ p^(t)` rather than
just outputting the weighted-majority prediction deterministically? Let me check what determinism costs.
If I must commit to a single 0/1 prediction each round, run weighted majority: predict the side holding
more weight, and on a wrong prediction scale the erring experts by `(1-η)`. Track the same `Φ = Σ w_i`.
Each time *I* err, the experts on the wrong side held at least half the total weight, and they all get
multiplied by `(1-η)`, so `Φ` drops by at least a factor `(1/2 + (1-η)/2) = (1 - η/2)`. After `M`
mistakes, `Φ^(T+1) ≤ n(1-η/2)^M`. And `Φ ≥ w_i = (1-η)^{m_i}` for the best expert. Taking logs with
`-ln(1-η) ≤ η + η²` gives `M ≤ 2(1+η) m_i + 2 ln(n)/η`. The factor *2* is the scar: a deterministic
predictor can be forced into twice the best expert's mistakes (the adversary splits the weight near
50/50 and makes my majority side wrong). Randomizing dissolves it, because my expected cost
`m^(t)·p^(t)` is *linear* in `p` — that linearity is exactly what let me write `Σ w_i m_i = Φ(m·p)` in
the potential step. So randomization isn't a flourish; it's what removes the factor of 2 and makes the
clean bound possible.

Now back to that suspicious generality. The cost vector can depend on my distribution and be
adversarial — that smells like a *game*. Let me make it a game. Take a two-player zero-sum game with
payoff matrix `A`, entries `A(i,j) ∈ [0,1]`, row player minimizing, column player maximizing. If I (row
player) play a distribution `p` over rows and the column player best-responds with column `j`, I pay
`A(p,j) = Σ_i p_i A(i,j)`. von Neumann's minimax theorem (1928) asserts
`min_p max_j A(p,j) = max_q min_i A(i,q) = λ*`, the value of the game — but classically that's an
existence statement proved by a separating-hyperplane / fixed-point argument; it doesn't hand me the
equilibrium. Can my regret rule *construct* it? Let me make the rows my "decisions." Each round I hold a
distribution `p^(t)` over rows; I ask an oracle for the column player's best response
`j^(t) = argmax_j A(p^(t), j)` — a single column, which I can compute by checking each of the finitely
many columns. The cost vector I feed my update is that column of `A`: `m^(t) = A(·, j^(t))`, so
`m_i^(t) = A(i, j^(t))`, all in `[0,1]`. Run the multiplicative-weights rule on these.

Apply the bound (the convex-combination form, taking the inequality against an arbitrary distribution
`p`): `Σ_t A(p^(t), j^(t)) ≤ (1+η) Σ_t A(p, j^(t)) + (ln n)/η`. Divide by `T`. On the left, each round
`A(p^(t), j^(t))` is the *best the column player could do against my play*, so it's `≥ λ*` (the column
player can always guarantee at least the value). On the right, set `p = p*`, the optimal row strategy,
for which `A(p*, j) ≤ λ*` for *every* column `j`, so `(1/T) Σ_t A(p*, j^(t)) ≤ λ*`. Stitching:

`λ* ≤ (1/T) Σ_t A(p^(t), j^(t)) ≤ λ* + η + (ln n)/(ηT)`.

Choosing `η = ε/2` and `T = ⌈4 ln(n)/ε²⌉` makes the slack at most `ε`. So the time-averaged value of my
play is within `ε` of `λ*` — using only `O(ln(n)/ε²)` calls to the best-response oracle. And the
strategies themselves are recovered: my average row distribution `p̄ = (1/T) Σ_t p^(t)` is an
`ε`-optimal minimizer (some round's `p^(t)` achieves value `≤ λ* + ε` against its best response), and
the empirical frequencies of the columns `j^(t)` the oracle played form an `ε`-optimal maximizer. The
two no-regret players' time-averaged play converges to an equilibrium — which is a *constructive,
algorithmic* proof that `min_p max ≤ max_q min`, i.e. that the value is well-defined and minimax holds.
The same machinery I built for prediction just produced von Neumann's theorem with an iteration count.
This is much sharper than fictitious play (Brown, 1951), which plays a *hard* best response and lacks a
quantitative rate; here the soft, weight-proportional play is exactly what the regret bound needs.

If a game is a repeated decision problem, so is a linear program — minimax is LP duality in disguise.
Let me push the game view onto feasibility. I want to know whether there's an `x` in some easy convex
set `P` (say nonnegativity and a budget) with `A x ≥ b`, the `m` hard constraints. Make each of the `m`
*constraints* a decision. Now here's the part that looks backwards until you see it: I'll set the cost
of constraint `i` at a point `x` to be `(1/ρ)(A_i x - b_i)` — *how much the constraint is satisfied*, not
violated, normalized by the width `ρ = max_i |A_i x - b_i|` so it lands in `[-1,1]`. Why reward
satisfaction? Because my update *lowers* weight on low-cost decisions. A constraint that's already
comfortably satisfied gets low weight; the weight piles onto the constraints that are currently
violated — exactly the hard ones I should be focusing on. The weighting `p^(t)` over constraints thus
points at the binding ones.

What does the column player / oracle do now? Given the weighting `p^(t)` over constraints, I ask for a
single `x ∈ P` satisfying the *one averaged constraint* `p^(t)^T A x ≥ p^(t)^T b`. That's a vastly
easier problem than the original `m` constraints — it's one inequality over the easy set `P`, often a
one-liner (maximize `p^T A x` over `P` and check). If the oracle ever *fails* to find such an `x`, then
`p^(t)` is a vector of nonnegative multipliers under which no point of `P` can satisfy the averaged
constraint — that's a Farkas-style certificate that the original system is infeasible, and I stop. So
assume the oracle always succeeds, returning `x^(t)`. Then by construction the expected cost each round
is `m^(t)·p^(t) = (1/ρ)(p^(t)^T A x^(t) - p^(t)^T b) ≥ 0` — the oracle's guarantee makes my per-round
cost nonnegative.

Now feed this into the regret bound for a tight constraint `i` (one that ends up binding). The bound
says `Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t |m_i^(t)| + (ln m)/η`. The left side is `≥ 0` every round.
The right side, written out with `m_i^(t) = (1/ρ)(A_i x^(t) - b_i)`, after dividing by `T`, multiplying
by `ρ`, and letting `x̄ = (1/T) Σ_t x^(t)` (which is in `P` because `P` is convex):
`0 ≤ (1+η)(A_i x̄ - b_i) + 2ηℓ + ρ ln(m)/(ηT)`, where `ℓ` is the bound on how *negative* the
normalized cost can get on the constrained subset (separating the lower width `ℓ` from the upper width
`ρ` tightens this). Choose `η = ε/(4ℓ)` and `T = ⌈8 ℓ ρ ln(m)/ε²⌉`, and the slack collapses to:
`A_i x̄ ≥ b_i - ε`. The *average* of the oracle's points is `ε`-approximately feasible, in
`O(ℓ ρ log(m)/ε²)` oracle calls. This is the Plotkin–Shmoys–Tardos (1995) Lagrangian framework, but now
it's plainly the *same* multiplicative-weights rule with constraints as experts — and the width `ρ`,
which entered as the cost-normalizer, is exactly what blows up the iteration count by `ρ²`. That tells
me where to optimize: shrink the width. (For routing problems one reroutes only as much flow as the
minimum-capacity edge allows, capping the per-edge cost at 1 — Garg & Könemann's width reduction — and
the `ρ²` factor disappears.)

The `η = 1` corner is worth a look because the rule degenerates into something I recognize. Take Set
Cover: universe of `n` elements, a family of sets, want the fewest sets covering everything. Make
*elements* the decisions; the cost of element `i` for a chosen set `C` is `1` if `i ∈ C` else `0`. Run
the rule with `η = 1`. Then the update `w_i ← w_i(1 - m_i)` sends every covered element's weight to
*zero* and leaves uncovered elements at weight `1`. So `p^(t)` is just the uniform distribution over the
*still-uncovered* elements. Maximizing my cost `m·p` — picking the set that covers the most weight —
becomes "pick the set covering the most uncovered elements." That's exactly the greedy set-cover
algorithm, dropping out of the meta-algorithm as the `η = 1` special case. And the potential analysis
gives its guarantee for free: since `OPT` sets cover everything, under any distribution on elements some
set covers at least a `1/OPT` fraction, so `m^(t)·p^(t) ≥ 1/OPT` each round. With `η = 1`,
`Φ^(t+1) = Φ^(t)(1 - m^(t)·p^(t)) < Φ^(t) e^{-1/OPT}`, so after `T = ⌈ln n⌉ · OPT` rounds
`Φ^(T+1) < n · e^{-⌈ln n⌉} ≤ 1`. With `η = 1` the surviving potential *is* the count of uncovered
elements, so `Φ < 1` means zero uncovered: everything's covered, in `⌈ln n⌉ · OPT` sets — a
`ln n`-approximation.

One more instantiation, and it's the most surprising because the "decisions" are training examples.
Boosting: I have a `γ`-weak learner — a black box that, given *any* distribution over a training set `S`
of `N` examples, returns a classifier with error at most `1/2 - γ` under that distribution. I want to
combine many such weak classifiers into one that's nearly perfect on `S`. Make the *examples* the
decisions. Each round I present my current distribution `p^(t)` over examples to the weak learner and
get back `h^(t)`. Encode the cost of example `x` as `m_x^(t) = 1 - |h^(t)(x) - c(x)|` — `1` if `h^(t)`
labels `x` correctly, `0` if it errs. Since `h^(t)` has error `≤ 1/2 - γ` under `p^(t)`, the cost I see
is `m^(t)·p^(t) ≥ 1/2 + γ` — comfortably above half every round. And because my update *lowers* weight
on low-cost decisions, the *misclassified* examples (cost 0) keep their weight and the correctly-classified
ones shrink: the distribution drifts toward exactly the examples the ensemble keeps getting wrong, so
the next weak classifier is forced to attend to them. The final classifier is the majority vote of
`h^(1), …, h^(T)`. Why does the vote work? An example `x` that the majority gets *wrong* must have been
misclassified by at least half the `h^(t)`, so its cumulative cost is `Σ_t m_x^(t) ≤ T/2`. But the
regret bound — here in the sharper restricted-distribution form, whose penalty term is tied to the
algorithm's own play — says the cumulative cost of *every* example is at least roughly the algorithm's
total `(1/2 + γ)T` minus a `(ln(N/|E|))/η` slack. Setting `η = γ` and running
`T = ⌈(2/γ²) ln(1/ε)⌉` rounds forces the fraction of training examples the majority vote gets wrong
down to `≤ ε`. So the very same reweighting rule, with examples as experts and "did the weak learner
get me right" as the cost, *is* AdaBoost (Freund & Schapire, 1997).

So the picture closes. One rule — keep a weight per decision, multiply it by `(1 - η · cost)`, draw in
proportion to weight — analyzed by one potential `Φ = Σ w_i` squeezed between `n · e^{-η Σ m·p}` and the
best decision's surviving weight, yields a `2√(T ln n)` regret bound; and by swapping in a
problem-specific cost encoding and a best-response oracle it becomes a constructive minimax solver, a
packing/covering LP solver with width-controlled iterations, greedy set cover at `η = 1`, and AdaBoost.
Let me write the rule down as code I can actually run.

```python
import random

# draw an index proportional to the given nonnegative weights:
# this realizes p_i = w_i / sum(w), the distribution we sample our decision from.
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight                 # walk the cumulative mass
        if choice <= 0:
            return choiceIndex           # land in the i-th bucket w.p. w_i / sum(w)
        choiceIndex += 1

# The multiplicative-weights update algorithm, in gain form.
# Each round: sample a decision from p^(t), observe the outcome the adversary/world
# reveals, then multiply every weight by (1 + eta * gain_i). Penalizing cost is the
# same rule run on -gain; here we phrase it as collecting gain.
def MWUA(objects, observeOutcome, reward, learningRate, numRounds):
    weights = [1] * len(objects)         # Phi^(1) = n: maximum-entropy start, total ignorance
    cumulativeReward = 0
    for t in range(numRounds):
        chosenObjectIndex = draw(weights)            # draw i ~ p^(t) = w^(t)/Phi^(t)
        chosenObject = objects[chosenObjectIndex]
        outcome = observeOutcome(t, weights, chosenObject)   # adversary reveals the round
        thisRoundReward = reward(chosenObject, outcome)
        cumulativeReward += thisRoundReward
        for i in range(len(weights)):
            # w_i <- w_i * (1 + eta * gain_i): geometric reward for decisions that pay off,
            # geometric decay for those that don't -- the multiplicative, not additive, nudge.
            weights[i] *= (1 + learningRate * reward(objects[i], outcome))
    return weights
```

And the linear-program instantiation, where the "decisions" are the `m` constraints and the oracle
plays the role of the adversary's best response:

```python
import numpy

class InfeasibleException(Exception):
    pass

# ORACLE for min c.x s.t. Ax >= b, x >= 0, given a guessed optimal value of c.x.
# Given a weighting over constraints, return a single point x in {x>=0, c.x = optimalValue}
# satisfying the averaged constraint  (A^T w).x >= w.b ; if none exists, the weighting w is
# a certificate of infeasibility for this guessed value.
def makeOracle(c, optimalValue):
    n = len(c)
    def oracle(weightedVector, weightedThreshold):
        def quantity(i):
            return weightedVector[i] * optimalValue / c[i] if c[i] > 0 else -1
        biggest = max(range(n), key=quantity)        # put all budget on the best coordinate
        if quantity(biggest) < weightedThreshold:
            raise InfeasibleException
        return numpy.array([optimalValue / c[i] if i == biggest else 0 for i in range(n)])
    return oracle

def solveGivenOptimalValue(A, b, linearObjective, optimalValue, learningRate=0.1):
    m, n = A.shape
    oracle = makeOracle(linearObjective, optimalValue)
    # reward of constraint i at point x is b_i - A_i.x : a violated constraint (A_i.x < b_i)
    # yields positive reward and keeps its weight, so weight concentrates on the hard ones.
    def reward(i, x):
        return b[i] - numpy.dot(A[i], x)
    def observeOutcome(_, weights, __):
        weights = numpy.array(weights)
        return oracle(A.transpose().dot(weights), weights.dot(b))   # solve the averaged constraint
    numRounds = 1000
    weights = MWUA(range(m), observeOutcome, reward, learningRate, numRounds)
    # collect the oracle's points and average them: x_bar = (1/T) sum_t x^(t) is feasible by convexity
    # (re-run capturing outcomes; the averaged point is the eps-approximately feasible solution)
    return weights

# Reduce optimization to feasibility by binary-searching the optimal value of c.x.
def solve(A, b, linearObjective, maxRange=1000):
    optRange = [0, maxRange]
    while optRange[1] - optRange[0] > 1e-8:
        proposedOpt = sum(optRange) / 2
        learningRate = min(1 / max(2 * proposedOpt * c for c in linearObjective), 0.1)
        try:
            result = solveGivenOptimalValue(A, b, linearObjective, proposedOpt, learningRate)
            optRange[1] = proposedOpt          # feasible: lower the target
        except InfeasibleException:
            optRange[0] = proposedOpt          # infeasible: raise the target
    return result
```

The causal chain, start to end: committing before seeing adversarial costs rules out chasing the best
decision per round, so I aim for the best *fixed* decision; majority and uniform-random both fail, so I
reweight; additive reweighting can't make a chronic loser negligible, so I multiply, making the weight
a product `(1-η)^{#mistakes}` that decays geometrically; the sum of weights `Φ` then satisfies
`Φ^(t+1) = Φ^(t)(1 - η m^(t)·p^(t)) ≤ Φ^(t) e^{-η m·p}` from above and `Φ ≥ (1-η)^{Σ m_i}` from below,
and squeezing the two gives regret `≤ ηT + (ln n)/η`, minimized at `η = √(ln n / T)` to `2√(T ln n)`;
randomizing (drawing `i ∼ p`) is what makes the cost linear in `p` and removes the deterministic factor
of 2; and since the bound never restricted the costs beyond `[-1,1]`, plugging in a best-response oracle
turns the rule into a constructive minimax solver, a width-`ρ` packing/covering LP solver, greedy set
cover at `η = 1`, and AdaBoost — all one update rule, one potential.
