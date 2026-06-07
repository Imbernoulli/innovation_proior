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
mistakes, `Φ^(T+1) ≤ n(1-η/2)^M`. And `Φ ≥ w_i = (1-η)^{m_i}` for any fixed expert. Taking logs gives
`M ln(1/(1-η/2)) ≤ ln n + m_i ln(1/(1-η))`. Since `ln(1/(1-η/2)) ≥ η/2` and
`ln(1/(1-η)) ≤ η + η²` for `η ≤ 1/2`, this becomes
`M ≤ 2(1+η) m_i + 2 ln(n)/η`. The factor *2* is the scar: a deterministic predictor can be forced into
twice the best expert's mistakes (the adversary splits the weight near 50/50 and makes my majority side
wrong). Randomizing dissolves it, because my expected cost
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
play is within `ε` of `λ*` — using only `O(ln(n)/ε²)` calls to the best-response oracle. The strategies
themselves come out of the same averages. For the row player,
`max_j A(p̄,j) = max_j A((1/T)Σ_t p^(t),j) ≤ (1/T)Σ_t max_j A(p^(t),j) ≤ λ* + ε`, so
`p̄ = (1/T)Σ_t p^(t)` is an `ε`-optimal minimizer. For the column player, use the regret inequality
against each pure row `i`: the empirical distribution of the played columns satisfies
`A(i,q̄) ≥ λ* - ε` for every row `i`, so `q̄` is an `ε`-optimal maximizer. The two no-regret players'
time-averaged play converges to an equilibrium — which is a *constructive, algorithmic* proof that
`min_p max ≤ max_q min`, i.e. that the value is well-defined and minimax holds. The same machinery I
built for prediction just produced von Neumann's theorem with an iteration count.
This is much sharper than fictitious play (Brown, 1951), which plays a *hard* best response and lacks a
quantitative rate; here the soft, weight-proportional play is exactly what the regret bound needs.

If a game is a repeated decision problem, so is a linear program — minimax is LP duality in disguise.
Let me push the game view onto feasibility. I want to know whether there's an `x` in some easy convex
set `P` (say nonnegativity and a budget) with `A x ≥ b`, the `m` hard constraints. Make each of the `m`
*constraints* a decision. Now here's the part that looks backwards until the sign is right: I'll set the
cost of constraint `i` at a point `x` to be `(1/ρ)(A_i x - b_i)` — how much the constraint is satisfied,
not how much it is violated, normalized by the width `ρ = max_i |A_i x - b_i|` so it lands in `[-1,1]`.
Why use satisfaction as the cost? Because the cost update multiplies by `1 - η cost`. A comfortably
satisfied constraint has positive cost and gets pushed down; a violated constraint has negative cost and
gets multiplied upward. The weighting `p^(t)` over constraints therefore drifts toward the binding and
violated constraints, exactly where the next point has to work harder. If I write code in gain form
`w_i ← w_i(1 + η reward_i)`, I just flip the sign and use `reward_i = b_i - A_i x`; it is the same
pressure on the same constraints.

What does the column player / oracle do now? Given the weighting `p^(t)` over constraints, I ask for a
single `x ∈ P` satisfying the *one averaged constraint* `p^(t)^T A x ≥ p^(t)^T b`. That's a vastly
easier problem than the original `m` constraints — it's one inequality over the easy set `P`, often a
one-liner (maximize `p^T A x` over `P` and check). If the oracle ever *fails* to find such an `x`, then
`p^(t)` is a vector of nonnegative multipliers under which no point of `P` can satisfy the averaged
constraint — that's a Farkas-style certificate that the original system is infeasible, and I stop. So
assume the oracle always succeeds, returning `x^(t)`. Then by construction the expected cost each round
is `m^(t)·p^(t) = (1/ρ)(p^(t)^T A x^(t) - p^(t)^T b) ≥ 0` — the oracle's guarantee makes my per-round
cost nonnegative.

Now feed this into the regret bound for a fixed constraint `i`. The bound says
`Σ_t m^(t)·p^(t) ≤ Σ_t m_i^(t) + η Σ_t |m_i^(t)| + (ln m)/η`, and the left side is nonnegative because the
oracle satisfies the averaged constraint every round. Substitute
`m_i^(t) = (1/ρ)(A_i x^(t) - b_i)`:
`0 ≤ Σ_t (A_i x^(t) - b_i)/ρ + ηΣ_t |A_i x^(t)-b_i|/ρ + (ln m)/η`. The absolute-value term is the only
place the width bites. If constraint `i` is in the side where negative violation is bounded by `ℓ`, then
on rounds with `A_i x^(t) - b_i < 0` the absolute value is at most `ℓ`, while on nonnegative rounds I can
fold `η(A_i x^(t)-b_i)` into the main sum. That gives
`0 ≤ (1+η)Σ_t (A_i x^(t)-b_i)/ρ + 2ηℓT/ρ + (ln m)/η`. Divide by `T`, multiply by `ρ`, and let
`x̄ = (1/T)Σ_t x^(t)`, which stays in `P` by convexity:
`0 ≤ (1+η)(A_i x̄ - b_i) + 2ηℓ + ρ ln(m)/(ηT)`. Choose `η = ε/(4ℓ)` and
`T = ⌈8ℓρ ln(m)/ε²⌉`; then `2ηℓ + ρ ln(m)/(ηT) ≤ ε`, so
`A_i x̄ ≥ b_i - ε`. The *average* of the oracle's points is `ε`-approximately feasible, in
`O(ℓρ log(m)/ε²)` oracle calls. This is the Plotkin–Shmoys–Tardos (1995) Lagrangian framework, but now
it's plainly the same reweighting rule with constraints as experts — and when `ℓ = ρ`, the width enters
as `ρ²`. That tells me where to optimize: shrink the width. For routing problems I can reroute only as
much flow as the minimum-capacity edge allows, so each edge's per-round load is capped at 1; that is the
Garg–Könemann width reduction.

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
is `m^(t)·p^(t) ≥ 1/2 + γ` — comfortably above half every round. And because my update lowers weight
on high-cost decisions, the *misclassified* examples (cost 0) keep their weight and the correctly-classified
ones shrink: the distribution drifts toward exactly the examples the ensemble keeps getting wrong, so
the next weak classifier is forced to attend to them. The final classifier is the majority vote of
`h^(1), …, h^(T)`. Why does the vote work? An example `x` that the majority gets *wrong* must have been
misclassified by at least half the `h^(t)`, so its cumulative cost is `Σ_t m_x^(t) ≤ T/2`. But the
restricted-distribution bound makes this precise. Let `E` be the examples the final majority vote gets
wrong, and compare against the uniform distribution on `E`. The algorithm's total expected cost is at
least `(1/2 + γ)T`. The comparator's average cost is at most `T/2`, because every example in `E` is
correct on at most half the rounds. The restricted-distribution bound gives
`(1/2 + γ)T ≤ (1+η)T/2 + ln(N/|E|)/η`. With `η = γ`, this says
`γT/2 ≤ ln(N/|E|)/γ`, hence `|E|/N ≤ exp(-γ²T/2)`. Running
`T = ⌈(2/γ²) ln(1/ε)⌉` rounds forces the fraction of training examples the majority vote gets wrong down
to `≤ ε`. So the very same reweighting rule, with examples as experts and "did the weak learner get me
right" as the cost, is AdaBoost (Freund & Schapire, 1997).

So the chain is clear now. Committing before seeing adversarial costs rules out chasing the best decision
per round, so I aim for the best fixed decision; majority and uniform-random both fail, so I reweight;
additive reweighting can't make a chronic loser negligible, so I multiply, making weights products that
separate geometrically; the sum of weights `Φ` then satisfies
`Φ^(t+1) = Φ^(t)(1 - η m^(t)·p^(t)) ≤ Φ^(t)e^{-η m·p}` from above and the single-decision product lower
bound from below, and squeezing the two gives regret `≤ ηT + (ln n)/η`, minimized at
`η = √(ln n/T)` to `2√(T ln n)`; randomizing is what makes cost linear in `p` and removes the
deterministic factor of 2; and because the bound never used anything except `[-1,1]` costs, a
problem-specific oracle turns the same rule into a constructive minimax solver, a width-controlled LP
solver, greedy set cover at `η = 1`, and boosting. The code I end up with uses the gain convention,
because then the LP reward `b_i - A_i x` is positive exactly on violated constraints.

```python
import random
import numpy


# Draw an index proportional to the given nonnegative weights.
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0
    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex
        choiceIndex += 1


# Multiplicative weights in gain form:
# w_i <- w_i * (1 + learningRate * reward_i).
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
        cumulativeReward += reward(chosenObject, outcome)

        for i in range(len(weights)):
            weights[i] *= (1 + learningRate * reward(objects[i], outcome))

    return weights, cumulativeReward, outcomes


class InfeasibleException(Exception):
    pass


# Create an oracle for the one-constraint problem:
# find nonnegative x with c.dot(x) = optimalValue and weightedVector.dot(x) >= weightedThreshold.
def makeOracle(c, optimalValue):
    n = len(c)

    def oracle(weightedVector, weightedThreshold):
        def quantity(i):
            return weightedVector[i] * optimalValue / c[i] if c[i] > 0 else -1

        biggest = max(range(n), key=quantity)
        if quantity(biggest) < weightedThreshold:
            raise InfeasibleException

        return numpy.array([optimalValue / c[i] if i == biggest else 0 for i in range(n)])

    return oracle


# Solve min c.dot(x) subject to Ax >= b, x >= 0, given a guessed objective value.
def solveGivenOptimalValue(A, b, linearObjective, optimalValue, learningRate=0.1):
    m, n = A.shape
    oracle = makeOracle(linearObjective, optimalValue)

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
    return sum(outcomes) / numRounds


# Reduce optimization to feasibility by binary-searching the objective value.
def solve(A, b, linearObjective, maxRange=1000):
    optRange = [0, maxRange]
    result = None

    while optRange[1] - optRange[0] > 1e-8:
        proposedOpt = sum(optRange) / 2
        learningRate = 1 / max(2 * proposedOpt * ci for ci in linearObjective)
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
