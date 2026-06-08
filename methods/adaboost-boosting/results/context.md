# Context: boosting a weak learner to high accuracy

## Research question

Suppose you have a learning algorithm that is barely useful — for any distribution over a
labelled sample it returns a classifier whose error is just *under* 1/2, say at most 1/2 − γ for
some small fixed γ > 0. Such a learner is cheap: it does no more than beat a coin flip by a
sliver. Now suppose you actually need a classifier with error at most ε, for an ε you get to
choose as small as you like. Can the cheap learner, used as a black box, be turned into one
that achieves arbitrarily small error — using only polynomially many calls and polynomially
much data?

This is the *hypothesis boosting* question. It was raised explicitly by Kearns and Valiant,
who, in formalising distribution-free (PAC) learning, separated two strengths of learnability:
a *strong* learner that can be driven to any accuracy ε, and a *weak* learner that need only beat
random guessing by an inverse polynomial. Strong learnability obviously implies weak
learnability. The open question is the converse — whether the two notions are in fact
**equivalent**. If they are, then accuracy is essentially free: any nontrivial learning ability
can be amplified to perfection. If they are not, then "a little better than chance" is a genuine
ceiling that no amount of cleverness can break through. The stakes are foundational: the answer
decides whether the difficulty of learning lives in *getting started at all* or in *getting the last
few percent*.

A solution would have to (i) produce, from black-box access to the weak learner, a final
classifier provably below any target ε; (ii) do so with the number of weak-learner calls and the
sample size growing only polynomially in 1/ε (and the other problem parameters); and (iii)
ideally not require knowing γ in advance, since in practice the edge of a real learner varies
wildly from one reweighting of the data to the next and is never known ahead of time.

## Background

**The PAC model and the two strengths of learnability.** In Valiant's distribution-free model a
*concept* c is a {0,1}-labelling of an instance space X. Examples (x, c(x)) arrive with x drawn
from an unknown, arbitrary distribution D. A learner outputs a hypothesis h; its error is
Pr_{x∼D}[h(x) ≠ c(x)]. A **strong** PAC learner, given ε, δ > 0, outputs with probability ≥ 1 − δ
a hypothesis of error ≤ ε, in time polynomial in 1/ε, 1/δ and the size of the concept. A **weak**
learner (Kearns and Valiant) relaxes only the accuracy demand: it need only output, with
probability ≥ 1 − δ, a hypothesis of error ≤ 1/2 − 1/p(n) for some polynomial p — that is,
accuracy at least 1/2 + 1/p(n), only inverse-polynomially better than a coin.

**Why the question is hard, and why the naive attempt fails.** The obvious idea — run the weak
learner many times on independent samples and take a majority vote of the resulting hypotheses
— does not work. Drawn from the *same* distribution D, the weak hypotheses all tend to make
their mistakes on the *same* hard region of the space; voting over correlated errors does not
escape them. Kearns gave a convincing argument that naive repeated-and-vote cannot boost
accuracy. So whatever works must somehow *change* what the weak learner sees on each call —
force it onto the parts of the space the previous hypotheses got wrong — so the new mistakes are
decorrelated from the old ones.

**It is well documented that mediocre learners are easy and accurate ones are hard.** Across
practical learning — decision rules, simple thresholds, shallow trees — it is routine to find a
rule that is *somewhat* predictive and very hard to find one that is highly accurate directly.
This asymmetry is exactly what makes the boosting question worth asking: if amplification works,
one builds the hard object (high accuracy) out of many copies of the easy one (slight edge).

**Reliability is free; accuracy is the hard part.** Increasing the *confidence* parameter δ of any
learner is trivial — run it O(log(1/δ)) times on fresh samples and validate each hypothesis on a
held-out set, keeping one that tests well. So the entire difficulty of the boosting question is the
accuracy axis ε, not the reliability axis δ.

**The on-line allocation / weighted-majority background.** Separately, a body of work on
*on-line prediction with expert advice* had matured. Littlestone and Warmuth's weighted-majority
algorithm maintains a weight over a set of "experts," predicts by their weighted vote, and after
each outcome multiplies the weight of each erring expert by a fixed factor β ∈ [0,1). One proves
that the learner's cumulative loss stays within a bounded factor of the best expert's loss, with a
regret that grows like √(T ln N) over T trials — sub-linear, so the per-trial gap vanishes. The
machinery is a multiplicative weight update plus an exponential potential argument. This is a
worst-case, distribution-free guarantee about combining many weak predictors by a weighted vote
under adaptively chosen losses — structurally the same shape of problem as combining weak
hypotheses, though it had not been connected to boosting.

## Baselines

**Schapire's recursive boosting (Schapire 1990).** The first proof that weak ⟹ strong. Given a
weak learner A with error α < 1/2, build A′ that calls A three times on three distributions over X:
D1 is the original; D2 is a *filtered* distribution under which A's first hypothesis h1 is forced to
chance (an instance is passed to A only after coin-flipping in a way that equalises h1's correct and
incorrect mass), yielding h2; D3 is the distribution of examples on which h1 and h2 *disagree*,
yielding h3. The output is the majority vote of h1, h2, h3. A direct calculation shows this
majority has error g(α) = 3α² − 2α³, which is strictly less than α for α < 1/2. Applying the
construction recursively — each level demanding a smaller error of its three sub-hypotheses —
drives the error below any ε in O(log(1/ε)) levels of recursion. **What it leaves open:** the final
hypothesis is a deep, bushy circuit of three-input majority gates whose structure varies run to
run; the filtering implicitly assumes control of the bias; and the analysis is governed by the
*worst* sub-hypothesis at each node. It proves the equivalence but the construction is clumsy and
does not exploit weak hypotheses that happen to be much better than the worst-case bias.

**Freund's boost-by-majority (Freund 1995).** A simpler, near-optimally efficient boosting
algorithm. Instead of a recursive circuit, it runs the weak learner T times, each time on a
reweighted distribution that puts more mass on the examples that the *running unweighted majority*
of the previous hypotheses has gotten wrong, and outputs a *single* unweighted majority gate over
all T weak hypotheses. The weights it assigns to examples come from a binomial-tail calculation
that is provably the optimal way to allocate "how many more correct votes do I still need," and the
number of rounds it needs matches an information-theoretic lower bound. **What it leaves open:** the
reweighting schedule, and the very definition of the majority gate, are derived from a γ that must
be **known before boosting begins** — the algorithm is told the worst-case edge of the weak learner
in advance. In practice this edge is unknown and, worse, *varies* from one reweighted distribution
to the next; because the final vote is *unweighted*, boost-by-majority cannot give extra credit to a
round on which the weak learner happened to return a much better hypothesis than the assumed γ. It
throws away exactly the information a practical booster most wants to use.

**Weighted majority / on-line allocation (Littlestone–Warmuth; and the general allocation
setting).** As above: maintain multiplicative weights over N strategies, predict by weighted
vote, update by a fixed β. The guarantee bounds cumulative loss against the best single strategy.
**What it leaves open:** it is posed as an *on-line* game against an adversary, with a *fixed* β set
ahead of time, and is about tracking the best expert — not, on its face, about manufacturing a
single accurate classifier out of weak ones. The connection to boosting (and the idea of letting
β float with the observed performance) is not present.

## Evaluation settings

The natural yardstick is the PAC framework itself: report the error Pr_{x∼D}[h(x) ≠ c(x)] of the
produced hypothesis, and the resources — number of weak-learner calls (rounds T), sample size,
running time — as functions of the target ε, the confidence δ, and the edge γ. For empirical study,
the standard setup is a fixed training set drawn from X × {0,1} ("boosting by sampling" /
batch learning), with error measured both on the training set and on a held-out test set as a
function of the number of rounds. The weak learner is instantiated by a simple base class: decision
stumps (single-feature, single-threshold rules) or shallow decision trees such as those produced by
C4.5; benchmark problems are the usual UCI-style classification datasets. Generalisation is studied
by comparing test error to training error, and by tracking how both move as T grows. The yardstick
for the *theory* is a bound on training error as a function of the per-round weighted errors ε_t, and
a bound on generalisation error in terms of sample size, the complexity (VC-dimension) of the base
class, and T.

## Code framework

The primitives that already exist: a labelled dataset, a reweightable *weak learner* that can be
fit to a weighted sample and returns a classifier, and the arithmetic to combine classifier outputs.
What does not yet exist is the rule for *how* to reweight between rounds, *how much* to trust each
returned hypothesis, and *how* to combine them into the final vote — those are the empty slots.

```python
import numpy as np

# --- existing primitive: a weak learner fit to a weighted sample ---
# e.g. a decision stump: pick the single feature/threshold minimizing weighted error.
class WeakLearner:
    def fit(self, X, y, sample_weight):   # y in {-1,+1}
        # returns a hypothesis h: X -> {-1,+1} with low *weighted* error
        ...
        return self
    def predict(self, X):
        ...

def weighted_error(h, X, y, w):
    # epsilon_t = sum_i w_i * 1[h(x_i) != y_i],  with w a distribution (sum=1)
    return np.sum(w * (h.predict(X) != y))

# --- the boosting harness: the slots the method will fill ---
def boost(X, y, T, weak_learner_factory):
    N = len(y)
    w = np.full(N, 1.0 / N)               # initial distribution over examples
    hypotheses, coeffs = [], []
    for t in range(T):
        h = weak_learner_factory().fit(X, y, sample_weight=w)
        eps = weighted_error(h, X, y, w)

        alpha = None                       # TODO: how much to trust h given eps
        # TODO: reweight the examples for the next round, then renormalize w
        #       (up-weight the ones h got wrong, down-weight the ones it got right)

        hypotheses.append(h); coeffs.append(alpha)
    return hypotheses, coeffs

def predict(hypotheses, coeffs, X):
    # TODO: combine the weak hypotheses into a single prediction
    pass
```
