# Context: turning weak learners into accurate predictors (circa 1995)

## Research question

A learning algorithm is *strong* (in the PAC sense of Valiant 1984) if, given enough data, it can
produce a hypothesis with arbitrarily small error with high probability. It is merely *weak* if it
can only be guaranteed to do slightly better than random guessing — on a two-class problem, error at
most `1/2 - gamma` for some small but positive edge `gamma`, but on *every* input distribution.
Kearns and Valiant (1988; JACM 1994) posed the question sharply: are these two notions actually
different, or is every weakly learnable concept class also strongly learnable? Equivalently — and this
is the form that matters for practice — given only a black-box subroutine that reliably returns a
hypothesis a hair better than chance, can we *drive its error to zero* by calling it repeatedly and
combining the results? A constructive positive answer would be worth a great deal: it would mean one
could build a highly accurate predictor out of a crude, easy-to-write rule-of-thumb generator, and it
would say something deep about the structure of learnability itself. The precise goal is a
combination procedure that (1) needs the weak learner only as a black box returning a hypothesis with
*some* edge; (2) provably amplifies that edge into arbitrarily small training error; (3) does so with
a small, predictable number of calls; and (4) — the practical sticking point left open by the prior
art below — does not require an accuracy parameter fixed before the learner is called. Closing (4)
without giving up (1)-(3) is the problem.

## Background

The field state rests on a few load-bearing pieces.

**The PAC / weak-learning frame.** Valiant's PAC model measures a learner by whether it returns, with
probability at least `1 - delta`, a hypothesis of error at most `epsilon`, for any target `epsilon`,
`delta`. Weak learning fixes a single modest accuracy instead: error bounded by `1/2 - gamma` for
some `gamma > 0`, uniformly over distributions. Kearns and Valiant formalized the weak condition as
accuracy at least `1/2 + 1/p(n)` for a polynomial `p`, and connected the difficulty of weak learning
to cryptographic hardness — so the weak-versus-strong question was known to be substantive, not a
triviality.

**Why the obvious approach fails — a diagnostic fact, not a measurement.** The naive idea is to run
the weak learner many times and take a majority vote. But every run sees the *same* distribution, so
nothing forces the runs to make *different* mistakes. The weak learner can keep latching onto the same
easy regularity and keep missing the same hard pocket of the input space; a majority of correlated
errors is still an error. This observation (recorded by Schapire 1990, attributed to Kearns) is the
pivotal background fact: collecting many weak hypotheses is not enough — one must somehow make their
errors *diverge*. The distribution-free promise of weak learning is exactly the lever that makes this
conceivable: if the weak learner has an edge on *every* distribution, then one is free to confront it
with distributions that emphasize whatever the current committee is getting wrong, and it must still
return something useful there.

**Reweighting examples to refocus a black-box learner.** Two mechanisms for "showing the learner a
chosen distribution" were standard. *Boosting by filtering* uses the weak learner inside a rejection
sampler: keep or discard incoming examples so that what reaches the learner is effectively drawn from
the desired distribution. *Boosting by reweighting* keeps a fixed training set and attaches a weight
to each example, passing those weights to a learner that accepts weighted samples (or, if it does not,
resampling the set with replacement according to the normalized weights). Either way, the controllable
object is a distribution `p` over the training examples, and the question of *how to move `p` from one
round to the next* is the open design choice.

**Multiplicative-weights / online allocation.** A separate line — the Weighted Majority algorithm
(Littlestone & Warmuth 1994) and its descendants for the on-line allocation / "experts" problem —
gives a general template for sequential decision-making: maintain a weight on each of several options,
suffer a loss each round, and update every weight by a *multiplicative* factor that decays the options
that did badly. The analyses bound the total accumulated loss against the best fixed option by
tracking how the *sum of all weights* evolves, using the convexity inequality
`beta^x <= 1 - (1 - beta) x` for `x in [0,1]`, `beta in [0,1]`. This is an algebraic shape — a
multiplicative update plus a total-weight potential argument — that has nothing intrinsically to do
with boosting, but it is sitting on the table.

**Weak learners that are actually used.** In practice the weak learner is something deliberately
simple and high-bias: a *decision stump* (a one-split threshold rule on a single feature) or a
shallow decision tree (e.g. depth 2-3). Such a learner is cheap, almost always clears the
better-than-chance bar on a reweighted problem, and is weak enough that it leaves room for later
rounds to contribute — a fully grown tree would fit a single weighted problem too well and leave no
edge for anyone else. Decision-tree learners (CART, C4.5) that accept per-example weights were widely
available.

## Baselines

These are the prior combination procedures a new method would be measured against and would react to.

**The recursive three-distribution construction (Schapire 1990).** The first proof that weak learning
implies strong learning. Given a weak learner with edge on every distribution, it manufactures three
sub-problems: run the learner on the original distribution `D1` to get `h1`; then form `D2`, a
distribution on which `h1` is correct exactly half the time (its advantage is neutralized), and learn
`h2`; then form `D3` concentrated on the examples where `h1` and `h2` disagree, and learn `h3`. The
majority vote of `h1, h2, h3` has error `g(a) = 3a^2 - 2a^3`, which is strictly less than `a` whenever
`a < 1/2`; recursing this construction drives the error to zero. This settles the question, but as a
*mechanism* it is awkward: the final predictor is a deep recursive majority-of-three circuit, the
analysis charges every sub-call at a single worst-case error level, and the structure has no way to
take extra advantage of a sub-call that happens to come back much better than that worst case.

**Boosting by majority (Freund 1995).** A sharper combiner that flattens the recursion. Instead of a
tree, it builds *one* majority vote over many weak hypotheses obtained sequentially. The example
weights each round are derived from a majority-vote game: roughly, an example's weight reflects how
many additional correct future votes it still needs in order to end up on the correct side of the
final tally, which yields a binomial-tail weighting schedule that is provably near-optimal and a
number of rounds that is near the information-theoretic minimum for a given edge. This is a real
advance in tightness. Its limitation, the one that matters here: the weighting schedule is computed
*from a fixed edge `gamma` that must be supplied before the run begins*. A schedule pinned to one
presumed worst-case edge cannot respond to easier and harder subproblems as they arise, and because
the final vote is an *unweighted* majority, all returned hypotheses are counted exactly the same in
the output.

**Bagging (Breiman 1996).** A different variance-reduction combiner: train each predictor on an
independent bootstrap resample of the data and average (classification: vote; regression: mean). It
diversifies hypotheses through resampling randomness rather than by targeting current errors, so the
distribution shown to the learner does not adapt to where the committee is failing, and every member
is given equal weight regardless of how good it is.

## Evaluation settings

The natural yardsticks already in use for such a combiner, all pre-existing:

- **UCI benchmark classification datasets** — small-to-medium tabular problems with categorical and
  numeric features, evaluated by test-set error rate under cross-validation or repeated train/test
  splits, with decision stumps or shallow trees (C4.5 / CART) as the base learner.
- **A binary classification dataset with class labels in `{-1, +1}` or `{0, 1}`**; for the
  amplification claim, the quantity watched over rounds is the *training* error of the combined
  predictor and how it falls as rounds accumulate, alongside the per-round weighted error of each
  returned weak hypothesis.
- **Regression benchmarks** — tabular datasets with a continuous target, evaluated by a test-set
  error such as root-mean-squared error or mean absolute error, with a shallow regression tree (e.g.
  `max_depth = 3`) as the base learner and a fixed train/test split.
- Protocol: the base learner, the number of rounds, and the data split are held fixed across the
  procedures being compared; the same weighted training set is handed to each combiner.

## Code framework

A combination procedure of this kind plugs into a fixed boosting harness that already exists: an outer
loop that, for a fixed number of rounds, fits a fresh weak learner on the current weighted training
set, records it, and then asks a *strategy object* whether the learner is usable, how much it should
count in the final ensemble, and how the examples should be presented on the next round. Everything
specific to the combiner — how the examples are presented, how much each learner counts, what each
learner is even asked to fit — lives behind the strategy interface, and that interface is exactly the
empty slot. The base learner (a shallow decision tree), the weighted fitting or weighted resampling,
and the aggregation skeleton already exist.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BoostingStrategy:
    """The per-round decisions of a sequential ensemble. Everything here is the
    open design: how examples are weighted, what each weak learner fits, how much
    each learner counts in the final vote, and how the weights move for next round."""

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]          # "classification" or "regression"
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        # Starting distribution over the training examples (should sum to 1).
        # TODO: the initial weighting we will choose.
        pass

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # What the next weak learner is asked to fit.
        # TODO: the per-round target we will design.
        pass

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        # Whether the just-fitted learner is usable, and how much it counts if it is.
        # TODO: the acceptance test and coefficient we will design.
        pass

    def update_weights(self, sample_weights, learner, X, y,
                       pseudo_targets, learner_weight, round_idx):
        # The distribution over examples for the next round.
        # TODO: the reweighting rule we will design.
        pass


# fixed harness the strategy plugs into
def fit_ensemble(X, y, strategy, make_weak_learner):
    sample_weights = strategy.init_weights(len(y))
    learners, learner_weights = [], []
    predictions = np.zeros(len(y))                     # current ensemble output on train
    for t in range(strategy.n_rounds):
        targets = strategy.compute_targets(y, predictions, sample_weights, t)
        learner = make_weak_learner()                  # a shallow tree
        learner.fit(X, targets, sample_weight=sample_weights)
        learner_weight = strategy.compute_learner_weight(learner, X, y, targets,
                                                         sample_weights, t)
        if learner_weight is None:
            break
        learners.append(learner); learner_weights.append(learner_weight)
        sample_weights = strategy.update_weights(sample_weights, learner, X, y,
                                                 targets, learner_weight, t)
        predictions = ensemble_predict(learners, learner_weights, X, strategy)  # aggregate so far
    return learners, learner_weights
```

The harness supplies the fitted learner, the current weights, the labels, and the round index; the
four stubs are where the combiner's actual decisions go.
