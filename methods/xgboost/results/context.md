# Context: scalable tree-ensemble learning on large, sparse, structured data (circa 2014-2015)

## Research question

Across spam filtering, ad click-through prediction, fraud detection, web-search ranking, and
physics event classification, the learner that most often wins is an ensemble of decision trees
trained by boosting. The accuracy is there. What is not there is a single system that delivers
that accuracy *and* scales the way the data has grown. The concrete problems a usable system
would have to solve at once:

- **Datasets no longer fit in memory.** Click logs and insurance/claims data reach hundreds of
  millions to billions of rows. The dominant single-machine tree-boosting packages assume the
  data sits in RAM; once it does not, they either fail or thrash.
- **Tree learning re-sorts the data every iteration.** The expensive inner step of growing a
  tree is finding the best split, which requires the instances in sorted order per feature. Doing
  that sort afresh at every node of every tree, over hundreds of trees, dominates the run time and
  carries an avoidable `log n` factor.
- **Real features are sparse, and sparsity is handled by ad-hoc hacks.** Missing values, frequent
  zeros, and one-hot encodings make the input matrix mostly empty. Existing tree learners are
  tuned for dense data or special-case categorical encodings; none handles all sparsity patterns
  in one principled, *fast* way (cost proportional to the non-zeros).
- **The approximate, distributed split-finder lacks a rigorous primitive.** When the data is
  bucketed for an out-of-core or distributed split search, the candidate split points are quantiles
  of a feature. But in second-order boosting each instance carries a *weight*, and the available
  streaming quantile machinery handles only *unweighted* points with a provable error bound;
  weighted-quantile proposals have been done by sorting a random subset (failure probability) or by
  guarantee-free heuristics.
- **Accuracy still leaks through overfitting.** A tree ensemble has enough capacity to memorize;
  the procedure needs regularization that is part of *what is optimized when a split is chosen*,
  not just an after-the-fact cap on the number of trees.

A solution has to be one end-to-end system: as accurate as the best boosting, and able to push a
terabyte of sparse data through a desktop or a small cluster.

## Background

**The base learner: a regression tree (CART).** Breiman, Friedman, Olshen & Stone (1984) define a
tree as a piecewise-constant function: a structure `q` routes an input `x` to one of `T` leaves,
and each leaf `j` carries a continuous score `w_j`, so the tree computes `f(x) = w_{q(x)}`. A tree
is grown greedily: at each node, enumerate candidate splits of one feature, score each by how much
it reduces an impurity measure of the node, and keep the best. A single tree of bounded depth is a
weak, high-bias learner; the power comes from combining many.

**Additive expansions and forward stagewise fitting.** Many approximators take the form
`F(x) = Σ_{m=1}^{M} f_m(x)`, a sum of simple base functions. When fitting all `M` jointly is
infeasible, a *greedy stagewise* strategy adds one term at a time, holding the previous terms
fixed: at step `m`, choose `f_m` to most reduce the training loss given `F_{m-1}`, then set
`F_m = F_{m-1} + f_m`. Previously entered terms are never readjusted.

**Gradient boosting as steepest descent in function space (Friedman 2001).** Treat the value
`F(x)` at each point as a parameter and minimize the expected loss by numerical optimization *in
function space*. The ideal steepest-descent direction at the data points is the negative gradient
of the loss with respect to the current predictions,
`-g_m(x_i) = -[∂L(y_i, F(x_i)) / ∂F(x_i)]_{F=F_{m-1}}`,
i.e. the "pseudo-residuals". That direction is only defined at the training points, so it is
*generalized* by fitting a base learner (by least squares) to the pseudo-residuals, then taking a
one-dimensional **line search** `ρ_m = argmin_ρ Σ_i L(y_i, F_{m-1}(x_i) + ρ h(x_i))` for the step
length. With regression trees as the base learner this becomes **TreeBoost**: grow a tree to the
pseudo-residuals, then replace the single global line search by a *separate optimal constant per
leaf*, `γ_jm = argmin_γ Σ_{x_i∈R_jm} L(y_i, F_{m-1}(x_i) + γ)` — for squared error this is the
leaf mean, for absolute error the leaf median. This is the de-facto recipe (`scikit-learn`,
R's `gbm`). It uses the gradient to pick the tree's shape, then a per-leaf optimization for the
values.

**The second-order / Newton view of boosting (Friedman, Hastie & Tibshirani 2000).** Boosting can
be read as fitting an additive model by *adaptive Newton steps*. Expanding the loss to second order
in the new increment `f`, the per-point update that minimizes the quadratic is the negative first
derivative divided by the second derivative of the loss. For the logistic loss this is LogitBoost:
each round fits a base learner by *weighted* least squares to a working response
`z_i = (y_i^* - p_i) / (p_i(1-p_i))` with weights `w_i = p_i(1-p_i)`. The working response is the
negative gradient divided by curvature, and the weight is the curvature itself; the Newton step is a
weighted least-squares fit in which **the weights are the curvature**. So boosting already secretly
carries a notion of a per-instance second-order weight — but the mainstream gradient-boosting
implementations use only the first-order gradient to fit the tree and then a separate per-leaf line
search.

**Regularization of additive models.** Friedman (2001) shows that **shrinkage** — scaling each
newly added term by a factor `0 < ν < 1` (a "learning rate"), `F_m = F_{m-1} + ν·f_m` — generally
generalizes better than simply truncating the number of trees `M`; `ν` and `M` trade off against
each other. Stochastic gradient boosting (Friedman 2002) additionally subsamples *rows* each
iteration, which both regularizes and speeds up. **Column (feature) subsampling** is the trick
borrowed from RandomForest (Breiman 2001; Friedman & Popescu 2003): at each tree (or split) only a
random subset of features is considered; commercial TreeNet used it for boosting, but open-source
boosting packages did not. Regularized Greedy Forest (Zhang & Johnson 2014) goes further by putting
an explicit penalty on the whole forest and doing fully-corrective updates — more accurate in
places but harder to parallelize.

**Streaming quantiles with merge and prune (Greenwald & Khanna 2001; Zhang & Wang 2007).** For
proposing candidate split points on huge or distributed data, the database community has
`ε`-approximate **quantile summaries**: a small data structure that answers rank/quantile queries
to relative error `ε`, supporting a **merge** of two summaries (error becomes `max(ε_1, ε_2)`) and
a **prune** down to `b+1` elements (error grows to `ε + 1/b`). These two operations are what let a
summary be built in a streaming/distributed fashion. The catch: the guarantees are for
*equally-weighted* points.

**The empirical facts about scale that any system inherits.** These are properties of the hardware
and data, available before the design: indirect, by-row memory access during a feature-sorted
scan causes CPU cache misses that stall the inner loop on large data; disk read throughput, not
compute, is the bottleneck once data spills out of memory; and one-hot encoding alone can make a
feature matrix 50× sparser than it is dense, so any algorithm whose cost is proportional to the
dense size is paying mostly for zeros.

## Baselines

**TreeBoost / gradient boosting machine (Friedman 2001; `scikit-learn`, R `gbm`).** Function-space
steepest descent with trees: fit a tree to the negative-gradient pseudo-residuals, then a per-leaf
constant by line search; shrink by `ν`. Core idea is exactly right and dominant in accuracy on
tabular data. *Limitation:* it fits the tree using only the first-order gradient and finds the leaf
values by a separate per-leaf optimization, so the criterion used to *choose splits* (least-squares
fit to pseudo-residuals) is not the same quantity as the loss being minimized, and it carries no
explicit model-complexity penalty inside the split criterion. The implementations sort the data per
node every iteration, are single-threaded (R `gbm`, `scikit-learn` for the tree growth here), are
in-memory only, and have no unified sparsity handling.

**LogitBoost / adaptive-Newton boosting (FHT 2000).** Fits each round by weighted least squares with
weights equal to the loss curvature and working response equal to negative gradient divided by
curvature — a per-round Newton step. *Limitation:* derived loss-by-loss (logistic, exponential),
presented as a fitting procedure rather than as an explicit regularized objective over the tree
structure and leaf scores; no scalable system, no sparsity or out-of-core story.

**Regularized Greedy Forest (Zhang & Johnson 2014).** An explicit regularizer over the forest plus
fully-corrective re-optimization of all leaf weights. *Limitation:* the fully-corrective objective
and structure search are comparatively heavy and hard to parallelize.

**pGBRT / parallel and distributed boosting (Tyree et al. 2011; Ye et al. 2009; PLANET 2009).**
Histogram/approximate split finding to parallelize or distribute tree growth. *Limitation:* they
address only the algorithmic parallelization of split finding; out-of-core computation,
cache-aware access, and a principled sparsity-aware split finder are not addressed, and the
approximate proposal step relies on unweighted quantiles.

**In-memory distributed analytics frameworks (Spark MLlib, H2O).** General distributed ML with tree
ensembles. *Limitation:* they require the data to live in cluster RAM; they cannot fall back to
disk, so on data larger than aggregate memory they either cannot run or slow down sharply, and
their tree learners support only a subset of approximate/sparse settings.

## Evaluation settings

The yardsticks that already exist for this kind of system:

- **Allstate insurance claim classification** — ~10M rows, ~4227 features, highly sparse (mostly
  from one-hot encoding); used to probe sparsity handling. Metric: classification accuracy / AUC.
- **Higgs boson event classification** (UCI) — ~10M rows, 28 dense kinematic + derived features;
  Monte-Carlo physics events. Metric: test AUC; also time-per-tree for speed.
- **Yahoo! Learning-to-Rank Challenge** — ~20K queries, ~22 documents each, 700 features. Metric:
  NDCG@10; the natural ranking benchmark, with LambdaMART-style objectives.
- **Criteo terabyte click logs** — ~1.7B rows, 67 features (13 integer features plus average-CTR
  and count statistics derived from ID features); used to stress out-of-core and distributed
  scaling. Metric: per-iteration and end-to-end wall-clock vs. data size and vs. number of
  machines.
- **Protocol** — boost trees with a common setting such as maximum depth 8 and shrinkage 0.1, no
  column subsampling unless specified; single-machine multicore for the first three datasets,
  distributed/out-of-core for Criteo. Standard 80/20-style train/eval splits or official splits.

## Code framework

The existing scaffold is only the forward-stagewise boosting loop: keep a running prediction, fit
one base learner against the current loss signal, shrink that learner, and append it. The single
open slot is the entire base-learner step: what signal it reads from the loss, what function class it
uses, how it scores partitions, how it assigns values, and how it stays efficient on large sparse
inputs.

```python
import numpy as np


class AdditiveEnsemble:
    """Forward stagewise boosting harness. Holds the running prediction and adds
    one base learner per round; shrinks each addition by a learning rate.
    The base learner only needs to expose fit/predict once the open slot below
    has been filled."""

    def __init__(self, n_rounds, learning_rate, initial_value=0.0):
        self.n_rounds = n_rounds
        self.lr = learning_rate
        self.initial_value = initial_value
        self.learners = []

    def fit(self, X, y):
        y_pred = np.full(len(y), self.initial_value, dtype=float)
        for _ in range(self.n_rounds):
            learner = self._fit_next_base_learner(X, y, y_pred)
            self.learners.append(learner)
            y_pred += self.lr * learner.predict(X)
        return self

    def _fit_next_base_learner(self, X, y, y_pred):
        # TODO: fill this single slot with the loss signal, partitioning rule,
        #       leaf-value rule, and scalable split-search strategy.
        raise NotImplementedError

    def predict(self, X):
        out = np.full(X.shape[0], self.initial_value, dtype=float)
        for learner in self.learners:
            out += self.lr * learner.predict(X)
        return out
```

The single open slot is the base learner step together with the criterion it is grown by and the
loss signal it is fed.
