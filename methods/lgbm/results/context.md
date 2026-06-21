# Context: scaling gradient boosting on big tabular data (circa 2016-2017)

## Research question

Gradient boosting decision trees (GBDT) are the consensus high-accuracy learner on structured
/ tabular data — winning at multi-class classification, click prediction, and learning-to-rank.
Training cost grows as datasets expand along *both* axes at once: many millions of rows **and**
tens of thousands to millions of features. The dominant cost in GBDT is split finding: to grow
one tree, at every node, for every feature, the learner must look at the data to evaluate the
gain of candidate split points. The per-tree cost scales like `#data × #feature`. The question
is how to reduce this cost while keeping the trees essentially as accurate as those grown on the
full data.

## Background

**GBDT as steepest descent in function space (Friedman 2001).** A GBDT model is an additive
ensemble of regression trees trained in sequence. At boosting round `t`, with current model
`F_{t-1}`, each instance gets a *gradient* of the loss w.r.t. the current prediction,
`g_i = ∂ L(y_i, F_{t-1}(x_i)) / ∂ F_{t-1}(x_i)`; the new tree is fit to the *negative*
gradients (the "pseudo-residuals"), and folded in shrunk by a learning rate. For the squared
loss `g_i = F_{t-1}(x_i) − y_i`, so `−g_i` is exactly the residual `y_i − F_{t-1}(x_i)` and the
round is ordinary residual fitting. Crucially, **GBDT instances carry no native weight** — each
round produces only a gradient per instance, nothing that marks one instance as more important
than another the way AdaBoost's reweighting does.

**Split finding and the variance gain.** Inside one tree, a node `O` is split on the feature `j`
and threshold `d` that maximize the gain. For GBDT the gain is the variance reduction, written
in sum-of-gradient form (Friedman 2001):

```
V_{j|O}(d) = (1 / n_O) [ ( Σ_{x_i∈O, x_ij≤d} g_i )² / n_l(d)
                       + ( Σ_{x_i∈O, x_ij>d}  g_i )² / n_r(d) ],
```

`n_O = Σ_i 1[x_i∈O]`, `n_l(d)`, `n_r(d)` the counts on each side. The learner picks
`d* = argmax_d V_{j|O}(d)` per feature and the best feature overall. Each split side's
contribution is the **square of a sum of gradients** divided by its count.

**Two ways to find splits.** (i) *Pre-sorted* (SLIQ, Mehta et al. 1996; SPRINT, Shafer et al.
1996; used by scikit-learn and gbm-in-R, and one mode of XGBoost): pre-sort each feature's
values and enumerate every candidate split on the sorted order. It finds the exact optimum and
stores sorted indices maintained across splits. (ii) *Histogram-based* (CLOUDS, Ranka & Singh
1998; Jin & Agrawal 2003; McRank, Li et al. 2007; used by pGBRT and one mode of XGBoost):
bucket each continuous feature into a small number of discrete bins (e.g. 255, so a bin index
fits in one byte), then for a node accumulate, per bin, the sum of gradients and the instance
count in a single pass over the node's rows — that is the node's *histogram*. Split search then
runs over the `#bin` bin boundaries instead of all sorted values. Building the histograms costs
`O(#data × #feature)`; searching costs `O(#bin × #feature)`. Since `#bin` is far smaller than
`#data`, **histogram building dominates**. A standard acceleration: a parent node's histogram
equals the sum of its two children's, so build the histogram for the smaller child and obtain
the sibling by subtraction in `O(#bin)`.

**Leaf-wise (best-first) tree growth (Shi 2007).** Two growth orders exist. *Level-wise* splits
every node of a depth before descending. *Leaf-wise* repeatedly splits the single leaf whose
split yields the largest loss reduction (max delta loss). For a fixed number of leaves,
leaf-wise reaches lower training loss — it always spends each split where it pays most — at the
risk of deeper, asymmetric trees that can overfit small data, controlled by capping the number
of leaves and the depth.

**Sparsity in real feature spaces.** Large-scale tabular datasets are usually very sparse:
one-hot encodings (one-hot word representations in text mining), indicator features, and engineered
binary features mean most entries are zero, and many feature pairs are (almost) never nonzero on
the same instance. The pre-sorted GBDT path can exploit this by skipping zero entries (Chen &
Guestrin 2016).

## Baselines

These are the prior methods a faster GBDT would be measured against and react to.

**XGBoost (Chen & Guestrin 2016).** The strongest GBDT system of the time, and the reference
baseline. It supports both the pre-sorted (exact greedy) and histogram-based split finders, uses
a regularized second-order objective (the gain `½ Σ G²/(H+λ) − γ` generalizes the variance gain
above through the loss's first and second derivatives `g_i, h_i`), a sparsity-aware split finder
that routes missing/zero entries to a learned default direction, column blocks, and out-of-core
support.

**Stochastic Gradient Boosting (SGB, Friedman 2002).** At each boosting round, draw a *uniform*
random subset of the data and fit the round's tree on that subset only. Reduces `#data` per round.

**Weight-based instance subsampling for boosting (Friedman, Hastie, Tibshirani 2000; Dubout &
Fleuret 2011; Appel et al. 2013).** A family that speeds up boosting by sampling instances
according to a per-instance importance — e.g. filtering instances whose weight is below a
threshold, or adapting the sampling ratio over the run. These are built on AdaBoost, where
each instance carries a maintained sample weight that serves as the importance signal.

**Feature filtering (PCA, Jolliffe 2002; projection pursuit, Jimenez & Landgrebe 1999; Appel et
al. 2013).** To reduce `#feature`, the standard move is to filter or project out weak features,
usually via principal-component analysis or projection pursuit.

**Per-feature nonzero tables for sparse histograms.** One can make the histogram build skip
zeros by maintaining, per feature, a table of the instances with nonzero values and scanning
only those, dropping the build cost for a feature from `O(#data)` to `O(#non-zero-data)`.

## Evaluation settings

The natural yardsticks already in use for a big-data GBDT trainer:

- **Datasets spanning both regimes.** Dense numerical: the Microsoft LETOR learning-to-rank set
  (~2M rows, 136 dense features). Sparse / one-hot heavy: Allstate insurance claim (~12M rows,
  4228 features) and Flight Delay (~10M rows, 700 features), both binary classification with
  many one-hot features. Very large and mixed dense/sparse: KDD Cup 2010 (~19M rows, 29M
  features) and KDD Cup 2012 (~119M rows, 54M features), using the published winning-solution
  feature sets. The pairing of sparse-and-huge with dense lets one stress both the
  row-reduction and the feature-reduction levers.
- **Metrics.** AUC for binary classification, NDCG@10 for ranking. Speed measured as average
  wall-clock training time per boosting iteration (so the comparison is fair across methods that
  converge in a similar number of iterations) and as time-to-accuracy curves; memory measured by
  whether a method runs at all on the largest sets.
- **Protocol.** Histogram-mode methods grown leaf-wise; fixed number of iterations with the
  best-scoring iteration reported; same parameter names as the reference system for comparison;
  hardware fixed (a multi-core server, fixed thread count) so wall-clock numbers are comparable.

## Code framework

A faster GBDT trainer plugs into the same model-wrapper shape already used for tabular boosting:
prepare train / validation matrices, hand a parameter dictionary and tabular data to a tree engine,
then return one score per test row. The wrapper, loss-to-gradient interface, validation callbacks,
and prediction surface are ordinary machinery.

```python
import numpy as np
import pandas as pd


class TabularBoostingModel:
    def __init__(self, params=None, early_stopping_rounds=50, num_boost_round=1000):
        self.params = {"objective": "mse", "verbosity": -1}
        if params is not None:
            self.params.update(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset):
        prepared = []
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"])
                if df.empty:
                    raise ValueError("empty data")
                x, y = df["feature"], df["label"]
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("one label column is required")
                prepared.append((make_engine_dataset(x.values, label=y), key))
        return prepared

    def fit(self, dataset, **kwargs):
        prepared = self._prepare_data(dataset)
        data, names = list(zip(*prepared))
        self.model = train_tree_ensemble(
            self.params,
            data[0],
            num_boost_round=self.num_boost_round,
            valid_sets=data,
            valid_names=names,
            callbacks=validation_callbacks(self.early_stopping_rounds),
            **kwargs,
        )

    def predict(self, dataset, segment="test"):
        if self.model is None:
            raise ValueError("model is not fitted yet")
        x_test = dataset.prepare(segment, col_set="feature")
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)


def train_tree_ensemble(params, train_data, num_boost_round, valid_sets, valid_names, callbacks, **kwargs):
    # TODO: provide the tree-training backend.
    pass


def make_engine_dataset(x, label):
    # TODO: adapt arrays to the tree engine's dataset object.
    pass


def validation_callbacks(early_stopping_rounds):
    # TODO: construct validation callbacks for the tree engine.
    pass
```

The wrapper has no opinion about the internal tree-building trick. It only fixes the contract the
engine must satisfy: consume tabular data and parameters, train an additive tree ensemble
with validation callbacks, and expose `predict` over the test feature matrix.
