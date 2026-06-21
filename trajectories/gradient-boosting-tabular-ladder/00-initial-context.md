## Research question

The workhorse task is **supervised learning on fixed tabular benchmarks**: rows of heterogeneous features (numeric, categorical, sparse, missing) and a target to predict. Success is measured by a held-out metric and by training wall-clock time. The free variable is the rule that turns a stream of weak tree learners into one strong predictor; everything else is pinned to public, reproducible benchmarks.

Three fixed datasets anchor the ladder:

- **Higgs** — binary classification, 10.5M training rows, 28 features (last 500k held out as test). Metrics: **test AUC** (higher is better) and **training wall-clock seconds per iteration** (lower is better).
- **MS LTR** — Microsoft Learning-to-Rank, 2.27M rows, 137 features. Metric: **NDCG@10** on the test fold (higher is better).
- **Amazon** — the Kaggle Amazon Employee Access challenge, a heavily categorical dataset (employee/resource IDs with thousands of distinct values). Metric: **test log-loss** (lower is better).

Hardware is held fixed per benchmark and reported alongside the numbers (Higgs/MS-LTR timings from a single Azure ND24s server, 2×E5-2690 v4, 448GB, 16 threads; Amazon log-loss from a tuned 5-fold protocol). Because the data, splits, metrics, and tree-capacity budget are frozen, the only lever is the algorithm itself, and the cost of pulling it is paid in AUC/NDCG points and in training seconds.

## Prior art / Background / Baselines

Two starting points define the landscape:

- **Bagging and random forests.** Many trees are trained independently on bootstrap resamples of the data and averaged. This reduces variance, but no tree is directed at the mistakes the others still make, so the ensemble cannot concentrate effort where it is failing.

- **The weak-learnability question.** It is known in principle that a learner only slightly better than random guessing can be boosted into one with arbitrarily high accuracy, but the existing constructions are complex and are not yet a practical training algorithm for large heterogeneous tables.

The open question is how to turn the theoretical boosting guarantee into a fast, general-purpose tabular learner.

## Fixed substrate / Code framework

The substrate every rung fills in is a *gradient-boosting harness*: an additive model built one tree per round, where each round computes a per-example target signal from the current model and fits a tree to it. The pieces that already exist before any specific algorithm is chosen are the tree base learner, the data matrix, the loss, and the round loop:

```python
class WeakLearner:                 # a regression/decision tree base learner
    def fit(self, X, target, sample_weight=None): ...
    def predict(self, X): ...

def loss_signal(y, F_pred):
    # per-example signal the next tree should fit
    # derived from the current predictions and the supervised loss
    raise NotImplementedError      # TODO: the round-target rule we will design

def find_split(node_examples, signal):
    # choose feature + threshold for a node
    raise NotImplementedError      # TODO: the split-finding rule we will design

def encode_feature(column, fit_context=None):
    # turn a raw feature column into something a tree can split on
    raise NotImplementedError      # TODO: the feature-encoding rule we will design

def boost(X, y, n_rounds, learning_rate):
    F = init_model(y)
    for t in range(n_rounds):
        signal = loss_signal(y, F.predict(X))     # what the next tree targets
        tree = WeakLearner()
        tree.fit(X, signal)                        # split-finding happens inside
        F.add(tree, weight=learning_rate)          # additive update
    return F
```

The benchmark harness that produces the AUC/NDCG/log-loss numbers wraps this loop; the datasets and metrics are loaded by it and held fixed.

## Editable interface

The editable surface is the three `TODO` slots in the scaffold above:

- `loss_signal(y, F_pred)` — what per-example target the next tree fits.
- `find_split(node_examples, signal)` — how to choose a feature and threshold at each node.
- `encode_feature(column, fit_context=None)` — how raw columns are represented before the tree sees them.

You may implement these functions and any helpers they need. The datasets, metrics, weak-learner API, and the outer `boost` loop must stay unchanged.

## Evaluation settings

Each rung is scored on the benchmark that exercises its contribution, using the public numbers from that algorithm's own published comparison:

- **Speed / dense accuracy** on Higgs: test AUC (higher is better) and training seconds per iteration (lower is better), at matched model capacity (255 leaves / depth-8, 500 rounds, learning rate 0.1, 16 threads).
- **Ranking quality** on MS LTR: NDCG@10 on the test fold (higher is better), same capacity.
- **Categorical handling** on Amazon: test log-loss (lower is better), under a tuned 5-fold protocol.

Every number is the algorithm's own published benchmark figure, not a re-run. The ladder is ordered weak→strong by the evidence trail: each step must remove a limitation visible in the previous step while preserving the fixed datasets, metrics, and tree-capacity budget.
