## Research question

Most real-world prediction problems do not arrive as images or text; they arrive as a table — rows of
examples, columns of heterogeneous features (some numeric, some categorical, many sparse or missing),
and a target to predict. The task here is the workhorse of applied machine learning: **supervised
learning on fixed tabular benchmark datasets**, where success is measured by a held-out metric and by
how long the model takes to train. The free variable is the *boosting algorithm* — the rule that turns
a stream of weak tree learners into one strong predictor. Everything else is pinned to public,
reproducible benchmarks so that each rung is a real number an outsider can re-pull, not a story.

Three fixed datasets and metrics anchor the ladder, all from public sources:

- **Higgs** — binary classification, 10.5M training rows, 28 features (last 500k held out as test).
  Metric: **test AUC** (higher is better), and **training wall-clock seconds per iteration** (lower is
  better). This is where the speed story lives, because the data is large and dense.
- **MS LTR** — Microsoft Learning-to-Rank, 2.27M rows, 137 features. Metric: **NDCG@10** on the test
  fold (higher is better) — a ranking quality measure.
- **Amazon** — the Kaggle Amazon Employee Access challenge, a heavily *categorical* dataset (its
  features are employee/resource IDs with thousands of distinct values). Metric: **test log-loss**
  (lower is better). This is where the categorical-features story lives.

The hardware is held fixed per benchmark and reported alongside the numbers (the Higgs/MS-LTR timings
come from a single Azure ND24s server, 2×E5-2690 v4, 448GB, 16 threads; the Amazon log-loss comes from
a tuned 5-fold protocol). What is *frozen* across the ladder is the task: the datasets, the train/test
splits, the metrics, the tree-ensemble model class. What *varies*, rung by rung, is the boosting
algorithm — how each new weak learner is aimed, how tree splits are chosen, and how raw tabular
columns are made available to the tree. Because the data and metrics are fixed, you cannot buy
accuracy with a bigger model or more features; the only lever is the *algorithm*, and the cost of
pulling that lever is paid in AUC/NDCG points and in training seconds.

## Prior art before the first rung

The ladder climbs out of the idea of an *ensemble of weak learners*. A single decision tree is a weak
predictor: shallow trees underfit, deep trees overfit and are unstable. The breakthrough that makes
trees competitive is to combine *many* of them, each one focused on the examples the current ensemble
gets wrong, so the committee is far stronger than any member.

- **Bagging and random forests (Breiman, 1996/2001).** Train many trees on bootstrap resamples of the
  data and average them. This reduces variance but every tree is trained *independently* on the same
  target — no tree is told to fix the mistakes of the others. The trees are built in parallel and the
  ensemble cannot concentrate effort where it is failing.
- **The weak-learnability question (Schapire/Kearns, 1990).** Can a learner that is only slightly
  better than random guessing be "boosted" into one with arbitrarily high accuracy? The affirmative
  answer is *constructive*: build the strong learner as a weighted sequence of weak learners, where
  each new weak learner is trained on a reweighting of the data that emphasizes the examples the
  current committee handles poorly. But the early constructions were complex and not directly usable as
  a practical algorithm.

So the state of the art the ladder starts from is: trees are good weak learners, and there is a known
*sequential* way to combine them where each new tree attacks the residual errors of the committee so
far. The open questions — the ones the rungs below resolve one at a time — are how to define the
per-round signal for the next learner, how to find tree splits cheaply when the data has tens of
millions of rows, and how to handle the categorical features that dominate real tabular data without the
encoding silently corrupting the model. Each rung is a single named algorithm that answers one of these
and is measured on the fixed benchmarks above.

## Baselines

The comparison starts from the tools a practitioner already has before choosing a particular boosting
algorithm: a shallow decision tree as the weak learner, a sequential ensemble loop, a supervised loss, a
numeric representation of the table, and a deterministic evaluation harness. A useful baseline must hold
the dataset, split, metric, thread count, number of boosting rounds, learning rate, and tree-capacity
budget fixed; otherwise a faster or more accurate score could come from the experimental setup rather
than from the algorithmic choice. The feedback after each rung records the first public benchmark number
that makes the preceding limitation visible under those fixed conditions.

## Evaluation settings

Each rung is scored on the benchmark that exercises its contribution, using the public numbers from
that algorithm's own published comparison:

- **Speed / dense accuracy** is read off the Higgs benchmark: training seconds per iteration (lower is
  better) and test AUC (higher is better), at matched model capacity (255 leaves / depth-8, 500 rounds,
  learning rate 0.1, 16 threads).
- **Ranking quality** is read off MS LTR: NDCG@10 on the test fold (higher is better), same capacity.
- **Categorical handling** is read off Amazon: test log-loss (lower is better), under a tuned 5-fold
  protocol.

The metric for a rung is always the one its change targets; the feedback file states the units and
the direction of "better," and every number is the algorithm's own published benchmark figure, not a
re-run. The ladder is ordered weak→strong by the evidence trail: each step must remove a limitation
visible in the previous step while preserving the fixed datasets, metrics, and tree-capacity budget.

## Code framework

The substrate every rung fills in is a *gradient-boosting harness*: an additive model F that is a sum
of trees, built one tree per round, where each round computes a per-example target signal from the
current model and fits a tree to it. The pieces that already exist before any specific algorithm is
chosen are the tree base learner, the data matrix, the loss, and the round loop:

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

The three `TODO` slots — the round target, the split-finding rule, and the feature encoding — are
where a candidate boosting algorithm makes its choices. The benchmark harness that produces the
AUC/NDCG/log-loss numbers wraps this loop; the datasets and metrics are loaded by it and held fixed.
