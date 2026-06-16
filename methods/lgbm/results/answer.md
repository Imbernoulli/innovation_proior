# LightGBM, distilled

LightGBM is a gradient boosting decision tree (GBDT) system built on a histogram-based,
leaf-wise tree engine, made fast on big tabular data by two techniques that attack the dominant
`#data × #feature` histogram-build cost. **Gradient-based One-Side Sampling (GOSS)** cuts the
rows: it keeps all the large-gradient (under-trained) instances and randomly samples `b·n`
instances from the small-gradient tail, reweighting that sample by `(1-a)/b` so the tail's
gradient mass is represented in the split gain. **Exclusive Feature Bundling (EFB)** cuts the
columns: it packs mutually exclusive (rarely-co-nonzero) sparse features into single columns via
a greedy graph-coloring, merging them by disjoint bin-offset ranges so no split is lost.

## Problem it solves

GBDT is the consensus high-accuracy learner on structured/tabular data, but its dominant cost is
split finding — for each feature, look at the data to evaluate candidate splits — so per-tree
work scales like `#data × #feature`. On datasets large in both rows and features this is too slow
and memory-heavy. The goal is a GBDT trainer sub-linear in `#data × #feature` with no accuracy
loss. Naive row dropping biases the gain; AdaBoost-style importance sampling needs an instance
weight GBDT lacks; feature filtering needs redundancy that may not exist.

## Foundation: histogram-based, leaf-wise GBDT

A GBDT model is an additive ensemble; at round `t`, gradient `g_i = ∂L(y_i, F_{t-1}(x_i))/∂F`
(for squared loss, `g_i = ŷ_i − y_i`, so `−g_i` is the residual), and a tree is fit to `−g`. The
split gain on a node is the variance gain in sum-of-gradient form:

```
V_j(d) = (1/n) [ ( Σ_{x_ij ≤ d} g_i )² / n_l(d) + ( Σ_{x_ij > d} g_i )² / n_r(d) ].
```

- **Histogram split finding.** Bucket each feature into `≤ max_bin` (default 255, fits in a byte)
  discrete bins; per node accumulate per-bin `Σg` (and count / `Σh`) in one pass, then search the
  `≤ #bin−1` bin boundaries. Build is `O(#data × #feature)`; search is `O(#bin × #feature)`; build
  dominates since `#bin ≪ #data`. **Subtraction trick:** a parent's histogram = sum of its two
  children's, so build the smaller child's and get the sibling by subtraction in `O(#bin)`.
- **Leaf-wise (best-first) growth.** Repeatedly split the leaf with max loss reduction. For a
  fixed leaf budget it reaches lower loss than level-wise; cap with `num_leaves` and `max_depth`
  to bound overfit.
- **Second-order / regularized gain (engine generalization).** With per-leaf `G = Σg`, `H = Σh`,
  leaf value `w* = −G/(H+λ2)` (L1 enters as a soft-threshold on `G`), split gain
  `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ`. Squared loss has `h_i = 1` and
  reduces to `V_j(d)`. Categorical features: sort bins by `G/H` and search the ordered partition,
  `O(k log k)`, instead of one-hot's `2^{k−1}−1`.

## GOSS — reduce #data

`|g_i|` is the importance signal GBDT does produce: small `|g|` means well-trained, large `|g|`
means under-trained, and large-`|g|` instances dominate the gradient-sum gain. Keep the top
`a·n` instances by `|g|` (set `A`), randomly sample `b·n` instances from the remaining tail
`A^c` (set `B`), and reweight `B`'s gradients by `(1-a)/b`. Since the tail has `(1-a)n`
instances, the sampling fraction inside the tail is `b/(1-a)`, so the reweighted sampled sum is
an unbiased estimate of the full tail gradient sum. The estimated gain:

```
Ṽ_j(d) = (1/n) [ ( Σ_{A_l} g_i + (1−a)/b · Σ_{B_l} g_i )² / n_l(d)
               + ( Σ_{A_r} g_i + (1−a)/b · Σ_{B_r} g_i )² / n_r(d) ].
```

`a = 0` recovers plain random sampling (SGB). Per round (Alg. 2): predict on the full data,
compute `g`, sort by `|g|`, `topN = a·|I|`, `randN = b·|I|`, `usedSet = top topN ∪ random randN
of the rest`, multiply the random set's weights by `fact = (1−a)/b`, fit a tree to `−g[usedSet]`
with those weights.

**Approximation theorem.** Let `E(d) = |Ṽ_j(d) − V_j(d)|`, with `V` and `Ṽ` already carrying the
common `1/n` gain normalization. With probability `≥ 1−δ`,

```
E(d) ≤ C²_{a,b} ln(1/δ) · max{ 1/n_l(d), 1/n_r(d) } + 2 D C_{a,b} √( ln(1/δ)/n ),
C_{a,b} = (1−a)/√b · max_{x_i∈A^c} |g_i|,   D = max(ḡ_l(d), ḡ_r(d)),
```

with `ḡ_l(d) = Σ_{x_i∈(A∪A^c)_l}|g_i| / n_l(d)`. **Proof sketch:** strip the common `1/n` by
writing `U_j = nV_j`. The unnormalized gain difference factors as
`C_l·((1-a)/b Σ_{B_l} g − Σ_{A^c_l} g) + C_r·(…)`, so after taking absolute values each side is
controlled by `max{|C_l|,|C_r|}` times the rescaled tail-sampling error; after dividing by `n`, bound
`|C_l|, |C_r|` by `D_{A^c}·(Hoeffding on an indicator difference) + 2D`, bound the tail gradient error
by Hoeffding, absorb the side/union constants into the logarithm, multiply, and collect with `C_{a,b}`. **Consequences:** the rate
is `O(1/n_l + 1/n_r + 1/√n)`; on a balanced split (`n_l, n_r ≥ O(√n)`) it is `O(1/√n) → 0`, so
GOSS is accurate precisely on large data. At equal total sampling budget `β`, random sampling is
`C_{0,β}` and GOSS is `C_{a,β-a}`. GOSS beats random sampling iff
`C_{0,β} > C_{a,β-a}`, equivalently
`α_a/√β > (1-a)/√(β-a)` with
`α_a = max_{A∪A^c}|g| / max_{A^c}|g|`, or
`β > a / (1 - ((1-a)/α_a)^2)`. The condition is easiest to satisfy when gradient magnitudes have
a wide range, because keeping the top set shrinks `max_{A^c}|g|`.

## EFB — reduce #feature

Sparse feature spaces have many mutually exclusive features (rarely nonzero together, e.g. one-hot
blocks). Pack each exclusive set into one bundle column → histogram build `O(#data × #feature) →
O(#data × #bundle)`, `#bundle ≪ #feature`, near-lossless.

- **Which to bundle (NP-hard).** Theorem: partitioning features into the fewest exclusive bundles
  is NP-hard. Given a graph, create one feature per vertex and one instance per edge; a feature is
  nonzero on an edge-instance iff its vertex is incident to that edge. Non-exclusivity is exactly
  graph adjacency, so an exclusive bundle is a color class. **Greedy (Alg. 3):** weighted-conflict graph, sort features by degree
  descending, place each into an existing bundle if its total conflict stays `≤ K = γ·n` else open
  a new bundle; `O(#feature²)` once before training. Cheaper ordering for millions of features:
  by nonzero-count (proxy for degree), no graph. Allowing a small conflict rate `γ` gives fewer
  bundles; **Proposition:** random `γ`-pollution changes the max variance gain by at most
  `[(1−γ)n]^{−2/3}` (sandwich `|V − V^γ| ≤ max_j |V_j − V^γ_j|`, then single-feature split-point
  perturbation), so small `γ` ⇒ negligible loss.
- **How to merge (Alg. 4).** Add per-feature bin offsets so exclusive features occupy disjoint bin
  ranges: A in `[0,10)`, B → `[10,30)`. A merged-bin value identifies the original feature and its
  bin; the bundle histogram is the concatenation of the originals', so every split is preserved.
  (Plus a sparse-aware optimization: per-feature nonzero tables make build `O(#non-zero-data)`,
  compatible with EFB.) Bundling also raises cache locality, an extra constant-factor speed-up.

## Final algorithm

```
# Once: EFB — greedy-color features into exclusive bundles (conflict budget K = γ·n),
#       merge each bundle into one column by disjoint bin offsets.
models ← {};  fact ← (1−a)/b;  topN ← a·|I|;  randN ← b·|I|
for i = 1 to d:                                   # boosting rounds
    preds ← models.predict(I)
    g ← loss.gradient(I, preds);  w ← {1,…,1}     # full-data gradients
    sorted ← indices sorted by |g| descending
    topSet  ← sorted[1:topN]                      # GOSS: keep large-gradient instances
    randSet ← random randN of sorted[topN+1:|I|]  #       sample small-gradient tail
    usedSet ← topSet ∪ randSet
    w[randSet] ← w[randSet] · fact                #       reweight tail by (1−a)/b
    newModel ← L(I[usedSet], −g[usedSet], w[usedSet])   # leaf-wise histogram tree on bundles
    models.append(newModel)
# split finder: per-bin Σ(g[,h]) histograms over bundle columns, parent−smaller-child subtraction,
#               search bin boundaries for max gain V(d) (or ½ ΣG²/(H+λ) − γ·T), grow leaf-wise.
```

## Working code (qlib deployment)

A thin qlib `LGBModel` wrapper hands feature matrices and parameters to the GBDT engine and trains
with early stopping. Hyperparameters are the Alpha158 CSI300 deployment config. `objective=mse`
means `g_i = yhat_i - y_i` (residual fitting). The large `lambda_l1`/`lambda_l2` shrink leaf values;
`num_leaves`/`max_depth` bound leaf-wise overfit; `subsample` and `colsample_bytree` are forwarded
sampling parameters in the default GBDT booster, not a request to use GOSS.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from qlib.model.base import ModelFT
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.model.interpret.base import LightGBMFInt
from qlib.workflow import R


class LGBModel(ModelFT, LightGBMFInt):
    """Faithful qlib LGBModel shape with the deployment parameters filled in."""

    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {
            "objective": loss,           # mse gives g_i = yhat_i - y_i
            "colsample_bytree": 0.8879,  # feature sub-sampling per tree
            "learning_rate": 0.2,        # per-tree shrinkage
            "subsample": 0.8789,         # forwarded bagging-fraction parameter
            "lambda_l1": 205.6999,       # L1 on leaf weights (soft-threshold on G)
            "lambda_l2": 580.9768,       # L2 on leaf weights (denominator H + lambda_l2)
            "max_depth": 8,              # depth cap on leaf-wise trees
            "num_leaves": 210,           # leaf budget (leaf-wise overfit guard)
            "num_threads": 20,
            "verbosity": -1,
        }
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        ds_l = []
        assert "train" in dataset.segments
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError("Empty data from dataset, please check your dataset config.")
                x, y = df["feature"], df["label"]
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")
                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError("Unsupported reweighter type.")
                ds_l.append((lgb.Dataset(x.values, label=y, weight=w, free_raw_data=False), key))
        return ds_l

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds
        )
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
            **kwargs,
        )
        for k in names:
            for key, val in evals_result[k].items():
                name = f"{key}.{k}"
                for epoch, metric in enumerate(val):
                    R.log_metrics(**{name.replace("@", "_"): metric}, step=epoch)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)
```
