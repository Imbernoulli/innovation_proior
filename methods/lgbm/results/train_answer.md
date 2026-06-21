Gradient boosting decision trees are the strongest learner I have on structured tabular data, but on the datasets I now face — millions of rows and a feature space running from hundreds to tens of millions of columns — training is simply too slow, and I need to know exactly where the time goes before touching anything. A GBDT is an additive sequence of regression trees: at round $t$ I have the ensemble $F_{t-1}$, I compute for every instance a gradient of the loss against its current prediction, $g_i = \partial L(y_i, F_{t-1}(x_i))/\partial F_{t-1}(x_i)$, fit a new tree to the negative gradients, and fold it in shrunk by a learning rate. The gradient pass is cheap; the cost lives in growing each tree, and inside that, in finding the best split at each node. To split a node I must evaluate, for every feature, the gain of every candidate split, and to get those I have to scan the node's data — so per tree the work scales like $\#\text{data} \times \#\text{feature}$. That product is the enemy.

Of the two ways to find splits, I build on the histogram path: bucket each continuous feature into a small fixed number of bins (say 255, so a bin index fits in one byte), and for a node make one pass over its rows accumulating, per bin, the sum of gradients and the instance count. Building histograms is $O(\#\text{data} \times \#\text{feature})$; searching the at most $\#\text{bin}-1$ boundaries is $O(\#\text{bin} \times \#\text{feature})$, negligible since $\#\text{bin} \ll \#\text{data}$. The build dominates utterly. A free trick helps: a parent's histogram equals the sum of its two children's, so I build only the smaller child's histogram and recover the sibling by subtraction in $O(\#\text{bin})$. But the headline is unchanged — building histograms is $O(\#\text{data} \times \#\text{feature})$, and the only way to go materially faster is to cut $\#\text{data}$ or $\#\text{feature}$ before the build. Doing either without losing accuracy is the whole difficulty. The obvious levers all fail: naive uniform row subsampling (stochastic gradient boosting) treats every instance as equally informative and measurably loses accuracy; AdaBoost-style importance sampling needs a maintained per-instance weight that GBDT does not have; and feature filtering by PCA or projection pursuit assumes feature redundancy that engineered tabular features usually do not have, so dropping any can hurt.

The method is LightGBM: a histogram-based, leaf-wise GBDT engine made fast by two reductions that attack the two factors of the build cost separately — Gradient-based One-Side Sampling for the rows and Exclusive Feature Bundling for the columns.

For the rows, the key realization is that GBDT does produce an importance signal every round, even though it stores no weight: the gradient magnitude itself. If $|g_i|$ is small the loss is nearly flat there and the instance is already well fit; if $|g_i|$ is large the instance is under-trained and there is error left to chase. So $|g_i|$ is a perfectly good per-round importance, computed fresh, no bookkeeping. The naive move — delete the small-gradient instances — fails, and instructively. The split gain
$$V_j(d) = \frac{1}{n}\left[\frac{\left(\sum_{x_{ij}\le d} g_i\right)^2}{n_l(d)} + \frac{\left(\sum_{x_{ij}> d} g_i\right)^2}{n_r(d)}\right]$$
depends on a *sum of gradients* over each side. The small gradients are individually tiny but there can be enormously many of them, so their summed mass is not negligible; deleting them shifts both $\sum g$ terms and both counts, biasing the gain and selecting wrong splits. The fix is GOSS: keep the large-gradient instances whole — there are few and each matters — but instead of discarding the tail, sample a controlled number from it and scale that sample back up to stand in for the whole tail. Sort by $|g_i|$ descending, take the top $a$-fraction as set $A$ and keep it intact; the remaining tail $A^c$ has $(1-a)n$ instances; sample $b\,n$ of them as set $B$. Inside the tail the inclusion fraction is $b/(1-a)$, so to make $B$ represent $A^c$ in the gradient sums the multiplier is forced to be its inverse, $(1-a)/b$. The estimated gain becomes
$$\tilde V_j(d) = \frac{1}{n}\left[\frac{\left(\sum_{A_l} g_i + \tfrac{1-a}{b}\sum_{B_l} g_i\right)^2}{n_l(d)} + \frac{\left(\sum_{A_r} g_i + \tfrac{1-a}{b}\sum_{B_r} g_i\right)^2}{n_r(d)}\right],$$
where $A$ enters with weight $1$ (never sampled) and the sampled tail is reweighted by $(1-a)/b$. Setting $a=0$ makes the tail the whole dataset, sampled and weighted by $1/b$ — exactly rescaled uniform sampling — so SGB is the degenerate case and the novelty lives entirely in $a>0$.

This needs to be provably near-lossless, not hand-waved. Bounding the error $E(d) = |\tilde V_j(d) - V_j(d)|$ on a fixed split, I strip the common $1/n$ by writing $U_j = n V_j$, so each unnormalized score is a sum of two terms (sum of gradients)$^2$/count with the *same* counts in both. Each side's difference of squares $\tilde S^2 - S^2$ factors as $(\tilde S - S)(\tilde S + S)$, where $\tilde S_l - S_l = \frac{1-a}{b}\sum_{B_l} g_i - \sum_{A^c_l} g_i$ is the gap between the rescaled sample and the true tail sum, giving $\tilde U_j - U_j = C_l(\cdot) + C_r(\cdot)$ with $C_l$ collecting the $(\tilde S_l + S_l)/n_l$ factor (the kept top set $A$ appears doubled there). Taking absolute values bounds everything by $\max\{|C_l|,|C_r|\}$ times the rescaled tail-sampling error. Both pieces are empirical-mean deviations: $B$ is drawn uniformly from $A^c$ with bounded summands (indicators in $[0,1]$, gradients at most $D_{A^c} = \max_{A^c}|g_i|$), so Hoeffding controls each, the $2D$ piece carrying the average side-gradient scale $D = \max(\bar g_l, \bar g_r)$. Collecting with the single constant $C_{a,b} = \frac{1-a}{\sqrt b}\max_{x_i\in A^c}|g_i|$, with probability $\ge 1-\delta$,
$$E(d) \le C_{a,b}^2\,\ln(1/\delta)\,\max\!\left\{\frac{1}{n_l(d)},\frac{1}{n_r(d)}\right\} + 2 D\, C_{a,b}\sqrt{\frac{\ln(1/\delta)}{n}}.$$
The rate is $O(1/n_l + 1/n_r + 1/\sqrt n)$, so on a non-pathological split with both sides at least $O(\sqrt n)$ it is $O(1/\sqrt n)$ and vanishes as $n$ grows — GOSS is accurate precisely on large data, which is exactly the regime that motivated it. It genuinely beats uniform sampling, too: at equal total budget $\beta$, random sampling carries $C_{0,\beta}$ and GOSS carries $C_{a,\beta-a}$, so GOSS wins iff $C_{0,\beta} > C_{a,\beta-a}$, i.e. $\alpha_a/\sqrt\beta > (1-a)/\sqrt{\beta-a}$ with $\alpha_a = \max_{A\cup A^c}|g_i|/\max_{A^c}|g_i|$, equivalently $\beta > a/(1-((1-a)/\alpha_a)^2)$ — easiest to satisfy when gradient magnitudes span a wide range, because pulling out the top $a$-fraction sharply lowers $\max_{A^c}|g_i|$ and raises $\alpha_a$.

For the columns I cannot simply drop features, but sparse spaces give me something better than redundancy: mutual exclusivity. In one-hot blocks and indicator features most entries are zero, and many feature pairs are essentially never nonzero on the same instance. If two features never fire together, at most one carries information per row, so I can store them in the same column with no collision — and pack a whole exclusive set into one bundle column, cutting the build from $O(\#\text{data} \times \#\text{feature})$ to $O(\#\text{data} \times \#\text{bundle})$ with no information lost, only relocated. Choosing which features bundle together is NP-hard: build a graph with one vertex per feature and an edge whenever two features collide; an exclusive bundle is an independent set, the minimum bundling is a minimum graph coloring. The reduction is direct — one feature per vertex, one instance per edge, a feature nonzero on an edge-instance iff its vertex is incident — so non-exclusivity is exactly adjacency and optimal bundling is optimal coloring. So I use a greedy degree-ordered coloring: build the weighted-conflict graph, sort features by degree descending (the conflict-prone ones placed first, while bundles are empty), and drop each feature into an existing bundle if its total conflict stays under a budget $K = \gamma n$, else open a new bundle; for millions of features, order by nonzero-count as a degree proxy and skip the graph entirely. Allowing a small conflict rate $\gamma$ rather than demanding perfect exclusivity cuts the bundle count sharply for almost no cost: a $\gamma$-collision is like randomly polluting a $\gamma$-fraction of a feature's values, and the maximum variance gain moves by at most $[(1-\gamma)n]^{-2/3}$ — proved by sandwiching $|V - V^\gamma| \le \max_j |V_j - V_j^\gamma|$ (since the clean optimum $V_{j_1} \ge V_{j_2}$ and the polluted optimum $V^\gamma_{j_2} \ge V^\gamma_{j_1}$) and then applying single-feature split-point perturbation. Merging exploits the histogram's discreteness: lay each bundle's features into *disjoint bin ranges* by adding per-feature offsets, so feature $A$ keeps $[0,10)$ and $B$ shifts to $[10,30)$, and a single merged-bin value identifies both the original feature and its bin, with $0$ meaning nothing fired. The bundle's histogram is then exactly the concatenation of the originals' histograms over their sub-ranges, so every split the unbundled features would have found is preserved. Bundling also packs scattered features into contiguous memory, raising cache locality for a free constant-factor speed-up, and it composes with a sparse-aware optimization that keeps per-feature nonzero tables to make the build $O(\#\text{non-zero-data})$.

The engine tying these together grows leaf-wise — best-first, always splitting the leaf with the largest loss reduction, which reaches lower training loss than level-wise at a fixed leaf budget, capped by `num_leaves` and `max_depth` to bound overfit — and generalizes the variance gain to the second-order objective: with per-leaf $G = \sum g$, $H = \sum h$, the optimal leaf value is the Newton step $w^* = -G/(H+\lambda_2)$ (L1 entering as a soft-threshold on $G$) and the split gain is $\tfrac{1}{2}[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma$, which for squared loss ($h_i = 1$) collapses back to the variance gain GOSS was analyzed on. The whole method is then: GOSS cuts the rows the build sees, EFB cuts the columns it builds over, histogram-with-subtraction and leaf-wise growth make each build and split cheap, and the second-order gain ties leaf values to the boosting objective. The shipped form sits inside a thin qlib wrapper that hands feature matrices and parameters to the engine; under `objective=mse` the gradient is $g_i = \hat y_i - y_i$ so the tree fits the residual, and the large `lambda_l1`/`lambda_l2` shrink leaf values while `num_leaves`/`max_depth` guard against leaf-wise overfit. The `subsample`/`colsample_bytree` parameters are forwarded bagging fractions in the default booster, not a request to switch on GOSS.

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
