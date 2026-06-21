The relation-aware model is the whole point of this ladder, but it has to beat *something*, and the honest floor is the strongest predictor that uses no cross-stock structure at all. The scaffold ships a per-stock Ridge — a single linear weight vector over the 360 Alpha360 features — and that is a weak hypothesis for a feature table this wide. The features are engineered alphas (momentum windows, volatility proxies, volume ratios) whose relationship to next-day return is neither linear nor additive: a momentum signal matters only in some volatility regime, a volume spike means one thing after a run-up and another after a drawdown, and linear regression cannot express "feature $A$ matters only when feature $B$ is high." So the right floor is a model that captures those non-linear, regime-dependent interactions while still, deliberately, seeing each $(stock, day)$ row in isolation — so that whatever a graph later buys me is measured against a real baseline rather than the trivial linear one.

I propose to fill `CustomModel` with a faithful wrapper around qlib's official `LGBModel`: a histogram-based, leaf-wise **gradient-boosted decision tree** (LightGBM). Boosting is an additive ensemble of shallow regression trees. At round $t$ I hold the current predictor $F_{t-1}$, compute for every instance the gradient of the loss against its current prediction, fit a new tree to the negative gradients, and fold it in shrunk by a learning rate. For the squared loss I use, the gradient is $g_i = F_{t-1}(x_i) - y_i$, so $-g_i$ is exactly the residual and each tree fits what the ensemble still gets wrong. The trees are the reason this class is right: each internal split is a threshold on a single feature and a leaf is a conjunction of such thresholds, so a tree of depth $d$ represents an order-$d$ feature interaction *for free* — precisely the "$A$ matters only when $B$ is high" structure the linear model could not reach — and boosting stacks hundreds of them, discovering which interactions matter without my having to name them, which on 360 alphas I could not do anyway.

The data shape is what makes the *engine* matter and not just the model class. The split criterion at a node is the variance gain in sum-of-gradient form,
$$V_j(d) = \frac{1}{n}\left[\frac{\left(\sum_{x_{ij}\le d} g_i\right)^2}{n_l(d)} + \frac{\left(\sum_{x_{ij}>d} g_i\right)^2}{n_r(d)}\right],$$
and evaluating it for every feature and every candidate threshold is where the time goes — per tree the work scales like $\#\text{data}\times\#\text{feature}$. The histogram method buckets each continuous feature into a small fixed number of bins (255, so a bin index fits in a byte) and makes one pass per node accumulating, per bin, the sum of gradients and the count; building the histogram is $O(\#\text{data}\times\#\text{feature})$ and searching the $\le \#\text{bin}-1$ boundaries is a rounding error next to it. A parent's histogram is the sum of its two children's, so the engine builds only the smaller child and recovers the sibling by subtraction in $O(\#\text{bin})$. Growth is leaf-wise — it splits the single leaf with the largest loss reduction rather than a whole depth level, reaching lower training loss for a fixed leaf budget, capped by `num_leaves` and `max_depth`. Leaf values follow the second-order regularized objective, $G=\sum g$, $H=\sum h$, optimal value $-G/(H+\lambda_2)$ with L1 acting as a soft-threshold on $G$. These properties are what make the forest accurate *and* cheap enough to train to convergence on the full 2008–2014 cross-section inside the harness budget.

The load-bearing part for *this* task is matching what the harness exposes, not the generic library default, and two things there are deliberately not textbook. First, the regularization is enormous and not optional: `lambda_l1 = 205.70`, `lambda_l2 = 580.98`, `num_leaves = 210`, `max_depth = 8`, `colsample_bytree = 0.8879`, `subsample = 0.8789`, `learning_rate = 0.0421`, with `num_boost_round = 1000` and early stopping at 50 rounds on the validation segment. Those $\lambda$ values look absurd by general-tabular standards, but they are the published Alpha360 settings and they exist because financial cross-sectional signal is faint and noisy: without heavy shrinkage the 210-leaf trees would fit microstructure noise in the training window and generalize to nothing on 2017–2020. The big $\lambda$ push most candidate splits' gains below the complexity charge, so the forest stays shallow in *effective* capacity even at 210 leaves, and the slow learning rate with up to 1000 rounds is the many-small-trees regime boosting needs when each tree chips only a little signal off the residual. I take these verbatim — they are the difference between a fair floor and a broken one.

Second, this rung must reset the dataset preprocessing, not just the model. The default handler carries neural-model preprocessing: `RobustZScoreNorm` + `Fillna` on features at inference and `CSRankNorm` on the label at training. The inference-time feature normalization belongs to the graph models that need clean tensors; it does not belong here, because a tree split is a threshold and any monotone rescaling of a feature leaves which rows fall on each side unchanged — per-feature normalization is a no-op for trees and only risks discarding information through the outlier clipping. So I blank the inference processors (`infer_processors: []`) and keep only `DropnaLabel` + `CSRankNorm` on the label. The `CSRankNorm` I keep deliberately: it rank-normalizes each day's labels cross-sectionally, which is exactly right for a ranking objective scored by IC and Rank IC — the model learns to order stocks *within a day* rather than to hit absolute return levels. There is no per-day batching and no concept-matrix lookup; the whole point of this rung is that it never opens the graph file. Its deliberate ceiling is that every row is an independent example: two co-moving stocks reacting to the same macro print on the same day are, to this model, two unrelated points in a 360-dimensional space, and the cross-section — the one structural fact the research question is built on — is invisible. That is the gap the next rung reaches for, by letting each day's stocks attend to each other.

```python
# =====================================================================
# EDITABLE: CustomModel -- implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    """LightGBM model -- faithful to qlib's official LGBModel (gbdt.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
    """

    def __init__(self):
        super().__init__()
        # Official benchmark kwargs (passed to lgb.train via self.params)
        self.params = {
            "objective": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "verbosity": -1,
        }
        self.early_stopping_rounds = 50
        self.num_boost_round = 1000
        self.model = None

    def _prepare_data(self, dataset):
        """Prepare LightGBM datasets -- matches LGBModel._prepare_data()."""
        import lightgbm as lgb

        ds_l = []
        for key in ["train", "valid"]:
            if key in dataset.segments:
                df = dataset.prepare(
                    key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                if df.empty:
                    raise ValueError(
                        "Empty data from dataset, please check your dataset config."
                    )
                x, y = df["feature"], df["label"]
                # Lightgbm need 1D array as its label
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError(
                        "LightGBM doesn't support multi-label training"
                    )
                ds_l.append(
                    (lgb.Dataset(x.values, label=y, free_raw_data=False), key)
                )
        return ds_l

    def fit(self, dataset: DatasetH):
        import lightgbm as lgb

        ds_l = self._prepare_data(dataset)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds
        )
        verbose_eval_callback = lgb.log_evaluation(period=20)
        evals_result = {}
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],  # training dataset
            num_boost_round=self.num_boost_round,
            valid_sets=ds,
            valid_names=names,
            callbacks=[
                early_stopping_callback,
                verbose_eval_callback,
                evals_result_callback,
            ],
        )

    def predict(self, dataset: DatasetH, segment="test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)
```

The companion edit resets the editable handler block of `workflow_config.yaml` to the LightGBM Alpha360 preprocessing (no inference processors; label `DropnaLabel` + `CSRankNorm`):

```yaml
          infer_processors: []
          learn_processors:
            - class: DropnaLabel
            - class: CSRankNorm
              kwargs:
                fields_group: label
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
```
