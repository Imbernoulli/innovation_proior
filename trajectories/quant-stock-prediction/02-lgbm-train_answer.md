The Transformer rung came back the way I had braced for, and the numbers say *which* failure it was. On csi300 the IC is 0.0117 and the Rank IC 0.0350 — about a quarter of what a working temporal model reaches — and, the sharper tell, the portfolio information ratio is **−0.44** with annualized return −0.036; csi100 is worse (IR −1.01) and csi300_recent's IC is essentially zero (0.0014, IR −1.09). A negative IR across all three universes is exactly the signature I called in advance: a near-noise ranking still gets forced through TopkDropout to hold fifty names and churn five a day, so it pays transaction costs on noise and the return goes negative. The signal never formed — and the cause was the optimization, not the inductive bias. A constant $10^{-4}$ Adam with no warmup, patience-5 early stop on a faint signal almost certainly early-stopped on a barely-trained model. So the lesson is not "add more architecture"; it is the opposite — I want a learner that needs *no* delicate optimization: no learning-rate schedule to thread, no initialization to survive, no early-stop gamble, and that is robust on noisy tabular data out of the box.

I propose a **histogram-based, leaf-wise gradient-boosted decision tree (LightGBM)**. A GBDT is an additive sequence of regression trees. At round $t$ the current ensemble is $F_{t-1}$; I compute for every instance the gradient of the loss against its current prediction, $g_i = \partial L(y_i, F_{t-1}(x_i))/\partial F$, fit a new tree to the negative gradients, and fold it in shrunk by a learning rate. For squared loss $g_i = \hat y_i - y_i$, so each tree fits the residual $y_i - \hat y_i$ — the classic residual-fitting form. The reason this is the directed fix against the measured failure is that boosting has essentially no optimization fragility: there is no global initialization that can poison the whole run, no single learning rate whose miscalibration kills convergence (shrinkage is a per-tree scalar that only ever slows things down safely), early stopping is on the number of *trees* — a discrete, monotone count — rather than a gamble on a noisy validation curve, and the trees discover nonlinear feature interactions automatically. On a low signal-to-noise regression where the transformer's training fragility was the whole failure, swapping in a learner with no optimization fragility is exactly right.

What the trees see is the one place this rung differs from a naive "just use a tree," and it is worth being precise. The transformer treated the 360 features as a 60×6 sequence; the tree treats them as a flat tabular row — six base ratios at sixty lags, each a column — and is blind to the fact that column 5 and column 65 are the same ratio one day apart. But on this data that blindness may not hurt much, because the engineered Alpha360 ratios already carry the relevant information per-column, and the tree's job is to find which lags and which interactions predict the forward return. A tree splitting on "the 1-day return is high AND the 20-day volume ratio is low" is exactly the kind of nonlinear factor interaction a hand-built factor model would have to specify by hand. So the loss of the temporal bias is real, but the gain in robustness and automatic interaction discovery is the better trade against a near-zero-signal baseline.

The engine is what makes this tractable. The cost of growing each tree is dominated by split finding: to split a node I need, for every feature, the gain of every candidate split, and that scans the node's data, so per-tree work is $O(\#\text{data} \times \#\text{feature})$. The **histogram** engine cuts this down — bucket each continuous feature into a small fixed number of bins (255, so a bin index fits in a byte), and for a node make one pass accumulating per-bin sum-of-gradients and count. Searching splits is then $O(\#\text{bin} \times \#\text{feature})$, a rounding error next to the build. A parent node's histogram is the sum of its two children's, so I build only the *smaller* child's histogram and recover the sibling by subtraction in $O(\#\text{bin})$. On top of that the engine grows **leaf-wise** — best-first, splitting the single leaf with the largest loss reduction at each step — which reaches lower training loss than level-wise at a fixed leaf budget, capped by `num_leaves` and `max_depth` to bound overfit.

The split criterion is the variance gain in sum-of-gradient form,
$$V_j(d) = \frac{1}{n}\left[\frac{\big(\sum_{x_{ij}\le d} g_i\big)^2}{n_l(d)} + \frac{\big(\sum_{x_{ij}>d} g_i\big)^2}{n_r(d)}\right],$$
which a second-order view generalizes to leaf value $w^\star = -G/(H+\lambda_2)$ with $G = \sum g$, $H = \sum h$ and an L2 penalty $\lambda_2$, L1 entering as a soft-threshold on $G$. For squared loss the Hessian is 1 and this reduces to the plain variance gain. The two famous accelerations the engine carries — GOSS, which keeps the large-gradient (under-trained) rows whole and samples the small-gradient tail reweighted by $(1-a)/b$ to keep the gain unbiased, and EFB, which packs mutually exclusive sparse features into one column by disjoint bin-offset ranges via greedy graph-coloring under a small conflict budget — are the reason the method is *fast* in general. But I should be honest about how much of that fires *here*. The deployment runs the default `gbdt` booster, not `goss`, so the row-sampling reduction is *not* the GOSS importance sampler: `subsample`/`bagging_fraction` just feeds the engine's plain bagging machinery (uniform row subsampling per round — the $a=0$ degenerate case of GOSS). EFB still applies internally wherever the columns are sparse and exclusive, but Alpha360 features are dense continuous ratios, so its column packing is largely inert on this input. The accuracy and robustness here come from the histogram-leaf-wise engine and the regularization, not from GOSS/EFB on this dense matrix.

The hyperparameters are the official qlib Alpha360 LightGBM benchmark and every one is doing real regularization work, pushed toward *more* regularization than a default tabular task would use, because the IC ceiling here is low and the failure mode is variance, not bias. `learning_rate = 0.0421` is heavy shrinkage, so each of up to `num_boost_round = 1000` trees nudges the prediction gently and the ensemble averages out noise rather than chasing it; early stopping at patience 50 on the validation set cuts the round count before it overfits. `num_leaves = 210` with `max_depth = 8` allows fairly expressive trees, but the two L-penalties are enormous — `lambda_l1 = 205.7`, `lambda_l2 = 580.98` — and that is deliberate and characteristic of financial data: with such large penalties on the leaf weights, the soft-threshold on $G$ zeros out splits whose gradient mass is small, so the trees act only on the strongest, most repeated signal and ignore the vast noise floor. `colsample_bytree = 0.8879` and `subsample = 0.8789` (with the default bagging) add column- and row-level stochasticity so successive trees decorrelate. `objective = mse` matches the regression label.

There is one task-specific edit that matters, and it is exactly the kind of same-named-but-different detail I must get right: this rung does *not* leave the workflow processors alone. The default workflow applies a neural-model preprocessing block — `RobustZScoreNorm` (clip-and-standardize the features) and `Fillna` as `infer_processors`. Those are there for the gradient-descent models, which need standardized inputs to train stably. A tree does not: it splits on order statistics, so monotone per-feature rescaling is irrelevant to it, and the robust-z clipping actually *removes* information the tree could split on (the clipped tails). So this edit sets `infer_processors: []` — feed the trees the raw features — and keeps only the label-side `DropnaLabel` + `CSRankNorm`. This is a genuine deviation from the Transformer rung, which used the default processor block unchanged, and it is the correct one: the preprocessing that *helped* the neural model would quietly *hurt* the tree.

So the delta from the Transformer rung is total: I throw out the sequence model and its delicate Adam training, treat the 60×6 window as a flat tabular row, fit a heavily-regularized histogram GBDT that needs no warmup, no initialization luck, and no early-stop gamble, and strip the neural-only feature normalization so the trees see the raw signal. The falsifiable claim is that the csi300 IR must go *positive* — the transformer's −0.44 was a near-noise ranking paying churn costs, and a GBDT that actually forms signal should rank well enough to earn the costs back; if csi300 IR returns negative, the robustness argument is wrong. The IC should climb substantially from 0.0117 toward the high-0.03s / low-0.04s a working model reaches. Where I am genuinely uncertain is csi100: a smaller, harder universe where holding fifty of a hundred names makes the TopkDropout churn bite harder, so its portfolio return may stay weak even as its IC improves — and if so, the diagnosis for the final rung is that the tree's per-row tabular view, however robust, leaves ranking quality on the table that a *robustly-trained* temporal model could recover.

```python
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    """LightGBM model — faithful to qlib's official LGBModel (gbdt.py).

    Hyperparameters from official Alpha360 benchmark:
    examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml
    """

    def __init__(self):
        super().__init__()
        # Official Alpha360 benchmark kwargs (passed to lgb.train via self.params)
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
        """Prepare LightGBM datasets — matches LGBModel._prepare_data()."""
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

The workflow processor block is reset to the LightGBM Alpha360 pipeline (replacing the editable
`infer_processors`/`learn_processors`/`label` region of `qlib/workflow_config.yaml`):

```yaml
          infer_processors: []
          learn_processors:
            - class: DropnaLabel
            - class: CSRankNorm
              kwargs:
                fields_group: label
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
```
