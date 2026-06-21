AdaRNN settles the question the last two rungs were circling, and not in the direction the drift framing wanted. On `csi300_shifted` it does well — mean IC 0.0525, info ratio 1.66, even edging TRA's 1.59 — but on `csi300_recent`, the regime that already broke TRA, it stays broken: mean info ratio 0.47, essentially tied with TRA's 0.48, and now with the seed instability I predicted (seed 42 at info ratio 0.004 and annualized return 0.0003, a near-dead run, while seed 456 recovers to 1.12). So explicit two-half temporal alignment — exactly the "learn the invariant" move — did *not* rescue the far regime; it reproduced the same collapse plus variance. And the aggregate tells the deeper story: across the three regimes, AdaRNN ($\approx 0.561$) and TRA ($\approx 0.560$) are essentially tied, two independent adaptation mechanisms — routing within a distribution, aligning across halves — converged to the same ceiling. The bottleneck is therefore not the adaptation *mechanism*; it is the fit-able signal a sequence model can extract from a thin, noisy CSI300 panel while paying adaptation overhead (a 20-factor `FilterCol` slice for TRA, a single-timestep GRU for AdaRNN). So I am going to do the thing the drift framing resists: drop sequence-model adaptation entirely and fit the strongest plain, full-feature, non-adaptive tabular learner, and let it be the control the adaptive models must beat.

I propose LightGBM as that control — a heavily-regularized gradient-boosted tree over all 158 raw Alpha158 factors. A GBDT is an additive sequence of regression trees. At round $t$, with current ensemble $F_{t-1}$, each sample gets a gradient $g_i = \partial L(y_i, F_{t-1}(x_i))/\partial F$; for the squared loss ($\texttt{objective="mse"}$) this is $g_i = \hat y_i - y_i$, so $-g_i$ is the residual and each round fits a tree to the residuals, folded in shrunk by the learning rate. Inside one tree the split maximizes the variance gain, which in sum-of-gradient form is

$$V_j(d) = \frac{1}{n}\left[\frac{(\sum_{x_{ij}\le d} g_i)^2}{n_l} + \frac{(\sum_{x_{ij}>d} g_i)^2}{n_r}\right],$$

searched per feature over candidate thresholds $d$. LightGBM does this on bucketed histograms — each factor binned into $\le 255$ bins, one pass per node accumulating per-bin gradient sums, the sibling histogram recovered by parent-minus-smaller-child subtraction — and grows leaf-wise, always splitting the leaf with the largest gain, which is why $\texttt{num\_leaves}$ and $\texttt{max\_depth}$ are both capped to bound leaf-wise overfit.

Why is this the right control for return prediction across regimes? Three reasons that bear directly on the failure I just saw. First, trees split on *thresholds* of individual factors, so they are invariant to any monotone rescaling of a factor — the split $x_{ij}\le d$ cares only about rank order, not scale or mean. That is the crux for drift: temporal covariate shift moves $P(x)$ — the level, spread, and correlations of the factors drift between 2008–2014 and 2019–2020 — but a relation expressed as a rank threshold on a single factor is far more portable across that shift than a neural encoder whose first layer takes a fixed linear combination of all 158 factors, a combination whose meaning changes the moment the factor marginals move. The very drift that made TRA's router read a stale error-history signal and AdaRNN align two halves that don't span the test regime does not move a tree's split points the same way; the tree's inductive bias is *already* the kind of robustness the adaptive models tried to learn. Second, the GBDT sees the *full* 158-factor view, not a curated slice — no `FilterCol` to 20 columns, no collapse to a single timestep — so it loses none of the factor information the adaptive models discarded. Third, boosting with heavy L1/L2 leaf regularization is a strong defense against the noise that dominates this panel: with leaf weight $w^* = -G/(H+\lambda_2)$ and L1 entering as a soft-threshold on $G$, large penalties shrink every leaf value toward zero, so the trees commit only to splits whose gradient mass clearly survives the regularizer — exactly the right posture on low-SNR data where the unregularized optimum would memorize idiosyncratic returns.

A couple of the scaffold's hyperparameters look like adaptation but are not, and that distinction is load-bearing. The fill is a faithful qlib `LGBModel` at the official Alpha158 CSI300 benchmark: $\texttt{learning\_rate}=0.2$, $\texttt{num\_leaves}=210$, $\texttt{max\_depth}=8$, and the two that matter most here, $\texttt{lambda\_l1}=205.6999$ and $\texttt{lambda\_l2}=580.9768$ — enormous leaf-weight penalties that are precisely the regularization keeping the trees off the unlearnable residuals. $\texttt{colsample\_bytree}=0.8879$ and $\texttt{subsample}=0.8789$ are the booster's own per-tree feature and row sub-sampling, forwarded to `lgb.train`; they are *not* a request for the gradient-based one-side sampling LightGBM is famous for, and *not* temporal-domain adaptation — they are vanilla stochastic-GBDT knobs adding a little member-to-member decorrelation. Training uses $\texttt{num\_boost\_round}=1000$ with $\texttt{early\_stopping\_rounds}=50$ against the 2015–2016 validation segment, so the tree count is chosen by validation, and it is deterministic, hence identical across seeds.

The data view is the other deliberate choice, and it is the right one for trees. The fill edits the processor block to $\texttt{infer\_processors: []}$ — it drops the `RobustZScoreNorm` and `Fillna` the neural baselines depend on, keeping only the label processors (`DropnaLabel`, `CSRankNorm`). That is correct GBDT practice: trees split on raw factor thresholds and gain nothing from per-feature z-scoring (which would, if anything, smear the natural split points), and LightGBM handles NaNs natively by learning a default direction, so there is nothing to fill. The tree therefore sees the *raw* Alpha158 factors over the default `DatasetH` — no sequence sampler, no domain split, no router, by design.

The decisive regime is `csi300_recent`, where both adaptive models died at info ratio $\approx 0.47$–$0.48$. My falsifiable prediction is that the tree, seeing all factors and immune to marginal rescaling, will *not* die there — info ratio comfortably above 1.0 and annualized return well above the adaptive models' $\approx 0.03$–$0.04$ — because a relation learned from raw thresholds is more portable to a distant regime than a drifted neural encoder. On the long `csi300` window I expect the best signal of the three (higher IC and rank IC) with a markedly shallower max drawdown, the heavy regularization making the portfolio steadier than the adaptive models' (whose drawdowns ran to $-0.10$). If those hold, the aggregate gmean puts LightGBM above both adaptive models and the lesson is sharp and a little subversive: on a thin, noisy CSI300 panel the most effective response to concept drift among these baselines is not an explicit drift-adaptation mechanism but a strong, heavily-regularized, full-feature non-adaptive learner whose inductive bias is already robust to the marginal shift. The symmetric failure that would refute it is LGBM's far regime also landing near 0.5 — then the ceiling is the data, not the model. And if LightGBM does win as the strong control, the natural next move is not to abandon it but to make *it* drift-robust by ensembling it deliberately — reweighting the samples it learns from and the features it uses.

```python
# EDITABLE region of custom_model.py (lines 16-103) — step 3: LightGBM model
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    """LightGBM model — faithful to qlib's official LGBModel (gbdt.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
    """

    def __init__(self):
        super().__init__()
        # Official benchmark kwargs (passed to lgb.train via self.params)
        self.params = {
            "objective": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.2,
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

The editable handler processor block (workflow_config.yaml lines 32-45) is set so trees see raw
factors (empty infer processors; NaNs handled natively by LightGBM):

```yaml
# workflow_config.yaml lines 32-45 — handler processors (raw features for trees)
          infer_processors: []
          learn_processors:
            - class: DropnaLabel
            - class: CSRankNorm
              kwargs:
                fields_group: label
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
```
