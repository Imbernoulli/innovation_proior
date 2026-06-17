**Problem.** Both adaptive sequence models hit the same ceiling: TRA and AdaRNN are essentially tied
on aggregate (gmean ≈0.560 vs ≈0.561) and *both* collapse on the far `csi300_recent` regime
(info ratio ≈0.47–0.48), with AdaRNN adding seed instability (seed-42 info ratio 0.004 there). Two
independent adaptation mechanisms converged, so the bottleneck is not the mechanism — it is the
fit-able signal a sequence model can pull from a thin, noisy CSI300 panel while paying adaptation
overhead. The next rung drops adaptation and fits the strongest plain, full-feature control.

**Key idea (LightGBM — strong non-adaptive control).** A heavily-regularized gradient-boosted tree
over all 158 raw Alpha158 factors. Trees split on individual-factor *thresholds*, so they are
invariant to monotone factor rescaling and far less sensitive to the marginal `P(x)` drift that made
TRA read a stale router signal and AdaRNN mis-align; large `lambda_l1`/`lambda_l2` shrink leaf values
hard, the right defense against the noise that dominates the panel; and the model spends *none* of its
capacity on adaptation machinery and *all* of it on cross-sectional signal over the full feature set.

**What the harness runs (vs the generic method).** This is a faithful qlib `LGBModel`; the params that
look like adaptation are not. `colsample_bytree=0.8879` / `subsample=0.8789` are the booster's vanilla
per-tree feature/row sub-sampling forwarded to `lgb.train` — **not** gradient-based one-side sampling
and **not** temporal-domain adaptation. The processor edit sets `infer_processors: []` (drops
`RobustZScoreNorm`/`Fillna` the neural baselines needed) because trees gain nothing from z-scoring and
LightGBM handles NaNs natively — so the tree sees the **raw** Alpha158 factors over the default
`DatasetH` (no sequence sampler, no domain split). Deterministic, hence identical across seeds.

**Hyperparameters (official Alpha158 CSI300 benchmark).** `objective="mse"` (so `g_i = ŷ_i − y_i`,
residual fitting), `learning_rate=0.2`, `num_leaves=210`, `max_depth=8`, `lambda_l1=205.6999`,
`lambda_l2=580.9768`, `colsample_bytree=0.8879`, `subsample=0.8789`, `num_threads=20`, `verbosity=-1`;
`num_boost_round=1000`, `early_stopping_rounds=50` against the 2015–2016 valid segment.

**What to watch.** Decisive regime is `csi300_recent`, where both adaptive models died: expect the tree
to *not* die there (info ratio comfortably above 1.0, annualized return well above their ≈0.03–0.04),
because raw-threshold splits port to a distant regime better than a drifted neural encoder. Expect best
signal (IC, rank IC) and shallowest max drawdown on the long `csi300` window from the heavy
regularization. If all three hold, the aggregate puts LGBM above both adaptive models and the lesson is
subversive: the best response to concept drift here is a strong regularized non-adaptive learner, not
an explicit adaptation mechanism. (Refuted only if LGBM's far regime also nears 0.5 — then the ceiling
is the data.)

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
