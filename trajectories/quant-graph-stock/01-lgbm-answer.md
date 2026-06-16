**Problem.** Before any graph, fix the strongest *instrument-independent* floor. The scaffold default is a
per-stock Ridge — a global linear response over 360 engineered alphas — which cannot express the
non-linear, regime-dependent feature interactions that drive returns. The floor should capture those
interactions while still, deliberately, treating each `(stock, day)` row in isolation, so that whatever a
relation-aware model buys is measured against a real baseline.

**Key idea.** Fill `CustomModel` with a faithful wrapper around qlib's official `LGBModel` — a
histogram-based, leaf-wise gradient-boosted decision tree. Boosting fits an additive sequence of shallow
trees to the residual; each depth-`d` tree expresses an order-`d` feature interaction for free, and the
histogram engine (per-bin `Σg/Σh`, parent−child subtraction, leaf-wise growth, second-order regularized
gain) trains it to convergence on the full training cross-section within budget. No concept matrix is ever
opened.

**Why.** Trees are scale-invariant (a split is a threshold), so per-feature normalization is a no-op —
hence the rung also blanks the neural-model inference processors and keeps only `DropnaLabel` +
`CSRankNorm` on the label, which rank-normalizes each day's labels cross-sectionally for the IC/Rank-IC
ranking objective. The very heavy `lambda_l1=205.70`, `lambda_l2=580.98` shrinkage is the published
Alpha360 setting and is essential: financial cross-sectional signal is faint, so 210-leaf trees must be
penalized hard to avoid fitting noise. The deliberate ceiling: every row is an independent example, so
two co-moving stocks on the same day are unrelated points — the cross-section is invisible.

**Hyperparameters (verbatim from qlib's `workflow_config_lightgbm_Alpha360.yaml`).** `objective=mse`,
`learning_rate=0.0421`, `num_leaves=210`, `max_depth=8`, `colsample_bytree=0.8879`, `subsample=0.8789`,
`lambda_l1=205.6999`, `lambda_l2=580.9768`, `num_threads=20`; `num_boost_round=1000`,
`early_stopping_rounds=50` on the valid segment.

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

The companion edit resets the editable handler block of `workflow_config.yaml` to the LightGBM Alpha360
preprocessing (no inference processors; label `DropnaLabel` + `CSRankNorm`):

```yaml
          infer_processors: []
          learn_processors:
            - class: DropnaLabel
            - class: CSRankNorm
              kwargs:
                fields_group: label
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
```
