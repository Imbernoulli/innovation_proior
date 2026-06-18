**Problem.** The Transformer rung failed on *optimization*, not bias: a warmup-less constant-rate Adam
with patience-5 early stop on a faint signal early-stopped near noise, giving IC ~0.01 and a **negative**
portfolio information ratio across all three universes (it paid TopkDropout churn costs on a near-random
ranking). The fix wanted is a learner with no optimization fragility that is robust on noisy tabular data.

**Key idea.** A histogram-based, leaf-wise gradient-boosted decision tree (LightGBM). Treat the 60×6
window as a flat tabular row; fit an additive ensemble where each tree fits the gradient residual
(`g_i = ŷ_i − y_i` for MSE). No initialization to survive, no learning-rate schedule to thread; early
stopping is on the *tree count*. The trees discover nonlinear factor interactions automatically.

**Why.** Boosting has essentially no optimization fragility — shrinkage only slows things safely, and the
ensemble averages out noise. The price is losing the temporal inductive bias, but the engineered Alpha360
ratios carry per-column signal, so robustness + automatic interaction discovery is the better trade
against a near-zero-signal baseline. On this *dense* feature matrix, GOSS does not fire (the deployment
runs the default `gbdt` booster, so `subsample` is plain bagging — the `a=0` case) and EFB is largely
inert; the win comes from the histogram-leaf-wise engine and heavy regularization.

**Hyperparameters (qlib Alpha360 benchmark).** `objective=mse`, `learning_rate=0.0421`, `num_leaves=210`,
`max_depth=8`, `colsample_bytree=0.8879`, `subsample=0.8789`, `lambda_l1=205.6999`,
`lambda_l2=580.9768`, `num_threads=20`; `num_boost_round=1000`, `early_stopping_rounds=50`. The huge
L1/L2 penalties are the financial-data regularizer — they zero out splits with small gradient mass so the
trees act only on the strongest signal.

**Task-specific edit.** This rung *also edits the workflow processor block*: it sets `infer_processors:
[]` (removing the neural-model `RobustZScoreNorm` + `Fillna` that the default workflow applies), keeping
only label-side `DropnaLabel` + `CSRankNorm`. A tree splits on order statistics, so feature
standardization is irrelevant and the robust-z clipping would *remove* tail information the tree could
split on. The Transformer rung left the default processors unchanged; this one must not.

**What to watch.** The csi300 IR must turn *positive* — that is the falsifiable claim that signal formed
where the transformer's noise ranking failed. IC should climb from 0.0117 toward the high-0.03s. csi100's
portfolio return may stay weak even with a better IC (a small, hard universe where 50/100 names makes
churn bite), which would point the next rung at recovering ranking quality with a *robustly-trained*
temporal model.

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
