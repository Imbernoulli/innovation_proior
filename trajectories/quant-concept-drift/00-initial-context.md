## Research question

Can a stock-return predictor be made robust to *temporal* distribution shift — concept drift,
the change over time in the joint law of features and returns — while holding everything else
fixed: the CSI300 universe, the Alpha158 factor set, the label `Ref($close,-2)/Ref($close,-1)-1`,
and a fixed qlib top-50/drop-5 backtest? The single thing being designed is the `CustomModel`
(its `fit`/`predict`) plus, where a method needs it, the editable dataset-adapter / processor
block of the workflow. The model is evaluated under **three different temporal regimes** — the
same CSI300 universe split into three test windows (`csi300`: 2017–2020; `csi300_shifted`:
2016–2018; `csi300_recent`: 2019–2020) — so a method is rewarded only if it generalizes across
*when* it is tested, not just *what* it is tested on.

## Prior art before the first rung (the lineage the first baseline reacts to)

The first rung is a sequence model with explicit multi-pattern routing. It reacts to a line of
prior approaches to the non-stationary-prediction problem, each with a gap:

- **Empirical risk minimization on one model (a single LSTM / GBDT).** Fit one predictor on all of
  history by average loss. Gap: it bakes in the i.i.d. assumption — one fixed joint `P`, one
  relation `p(y|x)` — which the market violates; two documented cross-sectional effects (momentum
  and reversal) have *opposite* sign and rotate over time, so one parameter vector can only average
  contradictory relations and fits neither.
- **Mixture-of-experts / conditional computation (Jacobs et al. 1991; Shazeer et al. 2017).**
  Several expert sub-models and a gate that routes per input. The right *shape* — several relations,
  a router — but trained naively the gate collapses onto one expert (the rich-get-richer gating
  pathology), and a soft load-balancing penalty equalizes mass without saying *which* sample
  belongs to *which* expert. Gap: no mechanism that assigns each sample to the expert that fits it
  while preventing collapse.
- **Domain-adaptation alignment (covariate shift, Shimodaira 2000; MMD/CORAL/DANN, 2012–2016).**
  Align feature distributions across domains so a shared conditional is learned. Gap as posed for a
  *stream*: it needs the test density and a known train/test pair, and it has no notion of *when*
  inside a time series the distribution turns over — the temporal version of the shift is left
  unaddressed.

The two reference adaptive baselines on this task — a routing sequence model and a
temporal-alignment recurrent model — are the two answers to those gaps; a non-adaptive
gradient-boosted tree is the strong control they must beat.

## The fixed substrate

The qlib workflow is frozen except for the editable region. Fixed: the CSI300 instrument list;
the **Alpha158** handler (158 engineered factors per stock per day — rolling mean/std/max-min of
returns and volume, ROC momentum, K-line ratios, price–volume CORR/CORD, volatility VSTD/WVMA,
residual RESI/RSQR — pre-normalized with `RobustZScoreNorm` and `Fillna`); the label; the
`train=[2008,2014] / valid=[2015,2016] / test=[2017,2020]` segments (the three regimes are three
such fixed workflow files); and the `PortAnaRecord` backtest (`TopkDropoutStrategy`, topk 50,
n_drop 5, costs and benchmark fixed). The qlib `SignalRecord` / `SigAnaRecord` / `PortAnaRecord`
chain computes the metrics. (The task brief mentions Alpha360; the actual frozen workflow handler
is **Alpha158**, and every baseline is filled against Alpha158 — that is the contract.)

## The editable interface

Two regions are editable. (1) `custom_model.py` lines 16–103: the `CustomModel(Model)` class with
`fit(dataset)` and `predict(dataset, segment="test") -> pd.Series` (indexed by
`(datetime, instrument)`). (2) `workflow_config.yaml`: the dataset block (the `dataset.class` /
adapter, lines 19–26) and the handler's processor block (lines 32–45), so a method that needs a
different dataset *view* — a sequence sampler, a filtered feature set — can request it. Everything
else (qlib init, segments, records, backtest) is fixed.

The starting point is the scaffold **default fill**: a Ridge-regression `CustomModel` over the
default `DatasetH` / Alpha158 pipeline. Each rung replaces exactly this editable region (and, for
the sequence models, the dataset/processor block) and nothing else.

```python
# EDITABLE region of custom_model.py (lines 16-103) — default fill (Ridge baseline)
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
class CustomModel(Model):
    """Custom stock prediction model for concept drift adaptation.

    fit(dataset) trains; predict(dataset, segment="test") returns a pd.Series
    indexed by (datetime, instrument). The dataset is a qlib DatasetH with
    Alpha158 features (158 per stock per day), pre-normalized and NaN-filled.
    """

    def __init__(self):
        super().__init__()
        self.fitted = False
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)

    def fit(self, dataset: DatasetH):
        df_train = dataset.prepare(
            "train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        features = df_train["feature"].values
        labels = df_train["label"].values.ravel()
        mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
        features = features[mask]
        labels = labels[mask]
        self.model.fit(features, labels)
        self.fitted = True

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("Model is not fitted yet!")
        df_test = dataset.prepare(
            segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
        )
        features = df_test["feature"]
        index = features.index
        features_np = np.nan_to_num(features.values, nan=0.0)
        preds = self.model.predict(features_np)
        return pd.Series(preds, index=index, name="score")
```

## Evaluation settings

Three temporal regimes, each a fixed CSI300 workflow with a different test window: `csi300`
(2017–2020), `csi300_shifted` (2016–2018), `csi300_recent` (2019–2020). Three seeds {42, 123, 456}
where a method is stochastic. Per regime, seven metrics — signal: IC, ICIR, Rank IC, Rank ICIR;
portfolio: annualized return, information ratio, max drawdown — all higher-is-better (max drawdown
is reported as a negative number, so closer to zero is better). The task aggregate is the geometric
mean across the three regimes of each regime's equally-weighted (sigmoid-mapped) seven-metric mean,
so a method that wins one regime but collapses in another is penalized.
