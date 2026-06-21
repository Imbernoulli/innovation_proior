## Research question

Can a stock-return predictor be made robust to *temporal* distribution shift — concept drift — while the universe, feature set, label, and backtest remain fixed? The substrate is the CSI300 universe, the Alpha158 factor set, the label `Ref($close,-2)/Ref($close,-1)-1`, and a qlib top-50/drop-5 backtest. The designed parts are the `CustomModel` (`fit`/`predict`) and, if needed, the dataset adapter/processor block. Evaluation runs across three temporal regimes — `csi300` (2017–2020), `csi300_shifted` (2016–2018), and `csi300_recent` (2019–2020) — so a method must generalize across *when* it is tested.

## Prior art / Background / Baselines

- **Empirical risk minimization with one model (single LSTM / GBDT).** A single predictor is trained by averaging loss over all historical samples.
- **Mixture-of-experts / conditional computation.** Several expert sub-models are combined by a per-input gate that selects or weights them for each input.
- **Domain-adaptation alignment (covariate shift, MMD/CORAL/DANN).** Feature distributions are aligned across domains so a shared conditional can be reused.

## Fixed substrate / Code framework

Frozen: the CSI300 instrument list; the **Alpha158** handler (158 engineered factors per stock per day, pre-normalized with `RobustZScoreNorm` and `Fillna`); the label; the `train=[2008,2014] / valid=[2015,2016] / test=[2017,2020]` segments (the three regimes are separate fixed workflow files); and the `PortAnaRecord` backtest (`TopkDropoutStrategy`, topk 50, n_drop 5, costs and benchmark fixed). The qlib `SignalRecord` / `SigAnaRecord` / `PortAnaRecord` chain computes the metrics.

## Editable interface

Two regions are editable. (1) `custom_model.py`: the `CustomModel(Model)` class with `fit(dataset)` and `predict(dataset, segment="test") -> pd.Series` indexed by `(datetime, instrument)`. (2) `workflow_config.yaml`: the dataset block (`dataset.class` / adapter) and the handler's processor block, so a method that needs a different dataset *view* — e.g., a sequence sampler or filtered feature set — can request it. Everything else (qlib init, segments, records, backtest) is fixed.

The starting point is the default Ridge-regression fill below. Each baseline or method replaces exactly this editable region (and, for sequence models, the dataset/processor block) and nothing else.

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

Three temporal regimes, each a fixed CSI300 workflow with a different test window: `csi300` (2017–2020), `csi300_shifted` (2016–2018), `csi300_recent` (2019–2020). Three seeds {42, 123, 456} for stochastic methods. Per regime, seven metrics — signal: IC, ICIR, Rank IC, Rank ICIR; portfolio: annualized return, information ratio, max drawdown — all higher-is-better (max drawdown is reported as a negative number, so closer to zero is better). The task aggregate is the geometric mean across the three regimes of each regime's equally-weighted (sigmoid-mapped) seven-metric mean, so a method that wins one regime but collapses in another is penalized.
