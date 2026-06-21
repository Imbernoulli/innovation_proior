## Research question

Can the `CustomModel` implementation be improved to better predict next-day return rankings within the fixed qlib pipeline — keeping the Alpha360 features, label, universes, splits, and TopkDropout backtest unchanged?

## Prior art / Background / Baselines

- **Per-stock linear / factor models (Ridge, OLS).** Fit a linear mapping from the 360 Alpha360 features to next-day return, treating each `(stock, day)` independently.

- **Gradient-boosted trees (GBDT) on the flat feature table.** Train a histogram-based boosted forest over the same 360 features per row. The histogram engine bins each feature, accumulates gradient/hessian statistics per bin, and grows trees leaf-wise with second-order regularized gain.

- **Sequence encoders (LSTM / GRU).** Reshape Alpha360 into `[N, 60, 6]` windows and encode each stock's own temporal history with a recurrent network.

- **Stock-relation graph methods (GAT-style attention over a static relation graph).** Build a graph from shared sector / concept memberships and propagate messages so related stocks can influence each other's representations.

## Fixed substrate / Code framework

The qlib pipeline is frozen. **Features:** Alpha360 — 6 series (open/high/low/close/volume/vwap-style) over 60 trailing days, 360 features per `(stock, day)`, reshaped to `[N, 60, 6]` for sequence models. **Label:** `Ref($close, -2) / Ref($close, -1) - 1`. **Handler:** default Alpha360 handler applies `RobustZScoreNorm` + `Fillna` on features and `DropnaLabel` + `CSRankNorm` on the label. **Splits:** train 2008–2014, valid 2015–2016, test 2017–2020-08. **Universes:** `csi300`, `csi100`, `csi300_recent`. **Backtest:** `TopkDropoutStrategy`, top 50 / drop 5, fixed exchange costs. **Auxiliary input:** pre-downloaded stock-concept matrix (`qlib_csi300_stock2concept.npy`) and `stock_index` map, exposed through `get_stock_index(instruments)` and `get_concept_matrix(stock_indices)`; relation-aware models may use them, instrument-independent models ignore them.

## Editable interface

Only `CustomModel` in `custom_model.py` (between the fixed graph-data loaders and the end of the file) and the dataset-preprocessor block of `workflow_config.yaml` may be edited. The contract is qlib's model interface:

- `fit(self, dataset)` — train on `train` (and optionally `valid`). Data is prepared with `dataset.prepare(seg, col_set=["feature","label"], data_key=DataHandlerLP.DK_L)`; `df["feature"]` is `(n_samples, 360)` indexed by `(datetime, instrument)`, `df["label"]` is `(n_samples, 1)`.
- `predict(self, dataset, segment="test")` — return a `pd.Series` of scores indexed by `(datetime, instrument)` for the requested segment (inference via `DataHandlerLP.DK_I`).

Relation-aware models batch per day (`groupby(level=0).size()` → `daily_index`) and may look up the concept matrix via `get_concept_matrix(get_stock_index(instruments))`. Available imports: `torch`, `torch.nn`, `numpy`, `pandas`, `lightgbm`, `sklearn`, `scipy`.

The starting scaffold is a per-stock Ridge regression that ignores the graph.

```python
# EDITABLE region of custom_model.py — default fill (per-stock Ridge, graph ignored)
class CustomModel(Model):
    """Default baseline: Ridge regression over the flat 360 features, no graph."""

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
        mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))   # drop NaN rows
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

Three universes — `csi300` (300-name pool), `csi100` (100 largest), and `csi300_recent` (recent out-of-window slice) — are scored by qlib's `SigAnaRecord` and `PortAnaRecord`. Per universe, seven metrics, all higher-is-better after scoring: signal IC, ICIR, Rank IC, Rank ICIR; annualized return; information ratio; and max drawdown (reported as a negative number, so closer to zero scores higher). Each universe score is the equal-weight mean of sigmoid-mapped metrics; the task score is the geometric mean across universes. Seed `42`.
