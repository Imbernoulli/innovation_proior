## Research question

Can a relation-aware predictor exploit cross-stock structure — sector / concept membership, learned
relations, attention across instruments — to deliver consistently better next-day return rankings than
an instrument-independent model, with the data, labels, splits, and backtest held fixed? The single
thing being designed is the `CustomModel` that fills the qlib model interface (`fit(dataset)` /
`predict(dataset, segment="test")`); everything around it — the Alpha360 feature handler, the label,
the universes, the splits, the TopkDropout backtest — is frozen.

## Prior art before the first rung (the cross-sectional-modeling lineage)

The first rung reacts to a long line of single-stock and shallow cross-sectional predictors. These are
the methods the ladder climbs away from; each leaves a gap the next rung tries to close.

- **Per-stock linear / factor models (Ridge, OLS on engineered alphas).** Treat each `(stock, day)` row
  independently, fit one weight vector over the 360 Alpha360 features, predict the return. Cheap and
  stable, but they (a) impose a global linear response and (b) see every stock in isolation — no
  cross-stock information at all. Gap: linear and instrument-independent.
- **Gradient-boosted trees (GBDT) on the flat feature table.** The standard non-graph reference: a
  histogram-based, leaf-wise boosted forest over the same 360 features, treating each row independently.
  Captures non-linear feature interactions a linear model cannot, and is the strongest *tabular* model
  here — but it still has no notion that two rows on the same day are two co-moving stocks; the
  cross-section is invisible to it. Gap: non-linear but still instrument-independent.
- **Sequence encoders (LSTM / GRU on the `[60, 6]` window).** Reshape the 360 features back into a
  60-day × 6-channel sequence and run a recurrent encoder, taking the last hidden state as the stock's
  representation. This recovers the temporal structure the flat models flatten away, and is the natural
  backbone for everything graph-based. But on its own each stock's hidden state still only ever saw that
  one stock. Gap: temporal but not relational.
- **Stock-relation graph methods (GAT-style attention over a relation graph; concept-aware aggregation).**
  Build a graph over stocks — edges from shared sector / concept membership — and pass messages along it
  so a stock's prediction can borrow from related stocks. This is the family the ladder lives in. The
  open questions it leaves — whether the relations to attend over should be the *whole* day's cross-section
  or a curated graph, and whether concept membership should be a frozen edge set or a date-specific,
  recomputed relevance — are exactly what the rungs separate.

## The fixed substrate

The qlib benchmarking pipeline is frozen and must not be touched. **Features:** Alpha360 — 6 base series
(open/high/low/close/volume/vwap-style) over 60 trailing days = 360 features per `(stock, day)`, reshaped
to `[N, 60, 6]` for sequence models. **Label:** `Ref($close, -2) / Ref($close, -1) - 1`, the return from
T+1 to T+2 predicted at T. **Handler:** the default Alpha360 handler applies `RobustZScoreNorm` +
`Fillna` on features at inference and `DropnaLabel` + `CSRankNorm` on the label at training. **Splits:**
train 2008–2014, valid 2015–2016, test 2017–2020-08, fixed; **universes:** `csi300`, `csi100`,
`csi300_recent`. **Backtest:** `TopkDropoutStrategy`, top 50 / drop 5, with the fixed exchange costs.
**Auxiliary input:** a pre-downloaded stock-concept membership graph (`qlib_csi300_stock2concept.npy`, a
binary `(num_stocks, num_concepts)` matrix) and a `stock_index` map, exposed to the editable region
through `get_stock_index(instruments)` and `get_concept_matrix(stock_indices)` — a relation-aware model
may pull these in, an instrument-independent one ignores them.

## The editable interface

Exactly one class is editable — `CustomModel` in `custom_model.py` (the region between the FIXED graph-data
loaders above it and the end of the file), plus the dataset preprocessor block of `workflow_config.yaml`.
The contract is the qlib model interface:

- `fit(self, dataset)` — train on the `train` (and `valid`) segments. Data is pulled with
  `dataset.prepare(seg, col_set=["feature","label"], data_key=DataHandlerLP.DK_L)`; `df["feature"]` is an
  `(n_samples, 360)` DataFrame indexed by `(datetime, instrument)`, `df["label"]` is `(n_samples, 1)`.
- `predict(self, dataset, segment="test")` — return a `pd.Series` of scores indexed by
  `(datetime, instrument)` matching the requested segment's index (inference data via `DK_I`).

A relation-aware model batches **per day** — all stocks present on a date form one cross-section — using
the provided pattern (`groupby(level=0).size()` → `daily_index`), and may look up that day's concept
matrix via `get_concept_matrix(get_stock_index(instruments))`. Available imports: `torch`, `torch.nn`,
`numpy`, `pandas`, `lightgbm`, `sklearn`, `scipy`.

The starting point is the scaffold **default**: a per-stock Ridge regression that ignores the graph
entirely. Each rung replaces exactly this class (and, where the preprocessing must change, the handler
block).

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

Three universes — **csi300** (the main 300-name pool), **csi100** (the 100 largest, a harder, lower-breadth
cross-section), and **csi300_recent** (a recent out-of-window slice, hidden) — each scored by qlib's
`SigAnaRecord` and `PortAnaRecord`. Per universe, seven metrics, all *higher is better* as scored:
signal IC, ICIR, Rank IC, Rank ICIR; portfolio annualized return and information ratio; and max drawdown
(reported as a negative number, so closer to zero scores higher). Each universe's score is the equal-weight
mean of its seven sigmoid-mapped metrics; the task score is the geometric mean across the three universes.
Seed `42`.
