## Research question

Daily cross-sectional return prediction on Chinese equity universes: at each trading day, score every
stock in the universe by its predicted next-period excess return, rank by that score, and feed the
ranking into a fixed portfolio routine. The single thing being designed is the **predictive model** — a
`CustomModel` that maps a fixed feature view to one score per stock-day. Everything around it — the
features (Alpha360), the label, the train/valid/test date splits, and the backtest — is frozen by the
workflow. The question is whether one reusable model can deliver consistently strong ranking signal
across heterogeneous universes (`csi300`, `csi100`, `csi300_recent`) under that common protocol.

## Prior art before the first rung (cross-sectional forecasting lineage)

The first rung reacts to the standard ways this forecasting problem has been attacked; the fixed
substrate below is the harness those approaches plug into.

- **Hand-engineered factor models + linear regression (Fama–French 1993 lineage, and the practitioner
  "alpha factor" tradition).** Build a handful of economically motivated cross-sectional factors and
  regress forward return on them. Interpretable and cheap, but the signal lives entirely in the
  hand-chosen factors; it cannot discover interactions among hundreds of raw price/volume ratios, and a
  single global linear map underfits a noisy, non-stationary cross-section. Gap: capacity and
  automation — the model is only as good as the human factor set.
- **Gradient-boosted decision trees on engineered features (Friedman 2001; the GBDT tradition).** An
  additive ensemble of regression trees fit to gradient residuals, which discovers nonlinear feature
  interactions automatically and is the consensus high-accuracy learner on tabular data. But each row
  is treated as an i.i.d. tabular sample: the 360 features are a flat vector with no notion that they
  are six base ratios unrolled over sixty days, so the *temporal* structure of the window is invisible
  to the trees. Gap: blind to sequence order within the feature window.
- **Plain feedforward / shallow nets on the flat feature vector.** More capacity than a linear map, but
  the same blindness to the time axis, and prone to overfitting the noise without strong
  regularization. Gap: no temporal inductive bias, fragile on noisy financial data.

## The fixed substrate

A `qlib` workflow is frozen and must not be touched. It supplies:

- **Features**: Alpha360 — 360 features per stock-day, which are 6 base ratios (open/close/high/low/
  volume/vwap, normalized to the latest close) over 60 days of history. The flat `[N, 360]` vector can
  be reshaped to a sequence with `x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]` (60 time steps of
  6 features) for any temporal model.
- **Label**: `Ref($close, -2) / Ref($close, -1) - 1` — the return from T+1 to T+2, predicted at T.
- **Universes / splits**: `csi300`, `csi100`, `csi300_recent`; instruments and date ranges are fixed by
  the workflow YAML (train 2008–2014, valid 2015–2016, test 2017–2020-08 for csi300).
- **Backtest**: `TopkDropoutStrategy`, top 50 / drop 5, run by the workflow runner with fixed costs.
- **Processors**: the default workflow applies `RobustZScoreNorm` + `Fillna` to features
  (`infer_processors`) and `DropnaLabel` + `CSRankNorm` to the label (`learn_processors`) — a
  neural-model preprocessing block. A model may change the editable processor region of the YAML if its
  input view requires it (e.g. a tree model that prefers raw, un-normalized features).

## The editable interface

Exactly one class is editable — `CustomModel` in `custom_model.py` (the `fit`/`predict` body), plus two
editable blocks of the workflow YAML (the dataset/processor region). The contract is the standard `qlib`
model interface: `fit(dataset)` trains on the `"train"`/`"valid"` segments; `predict(dataset,
segment="test")` returns a `pd.Series` of scores indexed by `(datetime, instrument)` matching the
segment's index. Data comes out via `dataset.prepare(seg, col_set=["feature","label"], data_key=...)`;
available imports inside the class are `torch`, `numpy`, `pandas`, `lightgbm`, `sklearn`, `scipy`.

The starting point is the scaffold default: a **Ridge regression** on the flat 360-dim feature vector.
Each rung on the ladder replaces exactly this class (and, where noted, the processor region).

```python
# EDITABLE region of custom_model.py — default fill (Ridge regression baseline)
class CustomModel(Model):
    """Default: Ridge regression on the flat 360-dim Alpha360 feature vector."""

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

Reported per universe (`csi300`, `csi100`, `csi300_recent`), all from `qlib`'s standard
`SignalAnalysisRecord` and `PortAnaRecord`:

- **Signal quality**: IC, ICIR, Rank IC, Rank ICIR — higher is better. (IC is the daily
  cross-sectional correlation of score with realized return, averaged over the test period; ICIR is its
  mean/std; the Rank variants use Spearman rank correlation, which is what the TopkDropout ranking
  actually consumes.)
- **Portfolio**: annualized return and information ratio — higher is better; max drawdown — closer to
  zero is better.

Seed 42 is the reference run for the ladder; the leaderboard records additional seeds where available.
