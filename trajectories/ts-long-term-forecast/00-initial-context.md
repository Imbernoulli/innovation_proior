## Research question

Multivariate long-term forecasting: given a fixed look-back window of every channel of a multivariate
series, predict the next horizon for every channel. The one thing being designed is the **forecasting
architecture** — the map from a `[batch, seq_len, enc_in]` history to a `[batch, pred_len, c_out]`
horizon — and the question is which inductive bias (linear decomposition, channel-independent patch
attention, cross-variate attention, multi-scale mixing, endogenous/exogenous separation) generalizes
across heterogeneous datasets at one fixed `seq_len=96 / pred_len=96` setting. Everything around the
model — the data pipeline, the standardization, the optimizer, the metric computation — is frozen by
the Time-Series-Library protocol; only the model class changes.

## Prior art before the first rung (the temporal-attention lineage the linear baseline reacts to)

The first rung is a linear forecaster, and it is a deliberate reaction to a line of ever-heavier
attention-based forecasters. These are the methods that precede the ladder; the linear baseline is the
falsifying control built to ask how much of their reported accuracy the attention machinery actually
earned.

- **Informer (Zhou et al., AAAI 2021).** A Transformer with ProbSparse attention (drop most query–key
  pairs), a distilling encoder, and a generative-style decoder that emits the whole horizon in one
  forward pass. Cuts the O(L²) attention cost to O(L log L). Gap: the saving is bought by mutilating the
  attention kernel, and the win over classical baselines is confounded with simply using direct
  multi-step instead of iterated prediction.
- **Autoformer (Wu et al., NeurIPS 2021).** Replaces dot-product attention with an Auto-Correlation
  block that aggregates over discovered period lags, and folds a moving-average **series decomposition**
  block (trend = moving average, seasonal = residual) into the encoder/decoder. The decomposition idea
  is genuinely load-bearing and gets reused below. Gap: the auto-correlation machinery is intricate and
  still attention-shaped over the time axis.
- **FEDformer (Zhou et al., ICML 2022).** Does attention in the frequency domain with a mixture of
  decomposition kernels, O(L) cost. Gap: yet another surgery on the attention kernel; the accuracy gains
  over Autoformer are incremental and the model is harder to reason about.
- **The shared symptom.** Every model in this lineage tokenizes **per time step** and runs attention
  over the time axis — a permutation-invariant operator over data whose entire content is order — and
  none of them improves when handed a *longer* look-back window. That, plus the confound that their
  baselines were iterated multi-step forecasters (error compounds over long horizons) while the
  Transformers were direct multi-step, is exactly what the linear baseline is built to expose.

## The fixed substrate

The Time-Series-Library training/evaluation loop is frozen and must not be touched: the data loader
(per-channel standardization fit on the train split, fixed train/val/test boundaries), the direct
multi-step objective (MSE over the whole horizon, no autoregression), Adam, early stopping on
validation loss, and the metric computation (MSE and MAE on all channels, on the inverse-standardized
predictions). Three datasets are run through the identical loop — **ETTh1** (7 channels), **Weather**
(21 channels), **ECL** (321 channels) — all at `seq_len=96`, `label_len=48`, `pred_len=96`,
`features=M` (multivariate in, multivariate out).

The loop hands the model a fixed `configs` object. Under the evaluated run command it always carries:
`task_name="long_term_forecast"`, `seq_len=96`, `pred_len=96`, `label_len=48`,
`enc_in=dec_in=c_out=` the channel count, `d_model=512`, `d_ff=512`, `e_layers=2`, `d_layers=1`,
`n_heads=8`, `dropout=0.1`, plus the Time-Series-Library argument defaults for everything a model might
need but the run command does not set — `embed="timeF"`, `freq="h"`, `activation="gelu"`, `factor=1`,
`moving_avg=25`, `use_norm=1`, `channel_independence=1`, `down_sampling_layers=0`,
`down_sampling_window=1`, `patch_len=16`. The single fixed `d_model=512` and `e_layers=2` across all
three datasets is the central constraint of this edit surface: each method's own training script tunes
these per dataset, but the model class here sees one config for all three, so any
architecture-specific capacity choice the method depends on must be set **inside the model**, not
expected from `configs`.

## The editable interface

Exactly one file is editable: `models/Custom.py`. It must define a `Model(nn.Module)` whose
`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)` returns `[batch, pred_len, c_out]`. The loop
calls `forward`; the convention is that `forward` delegates to a `forecast(...)` method and slices the
last `pred_len` steps. The model may import any layer from the Time-Series-Library `layers/` package
(it lives in the same tree): `series_decomp` and `moving_avg` from `layers.Autoformer_EncDec`,
`PatchEmbedding` / `DataEmbedding_inverted` / `DataEmbedding_wo_pos` / `PositionalEmbedding` from
`layers.Embed`, `FullAttention` / `AttentionLayer` from `layers.SelfAttention_Family`, `Encoder` /
`EncoderLayer` from `layers.Transformer_EncDec`, `Normalize` from `layers.StandardNorm`.

Every step of the ladder is one fill of this same contract. The starting point is the scaffold default
— a zero forecaster (the template ships a placeholder that returns zeros). Each method replaces the
body of the class with its architecture.

```python
# models/Custom.py — default scaffold fill (the placeholder to be replaced)
import torch
import torch.nn as nn


class Model(nn.Module):
    """Custom model for long-term time series forecasting."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # TODO: define the architecture here

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # placeholder: predict all zeros
        batch_size = x_enc.shape[0]
        return torch.zeros(batch_size, self.pred_len, self.c_out).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```

## Evaluation settings

The same `Custom` model is trained from scratch and evaluated on all three datasets under the identical
loop: `seq_len=96`, `pred_len=96`, `features=M`, Adam at `learning_rate=1e-4`, `batch_size=32`,
`train_epochs=10`, `patience=3`, seed 42. Two metrics per dataset, **MSE** and **MAE** on all channels,
lower is better. ETTh1 and Weather are reported; ECL is held out. A method is stronger when it lowers
MSE (tie-broken by MAE) averaged across the datasets.
