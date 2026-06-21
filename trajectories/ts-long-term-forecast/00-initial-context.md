## Research question

Multivariate long-term forecasting: given a fixed look-back window of every channel of a multivariate
series, predict the next horizon for every channel. The design target is the **forecasting
architecture** — the map from a `[batch, seq_len, enc_in]` history to a `[batch, pred_len, c_out]`
horizon — at fixed `seq_len=96 / pred_len=96`. The data pipeline, standardization, optimizer, and
metric computation are fixed by the Time-Series-Library protocol; only the model class changes.

## Prior art / Background / Baselines

- **Informer.** A Transformer with ProbSparse attention, a distilling encoder, and a generative decoder
  that emits the whole horizon in one forward pass. Gap: the O(L²) cost is reduced by dropping most
  query–key pairs, and the reported gains are confounded with the shift from iterated to direct
  multi-step prediction.
- **Autoformer.** Replaces dot-product attention with an Auto-Correlation block that aggregates over
  discovered period lags, and folds a moving-average series decomposition block into the encoder/decoder.
  Gap: the auto-correlation machinery remains intricate and attention-shaped over the time axis.
- **FEDformer.** Does attention in the frequency domain with a mixture of decomposition kernels, O(L)
  cost. Gap: the gains over Autoformer are incremental and come with another specialized attention
  kernel that is harder to reason about.

## Fixed substrate / Code framework

The Time-Series-Library training/evaluation loop is frozen: the data loader (per-channel standardization
fit on the train split, fixed train/val/test boundaries), the direct multi-step objective (MSE over the
whole horizon, no autoregression), Adam, early stopping on validation loss, and MSE/MAE metrics on
inverse-standardized predictions across all channels. Three datasets are run through the identical loop
— **ETTh1** (7 channels), **Weather** (21 channels), **ECL** (321 channels) — at `seq_len=96`,
`label_len=48`, `pred_len=96`, `features=M`.

The loop hands the model a fixed `configs` object. Under the evaluated run command it carries:
`task_name="long_term_forecast"`, `seq_len=96`, `pred_len=96`, `label_len=48`,
`enc_in=dec_in=c_out=` the channel count, `d_model=512`, `d_ff=512`, `e_layers=2`, `d_layers=1`,
`n_heads=8`, `dropout=0.1`, plus the Time-Series-Library argument defaults for everything a model might
need but the run command does not set — `embed="timeF"`, `freq="h"`, `activation="gelu"`, `factor=1`,
`moving_avg=25`, `use_norm=1`, `channel_independence=1`, `down_sampling_layers=0`,
`down_sampling_window=1`, `patch_len=16`. The single fixed `d_model=512` and `e_layers=2` across all
three datasets is the central constraint: any architecture-specific capacity choice must be set
**inside the model**, not expected from `configs`.

## Editable interface

Exactly one file is editable: `models/Custom.py`. It must define a `Model(nn.Module)` whose
`forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)` returns `[batch, pred_len, c_out]`. The loop
calls `forward`; conventionally `forward` delegates to a `forecast(...)` method and returns the last
`pred_len` steps. The model may import any layer from the Time-Series-Library `layers/` package in the
same tree: `series_decomp` and `moving_avg` from `layers.Autoformer_EncDec`, `PatchEmbedding` /
`DataEmbedding_inverted` / `DataEmbedding_wo_pos` / `PositionalEmbedding` from `layers.Embed`,
`FullAttention` / `AttentionLayer` from `layers.SelfAttention_Family`, `Encoder` / `EncoderLayer` from
`layers.Transformer_EncDec`, `Normalize` from `layers.StandardNorm`.

The starting point is the scaffold default — a zero forecaster. Each fill replaces the body of the
class with its architecture.

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
`train_epochs=10`, `patience=3`, seed 42. Metrics are MSE and MAE on all channels, lower is better.
ETTh1 and Weather are reported; ECL is held out. A method is stronger when it lowers MSE (tie-broken
by MAE) averaged across the datasets.
