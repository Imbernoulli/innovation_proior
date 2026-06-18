**Problem.** Fill the empty `Model` with the most minimal *honest* forecaster — the control that
establishes how far the target's own history gets you *before* any exogenous fusion, so every later
rung has a real number to beat. Only the last (target) channel is scored.

**Key idea.** Direct multi-step (emit all 96 future steps in one shot, no iteration, no error
compounding) by a single affine map along the **time** axis, after a moving-average **series
decomposition** into trend + remainder so the slow drift and the periodic component get their own
linear maps. The two maps are **shared across channels**, so this rung is deliberately
**channel-independent**: the exogenous covariates never reach the target.

**Why it works (and why it's the right control).** A time series' content is its *order*; a linear map
reads the whole window at once and sees the shape directly, with none of attention's
permutation-invariance fighting the grain. Splitting trend from seasonal lets each map specialize.
What it provably *cannot* do is use the side channels — which is exactly the gap the ladder will press
on, isolated cleanly here.

**Scaffold edit / hyperparameters.** `series_decomp(moving_avg=25)`; two shared
`nn.Linear(seq_len, pred_len)` maps (seasonal, trend) initialized to $1/\text{seq\_len}$; sum the two
forecasts; no normalization, no embedding, no attention. Adam `lr=1e-4`, `train_epochs=10`,
`batch_size=32`, `patience=3` (fixed by the loop).

**What to watch.** ETTh1 (small, homogeneous, smooth target) should be respectable; Weather and ECL —
where the target depends on the other channels — should be visibly loose, since the discarded
exogenous signal is exactly what is missing. That pattern is the case for fusion, in the metrics.

```python
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """Channel-independent decomposition-linear forecaster (DLinear)."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # series decomposition (moving-average trend + remainder)
        moving_avg = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(moving_avg)
        self.channels = configs.enc_in

        # one shared linear map per part, applied along the time axis (channel-independent)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)              # [B, L, C], [B, L, C]
        seasonal_init = seasonal_init.permute(0, 2, 1)               # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)        # [B, C, pred_len]
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)                                    # [B, pred_len, C]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
