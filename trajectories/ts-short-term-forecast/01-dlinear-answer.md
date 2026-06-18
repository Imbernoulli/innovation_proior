**Problem.** Univariate M4 short-term forecasting under a fixed protocol: map a short look-back
(`seq_len = 2·pred_len`) of one channel to a `pred_len` horizon, trained on SMAPE. The scaffold's
default returns zeros, so the floor is simply *the simplest model that can fit M4 at all* — one whose
failure I can read cleanly.

**Key idea.** On M4 the forecastable structure is a slow trend plus a repeating seasonal shape, both
linear in the look-back. So use a single direct-multi-step linear map `R^{seq_len} → R^{pred_len}`
(path length one, no recurrence to compound error, no attention over meaningless single steps). Split
the window first with a parameter-free moving-average decomposition (`trend = MovingAvg(x)`,
`seasonal = x − trend`) and give each part its own channel-shared linear map, summed — this adds no
capacity (the whole model is affine) but preconditions the fit so the loud trend cannot starve the
quiet seasonal component.

**Why.** Decomposition before fitting stops the large-magnitude trend from dominating the squared-error
fit and under-fitting seasonality; it should help most on the trending regimes (Yearly, trending
Monthly). The model ignores the harness's capacity knobs (`d_model`, `e_layers`, `n_heads`) entirely —
the linear maps are sized only by `seq_len`/`pred_len` — which is the cleanest statement of the
hypothesis that M4 does not need capacity. No instance normalization at the floor on purpose, so any
later gain from normalization is attributable, not confounded.

**Hyperparameters.** `moving_avg = 25` (replicate-padded, from `configs.moving_avg`); channel-shared
`nn.Linear(seq_len, pred_len)` for seasonal and trend; trainable footprint `2·pred_len·seq_len`
(e.g. Monthly `2·18·36`). Trained under the fixed Custom protocol: Adam `lr=1e-3`, batch 16, 10
epochs, patience 3, SMAPE loss.

**What to watch.** SMAPE should come out worst on Yearly (shortest window, trend-dominated, six-step
horizon) and most competitive on Monthly (window spans three 12-step cycles). The affine, un-normalized
form cannot decouple shape from per-series level, nor model any nonlinear interaction — exactly the two
things a normalized, higher-capacity rung above should reclaim.

```python
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Length-preserving moving average: highlights the trend of a series."""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)          # replicate-pad both ends
        x = self.avg(x.permute(0, 2, 1))               # pool along time
        x = x.permute(0, 2, 1)
        return x                                        # [B, L, C]


class series_decomp(nn.Module):
    """trend = moving average, seasonal = residual."""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):  # x: [B, L, C]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean                         # seasonal, trend


class Model(nn.Module):
    """DLinear: decomposition + two channel-shared linear maps, direct multi-step."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)
        # channel-shared linear maps along the time axis
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):  # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)          # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)                # [B, C, L]
        seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, C, T]
        trend_output = self.Linear_Trend(trend_init)           # [B, C, T]
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)                               # [B, T, C]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]               # [B, T, C]
        return None
```
