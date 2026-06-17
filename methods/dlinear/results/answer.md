# DLinear, distilled

DLinear (Decomposition-Linear) is an embarrassingly simple direct-multi-step forecaster: split
the look-back window into a moving-average trend and a seasonal residual, run each through its
own one-layer linear map along time (L → T), and sum the two. No attention, no recurrence, no
nonlinearity beyond the parameter-free decomposition; weights are shared across channels by
default and no cross-variate correlation is modeled. It was introduced as a baseline to test
whether the high-capacity Transformer forecasters dominating long-term TSF were actually using
temporal order or were winning for confounded reasons.

## Problem it solves

Long-term multivariate forecasting: map a length-`L` look-back of `C` channels to a length-`T`
horizon for all channels, with `T >> 1`, trained directly (one shot, no autoregression).

## Key idea

Two observations motivate stripping the model to almost nothing:

- **Self-attention is permutation-invariant**, so it represents order only through positional
  encodings. Numeric time series carry their signal almost entirely in order (a single value has
  no standalone semantics), so an order-agnostic core is working against the grain — and the
  deployed Transformers do not improve with longer look-back windows, the fingerprint of a model
  not exploiting long-range temporal structure.
- The reported Transformer wins were measured against **iterated multi-step** baselines that
  compound error over long horizons, while the Transformers used **direct multi-step** (DMS)
  prediction. So the gains conflated "attention" with "DMS."

DLinear keeps only the DMS strategy and discards everything else. The minimal DMS forecaster is
one linear layer along time, `X̂ = W X` with `W ∈ R^{T×L}` — every past step connects to every
future step with a learned weight (signal path length one), which captures the two structures
that make long-horizon forecasting feasible at all: trend (extrapolating a slow drift is linear)
and periodicity (reading off the value one period back is linear).

Two refinements:

- **Decomposition.** A single `W` fitting a strong trend *plus* seasonality is dominated by the
  large-magnitude trend in the squared-error fit and under-fits the seasonality. Splitting the
  window into `trend = MovingAvg(x)` and `seasonal = x − trend`, then giving each its own linear
  layer and summing, lets each specialize. This adds **no representational capacity** (moving
  average + two linear maps + sum is still affine end-to-end); it is preconditioning that
  separates the loud component from the quiet one. This is what distinguishes DLinear from the
  vanilla Linear, and it helps precisely when there is a clear trend.
- **Normalization (the NLinear sibling, for distribution shift).** Subtract the last value of the
  look-back, apply the linear map, add it back — re-centers each window on a common baseline so
  the level drift between train and test eras is neutralized.

## Design choices and why

- **Linear, one layer:** the cleanest falsification of "attention is necessary" is the minimal
  DMS model; trend and periodicity are linear-capturable, and a chaotic series is not
  long-horizon-predictable by anything.
- **DMS not IMS:** removes long-horizon error accumulation and matches the strategy the
  Transformers use, removing the confound.
- **Moving-average kernel = 25, replicate-padded:** the same smoothing scale as the established
  decomposition block (clean comparison); replicate-padding keeps the trend faithful at the
  window edges instead of pulling it toward zero. Odd kernel with `(k−1)//2` pad on each side
  preserves length.
- **Channel-shared weights, no cross-variate term:** channels of one dataset share temporal
  dynamics; sharing cuts parameters from `C·T·L` to `2·T·L` and avoids overfitting spurious
  cross-channel coupling. An `individual` variant (one linear pair per channel) exists for
  heterogeneous channels but is not the default.
- **Interpretability:** initializing the linear weights to the uniform average `1/L` and training
  yields a smooth weight heatmap that shows bright stripes at the periodic lags (e.g. 24 and 168
  on traffic data) — direct evidence the linear map represents periodicity. The `1/L` init is for
  visualization only; the forecaster itself uses default initialization.

## Final algorithm

```
input  x : [B, L, C]
trend    = MovingAvg_k(x)               # length-preserving, replicate-padded, k = 25
seasonal = x - trend
s_out    = Linear_Seasonal(seasonalᵀ)   # nn.Linear(L, T) along time  -> [B, C, T]
t_out    = Linear_Trend(trendᵀ)         # nn.Linear(L, T) along time  -> [B, C, T]
output   = (s_out + t_out)ᵀ             # [B, T, C]
loss     = MSE(output, y_true)          # direct multi-step over the whole horizon
```

Parameters: `2·T·L` (shared) or `2·C·T·L` (individual). Trained with Adam (lr `1e-4`), batch 32,
≤10 epochs with early stopping.

## Working code

Faithful to the canonical Time-Series-Library implementation: the `series_decomp` block from the
decomposition line, two channel-shared linear maps, summed, with the long-term-forecast dispatch.

```python
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Length-preserving moving average: highlights the trend of a series."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):                              # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)         # replicate-pad both ends
        x = self.avg(x.permute(0, 2, 1))              # pool along time
        return x.permute(0, 2, 1)                      # [B, L, C]


class series_decomp(nn.Module):
    """Series decomposition: trend = moving average, seasonal = residual."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):                              # x: [B, L, C]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean                        # seasonal, trend


class Model(nn.Module):
    """DLinear: decomposition + two linear maps, direct multi-step."""
    def __init__(self, configs, individual=False):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name in ('classification', 'anomaly_detection', 'imputation'):
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.decompsition = series_decomp(configs.moving_avg)   # kernel 25
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):                              # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)   # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)         # [B, C, L]
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)   # [B, C, T]
            trend_output = self.Linear_Trend(trend_init)            # [B, C, T]
        x = seasonal_output + trend_output                          # recombine
        return x.permute(0, 2, 1)                                    # [B, T, C]

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)                      # pred_len == seq_len: reconstruct

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]      # [B, T, C]
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)        # [B, L, C]
        return None
```

NLinear sibling — replace the decomposition with subtract-last / add-back normalization to
handle distribution shift:

```python
class NLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):                              # x: [B, L, C]
        seq_last = x[:, -1:, :].detach()              # current level
        x = x - seq_last                              # de-level the window
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last                              # re-anchor to current level
        return x                                       # [B, T, C]
```
