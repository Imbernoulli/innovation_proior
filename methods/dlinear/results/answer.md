# DLinear, distilled

DLinear (Decomposition-Linear) is a direct multi-step forecaster for long-term time
series: split the look-back window into a moving-average trend and a seasonal residual,
run each component through its own one-layer temporal linear map from `L` to `T`, and
sum the two predicted components. It has no attention, no recurrence, and no nonlinear
learned block; the moving average is a fixed linear smoothing operator.

## Problem it solves

Long-term multivariate forecasting: map a length-`L` look-back of `C` channels to a
length-`T` horizon for all channels, with `T > 1`, trained directly in one shot rather
than autoregressively rolling a one-step model forward.

## Key idea

The baseline isolates two confounds in Transformer-based long-term forecasting:

- Self-attention scores content pairs and needs positional/temporal embeddings to carry order,
  while raw numerical time series carry most of their information in ordered temporal change.
- Many reported wins compared direct multi-step Transformer forecasters against iterated
  multi-step non-Transformer baselines, so the gain could come from DMS rather than attention.

The minimal DMS model is a temporal affine map. For one variate `i`,
`X_hat_i = W X_i`, where `W in R^{T x L}` maps the look-back directly to the horizon.
This can represent the two structures that make long horizons forecastable: trend
extrapolation and periodic lookup.

DLinear keeps that temporal-linear model but reparameterizes it with a fixed additive
decomposition:

```
trend    = MovingAvg_25(x)
seasonal = x - trend
output   = Linear_Seasonal(seasonal^T)^T + Linear_Trend(trend^T)^T
```

If `A` is the moving-average matrix, the shared-channel model computes
`W_s(I - A)x + W_t A x + b_s + b_t`. This is still affine in the raw input, and it does
not enlarge the temporal affine function class: setting `W_s = W_t` recovers any single
linear temporal map. The decomposition changes conditioning and inductive bias, letting
trend and seasonal residuals fit through separate parameter matrices.

## Design choices

- **DMS not IMS:** predicts the whole horizon in one forward pass, avoiding recursive
  error accumulation and matching the forecasting strategy used by the Transformer baselines.
- **Kernel 25, endpoint repeated:** the official code uses a fixed odd moving-average
  kernel of 25. Repeating 12 endpoint values on each side makes AvgPool1d preserve length:
  `L + 24 - 25 + 1 = L`.
- **Seasonal residual sign:** `seasonal = x - moving_mean`, not the reverse.
- **Channel sharing:** the paper formulation applies one pair of temporal linears to every
  channel and does not model spatial correlation. The official code also supports
  `configs.individual`; when true, it uses one seasonal and one trend linear per channel.
- **Parameters:** PyTorch has biases, so the shared DLinear path has `2 * (T * L + T)`
  trainable scalar parameters. The paper reports `2 * T * L`, ignoring biases; the
  individual path multiplies the linear-layer count by `C`.
- **Visualization initialization:** the official code leaves `1 / seq_len` weight
  initialization commented out for weight-heatmap visualization. It is not active by default.

## Final algorithm

```python
input  x : [B, L, C]
moving_mean = MovingAvg_25(x)             # [B, L, C]
seasonal    = x - moving_mean             # [B, L, C]
s_out       = Linear_Seasonal(seasonal.permute(0, 2, 1))  # [B, C, T]
t_out       = Linear_Trend(moving_mean.permute(0, 2, 1))  # [B, C, T]
output      = (s_out + t_out).permute(0, 2, 1)            # [B, T, C]
loss        = MSE(output, y_true)
```

## Working code

Faithful to the official `cure-lab/LTSF-Linear` `models/DLinear.py` implementation:

```python
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):                       # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)               # [B, L, C]


class series_decomp(nn.Module):
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean                 # seasonal, trend


class Model(nn.Module):
    """Decomposition-Linear."""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):                        # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)               # [B, T, C]
```

NLinear is the sibling variant for distribution shift: subtract the last observed value,
apply the temporal linear map, and add that value back. The official implementation also
supports the `individual` channel-wise option:

```python
class NLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):                        # x: [B, L, C]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)],
                                 dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + seq_last                      # [B, T, C]
```
