# Context: multiscale deep time-series forecasting before the method

## Research question

Given a past window `x` of length `P` with `C` observed variates, predict a future window of length
`F` with the same variates. The hard part is not the final regression interface. The hard part is
that a real window is a superposition of several temporal behaviors: short oscillations, longer
periodic structure, a slow level or trend, local shocks, and distribution drift between train and
test windows.

A single sampling rate exposes only one view of that mixture. A five-minute traffic series can show
commute spikes; a daily aggregation suppresses those spikes and exposes weekday or holiday
structure; a still coarser view mostly exposes slow macro movement. The same physical process can
therefore look microscopic at a fine scale and macroscopic at a coarse scale. The open question is
how to use cheap neural components to forecast from this kind of signal.

The desired component should fit a standard forecasting harness: it receives `[B, P, C]` past
values, optional time features for the past and decoder side, and returns `[B, F, C]`. It should be
efficient enough for long horizons and many channels, so attention or recurrence is not a necessary
default. The architecture inside that slot is unsettled.

## Background

Deep time-series forecasters before this point organize around a few reusable ideas.

The first is direct multi-step prediction. Instead of rolling a one-step predictor forward and
feeding predictions back into itself, the model maps the whole look-back window to the whole horizon
in one forward pass. This avoids recursive error accumulation and matches the common long-term
forecasting protocol.

The second is seasonal-trend decomposition. Classical time-series analysis separates a slow
trend-like component from faster seasonal or residual variation because the two are easier to model
separately than as one mixed signal. In deep forecasting, a common differentiable version is a
length-preserving moving average: pad the ends by repeating boundary values, average-pool with
stride 1 to get a smooth trend, and define the seasonal residual as `x - trend`.

The third is multiscale representation. Pooling or aggregating along time is a low-pass operation:
it suppresses high-frequency detail and keeps slower structure. Models such as pyramidal attention
and splitting trees already show that a network can process several temporal resolutions internally.

The fourth is cheap mixing. MLP-style blocks can mix along a chosen axis using Linear layers and a
nonlinearity. A temporal Linear can map one sequence length to another; a channel FeedForward can
mix hidden features at a fixed time step. These primitives are much cheaper than attention and are
already available in the forecasting codebase.

The fifth is reversible instance normalization. Many forecasting benchmarks have train-test
level and variance shift. A per-window normalization can store the current window's mean and
standard deviation, train the model on normalized values, and then invert the transformation at the
output.

## Baselines

**Autoformer.** Uses moving-average series decomposition inside a Transformer-style forecasting
model. Its decomposition block provides a reusable primitive: `trend = MovingAvg(x)` and
`season = x - trend`, with length preserved by endpoint padding.

**DLinear.** Decomposes a look-back window into seasonal and trend components, maps each component
directly from past length to future length with one temporal Linear, and sums the two forecasts. It
shows that simple temporal linear maps can be strong and efficient.

**TimesNet and other multiperiodicity models.** Detect dominant periods, often with an FFT, and
process period-specific structures. They disentangle variation along a period axis.

**Pyraformer and SCINet.** Build multiscale internal representations using a pyramidal structure or
a splitting tree.

**PatchTST and related channel-independent forecasters.** Treat variates independently with shared
weights, which is useful when `C` ranges from a few channels to hundreds and cross-channel
correlations are weak or unstable.

## Evaluation settings

The surrounding protocol is a standard long- and short-term forecasting benchmark setup. The model
is trained with Adam, default beta values `(0.9, 0.999)`, and an MSE loss for long-term forecasting.
Long-term experiments use multivariate-in, multivariate-out windows on datasets such as ETT,
Weather, Solar-Energy, Electricity, and Traffic, typically sweeping horizons such as 96, 192, 336,
and 720. Short-term experiments include PeMS traffic-network datasets and the univariate M4
subsets.

The metrics are task dependent. Long-term forecasting commonly reports MSE and MAE over all horizon
steps and variates. PeMS short-term forecasting commonly reports MAE, MAPE, and RMSE. M4 uses
SMAPE, MASE, and OWA. These are yardsticks only: no outcome or ablation result is assumed in the
setup.

The data loader provides `x_enc`, `x_mark_enc`, `x_dec`, and `x_mark_dec`. The architecture may use
calendar/time features when present, but it must not inspect the true future values. Normalization
statistics must be computed from observed windows and then reused only to invert the prediction.

## Code framework

The available toolbox already contains the generic pieces below.

```python
import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    """Length-preserving moving-average decomposition."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):                         # x: [B, T, C]
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        season = x - trend
        return season, trend


class Normalize(nn.Module):
    """Reversible per-instance normalization over non-channel dimensions."""
    def __init__(self, num_features, eps=1e-5, affine=True, non_norm=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):                   # x: [B, T, C]
        if mode == "norm":
            dims = tuple(range(1, x.ndim - 1))
            self.mean = x.mean(dim=dims, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=dims, keepdim=True, unbiased=False) + self.eps).detach()
            if self.non_norm:
                return x
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.non_norm:
                return x
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise NotImplementedError


class TemporalMLP(nn.Module):
    """Generic length-changing temporal mixer."""
    def __init__(self, in_len, out_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.GELU(),
            nn.Linear(out_len, out_len),
        )

    def forward(self, x):                         # x: [B, D, in_len]
        return self.net(x)


class Model(nn.Module):
    """Forecasting slot to fill."""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in]
        # return: [B, pred_len, c_out]
        raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
```
