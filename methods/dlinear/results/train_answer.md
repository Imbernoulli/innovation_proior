The problem is long-term multivariate time-series forecasting: given a look-back window of length L over C channels, predict the next T steps for every channel, with T much larger than one. The established literature has converged on ever more elaborate Transformer variants, each claiming a new state of the art on standard benchmarks. But two issues make it hard to credit those gains to the architecture itself. First, self-attention is permutation-equivariant before positional encodings are added; it scores pairs of tokens by content and only recovers order through an additive positional side channel. That is a good fit for language or vision, where individual tokens carry rich standalone meaning, but a poor fit for raw numerical time series, where a single value like 0.42 carries almost no semantics and the information lives almost entirely in order and spacing. Second, the reported wins often compare direct multi-step Transformer forecasters against iterated multi-step statistical or RNN baselines. Iterated prediction feeds its own previous prediction back as input, so errors compound over a long horizon, while direct multi-step prediction emits the whole horizon in one forward pass and avoids that recursion. The comparison therefore confounds "Transformer" with "direct multi-step forecasting," and it is unclear how much of the accuracy comes from attention and how much simply comes from choosing the right forecasting strategy.

What is needed is a deliberately minimal direct multi-step baseline that removes the confound and measures how much of the reported gap actually requires a high-capacity sequence model. The simplest possible such forecaster is a single affine map along the time axis: for each channel, predict the horizon as a weighted sum of the look-back, X_hat_i = W X_i, where W is a T by L matrix. This is not as limited as it first appears. The two structures that make long horizons forecastable at all are trend and periodicity, and both are naturally expressed as linear functionals of the past: a trend extrapolates a slow drift, and a periodic signal reads off same-phase values from the look-back. If long-horizon forecasting is only viable for series with clear trend and periodicity, a linear temporal map should capture most of what is forecastable, while a more flexible nonlinear model risks fitting noise that does not generalize across the train-test boundary. There is also a structural advantage: every past observation connects directly to every future prediction with a single weight, so there is no recurrence to forget through and no attention bottleneck.

The method I propose is DLinear, short for Decomposition-Linear. It is a direct multi-step forecaster with no attention, no recurrence, and no learned nonlinearity beyond the decomposition. The core idea is to split the look-back window into a slow trend component and a seasonal residual using a fixed, parameter-free moving average, run each component through its own one-layer temporal linear map from L to T, and sum the two predicted streams. Concretely, trend is obtained by a length-preserving moving average with replicate endpoint padding, seasonal is x minus trend, and the output is Linear_Seasonal(seasonal) plus Linear_Trend(trend), with each linear acting along the time axis. The model is still affine in the raw input: if A is the moving-average operator, the shared-channel version computes W_s (I - A) x + W_t A x plus biases. Setting W_s equal to W_t recovers any single linear temporal map, so the decomposition does not enlarge the function class; it reparameterizes it so that gradient descent can fit the quiet seasonal component without its gradient being swamped by the louder trend component.

DLinear makes a few deliberate design choices. It predicts the whole horizon in one forward pass, matching the direct multi-step strategy of the Transformer baselines and avoiding the recursive error accumulation that handicaps iterated methods. Weights are shared across channels by default, encoding the prior that channels within a dataset usually share temporal dynamics and avoiding spurious cross-channel structure. The moving average uses an odd kernel of size 25 with replicate-padded endpoints so the trend stays faithful at the window edges rather than being pulled toward zero. The official code also supports an individual-channel variant, but the shared version is the default and the one that aligns with the paper's main results. For series whose level drifts between training and test periods, the sibling NLinear variant subtracts the last observed value of the look-back, applies a single temporal linear map, and adds that value back, re-anchoring the prediction to the current level. DLinear itself lets the trend linear absorb the level through the moving-average split.

The reason DLinear is the right move here is that it isolates the two confounds that muddy the Transformer comparisons. By stripping away attention entirely and keeping only direct multi-step regression, it tests whether the elaborate sequence modeling is doing the work credited to it. And by using an interpretable linear temporal map, it makes the source of accuracy visible: on periodic data, the learned weight matrix should show high weights at the relevant lags, directly confirming that the model is reading off same-phase past values. If this embarrassingly simple model is competitive with or better than the Transformer forecasters, then much of their reported gain was coming from direct multi-step forecasting and from trend-seasonal decomposition rather than from attention.

```python
import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Length-preserving moving average along the time axis."""
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):                       # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))       # pool along time
        return x.permute(0, 2, 1)              # [B, L, C]


class series_decomp(nn.Module):
    """Split a window into seasonal residual and trend."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):                       # x: [B, L, C]
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class Model(nn.Module):
    """DLinear (Decomposition-Linear) direct multi-step forecaster."""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = getattr(configs, 'individual', False)

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
        seasonal_init = seasonal_init.permute(0, 2, 1)   # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)         # [B, C, L]

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype, device=trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)   # [B, C, T]
            trend_output = self.Linear_Trend(trend_init)            # [B, C, T]

        x = seasonal_output + trend_output                          # recombine
        return x.permute(0, 2, 1)                                    # [B, T, C]
```
