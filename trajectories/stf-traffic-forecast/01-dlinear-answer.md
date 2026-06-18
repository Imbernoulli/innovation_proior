**Problem.** Spatial-temporal traffic forecasting in the fixed BasicTS harness: map a 12-step window of
`N` sensors to the next 12 steps for every sensor, in one shot, trained by masked direct-multi-step
regression. The floor must be the weakest *honest* forecaster — the simplest direct map — so every
spatial or nonlinear mechanism above it has a clean number to justify itself against.

**Key idea.** Strip the model to almost nothing and keep only the direct-multi-step strategy. The
minimal direct map is one linear layer along time per node — `x̂ = W x`, every past step wired to every
future step with a learned weight (signal path length one) — which already captures the two structures
that make traffic forecastable: trend (extrapolating a drift is linear) and periodicity (reading the
value one period back is linear). One refinement: a single `W` fitting a loud trend plus a quiet
seasonality is dominated by the trend under squared error, so split the window into
`trend = MovingAvg(x)` and `seasonal = x − trend` (replicate-padded so the trend stays faithful at the
edges) and give each its own `Linear(12, 12)`, then sum. This adds **no representational capacity** —
the whole thing is still one affine map — only conditioning, so each linear specializes. Weights are
**shared across nodes**, which both regularizes and makes the model size independent of `N` (the
generalization property the task asks for), at the cost of modeling **no cross-node coupling** — the
deliberate gap this floor leaves for the spatial rungs above.

**Why it is the floor.** No spatial mechanism, no nonlinearity beyond the parameter-free split, no use
of the timestamps. It measures how far pure per-node temporal extrapolation reaches, so the gap to it is
the value of spatial modeling — expected largest on the dataset where cross-node propagation matters most.

**Hyperparameters.** `moving_avg = 25` (clamped by the 12-step window in practice); node-shared
`Linear(12, 12)` for each of seasonal and trend; trained under the harness default `lr = 2e-3`,
`weight_decay = 1e-4` (no `CONFIG_OVERRIDES`).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    moving_avg: int = field(default=25)


class Custom(nn.Module):
    """DLinear: Decomposition-Linear baseline.

    Decomposes input into trend (moving average) and seasonal (residual),
    then projects each component independently to the prediction horizon.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        k = config.moving_avg
        self.pad_left = (k - 1) // 2
        self.pad_right = k // 2
        self.avg_pool = nn.AvgPool1d(k, stride=1)
        self.linear_seasonal = nn.Linear(config.input_len, config.output_len)
        self.linear_trend = nn.Linear(config.input_len, config.output_len)

    def _decompose(self, x):
        # x: [B, T, N] -> trend via moving average, seasonal = x - trend
        padded = F.pad(x.transpose(1, 2), (self.pad_left, self.pad_right), mode='replicate')
        trend = self.avg_pool(padded).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N]
        seasonal, trend = self._decompose(inputs)
        # Per-feature linear: [B, N, T] -> [B, N, T']
        seasonal_out = self.linear_seasonal(seasonal.transpose(1, 2))
        trend_out = self.linear_trend(trend.transpose(1, 2))
        prediction = (seasonal_out + trend_out).transpose(1, 2)  # [B, T', N]
        return prediction
```
