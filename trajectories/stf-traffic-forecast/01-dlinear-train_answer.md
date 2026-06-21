The scaffold default returns zeros, so the only thing I am being asked to design is the map from a twelve-step window of $N$ sensors to the next twelve steps for every sensor, and the first question is what the *floor* of that map should be. The lineage I inherit is a stack of ever-more-elaborate temporal Transformers — Informer, Autoformer, FEDformer, Pyraformer — each beating the last on long-horizon benchmarks, and one embarrassing result hanging over the whole trajectory: a single linear layer with a moving-average decomposition in front beat that entire family on accuracy and on cost. If attention over the time axis were really earning its keep, a plain affine map from past to future would have no business winning. Before I reach for any spatial graph or any attention, I want to plant a flag at the bottom that tests exactly this suspicion in *this* harness — to measure how much of the forecasting accuracy a high-capacity model would claim is actually available from an almost-trivial direct map. That flag is the weakest model that is still an honest forecaster, and everything above it has to justify its complexity against the number it posts.

I propose DLinear: a decomposition followed by two shared linear maps. The harness wants direct multi-step prediction — twelve steps out in one shot, no recursion — which already removes the error-accumulation disease that iterated baselines suffer over a long horizon. The absolute floor of a direct multi-step map is a single affine map along time, applied independently per node: each future step is a learned weighted sum of the twelve past steps of that same node, $\hat{x} = W x$ with $W$ a $12 \times 12$ matrix whose row $t$ says how to combine the twelve observed values into the $t$-th prediction. The reflex objection is "it is linear, it cannot model traffic," but consider what a single linear row expresses on a traffic series. If the series has a period — and traffic is almost *defined* by its daily and weekly rhythm — then a future value is, to first order, well predicted by the value one period back, and a row of $W$ can place its weight exactly on that lag. If the series has a slow drift — congestion building over the look-back — then a future value is an extrapolation of recent values, again a linear functional of the past. Periodicity and trend are precisely the two structures a linear temporal map captures naturally, and they are the *only* structures long-horizon forecasting can exploit at all, because a genuinely chaotic series is not predictable by anything. So a linear map should already reach most of what is forecastable, while a high-capacity nonlinear model risks fitting noise that does not cross the train/test boundary. There is also a structural nicety: every input step connects to every output step through one learned weight, so the signal path from any past observation to any future prediction has length one — no recurrence to forget through, no attention bottleneck.

The one refinement comes from running the bare linear map against the structure I know will bite it. Picture a series as a large slow congestion ramp with a smaller daily oscillation riding on top. A single $W$ now has to fit both at once, and the two want opposite weight patterns — the trend wants weights that extrapolate a smooth drift, the seasonality wants weights sharply concentrated at the periodic lags. Worse, the trend is large in magnitude and the oscillation is small, so under squared error the trend dominates the gradient: $W$ spends itself getting the loud ramp roughly right and under-fits the quiet oscillation that carries the fine structure. The remedy is the oldest move in time-series analysis, seasonal-trend decomposition. Estimate the trend as a moving average of the window, take the residual as the seasonal part, and give *each* its own linear map, then sum:
$$\text{trend} = \text{MovingAvg}(x), \qquad \text{seasonal} = x - \text{trend}, \qquad \hat{x} = W_s\,\text{seasonal} + W_t\,\text{trend}.$$
What makes this honest is that it adds *no representational capacity*: the moving average is itself a linear operator, and two linear maps plus a sum are still a single affine map end to end. I have added only *conditioning* — a fixed, parameter-free reparameterization that separates the loud trend from the quiet seasonality so gradient descent fits each well, the same solution set with far better-behaved learning. It is preconditioning, not depth, and it pays off precisely when there is a clear trend, the case the bare linear map handles worst.

A few details decide the boundary behavior and the generalization property. The moving average is an average pool over a sliding window of size $k$, stride 1, and to keep the trend the same length as the input I must pad. Zero-padding would drag the trend toward zero at the two ends — spurious dips exactly where I have least information — so I replicate-pad instead, extending the front and back with copies of the boundary values so the edge trend stays flat but faithful. With $\text{input\_len}=12$ the kernel cannot be the long $k=25$ that the original decomposition block uses on hourly data, so $k$ is configured at 25 but clamped by the short window in practice; the split still separates a smoother component from a residual, which is all the conditioning argument needs. Finally, I share the linear weights across nodes: every loop on the highway has the same daily and weekly rhythm, so one $W$ pair encodes that prior, slashes the parameter count from $N\cdot 12\cdot 12$ to $12\cdot 12$, and — most important for this task — makes the model size *independent of $N$*, the exact generalization property the research question asks for. That choice deliberately models *no* cross-node coupling at all. It is a real gap: traffic genuinely couples across sensors, congestion upstream showing up downstream minutes later. But that is the point of the floor — it measures how far pure per-node temporal extrapolation reaches, so every spatial mechanism above it has a clean number to beat, and I expect the gap to it to be largest on the dataset where cross-node propagation matters most. No timestamps are used and no overrides are needed; the model trains under the harness default $\text{lr}=2\times10^{-3}$, $\text{weight\_decay}=10^{-4}$.

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
