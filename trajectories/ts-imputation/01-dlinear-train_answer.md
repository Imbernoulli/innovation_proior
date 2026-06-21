The floor I start from is the identity fill, and it is worth staring at because it tells me exactly what the metric rewards. The loop punches a quarter of the $[B, 96, C]$ window to zero, hands me the holed window plus the binary mask, and scores me only at the holes. Returning the window untouched predicts zero at every hole; after standardisation zero is the channel mean, so the identity is just "guess the mean wherever a value is missing" — it ignores the timestep, the neighbours, and the other channels. So the first real rung should not chase the most powerful model; it should be the simplest thing that actually uses the rewarded context, a floor I understand completely before reaching for anything intricate. At a masked entry there are two reservoirs of information: the temporal one — the other 95 timesteps of the same channel, most of them observed — and the cross-channel one — the other channels at the same instant. With only 25% deleted uniformly at random, every gap is interior, surrounded on both sides by observed neighbours of its own channel, so the cheapest, highest-leverage cue is temporal interpolation.

I propose DLinear: a parameter-free seasonal-trend decomposition feeding two linear maps along the time axis, summed. The core hypothesis is that the recoverable structure at an interior gap — periodicity and trend — is a *linear* functional of the rest of the window, so the minimal model that uses the rewarded context is a single linear map along time, applied per channel, reading all 96 timesteps and emitting a reconstructed 96. I lean on this over a Transformer for a concrete reason: attention's core operation is permutation-invariant, which suits language where a word carries standalone meaning, but a single numeric timestep's entire content *is* its position in the order, so an architecture that throws away order and re-injects it as an additive code works against the grain. A linear map along time does the opposite — it reads the whole window through learned weights, sees the shape directly, and the maximum signal path from any observed timestep to any reconstructed one is length one, with no recurrence to forget through and no attention bottleneck. For imputation the argument is even stronger than for forecasting: the masked positions are interior with observed values on both sides, so I am interpolating, and interpolation is exactly what a linear functional of the surrounding points does well.

The one trick is what makes a bare map insufficient. A single $96 \times 96$ matrix $W$ has to fit a window that is typically a large slow trend with a small daily oscillation riding on it, and these want opposite weight patterns — the trend wants smooth weights that extrapolate a drift, the seasonality wants weights concentrated at the periodic lags. Worse, the trend is large in magnitude and the seasonal part small, so in a squared-error fit the trend dominates the gradient: $W$ spends itself getting the big ramp roughly right and under-fits the small oscillation that carries the fine structure I actually need at the holes. One matrix is being asked to be two filters, and the loss prioritises the loud one. The fix is the oldest move in time-series analysis, seasonal-trend decomposition: split the window additively, because each piece on its own is more regular than the sum. I use the library's moving-average block `series_decomp` as a fixed, parameter-free front end — estimate $\text{trend} = \text{MovingAvg}(x)$ by a length-preserving moving average, take $\text{seasonal} = x - \text{trend}$ as the residual — then give each stream its own time-axis linear map, $W_{\text{trend}}$ and $W_{\text{seasonal}}$, both $96 \to 96$, and sum the two predicted streams.

I want to be honest that this adds no capacity. Two linear maps plus a sum is affine, the moving average is linear, so decompose-then-two-linears-then-sum is end to end still a single affine map; the function class is unchanged. What I have added is *conditioning* — a fixed reparameterisation that separates the loud trend from the quiet seasonality so gradient descent on the masked MSE fits each well instead of letting the trend's magnitude swamp the oscillation. It is preconditioning, not depth, and it pays off precisely on the trended hourly series where a bare map would under-fit. The moving average's boundary behaviour is pinned down because the edges are where I know least: `series_decomp` uses an odd kernel (25, the library default and a sensible sub-daily smoothing scale on hourly data) with stride 1, replicate-padding the front with copies of the first value and the back with copies of the last before average pooling, so the trend stays flat-but-faithful at the edges rather than being dragged toward zero by zero-padding; with $(25-1)/2 = 12$ on each side the length is restored exactly to 96.

The task-specific part I must derive rather than copy is the absence of a normalisation wrapper. The forecasting version of this method subtracts the last observed value, predicts the de-levelled continuation, and adds it back. That is wrong here for two reasons. First, this is imputation: $\text{pred\_len} = \text{seq\_len}$, the masked positions are interior, and there is no single "current level" to subtract — the pipeline's standardisation already handles the window statistics and the holes are already zero. Second, I must never let the imputation depend on the punched-out values as if real; the decomposition's moving average does average those zeros in along with real neighbours, biasing the trend slightly toward zero near a gap, but with only 25% missing and a 25-wide kernel each average is still mostly observed values, so the bias is small and the seasonal residual absorbs the rest. So I deliberately add no de-levelling step — the standardised, mean-zero input is already on a common baseline and the affine maps can represent any residual offset — and the imputation path is just decompose, two linear maps along time, sum.

This model is channel-blind by design: the linear maps are shared across channels (`individual=False`) and act only along time, so channel $c$ is reconstructed from $c$'s own observed timesteps and nothing from the correlated channels at the same instant. On ECL, with 321 strongly co-moving electricity clients, that leaves the single most informative cross-channel cue on the table — and naming that blind spot is exactly what sets up the next rung. The bet is that within-channel temporal interpolation already crushes the mean-guessing identity everywhere, with the largest residual error landing on ECL, where ignoring cross-channel structure costs the most.

```python
# models/Custom.py — step 1: DLinear (decomposition + linear, channel-shared)
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """DLinear for imputation: moving-average decomposition + two linear maps along time."""

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len            # imputation: pred_len = seq_len
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[-1].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[-1].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)      # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output                   # [B, C, T]
        return x.permute(0, 2, 1)                             # [B, T, C]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return None
```
