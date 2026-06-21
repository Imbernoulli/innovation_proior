I am opening the ladder, so there is no prior measurement to beat — only the scaffold's zero forecaster and the lineage it sits in. The question is not "what is the best forecaster" but "where do I honestly start," and I want the first rung to be a control whose every part I understand, so that whatever the heavier rungs add can be measured against it. My doubt is about the whole attention-based forecasting line the context lays out — Informer, Autoformer, FEDformer — each a clever surgery on the attention kernel, each reporting a creeping accuracy gain over the last. Attention's core operation, $\mathrm{softmax}(QK^\top/\sqrt{d})\,V$, is permutation-invariant: shuffle the input tokens and you get the same multiset of pairwise scores. For words and image patches that is fine, because each token carries standalone semantic content and order can be re-injected as a positional side channel. But a single number in a load trace — $0.42$ — means nothing by itself; the entire signal lives in the *order*, that this value sits on the rising shoulder of the evening peak, the shape that recurred 24 and 168 hours ago. So the lineage applies an order-throwing operator to data whose entire content is order, and two symptoms confirm the mismatch: these models do not improve when handed a *longer* look-back (the fingerprint of latching onto something local), and their reported wins are confounded — the non-Transformer baselines they beat were *iterated* multi-step forecasters whose errors compound over the horizon, while the Transformers used *direct* multi-step. The win conflates "Transformer" with "direct multi-step," which the scaffold loop already hands everyone for free.

So rather than build a better Transformer, I build the simplest thing that removes the confound and ask how much of the reported gap survives. I propose **DLinear**: a single affine map along the time axis, applied to a moving-average-decomposed window, channel-shared, trained by the loop's direct MSE. The floor of a direct multi-step forecaster is one linear layer over time: for a univariate series, predict the horizon as $\hat{X} = W X$ with $W$ a $T\times L$ matrix, so row $t$ of $W$ says how to combine the $L$ past values into the $t$-th future value. This is not crippled by being linear, because the only structure long-horizon forecasting can recover *is* trend and periodicity, and both are at heart "the future is a fixed linear functional of the recent past": a periodic series lets a row of $W$ place its weight on the lag one period back; a trending series lets it place weight on a weighted difference that extrapolates the drift. A high-capacity nonlinear model would mostly fit noise that does not cross the train/test boundary. And structurally, every input step connects to every output step through a learned weight, so the longest signal path from any past observation to any future prediction is length one — no recurrence to forget through, no attention bottleneck; the long-range dependency lives in the weight matrix itself. Across the $C$ channels I share a single $W$ rather than learning one per channel, because within a dataset the channels share temporal dynamics (every electricity client has the same daily and weekly rhythm), which both encodes that prior and cuts the parameter count from $C\cdot T\cdot L$ to $T\cdot L$. I deliberately model *no* cross-channel coupling — that is the explicit bet this control is built to test.

The bare linear map has one weakness worth fixing, and the fix is what makes the method work. Picture a signal as a large slow ramp with a small daily oscillation riding on it. A single $W$ must fit both at once, but the two want different filters — the trend wants weights broadly smooth across the look-back, the seasonality wants weights sharply concentrated at the periodic lags — and because the trend is large in magnitude, in a squared-error fit its gradient dominates: $W$ spends itself getting the ramp roughly right and under-fits the small oscillation that carries the fine structure. So I separate the two streams with a parameter-free seasonal-trend decomposition before the linear maps. Estimate the trend by a moving average, take the residual as the seasonal part — the oldest idea in time-series analysis, already wired into Autoformer (Wu et al., 2021) and exposed in the scaffold as `series_decomp`. Decompose once: $\text{trend} = \mathrm{MovingAvg}(x)$, $\text{seasonal} = x - \text{trend}$. Give each stream its own linear map, $W_{\text{seasonal}}$ and $W_{\text{trend}}$, both $L\to T$, and sum the predicted streams:

$$\hat{X} = W_{\text{seasonal}}\cdot \text{seasonal} + W_{\text{trend}}\cdot \text{trend}.$$

I want to be honest that this adds no representational capacity: a moving average is linear, two linear maps and a sum are affine, so the whole thing is still a single affine map. What it adds is *conditioning* — a fixed reparameterization that separates the loud trend from the quiet seasonality so gradient descent fits each well instead of letting the trend's magnitude swamp the seasonality. It is exactly the preconditioning trick: same solution set, far better-behaved learning, paying off precisely when there is a clear trend, the case the bare map handled worst. The moving-average details decide the edge behavior: I average-pool with an odd kernel of size $k$, stride 1, and *replicate*-pad (front with $(k-1)/2$ copies of the first value, back with $(k-1)/2$ of the last) rather than zero-pad — zero-padding would drag the trend toward zero at the two ends and create spurious dips where I have least information; replicate-padding keeps the trend flat-but-faithful at the edges and, with $k$ odd, restores the length exactly. I take $k=25$, the value the decomposition block already uses (it smooths sub-daily wiggle on hourly data while preserving daily-and-slower trend), read from `configs.moving_avg`; under this fixed config every dataset shares one kernel where a per-dataset script could have tuned it. The shape bookkeeping is where this silently breaks, so I am explicit: the window is $[B, L, C]$; decompose along time to seasonal and trend, each $[B, L, C]$; permute time to last $[B, C, L]$, apply the two linears to get $[B, C, T]$, sum, permute back to $[B, T, C]$, slice the last $T$ steps.

```python
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """DLinear: moving-average decomposition + two linear maps, direct multi-step, channel-shared."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = getattr(configs, "individual", False)

        # parameter-free seasonal/trend split (replicate-padded moving average)
        self.decompsition = series_decomp(configs.moving_avg)          # configs.moving_avg == 25 here

        if self.individual:                                            # one linear pair per channel
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            for i in range(self.channels):
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:                                                          # default: weights shared across variates
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):                                             # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        # move time to the last axis so the linear maps act along time (L -> T)
        seasonal_init = seasonal_init.permute(0, 2, 1)               # [B, C, L]
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
            seasonal_output = self.Linear_Seasonal(seasonal_init)    # [B, C, T]
            trend_output = self.Linear_Trend(trend_init)             # [B, C, T]
        x = seasonal_output + trend_output                           # recombine the streams
        return x.permute(0, 2, 1)                                    # [B, T, C]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
