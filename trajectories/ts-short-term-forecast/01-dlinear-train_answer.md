The scaffold returns zeros, so there is no forecaster at all; the right place to begin is not with the heavy machinery that preceded this problem but with the simplest thing that can fit M4 and whose failures I can read without ambiguity. The lineage just before me — iterated RNNs, then the Informer/Autoformer/FEDformer line of attention forecasters — poured its capacity into long-range dependency modeling. But on M4 the series are short (Monthly $\text{seq\_len}=36$, Quarterly $16$, Yearly $12$), one channel, a direct horizon to fill ($18$, $8$, $6$), and the dominant structure is plain: a slow level drift plus a repeating seasonal shape. Both of those are *linear* in the look-back — extrapolating a drift is a linear extrapolation, and reading off the value one period back is a linear read of a past coordinate. So before I pay for anything heavier I want to test the minimal hypothesis that a single linear map from the window to the horizon captures most of what is forecastable here, and that the rest is unnecessary or actively harmful on these tiny windows.

I propose **DLinear** — a decomposition front-end followed by two channel-shared linear maps, predicting the whole horizon directly. The cleanest direct-multi-step forecaster is one linear layer along time: from a look-back $x \in \mathbb{R}^{L}$ produce $\hat{x} = W x$ with $W \in \mathbb{R}^{T \times L}$, so every past step connects to every future step through one learned weight and the signal path length is one. This is deliberate: there is no recurrence to compound error over the horizon — the alternative, predict-one-step-and-feed-it-back, accumulates noise, and on Yearly the horizon is six steps of pure trend where compounding would be most visible — and no attention to tokenize single steps that carry no standalone meaning.

The one refinement on top of a bare linear map is what makes it work. A single $W$ fit by the loss has a known pathology: when a series carries a strong trend, the large-magnitude trend dominates the error budget, the map spends its weights tracking the level, and the smaller-magnitude seasonal shape is under-fit. The remedy is to separate the loud component from the quiet one *before* fitting, which is exactly what the decomposition primitive from the attention-forecaster lineage gives for free. A length-preserving moving average extracts the trend, $\text{trend} = \text{MovingAvg}(x)$, and the residual is the seasonal part, $\text{seasonal} = x - \text{trend}$. Each gets its own linear map and the outputs are summed:
$$\hat{x} = \text{Linear}_T(\text{trend}) + \text{Linear}_S(\text{seasonal}).$$
Crucially this adds *no* representational capacity — a moving average followed by two linear maps and a sum is still affine end to end. It is not a more powerful model; it is *preconditioning* that lets each linear layer specialize on a component of roughly uniform scale, which should help most on the regimes with a clear trend (Yearly, and the trending Monthly series) and cost nothing on the rest.

The moving average uses the canonical kernel of 25, the same smoothing scale the decomposition block uses elsewhere, with replicate-padding at both ends — $(k-1)//2$ copies of the first and last value — so the trend stays faithful at the window edges instead of being pulled toward zero. The kernel is read from `configs.moving_avg` (default 25), and under the fixed Custom protocol that flag is not overridden, so it stays 25. On the short regimes a kernel of 25 is wider than the window (Yearly is only 12 steps), but with replicate padding the average still returns a valid length-$L$ output: it simply smooths very aggressively, the trend collapsing toward the window mean and almost all variation flowing into the seasonal residual. That is a perfectly reasonable "trend $\approx$ window mean, seasonal $\approx$ everything else" split for series this short, and it does not crash.

The linear maps are *channel-shared* — one $\text{Linear}_S$, one $\text{Linear}_T$ — not per-channel. On a multivariate dataset that cuts parameters from $C \cdot T \cdot L$ to $2 \cdot T \cdot L$ and avoids fitting spurious cross-channel coupling; here, with one channel, it collapses to a single pair of $T \times L$ maps (Monthly $18 \times 36$, a few hundred parameters), and that tiny footprint is the point. Under the fixed protocol ($\text{lr}=10^{-3}$, batch 16, 10 epochs, patience 3) a model this small cannot overfit M4's short series and trains to convergence well inside ten epochs. I deliberately ignore the harness's capacity knobs — it passes $d_{\text{model}}=512$, $e_{\text{layers}}=2$, $n_{\text{heads}}=8$, but DLinear has no embedding, no attention, no notion of width, and the maps are sized purely by $\text{seq\_len}$ and $\text{pred\_len}$. Being blind to those knobs is the cleanest statement of the hypothesis that capacity is not what M4 needs.

I also leave out instance normalization on purpose. DLinear in its plain decomposition form does not re-center the window (its NLinear sibling does, by subtracting the last value and adding it back). Keeping the floor un-normalized lets the *data* make the distribution-shift point rather than me conceding it in advance, and the SMAPE loss the harness optimizes, $(200/T)\sum_t |y_t - \hat{y}_t| / (|y_t| + |\hat{y}_t|)$, is already a percentage error that is scale-robust across series — it does not let a large-magnitude series dominate the gradient the way a raw MSE would. So this floor is *fair*: whatever SMAPE it reaches is genuinely the best a purely affine, un-normalized direct-multi-step model can do here, and any gain a richer rung shows is attributable to capacity or normalization rather than to the linear model having been set up to fail.

What this floor cannot do is the diagnosis I am setting up. Being affine, it can capture exactly two things — a linear extrapolation of the trend and a fixed linear combination of past values that reproduces a periodic shape — and two failure modes are baked in. First, every M4 series sits at a different level and scale, and a single shared map with no per-window normalization cannot decouple "what shape" from "what level," so series far from the training-set average level are systematically off. Second, an affine map cannot represent any interaction between trend and season or any shape that is not a fixed linear function of the window. I therefore expect SMAPE worst on Yearly (shortest window, trend-dominated, six-step horizon, no repeating shape to lock onto) and most competitive on Monthly (the 36-step window spans three full 12-step cycles, so the seasonal map has enough periods to fit), with Quarterly in between. The per-regime spread is the first clue to what the next architecture must add.

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
