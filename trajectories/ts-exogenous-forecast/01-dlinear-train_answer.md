The scaffold hands me an empty `Model` whose `forecast` returns zeros — a forecaster that scores nothing. The research question is exogenous fusion, so the tempting first move is a wide cross-channel attention machine. I am deliberately not making it. If I open with a 512-wide Transformer that fuses channels and it scores well, I cannot tell how much of the score is the fusion and how much is just "a big nonlinear model fit this dataset." The first rung has to be the most minimal *honest* forecaster I can write — the control that measures how far the target's own history gets you *before* any exogenous fusion at all, so every later rung has a real number it must beat.

I propose **DLinear**: a channel-independent decomposition-linear forecaster. It is direct multi-step — it emits all 96 future steps in one shot rather than rolling a one-step map forward 96 times. That choice is not incidental. The iterated route conditions each step on the previous step's prediction, so errors compound and by step ninety the model is mostly forecasting from its own accumulated mistakes; the long-horizon forecasters that actually work (Informer's generative decoder, Autoformer, FEDformer) all abandoned iteration for one direct map from the whole history to the whole horizon, and I take that as settled.

The core of DLinear is the recognition that the absolute floor of a direct multi-step map is *a single affine map along the time axis*. For one channel with length-96 history $x_i$ and length-96 forecast $\hat x_i$,

$$\hat x_i = W x_i, \qquad W \in \mathbb{R}^{96 \times 96},$$

where row $t$ of $W$ says exactly how to combine the ninety-six past values into the $t$-th future value. No attention, no recurrence, no positional encoding bolted on — one linear layer applied along time. This is not a strawman, and it is worth seeing why. The entire signal of a time series lives in its *order*: a value means something because it sits on the rising shoulder of the evening peak, because the same shape recurred twenty-four and one-hundred-sixty-eight hours ago. A linear map reads the whole window at once through learned weights, so it sees that shape directly. Attention's core operation, by contrast, is permutation-invariant over tokens and re-injects order only as an additive side channel — it works against the grain of data whose whole content *is* order.

On top of the bare affine map I add one structural refinement, because it is the right inductive bias and costs almost nothing: a moving-average **series decomposition** before the linear maps. A load or temperature trace is a slow trend with a periodic component riding on it, and asking one map to capture both forces it to reconcile a near-DC drift with a sharp daily oscillation at very different scales. So I decompose first: split the look-back into a smooth trend (a fixed-kernel moving average, width 25, with reflect-style end padding so the trend keeps the input's length) and the remainder (the "seasonal" part left after subtracting it), forecast each with its own $96 \to 96$ linear map, and sum the two forecasts. The Time-Series-Library's `series_decomp` gives me exactly this, returning `(seasonal, trend)`. I initialize both maps' weights to $1/\text{seq\_len}$ everywhere — a uniform-average start, the sensible prior of "begin by predicting the mean of the window, then learn the deviations."

The defining choice of this rung is made *by omission*, and it is the choice the whole ladder is about. The two linear weight matrices are **shared across channels** (the `individual=False` path — one $W_{\text{seasonal}}$ and one $W_{\text{trend}}$ for every channel), and the maps act along the *time* axis with the channel axis treated as a batch dimension. That makes the model **channel-independent**: each channel is forecast from its own history alone, and the exogenous covariates never touch the target. The weather observations cannot inform the wet-bulb forecast; the other 320 clients cannot inform the one client I score. This is precisely the thing the research question asks me to improve, and I am deliberately not doing it yet — because I want to measure how much the target's own history already buys. What this rung provably *cannot* do is read the side channels, and that gap, isolated cleanly, is what the next rungs exist to close.

I expect ETTh1 — small, homogeneous panel, smooth strongly-autocorrelated oil-temperature target — to be respectable, since fusion has the least to add there. I expect Weather and ECL to be visibly loose, because there the target leans on its covariates and the discarded exogenous signal is exactly what is missing. That pattern, written in the metrics, is the case for fusion.

```python
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """Channel-independent decomposition-linear forecaster (DLinear)."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # series decomposition (moving-average trend + remainder)
        moving_avg = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(moving_avg)
        self.channels = configs.enc_in

        # one shared linear map per part, applied along the time axis (channel-independent)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)              # [B, L, C], [B, L, C]
        seasonal_init = seasonal_init.permute(0, 2, 1)               # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)        # [B, C, pred_len]
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)                                    # [B, pred_len, C]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
