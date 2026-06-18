**Problem.** Fill the frozen direct-multi-step loop with the most honest possible control: the simplest
architecture that could forecast at all, so the heavier rungs above can be measured against something
whose every part is understood. The attention-based lineage (Informer/Autoformer/FEDformer) tokenizes
per step and runs a permutation-invariant operator over data whose content is order, and its reported
wins are confounded with simply using direct multi-step prediction — which the loop already provides.

**Key idea.** Strip attention entirely. Predict the horizon as a single affine map along time, L→T,
applied channel-shared, after a parameter-free moving-average **series decomposition**: trend =
MovingAvg(x), seasonal = x − trend, one linear map per stream, summed. Periodicity and trend are exactly
what a linear temporal map captures (each future step is a fixed linear functional of the past), and
decomposition is preconditioning — it stops the loud trend from swamping the quiet seasonality in the
squared-error gradient without adding any representational capacity (a moving average is linear, so the
whole thing is still one affine map).

**Why it works.** Long-horizon forecasting is only feasible for series with clear trend and
periodicity, both of which the two linear maps represent directly with a length-one signal path from
every past step to every future step. Sharing W across channels encodes the shared diurnal/weekly rhythm
and slashes parameters; refusing to model cross-channel coupling is the explicit bet being tested.

**Hyperparameters / edit-surface notes.** Kernel `moving_avg=25` read from `configs` (the loop's
default; note this is fixed across all three datasets here, where a per-dataset script could tune it).
`series_decomp` reused from `layers.Autoformer_EncDec` (replicate-padded so the trend stays faithful at
the window edges). Channel-shared linears by default (`individual=False`); weights initialized to the
uniform 1/L average as in the reference block. Trained by the loop's direct MSE, Adam `lr=1e-4`,
`batch_size=32`, `train_epochs=10`, `patience=3`.

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
