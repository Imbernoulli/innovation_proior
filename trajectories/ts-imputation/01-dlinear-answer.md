**Problem.** Fill a 25%-masked length-96 multivariate window; error is scored only at the holes. The
identity scaffold predicts zero (the channel mean) at every hole — it ignores the temporal neighbours,
the simultaneous channels, everything. The first rung must use the rewarded context with the fewest moving
parts I can fully understand.

**Key idea (decomposition + linear interpolation along time).** Reconstruct each channel as a single
affine map of its own 96 timesteps — interpolation, with a length-one signal path from every observed
timestep to every masked one, no recurrence and no permutation-invariant attention to fight. To stop the
loud trend from swamping the quiet seasonality in the masked MSE, first split the window with the library's
parameter-free moving-average decomposition (`series_decomp`, kernel 25, replicate-padded edges) into
`trend = MovingAvg(x)` and `seasonal = x − trend`, give each its own time-axis linear map (96→96), and sum.
The whole thing is still one affine map end to end — this is preconditioning, not added capacity.

**Why.** With only 25% missing-at-random, every gap is interior with mostly-observed neighbours, so a
linear functional along time is exactly an interpolator; periodicity and trend, the only structure that is
recoverable, are both linear functionals of the window. No de-levelling wrapper (the forecasting trick):
the input is already standardised and mean-zero with zeros at the holes, and `pred_len = seq_len`.

**Hyperparameters.** `moving_avg=25`; channel-shared linear maps (`individual=False`); the loop's Adam
`lr=1e-3`, `batch_size=16`, `train_epochs=10`, `patience=3`, masked-only MSE. Channel-blind by design —
that blind spot (worst on the 321-channel ECL) is what the next rung attacks.

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
