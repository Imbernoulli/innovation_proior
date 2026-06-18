## DLinear reconstructor, distilled

A deliberately under-flexible reconstructor: split the window into a moving-average trend and a seasonal
residual (parameter-free), reconstruct each with its own one-layer linear map along time, and sum. No
attention, no nonlinearity beyond the decomposition; weights shared across channels. The reconstruction
MSE is the anomaly score.

## Problem it solves

The patch Transformer reconstructs *too well* — its 512-wide encoder and flatten head partly reconstruct
the anomalies too, smoothing the error spikes and dropping recall on MSL (0.7130) and SMAP (0.5557).
DLinear spends capacity only on trend-plus-periodicity, the structure that characterizes a *normal*
window, so anomalies remain as residual error.

## Key idea

- **Linear reconstruction along time.** A normal window is trend + periodicity, both of which are linear
  functionals of the window; a single `seq_len → seq_len` linear map captures them with signal-path
  length one and almost no capacity to fit an anomaly's idiosyncratic shape. Less flexibility is the cure
  for the over-flexible backbone.
- **Trend/seasonal decomposition as preconditioning.** One linear map under-fits the quiet seasonality
  because the loud trend dominates the squared-error gradient. Split with a length-preserving moving
  average (replicate-padded edges, kernel 25), give the trend and seasonal streams their own linear map,
  and sum. End-to-end this is still affine — no added capacity, just conditioning so each component fits.
- **Channel-shared weights.** One linear map applied to every channel encodes the shared per-dataset
  dynamics and avoids overfitting / spurious cross-channel structure in the normal data.
- **No instance norm (scaffold-specific).** The TS-Lib anomaly path feeds the raw (per-dataset Z-scored)
  window straight into the decomposition; the trend linear absorbs the level. Weights are initialized to
  the uniform `1/seq_len` average.

## Why it should help over the patch backbone

An under-flexible linear reconstructor reproduces normal periodic structure but leaves anomalies as
residual, recovering the recall the patch backbone lost — expected to lift F1 most on MSL.

## Hyperparameters

`seq_len=100`, `pred_len=seq_len`, `moving_avg=25`, `enc_in=c_out=C`, channel-shared (`individual=False`);
Adam `lr=1e-4`, `batch_size=32`, `train_epochs=3` (TS-Lib uses 3/10/3 epochs for PSM/MSL/SMAP).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """Decomposition + two linear maps, reconstruction, channel-shared."""

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # for anomaly detection (reconstruction) the output length equals the input length
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' \
                or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # parameter-free seasonal-trend split (length-preserving moving average, replicate-padded)
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:                       # one linear pair per channel (rarely needed)
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:                                     # default: weights shared across variates
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):                          # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        # move time to the last axis so the linear maps act along time (L -> L)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output         # recombine the streams
        return x.permute(0, 2, 1)                  # [B, L, C]

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)                 # [B, L, C] reconstruction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out                         # [B, L, D]
        return None
```
