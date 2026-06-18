**Problem.** Fill `models/Custom.py` with the weakest honest classifier: read the padded window
`[B, seq_len, enc_in]` and emit `[B, num_class]` logits, using the cheapest model that actually reads the
window (not the zero-logit floor). It is the lower bound against which every later rung is measured.

**Key idea.** A decomposition-linear classifier. Split each channel's window into a moving-average
**trend** and a **seasonal** residual (`seasonal = x − MovingAvg(x)`), map each stream with its own
`Linear(seq_len, seq_len)` along time, sum them, then flatten the whole `[B, seq_len, enc_in]`
representation and project to `num_class`. The decomposition adds **no capacity** (moving-average + two
linear maps + sum is affine end to end) — it is preconditioning that lets the trend-dominated fit also
capture the quiet seasonal cue. The final flatten-and-project is the only place channels are mixed and the
class boundary is drawn.

**Why it works (and where it won't).** Trend carries the label on smooth spectral traces
(EthanolConcentration); the seasonal residual carries it on oscillatory gestures (Handwriting); decoupling
the two lets one affine model serve both. But the model is linear (no nonlinear boundary), ignores the
padding mask (`x_mark_enc` unused — padded zeros become learned features), and models cross-channel
interaction only through the head — so binary FaceDetection (forgiving) should be its relative best and
26-class Handwriting (blunt linear boundary) its relative worst.

**Scaffold edit / hyperparameters.** `series_decomp` with `moving_avg = 25` (odd kernel, stride 1,
replicate-padded `(k−1)//2` per side, length-preserving); channel-shared `Linear_Seasonal` and
`Linear_Trend`, both `Linear(seq_len, seq_len)`; head `Linear(enc_in · seq_len, num_class)`. Frozen
protocol: RAdam, `lr 1e-3`, `batch 16`, CrossEntropy, patience 10.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """DLinear classification fill: decomposition + two linear maps, flatten-and-project head."""

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # classification keeps pred_len == seq_len: the per-channel temporal map is square
        self.pred_len = configs.seq_len
        # series decomposition block (moving-average trend + seasonal residual), kernel = 25
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # classification head: flatten the (time x channel) representation, project to classes
        self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, seq_len, enc_in]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)            # [B, enc_in, seq_len]
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
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, enc_in, seq_len]
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output                         # recombine the streams
        return x.permute(0, 2, 1)                                  # [B, seq_len, enc_in]

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc)                              # [B, seq_len, enc_in]
        output = enc_out.reshape(enc_out.shape[0], -1)            # flatten time x channel
        output = self.projection(output)                          # [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
