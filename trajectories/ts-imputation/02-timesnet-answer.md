**Problem.** DLinear beat mean-guessing everywhere but stalled on ETTh1 (0.1498 MSE — affine maps smear
nonlinear transients) and ECL (0.1132 MSE — channel-blind exactly where 321 clients are most correlated).
The next rung needs real nonlinearity *and* cross-channel information.

**Key idea (periodic 2-D modelling).** Several variations — trend, daily, weekly, within-day — overlap on
one 1-D axis and interfere. Discover a window's dominant periods by FFT (top-*k* amplitudes, DC zeroed),
fold the length-96 window into a 2-D array `(96/p, p)` per period so within-period and between-period
variation become orthogonal axes, run an Inception-style 2-D conv stack (genuine nonlinearity, aligned
with the data's structure), unfold, and aggregate the *k* views by softmaxed amplitude. Stack `e_layers`
such blocks with residual + LayerNorm. The `DataEmbedding` mixes all channels into the `d_model` features,
so the reconstruction is no longer channel-blind.

**Why (and the imputation-specific parts).** Normalisation is computed over *observed entries only* — mean
and std use `sum(mask == 1)` as the denominator and re-zero the holes after centring — so the punched-out
zeros do not corrupt the per-window centre/scale; both stats are detached and undone after the blocks.
There is no `predict_linear` (forecasting-only): `pred_len = seq_len`, the length is preserved, and the
final linear projects `d_model → c_out` at every position. Time features (`x_mark_enc`) feed the embedding
and hand the model the calendar phase the FFT folding exploits.

**Hyperparameters (the Custom.py eval config).** `d_model=d_ff=512`, `e_layers=2`, `top_k=5`,
`num_kernels=6`, `embed='timeF'`, `freq='h'`, `dropout=0.1`; loop Adam `lr=1e-3`, `batch_size=16`,
`train_epochs=10`, masked-only MSE. Lengths are taken from `seq_len` so the module is correct whatever
`pred_len` the harness passes.

```python
# models/Custom.py — step 2: TimesNet (FFT period folding + 2-D conv) for imputation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0                                   # drop DC (trend)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if self.seq_len % period != 0:                 # pad up to a whole number of periods
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([B, length - self.seq_len, N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)                            # 2-D conv over (cycles, phase)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)            # amplitude-weighted aggregation
        return res + x                                      # residual


class Model(nn.Module):
    """TimesNet for imputation."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len                     # imputation: pred_len = seq_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Non-stationary normalisation over OBSERVED entries only
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)     # [B, T, d_model], mixes channels
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)                 # [B, T, c_out]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return None
```
