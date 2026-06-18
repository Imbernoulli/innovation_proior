## TimesNet reconstructor, distilled

A period-aware reconstruction backbone. For each window it discovers the dominant periods by FFT,
reshapes the 1D series into a set of 2D tensors (one per period) whose two axes are the *intraperiod*
(within-cycle) and *interperiod* (across-cycle, same-phase) variation, and processes them with shared
multi-scale 2D convolutions. The reshape turns both temporal dependencies into 2D locality, and the
per-window period discovery adapts the representation to whatever rhythm the window carries. The
reconstruction MSE is the anomaly score.

## Problem it solves

DLinear's single fixed, position-indexed linear map suits steady-period datasets (PSM, MSL) but fails on
SMAP, whose telemetry shifts its rhythm window to window — a fixed lag pattern is wrong for most windows,
smearing the periodic structure and dropping recall to 0.5383. The backbone must discover and adapt to
the periods present in each window, and separate the within-cycle shape from the across-cycle relation.

## Key idea

1. **Period discovery.** FFT the window, average amplitude over batch and channels, zero the DC term,
   take the top-`k` frequencies → periods `p_i = T // f_i` and amplitudes `A_{f_i}`. Per-window, so the
   representation adapts (the adaptivity the fixed linear map lacked).
2. **Reshape to 2D.** Zero-pad to a multiple of `p_i`, reshape to a `(length/p_i) × p_i` grid per
   channel: columns = intraperiod, rows = interperiod. Both dependencies become 2D-local.
3. **Multi-scale 2D conv.** A parameter-efficient inception block (parallel kernels of increasing size,
   mean-combined) reads each grid; shared across all `k` periods so model size is independent of `k`.
4. **Adaptive aggregation.** Fuse the `k` representations weighted by `softmax(A_{f_i})` — amplitude as
   the period's confidence — so each window gets its own convex combination.
5. **Residual stacking + reconstruction.** `X^l = TimesBlock(X^{l-1}) + X^{l-1}` with LayerNorm; project
   back to channels. Reconstruction (no horizon, no decoder), wrapped in per-window instance norm — the
   reversible normalization returns here because a deep conv backbone, unlike DLinear's trend linear,
   cannot absorb the level itself.

## Why it should beat DLinear

The per-window period adaptivity directly targets DLinear's SMAP regression; the multi-scale 2D conv
catches structure the flat linear map smears on PSM/MSL. Point-aware period modeling highlights
variations that violate the periodicity — exactly the anomalies — rather than being distracted by the
dominant normal points.

## Hyperparameters

`top_k=3`, `num_kernels=6`, `e_layers` per dataset (TS-Lib uses 2/1/3 for PSM/MSL/SMAP),
`d_model=d_ff` per dataset (64/8/128), `seq_len=100`, `pred_len=0`, `c_out=enc_in`; Adam `lr=1e-4`,
`batch_size=128`, MSE loss. Embedding uses value + positional branches only (no time marks).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)     # amplitude over batch and channels
    frequency_list[0] = 0                         # drop DC (the mean, not a period)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len          # 0 for anomaly detection
        self.k = configs.top_k
        # one shared inception, reused for every period -> size independent of k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()                                        # T = seq_len (pred_len = 0)
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # [B, length, N] -> [B, N, num_periods, period]: cols=intraperiod, rows=interperiod
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)                                  # multi-scale 2D conv
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)       # back to 1D
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)                            # [B, T, N, k]
        # adaptive aggregation by amplitude-softmax
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                            # residual


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # reconstruction: project the representation straight back to the input channels
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def anomaly_detection(self, x_enc):
        # per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)                 # [B, seq_len, d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))     # residual TimesBlocks + LayerNorm
        dec_out = self.projection(enc_out)                        # [B, seq_len, c_out]

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out                                        # [B, seq_len, c_out] reconstruction
        return None
```
