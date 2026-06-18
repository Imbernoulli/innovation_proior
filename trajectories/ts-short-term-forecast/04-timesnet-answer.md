**Problem.** TimeMixer beat PatchTST everywhere (12.80 / 10.21 / 13.38) but stalled on Quarterly
(10.21 vs PatchTST's 10.22) — the regime with one sharp dominant period, where a blind average-pooling
ladder is the wrong lens. The unexploited structure is the *one-period-back* (interperiod) relationship,
which a 1D layout and a pooling ladder represent only incidentally: the interperiod neighbor at `t−p` is
`p` steps away and never made local.

**Key idea.** Discover the period structure inside the window with an FFT (top-`k` amplitude peaks,
DC removed; amplitude = period confidence), and for each period reshape the 1D series into a 2D grid
whose two axes are *intraperiod* (within-cycle shape, along columns) and *interperiod* (same phase
across cycles, along rows) — making both dependencies local at once. Process each 2D grid with a
shared parameter-efficient inception block (parallel multi-size 2D kernels), fuse the `k`
representations by a softmax over their amplitudes, stack a couple of these TimesBlocks residually with
LayerNorm, wrapped in reversible per-window instance normalization.

**Why.** A 1D conv/MLP/attention can present intraperiod variation as locality but hides interperiod
variation; the 2D reshape turns the `p`-apart same-phase points into adjacent ones a 2D kernel reads
directly — the thing the pooling ladder structurally cannot do. FFT discovers *which* period a series
carries instead of assuming it; the amplitude-softmax is the principled confidence-weighted fusion.
Sharing the inception across periods keeps size independent of `k`.

**Forecasting specifics.** The horizon must be created before the period machinery runs: instance-
normalize, embed the window to `d_model` (value + positional; the harness passes no time marks, so the
embedding accepts `x_mark=None`), then a linear `predict_linear` extends the temporal axis
`seq_len → seq_len + pred_len`. The TimesBlocks operate on the extended sequence (periods discovered
over `seq_len + pred_len`), a linear projection maps `d_model → 1`, denormalize, truncate to the last
`pred_len` steps.

**Hyperparameters.** `top_k=5`, `num_kernels=6` (configs defaults); under the fixed protocol
`d_model=512`, `d_ff=512`, `e_layers=2`; Adam `lr=1e-3`, batch 16, 10 epochs, patience 3, SMAPE loss.
(Wider than TimesNet's own M4 script; on Yearly's 12-step window the FFT has few bins, so the period
machinery is thin there.)

**What to clear.** A *non-uniform* expectation: clearest win on Quarterly (FFT locks the sharp 4-step
period, interperiod relation made local — should clear 10.21 with margin), roughly tied on Monthly
(~12.80, both models period-aware), likely *not* a win on Yearly (no real period to discover; TimeMixer's
top-down trend mixing is the better tool — may land slightly above 13.38). To be the strongest baseline
it needs a *mean* win driven by Quarterly outweighing a possible Yearly loss.

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
    frequency_list = abs(xf).mean(0).mean(-1)      # amplitude over batch and channels
    frequency_list[0] = 0                          # drop DC (the mean, not a period)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # one shared inception, reused for every period -> size independent of k
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
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # [B, length, N] -> [B, N, num_periods, period]: cols=intraperiod, rows=interperiod
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)                            # [B, T, N, k]
        # adaptive aggregation by amplitude-softmax
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                            # residual


class Model(nn.Module):
    """TimesNet: FFT period discovery + 2D inception conv, forecasting."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)                 # [B, seq_len, d_model]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # extend time
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))          # residual TimesBlocks
        dec_out = self.projection(enc_out)                             # [B, seq_len+pred_len, c_out]

        # de-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
