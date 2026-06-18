**Problem (from step 2).** PatchTST gained on Handwriting (0.2306 → 0.2541) where local shape matters, was
flat on FaceDetection (0.6853, cross-channel), and slipped on EthanolConcentration (0.2890 → 0.2852) where
per-window normalization washes out the global trend — a sideways trade. Both rungs present time as a single
axis (no cross-period structure) and both ignore the padding mask.

**Key idea.** Reshape the 1-D series into 2-D so cross-period structure becomes locality. Discover the
top-`k` periods by FFT amplitude (drop DC), reshape the window into `k` grids of `num_periods × period` so
columns are intraperiod and rows are interperiod, run *one shared* multi-scale Inception block (2-D conv) over
each grid, reshape back, and fuse the `k` views by a softmax over their amplitudes. Stack residual TimesBlocks.
For the head, **multiply the per-timestep features by the padding mask before flattening** (`output *
x_mark_enc.unsqueeze(-1)`) — the mask-aware pooling both earlier rungs lacked — then flatten and project.

**Why it works (and where it won't).** A 2-D kernel reads within-cycle and across-cycle variation in one
receptive field — the cross-period structure patch attention could never span. TimesNet also does *not*
per-window normalize, so it keeps EthanolConcentration's global trend that PatchTST erased. Mask-aware pooling
removes padding's spurious contribution on variable-length datasets (Handwriting). It still mixes channels
only in the embedding/head, so FaceDetection (cross-channel) should be at best parity.

**Scaffold edit / hyperparameters.** `enc_embedding = DataEmbedding(enc_in, 128, embed, freq, 0.1)` (value +
positional, no marks); `e_layers=3` TimesBlocks, `top_k=3`, `num_kernels=6`, `d_ff=256`; head `Linear(128 ·
seq_len, num_class)`. Frozen protocol: RAdam, `lr 1e-3`, `batch 16`, CrossEntropy, patience 10.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]; find the k dominant periods by spectral amplitude
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0                              # drop DC: window mean, not a period
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]       # periods, per-batch amplitudes


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len               # 0 for classification
        self.k = configs.top_k
        # ONE shared inception (model size independent of k): d_model -> d_ff -> d_model
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # pad along time so length divides the period, reshape to 2-D
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # [B, length, N] -> [B, num_periods, period, N] -> [B, N, num_periods, period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)                       # multi-scale 2-D conv: intra + inter period
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # amplitude-softmax aggregation: amplitude = period confidence
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                 # residual connection


class Model(nn.Module):
    """TimesNet classification fill: 2-D period reshape + Inception, mask-aware pooling head."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)      # value + positional, no marks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)     # zero out padded positions before pooling
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)               # [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
