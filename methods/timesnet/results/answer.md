# TimesNet, distilled

TimesNet is a task-general time-series backbone built around one representation change: for each
window it discovers the dominant periods by FFT, reshapes the 1D series into a set of 2D tensors
(one per period) whose two axes are the *intraperiod* (within-cycle) and *interperiod*
(across-cycle, same-phase) variation, and processes those 2D tensors with shared 2D convolutions.
The reshape turns both kinds of temporal dependency into 2D locality, so ordinary vision-style 2D
kernels can model them simultaneously — something a 1D layout cannot do, because same-phase points a
full period apart are never adjacent along the single time axis. For unsupervised anomaly detection
the same backbone is used as a reconstruction model: the per-point reconstruction error is the
anomaly score.

## Problem it solves

Unsupervised anomaly detection on multivariate monitoring streams (server metrics, spacecraft and
satellite telemetry). The model is trained only on normal windows to reconstruct them; at test time
the points it reconstructs poorly are flagged. So the task reduces to modeling the temporal
variation of a normal window well. Real variation mixes several overlapping periodicities, and each
point depends both on its temporal neighbors (short-range, intraperiod) and on the same phase in
adjacent cycles (long-range, interperiod); a 1D model can present only the first as locality, and
plain point-wise attention reconstructs worst here because its similarity is dominated by the many
normal points.

## Key idea

For a window `X_1D` of length `T` with `C` channels:

1. **Period discovery.** Compute `A = Avg_C(Amp(FFT(X_1D)))`, the amplitude spectrum averaged over
   channels. Zero out the DC term (`A[0]=0`: it is the mean, not a period). Take the top-`k`
   frequencies `f_1..f_k` by amplitude (only `f ≤ T/2`, by conjugate symmetry; top-`k` denoises the
   sparse spectrum), giving periods `p_i = ceil(T/f_i)` and amplitudes `A_{f_i}`.
2. **Reshape to 2D.** Zero-pad along time to a multiple of `p_i`, reshape to a
   `(length/p_i) × p_i` grid per channel: columns = intraperiod (within a cycle), rows = interperiod
   (same phase across cycles). This yields `k` 2D tensors of shape `[num_periods, p_i, C]`.
3. **2D convolution.** Process each 2D tensor with a parameter-efficient **inception block** —
   several 2D kernels of increasing size in parallel (`1×1, 3×3, …`), averaged — so the block is
   multi-scale. The same inception weights are **shared** across all `k` periods, making the model
   size independent of `k`. Reshape back to 1D and truncate to `T`.
4. **Adaptive aggregation.** Fuse the `k` representations weighted by `softmax(A_{f_i})`: the
   amplitude is the confidence of each period, so a softmax over the `k` amplitudes gives convex
   aggregation weights.
5. **Residual stacking.** `X^l = TimesBlock(X^{l-1}) + X^{l-1}`, each followed by LayerNorm; stack
   `e_layers` blocks; project back to channel space.

## Anomaly-detection specifics

A **per-window instance normalization** wraps the backbone (Series Stationarization, Liu et al.
2022): subtract the window's temporal mean, divide by its standard deviation (biased var + 1e-5
under the root, both detached), run the backbone, then de-normalize the output with the same
statistics — letting one shared backbone reconstruct windows spanning very different magnitudes.
Unlike the forecasting setup there is **no temporal extension and no decoder**: the task is
reconstruction, so `pred_len = 0`, the window goes in and the reconstruction of that same window
comes out at the same length, and the embedding uses only the value + positional branches (no time
marks). The training objective is **MSE** between the reconstruction and the input window. The
framework then takes the per-point squared reconstruction error as the anomaly score, sets the
threshold at the anomaly-ratio percentile of the score distribution, and reports point-adjusted F1.

## Final algorithm (one TimesBlock)

```
A, {f_1..f_k}, {p_1..p_k} = Period(X)          # FFT amplitude, drop DC, top-k
for i in 1..k:
    X2D_i = Reshape_{p_i}(Pad(X))              # [num_periods, p_i, C]; cols=intra, rows=inter
    Y2D_i = Inception(X2D_i)                   # shared multi-scale 2D conv
    Y1D_i = Trunc(Reshape_back(Y2D_i))         # [T, C]
w   = Softmax(A_{f_1}, ..., A_{f_k})           # amplitude = period confidence
X'  = sum_i w_i * Y1D_i                        # adaptive aggregation
return X' + X                                  # residual
```

## Default configuration (anomaly detection)

`top_k = 3`, a few residual layers (`e_layers ≈ 3`), inception with `num_kernels = 6`, `seq_len =
100`, `pred_len = 0`, `c_out = enc_in` (reconstruction). Model width sized to the channel count:
`d_model = d_ff = min(max(2^ceil(log C), d_min=32), d_max=128)`. Adam `lr = 1e-4`,
`(beta_1, beta_2) = (0.9, 0.999)`, batch size 128, up to 10 epochs, MSE loss. Inputs are Z-score
normalized per dataset; threshold at the `anomaly_ratio` percentile; metric point-adjusted F1.

## Relation to prior methods

- **Autoformer (Auto-Correlation).** Source of two reused ideas: estimating periods via the FFT and
  aggregating period-specific representations by a **softmax over confidences**. TimesNet uses the
  amplitude as the confidence and, crucially, reshapes to 2D so intra- and interperiod variation are
  separate convolvable axes rather than folded into a 1D roll-and-correlate.
- **GoogLeNet / Inception.** The multi-scale parallel-kernel block used to read the 2D tensors;
  sharing it across periods is what keeps parameters independent of `k`.
- **ResNet.** Residual stacking of the blocks for stable depth.
- **Non-stationary Transformer (Series Stationarization).** The instance-normalize / de-normalize
  wrapper, restoring each window's own scale around the shared backbone.
- **Transformer reconstruction baseline.** Outperformed on anomaly detection because point-wise
  attention is distracted by the dominant normal points, while period-aware 2D modeling highlights
  variations that violate the periodicity.
- **Generality.** Because the input is now a 2D grid, the inception block can be swapped for any 2D
  vision backbone (ResNet, ResNeXt, ConvNeXt, Swin); the inception version is chosen for efficiency.

## Working code

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
        super().__init__()
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
        super().__init__()
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
