The linear interpolator did what I built it to do and its numbers say exactly where it runs out. On seed 42 it lands MSE 0.1498 / MAE 0.2693 on ETTh1, 0.0503 / 0.1143 on Weather, and 0.1132 / 0.2428 on ECL — a decisive win over mean-guessing everywhere. But the pattern across datasets is the diagnosis. Weather is its best result by a wide margin while ETTh1 and ECL are four-to-five times worse, and the two failures point in two different directions. ETTh1 is the trend-plus-seasonality strain: hourly transformer temperature has sharp local transients that an affine map of the window smears, and the decomposition only half-fixes it. ECL indicts the design directly — 321 strongly co-moving clients, smooth and periodic, so it is *not* a hard temporal problem, yet the channel-blind map throws away the simultaneous values of the hundreds of correlated channels and posts the worst MSE of the three. The next rung has to fix both: it needs genuine nonlinearity and it needs to look across channels.

I propose TimesNet: discover each window's dominant periods by FFT, fold the 1-D window into a 2-D array along those periods, and model it with 2-D convolution. The motivating observation is that the linear map got the *unit* right — a series is about shape over stretches, not isolated points — but modelled each channel as a flat 1-D vector, so the slow trend, the daily oscillation, the weekly one, and short fluctuations all live superimposed on one axis where they interfere, and disentangling them along a single axis is genuinely hard, linear or not. The fix is a change of representation. A periodicity is the statement that values one period apart are related, so if I know a channel's dominant period $p$ I can fold its length-96 window into a 2-D array of shape $(96/p, p)$: successive rows are successive periods, each column a fixed phase. Two kinds of variation then become two orthogonal axes — moving down a column traverses the same phase across cycles (the slow between-period variation: trend and the cycle's evolution), moving along a row traverses one full cycle (the fast within-period shape). The interference is gone because the scales sit on different axes, and a 2-D structure is precisely what 2-D convolution is built for: it mixes neighbouring phases and cycles at once, giving me real nonlinearity — stacked Inception-style convs with GELU — that is *aligned with* the data's structure rather than fighting it.

Which period? Real series carry several and I do not know them in advance, so I find them in the frequency domain. I take the rFFT of the window along time, average the amplitude over batch and channels to get one spectrum, zero the zero-frequency (DC) bin so the trend cannot masquerade as a period, and take the top-$k$ frequencies by amplitude, each giving a candidate $p_i = 96 / \text{freq}_i$. I fold the window by each of the top-$k$ periods separately, run the 2-D conv stack on each folded view, unfold back to 1-D, then combine the $k$ reconstructions. The natural weighting is the amplitudes themselves — a period carrying more spectral energy should count for more — so I softmax the top-$k$ amplitudes into weights and take the weighted sum:

$$\text{res} = \sum_{i=1}^{k} \text{softmax}(A)_i \cdot \text{Unfold}\big(\text{Conv2D}(\text{Fold}_{p_i}(x))\big),$$

so the model discovers per window which periodicities matter and leans on them proportionally, with no hand-set period and no single-period assumption. I stack `e_layers` such blocks, each with a residual connection so the representation is refined rather than replaced, and a LayerNorm between blocks for stability.

Two pieces are task-specific and I derive them rather than transplant the forecasting or anomaly-detection forms. First the input space: a 2-D conv needs a richer per-timestep representation than a raw scalar, since the conv channels carry the learned features, so I embed the masked window into a $d_\text{model}$-dimensional space with the library's `DataEmbedding` — value projection plus positional code plus the time-feature (`x_mark_enc`) embedding. The calendar stamp matters here in a way it did not for the linear map: it tells the model the phase of the daily and weekly cycle directly, exactly the structure the FFT folding exploits. Critically, that value projection mixes all `enc_in` channels into the shared $d_\text{model}$ features at each timestep, so unlike the channel-blind linear map the reconstruction *does* see cross-channel information — a masked ECL client can be filled from a representation that saw all 321 clients at that timestep. It is not explicit channel-attention, but it is not blind either.

Second — the part I must get right for imputation — normalisation under a mask. The Non-stationary normalisation that wraps the model (centre and scale per window, undo after) cannot use ordinary statistics, because a quarter of the entries are fake zeros that would corrupt the mean and width. So I compute statistics over the *observed entries only*: the mean is the sum of $x_\text{enc}$ over time divided by the observed count per channel, `torch.sum(mask == 1, dim=1)`; I subtract it and re-zero the masked positions with `masked_fill(mask == 0, 0)` so the holes stay holes after centring; the standard deviation is the root-mean-square of those centred, re-masked values, again over the observed count. Both statistics are detached so the normalisation is a fixed transform rather than something to backprop through, and after the blocks and the projection back to `c_out` I de-normalise by the stored std and mean. This masked normalisation is the single most important task-specific change — it lets the periodic 2-D modelling see the window's *shape* on a common scale without the punched-out zeros biasing the centre and width. And unlike the forecasting path there is no `predict_linear` stretching the temporal axis: for imputation $\text{pred\_len} = \text{seq\_len}$, the length is preserved end to end, the embedded window enters the blocks at its native length, and the final linear maps $d_\text{model} \to c\_\text{out}$ at every one of the 96 positions.

This answers both halves of the DLinear diagnosis. The nonlinearity is real — stacked 2-D convs with GELU over a folded window can represent the ETTh1 transients an affine map smeared — and the channel-blindness is addressed through the embedding's channel mixing, which on ECL should be the decisive difference. The largest gains should land on exactly DLinear's two failure points; on Weather, already DLinear's strong suit, I expect improvement at a smaller margin.

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
