# TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting

## Problem
Forecast a length-`F` future from a length-`P` past window of a `C`-variate series whose temporal
variations (season, trend, several rhythms, drift) are deeply mixed in a single stream. The same
process presents different patterns at different sampling scales — fine scales carry microscopic
detail, coarse scales carry macroscopic structure — and the future is jointly determined by, and
best forecast from, several scales together.

## Key idea
A fully MLP-based forecaster that (1) builds a multiscale ladder of the input by average pooling,
(2) inside each block decomposes every scale into season + trend and mixes them *across scales in
opposite directions* — seasons bottom-up (fine → coarse), trends top-down (coarse → fine) — via
**Past-Decomposable-Mixing (PDM)** blocks, and (3) predicts with one regressor per scale and sums
the forecasts — **Future-Multipredictor-Mixing (FMM)** — to exploit the complementary forecasting
skills of different scales. No attention, no recurrence.

## Method (precise form)
Downsample the input `x ∈ R^{P×C}` by average pooling with window 2 into `M+1` scales
`x_m ∈ R^{⌊P/2^m⌋ × C}`, `m = 0..M` (`x_0` finest). Embed each scale to `d_model` channels.

Stack `L` PDM blocks. For the `l`-th block, decompose each incoming scale representation with a
moving-average block (`trend = AvgPool`, `season = x − trend`):
- `s_m^l, t_m^l = SeriesDecomp(x_m^{l−1})`, `m = 0..M`
- Seasonal **Bottom-Up-Mixing** (residual), for `m: 1→M`:
  `s_m^l = s_m^l + BottomUpMixing(s_{m−1}^l)`, where
  `BottomUpMixing: ⌊P/2^{m−1}⌋ → ⌊P/2^m⌋` is two Linear layers with a GELU between, on the temporal
  dimension.
- Trend **Top-Down-Mixing** (residual), for `m: (M−1)→0`:
  `t_m^l = t_m^l + TopDownMixing(t_{m+1}^l)`, where
  `TopDownMixing: ⌊P/2^{m+1}⌋ → ⌊P/2^m⌋` is two Linear layers with a GELU.
- Recombine and apply a channel FeedForward with residual:
  `x_m^l = x_m^{l−1} + FeedForward(s_m^l + t_m^l)` (FeedForward = Linear→GELU→Linear over channels,
  width `d_ff`).

FMM: per-scale predictor (one Linear `⌊P/2^m⌋ → F` over time, then project channels → variates),
summed: `x̂_m = Predictor_m(x_m^L)`, `x̂ = Σ_{m=0}^{M} x̂_m ∈ R^{F×C}`.

Reversible per-scale instance normalization wraps the model; the final summed forecast is
de-normalized with the finest-scale statistics. The model runs channel-independently: each variate
is its own univariate sequence with shared weights (`1 → d_model` embedding, `d_model → 1`
projection). Trained with Adam `(β1,β2)=(0.9,0.999)` and an MSE (L2) loss. The usual stack uses
`L = 2`; for long-term forecasting the number of downsampling layers is `M = 3` with window 2,
`d_model = 16`, `d_ff = 32`.

## Why each choice
- **Average pooling** for downsampling: parameter-free, faithful coarsening; a learned conv is
  more expressive but less neutral, and max-pool tracks peaks rather than the central tendency a
  coarse view should show.
- **Decompose at every scale** before mixing: even coarse series remain a season+trend mixture;
  mixing entangled wholes fights the same superposition decomposition was meant to remove.
- **Bottom-up season / top-down trend** (the central asymmetry): coarse seasonality is an
  aggregation of finer seasonality, so detail flows up; fine detail is noise for macro trend, so
  the clean coarse trend flows down. Reversing either direction destroys the information that made
  that scale valuable.
- **Two-layer GELU MLP** as the mixing primitive: a single linear can only resample; an MLP can
  transform the donor scale into a useful supplement.
- **Per-scale predictors summed (FMM)**: scales have complementary forecasting skills; merging
  features before prediction wastes them. Sum vs. average differ only by a constant the network
  absorbs.
- **Channel independence**: robust and scalable on tens to hundreds of heterogeneous variates.

## Implementation (Time-Series-Library `Model` interface, channel-independent)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up: fine -> coarse."""
    def __init__(self, seq_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** i),
                          seq_len // (down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                          seq_len // (down_sampling_window ** (i + 1))),
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list):                       # each [B, D, L_i]
        out_high, out_low = season_list[0], season_list[1]
        out = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low = out_low + self.down_layers[i](out_high)
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out.append(out_high.permute(0, 2, 1))
        return out


class MultiScaleTrendMixing(nn.Module):
    """Top-down: coarse -> fine."""
    def __init__(self, seq_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                          seq_len // (down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** i),
                          seq_len // (down_sampling_window ** i)),
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):                        # each [B, D, L_i]
        rev = trend_list.copy(); rev.reverse()
        out_low, out_high = rev[0], rev[1]
        out = [out_low.permute(0, 2, 1)]
        for i in range(len(rev) - 1):
            out_high = out_high + self.up_layers[i](out_low)
            out_low = out_high
            if i + 2 <= len(rev) - 1:
                out_high = rev[i + 2]
            out.append(out_low.permute(0, 2, 1))
        out.reverse()
        return out


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.decomp = series_decomp(configs.moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(
            configs.seq_len, configs.down_sampling_layers, configs.down_sampling_window)
        self.trend_mixing = MultiScaleTrendMixing(
            configs.seq_len, configs.down_sampling_layers, configs.down_sampling_window)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list):
        season_list, trend_list = [], []
        for x in x_list:
            s, t = self.decomp(x)
            season_list.append(s.permute(0, 2, 1))
            trend_list.append(t.permute(0, 2, 1))
        season_list = self.season_mixing(season_list)
        trend_list = self.trend_mixing(trend_list)
        out_list = []
        for x, s, t in zip(x_list, season_list, trend_list):
            out_list.append(x + self.out_cross_layer(s + t))
        return out_list


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.down_layers = configs.down_sampling_layers
        self.down_window = configs.down_sampling_window
        D = configs.d_model

        self.down_pool = nn.AvgPool1d(configs.down_sampling_window)
        self.norm_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True) for _ in range(self.down_layers + 1)
        ])
        self.enc_embedding = DataEmbedding_wo_pos(1, D, configs.embed, configs.freq, configs.dropout)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(configs.e_layers)
        ])
        self.predict_layers = nn.ModuleList([
            nn.Linear(self.seq_len // (self.down_window ** i), self.pred_len)
            for i in range(self.down_layers + 1)
        ])
        self.projection_layer = nn.Linear(D, 1)

    def __multi_scale_inputs(self, x_enc):
        x = x_enc.permute(0, 2, 1)                          # [B, C, T]
        x_list = [x.permute(0, 2, 1)]
        for _ in range(self.down_layers):
            x = self.down_pool(x)
            x_list.append(x.permute(0, 2, 1))
        return x_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.size(0)
        x_list = self.__multi_scale_inputs(x_enc)

        ci_list = []
        for i, x in enumerate(x_list):
            x = self.norm_layers[i](x, "norm")
            _, Li, N = x.shape
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, Li, 1)
            ci_list.append(x)

        enc_list = [self.enc_embedding(x, None) for x in ci_list]

        for block in self.pdm_blocks:
            enc_list = block(enc_list)

        dec_list = []
        for i, enc in enumerate(enc_list):
            dec = self.predict_layers[i](enc.permute(0, 2, 1)).permute(0, 2, 1)
            dec = self.projection_layer(dec)
            dec = dec.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1)
            dec_list.append(dec)
        dec_out = torch.stack(dec_list, dim=-1).sum(-1)
        dec_out = self.norm_layers[0](dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
```
