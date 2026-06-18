**Problem.** PatchTST beat the linear floor on every regime (12.97 / 10.22 / 13.68) but only by a few
tenths — a 512-wide attention encoder over 2–4 patch tokens on a `2·pred_len` window is near its
ceiling. The information left on the table is not "more attention"; it is the *multi-scale* structure
of the series, which any single-resolution view (patched or not) never exposes.

**Key idea.** A fully MLP-based forecaster that reads the window at several resolutions. (1) Build a
ladder by average pooling (window 2). (2) In each block, decompose every scale into season + trend,
then mix across scales in *opposite directions* — season **bottom-up** (fine → coarse, because coarse
seasonality aggregates fine detail), trend **top-down** (coarse → fine, because the clean trend lives
at the coarse scale and fine detail is noise to it) — via two-layer GELU MLPs added as residuals
(Past-Decomposable-Mixing). (3) Give each scale its own linear predictor to the horizon and **sum**
the per-scale forecasts (Future-Multipredictor-Mixing), so each scale contributes what it forecasts
best. Reversible per-scale instance normalization wraps it; channel-independent, shared weights.

**Why.** The same process shows different patterns at different resolutions, and the future depends on
several scales jointly; forcing one resolution to carry trend and season together is what capped
PatchTST. Decompose-before-mixing avoids entangling wholes; the bottom-up/top-down asymmetry routes
each component to where it is informative; per-scale summed predictors use the harness's wide channels
on real cross-scale structure instead of stranding them on two patch tokens. Average pooling (not
conv/max) gives a neutral central-tendency coarsening.

**Critical harness fix.** The fixed Custom scripts do **not** pass `--down_sampling_layers / _window /
_method`, so they default to `0 / 1 / None` — which would collapse this to a single-scale MLP and
discard the entire method. So the downsampling configuration is set **inside** `Custom.py`
(`down_sampling_layers=1`, `down_sampling_window=2`, average pooling), `configs` used only as override.

**Hyperparameters.** `down_sampling_layers=1`, `down_sampling_window=2`, avg pooling; `moving_avg=25`
(from configs); channel-independent (`enc_in=1` embedded `1→d_model`, projected `d_model→1`). Under the
fixed protocol `d_model=512`, `d_ff=512`, `e_layers=2`; Adam `lr=1e-3`, batch 16, 10 epochs,
patience 3, SMAPE loss. (Wider/shallower than TimeMixer's own narrow M4 script, but the per-scale
predictors and cross-scale mixing give the width real work; instance norm + early stopping regularize.)

**What to watch.** Should beat PatchTST on every regime, largest gain where trend and season live at
different scales and the window supports a coarse view — Monthly and especially Yearly (trend regime,
exactly what top-down trend mixing is for). If Yearly's 6-step coarse scale is too thin to help, that
signals one downsampling step is insufficient scale separation on M4's tiniest windows — pointing the
next rung at finding period structure *inside* the window rather than via a fixed pooling ladder.

```python
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up: fine -> coarse."""

    def __init__(self, seq_len, down_sampling_layers, down_sampling_window):
        super(MultiScaleSeasonMixing, self).__init__()
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
        super(MultiScaleTrendMixing, self).__init__()
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
        rev = trend_list.copy()
        rev.reverse()
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
    def __init__(self, seq_len, d_model, d_ff, moving_avg, down_sampling_layers, down_sampling_window):
        super(PastDecomposableMixing, self).__init__()
        self.decomp = series_decomp(moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(seq_len, down_sampling_layers, down_sampling_window)
        self.trend_mixing = MultiScaleTrendMixing(seq_len, down_sampling_layers, down_sampling_window)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
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
    """TimeMixer: multi-scale decomposable mixing, channel-independent."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        # harness does not pass these -> set TimeMixer-M4 defaults internally
        self.down_layers = getattr(configs, 'down_sampling_layers', 0) or 1
        self.down_window = getattr(configs, 'down_sampling_window', 1)
        if self.down_window <= 1:
            self.down_window = 2
        moving_avg = getattr(configs, 'moving_avg', 25)
        D = configs.d_model

        self.down_pool = nn.AvgPool1d(self.down_window)
        self.norm_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True) for _ in range(self.down_layers + 1)
        ])
        self.enc_embedding = DataEmbedding_wo_pos(1, D, configs.embed, configs.freq, configs.dropout)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(self.seq_len, D, configs.d_ff, moving_avg,
                                   self.down_layers, self.down_window)
            for _ in range(configs.e_layers)
        ])
        self.predict_layers = nn.ModuleList([
            nn.Linear(self.seq_len // (self.down_window ** i), self.pred_len)
            for i in range(self.down_layers + 1)
        ])
        self.projection_layer = nn.Linear(D, 1)

    def _multi_scale_inputs(self, x_enc):
        x = x_enc.permute(0, 2, 1)                          # [B, C, T]
        x_list = [x.permute(0, 2, 1)]
        for _ in range(self.down_layers):
            x = self.down_pool(x)
            x_list.append(x.permute(0, 2, 1))
        return x_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.size(0)
        x_list = self._multi_scale_inputs(x_enc)

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
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return out[:, -self.pred_len:, :]
        return None
```
