The patch-Transformer beat the linear floor on every regime — Monthly 12.97 against DLinear's 13.39, Quarterly 10.22 against 10.50, Yearly 13.68 against 14.36 — with the largest absolute gain on Yearly (0.68 SMAPE), exactly where the affine map was most strained and where the reversible instance normalization and learned nonlinear representation were supposed to help most. So the two things DLinear structurally lacked were real, and adding them moved every number the right way. But the *size* of the move is the diagnostic: each gain is a few tenths, not a clear separation. That is the falsifiable case I had flagged — a 512-wide, 2-layer attention encoder over a handful of patch tokens (four for Monthly, two for Quarterly, less than one for Yearly) is near its ceiling. There simply are not enough patch tokens on a $2\cdot\text{pred\_len}$ window for self-attention to find structure a normalized linear map missed; most of the encoder's width is stranded. The lesson is concrete: stop adding generic capacity over a single-scale window. The information PatchTST leaves on the table is not "more attention" — it is the *multi-scale* structure of the series, which any single fixed-resolution view, patched or not, never exposes.

The structural argument for scale is this. The same underlying process presents *different* patterns at different sampling resolutions. At the finest resolution I see microscopic detail — the exact wiggle of the seasonal shape, short bursts. Coarsen by averaging adjacent steps and that detail washes out, leaving macroscopic structure — the slow trend, the broad cyclic envelope — far cleaner than it appears in the noisy fine view. A short M4 window holds both kinds superimposed, and the future is jointly determined by both: the next six Yearly steps are a trend question (a coarse-scale fact), the next eighteen Monthly steps a fine seasonal-shape question. A model reading the series at one resolution must disentangle trend from season within that view, and on a twelve-or-thirty-six-step window with two layers it cannot do that well. So the move is to build an explicit *ladder of scales* and let the model mix information across resolutions.

I propose **TimeMixer**: a fully MLP-based, channel-independent forecaster that reads the window at several resolutions, mixes a decomposed season/trend representation across scales, and predicts from every scale separately. The ladder is built by **average pooling** — downsample the window by a window of 2 a small number of times to get scales $x_0$ (the original), $x_1$ (half-length), and so on. Average pooling, not a learned conv or max-pool, because the coarse view should show the *central tendency* of each neighborhood: a learned conv is more expressive but distorts what "coarsening" means, and max-pool tracks peaks rather than the average a macroscopic view wants. Parameter-free coarsening keeps the ladder honest. On M4's short windows one downsampling step (window 2) is enough — a 36-step and an 18-step scale for Monthly, a 12-step and a 6-step for Yearly.

This is the first place the harness protocol bites, and it is load-bearing. The fixed Custom scripts do **not** pass `--down_sampling_layers`, `--down_sampling_window`, or `--down_sampling_method`, so under the harness those args default to $0$, $1$, and `None` — which would collapse the model to a *single scale*, throwing away the entire multi-scale thesis and leaving a fancy single-resolution MLP no better than PatchTST. So I set the downsampling configuration *inside* `Custom.py` (one downsampling layer, window 2, average pooling), reading `configs` only as an override. The whole contribution of this rung exists only if I do that explicitly.

The core is *how* to mix across scales, through a block I call Past-Decomposable-Mixing. Inside each block I first *decompose* every scale into season and trend with a moving-average block ($\text{trend} = \text{AvgPool}$, $\text{season} = x - \text{trend}$), because even a coarse series is still a season+trend mixture, and mixing entangled wholes would fight the very superposition the decomposition removes. Then the two components mix across scales in *opposite directions*, and this asymmetry is the heart of the method. Seasonal detail aggregates **bottom-up** (fine $\to$ coarse): coarse seasonality is literally an aggregation of finer seasonality — average a fine oscillation and you get the coarse envelope — so the fine scale holds information the coarse scale should receive. Trend mixes **top-down** (coarse $\to$ fine): for the macroscopic trend, fine detail is just noise, while the clean trend is most visible at the coarse scale, so the coarse scale holds the information the fine scale should receive. Reversing either direction destroys what made that scale valuable — pushing noisy fine detail into the trend, or coarse-smoothed season up where the detail was the point. The mixing primitive between adjacent scales is a two-layer MLP with a GELU between (resampling the temporal length from one scale to the next), added as a residual: a single linear could only resample, while an MLP can actually *transform* the donor scale into a useful supplement. After mixing, season and trend recombine and a channel feed-forward is applied with a residual. A few such blocks stack ($e_{\text{layers}}$, fixed at 2 by the harness).

The decoder is the second idea and the one that most directly answers PatchTST's stranded capacity. Instead of merging all scales into one representation and predicting once — wasting the complementary forecasting skills different scales have — I give **each scale its own predictor** (one linear map from that scale's length to the horizon) and **sum** the per-scale forecasts. This is Future-Multipredictor-Mixing: the coarse scale is good at the trend part of the forecast, the fine scale at the seasonal part, and summing lets each contribute what it forecasts best rather than forcing one merged feature through a single head. Summing versus averaging differs only by a constant the network absorbs, so the choice is immaterial; what matters is that the predictions stay *separate until the end*. The whole thing is wrapped in reversible per-scale instance normalization — the same level-decoupling that helped PatchTST, now applied per scale, with the summed forecast denormalized by the finest-scale statistics — and run channel-independently: each M4 series is its own univariate sequence ($\text{enc\_in}=1$), embedded $1 \to d_{\text{model}}$ and projected $d_{\text{model}} \to 1$ with shared weights, which is also the harness default ($\text{channel\_independence}=1$).

A note on where the protocol still strains this rung. The harness fixes $d_{\text{model}}=512$, $d_{\text{ff}}=512$, $e_{\text{layers}}=2$, whereas TimeMixer's own M4 script uses a *narrow* model ($d_{\text{model}}=16$, $e_{\text{layers}}=4$) — its premise is that an MLP mixer needs far less width than an attention model. So I again run it wider and shallower than it wants, but I expect the multi-scale structure to make the width *less* wasteful than for PatchTST: the per-scale predictors and cross-scale mixing give the 512-wide channels something real to do across two resolutions and a decomposed representation, rather than attending over two patch tokens. The genuine risk is the opposite of PatchTST's — with only one downsampling step on a short window the coarse scale is very short (six steps for Yearly), so the ladder is shallow and the multi-scale benefit may be modest on the shortest regime even if clear on the longer ones. I expect to beat PatchTST on *every* regime, with the largest gain where trend and season genuinely separate across scales and the window supports a real coarse view — Monthly and especially Yearly, where pushing the clean coarse trend top-down into the forecast is exactly the multi-scale move. If instead the gain is flat on the shortest regime, that signals one downsampling step is not enough scale separation, and points the next rung at finding the *period* structure directly inside the window rather than relying on a fixed pooling ladder to expose it.

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
