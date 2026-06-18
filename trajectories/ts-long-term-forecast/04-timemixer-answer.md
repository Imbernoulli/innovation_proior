**Problem (from step 3).** The ladder ping-ponged: PatchTST nails intra-series temporal detail but is
channel-blind (ETTh1 0.3794); iTransformer sees channels (ECL 0.1482) but crushes per-series detail and
regressed ETTh1 to 0.3950. Each rung picked one point on a hidden axis — the **scale** axis: patches are
a fine view, the whole-series token is a coarse view. The future depends on variations at several scales
at once, so collapsing to one representation before the head merges skills that should stay separate.

**Key idea.** Disentangle along *both* the season/trend axis and the scale axis, and predict per scale.
Build a downsampling ladder (avg-pool to window, /2, /4, /8). Past-Decomposable-Mixing (encoder): at
each scale decompose into season/trend, mix season **bottom-up** (fine→coarse) and trend **top-down**
(coarse→fine) — the two components carry information in opposite directions across scales.
Future-Multipredictor-Mixing (decoder): one linear predictor per scale maps to the full horizon, and the
per-scale forecasts are **summed** so each scale contributes its own skill. Channel-independent (the
setting that made the linear model and PatchTST competitive on the small datasets).

**Why it works.** Every series has multi-scale structure; only some datasets have strong channel
coupling — so multiscale disentanglement is the broader-generalizing lever than cross-variate attention.
Keeping a separate predictor per scale uses the fine view's high-frequency skill and the coarse view's
drift skill without merging them.

**Hyperparameters / edit-surface notes.** This is the sharpest "baseline ≠ paper" case: the loop passes
`down_sampling_layers=0`, `down_sampling_window=1`, `down_sampling_method=None`, which would collapse the
scale ladder to one scale and make the architecture inert. So the model **hardcodes its own scale config
internally** (`down_sampling_layers=3`, `window=2`, `avg`) — the only way the multiscale design exists
here. `series_decomp`, `DataEmbedding_wo_pos`, `Normalize` reused from the scaffold layers;
channel-independent path. The loop's fixed `lr=1e-4` / `batch_size=32` / 10 epochs is a real handicap
versus the method's `lr=0.01` / `batch_size=128` recipe.

```python
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class _Cfg:
    """Internal config shim: supply the multiscale knobs the harness does not pass."""
    def __init__(self, configs):
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, "label_len", 0)
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.embed = configs.embed
        self.freq = configs.freq
        self.e_layers = configs.e_layers
        self.moving_avg = configs.moving_avg
        self.use_norm = getattr(configs, "use_norm", 1)
        # hardcoded multiscale configuration (harness defaults would disable the architecture)
        self.down_sampling_layers = 3
        self.down_sampling_window = 2
        self.down_sampling_method = "avg"
        self.channel_independence = 1
        self.decomp_method = "moving_avg"
        self.top_k = 5


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up (fine->coarse) season mixing."""
    def __init__(self, configs):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                          configs.seq_len // (configs.down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                          configs.seq_len // (configs.down_sampling_window ** (i + 1))),
            ) for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down (coarse->fine) trend mixing."""
    def __init__(self, configs):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                          configs.seq_len // (configs.down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                          configs.seq_len // (configs.down_sampling_window ** i)),
            ) for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        self.decompsition = series_decomp(configs.moving_avg)
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff), nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]
        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decompsition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)               # channel-independent residual
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    """TimeMixer: multi-scale Past-Decomposable-Mixing + Future-Multipredictor-Mixing (channel-indep)."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        cfg = _Cfg(configs)
        self.configs = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.down_sampling_window = cfg.down_sampling_window
        self.channel_independence = cfg.channel_independence
        self.layer = cfg.e_layers

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(cfg) for _ in range(cfg.e_layers)])
        self.enc_embedding = DataEmbedding_wo_pos(1, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout)
        self.normalize_layers = nn.ModuleList([
            Normalize(cfg.enc_in, affine=True, non_norm=(cfg.use_norm == 0))
            for _ in range(cfg.down_sampling_layers + 1)
        ])
        self.predict_layers = nn.ModuleList([
            nn.Linear(cfg.seq_len // (cfg.down_sampling_window ** i), cfg.pred_len)
            for i in range(cfg.down_sampling_layers + 1)
        ])
        self.projection_layer = nn.Linear(cfg.d_model, 1, bias=True)

    def _multi_scale_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        x_enc = x_enc.permute(0, 2, 1)                          # B,C,T
        x_list = [x_enc.permute(0, 2, 1)]
        x_ori = x_enc
        for _ in range(self.configs.down_sampling_layers):
            x_samp = down_pool(x_ori)
            x_list.append(x_samp.permute(0, 2, 1))
            x_ori = x_samp
        return x_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_list = self._multi_scale_inputs(x_enc)
        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)   # channel-independent
            x_list.append(x)

        enc_out_list = [self.enc_embedding(x, None) for x in x_list]   # [B*N, T, d_model]
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)           # multipredictor sum
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None
```
