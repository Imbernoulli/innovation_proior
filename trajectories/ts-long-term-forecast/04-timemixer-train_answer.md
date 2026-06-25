The inversion delivered exactly the split I predicted, and the split is the lesson. ECL — the 321-channel laggard — fell from $0.1819$ to $0.1482$, the largest single-dataset MSE drop on the ladder, with MAE crashing $0.2743\to 0.2398$: the cross-variate lever paying off precisely where it should. But ETTh1 went the other way, $0.3794\to 0.3950$, worse than PatchTST and essentially back at the linear control — the over-parameterization of 7 variate tokens of dimension 512 with weak coupling, plus the inversion throwing away the fine intra-series temporal detail PatchTST's patches preserved. Weather barely moved. So the mean improved ($0.2451\to 0.2395$) but the improvement is entirely ECL, and the ladder has been ping-ponging: PatchTST nails intra-series temporal detail but cannot see channels; iTransformer sees channels but crushes per-series detail into one coarse vector. Each rung's virtue is the other's missing piece. The next rung has to stop choosing.

So I step back from "temporal attention vs cross-variate attention" entirely. I have a length-$P$ window and want the length-$F$ future, and the killer is not the regression but that the past window is a *tangle* — climbing, falling, oscillating at a couple of rhythms, and slowly drifting, all superimposed in the same scalar stream. Whatever I build must *disentangle* before predicting, the same tension the linear control hit and that decomposition fixed — and both stronger rungs quietly dropped that trend/seasonal split. The field has two ways to disentangle, and both cut along a single axis. Season-trend decomposition isolates the slowly varying trend (a moving average) from the fast repeating season (the residual), because the two halves behave differently; but it does this at *one resolution*, one kernel on the series at its native rate. Multiperiodicity finds dominant periods and folds the series into a 2D period-by-cycle tensor; also single-axis, and heavy. One observation keeps nagging: take a road sampled every five minutes — a sharp commute spike twice a day; average it to a daily series and the spike is *gone*, leaving weekday-versus-weekend structure; average to monthly and only a slow seasonal drift remains. An average pool is a low-pass filter, so climbing a downsampling ladder slides the dominant content monotonically from microscopic detail to macroscopic structure. Decomposition cuts season from trend, periodicity cuts one period from another, but nobody is cutting along the *scale axis itself* — and this reframes the ping-pong: PatchTST's patches are a *fine*-scale view, iTransformer's whole-series token is a *coarse*-scale view. Each rung literally picked one point on the scale axis. A second fact, specific to forecasting, sharpens it: the future is jointly determined by variations at several scales — the next hour depends on both the immediate five-minute momentum and on which day of the week it is — so a fine-view predictor and a coarse-view predictor have different *skills*, and collapsing everything to one representation before the head merges skills before I can use them. Pyraformer and SCINet build multiple resolutions internally but merge and predict once; multiscale *prediction* is the opening.

So I propose **TimeMixer**: disentangle along *both* the season/trend axis and the scale axis, and predict per scale. Stage one builds the scale ladder — downsample the input window with average pooling to window, window/2, window/4, window/8 — a list of progressively coarser views of the same series. Stage two has two distinct mixings. The encoder, *Past-Decomposable-Mixing*: at each scale, decompose into season and trend with `series_decomp`, then mix *across scales separately for each component*, in opposite directions. Season mixes **bottom-up** (fine→coarse): the fine-scale seasonal detail informs coarser seasonal patterns, because a clean micro-period constrains the macro-period. Trend mixes **top-down** (coarse→fine): the coarse-scale trend, the reliable slow drift, informs the finer trends, because the macro-drift sets the baseline the micro-trends ride on. The asymmetry is the point — the two components carry information in opposite directions across scales, so mixing them with the same direction, or merging them, would smear exactly the structure I want to keep. The decoder, *Future-Multipredictor-Mixing*: one linear predictor per scale maps that scale's encoded representation straight to the full horizon $F$, and I *sum* the per-scale forecasts, so each scale's predictor contributes its own skill — fine→high-frequency, coarse→drift — instead of one head over a merged representation. The channel decision is informed by the whole ladder: iTransformer's cross-variate attention won ECL but cost ETTh1, so I keep *channel-independence* (treat each channel as its own series through a shared backbone, the setting that made the linear control and PatchTST competitive on the small datasets). The bet here is different from iTransformer's — that multiscale disentanglement generalizes across datasets better than cross-variate correlation does, because every series has multi-scale structure while only some datasets have strong channel coupling — so TimeMixer should recover the ETTh1 ground iTransformer lost and hold most of the ECL ground.

The edit surface bites hardest here, and I must be explicit. The whole architecture depends on the downsampling configuration, but the loop passes `down_sampling_layers=0`, `down_sampling_window=1`, `down_sampling_method=None` (run.py defaults), which would collapse the scale ladder to one scale and make the architecture inert. So the model **hardcodes its own scale config internally** — 3 downsampling layers, window 2, average pooling — via a config shim; that is the only way the multiscale design this rung is about actually exists here. TimeMixer's natural recipe also trains at `lr=0.01` / `batch_size=128` for up to 20 epochs with a small `d_model`, while the loop fixes `lr=1e-4` / `batch_size=32` / 10 epochs — a real handicap, since TimeMixer is tuned to train fast at high learning rate. This is the sharpest "same-named baseline is not the original" case on the ladder: faithful only because the model supplies the multiscale config the harness omits.

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
