# TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting

## Problem

Forecast a length-`F` future from a length-`P` past window `x in R^{P x C}`. The core difficulty is
that one observed series mixes short seasonal detail, slower trend movement, and distribution drift,
and different sampling scales expose different parts of that mixture.

## Method

Build a multiscale ladder by downsampling the past observations:

```text
X = {x_0, ..., x_M},  x_0 = x,
x_m in R^{floor(P / 2^m) x C}.
```

The default downsampler is average pooling with window `2`. The implementation
also supports `max`, `avg`, `conv`, or no downsampling through `configs.down_sampling_method`.

Embed each scale and pass the list through stacked Past-Decomposable-Mixing blocks. In each block,
for every scale:

```text
s_m, t_m = SeriesDecomp(x_m)
t_m = MovingAvg(x_m)
s_m = x_m - t_m
```

Then mix the components across scales in opposite directions:

```text
season, bottom-up: for m = 1..M:
    s_m <- s_m + BottomUpMixing_m(s_{m-1})

trend, top-down: for m = M-1..0:
    t_m <- t_m + TopDownMixing_m(t_{m+1})
```

`BottomUpMixing_m` maps temporal length `floor(P / 2^{m-1}) -> floor(P / 2^m)`. `TopDownMixing_m`
maps `floor(P / 2^{m+1}) -> floor(P / 2^m)`. Both are `Linear -> GELU -> Linear` along the
temporal dimension. The signs are additive in both residual updates.

Future-Multipredictor-Mixing keeps scales separate until prediction:

```text
xhat_m = Predictor_m(x_m^L),  m = 0..M
xhat = sum_m xhat_m
```

Each `Predictor_m` first maps the scale's temporal length to `F` with one Linear layer, then
projects hidden features to output variates. Sum and average ensembles differ only by a constant:
an average ensemble mimics a sum ensemble by scaling predictors by `M + 1`, and a sum ensemble
mimics an average ensemble by scaling predictors by `1 / (M + 1)`.

## Implementation Notes

- Default decomposition is moving-average `series_decomp(configs.moving_avg)`. The
  implementation also includes an optional `DFT_series_decomp` branch.
- One reversible normalization module is created per scale. Each scale is normalized before
  embedding; the summed forecast is denormalized with scale-0 statistics.
- In `channel_independence == 1`, each variate is reshaped into the batch dimension, embedded as
  a univariate sequence, projected with `d_model -> 1`, reshaped back to `[B, F, C]`, and summed.
- In `channel_independence == 0`, the implementation uses a joint `enc_in -> d_model` embedding,
  applies a `cross_layer` to decomposed season/trend inside PDM, and adds a residual regression
  path in `out_projection`.
- The PDM formula has a residual FeedForward after recombining season and trend. This
  final `ori + out_cross_layer(...)` is applied only in the
  channel-independent branch.
- If `use_future_temporal_feature` is enabled, known decoder-side temporal features are embedded
  and added before projection. This uses known calendar/covariate information, not target future
  values.
- Default settings use two PDM blocks; long-term forecasting uses `M = 3`, while short-term
  forecasting uses `M = 1`. `d_model` varies by dataset in the reported configurations.

## Forecasting Code Core

A faithful forecast-path excerpt:

```python
import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list):                  # each [B, D, L_i], fine -> coarse
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
    def __init__(self, configs):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
                nn.GELU(),
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                ),
            )
            for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list):                   # each [B, D, L_i], fine -> coarse
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()                 # coarse -> fine traversal
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
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == "moving_avg":
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError("decompsition is error")

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]
        season_list, trend_list = [], []

        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        self.c_out = configs.c_out

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(configs.e_layers)
        ])
        self.preprocess = series_decomp(configs.moving_avg)

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )

        self.normalize_layers = nn.ModuleList([
            Normalize(
                configs.enc_in,
                affine=True,
                non_norm=True if configs.use_norm == 0 else False,
            )
            for _ in range(configs.down_sampling_layers + 1)
        ])

        self.predict_layers = nn.ModuleList([
            nn.Linear(
                configs.seq_len // (configs.down_sampling_window ** i),
                configs.pred_len,
            )
            for i in range(configs.down_sampling_layers + 1)
        ])

        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.out_res_layers = nn.ModuleList([
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])
            self.regression_layers = nn.ModuleList([
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == "max":
            down_pool = nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == "avg":
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.configs.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc

        x_enc = x_enc.permute(0, 2, 1)           # [B, C, T]
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]
        x_mark_sampling_list = [x_mark_enc]

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[
                    :, ::self.configs.down_sampling_window, :
                ]

        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list

        return x_enc_sampling_list, x_mark_enc

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return x_list, None
        out1_list, out2_list = [], []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)
        return out1_list, out2_list

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        return dec_out + out_res

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1)
                dec_out_list.append(dec_out.contiguous())
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, _, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list, x_mark_list = [], []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for x, x_mark in zip(x_list[0], x_mark_list):
                enc_out_list.append(self.enc_embedding(x, x_mark))
        else:
            for x in x_list[0]:
                enc_out_list.append(self.enc_embedding(x, None))

        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError("This excerpt covers the forecasting path only.")
```
