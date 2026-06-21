The forecasting problem I keep returning to is that a single observed window is never one behavior. A length-$P$ past $x \in \mathbb{R}^{P \times C}$ that I want to map to a length-$F$ future is, in reality, a slow level and trend, one or more seasonal rhythms, and local noise all summed into the same stream — and on top of that the level and spread drift between training and test windows. If I ask one representation at the native sampling rate to explain all of it, that representation has to treat microscopic wiggles and macroscopic drift as though they live at the same resolution, which is wrong before I have even chosen a neural block. The existing tools each capture one piece of the answer but not the whole. Moving-average seasonal-trend decomposition, as used inside Autoformer, splits a slow trend from a faster residual, but it is applied at the model's single chosen scale, so it never coordinates across resolutions. DLinear shows that plain temporal Linear maps from past length to future length are strong and efficient, but it works from one native-resolution view. Multiperiodicity models like TimesNet disentangle variation along a period axis, which is not the same as a fine-to-coarse sampling ladder. Pyramidal and splitting-tree models such as Pyraformer and SCINet do build multiscale internal representations, but their forecast interface still consumes one merged representation rather than keeping a distinct forecasting role for each scale. And channel-independent forecasters like PatchTST regularize well across wildly varying channel counts but leave open how to combine that regularization with multiscale temporal modeling. What is missing is a design that says explicitly what stays separate, what gets mixed, in which direction, and at what moment the scales collapse into a forecast.

I propose TimeMixer, a decomposable multiscale mixing architecture built entirely from average pooling, moving-average decomposition, Linear layers, GELU, and reversible per-instance normalization — no attention, no recurrence. The first move is to make the multiple views explicit by building a scale ladder from the past observations, $X = \{x_0, x_1, \dots, x_M\}$ with $x_0 = x$ and $x_m \in \mathbb{R}^{\lfloor P/2^m \rfloor \times C}$. Average pooling with window $2$ is the default coarsener precisely because it is parameter-free and expresses the resolution change directly; pooling is a low-pass operation, so fine scales preserve high-frequency detail while coarse scales emphasize slower structure. A learned strided convolution is available, but it is not the cleanest starting point when the goal is simply to expose resolution. Each scale is normalized before embedding, with its own reversible normalization module storing that scale's statistics, because pooling changes variance and the windows themselves drift in level and spread.

The crucial decision is not to mix the scales as raw features. Even the coarsest view still carries seasonality on top of a slow movement, so mixing raw scales would force one cross-scale operation to handle two components with opposite behavior. Instead, inside each Past-Decomposable-Mixing block I decompose every scale first, $s_m, t_m = \mathrm{SeriesDecomp}(x_m)$, with $t_m = \mathrm{MovingAvg}(x_m)$ the trend and $s_m = x_m - t_m$ the seasonal residual; the sign matters, the season is the raw signal minus the trend. Then I mix the two components across scales in opposite directions, and that asymmetry is the heart of the method. Seasonal information flows bottom-up, fine to coarse, because a coarse rhythm is assembled from finer rhythms — a weekly pattern is built from daily patterns, and the detailed phase information lives at the fine scale:
$$s_m \leftarrow s_m + \mathrm{BottomUpMixing}_m(s_{m-1}), \quad m = 1 \dots M.$$
Trend information flows top-down, coarse to fine, because fine-scale detail is exactly the disturbance that makes a macro trend noisy, so the cleaner slow movement at the coarse scale should guide the finer trend estimates:
$$t_m \leftarrow t_m + \mathrm{TopDownMixing}_m(t_{m+1}), \quad m = M-1 \dots 0.$$
Both mixers operate along the temporal dimension and change length — $\mathrm{BottomUpMixing}_m$ maps $\lfloor P/2^{m-1} \rfloor \to \lfloor P/2^m \rfloor$ and $\mathrm{TopDownMixing}_m$ maps $\lfloor P/2^{m+1} \rfloor \to \lfloor P/2^m \rfloor$ — and each is a small $\mathrm{Linear} \to \mathrm{GELU} \to \mathrm{Linear}$, because I want a learned transformation of the neighboring pattern rather than only a fixed resampling. The updates are additive residuals in both directions. After the two passes, each scale carries a mixed seasonal part and a mixed trend part; I recombine them at the same scale, and in the scalable channel-independent path the block returns $\mathrm{ori} + \mathrm{out\_cross\_layer}(s_{\text{mix}} + t_{\text{mix}})$ as a residual channel-FeedForward update. Stacking a small number of these blocks lets the model keep revising the scale interactions without becoming an attention stack.

Prediction is the second place I refuse to collapse information too early. A fine-scale representation and a coarse-scale representation can both predict the same horizon, but from different evidence — the fine one sees short oscillations directly, the coarse one sees slow movement clearly — so merging everything into one head before prediction would discard the very difference the ladder was built to expose. Future-Multipredictor-Mixing therefore gives every scale its own predictor and sums the results:
$$\hat{x}_m = \mathrm{Predictor}_m(x_m^L), \quad \hat{x} = \sum_{m=0}^{M} \hat{x}_m,$$
where each $\mathrm{Predictor}_m$ first maps its temporal length $\lfloor P/2^m \rfloor$ to $F$ with one Linear layer and then projects the hidden features to the output variates. The choice of a sum over an average is deliberate but carries no structural cost: under learned predictors and an MSE loss on the ensemble output, the two are equivalent up to a constant, since an average ensemble mimics a sum by scaling each predictor by $M+1$ and a sum mimics an average by scaling each by $1/(M+1)$. After stacking and summing the per-scale forecasts I denormalize once with the scale-$0$ statistics, because the requested output is in the original resolution and units; this is safe because every statistic comes only from observed input windows and no future target ever enters the input.

For many-variate data I keep the channel-independent path as the scalable default: when channel counts range from a handful to hundreds, a joint embedding can spend capacity on unstable cross-channel coincidences and scales poorly with $C$, so I reshape $[B, L_m, C]$ into $[B \cdot C, L_m, 1]$, share one univariate model across all variates, project $d_{\text{model}} \to 1$, and reshape back to $[B, F, C]$. The channel-joint path is genuinely different and I do not pretend otherwise: it uses a joint $\mathrm{enc\_in} \to d_{\text{model}}$ embedding, applies a $\mathrm{cross\_layer}$ to the decomposed season and trend inside the mixing block, omits the final $\mathrm{ori} + \mathrm{out\_cross\_layer}(\cdot)$ residual, and adds a residual regression path in the output projection. Known decoder-side calendar features, when enabled, are embedded and added before projection — these are known covariates, never target future values. Optional knobs exist (a DFT-based decomposition in place of moving average, max-pool or strided-conv downsampling in place of average pooling), but the method as derived uses moving-average season-trend decomposition and average-pool multiscale observation. The default reported configuration uses two PDM blocks, with $M = 3$ scales for long-term forecasting and $M = 1$ for short-term.

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
