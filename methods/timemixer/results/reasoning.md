Let me start from what actually defeats me on these series. I have a length-`P` window of a `C`-variate series and I want the length-`F` future. The regression is the easy part; the killer is that the past window is a tangle. In one window the signal is climbing, falling, oscillating at a couple of rhythms, and slowly drifting all at once, and these are not on separate channels — they are superimposed in the same scalar stream. So whatever I build has to *disentangle* before it predicts, because predicting a sum of a stationary wiggle and a non-stationary drift with one map is asking a single linear head to be two very different functions at once.

The field already has two ways to disentangle, and I should be honest about exactly where each stops. The first is season-trend decomposition. The intuition is old and solid: a moving average over time isolates the slowly-varying part, call it the trend, and the residual is the fast repeating part, the season; STL formalized this decades ago precisely because the two halves *behave differently* — season is short-term and roughly stationary, trend is long-term and non-stationary. Autoformer made it a differentiable block — `trend = AvgPool1d(x)` with edge padding to keep the length, `season = x − trend` — and DLinear showed you can decompose and then slap one linear layer on each component and beat far heavier Transformers. That tells me decomposition is genuinely load-bearing and that linear/MLP heads are enough. But it also tells me what's missing: every one of these does the decomposition at a *single resolution*. One moving-average kernel, on the series at its native sampling rate. The split it produces is whatever that one resolution happens to reveal.

The second way is multiperiodicity — find the dominant periods (TimesNet does it with an FFT, folds the 1D series into a 2D period-by-cycle tensor and runs 2D convolutions; N-BEATS uses trigonometric bases). This disentangles along the period axis. Also single-axis, and the 2D-conv machinery is heavy.

So both paradigms cut the signal along *one* axis and call it done. Let me sit with whether that's really the only axis, because I keep coming back to one stubborn observation about traffic. The same road, sampled every five minutes, shows a sharp commute spike twice a day. Average that same road to a daily series and the commute spike is *gone* — what's left is weekday-versus-weekend and holiday structure. Average to monthly and even that washes out; you see only a slow seasonal drift. It is one physical process, but it *presents a completely different pattern at each sampling scale.* That isn't a quirk of traffic; it's just what averaging does. An average pool is a low-pass filter — it kills high frequencies and keeps low ones — so as I climb a downsampling ladder the dominant content slides monotonically from microscopic detail to macroscopic structure. Decomposition cuts season from trend; periodicity cuts one period from another; but *nobody is cutting along the scale axis itself.* That's the gap. Fine and coarse views are not redundant copies of one series — they are genuinely different, complementary descriptions of it, and I'm throwing all but one of them away.

And there's a second thing, specific to forecasting and not to representation learning, that sharpens this. The future is jointly determined by variations at several scales: the next hour of traffic depends both on the immediate five-minute momentum and on which day of the week it is. So a predictor built off the fine view and a predictor built off the coarse view don't just see different inputs — they have different *skills*. The fine one will be good at the next few high-frequency wiggles; the coarse one will be good at the slow drift. If that's true, then collapsing everything to one representation before the prediction head — which is what even the multiscale models do — is exactly the wrong move, because it merges the skills before I can use them separately.

Let me check that "even the multiscale models" claim, because Pyraformer and SCINet *do* build multiple resolutions internally — Pyraformer with a pyramidal attention tree, SCINet with a bifurcate downsampling tree that splits even/odd subsequences and recombines them. So is the gap already closed? No — and the distinction is precisely where my second observation bites. They build the pyramid to *extract* a richer single representation, then merge it and predict once from the merged thing. The forecast never gets to draw on the scales *simultaneously and separately*. So the multiscale-extraction idea is on the table, but multiscale *prediction* — keeping a separate predictor per scale and combining their forecasts — is not. That's the opening.

So now I have a shape for the thing: build the series at several scales, learn across scales while keeping them separate, and then predict from each scale and combine. A vague "use multiscale" is not enough; every length change and every direction has to be nailed down.

How do I make the scales? I want a ladder `x_0, x_1, ..., x_M` where `x_0` is the raw input (finest) and each step is coarser. The cheapest honest coarsening of a time series is averaging consecutive chunks — which is exactly the low-pass-then-decimate I already invoked. So `x_{m+1}` = average-pool `x_m` with some window. Pick the window = 2: halve the length (and the temporal resolution) at each step, so the scales are a clean geometric ladder, `x_m` has length `floor(P / 2^m)`. Could I use a strided 1D convolution instead of a plain average? A learned conv has more capacity, but average pooling is parameter-free, a single op, and crucially the neutral coarsening — it doesn't bake in a learned filter that could distort what "coarse" even means before the rest of the model sees it. For a downsampler whose whole job is to be a faithful change of resolution, I'll take the simple, free one. Max-pool is the other option but max is the wrong summary for a level/trend signal — it tracks peaks, not the central tendency I want a coarse view to show. Average it is.

Each scale then needs to become deep features so the mixing layers have channels to work with — embed `x_m` to `d_model` channels. Nothing subtle there yet; it's the standard projection. (On a many-variate dataset I'll come back to *how* to embed, because channel-independence will matter; for now, embed each scale.)

Now the core question: how do I let the scales talk to each other? The naive thing is to just concatenate or add the multiscale features and mix the whole stack. Let me think about whether that can possibly be right, and I don't think it is, for the same reason single-scale decomposition wasn't enough — within each scale the signal is *still* a tangle of season and trend. Look at the coarsest scale: even after heavy averaging it isn't pure trend; a daily-averaged traffic series still has a clear weekly seasonality on top of its drift. So if I mix raw multiscale series as wholes, I'm mixing entangled-with-entangled and the cross-scale interaction has to fight through the same superposition that motivated decomposition in the first place. The fix is forced: decompose *first, at every scale*, and mix the clean components. Inside each block, for every scale `m`, run the moving-average decomposition: `s_m, t_m = SeriesDecomp(x_m)`, where `t_m` is the moving average and `s_m` the residual. Now I have `M+1` seasonal parts and `M+1` trend parts.

The tempting move is to mix seasons across scales and trends across scales with the same machinery. But season and trend are *opposite kinds of object* — STL told me so, and it changes which direction information should flow. Take the seasonal parts. A coarse-scale season is, by the Box–Jenkins observation, an *aggregation* of finer seasons: the weekly cycle is literally seven daily cycles stacked. So the detailed structure that defines a coarse seasonal pattern actually lives down in the fine scale. That means for seasonality I want to push information *upward*, fine → coarse, supplementing each coarser season with the detail from below. Bottom-up. Concretely, walk `m` from `1` up to `M` and do, in a residual way,

  `s_m = s_m + BottomUpMixing(s_{m-1})`,

where `BottomUpMixing` maps a length-`floor(P/2^{m-1})` sequence down to length `floor(P/2^m)` — it has to change the temporal length to match the coarser scale — and I'll make it two linear layers with a GELU in between, acting along the temporal dimension. Why an MLP and not just a single linear resample? Because MLP-Mixer already showed me that "mixing" — a learned linear map along an axis wrapped in a nonlinearity — is the right cheap primitive for integrating information along that axis, and a single linear map can only do a fixed linear resampling whereas I want the finer season to be *transformed* into a useful supplement, not just rescaled. Two layers with a nonlinearity, the smallest thing that's more than linear. The residual `s_m = s_m + (...)` keeps each scale's own season and adds the bottom-up contribution rather than replacing it.

Now the trends, and the asymmetry is the whole point. For trend, fine-scale detail is the *enemy*: the wiggles I want to keep for season are exactly the noise I want to suppress when estimating a slow macroscopic drift. The clean read on the trend comes from the coarse, heavily-averaged scale, not the jittery fine one. So trend information should flow the *other* way — coarse → fine, top-down, using the clean macro picture from above to guide the trend at finer scales. Walk `m` from `M−1` down to `0`:

  `t_m = t_m + TopDownMixing(t_{m+1})`,

with `TopDownMixing` mapping length `floor(P/2^{m+1})` *up* to length `floor(P/2^m)` (now expanding, since I'm going to a finer, longer scale), again two linear layers with a GELU. Same residual logic.

I want to double-check I haven't just told myself a pretty story, so let me imagine the opposite assignment: mix season top-down and trend bottom-up. Top-down on season would pour the coarse, detail-poor seasonal view down onto the fine scale — but the fine scale's value *was* its detail, so I'd be smearing it out, destroying the microscopic seasonal information I most wanted. Bottom-up on trend would pour fine-scale wiggles up into the coarse trend — injecting exactly the high-frequency noise that the coarse trend's cleanliness depended on. Both directions, reversed, attack the thing that made each scale useful. So the assignment isn't aesthetic symmetry; each direction is the one that *adds* the kind of information that scale is rich in and the target scale is poor in.

After seasonal mixing and trend mixing I recombine: for each scale the mixed season and mixed trend add back together, `out_m = s_m + t_m`, and then I want a little cross-channel interaction before closing the block — the mixing so far was all along the temporal axis, so a two-layer GELU MLP across the `d_model` channels (a FeedForward, width `d_ff`) lets the channels talk. Wrap the whole block in a residual on the original scale features: `x_m^l = x_m^{l-1} + FeedForward(s_m + t_m)`. That's one Past-Decomposable-Mixing block. I keep the stack small, with `L = 2` in the usual configuration; each block re-decomposes the current features and mixes again, so repeated passes give deeper cross-scale interaction without ballooning cost.

Let me get the bookkeeping of the seasonal-mixing loop exactly right, because the lengths change at every step and an off-by-one here silently corrupts everything. I hold the seasonal parts as a list ordered finest-to-coarsest, each permuted to `[B, d_model, L_m]` so a `Linear` hits the temporal length `L_m`. I keep two cursors. `out_high` starts as the finest season `season_list[0]`; `out_low` starts as the next one `season_list[1]`. The output list starts with the finest season unchanged (permuted back to `[B, L_0, d_model]`). Then I loop `i` over the scales: I map the current `out_high` down a level, `out_low_res = down_layers[i](out_high)`, add it into `out_low` (`out_low = out_low + out_low_res`), and that summed thing becomes the new `out_high` for the next rung. If there's still a coarser raw scale waiting (`i+2 <= len-1`), I reset `out_low` to that next raw season `season_list[i+2]` so the next rung mixes into the *original* coarser season, not into an already-doubly-mixed buffer. Append the new `out_high`. So information genuinely climbs the ladder one rung at a time, each coarser season receiving the cumulatively-mixed finer detail. The trend loop is the mirror image: reverse the list so it's coarsest-first, build the up-sampling layers with `reversed(range(...))` so their dimensions line up with the reversed traversal, run the identical two-cursor pattern climbing coarse→fine, then reverse the output list back to finest-first. The two loops are structurally identical; only the direction and the linear-layer shapes flip. Good — that's the genuine asymmetry, implemented once each way.

Now the prediction, and this is where I refuse to collapse the scales. I have `L` blocks' worth of mixed features at every scale, `x_m^L`, each still at its own temporal length `floor(P/2^m)`. The conventional move is to take the finest (or a merged) representation and run one head. But my whole second observation was that *different scales have different forecasting skills*, so I want a *predictor per scale* and then combine the forecasts, not the features. For each scale: one linear layer maps the temporal length `floor(P/2^m)` straight to the future length `F` (a single linear past→future regression, DLinear-style, which I already know is shockingly strong), then a projection from `d_model` channels back to the `C` output variates. That gives `xhat_m`, a full `[B, F, C]` forecast *from that scale alone*. The final forecast is the sum over scales,

  `xhat = sum_{m=0}^{M} xhat_m`.

This is an ensemble where the members are deliberately heterogeneous — each sees the world at a different resolution — so they contribute complementary forecasting capabilities: the fine predictors carry the seasonal detail, the coarse predictors carry the macro trend. Why sum rather than average? Under an MSE loss computed on the *ensemble* output, sum and average differ only by the constant `1/(M+1)`, and a network trivially absorbs a constant by scaling each predictor's weights — so the two are equivalent in what they can learn; I'll just use the plain sum. And why a single predictor at all instead of dropping all but the finest scale? Because dropping the multipredictor and forecasting only from the finest scale throws away exactly the coarse-scale forecasting skill I argued is complementary; that should measurably hurt, especially on longer horizons where the macro trend dominates.

There's a normalization issue I've been deferring that I now have to nail, because these series are non-stationary — the level and scale drift between the training window and the test window, and that distribution shift wrecks a naively-trained forecaster. So I standardize each input window: subtract its own mean, divide by its own standard deviation, and remember those statistics to invert at the end (a reversible per-series instance norm). I do this *per scale* — each scale gets its own norm — because averaging changes the variance, so the coarse views have a different scale than the fine ones and each deserves its own standardization. An affine `(weight, bias)` after standardizing lets the model re-introduce a learned per-series level if that helps. At the very end I de-normalize the summed forecast with the scale-0 (finest, original-resolution) statistics, since the prediction is in the original series' units.

One more thing the many-variate setting forces, and it interacts with the embedding choice I postponed. Across these datasets `C` ranges from a handful (7 on ETTh1) to hundreds (321 on Electricity) of channels with very different per-channel magnitudes and only loose cross-channel coupling. If I embed all `C` variates jointly into a shared representation, a model can overfit spurious cross-channel couplings and let a few high-variance channels dominate — and the embedding dimension would have to grow with `C`, which doesn't scale to hundreds of variates. The robust move on many-variate data is *channel independence*: treat each variate as its own univariate series sharing one set of weights. Concretely, after per-scale normalization I reshape `[B, L_m, C]` into `[B·C, L_m, 1]` so every channel becomes an independent length-`L_m` sequence in the batch, embed the single channel up to `d_model` (a token embedding lifting `1 → d_model`, no positional encoding since the temporal-mixing linear layers already see absolute position through their weights), run the entire PDM stack and per-scale predictors on these `B·C` independent sequences, and at the very end project `d_model → 1` and reshape back to `[B, F, C]`. Every channel is forecast by the same shared multiscale machinery, which both regularizes and scales gracefully to hundreds of variates. The whole architecture is now pure MLP with no attention and no recurrence, so it stays cheap on long inputs and wide variate counts — which was a hard requirement, not a bonus.

Let me also settle the decomposition kernel and confirm I'm not over-engineering. The trend is a moving average; I keep the stride at 1 and replicate edge values so the decomposed components keep the input length. I could split in the Fourier domain by taking top-`k` frequencies as season and the residual as trend, but that turns a simple local smoothing primitive into a frequency-selection step. For a component meant to be fast, differentiable, and easy to place inside every block and every scale, the moving average is the right building block. A small kernel suffices because the per-scale series, especially the coarse ones, are short.

So now I land it as the actual module, filling the one open slot in the harness — the `Model` whose `forecast` turns the `[B, seq_len, enc_in]` past into the `[B, pred_len, c_out]` future. Each piece below is exactly one step of the reasoning above:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class series_decomp(nn.Module):
    # trend = moving average (low-pass), season = residual; the STL-style split,
    # length preserved by edge replication so each component matches the input.
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):                                  # x: [B, T, C]
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        xp    = torch.cat([front, x, end], dim=1)
        trend = self.avg(xp.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend                           # season, trend


class Normalize(nn.Module):
    # Per-scale reversible instance norm: standardize a window in, invert it out,
    # to absorb the per-series level/scale drift that non-stationarity causes.
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps, self.affine = eps, affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        else:
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean


class MultiScaleSeasonMixing(nn.Module):
    # Bottom-up (fine -> coarse): coarse seasonality is an aggregation of finer
    # seasonality, so supplement each coarser season with detail from below.
    # Each layer is the cheap mixing primitive: length-changing Linear, GELU,
    # same-length Linear -- more than a fixed linear resample.
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
        out_high = season_list[0]                          # finest
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_layers[i](out_high)    # push detail one rung up
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]               # next rung mixes into the raw coarser season
            out_season_list.append(out_high.permute(0, 2, 1))
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    # Top-down (coarse -> fine): fine detail is noise for macro trend, so guide
    # each finer trend with the cleaner trend from above.
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
        trend_rev = trend_list.copy()
        trend_rev.reverse()                                # coarsest first
        out_low = trend_rev[0]                             # coarsest
        out_high = trend_rev[1]
        out_trend_list = [out_low.permute(0, 2, 1)]
        for i in range(len(trend_rev) - 1):
            out_high_res = self.up_layers[i](out_low)      # push macro one rung down
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))
        out_trend_list.reverse()                           # back to finest first
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    # one PDM block: decompose every scale, mix seasons bottom-up and trends
    # top-down, recombine, cross-channel FFN, residual on the original features.
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
            season, trend = self.decomp(x)
            season_list.append(season.permute(0, 2, 1))    # [B, D, L] so Linear hits time
            trend_list.append(trend.permute(0, 2, 1))
        season_list = self.season_mixing(season_list)
        trend_list = self.trend_mixing(trend_list)
        out_list = []
        for x, season, trend in zip(x_list, season_list, trend_list):
            out = season + trend                           # recombine the mixed components
            out = x + self.out_cross_layer(out)            # cross-channel FFN + residual
            out_list.append(out)
        return out_list


class Model(nn.Module):
    # Multiscale ladder -> stacked PDM -> per-scale predictors summed (FMM).
    # Channel-independent: every variate is its own univariate sequence, shared weights.
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_layers = configs.down_sampling_layers
        self.down_window = configs.down_sampling_window
        D = configs.d_model

        self.down_pool = nn.AvgPool1d(configs.down_sampling_window)  # parameter-free coarsening

        self.norm_layers = nn.ModuleList([                          # one RevIN per scale
            Normalize(configs.enc_in, affine=True)
            for _ in range(self.down_layers + 1)
        ])

        self.enc_embedding = DataEmbedding_wo_pos(                  # channel-indep: 1 -> D, no pos enc
            1, D, configs.embed, configs.freq, configs.dropout)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(configs.e_layers)
        ])

        self.predict_layers = nn.ModuleList([                       # per-scale past->future regress
            nn.Linear(self.seq_len // (self.down_window ** i), self.pred_len)
            for i in range(self.down_layers + 1)
        ])
        self.projection_layer = nn.Linear(D, 1)                     # channel-indep: D -> 1
        self.c_out = configs.c_out

    def __multi_scale_inputs(self, x_enc):
        x = x_enc.permute(0, 2, 1)                                  # [B, C, T]
        x_list = [x.permute(0, 2, 1)]                              # finest
        for _ in range(self.down_layers):
            x = self.down_pool(x)
            x_list.append(x.permute(0, 2, 1))
        return x_list                                              # finest -> coarsest

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.size(0)
        x_list = self.__multi_scale_inputs(x_enc)                  # each [B, L_i, C]

        # per-scale normalization, then channel independence
        ci_list = []
        for i, x in enumerate(x_list):
            x = self.norm_layers[i](x, "norm")
            _, Li, N = x.shape
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, Li, 1)  # [B*N, L_i, 1]
            ci_list.append(x)

        # embed each scale's single channel up to D
        enc_list = [self.enc_embedding(x, None) for x in ci_list]  # each [B*N, L_i, D]

        # stacked Past-Decomposable-Mixing
        for block in self.pdm_blocks:
            enc_list = block(enc_list)

        # Future-Multipredictor-Mixing: predict from every scale, then sum
        dec_list = []
        for i, enc in enumerate(enc_list):
            dec = self.predict_layers[i](enc.permute(0, 2, 1)).permute(0, 2, 1)  # [B*N, F, D]
            dec = self.projection_layer(dec)                                     # [B*N, F, 1]
            dec = dec.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1)     # [B, F, C]
            dec_list.append(dec)
        dec_out = torch.stack(dec_list, dim=-1).sum(-1)
        dec_out = self.norm_layers[0](dec_out, "denorm")           # invert with scale-0 stats
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]
```

The causal chain, start to finish: the past window is a superposition of stationary season and non-stationary trend, so a single head can't fit it — decomposition is needed. But decomposition at one resolution is half the story, because averaging is a low-pass filter and the same series shows microscopic detail at fine scales and macroscopic structure at coarse scales, so I build a downsampling ladder to expose all of them; average pooling because a coarsener should be neutral and free. Within each scale the signal is still entangled, so I decompose at every scale before mixing. Season and trend are opposite objects, which dictates opposite information flow: coarse seasonality is an aggregation of fine seasonality, so seasons mix bottom-up; fine detail is noise for macro trend, so trends mix top-down — and reversing either direction would destroy the very information that made that scale valuable. The mixing primitive is a two-layer GELU MLP along time because a single linear can only resample, while MLP-style mixing can transform. Recombine, add a cross-channel FFN, residual, and stack two such blocks. For prediction I keep the scales separate to the very end — one linear past→future predictor per scale, summed — because the scales have complementary forecasting skills and merging them first would waste that; sum and average are equivalent under the ensemble loss. Per-scale reversible normalization handles the non-stationary drift, and channel independence with shared weights makes the whole thing robust and cheap across hundreds of variates, with no attention or recurrence anywhere.
