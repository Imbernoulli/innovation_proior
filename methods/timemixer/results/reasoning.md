I start from the part of the forecasting problem that keeps defeating a single clean model: the past window is not one behavior. It is a slow level, a trend, one or more seasonal rhythms, and local noise all added into the same observed stream. If I ask one representation at the native sampling rate to explain all of that, it has to treat microscopic wiggles and macroscopic drift as though they live at the same resolution. That seems wrong before I even choose a neural block.

The first familiar tool is decomposition. A moving average gives me a slow component, and the residual `x - moving_average(x)` gives me the faster component. The sign matters: the season is the raw signal minus the trend, not the other way around. This split is useful because the two pieces ask for different treatment. The residual carries short-term repeating detail; the trend carries slower, less stationary movement. But doing this only once on the original sampling rate still leaves me with a single view of the series.

The second familiar tool is multiscale observation. If I average-pool the series and decimate it, I do not get a harmless duplicate. I get a lower-pass view. Fine scales preserve high-frequency detail, while coarse scales emphasize slower structure. So I want a ladder `x_0, x_1, ..., x_M`, with `x_0` the original series and each `x_m` having length about `floor(P / 2^m)` when the window is 2. Average pooling is the right default coarsener because it is parameter-free and expresses the resolution change directly. A learned convolution is possible, but it is not the cleanest starting point for the method.

Now I have a dilemma. I could mix all scales as raw features, but each scale is still a mixture of season and trend. Even the coarsest view can have seasonality on top of a slow movement. If I mix raw scales directly, I force one cross-scale operation to handle two components with opposite behavior. So I decompose every scale first. For each scale `m`, I compute `s_m, t_m = SeriesDecomp(x_m)`, where `t_m` is the moving average and `s_m = x_m - t_m`.

The seasonal components should not flow in the same direction as the trend components. For seasonality, a coarse rhythm is built from finer rhythms: a weekly pattern is made from daily patterns, and the detailed phase information lives at the fine scale. So I push seasonal information upward from fine to coarse. For `m = 1..M`, I update

`s_m <- s_m + BottomUpMixing_m(s_{m-1})`.

The mixer has to change length from `floor(P / 2^{m-1})` to `floor(P / 2^m)`, so it acts along the temporal dimension. I make it a small MLP, `Linear -> GELU -> Linear`, because I want a learned transformation of the finer seasonal pattern, not only a fixed resampling.

For trend, the direction reverses. Fine-scale detail is exactly the disturbance that makes a macro trend noisy. The coarse scale has the cleaner slow movement, so it should guide the finer trend estimates. For `m = M-1..0`, I update

`t_m <- t_m + TopDownMixing_m(t_{m+1})`.

This mixer changes length from `floor(P / 2^{m+1})` back to `floor(P / 2^m)`. The signs are additive in both cases. The asymmetry is not cosmetic: bottom-up season preserves and aggregates detail, while top-down trend injects the clean macro component into finer scales.

After those two passes, each scale has a mixed seasonal part and a mixed trend part. I add them back together at the same scale. The clean block is a residual update: take the old representation, add a channel FeedForward of the recombined mixed components, and pass the result to the next block. Repeating this for a small number of layers lets the model keep revising the scale interactions without turning into an attention stack.

I also need to keep the two operating modes separate. In the channel-independent path, which is the scalable path, the block returns `ori + out_cross_layer(season_mix + trend_mix)`. In the channel-joint path, the design first applies a `cross_layer` to the decomposed season and trend, then returns the mixed sum cropped to the original length; that path does not apply the final `ori + out_cross_layer(...)` residual. I cannot pretend those two branches are identical.

Prediction is the next place where I have to resist collapsing information too early. If I merge all scales into one representation and use one head, I lose the very difference I built the scale ladder to expose. A fine-scale representation and a coarse-scale representation can both predict the same future length, but they do so from different evidence. The fine one sees short oscillations more directly; the coarse one sees slower movement more clearly. So I give every scale its own predictor:

`xhat_m = Predictor_m(x_m^L)`,

where `Predictor_m` first maps the temporal length `floor(P / 2^m)` to `F` with one Linear layer, then projects the hidden dimension back to the output variates. The final forecast is

`xhat = sum_{m=0}^M xhat_m`.

The sum is deliberate and simple. If I used an average instead, the function class would be the same up to a constant: an average ensemble can mimic a sum ensemble by scaling each predictor by `M + 1`, and a sum ensemble can mimic an average ensemble by scaling each predictor by `1 / (M + 1)`. Under a learned predictor and an MSE loss on the ensemble output, that constant is not a structural difference.

Normalization has to be per scale. Pooling changes variance, and real forecasting windows drift in level and spread. So I normalize each scale before embedding, store the scale's statistics, and after summing all scale forecasts I denormalize with the original finest-scale statistics, because the requested output is in the original resolution and original units. This is safe as long as the statistics come only from observed input windows. They do; no target future value is used.

For many-variate data, I also want the channel-independent path. Some datasets have seven channels, others hundreds. If I force a joint embedding across all channels, the model can spend capacity on unstable cross-channel coincidences and scale poorly with `C`. The channel-independent path reshapes `[B, L_m, C]` into `[B*C, L_m, 1]`, shares one univariate model over all variates, and projects `d_model -> 1` before reshaping back to `[B, F, C]`. This keeps the temporal architecture shared and regularized. If known future calendar features are supplied, they can be embedded and added before projection, but those are known covariates, not future target values.

There are optional branches I should not confuse with the central construction. The decomposition method can switch to a DFT-based split, and downsampling can switch to max pooling or a strided convolution. Those branches are useful knobs, but the method I am deriving uses moving-average season-trend decomposition and average-pool multiscale observations. Likewise, the channel-joint path carries an extra residual regression path in the future projection, while the channel-independent path uses the cleaner per-scale predictor and `d_model -> 1` projection.

The resulting architecture has the shape I wanted from the beginning. It builds a scale ladder, decomposes each scale before mixing, sends seasonal information fine-to-coarse, sends trend information coarse-to-fine, keeps each scale alive until prediction time, and sums separate per-scale forecasts. It uses only average pooling, moving average, Linear layers, GELU, reversible normalization, and optional known time-feature embeddings. No future target leaks into the input, and the final output is a direct `[B, F, C]` forecast.
