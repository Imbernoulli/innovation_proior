The patch-Transformer did exactly what I predicted and no more, and the "no more" is the tell. It beat
the linear floor on every regime — Monthly 12.97 against DLinear's 13.39, Quarterly 10.22 against
10.50, Yearly 13.68 against 14.36 — and the largest absolute gain landed on Yearly (0.68 SMAPE), the
regime where the affine map was most strained, just as the reversible instance normalization and the
learned nonlinear representation were supposed to help most. So the two things DLinear structurally
lacked — per-window level decoupling and a nonlinear shape model — were real, and adding them moved
every number in the right direction. But the *size* of the move is the diagnostic. Each gain is a few
tenths of a SMAPE point, not a clear separation, and that is the falsifiable case I flagged: a
512-wide, 2-layer attention encoder over a handful of patch tokens (Monthly has four patches,
Quarterly two, Yearly's window is shorter than one patch) is near its ceiling. There simply are not
enough patch tokens on a `2·pred_len` window for self-attention to find structure that a normalized
linear map missed; most of the encoder's width is stranded capacity on two-to-four tokens. The lesson
is concrete: stop adding generic capacity over the single-scale window, because the window is too short
to feed it. The information PatchTST is leaving on the table is not "more attention" — it is the
*multi-scale* structure of the series, which a single fixed-resolution view, patched or not, never
exposes.

Here is the structural argument for why scale is the missing axis. The same underlying process presents
*different* patterns at different sampling resolutions. At the finest resolution I see microscopic
detail — the exact wiggle of the seasonal shape, short bursts. Coarsen the series by averaging adjacent
steps and that detail washes out, leaving the macroscopic structure — the slow trend, the broad cyclic
envelope — much cleaner than it ever appears in the noisy fine view. A short M4 window holds both kinds
of information superimposed, and the future is jointly determined by both: the next six Yearly steps
are a trend question (a coarse-scale fact), while the next eighteen Monthly steps are a fine seasonal-
shape question (a fine-scale fact). A model that reads the series at one resolution — the linear floor,
or PatchTST's single patch grid — has to disentangle trend from season within that one view, and on a
twelve-or-thirty-six-step window with two encoder layers it cannot do that well. So the move is to
build an explicit *ladder of scales* and let the model mix information across resolutions, rather than
forcing one resolution to carry everything. That is a different bet than PatchTST's: not "give attention
better tokens," but "give a simple mixer several resolutions and let trend and season flow between
them."

Build the ladder by average pooling. Take the input window and downsample it by a window of 2 a small
number of times, giving scales `x_0` (finest, the original), `x_1` (half-length), and so on. Average
pooling, not a learned conv or max-pool, because the coarse view should show the *central tendency* of
each neighborhood — a learned conv is more expressive but less neutral and can distort what
"coarsening" means, and max-pool tracks peaks rather than the average a macroscopic view wants.
Parameter-free coarsening keeps the ladder honest. On M4's short windows I do not need a deep ladder —
one downsampling step (window 2) is enough to expose a coarse view alongside the fine one; for Monthly
that is a 36-step scale and an 18-step scale, for Yearly a 12-step and a 6-step. This is the first place
the harness protocol bites, and I have to handle it deliberately: the fixed Custom scripts do **not**
pass `--down_sampling_layers`, `--down_sampling_window`, or `--down_sampling_method`, so under the
harness those args default to `0`, `1`, and `None` — which would collapse the model to a *single
scale*, throwing away the entire multi-scale thesis and leaving me with a fancy single-resolution MLP
no better than PatchTST. So I must set the downsampling configuration *inside* `Custom.py` (one
downsampling layer, window 2, average pooling), reading from `configs` only as an override. The whole
contribution of this rung exists only if I do that explicitly; this is exactly the kind of paper-vs-
harness gap where the same-named model would silently degenerate if I trusted the command line.

Now the core: how to mix across scales. Inside each block, first *decompose* every scale into season
and trend with a moving-average block (`trend = AvgPool`, `season = x − trend`), because even a coarse
series is still a season+trend mixture — mixing entangled wholes would fight the very superposition the
decomposition was meant to remove. Then mix the two components across scales in *opposite directions*,
and this asymmetry is the heart of the method, so let me reason it out rather than assert it. Seasonal
detail aggregates upward: coarse seasonality is literally an aggregation of finer seasonality (average
a fine oscillation and you get the envelope of the coarse one), so the fine scale has information the
coarse scale should receive — season mixes **bottom-up**, fine → coarse. Trend goes the other way: for
the macroscopic trend, fine-scale detail is just noise, while the *clean* trend is most visible at the
coarse scale, so the coarse scale has the information the fine scale should receive — trend mixes
**top-down**, coarse → fine. Reversing either direction destroys the information that made that scale
valuable: pushing noisy fine detail down into the trend, or coarse-smoothed season up where the detail
was the point. The mixing primitive between adjacent scales is a two-layer MLP with a GELU between
(resampling the temporal length from one scale to the next), added as a residual — a single linear
could only resample, while an MLP can actually transform the donor scale into a useful supplement.
After mixing, recombine season and trend and apply a channel feed-forward with a residual. Stack a few
such blocks (`e_layers`, which the harness fixes at 2).

The decoder is the second idea, and it is the one that most directly answers PatchTST's stranded-
capacity problem. Instead of merging all scales into one representation and predicting once — which
wastes the complementary forecasting skills different scales have — give **each scale its own
predictor** (one linear map from that scale's length to the horizon) and **sum** the per-scale
forecasts. This is Future-Multipredictor-Mixing: the coarse scale is good at the trend part of the
forecast, the fine scale at the seasonal part, and summing lets each contribute what it is good at,
rather than forcing one merged feature through a single head. Summing versus averaging differs only by
a constant the network absorbs, so the choice is immaterial; what matters is that the predictions stay
*separate until the end*. Wrap the whole thing in reversible per-scale instance normalization — the
same level-decoupling that helped PatchTST, now applied per scale — and denormalize the summed forecast
with the finest-scale statistics. And run channel-independently, exactly as the channel argument from
the previous rung established: on M4 each series is its own univariate sequence (`enc_in = 1`), embedded
`1 → d_model` and projected `d_model → 1` with shared weights, which is also the harness default
(`channel_independence = 1`) so I get it for free.

I should be honest about where the protocol still strains this rung, because PatchTST taught me to
watch exactly that. The harness fixes `d_model = 512`, `d_ff = 512`, `e_layers = 2`, whereas
TimeMixer's own M4 script uses a *narrow* model (`d_model = 16`, `e_layers = 4`) — its whole design
premise is that an MLP mixer needs width far less than an attention model. So I am again running this
architecture wider and shallower than it wants. But here I expect the multi-scale structure to make the
width *less* wasteful than it was for PatchTST: the per-scale predictors and the cross-scale mixing give
the 512-wide channels something real to do across two resolutions and a decomposed season/trend
representation, rather than attending over two patch tokens. The reversible normalization and channel
sharing remain the regularizers, and 10-epoch early stopping caps overfitting. The genuine risk is the
opposite of PatchTST's: with only one downsampling step on a short window, the coarse scale is very
short (6 steps for Yearly), so the ladder is shallow and the multi-scale benefit may be modest on the
shortest regime even if it is clear on the longer ones.

So the falsifiable expectations against PatchTST's measured numbers are specific and directional. I
expect to beat PatchTST on **every** regime, because explicit multi-scale decomposition-mixing is
strictly more structure than a single-resolution patch encoder can express on these short windows, and
because the per-scale predictors put the harness's stranded width to use. I expect the **largest** gain
on the regimes where trend and season genuinely live at different scales and the window is long enough
to support a real coarse view — Monthly (clear 12-step seasonality plus drift, 36-step window) and
Yearly (the trend regime, where pushing the clean coarse trend top-down into the forecast is exactly
the multi-scale move). Concretely I expect to clear PatchTST's 12.97 (Monthly), 10.22 (Quarterly), and
13.68 (Yearly), with the Yearly improvement the one I am most confident about, since trend-dominated
forecasting is precisely what top-down trend mixing is built for. If instead the gain is flat on the
shortest regime (Yearly's 6-step coarse scale too thin to help), that is the signal that one
downsampling step is not enough scale separation on M4's tiniest windows — and that the next move is a
model that finds the *period* structure directly inside the window rather than relying on a fixed
pooling ladder to expose it.
