The patch-Transformer did exactly what I predicted and no more, and the "no more" is the tell. It beat
the linear floor on every regime — Monthly 12.97 against DLinear's 13.39, Quarterly 10.22 against
10.50, Yearly 13.68 against 14.36 — and the largest gain landed on Yearly, both absolutely (0.679) and
relatively (4.7% against Monthly's 3.2% and Quarterly's 2.7%), just where the reversible instance
normalization and the FFN nonlinearity were supposed to pay off most. The two things DLinear
structurally lacked were real, and adding them moved every number in the right direction. But the
*size* of the move is the second half of the diagnosis: every gain is a few tenths, not a clear
separation. That is the falsifiable case I flagged — a 512-wide, 2-layer encoder over the handful of
patch tokens those short windows yield (four Monthly, two Quarterly, one Yearly) is near its ceiling.
On two of three regimes attention had essentially nothing to relate, so the Yearly gain is carried by
RevIN and the FFN, not by attention finding cross-patch structure; most of the encoder's width is
stranded on one-to-four tokens. The lesson is concrete: stop adding generic capacity over the
single-scale window, because the window is too short to feed it. The information PatchTST leaves on the
table is not "more attention" — it is the *multi-scale* structure of the series, which a single
fixed-resolution view, patched or not, never exposes.

Here is why scale is the missing axis. The same underlying process presents different patterns at
different sampling resolutions. At the finest resolution I see microscopic detail — the exact wiggle of
the seasonal shape, short bursts. Coarsen by averaging adjacent steps and that detail washes out,
leaving the macroscopic structure — the slow trend, the broad cyclic envelope — much cleaner than it
ever appears in the noisy fine view. A short M4 window holds both kinds superimposed, and the future is
jointly determined by both: the next six Yearly steps are a trend question (coarse-scale), the next
eighteen Monthly steps a fine seasonal-shape question. A model reading at one resolution has to
disentangle trend from season within that one view, and on a twelve-to-thirty-six-step window with two
layers it cannot do that well. So build an explicit *ladder of scales* and let the model mix
information across resolutions rather than forcing one resolution to carry everything — a different bet
than PatchTST's, and one that gives the 512 channels real work across scales and a decomposed
season/trend representation rather than attending over two tokens.

Build the ladder by average pooling, and the depth matters first, because the windows are tiny and a
ladder can bottom out into noise. Downsample by a window of 2: one step gives Monthly 36/18, Quarterly
16/8, Yearly 12/6. A second step would give Yearly 12/6/**3** — and a three-step coarse view is noise,
not a macroscopic trend; its per-scale predictor would be a `Linear(3 → 6)` fitting structure that is
not there. A deep ladder actively hurts the shortest regime, the one I most need to help. So one
downsampling step is the right depth on M4: a coarse view alongside the fine one, without any scale
collapsing to noise. I use *average* pooling deliberately over the alternatives — a learned strided
conv would distort what "coarsening" means by learning a filter, and max-pool tracks peaks rather than
central tendency, which is wrong for a macroscopic view. Parameter-free average pooling keeps the
coarse view an honest central-tendency summary, which matters because the whole method rests on it
genuinely being "the trend, cleaner."

This is the first place the harness protocol bites. The fixed Custom scripts do **not** pass
`--down_sampling_layers`, `--down_sampling_window`, or `--down_sampling_method`, so under the harness
those default to `0`, `1`, `None` — which collapses the model to a single scale (zero downsampling
layers, pool window 1): the multi-scale input list has one entry, the mixing loops over zero adjacent
pairs, and the decoder has one predictor. That is a fancy single-resolution MLP no better than
PatchTST, and the degeneration is not a crash but a quiet collapse to the previous ceiling, the worst
kind of failure to miss. So I set the downsampling configuration *inside* `Custom.py`
(`self.down_layers = getattr(configs, 'down_sampling_layers', 0) or 1` forces at least one layer;
`if self.down_window <= 1: self.down_window = 2` forces a real pool window), reading `configs` only as
an override. The whole contribution here exists only if I do that explicitly.

Now the core: how to mix across scales. Inside each block, first *decompose* every scale into season
and trend with a moving-average block, because even a coarse series is still a season+trend mixture. To
see why decomposing first is necessary, consider the shortcut of mixing the raw scales directly.
Bottom-up I would push the *entire* fine scale — its clean trend and its noisy seasonal detail together
— up into the coarse scale, contaminating the coarse scale's clean trend with fine-scale noise, the
very thing that made the coarse view valuable. Top-down I would push the whole coarse scale — trend and
its smoothed-away season together — down, smearing the fine seasonal detail with a coarse blur. Mixing
entangled wholes fights the superposition I am trying to exploit: each direction carries the wrong
component along with the right one. Decomposing each scale first is what lets bottom-up carry *only*
season and top-down carry *only* trend.

Then mix the two components across scales in *opposite* directions, and this asymmetry is the heart of
the method. Seasonal detail aggregates upward: coarse seasonality is literally an aggregation of finer
seasonality (average a fine oscillation and you get the envelope of the coarse one), so the fine scale
has information the coarse scale should receive — season mixes **bottom-up**, fine → coarse. Trend goes
the other way: for the macroscopic trend, fine-scale detail is just noise while the clean trend is most
visible at the coarse scale, so the coarse scale has what the fine scale should receive — trend mixes
**top-down**, coarse → fine. Reversing either direction destroys the information that made that scale
valuable. The mixing primitive is a two-layer MLP with a GELU rather than a single linear — a single
linear could only resample the length, while an MLP can actually transform the donor scale into a
useful supplement — added as a residual so the recipient scale is never overwritten. After mixing,
recombine season and trend and apply a channel feed-forward `d_model → d_ff → d_model` with a residual.
Stack `e_layers` such blocks (the harness fixes 2).

I should price this block, because the whole complaint about PatchTST was stranded width. The channel
feed-forward is `2·512·512 ≈ 524k` weights per block, so two blocks is about a million parameters, and
that is where almost all the capacity sits — the season and trend mixing MLPs (`36→18→18`, `18→36→36`)
are a few thousand weights each, the embedding a thousand, the predictors a few hundred. Roughly a
third of PatchTST's ~3M, and — the point — the width is spent on a *channel* FFN that does genuine
per-timestep nonlinear work on every scale, not attention idling over two tokens. That is the concrete
sense in which the mixer uses the harness's width where the patch encoder could not: the capacity feeds
a pointwise MLP that always has something to do, independent of how many tokens the window yields.

The decoder is the second idea, and the one that most directly answers the stranded-capacity problem.
Instead of merging all scales into one representation and predicting once, give **each scale its own
predictor** (one linear map from that scale's length to the horizon) and **sum** the per-scale
forecasts: `Linear(36 → 18)` and `Linear(18 → 18)` on Monthly, and analogously on the others. This is
Future-Multipredictor-Mixing: the coarse scale is good at the trend part of the forecast, the fine
scale at the seasonal part, and summing lets each contribute what it is good at. The tempting
alternative is to merge the scales first — concatenate or interpolate to a common length and run a
single head — but that re-entangles exactly what the block spent its effort separating: forced through
one head, the coarse and fine features have to compromise on one set of weights, the same "one
resolution carries everything" failure that capped PatchTST, moved to the output. Keeping predictions
separate until the end is what lets the complementary skills pay off. (Summing versus averaging differs
only by a constant the weights absorb.) Wrap the whole thing in reversible per-scale instance
normalization — the level-decoupling that helped PatchTST, now per scale so each resolution centers on
its own statistics — and denormalize the summed forecast with the finest-scale statistics. Run
channel-independently, as the previous model's channel argument established: each M4 series is its own
univariate sequence, embedded `1 → d_model` and projected `d_model → 1` with shared weights, which is
also the harness default (`channel_independence = 1`).

The forward composes on all three regimes because only `L_i` and `pred_len` change: the input is pooled
once into a two-scale list, each scale is per-scale-normalized, the channel is folded and embedded
`1 → 512`, the PDM blocks mix the two scales and return the same two shapes, each per-scale predictor
maps to the horizon, the projection maps `512 → 1`, and the two per-scale forecasts are stacked, summed,
and denormalized with the finest-scale statistics.

Where does the protocol still strain this model? The harness fixes `d_model = 512`, `d_ff = 512`,
`e_layers = 2`, whereas TimeMixer's own M4 script uses a narrow model (`d_model = 16`, `e_layers = 4`)
— its premise is that an MLP mixer needs width far less than an attention model. So I run it wider and
shallower than it wants. But I expect the width less wasteful than for PatchTST: the per-scale
predictors and cross-scale mixing give the channels real work across two resolutions. The genuine risk
is the opposite of PatchTST's — with only one downsampling step, the coarse scale is very short (6
steps for Yearly), so the ladder is shallow and the multi-scale benefit may be modest on the shortest
regime even if clear on the longer ones; I chose that shallowness deliberately, because the deeper
alternative bottomed out into a 3-step noise scale.

So the expectations against PatchTST's numbers are directional. I expect to beat PatchTST on every
regime, because explicit multi-scale decomposition-mixing is strictly more structure than a
single-resolution patch encoder can express on these windows, and the per-scale predictors put the
stranded width to use. The largest gain should land where trend and season genuinely live at different
scales and the window supports a real coarse view — Monthly (12-step seasonality plus drift, 36-step
window) and Yearly (the trend regime, where pushing the clean coarse trend top-down is exactly the
multi-scale move; this is the one I am most confident about, since top-down trend mixing is built for
trend-dominated forecasting). Quarterly is where I am least confident: its window is already the second
shortest, its structure is a single clean 4-step period rather than a trend/detail split across scales,
and one pooling step to an 8-step coarse view may not separate much PatchTST did not already capture —
a near-tie there would not surprise me. If the gain is flat on that clean-period regime, that is the
signal that a fixed pooling ladder is the wrong lens when the structure is *one sharp period* — and
that the next move is a model that finds the period structure directly inside the window.
