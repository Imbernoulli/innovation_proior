The patch-Transformer did exactly what I predicted and no more, and the "no more" is the tell. It beat
the linear floor on every regime — Monthly 12.97 against DLinear's 13.39, Quarterly 10.22 against
10.50, Yearly 13.68 against 14.36 — and the largest absolute gain landed on Yearly, just as the
reversible instance normalization and the learned nonlinear representation were supposed to help most.
Let me read the three gains as numbers, because their pattern is the whole diagnosis. Monthly improved
0.427 (a 3.2% relative drop), Quarterly 0.281 (2.7%), Yearly 0.679 (4.7%). So the improvement is largest
both absolutely and *relatively* on Yearly — the regime where DLinear had no periodic structure to
exploit and the largest per-series level drift to remove — which is exactly the fingerprint of the fix I
built: RevIN's level-decoupling and the FFN nonlinearity paying off most where the affine map was most
strained. The two things DLinear structurally lacked were real, and adding them moved every number in the
right direction. But the *size* of the move is the second half of the diagnosis. Every gain is a few
tenths of a SMAPE point, not a clear separation, and that is the falsifiable case I flagged: a
512-wide, 2-layer attention encoder over a handful of patch tokens is near its ceiling. I counted those
tokens last rung — Monthly four, Quarterly two, Yearly a single padded patch — so on two of three regimes
attention had essentially nothing to relate, and the Yearly gain I just measured is carried by RevIN and
the FFN, not by attention finding cross-patch structure. There simply are not enough patch tokens on a
`2·pred_len` window for self-attention to find structure that a normalized linear map missed; most of
the encoder's width is stranded capacity on one-to-four tokens. The lesson is concrete: stop adding
generic capacity over the single-scale window, because the window is too short to feed it. The
information PatchTST is leaving on the table is not "more attention" — it is the *multi-scale* structure
of the series, which a single fixed-resolution view, patched or not, never exposes.

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
them." And it directly answers the stranded-width finding — a mixer that operates across scales and a
decomposed season/trend representation gives the 512 channels real work, rather than attending over two
tokens.

Build the ladder by average pooling, and I have to decide how deep before anything else, because the
windows are tiny and a ladder can bottom out into noise. Take the input window and downsample it by a
window of 2 some number of times. One step gives Monthly a 36-step scale and an 18-step scale, Quarterly
16 and 8, Yearly 12 and 6. A second step would give Monthly 36/18/9, Quarterly 16/8/4, and Yearly
12/6/**3** — and a three-step coarse view is noise, not a macroscopic trend; there is nothing meaningful
to read off three averaged points, and the per-scale predictor for it would be a `Linear(3 → 6)` fitting
structure that is not there. So a deep ladder actively hurts the shortest regime, which is the one I most
need to help. One downsampling step (window 2) is the right depth on M4: it exposes a coarse view
alongside the fine one without any scale collapsing to noise. I also choose *average* pooling
deliberately over the two alternatives. A learned strided conv is more expressive but less neutral — it
would distort what "coarsening" means by learning a filter, and I want the coarse view to be an honest
central-tendency summary I can reason about, not another learned transform. Max-pool tracks peaks rather
than the average, which is wrong for a macroscopic view that should show central tendency. Parameter-free
average pooling keeps the ladder honest and interpretable, which matters because the whole method rests
on the coarse view genuinely being "the trend, cleaner."

This is the first place the harness protocol bites, and I have to handle it deliberately: the fixed
Custom scripts do **not** pass `--down_sampling_layers`, `--down_sampling_window`, or
`--down_sampling_method`, so under the harness those args default to `0`, `1`, and `None` — which would
collapse the model to a *single scale* (zero downsampling layers, pool window 1), throwing away the
entire multi-scale thesis and leaving me with a fancy single-resolution MLP no better than PatchTST. I
can watch it happen in the arithmetic: with `down_sampling_layers = 0` the multi-scale input list has one
entry, the season/trend mixing loops over zero adjacent pairs and do nothing, and the decoder has one
per-scale predictor — a plain single-scale MLP. So I must set the downsampling configuration *inside*
`Custom.py` (one downsampling layer, window 2, average pooling), reading `configs` only as an override:
`self.down_layers = getattr(configs, 'down_sampling_layers', 0) or 1` forces at least one layer, and
`if self.down_window <= 1: self.down_window = 2` forces a real pool window. The whole contribution of
this rung exists only if I do that explicitly; this is exactly the kind of paper-vs-harness gap where the
same-named model would silently degenerate if I trusted the command line, and the degeneration is not a
crash — it is a quiet collapse to the previous rung's ceiling, which would be the worst kind of failure
to miss.

Now the core: how to mix across scales. Inside each block, first *decompose* every scale into season
and trend with a moving-average block (`trend = AvgPool`, `season = x − trend`), because even a coarse
series is still a season+trend mixture — mixing entangled wholes would fight the very superposition the
decomposition was meant to remove. Let me make the necessity of decomposing-first concrete by imagining the shortcut of mixing the raw
scales directly. Suppose I skip the decomposition and mix the whole coarse and whole fine series against
each other. Bottom-up I would push the *entire* fine scale — its clean trend and its noisy seasonal
detail together — up into the coarse scale, contaminating the coarse scale's clean trend with fine-scale
noise, which is precisely the thing that made the coarse view valuable. Top-down I would push the whole
coarse scale — trend and its smoothed-away season together — down into the fine scale, smearing the fine
seasonal detail with a coarse blur. Mixing entangled wholes therefore fights the very superposition I am
trying to exploit: each direction carries the wrong component along with the right one. Decomposing each
scale first is what lets bottom-up carry *only* season and top-down carry *only* trend, so each direction
transfers exactly the component that scale is authoritative about and nothing else. That is why the
decomposition is not a preprocessing nicety here but the precondition that makes the directional mixing
mean anything.

Then mix the two components across scales in *opposite directions*,
and this asymmetry is the heart of the method, so let me reason it out rather than assert it. Seasonal
detail aggregates upward: coarse seasonality is literally an aggregation of finer seasonality (average
a fine oscillation and you get the envelope of the coarse one), so the fine scale has information the
coarse scale should receive — season mixes **bottom-up**, fine → coarse. Trend goes the other way: for
the macroscopic trend, fine-scale detail is just noise, while the *clean* trend is most visible at the
coarse scale, so the coarse scale has the information the fine scale should receive — trend mixes
**top-down**, coarse → fine. Reversing either direction destroys the information that made that scale
valuable: pushing noisy fine detail down into the trend, or coarse-smoothed season up where the detail
was the point. Let me confirm the two directions actually fall out of the wiring on a two-scale ladder
(`L0 = 36`, `L1 = 18` on Monthly), because a reversed index here would silently invert the whole thesis.
Season mixing walks `down_layers[0]`: a `Linear(36 → 18)`, GELU, `Linear(18 → 18)` that maps the fine
season down to the coarse length and *adds* it to the coarse season — donor is the fine scale, recipient
the coarse, so fine → coarse, bottom-up. Trend mixing walks `up_layers[0]` on the reversed list: a
`Linear(18 → 36)`, GELU, `Linear(36 → 36)` that maps the coarse trend up to the fine length and adds it
to the fine trend — donor coarse, recipient fine, so coarse → fine, top-down. The directions check out.
The mixing primitive is a two-layer MLP with a GELU between rather than a single linear, because a single
linear could only resample the length, while an MLP can actually *transform* the donor scale into a
useful supplement, and it is added as a residual so the recipient scale is never overwritten. After
mixing, recombine season and trend and apply a channel feed-forward `d_model → d_ff → d_model` with a
residual. Stack a few such blocks (`e_layers`, which the harness fixes at 2).

Before I fix the decoder I should price this block, because the whole complaint about PatchTST was
stranded width and I do not want to reproduce it. The channel feed-forward `d_model → d_ff → d_model`
is `2·512·512 ≈ 524k` weights per block, so two blocks is about a million parameters, and that is where
almost all the capacity sits — the season and trend mixing MLPs are tiny (`36→18→18` and `18→36→36` are a
few thousand weights each), the embedding `1 → 512` is a thousand, the predictors a few hundred. So
roughly a third of PatchTST's ~3M, and — this is the point — the width is spent on a *channel* FFN that
transforms the 512-dimensional per-timestep representation, which is genuine per-position nonlinear work
on every scale, not attention idling over two tokens. That is the concrete sense in which the mixer puts
the harness's width to use where the patch encoder could not: the capacity feeds a pointwise MLP that
always has something to do, independent of how many tokens the short window yields.

The decoder is the second idea, and it is the one that most directly answers PatchTST's stranded-
capacity problem. Instead of merging all scales into one representation and predicting once — which
wastes the complementary forecasting skills different scales have — give **each scale its own
predictor** (one linear map from that scale's length to the horizon) and **sum** the per-scale
forecasts. Concretely the predictors are `Linear(36 → 18)` and `Linear(18 → 18)` on Monthly,
`Linear(16 → 8)` and `Linear(8 → 8)` on Quarterly, `Linear(12 → 6)` and `Linear(6 → 6)` on Yearly — the
finest scale's predictor reads the full window, the coarse one reads the halved view, and their horizon
outputs are summed. This is Future-Multipredictor-Mixing: the coarse scale is good at the trend part of
the forecast, the fine scale at the seasonal part, and summing lets each contribute what it is good at,
rather than forcing one merged feature through a single head. Summing versus averaging differs only by a
constant the network absorbs into its weights, so the choice is immaterial; what matters is that the
predictions stay *separate until the end*. The tempting alternative here is to merge the scales first —
concatenate the two scales' features into one `[B, L0 + L1, 512]` (or interpolate to a common length)
and run a single prediction head. It is the obvious "fuse then predict" design and it is what a merged
decoder would do. But it re-entangles exactly what I spent the whole block separating: forced through one
shared head, the coarse features (good at the trend part of the forecast) and the fine features (good at
the seasonal part) have to compromise on one set of head weights, which is the same "one resolution
carries everything" failure that capped PatchTST, just moved to the output. Keeping a *separate* linear
predictor per scale and summing lets the coarse predictor specialize on the trend contribution and the
fine one on the seasonal contribution, and the sum reconstructs the whole — the complementary-skills
argument only pays off if the skills are not merged before the horizon is formed. So I reject the merged
head and keep the per-scale predictors distinct. Wrap the whole thing in reversible per-scale instance
normalization — the same level-decoupling that helped PatchTST, now applied per scale so each resolution
is centered on its own statistics — and denormalize the summed forecast with the finest-scale
statistics. And run channel-independently, exactly as the channel argument from the previous rung
established: on M4 each series is its own univariate sequence (`enc_in = 1`), embedded `1 → d_model` and
projected `d_model → 1` with shared weights, which is also the harness default (`channel_independence =
1`) so I get it for free.

Let me trace the forecast shapes once, because the channel-independent multi-scale pipeline reshapes
between a batch-of-series view and a batch-of-scale-tokens view several times and a wrong axis would
scramble the scales. On Monthly the input `[B, 36, 1]` is pooled once to build the list
`[[B,36,1], [B,18,1]]`. Each scale is per-scale-normalized, then the channel is folded so scale `i`
becomes `[B·1, L_i, 1]` and embedded `1 → 512` to `[B, L_i, 512]`. The two PDM blocks mix the two scales
against each other and return the same two shapes. Each per-scale predictor maps `[B, 512, L_i] →
[B, 512, 18]` along time, the projection maps `512 → 1`, and I reshape back to `[B, 18, 1]`; stacking the
two per-scale forecasts and summing gives one `[B, 18, 1]`, denormalized with the finest-scale mean and
std. The last-`pred_len` slice is a no-op since the output is already length 18. Shapes compose on all
three regimes because only `L_i` and `pred_len` change.

I should be honest about where the protocol still strains this rung, because PatchTST taught me to watch
exactly that. The harness fixes `d_model = 512`, `d_ff = 512`, `e_layers = 2`, whereas TimeMixer's own
M4 script uses a *narrow* model (`d_model = 16`, `e_layers = 4`) — its whole design premise is that an
MLP mixer needs width far less than an attention model. So I am again running this architecture wider and
shallower than it wants. But here I expect the multi-scale structure to make the width *less* wasteful
than it was for PatchTST: the per-scale predictors and the cross-scale mixing give the 512-wide channels
something real to do across two resolutions and a decomposed season/trend representation, rather than
attending over two patch tokens. The reversible normalization and channel sharing remain the
regularizers, and 10-epoch early stopping caps overfitting. The genuine risk is the opposite of
PatchTST's: with only one downsampling step on a short window, the coarse scale is very short (6 steps
for Yearly), so the ladder is shallow and the multi-scale benefit may be modest on the shortest regime
even if it is clear on the longer ones — and I chose that shallowness deliberately, because the deeper
alternative bottomed out into a 3-step noise scale.

So the falsifiable expectations against PatchTST's measured numbers are specific and directional. I
expect to beat PatchTST on **every** regime, because explicit multi-scale decomposition-mixing is
strictly more structure than a single-resolution patch encoder can express on these short windows, and
because the per-scale predictors put the harness's stranded width to use. I expect the **largest** gain
on the regimes where trend and season genuinely live at different scales and the window is long enough
to support a real coarse view — Monthly (clear 12-step seasonality plus drift, 36-step window) and
Yearly (the trend regime, where pushing the clean coarse trend top-down into the forecast is exactly
the multi-scale move). Concretely I expect to clear PatchTST's 12.97 (Monthly), 10.22 (Quarterly), and
13.68 (Yearly), with the Yearly improvement the one I am most confident about, since trend-dominated
forecasting is precisely what top-down trend mixing is built for. Quarterly is where I am least
confident: its window (16) is already the second shortest, its structure is a single clean 4-step period
rather than a trend/detail split across scales, and one pooling step to an 8-step coarse view may not
separate much that PatchTST's 10.22 did not already capture — so a near-tie there would not surprise me.
If the gain is flat on that clean-period regime, that is the signal that a fixed pooling ladder is the
wrong lens when the structure is *one sharp period* rather than a trend/detail split — and that the next
move is a model that finds the *period* structure directly inside the window rather than relying on a
fixed pooling ladder to expose it.
