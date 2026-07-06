The linear interpolator did exactly what I built it to do, and its numbers tell me precisely where it
runs out. On seed 42 it lands MSE 0.1498 / MAE 0.2693 on ETTh1, 0.0503 / 0.1143 on Weather, and 0.1132
/ 0.2428 on ECL — a decisive win over mean-guessing everywhere, confirming that a length-one linear map
along time already captures most of the interpolable structure. But the pattern across the three datasets
is the interesting part, and it falls out along the axis I flagged. Let me read it quantitatively rather
than by eye. Weather is the best result by a wide margin, and the ratios make the gap concrete: ETTh1's
MSE is 0.1498 / 0.0503 ≈ 2.98× Weather's, and ECL's is 0.1132 / 0.0503 ≈ 2.25× Weather's; in MAE the
spread is gentler (0.2693 / 0.1143 ≈ 2.36× and 0.2428 / 0.1143 ≈ 2.12×), which already tells me something
— the MSE gap is worse than the MAE gap on ETTh1, and MSE punishes large individual errors quadratically
while MAE averages their magnitude, so ETTh1's excess error is concentrated in a *few large misses*
rather than spread as uniform low-grade blur. That is the fingerprint of sharp local transients that an
affine map of the window smears: hourly transformer temperature has abrupt local excursions, and the
decomposition only half-fixes them because a moving-average trend plus a linear seasonal map still cannot
represent a genuinely nonlinear local feature — it interpolates *through* a spike rather than
reconstructing it.

ECL is the one that indicts the design differently, and the ratios point somewhere else entirely. Its MSE
(0.1132) is actually the worst of the three, yet ECL is *not* a hard interpolation problem in the temporal
sense — the electricity clients are smooth and strongly periodic, the opposite of ETTh1's transients. So
the quadratic-heavy error on ECL cannot be blamed on temporal nonlinearity the way ETTh1's can; the only
reservoir of information the model is leaving untouched is the cross-channel one. With 321 strongly
co-moving clients, the channel-blind map throws away the single most informative cue at a masked entry —
the simultaneous values of the hundreds of correlated channels — and it pays for it. So the diagnosis is
sharp and it is twofold: first, an affine map of a window cannot represent genuinely nonlinear temporal
variation (the ETTh1 transients, betrayed by the MSE-heavy error signature); second, and separately, the
model is channel-blind exactly where channels are most correlated (ECL, where temporal difficulty cannot
explain the error). The next rung has to fix both — it needs nonlinearity and it needs to look across
channels.

Let me think about what *kind* of nonlinearity, because I do not want to reach blindly for a Transformer
and re-import the tokenisation problem that made the linear map competitive in the first place. The thing
the linear map got right is that a time series is about *shape over stretches*, not isolated points. The
thing it got wrong is that it modelled each channel's window as a flat 1-D vector, so all temporal
structure — the daily cycle, the weekly cycle, the within-day shape — got mashed into one axis where it
overlaps and interferes. On a real channel several variations live in the same 1-D record at once: a slow
trend, a daily oscillation, a weekly one, and short fluctuations, all superimposed. A 1-D model, linear or
not, has to disentangle all of them along a single axis, and that is genuinely hard. So the question I want
to ask is: is there a representation in which these overlapping variations stop overlapping?

Here is the observation that cracks it. A periodicity is a statement that values one period apart are
related. If I knew a channel's dominant period *p*, I could fold its length-96 window into a 2-D array of
shape `(96/p, p)` — successive rows are successive periods, each column is a fixed phase of the cycle. Then
two kinds of variation become two *orthogonal axes*: moving down a column traverses the same phase across
successive cycles (the slow, between-period variation — trend and the cycle's evolution), and moving along
a row traverses one full cycle (the fast, within-period shape). What was one tangled 1-D axis becomes two
clean 2-D axes, and crucially a 2-D structure is exactly what a 2-D convolution is built to model — local
neighbourhoods in both the within-period and between-period directions at once. Make this concrete with
the numbers I actually have: on hourly data the daily cycle is period 24, so a length-96 window folds into
`(96/24, 24) = (4, 24)` — four stacked days, each a 24-hour row; the half-day period 12 folds into
`(8, 12)`; the whole-window period 96 folds into `(1, 96)`, i.e. no folding at all. A 3×3 conv over the
`(4, 24)` fold touches the same hour on adjacent days *and* adjacent hours on the same day in one
receptive field — precisely the two-scale locality a flat linear map cannot express. The interference is
gone because the two scales are now on different axes. That reframing — fold the 1-D window into 2-D by
period, then use 2-D convolution — is the move I want, because it gives me real nonlinearity (stacked
convs with GELU) that is *aligned with the structure of the data* rather than fighting it, and 2-D convs
over a folded window naturally mix neighbouring phases and cycles in a way a flat linear map cannot.

But which period? Real series do not have one clean period; they have several, and I do not know them in
advance. The principled way to find the dominant periodicities of a window is the frequency domain: take
the FFT of the window along time, look at the amplitude spectrum, and the frequencies with the largest
amplitude are the dominant periodicities; their reciprocals (scaled by the window length) are the periods.
Let me work the arithmetic so I know exactly what the candidate periods can be. An rFFT of a length-96
window returns 96/2 + 1 = 49 complex bins indexed by integer frequency 0..48; bin *f* corresponds to
period 96/f. Frequency 0 is the DC (mean) term, frequency 1 is period 96 (one cycle spans the window),
frequency 2 is period 48, frequency 4 is period 24 (the daily cycle on hourly data), frequency 8 is period
12. So I compute the rFFT of the window, average the amplitude over batch and channels to get one spectrum
per window-set, zero out the zero-frequency (DC) bin so the trend does not masquerade as a period — this
matters, because the standardised windows still carry a large low-frequency component and without zeroing
it the arg-max would forever pick "the whole window" as the period — and take the top-*k* frequencies by
amplitude. With `top_k = 5` I keep the five loudest, each giving a candidate period `p_i = 96 // freq_i`.
I fold the window by each of the top-*k* periods separately, run the 2-D conv stack on each folded view,
unfold back to 1-D, and then I need to combine the *k* reconstructions.

Folding by an arbitrary integer period needs a divisibility guard, and I want to get it right because a
wrong reshape silently corrupts everything downstream. If `96 % p ≠ 0` the window does not tile a whole
number of periods, so before reshaping I pad the time axis up to the next multiple of *p* with zeros:
`length = (96 // p + 1) * p`. Take a period the FFT can actually produce, say frequency 5 giving
`p = 96 // 5 = 19`: then `96 // 19 = 5`, so `length = 6 × 19 = 114`, and I append 114 − 96 = 18 zero rows
before folding into `(6, 19)`, then crop back to the first 96 positions after the conv and unfold. The
zeros I append are outside the real window and get cropped off, so they do not pollute the scored
positions; the guard just makes the reshape legal. The natural weighting for recombining the *k* views is
the amplitudes themselves: a period that carried more spectral energy should count for more, so I softmax
the top-*k* amplitudes into weights and take the weighted sum of the *k* unfolded outputs. That adaptive
aggregation means the model discovers, per window, which periodicities matter and leans on them
proportionally — no hand-set period, no single-period assumption.

Why five periods and not one or twenty? The count `top_k` trades coverage against noise, and I can bound
it from the spectrum's structure. Of the 49 rFFT bins, the vast majority carry noise-floor amplitude; the
recoverable periodic structure on these datasets lives in a handful of low-frequency peaks — the daily
period 24, its harmonics at 12 and 8, the whole-window 96 and half-window 48, and on hourly electricity a
weekly-ish component if it fits the 96-hour window. That is roughly five to six genuinely informative
periods; keeping `k = 5` captures the real peaks while excluding the long tail of noise bins whose folded
reconstructions would only inject variance into the amplitude-weighted sum. Push `k` much higher and the
softmax would spread weight onto meaningless folds; push it to 1 and I would bet the whole block on a
single period, throwing away the multi-scale structure that was the entire point of moving to 2-D. Five is
the count that matches the number of real spectral peaks these windows actually have, so the aggregation
sums over signal and not over noise.

That is one block; I stack a few of them with a residual connection so the representation can be refined,
and a LayerNorm between blocks to keep it stable. Each block: FFT to find the top-*k* periods and their
amplitudes, fold-conv-unfold for each period, amplitude-weighted sum, add the input back. With
`e_layers = 2` I get two such blocks, and the residual connection is what makes stacking safe: each block
adds a periodic correction on top of its input rather than replacing it, so the second block refines the
first block's reconstruction instead of having to rediscover the window from scratch, and the LayerNorm
between them keeps the feature scale from drifting across the two stages. I should be clear-eyed that this is a heavy rung compared with the
last one: the conv is an Inception stack of `num_kernels = 6` parallel 2-D convolutions with kernel sizes
1, 3, 5, 7, 9, 11, and at `d_model = d_ff = 512` the weight count of a single Inception block scales as
d_model × d_ff × Σ k² = 512 × 512 × (1+9+25+49+81+121) = 512 × 512 × 286 ≈ 75M parameters, two such blocks
per TimesBlock, two TimesBlocks — orders of magnitude past DLinear's 18,624-weight map. That is the price
of genuine nonlinearity aligned to the data, and it is a price the budget (Adam, 10 epochs, batch 16) can
pay; but it tells me this rung is buying capacity, and I should attribute any win to the periodic
*structure* it exploits, not merely to having more parameters. Now I have to wire this into the imputation
contract, and there are several task-specific details that are *not* the same as the forecasting or
anomaly-detection versions of this idea, so I derive them rather than transplant them.

First, the input space. A 2-D conv over a folded window needs a richer per-timestep representation than a
raw scalar — the conv channels are what carry the learned features. So I embed the masked window into a
`d_model`-dimensional space first with the library's `DataEmbedding`: a value projection plus a positional
code plus the time-feature (`x_mark_enc`) embedding. The value projection is a `Linear(enc_in, d_model)`,
and this is the exact place where cross-channel information enters: on ECL it maps all 321 client values at
a timestep into the shared 512-dim feature, on ETTh1 it maps 7, and downstream everything operates on that
mixed vector. The time features matter here in a way they did not for the linear map: the calendar stamp
tells the model the phase of the daily and weekly cycle directly, which is exactly the periodic structure
the FFT folding exploits. After the embedding the FFT for period discovery runs in the embedded space, and
the folding reshapes `(B, 96, d_model)` into `(B, 96/p, p, d_model)`, permutes `d_model` to the
conv-channel axis, and the 2-D conv mixes within-period and between-period neighbourhoods.

Second — and this is the part I must get right for *imputation specifically* — normalisation under a mask.
The Non-stationary-Transformer normalisation that wraps the model (centre and scale per window, undo
after) cannot use the standard mean/variance over all timesteps, because a quarter of the entries are
fake zeros and would corrupt the statistics. Let me quantify how badly. If I naively averaged over all 96
timesteps, treating the punched-out zeros as data, I would get (sum of the 72 observed values) / 96, which
is 0.75 times the true observed mean — a 25% shrink toward zero, and the variance would be corrupted the
same way because the fake zeros sit far below the real level and inflate the spread. So I compute the
statistics over the *observed entries only*: the mean is the sum of `x_enc` over time divided by the count
of observed entries per channel (`sum(mask == 1)`, which averages 72 not 96), then I subtract it and
re-zero the masked positions (`masked_fill(mask == 0, 0)`) so the holes stay holes after centring — if I
did not re-zero, subtracting the mean would turn every hole from 0 into `−mean`, a spurious nonzero that
the conv would happily treat as signal. The standard deviation is then the root-mean-square of those
centred, re-masked values, again normalised by the observed count. Both statistics are detached so the
normalisation is treated as a fixed transform, not something to backprop through — I do not want gradients
trying to game the per-window scale. After the TimesBlocks and the linear projection back to `c_out`
channels, I de-normalise by multiplying by the stored std and adding the stored mean. This masked
normalisation is the single most important task-specific change: it is what lets the periodic 2-D
modelling see the window's *shape* on a common scale without the punched-out zeros biasing the centre and
width. Note that, unlike the forecasting path, there is no `predict_linear` to stretch the temporal axis —
for imputation `pred_len = seq_len`, the sequence length is preserved end to end, so the embedded window
goes straight into the TimesBlocks at its native length and the projection maps `d_model → c_out` at every
one of the 96 positions. I deliberately key every length off `seq_len` rather than `configs.pred_len`,
because the harness passes a forecasting-style `pred_len` that does not match the imputation convention;
reading `seq_len` keeps the module correct whatever length the config carries.

Does this answer both halves of the DLinear diagnosis? The nonlinearity is real now — stacked 2-D convs
with GELU over a folded window are a genuine nonlinear map, so the ETTh1 transients that an affine map
smeared can be represented, and the MSE-heavy error signature I read off DLinear's ETTh1 line is exactly
what a nonlinear local operator should be able to cut. And the channel-blindness is addressed, though
indirectly and this is the honest caveat: TimesNet is not channel-independent like the linear map was. The
`DataEmbedding`'s value projection mixes all `enc_in` channels into the `d_model` features at each
timestep, and the 2-D convs and the final projection operate on those mixed features, so cross-channel
information *is* available to the reconstruction — a masked client on ECL can be filled using the embedded
representation that saw all 321 clients at that timestep. It is not an explicit channel-attention, but it
is not blind either, and on ECL that should be the decisive difference versus rung one. I do want to flag
the one thing that could go wrong with this indirect fix: cramming 321 distinct clients through a single
`Linear(321, 512)` into one shared feature basis might blur the per-client identity, in which case the
mixing would help less than I hope. I will watch ECL specifically for whether the implicit mixing is
enough.

Before committing I want two checks that the folding machinery does what I claim, one a degenerate limit
and one a small trace, because a reshape that is subtly wrong would corrupt every downstream number
silently. The limit first: suppose a channel's window is *exactly* periodic with period 24 — the same
24-hour shape repeated four times, no trend, no evolution. Its rFFT concentrates all non-DC energy at
frequency 4 (period 96/4 = 24), so the top-*k* search picks p = 24, the fold is `(4, 24)`, and each of the
24 columns holds four *identical* values down its rows. A 2-D conv then sees, along the between-period
axis, a perfectly flat column (nothing to learn there — correct, because the cycle does not evolve), and
along the within-period axis the 24-point daily shape, which is exactly the object worth modelling. So in
the clean-periodic limit the fold isolates the signal onto the within-period axis and leaves the
between-period axis constant — the representation behaves as designed. Now the softmax-weighting limit: if
one period utterly dominates the amplitude spectrum, `F.softmax` over the top-*k* amplitudes puts almost
all its mass on that single period, and the amplitude-weighted sum collapses to that one folded
reconstruction — the multi-period model degrades gracefully to the single-period model when the data
really has one period, and only spreads weight across several folds when the spectrum genuinely has
several peaks. Both checks say the aggregation is doing sensible book-keeping rather than smearing.

Now a shape trace through one block to be sure the wiring is legal end to end. The embedded tensor is
`(B, 96, 512)`. For period p = 24: no padding needed since 96 % 24 = 0, reshape to `(B, 4, 24, 512)`,
permute the feature axis to the conv-channel slot giving `(B, 512, 4, 24)`, run the Inception 2-D conv
(which preserves the spatial `4 × 24` and returns `(B, 512, 4, 24)`), permute back and flatten the two
period axes to `(B, 96, 512)`, and crop to the first 96 positions (a no-op here since 4 × 24 = 96 exactly).
Repeat for each of the five periods, stack the five `(B, 96, 512)` outputs on a new last axis to get
`(B, 96, 512, 5)`, broadcast the softmaxed period weights over the time and feature axes, sum over the
period axis back to `(B, 96, 512)`, and add the residual. Length in equals length out at every step, the
feature width never changes, and the only tensors that ever get cropped are padded-then-cropped periods
that do not divide 96 — so the scored positions never see a padding zero. The trace closes; the block is
a well-typed same-length map, which is exactly what the imputation contract needs.

So rung two is the periodic 2-D model: masked-statistics normalisation, embed with time features, stack
`e_layers` TimesBlocks that FFT-discover the top-*k* periods, fold each into 2-D, run an Inception-style
2-D conv stack, amplitude-weight and sum, residual and LayerNorm, then project and de-normalise — the full
scaffold module is in the answer. Now the falsifiable expectations against the DLinear numbers. I expect a
clear improvement on all three datasets, because every one of them has the periodic structure this model is
built to exploit. The two places I expect the *largest* gains are exactly DLinear's two failure points:
ETTh1, where the nonlinearity should cut the transient-smearing error well below 0.1498 MSE; and ECL,
where the cross-channel mixing in the embedding should beat the channel-blind 0.1132 MSE substantially. On
Weather, already DLinear's strong suit at 0.0503, I expect improvement but a smaller margin — there is less
error left to remove. If I am wrong — if ECL does *not* improve much — that would say the embedding's
implicit channel mixing is too weak to exploit the correlation, and the next rung would need explicit
cross-channel structure. If ETTh1 does not improve, the nonlinearity is not the bottleneck and the problem
is elsewhere. Concretely, the bar this rung must clear is DLinear's seed-42 line on every dataset, and I
expect it to clear it most decisively on ETTh1 and ECL — the two the linear interpolator handled worst.
