The linear interpolator did exactly what I built it to do, and its numbers tell me where it runs out. On
seed 42 it lands MSE 0.1498 / MAE 0.2693 on ETTh1, 0.0503 / 0.1143 on Weather, and 0.1132 / 0.2428 on ECL
— a decisive win over mean-guessing everywhere, confirming that a length-one linear map already captures
most of the interpolable structure. The pattern across datasets is the interesting part. Weather is best
by a wide margin: ETTh1's MSE is 2.98× Weather's and ECL's is 2.25×, but in MAE the spread is gentler
(2.36× and 2.12×). The MSE gap is worse than the MAE gap on ETTh1, and since MSE punishes large individual
errors quadratically while MAE averages their magnitude, ETTh1's excess error is concentrated in a *few
large misses* rather than spread as uniform blur. That is the fingerprint of sharp local transients that an
affine map smears: hourly transformer temperature has abrupt excursions, and the decomposition only
half-fixes them because a moving-average trend plus a linear seasonal map still cannot represent a
genuinely nonlinear local feature — it interpolates *through* a spike rather than reconstructing it.

ECL indicts the design differently. Its MSE (0.1132) is the worst of the three, yet ECL is *not* a hard
temporal-interpolation problem — the electricity clients are smooth and strongly periodic, the opposite of
ETTh1's transients. So the quadratic-heavy error on ECL cannot be temporal nonlinearity; the only untouched
reservoir is the cross-channel one. With 321 co-moving clients, the channel-blind map throws away the most
informative cue at a masked entry — the simultaneous values of hundreds of correlated channels — and pays
for it. So the diagnosis is twofold and separate: an affine map cannot represent nonlinear temporal
variation (ETTh1, betrayed by the MSE-heavy signature), and the model is channel-blind exactly where
channels are most correlated (ECL, where temporal difficulty cannot explain the error). The next rung has
to fix both — nonlinearity, and looking across channels.

What *kind* of nonlinearity, though — I do not want to reach blindly for a Transformer and re-import the
tokenisation problem that made the linear map competitive. What the linear map got right is that a series
is about *shape over stretches*, not isolated points. What it got wrong is that it modelled each channel as
a flat 1-D vector, so all temporal structure — daily cycle, weekly cycle, within-day shape — got mashed
into one axis where it overlaps and interferes. A 1-D model, linear or not, has to disentangle all of that
along a single axis, which is genuinely hard. So: is there a representation in which these overlapping
variations stop overlapping?

Here is the observation that cracks it. A periodicity is a statement that values one period apart are
related. If I knew a channel's dominant period *p*, I could fold its length-96 window into a 2-D array of
shape `(96/p, p)`: successive rows are successive periods, each column is a fixed phase. Then two kinds of
variation become two *orthogonal* axes — moving down a column traverses the same phase across cycles (the
slow between-period variation, trend and the cycle's evolution), moving along a row traverses one full
cycle (the fast within-period shape). What was one tangled 1-D axis becomes two clean 2-D axes, and 2-D
structure is exactly what a 2-D convolution models — local neighbourhoods in both directions at once. With
the numbers I have: on hourly data the daily cycle is period 24, so a length-96 window folds into
`(4, 24)` — four stacked days, each a 24-hour row; period 12 folds into `(8, 12)`; period 96 folds into
`(1, 96)`, no folding. A 3×3 conv over the `(4, 24)` fold touches the same hour on adjacent days *and*
adjacent hours on the same day in one receptive field — the two-scale locality a flat map cannot express,
with the interference gone because the two scales are now on different axes. So the move is: fold the 1-D
window into 2-D by period, then use 2-D convolution — real nonlinearity (stacked convs with GELU) aligned
with the structure of the data rather than fighting it.

But which period? Real series have several, and I do not know them in advance. The principled way to find
the dominant periodicities is the frequency domain: FFT the window along time, and the frequencies with
largest amplitude are the dominant periodicities; their reciprocals (scaled by window length) are the
periods. An rFFT of a length-96 window returns 96/2 + 1 = 49 complex bins at integer frequency 0..48; bin
*f* corresponds to period 96/f. Frequency 0 is DC (mean), 1 is period 96, 2 is period 48, 4 is period 24
(the daily cycle), 8 is period 12. So I average the amplitude over batch and channels to get one spectrum,
zero out the DC bin so the trend does not masquerade as a period — this matters, because the standardised
windows still carry a large low-frequency component and without zeroing it the arg-max would forever pick
"the whole window" — and take the top-*k* frequencies. With `top_k = 5` I keep the five loudest, each
giving a candidate period `p_i = 96 // freq_i`. I fold by each, run the 2-D conv stack on each folded view,
unfold, and combine the *k* reconstructions.

Folding by an arbitrary integer period needs a divisibility guard. If `96 % p ≠ 0` the window does not tile
a whole number of periods, so before reshaping I pad the time axis up to `length = (96 // p + 1) * p` with
zeros — take `p = 96 // 5 = 19`: then `96 // 19 = 5`, so `length = 6 × 19 = 114`, I append 18 zero rows,
fold into `(6, 19)`, and crop back to the first 96 positions after the conv. The appended zeros are outside
the real window and get cropped, so they never pollute the scored positions. For recombining the *k* views,
the natural weighting is the amplitudes themselves — a period carrying more spectral energy should count
for more — so I softmax the top-*k* amplitudes and take the weighted sum. The model discovers per window
which periodicities matter and leans on them proportionally, with no hand-set period.

Why five and not one or twenty? Of the 49 rFFT bins the vast majority carry noise-floor amplitude; the
recoverable periodic structure lives in a handful of low-frequency peaks — the daily 24, its harmonics 12
and 8, the whole-window 96 and half-window 48, and on hourly electricity a weekly-ish component if it fits
96 hours. That is roughly five to six genuinely informative periods; `k = 5` captures the real peaks while
excluding the long tail of noise bins whose folded reconstructions would only inject variance into the
weighted sum. Push `k` much higher and the softmax spreads weight onto meaningless folds; push it to 1 and
I bet everything on a single period, throwing away the multi-scale structure that was the whole point.

That is one block; I stack `e_layers = 2` of them with a residual connection and a LayerNorm between, so
each block adds a periodic correction on top of its input rather than replacing it — the second block
refines the first's reconstruction instead of rediscovering the window from scratch, and the LayerNorm
keeps the feature scale from drifting. I should be clear-eyed that this is a heavy rung: the conv is an
Inception stack of `num_kernels = 6` parallel 2-D convolutions with kernel sizes 1..11, and at
`d_model = d_ff = 512` a single Inception block scales as 512 × 512 × (1+9+25+49+81+121) = 512 × 512 × 286
≈ 75M parameters, two blocks per TimesBlock, two TimesBlocks — orders past DLinear's 18,624-weight map.
That is the price of genuine nonlinearity aligned to the data; the budget (Adam, 10 epochs, batch 16) can
pay it, but it means any win should be attributed to the periodic *structure* it exploits, not merely to
having more parameters. Now the wiring into the imputation contract, with the task-specific details that
differ from the forecasting or anomaly versions.

First, the input space. A 2-D conv over a folded window needs a richer per-timestep representation than a
raw scalar — the conv channels carry the learned features. So I embed the masked window into `d_model`
dimensions with the library's `DataEmbedding`: a value projection plus a positional code plus the
time-feature (`x_mark_enc`) embedding. The value projection is `Linear(enc_in, d_model)`, and this is
exactly where cross-channel information enters — on ECL it maps all 321 client values at a timestep into
the shared 512-dim feature, on ETTh1 it maps 7, and everything downstream operates on that mixed vector.
The time features matter here in a way they did not for the linear map: the calendar stamp tells the model
the phase of the daily and weekly cycle directly, exactly the periodic structure the FFT folding exploits.

Second — the part I must get right for *imputation specifically* — normalisation under a mask. The
non-stationary normalisation (centre and scale per window, undo after) cannot use the mean/variance over
all timesteps, because a quarter of the entries are fake zeros. Naively averaging over all 96 timesteps
gives (sum of 72 observed) / 96 = 0.75 times the true observed mean, a 25% shrink, and the variance is
corrupted the same way because the fake zeros sit far below the real level. So I compute the statistics
over the *observed entries only*: mean is the sum of `x_enc` over time divided by the observed count
(`sum(mask == 1)`, which averages 72 not 96), then subtract it and re-zero the masked positions
(`masked_fill(mask == 0, 0)`) so the holes stay holes — without re-zeroing, subtracting the mean turns
every hole from 0 into `−mean`, a spurious nonzero the conv would treat as signal. The std is the RMS of
those centred, re-masked values over the observed count. Both statistics are detached, so the
normalisation is a fixed transform I do not backprop through. After the TimesBlocks and the projection to
`c_out`, I de-normalise by the stored std and mean. Note that, unlike the forecasting path, there is no
`predict_linear` to stretch the temporal axis — `pred_len = seq_len`, the length is preserved end to end,
and I key every length off `seq_len` rather than `configs.pred_len`, because the harness passes a
forecasting-style `pred_len` that does not match the imputation convention.

Does this answer both halves of the diagnosis? The nonlinearity is real — stacked 2-D convs with GELU over
a folded window are a genuine nonlinear map, so the ETTh1 transients an affine map smeared can now be
represented, and the MSE-heavy signature is what a nonlinear local operator should cut. The
channel-blindness is addressed, though indirectly, and this is the honest caveat: TimesNet is not
channel-independent. The `DataEmbedding`'s value projection mixes all `enc_in` channels into the `d_model`
features at each timestep, and the convs and projection operate on those mixed features, so cross-channel
information *is* available — a masked ECL client can be filled using the representation that saw all 321
clients at that timestep. It is not explicit channel-attention, but it is not blind either, and on ECL that
should be the decisive difference versus rung one. The one thing that could go wrong: cramming 321 distinct
clients through a single `Linear(321, 512)` into one shared feature basis might blur the per-client
identity, in which case the mixing helps less than I hope. I will watch ECL for whether the implicit mixing
is enough.

So rung two is the periodic 2-D model: masked-statistics normalisation, embed with time features, stack
`e_layers` TimesBlocks that FFT-discover the top-*k* periods, fold each into 2-D, run an Inception-style
2-D conv stack, amplitude-weight and sum, residual and LayerNorm, then project and de-normalise. A priori I
expect improvement on all three datasets, since each has the periodic structure this model exploits, with
the largest gains at DLinear's two failure points — the nonlinearity should cut ETTh1's transient-smearing
error, and the cross-channel mixing should cut ECL's channel-blind error — and a smaller margin on Weather,
where less error is left to remove. The two ways I could be wrong are diagnostic: if ECL does not improve
much, the embedding's implicit mixing is too weak and the next rung needs explicit cross-channel structure;
if ETTh1 does not improve, the nonlinearity is not the bottleneck.
