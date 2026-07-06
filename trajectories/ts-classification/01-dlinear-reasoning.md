The scaffold default returns zeros, which means every window gets the same logit vector and the
classifier predicts the majority class — chance, modulated by class imbalance. Let me pin what that floor
actually is per dataset, because "chance" is not one number here. FaceDetection is binary and roughly
balanced, so the zero-logit model sits near 0.50. EthanolConcentration has four classes; if they are close
to balanced, the majority-class guess is around 0.25. Handwriting has twenty-six classes, so blind majority
guessing is near 1/26 ≈ 0.038. Those three numbers — 0.50, 0.25, 0.038 — are the anchors, and they already
tell me the datasets live in completely different regimes: on FaceDetection a model has to beat a coin, on
Handwriting it has to beat one-in-twenty-six. The floor exists to anchor one question: how much accuracy is
available from the *cheapest* honest model that actually reads the window? I want the weakest rung to be a
model I cannot accuse of cleverness, so that whatever it scores is a clean lower bound on "a real encoder
helps." If an elaborate temporal encoder later beats it by a wide margin, the encoder earned its keep; if it
barely beats it, the encoder was mostly decoration. So I deliberately start from the most minimal thing that
can map a multivariate window to class logits, and I let the discomfort that motivates the whole ladder pick
which minimal thing.

The discomfort is concrete. The standard deep recipe for multivariate time-series classification embeds
the whole channel-vector at each timestamp into one token and runs self-attention over the `seq_len`
tokens. But a single timestamp of a load trace or an MEG sweep has no standalone meaning — unlike a word,
a scalar at time *t* is nothing in isolation; the signal lives in the *shape over a stretch* of time. So
point-wise attention is scoring pairwise similarity between objects that individually carry no semantics,
and on top of that it pays a quadratic cost in `seq_len`. Quantify that cost on the dataset that punishes
it: EthanolConcentration's windows are long — the loader pads to the maximum series length, which for these
absorbance traces is on the order of 1,750 timesteps. A per-timestep attention map is then roughly
1,750² ≈ 3.06 million entries per head per layer, for series whose discriminative content is a slow smooth
curve that a handful of numbers would summarize. The fact that nags is that a plain linear map over the
flattened window is reported to be competitive with these encoders. If a linear map is competitive, then
either attention is the wrong tool here, or the leverage is in the *input representation*, not the kernel.
I want the weakest rung to be exactly that linear map — both because it is the right floor and because it
operationalizes the doubt: I am measuring how much of the achievable accuracy a one-layer affine map already
captures, before any temporal machinery.

So what is the minimal model? Read the window `x_enc` of shape `[B, seq_len, enc_in]`, and produce
`[B, num_class]`. The absolute floor is: flatten the window into a single `enc_in · seq_len` vector and
apply one linear layer to `num_class`. That is a logistic regression on the raw flattened window. It
treats every (channel, timestep) pair as an independent feature and learns one weight per pair per class.
It has no notion of temporal order beyond what the weights memorize, no cross-channel modeling beyond the
shared linear combination, and no nonlinearity. It is the honest floor.

Before I settle, let me walk the small design space of "minimal" models and price each, because the choice
between them is not aesthetic — it is an arithmetic about capacity versus the data I have. Option one is the
bare flatten-and-project I just described: parameters `enc_in · seq_len · num_class`. On EthanolConcentration
that is 3 · 1,750 · 4 ≈ 21,000 weights; on FaceDetection 144 · 62 · 2 ≈ 18,000; on Handwriting 3 · 152 · 26 ≈
12,000. All small, all trainable on these tiny UEA splits (EthanolConcentration has only ~260 training
series, Handwriting ~150). Option two adds a hidden nonlinear layer — a two-layer MLP over the flattened
window. That immediately explodes: a single `Linear(enc_in · seq_len, h)` with even h = 128 is 3 · 1,750 · 128
≈ 672,000 weights on EthanolConcentration against 260 examples, a 2,600:1 ratio that will memorize the train
split. And it is no longer the *honest floor* — a nonlinear boundary is exactly the thing I want later rungs
to earn, so buying it now would blur the measurement. Reject. Option three is per-channel *individual* linear
maps along time — give every channel its own temporal weight matrix. I will compute its cost in a moment,
because there is a cheaper variant of the same idea that keeps the floor honest, and that variant is what I
actually ship.

That variant is seasonal-trend decomposition, and I will not ship the *bare* flatten-and-project, because
decomposition is a free, parameter-light refinement that costs nothing in capacity and that I expect to
matter on at least one of these datasets — and using it keeps this rung aligned with the linear-model lineage
that motivated starting here. The idea is the oldest move in time-series analysis: write the series additively
as a slow trend (a moving average) plus a seasonal residual (what is left after subtracting the trend),
because each piece on its own is more regular than the sum. In a classifier this matters because the
discriminative cue can live in either piece. EthanolConcentration is the clearest case: the spectral
absorbance traces are long, smooth curves whose *level and slow shape* carry the concentration information —
that is trend. Handwriting's accelerometer gestures, by contrast, are about the *oscillatory residual* once
the slow drift of the hand is removed — that is seasonal. A single flatten-and-project must fit both kinds of
cue with one weight matrix, and here is the concrete failure mode: the trend component of a spectral trace
has magnitude on the order of the raw absorbance, while the seasonal residual after subtracting a 25-wide
moving average is a small wiggle an order of magnitude smaller. Under a cross-entropy fit driven by the
loud-component gradients, the large-magnitude trend dominates the weight updates and the quiet seasonal cue is
under-fit — the optimizer spends its capacity where the energy is, not necessarily where the label is.
Splitting the window into `trend = MovingAvg(x)` and `seasonal = x − trend`, mapping each with its own linear
layer, and summing the two streams lets each specialize. Crucially this adds *no representational capacity*: a
moving average is linear, two linear maps plus a sum is still affine end to end. Let me verify the losslessness
explicitly, because the whole argument rests on it — `trend + seasonal = MovingAvg(x) + (x − MovingAvg(x)) = x`
exactly, so the decomposition throws nothing away; it only re-bases the two linear maps so each sees a
component of homogeneous magnitude. It is preconditioning, not depth — it separates the loud component from
the quiet one so the optimizer can fit each, exactly the move that distinguishes this rung from the bare
linear map. This is the model I commit to for rung one.

Now let me settle the individual-vs-shared question with the arithmetic I deferred, because it is the one
place this rung could quietly overfit. The temporal maps are the parameter-dominant part: two
`Linear(seq_len, seq_len)` matrices. Shared across channels (the choice I make), that is 2 · seq_len² weights,
*independent of channel count*. On EthanolConcentration seq_len ≈ 1,750, so 2 · 1,750² ≈ 6.13 million weights
— already a startling number against ~260 training series, a 23,000:1 ratio, and this is the *cheap* option.
Now price the individual variant, one temporal map per channel: multiply by `enc_in`. On FaceDetection with
144 channels that turns 2 · 62² ≈ 7,700 weights into 144 · 7,700 ≈ 1.1 million — a 144× blow-up on the very
dataset with the most channels, precisely the recipe for fitting per-sensor noise. So individual maps cost the
most exactly where they help the least. Share the weights. The head, by contrast, *is* where I let channels
differ: a single `Linear(enc_in · seq_len, num_class)`, which is the 21,000 / 18,000 / 12,000-weight projection
I priced above. Note the striking imbalance this exposes — on EthanolConcentration the 6.13M temporal weights
dwarf the 21k head by ~290×, so almost the entire model is the shared per-channel smoothing-and-reweighting of
time, and the actual class decision is a thin linear read-off. That is a warning I will carry: this rung's
capacity is spent on temporal reshaping, not on the decision boundary, and on the long-window dataset it is
wildly over-parameterized and will lean hard on early stopping (patience 10) to avoid memorizing.

Now the mechanics of the moving average, because the details decide whether it behaves. The moving average
must preserve length so I can subtract it from the input. I use an `AvgPool1d` with an odd kernel `k = 25`
(the same smoothing scale the decomposition line uses, so the comparison stays clean), stride 1, and I
replicate-pad the two endpoints by `(k−1)//2 = 12` each. Let me trace the length to be sure it is preserved:
input length `L`, pad 12 on each side gives `L + 24`, an averaging window of width 25 with stride 1 produces
`(L + 24) − 25 + 1 = L` outputs. Length preserved, subtraction well-defined. Replication matters and I should
say why concretely: zero-padding would drag the trend toward zero at the window edges, so at the first
timestep the "average" would be (12 zeros + 13 real values)/25 ≈ roughly half the true local level, inventing
a spurious dip exactly where I have least information; replicating the endpoint value into those 12 slots keeps
the trend flat-but-faithful there. The seasonal part is then `x − trend`, and both streams have shape
`[B, seq_len, enc_in]`.

Let me trace the decomposition on a deliberately tiny window to confirm it does what I claim rather than
just asserting it. Take one channel of six values with a rising trend plus a small alternating wiggle:
`x = [10, 12, 11, 13, 12, 14]` (a slow climb of ~+0.8/step with a ±1 oscillation on top). With a small
smoothing kernel the moving average yields a trend near `[10.7, 11.0, 12.0, 12.0, 13.0, 13.3]` — the climb
survives, the ±1 wiggle is gone. The seasonal residual `x − trend ≈ [−0.7, +1.0, −1.0, +1.0, −1.0, +0.7]`
is exactly the alternating component, summing to ≈ 0 as the losslessness argument predicted. Now the point:
the trend values live around 10–14 while the seasonal values live around ±1, an order-of-magnitude gap. A
single linear map fitting both would see gradients dominated by the 10–14 band and would barely move the
weights that read the ±1 band; giving the seasonal residual its own map lets its ±1 structure drive a
gradient of its own scale. That is the preconditioning made concrete on numbers I can hold in my head.

One more hyperparameter I should not set on autopilot: the smoothing kernel `k = 25`. It is the knob that
decides what counts as "trend" versus "seasonal," and the split it induces is dataset-dependent whether I
like it or not, because the datasets have wildly different window lengths. On EthanolConcentration's ~1,750
steps, a 25-wide average smooths over ~1.4% of the window — it removes fine spectral noise and leaves the
broad absorbance shape as trend, which is what I want there. On FaceDetection's ~62 steps, 25 is ~40% of the
window, so the "trend" is an aggressive smoothing that leaves almost everything in the seasonal residual —
which is fine, because on MEG the fast fluctuation *is* the signal. On Handwriting's ~150 steps, 25 is ~17%,
sitting between the two. A single fixed kernel therefore does something reasonable-but-different on each, and
that is acceptable for a floor: I am not trying to tune the split per dataset, only to give the optimizer two
magnitude-separated streams instead of one. A much larger kernel would push nearly everything into seasonal
on the short datasets (collapsing the decomposition back toward the bare linear map); a much smaller one
would leave high-frequency noise in the trend on EthanolConcentration. 25 is the standard scale and I keep it,
noting that the split is coarse and not adapted — a limitation a later, period-aware rung could improve on.

There is an initialization detail that decides what the model *is* at step zero, and tracing it is the small
verification that the design does something sensible before any training. The two temporal linear maps are
initialized to `(1/seq_len) · ones`, i.e. every output position is the *mean* of its inputs. So at
initialization `Linear_Trend(trend)` outputs, at every time position, the temporal mean of the trend — which
is essentially the temporal mean of the raw window, since the moving average preserves the mean. And
`Linear_Seasonal(seasonal)` at init outputs the temporal mean of the seasonal residual, which is ≈ 0 because
`seasonal = x − MovingAvg(x)` sums to nearly zero over time. So the summed encoder output at initialization is
a constant-per-channel signal equal to each channel's window mean, broadcast across all `seq_len` positions.
Flatten that and the head sees, for each channel, its mean level repeated. In other words, the model *starts*
as "classify by per-channel average level" and then learns, over training, to reweight which timesteps and
which decomposed components matter. That is a sane, interpretable starting point — a mean-level classifier —
and it tells me the decomposition-linear is not initialized in some pathological regime; it is initialized as
the simplest possible feature and refines from there.

A constraint I have to respect while making all these choices: the substrate is frozen and there is exactly
*one* `Custom.py` for all three datasets. I cannot special-case EthanolConcentration's long window with a
different kernel or FaceDetection's 144 channels with a different head. The optimizer is RAdam at lr 1e-3, the
loss is cross-entropy, and early stopping watches with patience 10 over up to 100 epochs — none of that is
mine to change. That matters for the floor specifically because the model that is most over-parameterized —
EthanolConcentration with its 6.13M temporal weights on ~260 series — is entirely at the mercy of early
stopping to halt before it memorizes; there is no weight decay knob I can reach for and no per-dataset
regularization. So the honest reading is that this rung's EthanolConcentration behavior is as much a statement
about the frozen protocol's implicit regularization (patience 10) as about the decomposition itself. On the
small-parameter datasets (FaceDetection ~18k head weights against several thousand series, Handwriting ~58k
total against ~150 series) the ratios are far tamer, so early stopping is doing less work there. This is the
kind of thing a floor is supposed to expose: not just "a linear map scores X," but "under this exact frozen
budget, a linear map with this parameter profile scores X," which is the only comparison later rungs are held
to.

Here is the design decision that ties this to the classification head, and where I depart from how a
*forecaster* would use the same decomposition. A forecaster maps the time axis `L → T` and keeps the
channel axis separate, because it must emit a future trajectory per channel. I do not emit a trajectory;
I emit class logits. So after decomposing, the classification head flattens *everything* — both the time
axis and the channel axis — into one `enc_in · seq_len` vector and projects straight to `num_class`. The
two linear streams (seasonal and trend) each run a `Linear(seq_len, seq_len)` along time (the canonical
decomposition-linear keeps `pred_len = seq_len` for non-forecast tasks, so the per-channel temporal map is
square), they are summed, and the summed `[B, seq_len, enc_in]` representation is reshaped to
`[B, enc_in · seq_len]` and fed to a single `Linear(enc_in · seq_len, num_class)`. That final projection
is where the channels are mixed and the class decision is made; the per-time linear maps before it are the
decomposed temporal encoder. The whole thing is still affine from input to logits.

I have to be honest about what this rung throws away, because the throwaways are exactly what the later
rungs will exploit. First: it ignores the padding mask `x_mark_enc` entirely. The flattened projection
sees the padded positions as ordinary zeros and learns weights for them; on datasets where windows vary
in length within the dataset, the right-padding is a block of zeros that the linear map happily fits a
weight to, which is at best wasted capacity and at worst a spurious cue (padding length can correlate with
class through dataset quirks). Concretely, Handwriting's series vary in length and get right-padded to the
per-dataset maximum, so some fraction of each flattened window is a constant zero tail that the head learns a
column of weights against — capacity spent on positions that carry no signal. The decomposition-linear
classification path does not consult the mask — that is a real limitation I am inheriting, and a later rung
that *does* zero out padding before pooling should gain on the variable-length datasets. Second: it models
cross-channel interaction only through the final flatten-and-project. There is no mechanism for the model to
learn that two MEG channels covary; their covariation can only be expressed as a fixed linear combination in
the head. FaceDetection, whose entire signal is cross-channel covariance over ~144 noisy channels, is the
dataset I expect this to hurt most. Third: it is linear, so any class boundary that is not (after
decomposition) a hyperplane in the flattened-window space is unreachable. Handwriting's 26 classes of
oscillatory gestures are the case where I expect the linear boundary to be too blunt.

So I can already predict the *shape* of where this rung lands, which is the point of a floor. On
EthanolConcentration — smooth spectral traces where the slow trend carries the label and four
well-separated classes — the decomposition should give the linear head a fair shot, and I expect a modest
but real lift over chance (chance with four classes is ~0.25; I expect to land somewhere in the high
0.2s, only a little above chance, because the within-class shape variation in spectra is large for a
purely linear map and the ~23,000:1 parameter-to-example ratio means early stopping halts training before the
map can carve the four overlapping concentration levels cleanly). On FaceDetection — binary, so chance is
~0.5, and the signal is cross-channel covariance the flatten-head can only crudely capture — I expect the
linear map to do *better than its nature suggests* simply because binary problems are forgiving and a linear
combination of 144 channels over a 62-step window has ~18,000 features to draw a single hyperplane through,
with several thousand training series to fit it — a comfortable ratio; I would not be surprised to see the
high 0.6s here, possibly this rung's relative best. On Handwriting — 26 classes (chance ~0.04) of oscillatory
gestures where a linear boundary is too blunt and the discriminative residual is exactly what gets under-fit —
I expect the worst relative result, landing well above chance but clearly the weakest of the three datasets,
somewhere in the low 0.2s.

The falsifiable expectation for this rung, then, is not a single number but a profile: clearly above
chance everywhere (so the model is genuinely reading the window, not the floor), strongest in *relative*
terms on the binary FaceDetection, and weakest on the 26-class Handwriting where linear blunt-ness and the
ignored padding mask both bite. If a later rung that adds a learned nonlinear temporal encoder and a
mask-aware pooling beats this on Handwriting by a large margin while only matching it on FaceDetection,
that will tell me the leverage is in nonlinear temporal feature extraction and padding-awareness, not in
cross-channel attention — and that is exactly the diagnosis I want the floor to set up. The distilled module
and the literal `Custom.py` fill are in the answer.
