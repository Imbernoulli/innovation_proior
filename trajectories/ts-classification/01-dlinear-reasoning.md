The scaffold default returns zeros, so every window gets the same logit vector and the classifier predicts
the majority class — chance, modulated by class imbalance. That floor is not one number here. FaceDetection
is binary and roughly balanced, so the zero-logit model sits near 0.50. EthanolConcentration has four
classes, so a balanced majority guess is around 0.25. Handwriting has twenty-six, so blind guessing is near
1/26 ≈ 0.038. Those three anchors already say the datasets live in different regimes: on FaceDetection a
model has to beat a coin, on Handwriting it has to beat one-in-twenty-six. What I want from the weakest rung
is a clean lower bound on "a real encoder helps" — the cheapest honest model that actually reads the window,
one I cannot accuse of cleverness. If an elaborate temporal encoder later beats it by a wide margin, the
encoder earned its keep; if it barely beats it, the encoder was decoration. So I start from the most minimal
thing that maps a multivariate window to class logits.

The discomfort that picks *which* minimal thing is concrete. The standard deep recipe for multivariate
time-series classification embeds the whole channel-vector at each timestamp into one token and runs
self-attention over the `seq_len` tokens. But a single timestamp of a load trace or an MEG sweep has no
standalone meaning — unlike a word, a scalar at time *t* is nothing in isolation; the signal lives in the
*shape over a stretch* of time. So point-wise attention scores pairwise similarity between objects that
individually carry no semantics, and it pays a quadratic cost in `seq_len`. On the dataset that punishes
that most — EthanolConcentration, whose absorbance traces pad to a maximum length on the order of 1,750 —
a per-timestep attention map is roughly 1,750² ≈ 3.06 million entries per head per layer, for series whose
discriminative content is a slow smooth curve a handful of numbers would summarize. And a plain linear map
over the flattened window is reported competitive with these encoders. If a linear map is competitive, then
either attention is the wrong tool here or the leverage is in the *input representation*, not the kernel. I
want the weakest rung to be exactly that linear map — it is both the right floor and an operationalization
of the doubt: I am measuring how much achievable accuracy a one-layer affine map already captures before any
temporal machinery.

The absolute floor is: flatten the window `[B, seq_len, enc_in]` into one `enc_in · seq_len` vector and
apply a single linear layer to `num_class`. That is logistic regression on the raw flattened window — every
(channel, timestep) pair an independent feature, no temporal order beyond what weights memorize, no
cross-channel modeling beyond a shared linear combination, no nonlinearity. The honest floor.

Before settling I price the small space of "minimal" models, because the choice is arithmetic, not
aesthetic. The bare flatten-and-project costs `enc_in · seq_len · num_class`: on EthanolConcentration
3 · 1,750 · 4 ≈ 21,000, on FaceDetection 144 · 62 · 2 ≈ 18,000, on Handwriting 3 · 152 · 26 ≈ 12,000 — all
small, all trainable on these tiny UEA splits (EthanolConcentration has only ~260 training series). Adding a
hidden nonlinear layer explodes it: `Linear(enc_in · seq_len, 128)` alone is 3 · 1,750 · 128 ≈ 672,000
weights against 260 examples, a 2,600:1 ratio that will memorize the split — and a nonlinear boundary is
exactly what I want later rungs to earn, so buying it now would blur the measurement. Reject.

What I ship instead is not the *bare* projection but a seasonal-trend decomposition in front of it, because
decomposition is a free, parameter-light refinement that costs nothing in capacity and keeps this rung
aligned with the linear-model lineage that motivated starting here. The idea is the oldest move in
time-series analysis: write the series additively as a slow trend (a moving average) plus a seasonal
residual (what is left after subtracting the trend), because each piece is more regular than the sum. The
discriminative cue can live in either piece. EthanolConcentration is the clearest case: the spectral traces
are long smooth curves whose *level and slow shape* carry the concentration — that is trend. Handwriting's
accelerometer gestures are about the *oscillatory residual* once the slow drift of the hand is removed —
that is seasonal. A single flatten-and-project must fit both cues with one weight matrix, and the failure
mode is concrete: the trend of a spectral trace has magnitude on the order of the raw absorbance, while the
seasonal residual after subtracting a 25-wide moving average is a wiggle an order of magnitude smaller.
Under a cross-entropy fit the loud-component gradients dominate the weight updates, so the quiet seasonal
cue is under-fit — the optimizer spends capacity where the energy is, not where the label is. Splitting the
window into `trend = MovingAvg(x)` and `seasonal = x − trend`, mapping each with its own linear layer, and
summing the two streams lets each specialize. This adds *no representational capacity*: `trend + seasonal =
MovingAvg(x) + (x − MovingAvg(x)) = x` exactly, so two linear maps plus a sum is still affine end to end. It
is preconditioning, not depth — it re-bases the two maps so each sees a component of homogeneous magnitude
and the optimizer can fit both. This is the model I commit to.

The one place this rung could quietly overfit is the temporal maps, which are the parameter-dominant part:
two `Linear(seq_len, seq_len)`. Shared across channels, that is 2 · seq_len² weights, *independent of
channel count* — on EthanolConcentration 2 · 1,750² ≈ 6.13 million against ~260 series, a 23,000:1 ratio,
and this is the *cheap* option. The individual variant (one temporal map per channel) multiplies by
`enc_in`: on FaceDetection's 144 channels it turns 2 · 62² ≈ 7,700 into ~1.1 million — a 144× blow-up on the
very dataset with the most channels, exactly the recipe for fitting per-sensor noise. Individual maps cost
the most where they help the least. Share the weights. The head is where I let channels differ: a single
`Linear(enc_in · seq_len, num_class)`, the 21k / 18k / 12k projection priced above. The imbalance is
striking — on EthanolConcentration the 6.13M temporal weights dwarf the 21k head by ~290×, so almost the
whole model is shared per-channel smoothing of time and the class decision is a thin linear read-off. That
is a warning I carry: this rung's capacity goes to temporal reshaping, not the decision boundary, and on the
long-window dataset it is wildly over-parameterized and will lean hard on early stopping (patience 10) to
avoid memorizing.

The moving-average mechanics decide whether this behaves. It must preserve length so I can subtract it: an
`AvgPool1d` with odd kernel `k = 25`, stride 1, replicate-padding `(k−1)//2 = 12` per endpoint — input `L`,
padded to `L + 24`, a width-25 stride-1 average yields `(L + 24) − 25 + 1 = L`. Replication matters
concretely: zero-padding would drag the trend toward zero at the edges (at the first timestep the "average"
would be roughly half the true local level, inventing a spurious dip exactly where I have least
information), whereas replicating the endpoint keeps the trend flat-but-faithful there. The seasonal part is
`x − trend`, both streams `[B, seq_len, enc_in]`.

The kernel `k = 25` is itself a knob I should not set on autopilot, because it decides what counts as trend
versus seasonal and the split it induces is dataset-dependent — the windows differ wildly in length. On
EthanolConcentration's ~1,750 steps, 25 smooths over ~1.4% of the window, removing fine spectral noise and
leaving the broad absorbance shape as trend — what I want there. On FaceDetection's ~62 steps, 25 is ~40% of
the window, so almost everything falls into the seasonal residual — fine, because on MEG the fast
fluctuation *is* the signal. On Handwriting's ~150 steps, 25 is ~17%, in between. A single fixed kernel does
something reasonable-but-different on each, which is acceptable for a floor: I am not tuning the split per
dataset, only giving the optimizer two magnitude-separated streams. A much larger kernel would collapse the
decomposition back toward the bare linear map on the short datasets; a much smaller one would leave
high-frequency noise in the trend on EthanolConcentration. I keep 25, noting the split is coarse and not
adapted — something a later period-aware rung could improve.

The two temporal maps are initialized to `(1/seq_len) · ones`, i.e. every output position is the *mean* of
its inputs. So at init `Linear_Trend(trend)` outputs, at every position, the temporal mean of the trend
(≈ the mean of the raw window, since the moving average preserves the mean), and `Linear_Seasonal(seasonal)`
outputs the temporal mean of a residual that sums to ≈ 0. The summed encoder output at init is therefore a
constant-per-channel signal equal to each channel's window mean, and the head sees, per channel, its mean
level repeated. The model *starts* as "classify by per-channel average level" and learns from there which
timesteps and components matter — a sane, interpretable starting point, not a pathological regime.

All of this is under a frozen substrate with one `Custom.py` for all three datasets: I cannot special-case
EthanolConcentration's long window or FaceDetection's 144 channels, and RAdam at lr 1e-3, cross-entropy, and
early stopping at patience 10 are not mine to touch. That matters most for the over-parameterized case —
EthanolConcentration's 6.13M temporal weights on ~260 series are entirely at the mercy of early stopping,
with no weight-decay knob to reach for. So the honest reading is that this rung's EthanolConcentration
behavior is as much a statement about the protocol's implicit regularization as about the decomposition. On
the smaller-parameter datasets the ratios are far tamer and early stopping does less work.

The head is where the classification path departs from a forecaster's. A forecaster maps `L → T` and keeps
the channel axis separate to emit a per-channel trajectory. I emit class logits, so after decomposing I
flatten *everything* — time and channel — into one `enc_in · seq_len` vector and project to `num_class`. The
two streams each run a `Linear(seq_len, seq_len)` along time (square, since I keep `pred_len = seq_len`),
are summed, and the `[B, seq_len, enc_in]` result is reshaped and fed to `Linear(enc_in · seq_len,
num_class)`. That final projection is where channels are mixed and the class decision is made; everything
before it is the decomposed temporal encoder, and the whole thing is affine input-to-logits.

What this rung throws away is exactly what later rungs will exploit. First, it ignores the padding mask
`x_mark_enc`: the flattened projection sees padded positions as ordinary zeros and learns weights for them.
Handwriting's series vary in length and get right-padded to the per-dataset maximum, so some fraction of
each flattened window is a constant zero tail the head learns a column of weights against — wasted capacity,
and potentially a spurious cue if padding length correlates with class. A later rung that zeros padding
before pooling should gain on the variable-length datasets. Second, cross-channel interaction is only the
final flatten-and-project — a fixed linear combination, no mechanism to learn that two MEG channels covary.
FaceDetection, whose entire signal is cross-channel covariance over ~144 noisy channels, is where I expect
that to hurt most. Third, it is linear, so any class boundary that is not a hyperplane in flattened-window
space is unreachable — Handwriting's 26 oscillatory-gesture classes are where I expect the linear boundary
to be too blunt.

So I can predict the *shape* of where this lands, which is the point of a floor. It should be clearly above
chance everywhere — genuinely reading the window, not the floor. In *relative* terms I expect the binary
FaceDetection to be its best: a forgiving two-class problem where a linear combination of ~144 channels over
a 62-step window has ~18,000 features and several thousand series to fit a single hyperplane, a comfortable
ratio. EthanolConcentration should show a modest lift over its four-class chance — the decomposition gives
the linear head a fair shot at the smooth trend, but within-class spectral variation is large for an affine
map and the ~23,000:1 parameter ratio means early stopping halts before the four overlapping concentration
levels are carved cleanly. Handwriting I expect to be the weakest of the three in absolute distance: well
above its 1/26 chance, but the linear boundary is too blunt for 26 oscillatory regions and the under-fit
seasonal residual is exactly the discriminative part. If a later rung with a nonlinear temporal encoder and
mask-aware pooling beats this on Handwriting by a wide margin while only matching it on FaceDetection, the
leverage is nonlinear temporal feature extraction and padding-awareness, not cross-channel attention — the
diagnosis I want the floor to set up.
