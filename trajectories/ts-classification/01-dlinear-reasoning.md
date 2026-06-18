The scaffold default returns zeros, which means every window gets the same logit vector and the
classifier predicts the majority class — chance, modulated by class imbalance. That is the floor, and
the floor exists to anchor one question: how much accuracy is available from the *cheapest* honest model
that actually reads the window? I want the weakest rung to be a model I cannot accuse of cleverness, so
that whatever it scores is a clean lower bound on "a real encoder helps." If an elaborate temporal
encoder later beats it by a wide margin, the encoder earned its keep; if it barely beats it, the encoder
was mostly decoration. So I deliberately start from the most minimal thing that can map a multivariate
window to class logits, and I let the discomfort that motivates the whole ladder pick which minimal
thing.

The discomfort is concrete. The standard deep recipe for multivariate time-series classification embeds
the whole channel-vector at each timestamp into one token and runs self-attention over the `seq_len`
tokens. But a single timestamp of a load trace or an MEG sweep has no standalone meaning — unlike a word,
a scalar at time *t* is nothing in isolation; the signal lives in the *shape over a stretch* of time. So
point-wise attention is scoring pairwise similarity between objects that individually carry no semantics,
and on top of that it pays a quadratic cost in `seq_len`. The fact that nags is that a plain linear map
over the flattened window is reported to be competitive with these encoders. If a linear map is
competitive, then either attention is the wrong tool here, or the leverage is in the *input
representation*, not the kernel. I want the weakest rung to be exactly that linear map — both because it
is the right floor and because it operationalizes the doubt: I am measuring how much of the achievable
accuracy a one-layer affine map already captures, before any temporal machinery.

So what is the minimal model? Read the window `x_enc` of shape `[B, seq_len, enc_in]`, and produce
`[B, num_class]`. The absolute floor is: flatten the window into a single `enc_in · seq_len` vector and
apply one linear layer to `num_class`. That is a logistic regression on the raw flattened window. It
treats every (channel, timestep) pair as an independent feature and learns one weight per pair per class.
It has no notion of temporal order beyond what the weights memorize, no cross-channel modeling beyond the
shared linear combination, and no nonlinearity. It is the honest floor.

But I will not ship the *bare* flatten-and-project, because there is a free, parameter-light refinement
that costs nothing in capacity and that I expect to matter on at least one of these datasets — and using
it keeps this rung aligned with the linear-model lineage that motivated starting here. The refinement is
seasonal-trend decomposition. The idea, the oldest move in time-series analysis: write the series
additively as a slow trend (a moving average) plus a seasonal residual (what is left after subtracting
the trend), because each piece on its own is more regular than the sum. In a classifier this matters
because the discriminative cue can live in either piece. EthanolConcentration is the clearest case: the
spectral absorbance traces are long, smooth curves whose *level and slow shape* carry the concentration
information — that is trend. Handwriting's accelerometer gestures, by contrast, are about the *oscillatory
residual* once the slow drift of the hand is removed — that is seasonal. A single flatten-and-project must
fit both kinds of cue with one weight matrix, and the large-magnitude trend dominates the fit. Splitting
the window into `trend = MovingAvg(x)` and `seasonal = x − trend`, mapping each with its own linear layer,
and summing the two streams lets each specialize. Crucially this adds *no representational capacity*: a
moving average is linear, two linear maps plus a sum is still affine end to end. It is preconditioning,
not depth — it separates the loud component from the quiet one so the optimizer can fit each, exactly the
move that distinguishes this rung from the bare linear map. This is the model I commit to for rung one.

Now the mechanics, because the details decide whether it behaves. The moving average must preserve length
so I can subtract it from the input. I use an `AvgPool1d` with an odd kernel `k = 25` (the same smoothing
scale the decomposition line uses, so the comparison stays clean), stride 1, and I replicate-pad the two
endpoints by `(k−1)//2` each. Replication matters: zero-padding would drag the trend toward zero at the
window edges, inventing spurious dips exactly where I have least information; replicating the endpoint
keeps the trend flat-but-faithful there. With `k` odd and `(k−1)//2` on each side, the pooled output has
exactly the input length. The seasonal part is then `x − trend`, and the two streams have shape
`[B, seq_len, enc_in]`.

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
class through dataset quirks). The decomposition-linear classification path does not consult the mask —
that is a real limitation I am inheriting, and a later rung that *does* zero out padding before pooling
should gain on the variable-length datasets. Second: it models cross-channel interaction only through the
final flatten-and-project. There is no mechanism for the model to learn that two MEG channels covary;
their covariation can only be expressed as a fixed linear combination in the head. FaceDetection, whose
entire signal is cross-channel covariance over ~144 noisy channels, is the dataset I expect this to hurt
most. Third: it is linear, so any class boundary that is not (after decomposition) a hyperplane in the
flattened-window space is unreachable. Handwriting's 26 classes of oscillatory gestures are the case where
I expect the linear boundary to be too blunt.

So I can already predict the *shape* of where this rung lands, which is the point of a floor. On
EthanolConcentration — smooth spectral traces where the slow trend carries the label and four
well-separated classes — the decomposition should give the linear head a fair shot, and I expect a modest
but real lift over chance (chance with four classes is ~0.25; I expect to land somewhere in the high
0.2s, only a little above chance, because the within-class shape variation in spectra is large for a
purely linear map). On FaceDetection — binary, so chance is ~0.5, and the signal is cross-channel
covariance the flatten-head can only crudely capture — I expect the linear map to do *better than its
nature suggests* simply because binary problems are forgiving and a linear combination of 144 channels
over a window has plenty of features; I would not be surprised to see the high 0.6s here, possibly this
rung's relative best. On Handwriting — 26 classes (chance ~0.04) of oscillatory gestures where a linear
boundary is too blunt and the discriminative residual is exactly what gets under-fit — I expect the worst
relative result, landing well above chance but clearly the weakest of the three datasets, somewhere in the
low 0.2s.

The falsifiable expectation for this rung, then, is not a single number but a profile: clearly above
chance everywhere (so the model is genuinely reading the window, not the floor), strongest in *relative*
terms on the binary FaceDetection, and weakest on the 26-class Handwriting where linear blunt-ness and the
ignored padding mask both bite. If a later rung that adds a learned nonlinear temporal encoder and a
mask-aware pooling beats this on Handwriting by a large margin while only matching it on FaceDetection,
that will tell me the leverage is in nonlinear temporal feature extraction and padding-awareness, not in
cross-channel attention — and that is exactly the diagnosis I want the floor to set up. The distilled
module and the literal `Custom.py` fill are in the answer.
