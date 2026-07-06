The decomposition-linear floor landed exactly where its nature predicted, and reading the three numbers
tells me precisely what to attack. EthanolConcentration 0.2890 — four classes, chance ~0.25, so the linear
map is barely above chance: it is reading *something* from the smooth spectral traces but the affine
boundary is too blunt to separate four concentration levels whose absorbance shapes overlap. Handwriting
0.2306 — 26 classes, chance ~0.04, so 0.23 is well above chance and the model is genuinely picking up the
gestures, but it is the weakest dataset in absolute distance from the others, exactly the case I flagged
where a linear boundary over a flattened window cannot carve 26 oscillatory regions. FaceDetection 0.6822
— binary, chance ~0.5, and this is the floor's relative best, which confirms the read that a linear
combination of ~144 channels over a window is plenty for a forgiving two-class problem. So the floor's
profile is the one I predicted: above chance everywhere, relatively strongest on the binary problem,
weakest on the many-class gesture problem. The diagnosis is therefore sharp. The flatten-and-project head
is linear, so the decision boundary is a hyperplane and the multi-class gesture geometry is unreachable;
and the model never forms a *local temporal feature* — it has no notion that a rising edge over a few steps
is a thing — it only has per-(channel, timestep) weights. Both of those are representation problems, not
optimization problems, and both point at the same fix: stop comparing and weighting raw timesteps, and
start from local sub-sequences.

That is the doubt that motivated starting at the linear floor in the first place, and now I get to act on
it. The standard deep recipe tokenizes per timestep — embed the whole channel-vector at each instant and
attend over `seq_len` tokens — and I argued at the floor that this is broken because a single timestamp has
no standalone meaning. The floor confirmed the negative half of that argument (a per-timestep linear map is
weak); now I want the positive half. What is the *token* supposed to be? In vision, a pixel is meaningless
and the answer was to cut the image into patches, each a local visual concept. The analogue is immediate:
cut each channel's length-`seq_len` series into contiguous sub-series patches, and let each patch be a
token. A patch of, say, sixteen consecutive timesteps is a little shape — a ramp, a bump, an oscillation —
which is exactly the object that carries discriminative content in a time series, and exactly what the
linear floor could not form. This is the move: change what a token is, and keep the attention kernel
vanilla, because the floor told me the leverage is in the representation, not the kernel.

Make it concrete. Take one channel's series of length `L = seq_len`. Pick a patch length `P = 16` and a
stride `S = 8`. Slide a window of width `P` along the series in steps of `S`; each placement is one patch,
a vector in `R^P`. I want the patching not to drop the end of the series, so before patching I
replication-pad the end by `S` so one more full window slides into existence that reaches the last
timestep. The patch count is then `N = floor((L − P)/S) + 2`. Two payoffs fall out. First, each token now
carries a local shape, so attention is finally comparing meaningful objects — "does this ramp resemble that
ramp `k` patches later?" — instead of comparing individual scalars. Second, the token count drops from `L`
to about `L/S`, so the `N × N` attention map shrinks by roughly `S²`; with `S = 8` that is a ~64× cut in
attention cost, which is why this is affordable even though I am about to stack three encoder layers with
sixteen heads. The stride smaller than the patch length (`8 < 16`) means consecutive patches overlap by
half, so no local shape gets cleanly split down the middle between two patches.

Let me put the patch counts on the actual datasets, because the whole affordability argument is numeric. With
`P = 16`, `S = 8`, `N = floor((L − 16)/8) + 2`. EthanolConcentration, `L ≈ 1,750`: `floor(1734/8) + 2 =
216 + 2 = 218` patches per channel. FaceDetection, `L ≈ 62`: `floor(46/8) + 2 = 5 + 2 = 7` patches.
Handwriting, `L ≈ 152`: `floor(136/8) + 2 = 17 + 2 = 19` patches. Now the attention-cost check that made me
comfortable stacking three 16-head layers: the map is `N × N` per head. On EthanolConcentration that is
218² ≈ 47,500 entries versus the per-timestep 1,750² ≈ 3.06M — a 64× reduction, exactly the `S² = 64` I
predicted. On FaceDetection 7² = 49 versus 62² = 3,844, ~78×; on Handwriting 19² = 361 versus 152² = 23,100,
~64×. So patching turns EthanolConcentration from a 3-million-entry attention problem into a 47-thousand-entry
one, which is the difference between "cannot stack this" and "cheap." But the FaceDetection number is also a
quiet warning: 7 patches is a *very* short token sequence, so on the dataset where I most want rich structure
there is almost nothing for attention to attend over — seven tokens is barely a sequence, and self-attention
over seven objects is close to a small fixed pooling. That reinforces my prior that FaceDetection's leverage
is not going to come from this rung's temporal encoder.

The patch length and stride are the two knobs that most shape this rung, so let me not set them on reflex.
There is a genuine tension: a longer patch makes each token a richer local shape but produces fewer tokens
(less for attention to relate), while a shorter patch gives more tokens but each carries less shape. On the
short datasets this bites hard. Suppose I picked `P = 24, S = 12` to capture broader shapes: FaceDetection's
`N = floor((62 − 24)/12) + 2 = floor(38/12) + 2 = 3 + 2 = 5` — only five tokens, and Handwriting drops to
`floor((152 − 24)/12) + 2 = 12`. Suppose instead I went fine with `P = 8, S = 4`: EthanolConcentration would
balloon to `floor((1750 − 8)/4) + 2 = 437` tokens, pushing the attention map to 437² ≈ 191k entries and the
head width to `128 · 437 · 3 ≈ 168k` inputs per example — 8× the cost, on the dataset that least needs fine
temporal resolution (its cue is a smooth curve). `P = 16, S = 8` is the compromise that keeps
EthanolConcentration's token count sane (218) while leaving FaceDetection and Handwriting with at least a
handful (7 and 19). The overlap ratio `S/P = 1/2` is the other deliberate choice: with stride equal to patch
length there would be no overlap and a discriminative shape straddling a patch boundary would be split across
two tokens with neither seeing it whole; half-overlap guarantees every length-8 sub-window appears intact
inside at least one patch. I keep `P = 16, S = 8`, aware that it is a single global setting the frozen
one-file constraint forces me to apply identically to a 62-step MEG window and a 1,750-step spectrum.

Now the second decision the floor forces me to confront: the channels. The linear floor mixed channels only
in its flatten head — a single linear combination — and it was the floor's relative *best* exactly on the
many-channel FaceDetection, which is a warning. The instinct is "model cross-channel correlation with
attention," but the linear-model evidence cuts the other way: the channel-*independent* linear map is what
is competitive with mixing Transformers on these benchmarks, and mixing models are observed to overfit on
the smaller datasets. So I will go channel-independent: run one shared Transformer backbone over each
channel's patch sequence separately, weights shared across channels, no cross-channel attention at all.
Three reasons, each of which I can trace. Adaptability: if I mixed channels into one token stream they would
all share a single attention pattern, but a spectral channel, an MEG channel, and an accelerometer axis
have completely different temporal behavior; processing each separately lets each form its own attention map.
Data efficiency: learning cross-channel interaction *jointly* with temporal structure is a far bigger
hypothesis space, and these UEA datasets are small (EthanolConcentration in particular has few training
examples), so the smaller temporal-only space converges with the data I have. Overfitting: mixing can fit
spurious cross-channel coincidences that hold in the training split and don't generalize. The cost of
channel-independence is nearly nothing in code — patch the batch `[B, enc_in, L]` into `[B, enc_in, N, P]`,
fold the channel axis into the batch to get `[B·enc_in, N, P]`, and the Transformer just sees a larger batch
of length-`N` token sequences; reshape back at the end.

The data-efficiency argument deserves a number, because it is the crux of choosing independence over mixing.
The encoder weights are shared across channels, so their count is fixed regardless of how many channels a
dataset has: per layer, attention is roughly `4 · d_model² = 4 · 128² ≈ 65.5k` (the Q, K, V, O projections)
and the feed-forward is `2 · d_model · d_ff = 2 · 128 · 256 ≈ 65.5k`, so ~131k per layer, ~393k over the three
layers, plus the patch embedding `Linear(16, 128) ≈ 2k`. Call it ~0.4M shared encoder parameters. Because
they are shared, folding channels into the batch means FaceDetection's 144 channels give the encoder 144× more
*training sequences* per batch (each channel is an independent length-`N` example) rather than 144× more
*parameters*. That is the whole point: channel-independence converts the many-channel datasets into an
effective data-augmentation for a small shared backbone, which is exactly what I want when EthanolConcentration
has only ~260 training series. A channel-mixing alternative — say, one token per timestep carrying all
`enc_in` values, embedded by `Linear(enc_in, d_model)` — would instead make the input dimension scale with
channel count (144 on FaceDetection) and force a single attention pattern to serve a spectral channel, an MEG
channel, and an accelerometer axis at once. So mixing costs more where data is scarcest and constrains
expressiveness where the channels are most heterogeneous. Independence it is.

There is a tension I have to be honest about, and it shapes my expectation for FaceDetection specifically.
FaceDetection's entire signal is cross-channel covariance over ~144 MEG channels. A channel-independent
model *by construction* cannot look at two channels together inside the encoder — it can only mix them in
the final head. So on the one dataset where cross-channel structure is the whole game, I am betting that the
shared per-channel temporal encoder plus a head that pools all channels is enough, and I should not expect a
large gain over the linear floor's 0.6822 there — the floor already mixed channels linearly in its head and
FaceDetection was forgiving. Where I *do* expect to gain is exactly where the floor was weakest: the
multi-class gesture and spectral problems, because those need local temporal shapes and a nonlinear boundary,
both of which patching plus a real encoder provide and the linear floor lacked.

The backbone is a vanilla Transformer encoder, deliberately. Project each `P`-vector patch to `d_model = 128`
with a bias-free linear embedding (a patch already normalized has its level removed, so a per-patch additive
offset buys nothing), add a positional embedding so the encoder knows patch order (attention is
permutation-invariant and patch order is everything — without it the encoder would treat a Handwriting
gesture's 19 patches as an unordered bag, and a stroke that goes up-then-down would be indistinguishable from
down-then-up, which is often the entire label; the positional embedding is not optional decoration here), then `e_layers = 3` of multi-head self-attention
(`n_heads = 16`) over the `N` patch tokens, each layer with a position-wise feed-forward `d_model → d_ff = 256
→ d_model` and residual connections. One non-default choice I will not make on autopilot: the normalization
inside the encoder. The default Transformer uses LayerNorm across the feature dimension per token, but time
series have outliers — a sensor glitch, a regime jump — and an outlier patch would skew its own per-token
statistics. BatchNorm instead normalizes each feature across the batch of patch positions, diluting a single
outlier among all the others; this is measured to beat LayerNorm for time-series Transformers, so the encoder
uses a BatchNorm-in-disguise (transpose, BatchNorm1d over `d_model`, transpose back). Concretely: if one patch
token embeds to a feature vector with a component of 50 while the typical value across the other ~217
EthanolConcentration patches (or across the batch of folded channel-sequences) is ~1, LayerNorm computes that
token's mean and variance from its own `d_model = 128` components, so the 50 inflates the token's own std and
squashes every *other* component of that same token toward zero — the glitch corrupts the whole token's
representation. BatchNorm computes the statistics of that one feature *across all the tokens in the batch*, so
the 50 is one entry among hundreds and barely moves the per-feature mean and std; the other tokens' values of
that feature pass through cleanly. On time series, where a single sensor spike is common, that difference is
the difference between one glitch poisoning a token and one glitch being averaged away — which is why I pay the
transpose cost to use BatchNorm here. And one input-side fix
the floor lacked: per-instance normalization. The channels and datasets span very different magnitudes — MEG
microvolts versus normalized absorbance versus acceleration — so before patching I subtract each window's
temporal mean and divide by its temporal standard deviation (biased variance, detached, with a `1e-5` floor
under the root), which decouples shape-learning from level-tracking and is what lets one shared backbone read
heterogeneous channels. (For classification I do not de-normalize an output trajectory — there is no
trajectory — so the normalization is purely an input conditioning.)

I have to be careful about this normalization, though, because on one dataset it may erase signal rather than
condition it. Trace it on EthanolConcentration: the discriminative cue there is the *absolute level and slow
shape* of the absorbance curve — different concentrations sit at different overall absorbance. Per-instance
normalization subtracts each window's temporal mean and divides by its temporal std, which is precisely the
operation that removes overall level and overall scale. So on the spectral dataset I am, by conditioning,
throwing away part of the thing the floor's *trend* stream captured directly. The floor mapped the raw trend
(level intact) with a dedicated linear map; here I subtract the level before the encoder ever sees it. That is
a concrete mechanism by which PatchTST could *lose* ground on EthanolConcentration relative to the floor, and
it is separate from patching. On Handwriting and FaceDetection the cue is shape and covariance, not absolute
level, so removing per-window level there is harmless-to-helpful. This asymmetry is why I will predict a
possible small regression on EthanolConcentration specifically while expecting gains elsewhere — the
normalization that helps the heterogeneous-magnitude datasets is the same one that flattens the spectral level
cue.

Now the head, and this is where the classification path departs from how a forecaster uses the same backbone.
A forecaster flattens each channel's `[d_model, N]` representation and projects to the horizon `T`, per
channel, keeping channels separate to emit a per-channel trajectory. I am not emitting a trajectory; I am
emitting one class decision for the whole multivariate window. So after the encoder gives me
`[B, enc_in, d_model, N]` (channels unfolded back out of the batch), I flatten *across both* the patch axis
and the channel axis — `head_nf = d_model · N` per channel, times `enc_in` channels — into one
`enc_in · d_model · N` vector, apply dropout, and project to `num_class` with a single linear layer. The
flatten across channels is the *only* place the channels meet, which is the channel-independence story made
literal: independent inside the encoder, joined once at the decision. This is also where FaceDetection's
cross-channel signal must be recovered if it is recovered at all — through that final linear join — so it is
the same crude channel-mixing the linear floor had, just sitting on top of a far richer per-channel temporal
representation.

The head width is worth computing, because it exposes where this rung spends its parameters and it is not
where the intelligence is. Per channel the flattened width is `head_nf = d_model · N = 128 · N`. On
EthanolConcentration `128 · 218 = 27,904`; times `enc_in = 3` gives 83,712 inputs, times `num_class = 4` is
~335k weights in the final linear. On FaceDetection `128 · 7 = 896`, times 144 channels = 129,024 inputs,
times 2 classes = ~258k. On Handwriting `128 · 19 = 2,432`, times 3 = 7,296, times 26 = ~190k. So on every
dataset the final flatten-and-project (~0.2–0.34M weights) is comparable in size to the entire shared
encoder (~0.4M) — a large fraction of the model is again a single linear read-off, just as in the floor. The
difference from the floor is *what* it reads: the floor's head read raw (channel, timestep) values, whereas
this head reads `d_model`-dimensional learned patch features. That is the intended upgrade — a richer input
to the same crude linear decision — and it is exactly why I expect gains where local-shape features matter
(Handwriting, EthanolConcentration) and near-parity where the bottleneck is the crude cross-channel join
(FaceDetection), since the join itself is no better than the floor's.

One thing I will carry forward as a known gap, because the next rung should attack it: like the floor, this
PatchTST classification path does *not* consult the padding mask `x_mark_enc`. The patches over the
right-padded region are patches of replicated/zero tail, embedded and attended like any other, and the
flatten head learns weights for them. On datasets where the within-dataset window length varies, that is
wasted capacity and a possible spurious cue. I am choosing not to fix it here because the whole point of this
rung is to test patching-plus-channel-independence cleanly against the linear floor; folding in mask-aware
pooling now would confound the two changes.

So the falsifiable expectations against the floor's numbers. On Handwriting (floor 0.2306): this is where I
expect the clearest gain, because local-shape tokens and a nonlinear encoder are exactly the missing
ingredients for 26 oscillatory gesture classes — I expect a meaningful lift above 0.23, into the mid-to-high
0.2s. On EthanolConcentration (floor 0.2890): a model that forms local spectral shapes and has a nonlinear
boundary should edge past the blunt linear map, but the dataset is small and the four classes overlap, so I
expect at best a small move, landing near or just below the floor — and if it lands *below*, that will tell
me patching's overlap is washing out the slow global trend that the decomposition-linear captured directly,
a real possibility on smooth spectra. On FaceDetection (floor 0.6822): because channel-independence forbids
in-encoder cross-channel modeling on the one dataset that is all cross-channel, I expect at best parity with
the floor, around 0.68 — possibly a hair above from the richer per-channel features, but I would not bet on a
real gain, and a result at or just above 0.68 is the honest expectation. The seven-token attention there is so
short it is barely more than a learned pooling over patches, so almost all of FaceDetection's decision still
rests on the final cross-channel linear join — the same crude join the floor had — which is precisely why I
expect parity rather than a jump. The single sharpest test is
Handwriting: if patching-plus-encoder does *not* clearly beat the linear floor there, then local temporal
shapes are not the bottleneck and I have misdiagnosed the task. The distilled module and the literal
`Custom.py` fill are in the answer.
