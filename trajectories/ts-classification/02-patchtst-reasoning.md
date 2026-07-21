The decomposition-linear floor landed where its nature predicted, and the three numbers say what to attack.
EthanolConcentration 0.2890 — four classes, chance ~0.25, so the affine boundary is barely above chance:
reading *something* from the smooth spectra but too blunt to separate four overlapping concentration levels.
Handwriting 0.2306 — 26 classes, chance ~0.04, well above chance but the weakest dataset in absolute terms,
exactly the case I flagged where a linear boundary over a flattened window cannot carve 26 oscillatory
regions. FaceDetection 0.6822 — binary, chance ~0.5, the floor's relative best, confirming that a linear
combination of ~144 channels is plenty for a forgiving two-class problem. So the profile is the one I
predicted: above chance everywhere, strongest on the binary problem, weakest on the many-class gesture
problem. The diagnosis is sharp: the flatten-and-project head is a hyperplane so the multi-class gesture
geometry is unreachable, and the model never forms a *local temporal feature* — it has no notion that a
rising edge over a few steps is a thing, only per-(channel, timestep) weights. Both are representation
problems, not optimization problems, and both point at the same fix: stop weighting raw timesteps, start
from local sub-sequences.

That is the doubt that motivated the linear floor, and now I act on it. The floor confirmed the negative
half of my argument (a per-timestep linear map is weak); I want the positive half. What is the *token*
supposed to be? In vision a pixel is meaningless and the answer was to cut the image into patches, each a
local visual concept. The analogue is immediate: cut each channel's length-`seq_len` series into contiguous
sub-series patches, one token each. A patch of ~sixteen consecutive timesteps is a little shape — a ramp, a
bump, an oscillation — exactly the object that carries discriminative content in a time series and exactly
what the linear floor could not form. Change what a token is, and keep the attention kernel vanilla, because
the floor said the leverage is in the representation.

Concretely: take one channel's series of length `L`, pick patch length `P = 16`, stride `S = 8`, slide a
width-`P` window in steps of `S`; each placement is a vector in `R^P`. To avoid dropping the end, I
replication-pad the tail by `S` so one more full window reaches the last timestep, giving `N = floor((L −
P)/S) + 2` patches. Two payoffs: each token now carries a local shape, so attention compares meaningful
objects ("does this ramp resemble that ramp `k` patches later?"); and the token count drops from `L` to
about `L/S`, so the `N × N` attention map shrinks by roughly `S² = 64`. Stride below patch length (`8 < 16`)
means consecutive patches overlap by half, so no local shape gets cleanly split down the middle.

Put the patch counts on the datasets, because the affordability argument is numeric. EthanolConcentration,
`L ≈ 1,750`: `floor(1734/8) + 2 = 218` patches, turning a 1,750² ≈ 3.06M-entry attention problem into
218² ≈ 47,500 — the difference between "cannot stack this" and "cheap," and exactly the 64× I predicted.
FaceDetection, `L ≈ 62`: 7 patches. Handwriting, `L ≈ 152`: 19 patches. The FaceDetection number is a quiet
warning: 7 patches is barely a sequence, and self-attention over seven objects is close to a small fixed
pooling — so on the dataset where I most want rich structure there is almost nothing for attention to
attend over. That reinforces my prior that FaceDetection's leverage will not come from this rung's temporal
encoder.

Patch length and stride are the two knobs that most shape this rung, and there is a genuine tension: a
longer patch makes each token richer but produces fewer tokens, a shorter patch gives more tokens each
carrying less shape. On the short datasets it bites. `P = 24, S = 12` would give FaceDetection only 5 tokens
and Handwriting 12; `P = 8, S = 4` would balloon EthanolConcentration to 437 tokens (attention map 437² ≈
191k, head width `128 · 437 · 3 ≈ 168k`), 8× the cost on the dataset that least needs fine resolution. `P =
16, S = 8` keeps EthanolConcentration sane (218) while leaving the others a handful (7 and 19), with the
half-overlap guaranteeing every length-8 sub-window appears intact in some patch. I keep it, aware it is a
single global setting the one-file constraint forces onto a 62-step MEG window and a 1,750-step spectrum
alike.

Now the channels. The linear floor mixed them only in its flatten head and was its relative *best* on the
many-channel FaceDetection — a warning against elaborate mixing. The instinct is "model cross-channel
correlation with attention," but the evidence cuts the other way: the channel-*independent* linear map is
what is competitive with mixing Transformers on these benchmarks, and mixing models are observed to overfit
the smaller sets. So I go channel-independent: one shared Transformer backbone over each channel's patch
sequence separately, weights shared, no cross-channel attention. Three reasons. Adaptability: a spectral
channel, an MEG channel, and an accelerometer axis have completely different temporal behavior, and
processing each separately lets each form its own attention map rather than share one. Data efficiency:
learning cross-channel interaction *jointly* with temporal structure is a far bigger hypothesis space, and
these UEA sets are small, so the temporal-only space converges with the data I have. Overfitting: mixing can
fit spurious cross-channel coincidences that hold only in the train split. The cost in code is nearly
nothing — patch `[B, enc_in, L]` into `[B, enc_in, N, P]`, fold the channel axis into the batch to get
`[B·enc_in, N, P]`, and the Transformer just sees a larger batch of length-`N` sequences.

The data-efficiency argument has a number behind it. The shared encoder's parameter count is fixed
regardless of channel count: per layer ~`4 · 128² ≈ 65.5k` attention plus ~`2 · 128 · 256 ≈ 65.5k`
feed-forward, ~131k per layer, ~0.4M over three layers plus the patch embedding. Because they are shared,
folding channels into the batch means FaceDetection's 144 channels give the encoder 144× more training
*sequences* rather than 144× more *parameters* — channel-independence turns the many-channel datasets into
data augmentation for a small shared backbone, which is what I want when EthanolConcentration has only ~260
series. A channel-mixing alternative (one token per timestep carrying all `enc_in` values) would instead
make the input dimension scale with channel count and force one attention pattern to serve all channel
types. Mixing costs more where data is scarcest and constrains expressiveness where channels are most
heterogeneous. Independence it is — with the honest caveat that FaceDetection's entire signal is
cross-channel covariance, and a channel-independent model can only mix in the final head, so I should not
expect a large gain there over the floor's 0.6822.

The backbone is a vanilla Transformer encoder: project each `P`-vector patch to `d_model = 128` with a
bias-free linear embedding (a patch's level is removed, so a per-patch offset buys nothing), add a
positional embedding — not optional, since attention is permutation-invariant and patch order is everything:
without it a Handwriting gesture's 19 patches are an unordered bag and a stroke that goes up-then-down is
indistinguishable from down-then-up, often the whole label — then `e_layers = 3` of `n_heads = 16`
self-attention over the `N` tokens with a position-wise `d_model → 256 → d_model` feed-forward and
residuals. One non-default choice: the encoder normalizes with BatchNorm, not LayerNorm. Time series carry
outliers — a sensor glitch, a regime jump — and LayerNorm normalizes within a token, so an outlier landing
in one token skews that token's own statistics and squashes its other components; BatchNorm normalizes each
feature across the batch of patch positions, diluting a single outlier among all the others. On time series
that difference is measured to matter, so I pay the transpose cost (transpose, BatchNorm1d over `d_model`,
transpose back). And one input-side fix the floor lacked: per-instance normalization. The channels span very
different magnitudes — MEG microvolts, normalized absorbance, acceleration — so before patching I subtract
each window's temporal mean and divide by its temporal std (biased, detached, `1e-5` floor), decoupling
shape-learning from level-tracking so one backbone can read heterogeneous channels. (No de-normalization —
there is no output trajectory; the normalization is purely input conditioning.)

That normalization is exactly where this rung may *lose* ground, and only on one dataset. On
EthanolConcentration the cue is the *absolute level and slow shape* of the absorbance curve — different
concentrations sit at different overall absorbance — and subtracting each window's mean and dividing by its
std is precisely the operation that removes overall level and scale. The floor mapped the raw trend (level
intact) with a dedicated linear map; here I strip the level before the encoder sees it. On Handwriting and
FaceDetection the cue is shape and covariance, not absolute level, so removing per-window level there is
harmless-to-helpful. This asymmetry is why I expect a possible small regression on EthanolConcentration
while expecting gains elsewhere — the conditioning that helps the heterogeneous-magnitude datasets is the
same one that flattens the spectral level cue.

The head departs from a forecaster's again. A forecaster flattens each channel's `[d_model, N]` and projects
to a horizon per channel, keeping channels separate. I emit one class decision for the whole window, so
after the encoder returns `[B, enc_in, d_model, N]` I flatten *across both* patch and channel axes into one
`enc_in · d_model · N` vector, apply dropout, and project to `num_class` with a single linear layer. The
flatten across channels is the *only* place channels meet — channel-independence made literal: independent
inside the encoder, joined once at the decision. That join is the same crude channel-mixing the floor had,
now sitting on top of a far richer per-channel temporal representation, and it is where FaceDetection's
cross-channel signal must be recovered if at all. The head width confirms most of the model is again a
single linear read-off: per channel `head_nf = 128 · N` (27,904 on EthanolConcentration, 896 on
FaceDetection, 2,432 on Handwriting), and the final linear runs ~0.2–0.34M weights per dataset, comparable
to the ~0.4M encoder. The upgrade over the floor is *what* it reads — `d_model`-dimensional learned patch
features instead of raw (channel, timestep) values — which is exactly why I expect gains where local-shape
features matter and near-parity where the bottleneck is the crude cross-channel join.

Like the floor, this path does *not* consult the padding mask: patches over the right-padded tail are
embedded and attended like any other, wasted capacity and a possible spurious cue on variable-length
datasets. I leave it unfixed here on purpose — the point of this rung is to test patching-plus-channel-
independence cleanly, and folding in mask-aware pooling now would confound the two changes. The next rung
attacks it.

So the expectations against the floor. On Handwriting (floor 0.2306) I expect the clearest gain, into the
mid-to-high 0.2s: local-shape tokens and a nonlinear encoder are exactly the missing ingredients for 26
oscillatory classes — this is the sharpest test, and if patching does *not* clearly beat the floor here then
local temporal shapes are not the bottleneck and I have misdiagnosed the task. On EthanolConcentration
(floor 0.2890) I expect at best a small move, landing near or just below the floor: forming local spectral
shapes with a nonlinear boundary should help, but the per-window normalization washes out the slow global
trend the decomposition captured directly, and a result below the floor would confirm that mechanism. On
FaceDetection (floor 0.6822) I expect parity, around 0.68: channel-independence forbids in-encoder
cross-channel modeling on the one dataset that is all cross-channel, and with only seven tokens the
attention is barely more than a learned pooling, so almost the entire decision still rests on the same crude
cross-channel join the floor had.
