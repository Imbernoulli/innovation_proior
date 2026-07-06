The linear control came back, and I read its numbers carefully before deciding what to build next. On
ETTh1 it landed MSE 0.3962 / MAE 0.4108 — exactly the "genuinely competitive" outcome I predicted: a
single affine map over a decomposed window is already in the high 0.30s on the small, strongly
trend-and-seasonal dataset, and any attention model that cannot clearly beat that there has no business
claiming the temporal-attention machinery was load-bearing. So the embarrassment the control was built
to deliver, it delivered. But the other two datasets tell the more useful story. On Weather it is 0.1962
/ 0.2561 and on ECL 0.2104 / 0.3016, for a mean MSE of 0.2676. ECL is the tell: 321 channels, and the
deliberately channel-blind, channel-shared linear map leaves the most on the table exactly there, just
as I expected. The sharpest way to see it is not the raw MAE but the MAE-over-MSE gap: on ECL 0.3016 sits
a full 43% above its own MSE of 0.2104, versus 0.2561/0.1962 ≈ 31% on Weather and 0.4108/0.3962 ≈ 4% on
ETTh1. A high MAE relative to MSE means the errors are many and moderate rather than a few large ones —
the fingerprint of a systematic bias spread across predictions, which is exactly what ignoring 321
coupled channels would produce: every client is nudged the same wrong way because the shared row cannot
read the regional signal the others carry. ETTh1's near-parity of MAE and MSE, by contrast, says its
residual is dominated by a few hard windows, not a pervasive channel bias. So the diagnosis splits
cleanly. The temporal modeling is *fine* — the linear map already captures the forecastable trend and
periodicity. What is missing has two candidate names: richer temporal structure, or cross-channel
structure. Before I reach for cross-channel attention I want to settle whether attention over *time*,
done right, can extract more than the linear map did, because if it cannot, then the heavy temporal
Transformers really were a dead end and the only remaining lever is channels.

So I refuse to touch the attention kernel and ask the question the linear result reframes: the premise
of the whole Informer/Autoformer/FEDformer line was "attention is too expensive, make it cheaper." But a
plain linear projection matched or beat all of them. That means either self-attention is the wrong tool
for this data, or — the possibility I want to chase — attention is fine and we have been feeding the
series into it in a way that destroys exactly the structure attention is good at finding. What is the
*token*, and is the token the bug?

In every one of those models, and in the scaffold's per-step convention, a token is a single time step:
one scalar at time t for a univariate series, or the full channel vector at time t for a multivariate
one. Sit with that. In language a token is a word — a standalone semantic unit, and attention between
two words compares two meanings. What is the meaning of a sensor's value at 14:03? Nothing, in
isolation — less than a character, which at least belongs to a small alphabet. The information in a
series lives in *shapes over short stretches*: a rising edge, a dip, a local oscillation, the slope of a
ramp. Point-wise attention asks "how does the scalar at 14:03 relate to the scalar at 09:17?" and the
answer is almost always noise, because neither scalar means anything alone. The attention map is
computed over the wrong objects. That is exactly why the linear model won the temporal comparison: it
never does point-wise comparison at all — it reads the whole window at once through learned weights, so
it sees the *shape*. The Transformers' temporal failure may be a tokenization failure, not an attention
failure. That is the hypothesis the ECL gap does *not* yet adjudicate, so I test it directly.

There are three genuinely different ways I could try to extract more temporal structure than the linear
map did, and I want to weigh them before committing. The first is to stay linear but go wider — stack the
decomposed linear maps into an MLP-Mixer over time, alternating time-mixing and feature-mixing layers.
That buys nonlinearity but keeps the operator token-agnostic and adds no mechanism to compare local
shapes at a distance; it is a plausible incremental gain, not a test of whether attention is salvageable,
so it does not answer the question I actually posed. The second is the lineage's own move: keep per-step
tokens and cheapen the L×L attention with sparsity or frequency tricks so I can afford a longer window.
But this accepts the tokenization I just diagnosed as the bug and spends all its ingenuity working around
the quadratic cost of comparing meaningless scalars — it is optimizing the wrong object faster, and the
linear result says that object was never worth computing. The third is to change what a token is so that
attention compares meaningful units, leaving the kernel stock. Only the third both keeps attention (so a
win is attributable to attention done right) and removes the diagnosed defect, so that is the one I test
directly.

A second symptom points the same way. Token count N equals sequence length L when every step is a token,
so the attention map is L×L and lengthening the look-back is quadratically punished — which is why the
field defaults to short windows and throws away older history that, on these datasets, genuinely helps.
And people who kept a long window cheaply by *down-sampling* — take every fourth step — still forecast
well, sometimes better. Read carefully, that says the time axis is *redundant*: neighboring steps carry
overlapping, compressible structure, not independent information. So the move is staring at me. If
neighboring steps are redundant and a single step is meaningless, do not tokenize per step — group a
local stretch into one token. This is exactly what vision did at the same wall: an image has H×W pixels,
one pixel is meaningless, per-pixel attention is hopeless, so cut the image into patches and call each
patch a token. The analogy to a series is immediate: cut the length-L series into contiguous sub-series
patches and let each patch be a token. A patch of sixteen consecutive steps is a little shape — a ramp,
a bump, a level — which is precisely what attention should be comparing.

The two knobs — patch length P and stride S — trade off against each other and I want to pick them for a
reason, not by default. Push P up toward L and I approach one token for the whole series: maximal
receptive field per token but only a handful of tokens, so attention has almost nothing to relate and I
have thrown away the intra-series resolution that made the shape idea worth having. Push P down toward 1
and I am back to the per-step tokenization I just indicted. Sixteen is a sensible middle on hourly-ish
data: long enough that a patch spans a meaningful local motif — most of an intraday up-and-down — short
enough that twelve of them tile the window and attention has real structure to compare. For the stride,
S = P would tile the window with disjoint patches, but a boundary that falls in the middle of a motif
splits it across two tokens and neither sees it whole; S = P/2 = 8 overlaps consecutive patches by half,
so every local shape is captured intact by at least one patch regardless of phase, at the cost of roughly
doubling the token count from six to twelve. I take the overlapping choice because the phase-robustness is
worth the extra six tokens when twelve is already tiny.

Make it concrete. Take the i-th univariate series, pick a patch length P and a stride S (the hop between
consecutive patch starts), slide a width-P window in steps of S; each placement is one patch in R^P. The
patch count is N = floor((L − P)/S) + 1. I want to be careful at the boundary, because the last steps of
the look-back matter most for forecasting and I do not want the windowing to drop them: if (L − P) is
not a multiple of S the last window does not reach the final value, the one I least want to lose. So pad
the end with S copies of the last value before patching; that guarantees one more full window including
the true end and adds exactly one patch, so N = floor((L − P)/S) + 2. With the scaffold's L = 96,
P = 16, S = 8: floor(80/8) + 2 = 12 patches per channel. The attention map is then 12×12 instead of
96×96 — the quadratic-in-L wall everyone attacked by mutilating the kernel, I walk around by changing
what a token is, an S² ≈ 64× cut in attention cost.

I should trace the padding to be sure it does what I claim and drops nothing. With L = 96, P = 16, S = 8,
the unpadded window starts sit at 0, 8, 16, …, 80 — eleven of them, the last patch covering steps 80–95,
which does reach the final step here because (96 − 16) = 80 happens to be a multiple of 8. The padding
rule adds one more start at 88, whose patch covers steps 88–103 with steps 96–103 filled by copies of the
last value; that guarantees the true endpoint sits in the interior of a patch rather than only at the
edge of the last one, so the most recent shape is represented twice and never clipped. Twelve patches,
each in R^16, and the +2 in N = floor((L − P)/S) + 2 is that guaranteed extra window. The cost of the
head follows directly: the flattened per-channel feature is N·d_model = 12 × 512 = 6144, and the head is
one linear map 6144 → 96, about 590k weights, shared across all channels. That single head is where most
of the model's parameters live, and sharing it across channels is the same channel-independence bet as
the backbone — one temporal read-out for every series.

Now the channel decision, and here the linear result is decisive. The linear control was channel-shared
and channel-blind, and it was *competitive* — so channel-independence is not what cost it; the missing
piece, if any, is structure, not coupling, at least for the temporal question I am isolating. So I keep
channel independence: process each of the C channels through the *same* shared Transformer backbone,
each as its own sequence of patch tokens, never mixing channels in attention. This is the cleanest
possible test of "is patched temporal attention better than a linear temporal map," because it holds the
channel treatment fixed at the linear model's setting and changes only the temporal operator. It also
means the C channels become a batch dimension: reshape [B, C, L] so the backbone sees B·C independent
patch sequences, which makes the model see C× more training sequences — a real regularizer on the
small datasets. The arithmetic is stark on ECL: with batch 32 and 321 channels the backbone processes
32 × 321 = 10272 patch sequences per step, so the shared weights get thousands of gradient signals per
batch even though there are only 321 distinct series — a single well-trained temporal backbone rather
than 321 under-trained ones. On the seven-channel data the multiplier is only ×7, which is part of why I
expect the fixed d_model=512 to sit uneasily there: many parameters, comparatively few effective
sequences to constrain them.

I need the per-window normalization the linear model could mostly lean on the loader for, because
patching does not absorb level drift. I fold in instance normalization the reversible way: per window,
per channel, subtract the look-back mean and divide by the look-back standard deviation before
patching, then multiply back the std and add back the mean after the prediction head. This is the
non-stationary normalization the scaffold's other models use, applied at the window level so the
backbone only ever sees zero-mean, unit-variance shapes and the level/scale are restored at the end.

I check that the wrap is exactly invertible so it cannot silently bias the forecast. Forward I compute
per window and channel m = mean(x), s = sqrt(var(x) + ε), and feed (x − m)/s into the backbone; the head
emits a normalized horizon ŷ, and I return s·ŷ + m. If the backbone were the identity on the statistics —
if the horizon truly continued the window's own level and scale — this returns exactly the right absolute
values, and in general it means the network only ever has to predict *shape* while the level and scale it
should carry forward are reinstated arithmetically. The ε = 1e-5 guards the division when a window is
flat, and detaching m from the graph keeps the recentering from leaking a gradient path that would let the
model game the mean instead of learning the shape. This is the same recentering I deferred at the linear
rung as "the obvious next handle"; patching forces my hand because a patch embedding has no way to absorb
a level offset the way a full-window linear row implicitly could, so the normalization has to be explicit
here.

The backbone itself I keep stock, because the whole point is that the *tokenization* is the fix, not a
new attention kernel. Embed each patch with a linear map P→d_model plus a positional embedding over the
N patch positions (now that I have order-bearing local tokens, the positional encoding is doing
honest work, not papering over a mismatch). The distinction is worth making precise: with twelve patch
tokens, shuffling them genuinely scrambles the series — the third-day motif landing where the first
morning belongs is a real corruption the model should be able to detect — so a positional encoding that
lets attention recover which patch sits where is adding recoverable information, not undoing damage the
tokenization itself inflicted. In the per-step regime the positional encoding was fighting the operator's
own permutation invariance just to reconstruct an order the tokenization had dissolved; here it rides on
top of tokens that already carry within-patch order intact, and only twelve inter-patch positions need
tagging. Run e_layers of standard pre-LN-style encoder blocks —
multi-head self-attention over the N patch tokens, then a position-wise feed-forward — and here a
detail from the reference matters: the encoder norm is a BatchNorm over the feature axis rather than
LayerNorm, which is steadier when the token count N is small (12 here) and the per-token statistics are
noisy. The reason is a statistics-count argument. LayerNorm computes its mean and variance across the
d_model = 512 features of a single token, and here a token is one patch of sixteen raw steps embedded up
to 512 dimensions — the effective information is only sixteen numbers, so the per-token statistics
LayerNorm relies on are estimated from a thin, noisy slice and it renormalizes each of twelve patches on
its own shaky footing. BatchNorm instead pools each feature's statistics across the whole batch of
B·C·N patch instances — thousands of samples on ECL, hundreds even on ETTh1 — so the normalization
constants are estimated from a large, stable population and do not wobble with an individual patch. When
N is small and the tokens are shape-fragments rather than rich semantic vectors, the cross-batch estimate
is simply the better-conditioned one, and it is a deliberate departure from the language-Transformer
default that suits the shape of this data. The scaffold exposes `Encoder`/`EncoderLayer` and `FullAttention`/`AttentionLayer`; I assemble
them with `patch_embedding` and a flatten head. The head takes the [B·C, N, d_model] encoded patches,
flattens the N·d_model patch features per channel, and maps them with one linear layer to the full
pred_len horizon — direct multi-step again, no decoder, consistent with the linear model's generation
step that already worked.

I have to be explicit about what this edit surface does *not* let me match from the method's own tuned
recipe, because it changes what I should expect. The PatchTST training script tunes `e_layers`,
`n_heads`, and `batch_size` per dataset — 1 layer / 2 heads on ETTh1, more on Weather and ECL — and uses
a much longer look-back (336/512) where patching pays off most. The Custom edit gets one fixed config
for all three: `seq_len=96`, `e_layers=2`, `n_heads=8`, `d_model=512`. So I am running PatchTST at a
*short* look-back and a *fixed* capacity, which blunts its signature advantage — patching's headroom to
afford a long, informative history is unavailable here, and on tiny ETTh1 the fixed `d_model=512` with 2
layers is over-parameterized for 12 patch tokens. I can quantify the blunting: at the method's own
look-back of 336 the same P = 16, S = 8 gives floor((336 − 16)/8) + 2 = 42 patch tokens, a 42×42
attention map with genuine long-range structure for the operator to find, and the whole reason patching
pays is that it makes such a long window affordable. Clamped to seq_len = 96 I get only twelve tokens, so
attention is relating a dozen fragments over a single short window — the mechanism still works but its
headroom is gone, and any advantage it shows here is the floor of what the method can do, not its tuned
ceiling. I keep `patch_len=16` (the config default) and set the stride internally to 8; I cannot rely on
the loop to pass a stride, so it is hardcoded. This is the
"same-named baseline is not the original method here" gap in concrete form: the architecture is faithful, the
regime is the scaffold's, not the recipe's.

Falsifiable expectations against the linear numbers. If patched temporal attention genuinely extracts
more than the linear map, I expect it to beat DLinear's mean MSE of 0.2676 — and the cleanest place to
see it is ETTh1, where the temporal-only test is purest: I expect PatchTST to drop ETTh1 MSE below
0.3962, into the high 0.37s, beating the linear control on the dataset the linear control was strongest
on. On Weather I expect a clear improvement over 0.1962 — patch-level shapes plus instance norm should
help the noisy 21-channel data. The risk is ECL: the linear model's worst dataset (0.2104) is worst
because of *channels*, not time, and PatchTST is still channel-independent, so I expect it to improve ECL
only modestly and possibly less than a cross-channel model would — if ECL stays high while ETTh1 and
Weather drop, that is the signal that the next lever must be cross-variate, not temporal. That is exactly
the fork this rung is built to expose.

To make the fork quantitative: the linear control's mean MSE is 0.2676, split 0.3962 / 0.1962 / 0.2104.
If patching helps time but not channels, the two temporal-limited datasets should carry the improvement
while ECL lags, so I expect ETTh1 and Weather to move by the larger fractions and ECL by the smaller one,
dragging the mean down toward the low 0.24s but leaving ECL as the visible residue. If instead all three
move together, the temporal fix helped everywhere and the channel story is weaker than I think. And if
ECL barely moves at all while the other two fall, that is the strongest possible instruction that the
remaining error on the wide dataset is cross-channel and untouchable by any channel-independent operator,
however good its temporal tokenization — which is precisely the condition that would force the next rung
to model cross-channel structure directly rather than reach for a still-better temporal operator.
