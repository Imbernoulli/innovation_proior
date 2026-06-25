The linear control came back, and I read its numbers carefully before deciding what to build next. On
ETTh1 it landed MSE 0.3962 / MAE 0.4108 — exactly the "genuinely competitive" outcome I predicted: a
single affine map over a decomposed window is already in the high 0.30s on the small, strongly
trend-and-seasonal dataset, and any attention model that cannot clearly beat that there has no business
claiming the temporal-attention machinery was load-bearing. So the embarrassment the control was built
to deliver, it delivered. But the other two datasets tell the more useful story. On Weather it is 0.1962
/ 0.2561 and on ECL 0.2104 / 0.3016, for a mean MSE of 0.2676. ECL is the tell: 321 channels, and the
deliberately channel-blind, channel-shared linear map leaves the most on the table exactly there, just
as I expected — its MAE of 0.3016 is the worst of the three by a wide margin. So the diagnosis splits
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

Now the channel decision, and here the linear result is decisive. The linear control was channel-shared
and channel-blind, and it was *competitive* — so channel-independence is not what cost it; the missing
piece, if any, is structure, not coupling, at least for the temporal question I am isolating. So I keep
channel independence: process each of the C channels through the *same* shared Transformer backbone,
each as its own sequence of patch tokens, never mixing channels in attention. This is the cleanest
possible test of "is patched temporal attention better than a linear temporal map," because it holds the
channel treatment fixed at the linear model's setting and changes only the temporal operator. It also
means the C channels become a batch dimension: reshape [B, C, L] so the backbone sees B·C independent
patch sequences, which makes the model see C× more training sequences — a real regularizer on the
small datasets.

I need the per-window normalization the linear model could mostly lean on the loader for, because
patching does not absorb level drift. I fold in instance normalization the reversible way: per window,
per channel, subtract the look-back mean and divide by the look-back standard deviation before
patching, then multiply back the std and add back the mean after the prediction head. This is the
non-stationary normalization the scaffold's other models use, applied at the window level so the
backbone only ever sees zero-mean, unit-variance shapes and the level/scale are restored at the end.

The backbone itself I keep stock, because the whole point is that the *tokenization* is the fix, not a
new attention kernel. Embed each patch with a linear map P→d_model plus a positional embedding over the
N patch positions (now that I have order-bearing local tokens, the positional encoding is doing
honest work, not papering over a mismatch). Run e_layers of standard pre-LN-style encoder blocks —
multi-head self-attention over the N patch tokens, then a position-wise feed-forward — and here a
detail from the reference matters: the encoder norm is a BatchNorm over the feature axis rather than
LayerNorm, which is steadier when the token count N is small (12 here) and the per-token statistics are
noisy. The scaffold exposes `Encoder`/`EncoderLayer` and `FullAttention`/`AttentionLayer`; I assemble
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
layers is over-parameterized for 12 patch tokens. I keep `patch_len=16` (the config default) and set the
stride internally to 8; I cannot rely on the loop to pass a stride, so it is hardcoded. This is the
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
