The periodic 2-D model moved every number in the right direction, and where it moved them most is the
tell. On seed 42 it cut ETTh1 from 0.1498 to 0.0803 MSE and ECL from 0.1132 to 0.0915, with Weather
dropping to 0.0293. Turning those into fractions is the diagnosis: ETTh1 fell by (0.1498 − 0.0803) /
0.1498 ≈ 46%, Weather by ≈ 42%, but ECL by only (0.1132 − 0.0915) / 0.1132 ≈ 19%. So the rung built to fix
*both* the temporal-nonlinearity failure and the channel-blindness failure delivered a big cut on the two
datasets whose problem was temporal (ETTh1, and to whatever extent Weather had one) and a feeble cut on the
one whose problem was cross-channel (ECL). And ECL is *still* the worst of the three at 0.0915 — now worse
than ETTh1's 0.0803, a reversal from rung one where ETTh1 was worst. That reversal is the signal: ECL is
the dataset where the problem is least about hard temporal variation (smooth, strongly periodic clients)
and most about exploiting correlation across hundreds of channels, and it is precisely there that
TimesNet's improvement stalled. The model that was supposed to fix channel-blindness still does its worst
work where channel structure matters most. So I want to understand *why* the cross-channel handling is the
bottleneck on ECL, because that diagnosis decides the next rung.

I can reason it from the mechanism. TimesNet's `DataEmbedding` projects all `enc_in` channels into the
shared `d_model` features through a single `Linear(enc_in, d_model)`, and everything downstream operates on
that mixed representation. On 7-channel ETTh1 that projection is `Linear(7, 512)`: an over-complete lift
where the network has ample room to keep each channel separable — the mixing is benign, even helpful,
letting the few channels share strength. But on ECL the same projection is `Linear(321, 512)`: it
compresses 321 correlated-but-distinct clients into 512 shared features, barely more slots than inputs, and
must find one common basis serving all 321 at once. The per-client idiosyncrasies — exactly the detail
needed to reconstruct a *specific* masked client — get averaged into that shared basis and cannot be pulled
back out cleanly by the projection head. So the very mechanism that helped ECL versus DLinear is what caps
it: forced channel mixing at the input, a lift when channels are few but a lossy compression when they are
many. The 19%-versus-46% asymmetry is what that compression looks like in the metric. That points at a
specific hypothesis: on high-channel data, do *not* mix channels in the representation at all; model each
channel independently and let parameter sharing, not feature mixing, do the cross-channel work. A single
shared model trained on all 321 clients still learns the common dynamics from all of them, but each client
is reconstructed from *its own* observed values, with its identity intact.

That is one half of the next rung: channel independence. The other half I get by revisiting what made the
linear interpolator competitive — that a single timestep carries almost no standalone meaning, and the
signal lives in *shapes over short stretches*. TimesNet's token is still effectively a single embedded
timestep; its FFT-folding is a clever way to expose multi-scale structure, but the atomic unit is
point-wise. What if the atomic unit is a *patch* — a short contiguous run of timesteps — instead of a
single point? A length-16 patch is a little shape: a rising edge, a dip, a local oscillation, a unit with
real content worth comparing with attention. Patching does three things at once: it gives each token
genuine semantic content, so self-attention finally compares objects worth comparing; it cuts the sequence
length attention sees from 96 to ~96/stride patches, dropping both the quadratic cost and the point-wise
noise; and it lets the model aggregate within a patch before any global mixing, so a masked entry is
reconstructed first from its immediate patch-mates and then from patch-level context — exactly the locality
interpolation wants.

Before building it I should ask the honest alternative: if the bottleneck is cross-channel, why not add
*explicit* cross-channel attention — a second attention axis over the 321 channels — rather than removing
mixing entirely? Explicit channel attention over ECL means a 321 × 321 ≈ 103k-score matrix per layer per
head, *learned* from the training windows; with only the loop's budget (batch 16, 10 epochs) that is a
large, noisy object to estimate, and worse, it re-introduces the failure I am escaping — it lets one
client's reconstruction depend on fitted weights to hundreds of others, so a mis-estimated cross-channel
weight corrupts the fill. Channel independence gets the cross-channel benefit for free and robustly: a
shared encoder trained across all 321 clients sees 321× more sequences per batch, so the *shared weights*
are estimated from an enormous, diverse sample while each client is reconstructed only from its own shape.
Parameter sharing gives the statistical strength, channel independence keeps the identity — the strictly
safer way to spend the capacity, so I drop explicit channel attention and commit to independence.

So the rung is: split each channel's window into overlapping patches, embed each into `d_model`, run a
standard Transformer encoder over the patch sequence *per channel independently*, and project the encoded
patches back to a length-96 reconstruction. Several pieces are task-specific and differ from the
forecasting form, and the patch geometry needs its numbers checked.

Patching first. With `patch_len = 16` and `stride = 8` the patches overlap by half, which matters: a masked
entry near a patch boundary still sits in the interior of a neighbouring patch, so the overlap gives every
position multiple patch contexts and avoids boundary blind spots. The library's `PatchEmbedding`
replication-pads the end by `stride = 8` before unfolding, so the number of patches is
`(seq_len − patch_len) / stride + 2 = (96 − 16) / 8 + 2 = 12`. Each channel's 96-point window becomes 12
patch tokens: self-attention now runs over a length-12 sequence, its score matrix shrinking from 96² = 9216
to 12² = 144 — a 64× reduction in the quadratic term — while every token carries a 16-point shape instead
of a lone scalar. Each patch is projected from its 16-vector to `d_model = 512` and gets a positional code.
Crucially, `PatchEmbedding` folds the batch and channel axes together: the encoder runs on
`[B·C, num_patches, d_model]`, which is channel independence in implementation form — every channel of
every batch element is its own sequence through the same shared encoder weights. On ECL with batch 16 that
is 16 × 321 = 5136 independent patch-sequences streaming through one shared encoder per step; on ETTh1,
16 × 7 = 112. Shared parameters estimated from thousands of per-channel sequences, no feature mixing across
channels, identity preserved. On ECL this should be the decisive change versus TimesNet, because the
`Linear(321, 512)` compression is simply gone.

The encoder is a plain Transformer encoder with `n_heads = 8` over `d_model = 512`, so each head works in a
64-dimensional subspace — full self-attention over the patch sequence, feed-forward, with the one library
detail that the norm is `BatchNorm1d` over the `d_model` axis (transposed in and out) rather than
LayerNorm. BatchNorm is steadier for patch tokens because it normalises each feature across the batch of
patch-sequences rather than across the 12 patches of one sequence, and 12 is a short axis to compute a
stable per-token statistic over. Self-attention over patches now asks "which other local shapes in this
channel's window inform the shape around the masked region," and because the tokens are real shapes the
attention map is meaningful rather than the point-wise noise that made attention lose to a linear map.
After the encoder I have `[B·C, num_patches, d_model]`; I reshape to `[B, C, num_patches, d_model]`,
permute to `[B, C, d_model, num_patches]`, and a `FlattenHead` flattens the last two axes into a
`d_model × num_patches = 512 × 12 = 6144`-vector per channel and linearly projects it to the full
`seq_len = 96` reconstruction — a `Linear(6144, 96)`, about 590k weights, applied identically to every
channel.

The flatten-then-project head is where imputation diverges most from the forecasting use of patches. In
forecasting the head projects to `pred_len` *future* steps never in the input; here it projects to the 96
steps that *are* the input window, holes and all, and the loss reads only the masked positions. So the head
reconstructs an interior, not a continuation — every output timestep draws on the whole encoded patch
sequence at once (a length-one path from any patch to any reconstructed position), which is what
interpolation wants: an interior hole at *t* should draw on patches on *both* sides, which a whole-window
flatten provides and a causal or per-patch decoder would not. And because the head keys its output length
off `seq_len`, it is correct even though the harness hands the module a forecasting-style `pred_len` to
ignore.

The imputation-specific normalisation is the same masked-statistics device I derived for the periodic
model, and I keep it — patching is even more sensitive to corrupted statistics, since a 16-wide patch has
~4 of every 16 entries masked, so an uncorrected statistic would bias every single patch token. Mean and
std over observed entries only (`sum(mask == 1)` denominator), subtract the mean, re-zero the holes with
`masked_fill(mask == 0, 0)` so patches see a centred zero rather than a spurious `−mean`, divide by std,
run the encoder, and undo the detached normalisation after the head, repeated over `seq_len` not
`pred_len`. Time features are *not* used here — patch embedding consumes only the value patches plus
positional codes — a deliberate part of the channel-independent design: each channel is reconstructed from
its own shape sequence, and a shared time-feature stamp would only re-introduce the cross-channel common
signal I just worked to remove. The channel fold and unfold are consistent because the fold is a plain
reshape of the leading axis, so `B·C → (B, C)` recovers the original channel order, and the mask is
consulted only in the normalisation statistics — the reconstruction never peeks at which entries were
holes.

One geometric check worth making, since a gap in patch coverage would orphan masked positions: patch *j*
covers timesteps `[8j, 8j + 15]` for j = 0..11 with the tail replication-padded, so patch 0 covers 0..15,
patch 1 covers 8..23, and so on — consecutive patches overlap on 8 positions with no interior gap, every
timestep from 0 to 95 lies inside at least one patch, and interior timesteps inside two. So a masked entry
always has at least one patch containing it *and* its observed neighbours, usually two offering different
local contexts, and the padding guarantees the tail positions 88..95 are covered rather than clipped.

Part of my argument is that channel independence is not just more accurate on ECL but *cheaper and
better-conditioned* there. The input side of PatchTST is a
shared `Linear(patch_len, d_model) = Linear(16, 512)`, about 8k weights, identical for every channel and
dataset — there is no `Linear(enc_in, d_model)` whose width scales with the channel count, so ECL's 321
channels cost exactly the same per-channel projection as ETTh1's 7. Contrast the periodic model's Inception
2-D convs, tens of millions of weights per block; the patch encoder's cost lives in two Transformer layers
over a length-12 sequence, a trivial 12² attention term. So on the high-channel dataset that was the whole
problem, this rung is both structurally correct (no channel compression) and computationally lighter on the
input path, with the capacity pooled into the shared encoder all 5136 ECL sequences per batch train jointly.

Does this answer the diagnosis? The bottleneck was forced channel mixing capping ECL; channel independence
removes that cap directly, so a priori I expect the largest gain of the whole ladder to land on ECL. The
patch tokenisation should also help ETTh1 and Weather by giving attention meaningful units, but those were
already in good shape, so the margins should be smaller — and it is even possible that on a very-few-channel,
transient-heavy dataset like ETTh1 the channel-independent patch model gives up a little to the periodic
model's explicit multi-scale folding, since a 16-wide patch with an 8-step stride averages a sharp one-hour
spike across a token spanning most of a shift, where period-folding 2-D convs would capture it. That is the
honest risk: channel independence is unambiguously right for ECL and a wash-to-win elsewhere, but not
guaranteed to dominate the periodic model on every dataset. The signature that would confirm the diagnosis
I started from is a method that wins by fixing the channel bottleneck rather than the temporal one — a large
ECL gain dragging the mean down even if ETTh1 is roughly a tie.
