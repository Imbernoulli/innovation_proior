The periodic 2-D model moved every number in the right direction, and where it moved them most is the
tell. On seed 42 it cut ETTh1 from 0.1498 to 0.0803 MSE and ECL from 0.1132 to 0.0915, with Weather
dropping to 0.0293 — so the nonlinearity bought the ETTh1 transients exactly as I expected, and the
implicit channel mixing in the embedding bought a real chunk of ECL. But ECL is *still* the worst of the
three at 0.0915, and that bothers me, because ECL is the dataset where the problem is least about hard
temporal variation (the electricity clients are smooth and strongly periodic) and most about exploiting
correlation across hundreds of channels. The model that was supposed to fix channel-blindness still does
its worst work precisely there. So I want to understand *why* the cross-channel handling is the bottleneck
on ECL, because that diagnosis decides the next rung.

Here is what I think happened. TimesNet is not channel-independent — its `DataEmbedding` projects all
`enc_in` channels into the shared `d_model` features at each timestep, and everything downstream operates
on that mixed representation. On a 7-channel dataset like ETTh1 that mixing is benign and even helpful. But
on ECL there are 321 channels and `d_model` is fixed; cramming 321 correlated-but-distinct client series
into one shared feature vector per timestep forces the model to find one common basis for all of them, and
the per-client idiosyncrasies — exactly the detail you need to reconstruct a *specific* masked client — get
averaged away in the projection. The mixing is a blunt instrument: it shares statistical strength across
channels (good when channels are few and similar) but it cannot preserve per-channel identity when channels
are many. So the very mechanism that helped ECL versus DLinear is also what caps it: forced channel mixing
at the input. That points at a specific hypothesis — on high-channel data, I should *not* mix channels in
the representation at all; I should model each channel independently and let parameter sharing, not feature
mixing, do the cross-channel work. The cross-channel correlation can still be exploited implicitly, because
a single shared model trained on all 321 clients learns the common dynamics from all of them, but each
client is reconstructed from *its own* observed values, with its identity intact.

That is one half of the next rung: channel independence. The other half I get by revisiting the thing that
made the linear interpolator competitive in the first place — that a single timestep carries almost no
standalone meaning, and the signal lives in *shapes over short stretches*. TimesNet's token is still
effectively a single timestep (embedded), and its FFT-folding is a clever way to expose multi-scale
structure, but the atomic unit it attends to is point-wise. What if I make the atomic unit a *patch* — a
short contiguous run of timesteps — instead of a single point? A length-16 patch of a channel is a little
shape: a rising edge, a dip, a local oscillation. That is a unit with real content, the kind of object it
makes sense to compare with attention. Patching does three things at once. It gives each token genuine
semantic content (a local shape rather than a meaningless scalar), so self-attention finally compares
objects worth comparing. It cuts the sequence length the attention sees from 96 to roughly 96/stride
patches, so the quadratic cost and the noise of point-wise attention both drop. And it lets the model
aggregate local information within a patch before any global mixing, which for imputation means a masked
entry is reconstructed first from its immediate patch-mates and then from the patch-level context — exactly
the locality interpolation wants.

So the rung is: split each channel's window into overlapping patches, embed each patch into `d_model`,
run a standard Transformer encoder over the patch sequence *per channel independently*, and project the
encoded patches back to a length-96 reconstruction. Let me work the pieces, because several of them are
task-specific and differ from the forecasting form of this idea.

Patching first. With `patch_len=16` and `stride=8` the patches overlap by half, which matters: a masked
entry near a patch boundary still sits in the interior of a neighbouring patch, so the overlap gives every
position multiple patch contexts and avoids boundary blind spots. To make the patch count clean and to
give the last positions a full patch, the library's `PatchEmbedding` replication-pads the end by `stride`
before unfolding, then projects each `patch_len`-vector to `d_model` with a linear layer and adds a
positional code. Crucially, `PatchEmbedding` reshapes the input so the batch-and-channel axes are folded
together: the encoder runs on `[B·C, num_patches, d_model]`, which is precisely channel independence in
implementation form — every channel of every batch element is its own sequence through the same shared
encoder weights. That is the mechanism I argued for: shared parameters, no feature mixing across channels,
per-channel identity preserved. On ECL this should be the decisive change versus TimesNet.

The encoder is a plain Transformer encoder — full self-attention over the patch sequence, feed-forward,
with the one library detail that the norm is `BatchNorm1d` over the `d_model` axis (transposed in and out)
rather than LayerNorm, which is empirically steadier for patch tokens. Self-attention over patches is now
doing something sensible: it asks "which other local shapes in this channel's window resemble or inform the
shape around the masked region," and because the tokens are real shapes the attention map is meaningful
rather than the point-wise noise that made attention lose to a linear map. After the encoder I have
`[B·C, num_patches, d_model]`; I reshape back to `[B, C, num_patches, d_model]`, permute to put `d_model`
before the patches, and a `FlattenHead` flattens `(d_model, num_patches)` and linearly projects to the
full `seq_len=96` reconstruction per channel. The flatten-then-project head is the right choice for
imputation: it lets every output timestep draw on the whole encoded patch sequence at once (a length-one
path from any patch to any reconstructed position), which is what interpolation across a window wants, as
opposed to a per-patch decoder that would localise too much.

Now the imputation-specific normalisation, which is the same masked-statistics device I derived for the
periodic model and which I must keep, because patching is even more sensitive to corrupted statistics — a
patch that straddles a hole would otherwise carry a fake-zero into its embedded shape. I compute the mean
and std over observed entries only (`sum(mask == 1)` as the denominator), subtract the mean, re-zero the
holes with `masked_fill(mask == 0, 0)` so the patches see real holes rather than spurious values, divide by
the std, run the patch encoder, and undo the normalisation after the head with the stored, detached mean
and std repeated over `seq_len`. The de-normalisation repeat uses `seq_len` (not the forecasting
`pred_len`), because for imputation the output length equals the input length. Time features are *not* used
here — patch embedding consumes only the value patches plus positional codes — which is a deliberate part of
the channel-independent design: each channel is reconstructed from its own shape sequence, and the calendar
phase is recoverable from the periodic structure within the patches themselves.

Does this answer the TimesNet diagnosis? The bottleneck I identified was forced channel mixing capping ECL;
channel independence removes that cap directly, so I expect the largest gain of the whole ladder to land on
ECL. The patch tokenisation should also help ETTh1 and Weather by giving attention meaningful units, but
those were already in good shape, so the margins there should be smaller and it is even possible that on a
very-few-channel, transient-heavy dataset like ETTh1 the channel-independent patch model gives up a little
to the periodic model's explicit multi-scale folding — ETTh1's sharp hourly transients are exactly the kind
of fine within-window nonlinearity that period-folding 2-D convs capture and patch-level attention might
smooth over. That is the honest risk: channel independence is unambiguously right for ECL and a wash-to-win
elsewhere, but it is not guaranteed to dominate TimesNet on every single dataset.

So rung three is the channel-independent patch Transformer: masked-statistics normalisation, overlapping
patch embedding folding channels into the batch, a `BatchNorm` Transformer encoder over the patch sequence,
a flatten head projecting back to 96 per channel, de-normalise — the full scaffold module is in the answer.
The falsifiable expectations against the TimesNet numbers are sharp. ECL must improve substantially below
0.0915 MSE — that is the whole thesis of channel independence, and if it does not, the thesis is wrong and
the next move would have to be *explicit* cross-channel attention rather than independence. Weather should
hold or improve (already small at 0.0293). ETTh1 is the one I am least sure about: I expect it to stay
competitive but I would not be shocked if the periodic model edges it there, because period-folding suits
ETTh1's transients. The aggregate bar this rung must clear is the TimesNet mean (0.067 MSE / 0.151 MAE);
the way I expect it to clear it is by a large ECL gain dragging the mean down even if ETTh1 is roughly a
tie — and that asymmetry, a method that wins by fixing the channel bottleneck rather than the temporal one,
is exactly the signature that would confirm the diagnosis I started from.
