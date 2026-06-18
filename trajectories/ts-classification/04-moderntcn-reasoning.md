TimesNet's three numbers land almost exactly where I aimed it, and the one place it does *not* improve is
the place that now defines the next move. On Handwriting it jumped to 0.3353 from PatchTST's 0.2541 and the
linear floor's 0.2306 — the largest gain on the ladder, vindicating cross-period 2-D structure plus
mask-aware pooling on the phase-recurrent gestures. On EthanolConcentration it reached 0.3194, clearing both
PatchTST (0.2852) and the floor (0.2890), because it never per-window-normalizes away the slow spectral trend
and its multi-scale convolution reads the curve at several scales. But on FaceDetection it sat at 0.6745 —
*below* PatchTST's 0.6853 and the floor's 0.6822, a small but real loss. That is the prediction I made
explicitly when I built it: TimesNet mixes channels only in the value embedding and the flatten-and-project
head, never inside the encoder, so on the one dataset whose entire signal is cross-channel covariance across
~144 MEG sensors it trades a little channel-friendliness for cross-period 2-D structure that this dataset
needs less. So the strongest rung so far has a clean, named weakness: no explicit cross-variable modelling.
Every rung has now mixed channels the same lazy way — a linear combination in the head — and FaceDetection
has refused to move past the high-0.68 ceiling for any of them. That is the gap I want to attack, without
giving back what TimesNet won on Handwriting and EthanolConcentration.

Let me be precise about what "mix channels" has meant and why it keeps failing on FaceDetection. The linear
floor mixed channels only in the final `Linear(enc_in · seq_len, num_class)` — one fixed linear functional per
class over the flattened window, so cross-channel covariance can be expressed only as a static weighting, never
as a learned, position-wise interaction. PatchTST went the other way and forbade in-encoder channel mixing
entirely (channel-independent, weights shared, channels folded into the batch), which is exactly why it could
not improve FaceDetection. TimesNet fuses channels in the embedding conv and the head, but its TimesBlocks
operate per-channel-feature inside the 2-D reshape. None of the three has a *dedicated, learnable stage whose
job is to mix information across the variables at each temporal position*. FaceDetection is the dataset that
punishes this: an MEG decision is not in any single sensor, it is in how sensors covary moment to moment. So
the missing component is concrete — an explicit cross-variable mixing operator inside the encoder — and I want
to add it without abandoning the convolutional, mask-tolerant temporal modelling that carried Handwriting and
EthanolConcentration.

Now the temporal side, because I also want to push past TimesNet's strengths, not just patch its weakness.
TimesNet's whole trick was to manufacture long-range (cross-period) locality by reshaping to 2-D, because a
1-D convolution kernel of sane width only reaches its immediate temporal neighbours — same-phase points one
period apart never land in one kernel without a kernel as wide as the period. But that framing quietly concedes
that wide kernels are impossible, and I want to question that. What if I just *use a very large 1-D kernel*? The
reason nobody did this historically is that a large dense kernel is expensive and hard to train — but a
**depthwise** large kernel is cheap: one kernel per channel, so a kernel of size 31 or 51 costs only
`channels × kernel` parameters, not `channels² × kernel`. A depthwise kernel that spans tens of timesteps gives
each channel a genuinely large effective receptive field in a single layer, which is exactly the long-range
temporal reach TimesNet went to 2-D to get — but kept in the natural 1-D layout, where the padding mask and
variable length are trivial to respect. This is the modern-convolution insight from vision: large depthwise
kernels plus pointwise mixing recover what attention offered, at convolutional cost. So my temporal operator is
a large-kernel depthwise 1-D convolution along time, applied per channel-feature.

A large kernel has a known training pathology, though: it is hard to optimize from scratch because the gradient
has to find structure across a wide support, and small-scale local detail (a two- or three-step wiggle that
matters for the fine gesture distinctions in Handwriting) is easy for a wide kernel to smear. The fix that the
structural-reparameterization line established is to train the large kernel *in parallel with a small kernel* —
a large branch (say size 31) and a small branch (say size 5), each its own depthwise conv with its own
BatchNorm, summed at the output. The small branch guarantees the fine local detail has a clean, easy-to-train
path while the large branch learns the long-range support; at inference the two branches and their BatchNorms
fuse into a single equivalent kernel (the small kernel zero-padded up to the large kernel's width and added), so
the deployed model is one large depthwise conv with no overhead. I will keep both branches explicit in training
— the large conv+BN and the small conv+BN summed — because that is what the forward pass actually computes; the
fusion is an inference-time identity, not a change to the function. This is the reparameterized large-kernel
depthwise conv, and it is my replacement for both TimesNet's 2-D reshape and PatchTST's patch attention: it gets
the long-range temporal reach in 1-D, where masking is free, with a small-kernel branch protecting local detail.

Now I have to organise the channel axis carefully, because this is where the cross-variable idea has to be made
real and it is the subtlest part. Take the window and lift each variable to a feature vector of width `D` (the
model dimension), so the working tensor is `[B, M, D, N]` — batch, M variables, D features per variable, N
temporal positions. The depthwise temporal conv runs over `M·D` grouped channels (genuinely depthwise: groups =
`M·D`), so it mixes only along time, never across variables or features — that keeps the temporal operator clean
and per-stream. Then I need two *different* pointwise mixers, and the distinction between them is the heart of
the design. The first, call it ConvFFN1, mixes across the **feature** dimension D within each variable: a
grouped 1×1 conv with `groups = M`, so each variable's D features are mixed among themselves but variables stay
separate — this is the per-variable channel MLP, the standard feature-mixing FFN. The second, ConvFFN2, mixes
across the **variable** dimension M: by permuting so the variable axis becomes the convolution's channel axis
and grouping by D (`groups = D`), each feature index is mixed *across all M variables* while features stay
separate. ConvFFN2 is the explicit cross-variable operator that every earlier rung lacked — a learnable,
per-feature interaction across the MEG sensors, applied at every temporal position. Both FFNs are the usual
two-layer 1×1 conv with a GELU and dropout between, expanding `D → ffn_ratio·D → D`. A block is: depthwise
large-kernel temporal conv, BatchNorm, ConvFFN1 (feature mix), ConvFFN2 (variable mix), all wrapped in a
residual connection. That decomposition — separable temporal conv, then feature mixing, then variable mixing —
is the whole architecture in one block, and it cleanly separates the three kinds of structure (time, feature,
variable) that the earlier rungs entangled.

A few mechanics I have to get right or this misbehaves on the heterogeneous UEA shapes. First, the stem. I embed
each variable independently: a `Conv1d(1, D, kernel=patch_size, stride=patch_stride)` applied with the variable
axis folded into the batch, so the same embedding weights process every variable (channel-independent stem,
shared weights — the data-efficiency argument from PatchTST still holds and matters because FaceDetection has
144 variables and little data per class). With `patch_stride < patch_size` I replicate-pad the tail by
`patch_size − patch_stride` so the last timestep is never dropped — the same edge-faithful padding logic as the
decomposition's replicate-pad and PatchTST's last-value pad, reused so no recent information is lost. The patch
count is `N = seq_len // patch_stride`. Second, multi-stage depth with downsampling: between stages a `BatchNorm`
then a strided `Conv1d(dims[i], dims[i+1], kernel=downsample_ratio, stride=downsample_ratio)` `r`-folds the
temporal length and grows the feature width, so later blocks see a coarser, wider representation — the standard
convolutional pyramid, which lets the large-kernel reach compound across scales. When the length is not divisible
by the downsample ratio I replicate-pad the tail before downsampling, again to avoid inventing zeros. The number
of stages and blocks is small (a couple of blocks per stage) because the UEA datasets are small and a deep stack
would overfit — the same restraint that sized the earlier rungs.

Third, the normalization placement, which is deliberate and not LayerNorm. Time series carry outliers — a sensor
glitch, a regime jump — and LayerNorm, normalizing within a token, gets dragged around by an outlier landing in
that token; BatchNorm normalizes each feature across the batch so a single outlier is diluted. This is the same
reason PatchTST used BatchNorm in its encoder, and it is doubly right here because the whole architecture is
convolutional and BatchNorm is the native normalization for conv stacks. The depthwise conv branches each carry
their own BatchNorm (which is what makes the train-time-two-branch / inference-time-one-kernel fusion exact), and
there is a BatchNorm over the feature dimension after the depthwise conv.

Now the head, and here I keep TimesNet's hard-won mask-awareness in spirit while adapting it to the convolutional
backbone. After the stages I have `[B, M, D, N_final]`. The canonical classification head applies a GELU and
dropout, flattens the whole `M · D · N_final` representation into one vector, and projects with a single
`Linear(M · D · N_final, num_class)`. That flatten-and-project is where the final class decision is drawn — but
it is the cross-variable ConvFFN2 *inside* every block that has already done the real channel mixing, so the
head no longer carries the entire burden of cross-channel modelling the way the linear floor's head did. The one
wrinkle is the padding mask: the stem's strided conv changes the temporal length from `seq_len` to `N_final`, so
I cannot multiply by `x_mark_enc` at full resolution the way TimesNet did. The reference classification head does
not consult the mask at all — it relies on the replicate-padding (not zero-padding) of the tail before every conv,
so the padded region carries the boundary value rather than a spurious zero, and on the learned flatten head the
optimizer can suppress the constant padded tail. I keep the head faithful to that reference — GELU, dropout,
flatten, linear — since that is exactly what produced ModernTCN's measured UEA results, and adding an ad-hoc mask
multiply at a downsampled resolution would diverge from the canonical implementation without a principled gain.

Let me make sure I have not just rebuilt TimesNet with extra parts. The temporal operator is different: a
large-kernel depthwise 1-D conv (with a small-kernel reparam branch) instead of the FFT-period 2-D reshape — it
gets long-range reach in the natural 1-D layout where masking and variable length are trivial, and it costs less
because depthwise. The channel handling is different and is the whole point: an explicit cross-variable mixing
stage (ConvFFN2) that no earlier rung had, separated cleanly from feature mixing (ConvFFN1) and from temporal
mixing (the depthwise conv). The stem is channel-independent with shared weights, keeping PatchTST's data
efficiency. So this is a genuinely different architecture aimed squarely at the cross-channel ceiling that
pinned FaceDetection, while retaining the convolutional, multi-scale, mask-tolerant temporal modelling that won
Handwriting and EthanolConcentration for TimesNet.

The falsifiable expectations, against TimesNet's measured numbers, dataset by dataset. On FaceDetection
(TimesNet 0.6745, the ceiling none of the three cracked, all stuck in the high-0.68s at best): this is the
sharpest test, because ConvFFN2 is the first dedicated cross-variable operator on the ladder and FaceDetection's
signal *is* cross-variable covariance — I expect ModernTCN to finally break the high-0.68 ceiling, into roughly
the 0.69–0.71 range, and if it does not, then explicit cross-variable mixing is not the missing piece and the
FaceDetection ceiling is set by something else (label noise, the binary task's intrinsic limit). On
EthanolConcentration (TimesNet 0.3194, the best so far): the large-kernel depthwise conv reads the smooth
spectral curve at long range without per-window normalization, so I expect at least parity and plausibly a small
gain into the low-0.32s — it should not regress, because nothing here washes out the trend. On Handwriting
(TimesNet 0.3353, the ladder's biggest mover): the phase-recurrent strokes are reachable by a large temporal
kernel (long-range) plus the small-kernel branch (the fine local stroke detail), so I expect parity-to-gain,
around 0.33–0.36; the risk is that the strided downsampling coarsens the fine gesture detail, in which case the
small-kernel branch is what saves it. The single decisive question is FaceDetection: if explicit cross-variable
mixing finally moves the one dataset that defeated every prior rung's lazy head-only channel mixing, while
holding the other two, then ModernTCN is the strongest endpoint on this ladder and the diagnosis — that the
recurring weakness was the absence of a dedicated cross-channel stage — was correct. The distilled module and
the literal `Custom.py` fill are in the answer.
