TimesNet lands almost exactly where I aimed it, and the one place it does *not* improve now defines the next
move. On Handwriting it jumped to 0.3353 from PatchTST's 0.2541 and the floor's 0.2306 — the largest gain on
the ladder, vindicating cross-period 2-D structure plus mask-aware pooling on the phase-recurrent gestures.
On EthanolConcentration it reached 0.3194, clearing PatchTST (0.2852) and the floor (0.2890), because it
never normalizes away the slow spectral trend and reads the curve at several scales. But on FaceDetection it
sat at 0.6745 — *below* PatchTST's 0.6853 and the floor's 0.6822, a small but real loss, the prediction I
made when I built it: TimesNet mixes channels only in the value embedding and the head, never inside the
encoder, so on the one dataset whose entire signal is cross-channel covariance across ~144 MEG sensors it
trades a little channel-friendliness for cross-period structure this dataset needs less. So the strongest
rung so far has a clean, named weakness: no explicit cross-variable modelling.

The deltas sharpen what to hold and what to move. TimesNet vs PatchTST: Handwriting +0.0812,
EthanolConcentration +0.0342, FaceDetection −0.0108; means 0.4431 vs 0.4082 vs the floor's 0.4006, so
TimesNet added +0.0349 mean, and unlike PatchTST's gain (almost entirely one dataset) this one is broad —
two big moves up partly offset by one small move down. Now read the FaceDetection column *down the whole
ladder*: 0.6822 → 0.6853 → 0.6745. Three architectures, a spread of just 0.0108, and the best of them is the
linear floor's head essentially tied with PatchTST's. FaceDetection has been *flat* across every rung
regardless of temporal machinery — the temporal encoder is not the variable that moves it. That is the
strongest evidence its bottleneck is orthogonal to everything the ladder has varied, and it is cross-channel.
Every rung has mixed channels the same lazy way: the floor only in the final `Linear(enc_in · seq_len,
num_class)`, a static per-class weighting that can never express a learned position-wise interaction;
PatchTST forbade in-encoder mixing entirely; TimesNet fused only in the embedding and head. None has a
*dedicated, learnable stage whose job is to mix across variables at each temporal position*, and an MEG
decision is not in any single sensor, it is in how sensors covary moment to moment. So the target is precise:
add an explicit cross-variable stage to move the flat 0.68 column, without regressing the +0.081 and +0.034
that cross-period 2-D and no-normalize bought.

I also want to push past TimesNet's strengths on the temporal side, not just patch its weakness. TimesNet's
whole trick was to manufacture cross-period locality by reshaping to 2-D, because a sane-width 1-D kernel
only reaches immediate neighbours — same-phase points one period apart never land in one kernel without a
kernel as wide as the period. But that quietly concedes wide kernels are impossible, and I want to question
it. What if I just *use a very large 1-D kernel*? A large dense kernel is expensive and hard to train — but a
**depthwise** large kernel is cheap: one kernel per channel, so size 31 or 51 costs only `channels × kernel`
parameters, not `channels² × kernel`. A depthwise kernel spanning tens of timesteps gives each channel a
genuinely large receptive field in a single layer — the long-range reach TimesNet went to 2-D to get — but
kept in the natural 1-D layout, where the padding mask and variable length are trivial to respect. This is
the modern-convolution insight from vision: large depthwise kernels plus pointwise mixing recover what
attention offered, at convolutional cost.

The cheapness is the whole reason this is viable. The temporal conv runs over `M · D` genuinely depthwise
channels (groups = `M · D`). With `D = 64`, FaceDetection's `M = 144` gives `144 · 64 = 9,216` depthwise
channels, so a size-31 kernel costs `9,216 · 31 ≈ 285,700` weights (plus a size-5 branch at ~46,100). The
*dense* large conv the historical objection assumed would be `9,216² · 31 ≈ 2.6 billion` — untrainable on
~5,900 FaceDetection series. Depthwise cuts that by a factor of `channels` to a quarter-million: the
difference between impossible and routine. On EthanolConcentration the depthwise channels are `3 · 64 = 192`,
so the size-31 kernel is ~6,000 weights, negligible. Translate the receptive field into raw timesteps: the
stem strides by 4, so one position is 4 original steps, a size-31 kernel spans `31 · 4 ≈ 124` steps in one
conv, and after the stage-1 downsample-by-2 the reach doubles to ~248. On FaceDetection's ~62-step window
the stem produces only ~15 positions, so a padded size-31 kernel already covers the *entire* sequence in a
single layer — full-window temporal context for free.

A large kernel has a known training pathology: the gradient must find structure across a wide support, and
small-scale local detail (a two- or three-step wiggle that matters for fine Handwriting distinctions) is
easy for a wide kernel to smear. The structural-reparameterization fix is to train the large kernel *in
parallel with a small kernel* — a large branch (size 31) and a small branch (size 5), each its own depthwise
conv with its own BatchNorm, summed. The small branch gives fine local detail a clean, easy-to-train path
while the large branch learns the long-range support; at inference the two branches and their BatchNorms
fuse into one equivalent kernel (the small kernel zero-padded up to the large width and added), so the
deployed model is one large depthwise conv with no overhead. I keep both branches explicit in training,
since that is what the forward pass actually computes — the fusion is an inference-time identity, not a
change to the function.

Now the channel axis, the subtlest part. Lift each variable to a feature vector of width `D`, so the working
tensor is `[B, M, D, N]`. The depthwise temporal conv over `M·D` grouped channels mixes only along time,
never across variables or features — clean and per-stream. Then two *different* pointwise mixers, and their
distinction is the heart of the design. ConvFFN1 mixes across the **feature** dimension D within each
variable: a grouped 1×1 conv with `groups = M`, so each variable's features mix among themselves but
variables stay separate — the standard per-variable feature-mixing FFN. ConvFFN2 mixes across the
**variable** dimension M: permute so the variable axis is the conv's channel axis and group by D (`groups =
D`), so each feature index mixes across all M variables while features stay separate. ConvFFN2 is the
explicit cross-variable operator every earlier rung lacked — a learnable per-feature interaction across the
MEG sensors, applied at every temporal position. The obvious alternative is cross-variable *attention* —
treat the `M` variables as tokens and self-attend over them per timestep — but on FaceDetection that is a
`144 × 144` map per timestep per layer plus Q/K/V/O projections, and its softmax-normalized dynamic mixing
is exactly the kind of thing these small sets (EthanolConcentration ~260 series) overfit; attention's data
appetite is what pushed me to channel-independence at the PatchTST rung. A grouped 1×1 conv learns a fixed
`M × M` interaction per feature with no softmax and no per-timestep dynamics, and composes cleanly with the
convolutional grammar of the rest. So I keep the whole architecture convolutional.

The parameter arithmetic shows where the cost of cross-variable modelling lands. Both FFNs are two-layer 1×1
convs with a GELU and dropout between, `D → ffn_ratio·D → D` with `ffn_ratio = 1` (hidden width = `D = 64`,
lean on the small sets). ConvFFN1's `groups = M` maps each variable's `D` features to `D` at `D² = 4,096` per
group — `4,096 · M` per layer, *linear* in M (FaceDetection ~590k). ConvFFN2's `groups = D` mixes across all
M variables at `M²` per group — `D · M²` per layer, *quadratic* in M (FaceDetection `64 · 144² ≈ 1.33M`). So
the cross-variable operator is by construction the expensive one exactly on the high-channel dataset — and
that is the point: FaceDetection is where I spend parameters to model 144-way covariance, precisely the
dataset whose flat 0.68 column says the covariance was never modelled. On the two low-channel datasets
ConvFFN2 is nearly free (`64 · 9 ≈ 576` weights). A block is: depthwise large-kernel temporal conv,
BatchNorm, ConvFFN1 (feature mix), ConvFFN2 (variable mix), all residual — cleanly separating the three kinds
of structure the earlier rungs entangled.

A few mechanics. The stem embeds each variable independently: `Conv1d(1, D, kernel=patch_size,
stride=patch_stride)` with the variable axis folded into the batch, so shared weights process every variable
— PatchTST's data-efficiency argument still holds and matters because FaceDetection has 144 variables and
little data per class. With `patch_stride < patch_size` I replicate-pad the tail by `patch_size −
patch_stride` so the last timestep is never dropped, giving `N = seq_len // patch_stride`. Between stages a
BatchNorm then a strided `Conv1d(kernel=downsample_ratio, stride=downsample_ratio)` folds the temporal length
and grows the feature width — the convolutional pyramid that lets the large-kernel reach compound across
scales — and I replicate-pad the tail before downsampling when the length is not divisible, again to avoid
inventing zeros. The stage/block counts are small because the UEA sets are small and a deep stack would
overfit, the same restraint that sized the earlier rungs. Normalization is BatchNorm, not LayerNorm, for the
outlier-dilution reason PatchTST already established, doubly right here because the whole architecture is
convolutional and BatchNorm is its native normalization; each depthwise branch carries its own (which makes
the branch fusion exact) plus a BatchNorm over the feature dimension after the depthwise conv.

The head keeps TimesNet's mask-awareness in spirit but adapts to the convolutional backbone. After the
stages I have `[B, M, D, N_final]`; I apply GELU and dropout, flatten the whole `M · D · N_final`
representation, and project with a single `Linear`. That flatten-and-project is where the class decision is
drawn — but ConvFFN2 inside every block has already done the real channel mixing, so the head no longer
carries the whole cross-channel burden the floor's head did. Size it: the stem strides by 4, so `patch_num =
seq_len // 4` (EthanolConcentration 437, FaceDetection 15, Handwriting 38), and one downsample-by-2 folds
that once to 219 / 8 / 19; times `D = 64` gives `head_nf` per variable, and the head `Linear(M · head_nf,
num_class)` is `3 · 14,016 · 4 ≈ 168k` / `144 · 512 · 2 ≈ 147k` / `3 · 1,216 · 26 ≈ 95k` — all in the
hundred-thousand range, and crucially FaceDetection's head (~147k) is now *smaller* than its per-block
ConvFFN2 (~1.3M), the structural statement I want: the cross-channel work has moved out of the head into a
dedicated in-encoder stage. One wrinkle: the strided stem changes the temporal length from `seq_len` to
`N_final`, so I cannot multiply by `x_mark_enc` at full resolution the way TimesNet did. I choose not to
apply a mask multiply at the downsampled resolution: instead I replicate-pad (not zero-pad) the tail before
every conv, so the padded region carries the boundary value rather than a spurious zero, and the learned
flatten head can suppress that constant tail. An ad-hoc mask multiply at a coarsened resolution would add
complication without a principled gain, so I keep the head plain — GELU, dropout, flatten, linear.

Let me make sure this is not just TimesNet with extra parts. The temporal operator is different: a
large-kernel depthwise 1-D conv with a small-kernel reparam branch instead of the FFT-period 2-D reshape —
long-range reach in the natural 1-D layout where masking and variable length are trivial, and cheaper
because depthwise. The channel handling is the whole point of departure: an explicit cross-variable stage
(ConvFFN2) no earlier rung had, separated from feature mixing (ConvFFN1) and temporal mixing. The stem is
channel-independent with shared weights, keeping PatchTST's data efficiency. So it is genuinely different,
aimed at the cross-channel ceiling that pinned FaceDetection while retaining the convolutional, multi-scale,
mask-tolerant temporal modelling that won Handwriting and EthanolConcentration.

The expectations against TimesNet, dataset by dataset. On FaceDetection (0.6745, the ceiling none of the
three cracked) this is the sharpest test, because ConvFFN2 is the first dedicated cross-variable operator and
FaceDetection's signal *is* cross-variable covariance — I expect it to finally break the high-0.68 ceiling,
and if it does not, then explicit cross-variable mixing is not the missing piece and the ceiling is set by
something else (label noise, the binary task's intrinsic limit). On EthanolConcentration (0.3194, the best so
far) the large-kernel depthwise conv reads the smooth curve at long range without per-window normalization,
so I expect at least parity and plausibly a small gain — nothing here washes out the trend. On Handwriting
(0.3353, the ladder's biggest mover) the phase-recurrent strokes are reachable by a large temporal kernel
plus the small-kernel branch for fine local detail, so I expect parity-to-gain; the risk is that strided
downsampling coarsens the fine gesture detail, in which case the small-kernel branch is what saves it. The
decisive question is FaceDetection: if explicit cross-variable mixing finally moves the one dataset that
defeated every prior rung's head-only channel mixing while holding the other two, then the diagnosis — that
the recurring weakness was the absence of a dedicated cross-channel stage — was correct.
