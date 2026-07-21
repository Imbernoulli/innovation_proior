69.10, and 71.05 with multi-scale-and-flip test-time augmentation. The encoder–decoder with skips works —
the prediction is dense and the boundaries are no longer mush — and the ms+flip augmentation buying almost
two full points (a +1.95 gap) tells me the single-scale prediction still has real scale-sensitivity and
localization jitter that averaging over scales and a flip smooths out. But before I chase that jitter, look
at what that encoder *is*: a stack of plain two-conv blocks trained from scratch on a few thousand
Cityscapes images. A perfectly reasonable feature extractor, but it has never seen ImageNet. Put the data
disparity in numbers, because it is the whole argument: the Cityscapes fine-annotated training set is 2975
images; ImageNet-1k is about 1.28 million. A pretrained ResNet-50 has been taught "what objects look like"
from roughly 430× more labeled images, across a thousand categories rather than nineteen. Vision has
converged on one fact from exactly this disparity — a deep residual network pretrained on a million images
learns features so transferable that almost every downstream task does better starting from them than from
a bespoke encoder trained on the task's own small set. My U-shaped net leaves that on the table: its
semantics come entirely from the narrow Cityscapes set.

So can I keep what worked — a fully-convolutional net producing a dense per-pixel prediction, no
fully-connected bottleneck collapsing the grid — but mount it on a *pretrained* deep backbone? The tension
is the one that started this line of work: a pretrained classification backbone is built to shrink
resolution, its deepest stage at output stride 32. Read that off and upsample and I am back to the blurry
baseline, a 16×32 map on the crop, only now with much better features — and better semantics on a coarse
grid is not obviously better than weaker semantics on a fine grid. There are two distinct levers, and I
take them one at a time: getting the resolution up *inside* the backbone before the head ever runs, and
*what the head does* with the result.

The resolution lever first. The downsampling in the last backbone stages comes from stride-2 convolutions,
each halving the map. The alternative for enlarging receptive field without shrinking the grid is
*dilation*: spread the kernel taps apart instead of moving the window in bigger steps. This is nearly free
rather than a retrain-from-scratch move. A dilated 3×3 conv has the *same* 9 weights as an undilated one;
only the spacing at which they sample the input changes. So the pretrained weights stay valid — I reuse the
exact kernels, read over a wider stencil — and I add zero new parameters. What I pay is compute, because
removing the stride means the later stages run on a larger map. Replace the strides in the last two stages
with dilation and the backbone stops shrinking after stride 8: the map goes from 16×32 (OS-32) to 64×128
(OS-8) on the crop — 4× finer each way, 16× more spatial locations — and every conv in those stages plus
the whole head now runs over 16× the area. The concrete backbone is ResNet-50 with `strides=(1,2,1,1)`,
`dilations=(1,1,2,4)`, output stride 8 — R-50-D8.

How far to push is a cost curve. Dilate only the last stage and stop at OS-16 — a 32×64 map, cheaper — but
an ×16 upsample owns 16×16 = 256-pixel cells, close to the baseline blur I am escaping. Dilate three
stages to OS-4 for a 128×256 map — finer, but it quadruples the already-16×-heavier tail again and demands
larger dilation rates in the deepest stage to hold the receptive field, and large rates bring *gridding*: a
dilated kernel samples only at its spread-apart taps, so stacking convs at a large common rate lets whole
sub-lattices of the feature map stop talking to each other. The `(1,1,2,4)` schedule keeps the rates small
and staggered (2 then 4) so successive stages' taps interleave and cover the map without gaps. OS-8 sits at
the knee: fine enough that the ×8 upsample's 8×8 cells are much smaller than the objects, deep enough that
two modest rates suffice, and only 16× — not 64× — heavier than the OS-32 tail.

The receptive-field preservation is what lets the pretrained weights transfer, so make the tap arithmetic
explicit. A 3×3 conv at dilation 1 samples offsets {−1, 0, +1}, its taps spanning 2 pixels; at dilation 2,
{−2, 0, +2}, spanning 4; at dilation 4, offsets spanning 8. In the original backbone the stage I am now
dilating sat *after* a stride-2 reduction, so one output step corresponded to 2 input pixels and its
undilated 3×3 already reached across 4 input pixels through that downsampling. Remove the stride and set
dilation 2, and a single output step is back to 1 input pixel while the dilation-2 taps again reach across
4 input pixels — the same input span, now evaluated at every fine location instead of every other. The
kernel integrates the same spatial footprint, the pretrained weights read the same relative pattern they
were trained on, and the only change is that the output is dense. So the semantics survive dilation, which
is why this is a weight-preserving edit and not a new network.

Now the head. At OS-8 the map is rich — 2048 channels of pretrained semantics — and reasonably fine, and I
have a choice about how much the head should do. The minimal head, a single 1×1 straight to 19 classes then
×8, is the cleanest test of "do pretrained OS-8 features already suffice," but a pointwise 1×1 decides each
pixel in isolation and cannot use the neighborhood to clean up a locally noisy prediction. The maximal head
reinstates the full U-Net decoder, climbing back to full resolution with per-level skips — tempting given
how much localization the decoder bought last time, but wrong here for two concrete reasons. First, at OS-8
the map is already 4× finer than the OS-32 baseline that made the U-Net necessary, so most of that
machinery would be recovering detail the dilated backbone already kept. Second, a pretrained ResNet's
stages — widths and resolutions designed for classification — do not form the clean symmetric U that
per-level concatenation wants, so bolting a full decoder on is both awkward and poor value for its compute.
So I take the middle option: a couple of 3×3 conv+norm+ReLU over the deep map for learned local refinement —
resolving a flickering pixel by looking at its neighbors, which a 1×1 cannot — plus the one cheap piece of
the U-Net lesson clearly worth keeping.

That piece: reading off only the deepest processed map throws away a less-processed, more faithful copy of
the features. In miniature, I run the 3×3 convs and then concatenate their output with the head's *input*
feature before the final classifier, so the 1×1 sees refined and raw deep features side by side — a single
"concat the input" skip rather than a full decoder, because the dilated backbone already kept the
resolution I need. And the whole thing stays convolutional end to end, which is not just tidiness. The
alternative the baselines offer is the per-patch classifier — run a classification CNN on a window centered
at each pixel and take its class — which is computationally hopeless: on the order of one forward pass per
pixel, hundreds of thousands per crop, each recomputing almost the same convolutions its overlapping
neighbors just did. A fully-convolutional net does the entire dense prediction in a single pass, sharing
every convolution across all output locations by construction; the sliding window is already implicit in a
conv. And the ×8 "upsampling" back to full resolution is an in-network resize the loss is computed through,
so the net learns dense prediction directly against the dense labels rather than a downsampled proxy.

The head is a couple of 3×3 conv blocks, an optional concat of the input, then the 1×1 classifier — the
full module is in the answer; trace the channels, because the concat has to line up. The input `x` is the
OS-8 deep feature, 2048 channels; the first 3×3 maps 2048 → `channels` (512), the second 512 → 512, so the
refined `feats` is 512 channels. The concat stacks the raw 2048 on the refined 512 = 2560 channels, which
is exactly why `conv_cat` takes `self.in_channels + self.channels` = 2560 and folds back to 512; the 1×1
classifier maps 512 → 19 and the ×8 resize lifts the 64×128 score map to 512×1024. Every dimension closes.
The concat is not a resolution fusion (both tensors are at OS-8) but a *processing-depth* fusion — refined
and neighborhood-aware alongside raw and faithful — the miniature of the U-Net skip appropriate to a
backbone that already kept its resolution.

The 512 head channels and two convs are a compute decision. The first 3×3 reducing 2048 → 512 costs about
9·2048·512 ≈ 9.4M multiply-adds per location over 64×128 ≈ 8200 locations — on the order of 77G MACs, run
over the 16×-enlarged OS-8 map. Holding the head at 512 channels (a 4× reduction off the 2048-wide backbone
feature) and only two convs keeps this in check; widening to 2048-channel head convs would roughly
quadruple the head cost for a refinement gain I have no reason to expect is worth it. The head is
deliberately a thin learned cap on a strong backbone: most of the representational work is the pretrained
ResNet's, and the head only projects and locally cleans up.

The prediction, against 69.10, and two things move in opposite directions on the two axes I separated. On
the *what* axis the head reads a pretrained ResNet-50 rather than a from-scratch encoder, so the semantics
are far stronger; whole-object misclassifications — the kind where a from-scratch net had not learned the
category from 2975 images — should drop broadly, concentrating on the rarer object categories where 2975
images is genuinely thin and 1.28M ImageNet images of related objects is a real prior. On the *where* axis I
have given up the U-Net's full-resolution decoder: at OS-8 the prediction is produced on a 64×128 grid then
blown up ×8, so each cell owns an 8×8 = 64-pixel patch and the finest boundaries — pole silhouettes,
pedestrian legs — will be softer than the U-Net drew them, costing me on the thin/elongated classes whose
IoU is dominated by edges. The two effects act on partly different classes, which is why I expect the net
movement to be positive: gaining broadly on semantics, conceding narrowly on the finest boundaries, and on
an equal-weight average the broad gain should outweigh the narrow concession. I expect the mIoU to clear
69.10; the honest uncertainty is whether the boundary concession eats the semantic gain, and the val mIoU
will tell.

Two training details the backbone makes worth stating. The fixed substrate offers an auxiliary
cross-entropy head on an intermediate stage; the dilated ResNet-50 is deep and I am driving gradients back
through all of it while it adapts its pretrained weights to street scenes, so hanging a lightweight
FCN-style head off the OS-8 stage-3 feature with its own loss gives the deep middle a shorter path to
supervision than the one that has to survive the whole climb from the final loss. It is a training-only
scaffold, dropped at evaluation, and does not touch the architecture whose mIoU I report. And whether to
*freeze* the pretrained weights or *fine-tune* them: freezing is tempting on 2975 images, but ImageNet
features were learned for whole-image classification, where the net is rewarded for becoming *invariant* to
exactly the fine spatial detail segmentation needs to keep. Left frozen, the backbone would hand the head
classification-tuned features and never adapt them to dense prediction. So I fine-tune the whole backbone
end to end at a modest learning rate, with synchronized batch-norm so the statistics are estimated over a
real batch of crops — the pretraining is a strong *initialization* to move away from, not a fixed feature
extractor. This is the mirror of last time's problem: there I had adaptation but no prior; here a strong
prior that the 40k schedule adapts.

One prediction from the metric pair. The U-Net's ms+flip gap was large (+1.95), which I read as a
from-scratch fine-detail decoder being scale-sensitive; a pretrained backbone at a fixed moderate output
stride, with no full-resolution decoder amplifying scale-dependent jitter, should be more scale-robust, so
I expect this step's single-scale-to-ms+flip gap to be *narrower* than +1.95 — the multi-scale
augmentation should have less to fix. I only have the single-scale value to reason from now. The concrete
change is the dilated pretrained backbone plus the fully-convolutional refine-and-classify head with a
single input skip; the full head is in the answer.
