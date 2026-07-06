69.10, and 71.05 with multi-scale-and-flip test-time augmentation. The encoder–decoder with skips
works — the prediction is dense and the boundaries are no longer mush, and the fact that the ms+flip
augmentation buys almost two full points (69.10 → 71.05, a +1.95 gap) tells me the single-scale
prediction still has real scale-sensitivity and localization jitter that averaging over scales and a
flip smooths out. But before I chase that jitter, look at what that encoder *is*: a stack of plain
two-conv blocks trained from scratch on a few thousand Cityscapes images. It is a perfectly reasonable
feature extractor, but it has never seen ImageNet. Put the data disparity in numbers, because it is the
whole argument: the Cityscapes fine-annotated training set is 2975 images; ImageNet-1k is about 1.28
million. A pretrained ResNet-50 has therefore been taught "what objects look like" from roughly 430×
more labeled images than my encoder ever saw, across a thousand categories rather than nineteen. The
rest of computer vision has converged on one fact from exactly this disparity — that a deep residual
network pretrained on a million labeled images learns features so transferable that almost every
downstream task does better starting from them than from a bespoke encoder trained on the task's own
small set. My U-shaped net is leaving that on the table. Its semantics come entirely from the small,
narrow Cityscapes training set; a pretrained ResNet-50 brings a far richer notion of object identity
before it sees a single street scene.

So the question is whether I can keep the part of the last rung that worked — a fully-convolutional net
that produces a dense per-pixel prediction, with no fully-connected bottleneck collapsing the spatial
grid — but mount it on a *pretrained* deep backbone instead of a from-scratch encoder. The tension is
the same one that started this whole line of work: a pretrained classification backbone is *built to
shrink resolution*. Its deepest stage is at output stride 32. If I just read that off and upsample, I am
back to the blurry baseline from two rungs ago — a 16×32 map on the 512×1024 crop — only now with much
better features. Better semantics on a coarse grid is not obviously better than weaker semantics on a
fine grid; that is exactly the trade I have to get right, so I should not conflate the two things I am
changing. There are two distinct levers here, and I will take them one at a time: *getting the
resolution up inside the backbone*, before the head ever runs, and *what the head does* with the
resulting feature map.

Take the resolution lever first. I do not have to accept output stride 32. The downsampling in the last
backbone stages comes from stride-2 convolutions — each one halves the map. The alternative to striding,
for enlarging receptive field without shrinking the grid, is *dilation*: spread the kernel taps apart by
a factor instead of moving the window in bigger steps. Here is why that is nearly free rather than a
retraining-from-scratch move. A dilated 3×3 conv has the *same* 3×3 = 9 weights as an undilated one; the
only thing that changes is the spacing at which those 9 weights sample the input. So the pretrained
weights are still valid — I am reusing the exact same kernels, merely reading them over a wider stencil
— and I add *zero* new parameters. What I pay is compute, because removing the stride means the later
stages now run on a larger spatial map. Concretely: replace the strides in the last two stages with
dilation and the backbone stops shrinking after stride 8. The map goes from 16×32 (OS-32) to 64×128
(OS-8) on the 512×1024 crop — 4× finer in each dimension, 16× more spatial locations — and every conv in
those dilated stages, plus the whole head on top, now runs over 16× the area. That is the real cost of
the "free" resolution: no parameters, no lost pretraining, but a 16× heavier tail. The concrete backbone
is ResNet-50 with `strides=(1, 2, 1, 1)` and `dilations=(1, 1, 2, 4)`, output stride 8 — R-50-D8.

How far to push this is itself a choice with a cost curve. I could dilate only the last stage and stop
at output stride 16 — a 32×64 map, half as many locations, so cheaper — or dilate three stages down to
output stride 4 for a 128×256 map, finer but heavier still. Output stride 16 keeps the prediction on a
coarse grid where an ×16 upsample owns 16×16 = 256-pixel cells, which is close to the baseline blur I am
trying to get away from; output stride 4 quadruples the already-16×-heavier tail again and, more subtly,
demands even larger dilation rates in the deepest stage to hold the receptive field, and large rates
bring their own problem. That problem is *gridding*: a dilated kernel samples the input only at its
spread-apart taps and skips everything in between, so if I stack dilated convs at a large common rate,
whole sub-lattices of the feature map stop talking to each other — the effective sampling develops
holes. The `dilations=(1, 1, 2, 4)` schedule keeps the rates small and staggered (2 then 4) precisely so
the taps of successive stages interleave and cover the map without gaps. Output stride 8 sits at the
knee of this curve: fine enough that the ×8 upsample's 8×8 cells are much smaller than the objects, deep
enough that two modest dilation rates suffice, and only 16× — not 64× — heavier than the OS-32 tail.

Let me sanity-check the receptive-field claim, because "same receptive field" is the thing that lets the
pretrained weights transfer. In the original backbone, stage 4 sat behind two stride-2 reductions, so a
unit there pooled over a wide swath of the input through the downsampling. When I delete those strides
and dilate instead, a stage-4 unit still reaches across a comparable input extent — the taps of its
now-dilated kernels land far apart in input pixels precisely because the earlier downsampling that used
to spread them is replaced by explicit dilation. The abstraction the pretrained stage learned is a
function of the spatial pattern of inputs it integrates, and that pattern is preserved; what changes is
that the integration now produces an output at *every* fine location instead of one per coarse cell. So
the semantics survive dilation, which is the whole reason this is a weight-preserving edit and not a new
network.

Let me make the tap arithmetic explicit on one kernel, because "same receptive field" is easy to assert
and worth actually checking. A 3×3 conv at dilation 1 samples input offsets {−1, 0, +1} in each axis —
its taps span 2 pixels. Give it dilation 2 and the same nine weights now sample offsets {−2, 0, +2},
spanning 4 pixels; at dilation 4, offsets {−4, 0, +4}, spanning 8. In the original backbone the stage I
am now dilating sat *after* a stride-2 reduction, so one output step there corresponded to 2 input
pixels — its undilated 3×3 already reached across 4 input pixels through that downsampling. Remove the
stride and set dilation 2, and a single output step is back to 1 input pixel, but the dilation-2 taps
again reach across 4 input pixels — the same input span, now evaluated at every fine location instead of
every other one. That is the swap working exactly as intended: the kernel integrates the same spatial
footprint of input, the pretrained weights read the same relative pattern they were trained on, and the
only change is that the output is dense. The `(1, 1, 2, 4)` schedule is precisely the sequence of rates
that keeps each dilated stage's footprint matched to the stride it replaced.

Now the head lever. At output stride 8 the feature map is rich — 2048 channels of pretrained semantics —
and reasonably fine. I want to turn those 2048 channels into 19 class scores per location, and I have a
choice about *how much* the head should do. Option one, the minimal head: a single 1×1 conv straight to
19 classes, then upsample ×8. This is the cleanest test of "do pretrained features at OS-8 already
suffice," but a pointwise 1×1 makes each pixel's decision in isolation — it cannot use the neighborhood
to clean up a locally noisy prediction. Option two: a couple of 3×3 conv+norm+ReLU over the deep map
before the 1×1, so the head can do learned local smoothing and refinement — resolve a flickering pixel
by looking at its neighbors — which a 1×1 cannot. Option three, the maximal head: reinstate the full
U-Net decoder on top of the pretrained backbone, climbing all the way back to full resolution with
per-level skips. That is tempting because the last rung showed how much localization the decoder buys —
but it is the wrong tool *here*, for two reasons I can name concretely. First, at OS-8 the map is already
4× finer than the OS-32 baseline that made the U-Net necessary; the boundary problem the full decoder was
built to solve is much milder now, so most of that machinery would be recovering detail the dilated
backbone already kept. Second, a pretrained ResNet's stages do not form the clean symmetric U that
per-level concatenation wants — their channel widths and resolutions were designed for classification,
not for a mirror-image decoder — so bolting a full decoder on is both awkward and, given the mild
boundary gap, poor value for its compute. So I take option two: a light, learned, local head, and I add
only the *one* piece of the U-Net lesson that is cheap and clearly worth it.

That one piece: reading off only the very deepest processed map throws away a less-processed, more
faithful copy of the features. In the U-Net the fix was per-level skips; here, in miniature, I run the
3×3 convs and then concatenate their output with the head's *input* feature before the final classifier,
so the 1×1 sees both the refined features and the original deep features side by side. It is a single
"concat the input" skip rather than a full decoder — the dilated backbone has already kept the
resolution I need, so one skip at the head suffices. The whole thing stays convolutional end to end:
every layer is a convolution, the spatial grid is never collapsed into a vector, so the net accepts any
input size and emits a correspondingly-sized dense map, and the ×8 "upsampling" back to full resolution
is itself an in-network resize the loss is computed through — so the net learns dense prediction directly
against the dense labels rather than against some downsampled proxy.

It is worth being explicit about why staying fully convolutional matters, because the alternative the
baselines offer is the per-patch classifier — run a classification CNN on a fixed-size window centered
at each pixel and take its class. That is not just clumsy, it is computationally hopeless and
statistically wasteful: it would require on the order of one forward pass per pixel — hundreds of
thousands per crop — and each pass recomputes almost the same convolutions its neighbors just computed,
because adjacent windows overlap by all but a row. A fully-convolutional net does the entire dense
prediction in a *single* pass over the image, sharing every convolution across all output locations by
construction; the sliding window is already implicit in a conv. So keeping the grid uncollapsed is not
only about preserving resolution — it is what turns an intractable per-pixel loop into one forward pass,
and it is the property I most want to carry forward from the last rung onto the new backbone.

Let me write the head and check its channel arithmetic, because the concat has to line up. A couple of
3×3 conv blocks, an optional concat of the input, then the 1×1 classifier:

```python
class FCNHead(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True,
                 dilation=1, **kwargs):
        super().__init__(**kwargs)
        conv_padding = (kernel_size // 2) * dilation
        convs = [ConvModule(self.in_channels, self.channels, kernel_size,
                            padding=conv_padding, dilation=dilation,
                            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]
        for _ in range(num_convs - 1):
            convs.append(ConvModule(self.channels, self.channels, kernel_size,
                                    padding=conv_padding, dilation=dilation,
                                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*convs)
        if concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels,
                                       self.channels, kernel_size,
                                       padding=kernel_size // 2,
                                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)   # the backbone's deep (OS=8) feature
        feats = self.convs(x)                # learned local refinement
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))  # skip: refined + raw
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)          # 1x1 -> 19 classes, then upsample x8
```

Trace the channels. The head's input `x` is the OS-8 deep feature, 2048 channels. The first 3×3 conv
maps 2048 → `channels` (I will use 512); the second maps 512 → 512. So `feats` is 512 channels of
locally-refined feature. The concat `torch.cat([x, feats])` stacks the raw 2048 on the refined 512, giving
2560 channels — which is exactly why `conv_cat` is built to take `self.in_channels + self.channels` =
2048 + 512 = 2560 inputs and fold them back to 512. The final 1×1 `conv_seg` takes 512 → 19, and the ×8
bilinear resize lifts the 64×128 score map to 512×1024. Every dimension closes; the concat is not a
resolution fusion (both tensors are at OS-8) but a *processing-depth* fusion — refined-and-neighborhood-
aware alongside raw-and-faithful — which is the miniature of the U-Net skip appropriate to a backbone
that already kept its resolution.

The choice of 512 head channels and two convs is a compute decision, not a free parameter. The first
3×3 conv reducing 2048 → 512 costs about 9·2048·512 ≈ 9.4M multiply-adds per output location, and there
are 64×128 ≈ 8200 locations, so that one layer is on the order of 77G MACs — already a substantial
fraction of the whole head, run over the 16×-enlarged OS-8 map. Keeping the head at 512 channels
(a 4× reduction off the 2048-wide backbone feature) and only two convs holds this in check; widening to
2048-channel head convs would roughly quadruple the head cost for a refinement gain I have no reason to
expect is worth it. So the head is deliberately a *thin* learned cap on a *strong* backbone: most of the
representational work is the pretrained ResNet's, and the head only has to project and locally clean up.

The prediction, against 69.10, and I should be honest that two things move in opposite directions on
the two axes I separated. On the *what* axis the head is now reading a pretrained ResNet-50 rather than a
from-scratch encoder, so the semantics are far stronger — whole-object misclassifications, the kind
where a from-scratch net simply had not learned the category well from 2975 images, should drop broadly
across classes. On the *where* axis, though, I have deliberately given up the U-Net's full-resolution
decoder: at OS-8 the prediction is produced on a 64×128 grid and then blown up ×8, so each output cell
still owns an 8×8 = 64-pixel patch, and the finest boundaries — the pole silhouettes, the pedestrian
legs — will be softer than the U-Net's full-resolution decoder drew them. So this rung is a bet that the
semantics win the class-averaged metric even while conceding some of the finest boundary sharpness: the
pretrained features fix a broad swath of whole-region errors, and the OS-8 grid is fine enough that the
boundary loss is a smaller effect than the semantic gain. I expect the mIoU to clear 69.10; I do not
know by how much, and the honest uncertainty is precisely whether the boundary concession eats into the
semantic gain — the val mIoU will tell.

I can be a little more precise about where on the class average the two effects should land, since mIoU
weights all 19 classes equally. The pretrained-semantics gain should show up broadly but concentrate on
classes the from-scratch encoder had too few examples to learn well — the rarer object categories, the
ones with only a handful of instances per image, where 2975 images is genuinely thin training data and
1.28M ImageNet images of related objects is a real prior. The boundary concession, by contrast, costs me
on the thin/elongated classes whose IoU is dominated by their edges, where an ×8 upsample is coarser
than a full-resolution decoder. So the two effects act on partly different classes, which is why I
expect the net movement to be positive: I am gaining broadly on semantics and conceding narrowly on the
finest boundaries, and on an equal-weight average the broad gain should outweigh the narrow concession.
If instead the mIoU barely moves or drops, the read would be that OS-8 is too coarse to bank the
semantic gain and I have conceded too much localization — but I will not presume what to do about it
before the number is in and I can see which classes actually moved.

One training aid the fixed substrate already offers, and that this backbone makes worth taking, is an
auxiliary cross-entropy head on an intermediate stage. The dilated ResNet-50 is deep, and I am asking
the main head to drive gradients back through all of it while the backbone is simultaneously adapting
its pretrained weights to street scenes. Hanging a second, lightweight FCN-style head off an earlier
stage — say the output-stride-8 stage-3 feature — with its own per-pixel loss gives the deep middle of
the network a shorter path to a supervision signal, so those stages get a cleaner gradient than the one
that has to survive the whole climb from the final loss. It is a pure training-time scaffold: it adds no
cost at inference (I drop it for evaluation) and does not touch the architecture whose mIoU I am
reporting. I mention it because it is the standard way to keep a deep segmentation backbone trainable,
and I would use it here.

One decision the "pretrained backbone" framing hides is whether to *freeze* the pretrained weights or
fine-tune them. Freezing is tempting on a 2975-image set — fewer parameters to fit, less overfitting risk
— but it is the wrong call here, because ImageNet features were learned for whole-image classification,
where the network is rewarded for becoming *invariant* to exactly the fine spatial detail segmentation
needs to keep. Left frozen, the backbone would hand the head classification-tuned features and never adapt
them to dense prediction. So I fine-tune the whole backbone end to end at a modest learning rate, with
synchronized batch-norm across devices so the norm statistics are estimated over a real batch of crops
rather than a handful per device — the pretraining is a strong *initialization* to move away from, not a
fixed feature extractor to read off. This is the mirror of last rung's problem: there I had adaptation but
no prior; here I have a strong prior and let the 40k schedule adapt it.

There is a second prediction I can make from the metric pair, using what the last rung showed. The
U-Net's ms+flip gap was large, +1.95, which I read as a from-scratch, fine-detail decoder being
scale-sensitive — its predictions moved a lot when the input scale changed. A pretrained backbone at a
fixed moderate output stride should be *more* scale-robust: pretrained features generalize across scale
better, and there is no full-resolution decoder amplifying scale-dependent jitter. So I expect this
rung's single-scale-to-ms+flip gap to be *narrower* than the U-Net's +1.95 — the multi-scale
augmentation should have less to fix. I only have the single-scale value to reason from now; the ms+flip
column will tell whether that read is right. The concrete change is the dilated pretrained backbone plus
the fully-convolutional refine-and-classify head with a single input skip; the full head is in the
answer.
