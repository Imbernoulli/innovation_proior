77.85, and 79.18 with multi-scale-and-flip — up +5.60 from the FCN head's 72.25, the biggest jump on
the ladder so far. So context was indeed the dominant remaining problem, and the metric pair backs the
mechanism: the ms+flip gap moved only modestly, from +1.11 to +1.33, which is what I predicted for a gain
that comes from scene-level reasoning rather than boundary sharpness — the improvement is already there at
single scale, not something test-time augmentation had to rescue. Good. Now look closely at *how* the
pyramid gets its context and what that costs, because a +5.60 jump that leaves a clear structural wart is
an invitation to find the next lever.

Each pyramid level adaptive-average-pools the feature map down to a tiny grid — 1×1, 2×2, 3×3, 6×6 — then
upsamples the summary back. Pooling to a 2×2 grid means I have averaged everything in each quadrant into
one number per channel; the *within-region* spatial detail of the context is gone, and when I bilinearly
broadcast that quadrant summary back to full resolution it is a smooth blanket that knows nothing about
where the boundaries inside the quadrant are. So the pyramid buys global reach by *destroying resolution*
to get it, then upsampling. The context is right but spatially coarse, and that coarseness shows up most
at object boundaries — exactly the place a smooth broadcast blanket has nothing to say. It worked, and it
worked well, but it is context computed at the wrong resolution and then blurred back up.

Two problems are tangled together here, and I should pull them apart before designing, because a fix for
one is not automatically a fix for the other. First: can I gather multi-scale context *without* pooling
resolution away — get the same global reach the pyramid gave me but keep a distinct value at every pixel
while I gather it? Second, and separate: regardless of how I get context, the whole head still produces
its prediction at output stride 8 and then bilinearly upsamples by 8 to full resolution — so the finest
boundaries, the thin poles, the exact silhouette of a pedestrian, are blurry no matter how good the
context is, because an 8× bilinear upsample of a 64×128 map cannot draw a sharp edge inside an 8×8 cell.
The U-Net solved that second problem with a full decoder; I dropped the decoder when I moved to the
dilated backbone, and the pyramid never addressed it. Maybe I gave up too much there. So I have a context
problem and a boundary problem, and I will fix them with two separate pieces.

Take the first problem. I want context at multiple scales, but applied *densely*, at the feature map's
own resolution, not via pooling. The tool I already have for "see farther without downsampling" is
dilation — it is exactly what made the backbone output stride 8 in the first place, and it enlarges the
window of a conv while keeping a value at every location. So instead of pooling into grids, run *several
convolutions in parallel over the same OS-8 feature map, each at a different dilation rate*. Work out what
each rate reaches. A 3×3 conv at rate r places its taps at offsets {−r, 0, +r}, so it spans 2r+1 taps and
reaches ±r feature-map cells from center. At rate 12 that is ±12 cells = 24 cells across = 192 image
pixels at OS-8; rate 24 spans 48 cells = 384 image pixels; rate 36 spans 72 cells = 576 image pixels —
wider than the 512-tall crop. Add a rate-1 branch (a plain 1×1) for the purely-local view, and I have a
*pyramid of dilations* rather than a pyramid of pools: four branches sampling context from ~1 pixel out
to wider-than-the-crop, all at full feature resolution, no pooling, every output pixel keeping its
location. Concatenate the branches and fuse with a 1×1 conv.

The reason dilation-instead-of-pooling helps is precisely the within-region detail that pooling threw
away: a rate-12 atrous conv sees far-apart context but still produces a *distinct* value at every pixel,
computed from that pixel's own wide neighborhood, so context near a boundary is a real local computation
rather than a single regional average smeared back. Picture a pixel right at the road/sidewalk seam. Under
the pooling pyramid, its 2×2 and 3×3 context cells each hand it the *average* of a region that straddles
the seam — a blend of road and sidewalk statistics that is, by construction, ambiguous exactly where I
need it sharpest. Under the atrous pyramid, that same pixel's rate-12 branch reads its own offset taps —
some landing on definite road, some on definite sidewalk — and passes those distinct samples to the
fusion, which can resolve the seam from the pattern rather than from a pre-averaged blur. That is the
concrete meaning of "keep the resolution while gathering context": the context tensor still has an edge in
it at the edge, instead of a smooth ramp. But there is a catch in the arithmetic I just did,
and it is worth catching now rather than being surprised by it. Push the dilation rate too high and the
3×3 kernel's outer taps fall off the feature map. Check it on the real map: at OS-8 the feature is 64×128.
A rate-36 branch, at any row of the 64-tall map, wants vertical taps at row±36 — but from a center row r,
r−36 ≥ 0 needs r ≥ 36 and r+36 ≤ 63 needs r ≤ 27, which cannot both hold, so *every* row has at least one
vertical tap off the map, hitting zero-padding. With two of its three vertical taps effectively dead, the
rate-36 conv degenerates toward a 1×1 that mostly reads its center tap. Even rate-24 (±24) only keeps
both vertical taps in range for rows 24…39, a thin central band. So the largest atrous branches do not
actually deliver honest wide context on the short axis of this crop — they collapse toward pointwise.
That is a real limitation, and it names the fix: I keep a single image-level global-average-pool branch
alongside the atrous branches, broadcast back, as the one *honest* whole-image summary — the same global
cue the pooling pyramid's 1×1 level gave me — rather than pretending the rate-36 branch supplies it. So
the module is: 1×1 branch, three atrous branches at rates 12/24/36, and one global-pool branch, all
concatenated and fused.

Why 12/24/36 specifically, and not the smaller ladder one might reach for first? The rate has to be read
relative to the output stride, because a rate-r atrous conv reaches ±r *feature-map* cells, and at OS-8
each cell is 8 input pixels. The canonical atrous ladder for this kind of module is tuned at output
stride 16; run the same physical reach at OS-8 — half the cell size, so twice the cells per pixel — and
the rates have to double to cover the same image extent. Hence 12/24/36 rather than 6/12/18: the rates
scale with the resolution I chose for the backbone, and choosing OS-8 for the finer prediction forces the
larger rates, which is what pushed the biggest branch into the partial-degeneration I just checked. The
three rates are geometrically spaced — each doubles the reach of the last — so the branches tile the scale
axis roughly evenly rather than clustering, the same geometric-spacing logic the pooling pyramid used, now
on dilation reach instead of pool-grid count.

There is a cost problem to pay attention to: three parallel 3×3 atrous convs over a 2048-channel OS-8 map
is heavy. Price one of them — a standard 3×3 from 2048 → 512 is 3·3·2048·512 ≈ 9.4M multiply-adds per
location, times three branches, over 8200 locations. I can cut this sharply with *depthwise-separable*
convolution: factor each 3×3 atrous conv into a depthwise 3×3 (one filter per channel — the spatial and
atrous part) followed by a pointwise 1×1 (the cross-channel mixing). The depthwise part costs 3·3·2048 =
18.4k per location and the pointwise 2048·512 ≈ 1.05M, totalling ≈ 1.07M against the dense conv's 9.4M —
about a 9× reduction, matching the general ratio (9+C_out)/(9·C_out) ≈ 1/9 for C_out = 512. Same
receptive field, a fraction of the multiply-adds, and the dilation lives entirely in the cheap depthwise
part. So the atrous pyramid becomes a *separable* atrous pyramid, and the three-branch context module
costs about what one dense branch would have.

That separability is what keeps the whole head from being more expensive than the pyramid it replaces,
even though I am now running two modules (context *and* decoder) instead of one. The pooling pyramid's
dominant cost was a dense 3×3 over 4096 → 512 ≈ 18.9M MACs per OS-8 location. Here the three dense atrous
branches would have been ≈ 3·9.4M ≈ 28M per location, already more than the whole PSPNet head; made
separable they drop to ≈ 3·1.07M ≈ 3.2M, plus the 1×1 local branch and the 1×1 fusion. The decoder adds
its two separable 3×3 convs, but those run at OS-4 on only 512+48 channels and are themselves separable,
so they are cheap per location even at 4× the pixel count. Netting it out, the separable ASPP plus decoder
lands in the same rough compute class as the pooling-pyramid head — I am buying dense multi-scale context
and a boundary decoder for roughly the price of the pooled context alone, and depthwise-separable
convolution is the entire reason that trade is available.

Now the second problem — the boundaries. The context module gives me a strong OS-8 feature, but I am
still one 8× bilinear jump from the labels, and no amount of context sharpness survives an 8× blur.
Rather than reinstate the whole U-Net — five symmetric levels, per-level skips — I ask what the *minimum*
decoder is that recovers boundaries on this backbone. The boundary information lives in the *shallow*
backbone stage: the OS-4 feature, which is sharp but low-level. So I add exactly one skip. Take the rich
context feature, bilinearly upsample it from OS-8 to OS-4 — a 2× jump, not 8× — grab the OS-4 backbone
feature, run a 1×1 conv on it to squeeze its channel count down, concatenate the two, and refine with a
couple of 3×3 separable convs. Then the classifier, then a final 4× upsample to full resolution. It is
the U-Net skip idea, but a *single* skip at OS-4 bolted onto the dilated backbone — semantics from the
context module, boundary detail from the shallow skip, fused once.

Why one skip and not the full U-Net stack? Because the situations are different in a way the arithmetic
makes plain. The U-Net needed five levels because its prediction was born at OS-16/32 and had to climb all
the way back through resolutions the from-scratch encoder had thrown away. Here the context feature is
already at OS-8 — it only has to climb to OS-4 and then upsample ×4 — and the dilated backbone kept the
intermediate resolution the U-Net had to reconstruct. The boundary headroom left is the difference
between an ×8 blur and an ×4 blur plus one sharp skip, and a single OS-4 fusion captures most of that; a
second or third skip would be recovering ever-finer detail against ever-smaller returns while adding
decoder levels the backbone does not cleanly provide. So one skip is the value pick, not a compromise.

There is a tempting shortcut for the boundary problem that I should rule out explicitly: instead of a
decoder skip, just dilate the backbone one stage further, to output stride 4, and shrink the final
upsample from ×8 to ×4. That keeps the prediction on a finer native grid with no decoder at all. But it
fails on both cost and substance. On cost, going to OS-4 runs the whole dilated tail *and the entire ASPP
module* over a 128×256 map instead of 64×128 — 4× the locations, so roughly 4× the head compute, on top
of a heavier backbone tail, and it would force even larger atrous rates to hold the reach, deepening the
degeneration problem. On substance, an OS-4 backbone feature is still a *deep, context-heavy* feature that
has been smoothed by all the semantic processing above it; making it denser does not give it sharp
low-level edges. The decoder skip does something the dilation cannot: it reaches back to the *shallow*
OS-4 stage, which still carries crisp edge and texture response the deep feature has abstracted away. So
the skip is not merely the cheaper boundary fix, it is the one that supplies a genuinely different, sharper
signal — dilating the backbone to OS-4 would cost far more and still leave me upsampling a blurred deep
feature.

Why reduce the skip's channels before fusing? This is the crisp-but-mislabeled risk from the U-Net rung,
and here it is acute because the OS-4 feature is *low-level* — edges and textures with weak semantics. If
I concatenate it at full width against the context feature, its many channels could dominate the fusion
and pull the prediction toward sharp-but-wrong. So I 1×1-reduce it to 48 channels, deliberately small,
before the concat: set against the 512-channel context feature that is roughly a 10:1 ratio in favor of
semantics, so the skip contributes geometry — *where* the edge is — without outvoting the context on
*what* the region is. That channel ratio is the hedge, and the separable-conv fusion that follows is what
learns to read edge from the skip and class from the context.

Which shallow stage should the skip come from is itself a choice on a short spectrum. The backbone offers
an OS-2 stage (sharpest, but almost raw pixels — barely more than an edge map, weak enough that even
48 channels of it could inject noise), the OS-4 stage (still sharp, but one block deeper so it has begun
to form low-level parts rather than bare gradients), and the OS-8 stage (which is where the context
feature already lives, so no new detail). OS-4 is the sweet spot: fine enough that upsampling the context
to it is only a 2× step and the final upsample only ×4, yet processed enough that its features are
low-level *structure* rather than pixel noise. That is why the decoder reaches to `inputs[0]` at OS-4 and
not to a shallower or deeper stage — it is the shallowest feature that still carries structure worth
trusting.

Let me write the head. The separable atrous pyramid (the dilation>1 branches become depthwise-separable):

```python
class DepthwiseSeparableASPPModule(ASPPModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels, self.channels, 3,
                    dilation=dilation, padding=dilation,
                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
```

and the head: build the context feature from the image-pool branch plus the atrous branches, then the
single OS-4 decoder skip:

```python
def forward(self, inputs):
    x = self._transform_inputs(inputs)
    aspp_outs = [resize(self.image_pool(x), size=x.size()[2:],   # global branch
                        mode='bilinear', align_corners=self.align_corners)]
    aspp_outs.extend(self.aspp_modules(x))      # parallel atrous branches (rates 1,12,24,36)
    aspp_outs = torch.cat(aspp_outs, dim=1)
    output = self.bottleneck(aspp_outs)         # fuse the pyramid -> context feature
    if self.c1_bottleneck is not None:
        c1_output = self.c1_bottleneck(inputs[0])   # low-level OS-4 skip, channel-reduced
        output = resize(output, size=c1_output.shape[2:],   # context up OS-8 -> OS-4
                        mode='bilinear', align_corners=self.align_corners)
        output = torch.cat([output, c1_output], dim=1)
    output = self.sep_bottleneck(output)        # fuse context + detail (separable convs)
    return self.cls_seg(output)
```

Trace the decoder shapes to be sure the fusion lands where I think. The context feature `output` is 512
channels at OS-8 = 64×128. The OS-4 backbone feature `inputs[0]` is 256 channels at 128×256; the
`c1_bottleneck` 1×1 reduces it to 48 channels at 128×256. `resize` lifts the context 64×128 → 128×256, so
both tensors are now 128×256 and the concat gives 512 + 48 = 560 channels at OS-4. The `sep_bottleneck`'s
two separable 3×3 convs fold 560 → 512, the 1×1 classifier maps 512 → 19, and the final ×4 resize lifts
128×256 → 512×1024. Every size closes, and the largest single upsample anywhere in the head is now ×4, not
×8 — the boundary blur is halved before the skip even contributes its sharp edges.

The prediction, against 77.85, and I should temper it. Two things should each add a little. The atrous
pyramid gathers the same multi-scale context as the pooling pyramid but *without* averaging away
within-region resolution, so the context near boundaries is sharper — but the pooling pyramid already
captured most of the *available* context gain (that is what the +5.60 was), so the delta from
dense-versus-pooled context is a refinement, not a second big jump. And the single OS-4 decoder skip lets
the prediction recover fine boundaries that an ×8 bilinear upsample blurred — real, but it acts almost
entirely on the thin/boundary classes. So I expect the gain here to be *smaller* than the PSPNet jump,
and to come mostly from boundary classes — poles, fences, thin riders, traffic signs, the elongated
things whose IoU is dominated by their edges — rather than a wholesale rise across all nineteen. On the
metric pair, since this rung's gain is boundary-flavored, I would expect the ms+flip gap to hold or widen
slightly relative to the +1.33 the pyramid showed — multi-scale/flip augmentation helps boundary
localization, so a boundary-improving architecture and boundary-helping augmentation should stack rather
than cancel; but I only have the single-scale value to reason from now, and the val mIoU will tell.

It is worth reading the ms+flip trend across the whole ladder as a diagnostic, because it has been
telling a consistent story. The gap was +1.95 for the from-scratch U-Net (scale-fragile, its
full-resolution decoder amplifying jitter), fell to +1.11 when the pretrained backbone made features
scale-robust, then ticked back up to +1.33 when the pyramid added context. Each move in that gap tracked
*what kind* of improvement the rung made: pretraining shrank it, context nudged it. A boundary-focused
rung is the case where test-time multi-scale and flip have the most to contribute — seeing a thin pole at
a second scale and averaging a flip is exactly how augmentation cleans up edge predictions — so if this
rung really does its work at the boundaries, I would expect the gap to hold around +1.3 or edge a little
higher rather than collapse. If instead the gap shrinks sharply, that would hint the decoder skip did less
for boundaries than I hoped and the gain came from somewhere else; the paired numbers will discriminate
these, which is more than the single headline mIoU can do alone.

The risk is diminishing returns made concrete: if dense atrous context is not meaningfully sharper than
pooled context on this crop — recall that the two largest atrous branches partly degenerate on the
64-tall axis — then most of the headroom rests on the single OS-4 skip alone, and if that skip is too
low-level it could sharpen noise into crisp-but-wrong boundaries. The channel-reducing 1×1 to 48 and the
separable-conv fusion are the hedges against that; if they fail I would expect ragged, over-sharp edges on
small objects rather than a global drop, and the class-averaged number should still edge up on the
strength of the boundary classes. As on the last two rungs I keep the intermediate-stage auxiliary cross-entropy head during training — a
lightweight FCN-style head on the stage-3 feature with its own loss — because the backbone is still the
deep dilated ResNet-50 and the new capacity sits at the very top, in the ASPP and the decoder, so the
middle stages benefit from a short gradient path. It is a training-only scaffold, dropped at inference,
and does not touch the reported architecture; I mention it only so the comparison to the previous rungs is
honest about what is held fixed.

The concrete change is the depthwise-separable atrous spatial pyramid
plus the single OS-4 decoder; the full head is in the answer.
