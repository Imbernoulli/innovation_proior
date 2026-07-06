72.25 with the pretrained dilated backbone and the local conv head, up +3.15 from the U-Net's 69.10 —
and the metric pair tells me something beyond the headline. The multi-scale-and-flip gap has narrowed
from the U-Net's +1.95 (69.10 → 71.05) to +1.11 here (72.25 → 73.36). That is the confirmation I was
hoping for last rung: the pretrained backbone really is more scale-robust, so test-time multi-scale
augmentation has less jitter left to fix. Good — the semantics gambit paid, the *where*/*what* trade
came out on the side of the pretrained features, and I have a strong, moderately-fine feature map to
build on. Now I want to find out *where* it is still wrong, because that is what tells me what to add.

Think about the head's receptive field, and actually compute it rather than wave at it. The head is a
couple of 3×3 convs over the output-stride-8 feature map. Two stacked 3×3 convs see a 5×5 window at
OS-8; at output stride 8 each step is 8 input pixels, so that is a 40×40-pixel window in the original
image. The dilated backbone underneath adds more reach in principle, but there is a well-known gap
between a deep CNN's *theoretical* receptive field and its *effective* one: the influence of far-away
inputs on a unit falls off toward the edges of the theoretical window roughly like a Gaussian, so the
region that actually drives a unit's response is much smaller than the nominal one. Net effect: each
output pixel's decision is informed by a modest neighborhood — a few tens of pixels of context. For a
lot of Cityscapes that is plenty. But on a 1024-wide crop, 40-odd pixels is a few percent of the frame,
and the errors that survive are exactly the ones where the *local* evidence is genuinely ambiguous and
only the *whole scene* resolves it.

Make that concrete with the classes it bites. A gray planar surface, looked at through a 40-pixel
window, can be road or sidewalk — the texture is identical; what disambiguates it is the global layout,
where it sits relative to the buildings and the curb, which is nowhere in a 40-pixel window. A tall
narrow vertical thing can be a pole, a tree trunk, or the leg of a traffic-sign post — locally identical,
separable only if you know what is around it across the frame. The classic failure of a purely-local
segmenter is a patch of "boat" floating in a parking lot, or "building" predicted on a stretch that the
rest of the scene makes obviously a wall: the head decided each pixel from too small a window and never
consulted the global context that would have vetoed the mismatch.

Someone will object that the dilated backbone's deepest units have a huge theoretical receptive field —
so why is context missing at all? Because the theoretical number is not the operative one. Empirically
the effective receptive field of a deep CNN scales roughly with the square root of its depth and with the
kernel size, and it is Gaussian-weighted, so even when the nominal window covers the whole image the mass
that actually influences a unit is concentrated in a central blob a fraction of that size. The backbone
*can* in principle route a signal from the far side of the frame to a given unit, but the weight it puts
on that far signal, after passing through dozens of small kernels, is tiny. So "the backbone already sees
everything" is true on paper and false in effect, and that gap is precisely why bolting on an *explicit*
whole-map summary — pooled context that reaches every pixel with full weight, not attenuated through
depth — buys something the backbone's own receptive field does not deliver. The deficiency that is left,
now that pretraining fixed the raw semantics, is *global context* with real weight: each pixel's
prediction needs to see a summary of the whole feature map, not just its Gaussian-attenuated
neighborhood.

How do I summarize the whole map and feed it back to every location? Lay out the options before picking.
The bluntest is a single global-average-pool: collapse the OS-8 feature map to one C-dimensional vector
— the scene descriptor — and broadcast it back to every spatial position, concatenated with the local
features. Now every pixel's decision can be conditioned on "what kind of scene is this overall." That is
a real fix and it is cheap. But it is *too* blunt, and the reason is spatial: a single global vector
collapses the entire frame into one number per channel, discarding all spatial structure of the context.
A driving scene has gross spatial organization — sky at the top, road at the bottom, buildings flanking —
and the single global average throws that away, so it cannot tell a pixel "you are in the lower band
where road lives," only "this is a street scene." A second option is to fight locality within the head:
stack more, or larger-dilated, 3×3 convs to grow the head's window toward the whole image. But to cover
a 1024-wide frame at OS-8 I would need enormous dilation rates, which bring the gridding problem back and
cost a great deal, and I would still be committing to *one* window size baked into the architecture. What
I actually want is context that is global in reach but not structureless, and available at more than one
scale at once.

That points to pooling at *several* region scales rather than one. Pool the feature map into a 1×1 grid —
the single global vector, the coarsest context — and *also* into a 2×2 grid (four region summaries, the
quadrants of the scene), a 3×3 grid, and a 6×6 grid (finer sub-region summaries). Each grid cell is the
average of the features over its region, so it is a context summary at that region's scale, and the
stack of grids is context at a *pyramid* of scales, from "the whole image" down to "this sixth of the
image." A pixel that needs the whole-scene cue gets it from the 1×1 level; a pixel that needs to know "I
am in the lower-left region where road usually is" gets it from the 2×2 or 3×3 level. This is adaptive in
a way the per-patch baseline never was: the network can lean on whichever scale of context actually
disambiguates a given class, instead of being locked to one window size. And unlike the single global
vector, the 2×2/3×3/6×6 levels *keep* the gross spatial layout — a 2×2 summary literally distinguishes
top-left from bottom-right — so I get global reach without discarding where things are.

Trace the 2×2 level on a typical Cityscapes frame to see that this really preserves layout. The frame is
horizon-organized: sky and upper building near the top, road and its markings near the bottom, with the
vertical mid-band holding cars, people, and vegetation. Average-pool that to 2×2 and the two bottom cells
summarize the lower half — dominated by road/sidewalk/car features — while the two top cells summarize
the upper half — dominated by sky/building/vegetation. So when I broadcast that grid back, a pixel in the
lower band is handed a context vector that literally encodes "your region reads as road-and-vehicles,"
and a pixel up top gets "your region reads as sky-and-facades." That is a usable prior for the exact
road/sidewalk and wall/building confusions I need to break, and it is information a single global average
— one vector for the whole frame — cannot carry, because it has already averaged the road band and the
sky band together into one number.

The mechanics have to make that pooled context *dense* again before it can be fused per pixel. For each
pooling scale I adaptive-average-pool the OS-8 feature down to that grid (1×1, 2×2, 3×3, 6×6), run a 1×1
conv to reduce its channels, then bilinearly upsample the pooled map back to the full feature-map
resolution — so a 2×2 summary becomes a full-resolution map where every pixel carries the summary of its
quadrant. Concatenate all four upsampled context maps with the original feature map, and run a 3×3
bottleneck conv to fuse local features with multi-scale global context. Then the usual 1×1 classifier and
×8 upsample.

Check what the adaptive pool actually pools over on the real map, because that sets each level's context
scale. The OS-8 feature on the 512×1024 crop is 64×128. Adaptive-average-pool to 1×1 averages all
64×128 ≈ 8200 locations — the whole scene. To 2×2, each cell averages a 32×64 block — a full quadrant. To
3×3, roughly 21×43 per cell — a horizontal band by a third of the width. To 6×6, roughly 11×21 per cell —
about a sixth of the frame each way. In *image* pixels those regions are 8× larger again: the 6×6 cell
summarizes about a 88×168-pixel patch of the original image, which is far wider than the head's ~40-pixel
window yet still local enough to distinguish "the road region" from "the sidewalk region" a few cells
over. So the four levels genuinely tile the scale axis from ~90 image pixels up to the whole frame, with
no gap that the head's own window does not already cover on the small end — which is the span I argued I
wanted.

Let me write the pyramid pooling module:

```python
class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, **kw):
        super().__init__()
        self.pool_scales = pool_scales        # (1, 2, 3, 6)
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),     # summarize each region
                ConvModule(in_channels, channels, 1, **kw)))  # reduce channels

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled = resize(ppm_out, size=x.size()[2:],
                               mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled)        # broadcast region summary to full res
        return ppm_outs
```

and the head: concatenate the input feature with all the pyramid context maps, then a 3×3 bottleneck:

```python
def _forward_feature(self, inputs):
    x = self._transform_inputs(inputs)        # OS=8 deep feature
    psp_outs = [x]
    psp_outs.extend(self.psp_modules(x))      # + 1x1, 2x2, 3x3, 6x6 context, upsampled
    psp_outs = torch.cat(psp_outs, dim=1)
    feats = self.bottleneck(psp_outs)         # fuse local + multi-scale global
    return feats
```

Check the channel budget, because the concat has to close and the 1×1 channel-reduction is what keeps it
from exploding. The four levels together hold 1 + 4 + 9 + 36 = 50 grid cells, but the *cost* is set by
channels, not cell count, because each pooled map is reduced by its 1×1 conv to a common width — call it
512 — before being upsampled. So the concatenation is the original 2048-channel feature stacked with four
512-channel context maps: 2048 + 4·512 = 4096 channels, which is exactly what the bottleneck's input
width `self.in_channels + len(pool_scales) * self.channels` computes. The bottleneck 3×3 then folds 4096
back to 512. Without the per-level 1×1 reduction the concat would be five copies of 2048 = 10240
channels and the bottleneck would be four times heavier; the reduction is why a four-level pyramid is
affordable. And note the pooling itself is nearly free — adaptive-average-pool to a 6×6 grid is a cheap
reduction, and the 1×1 convs run on tiny 1×1…6×6 maps — so almost all the module's cost is the single
3×3 bottleneck over the fused 4096-channel map at OS-8.

It helps to spell out what one output pixel now receives, because that is the whole point of the module.
Before, a pixel's fused feature was just its own 2048 local channels — its neighborhood, nothing more.
After the pyramid, its fused vector is the concatenation of: its own 2048 local channels; the single
1×1 global summary (identical for every pixel — "this is a street scene of this overall composition");
the 512-channel summary of *its* quadrant from the 2×2 grid; the summary of its ninth from the 3×3 grid;
and the summary of its thirty-sixth from the 6×6 grid. So each pixel sees itself plus its region at four
zoom levels simultaneously, and the bottleneck learns how to weigh "what I look like locally" against
"what my region usually is" at each of those zooms. That is a genuinely different input to the classifier
than the local head had, and it is why I expect the module to reach errors the extra convs of the last
rung could not.

The cost of all this lands almost entirely on the one wide bottleneck, so it is worth pricing against the
head it replaces. The FCN head's dominant layer was a 3×3 over 2048 → 512 ≈ 9.4M MACs per location. The
pyramid bottleneck is a 3×3 over 4096 → 512 ≈ 18.9M per location — twice as wide because the concat
doubled the input channels — plus the four 1×1 reductions, which are negligible because they run on
1×1…6×6 maps. So the module roughly doubles the head's compute while leaving the (much larger) backbone
untouched, which is a cheap price for turning a purely-local decision into a globally-informed one. A
detail in the broadcast that matters: I upsample the pooled grids *bilinearly*, not nearest-neighbor, so
the region summaries blend smoothly across grid-cell borders instead of laying down a hard 2×2 seam down
the middle of the image. A nearest-neighbor broadcast would inject a spurious vertical/horizontal edge at
every pooling boundary — an artifact the classifier would have to learn to ignore — whereas the bilinear
ramp between adjacent region summaries is exactly the "context varies gradually across the scene" prior I
want.

Two design choices carry this, and both have a reason rather than a preference. Why *these* four scales
and not, say, just 1×1, or a dozen levels? One level (the global vector) is structureless, as I already
argued. Too many fine levels would approach the local conv head I already have — a 12×12 or 24×24 grid
pools over such small regions that its "context" is barely wider than the head's own 40-pixel window, so
it adds cost without new information. The 1/2/3/6 ladder spans the useful range — whole image, halves,
thirds, sixths — coarsely enough to bring genuinely global context, finely enough to keep some spatial
organization of it, and the four scales are geometrically spaced so each roughly doubles the region count
of the last. And average pooling, not max, because I want each region's *typical* content as a context
descriptor — the prior of what the region usually is — not its single most-activated location. Max would
report the strongest response anywhere in the quadrant, which is the wrong statistic for "what class does
this region tend to be"; average is the region-level class prior I actually want to condition on.

The prediction, against 72.25, and I think it should be a large one. The errors I expect to clear are
exactly the context-starved ones the local head left: the road/sidewalk confusion, the floating
mislabeled patches, the locally-ambiguous verticals — anywhere the right answer was determined by the
scene rather than the neighborhood. Crucially those are a *different* deficiency from the ones the last
two rungs fixed. The U-Net fixed localization; FCN fixed raw semantics; neither touched global context,
so if context-starvation is a big share of the remaining error, piping multi-scale global context into
every pixel's decision should move the class-averaged metric more than the +3.15 that swapping to a
pretrained backbone bought. On the metric pair, the mechanism suggests the ms+flip gap should *not* widen
much: this rung's gain is about scene-level reasoning rather than boundary sharpness, and multi-scale
test augmentation mostly helps boundaries and scale, so I would expect the +1.11-ish gap to hold rather
than blow up — the improvement should already be present at single scale. I will not put a number on the
jump; the val mIoU will tell whether context was indeed the dominant remaining error.

I can say which classes should carry the gain, since the equal-weighted average makes that the mechanism.
The classes that are locally confusable but globally determined are the ones to watch: road versus
sidewalk (same texture, disambiguated by scene position and the curb), wall versus building versus fence
(same facade texture at a patch scale, separated by their role in the layout), terrain versus vegetation,
and the "floating patch" errors where a region gets a class that the whole scene makes impossible. Those
should move up because the pyramid hands each pixel exactly the region-scale prior that resolves them.
The classes I do *not* expect this rung to help much are the thin, boundary-dominated ones — pole,
traffic sign, rider — whose IoU is set by edge localization, which pooled context does not sharpen and
may even slightly smear. So the shape of the improvement I am predicting is: large gains on the confusable
area classes, roughly flat on the thin classes, and since mIoU averages all nineteen equally, a big net
move driven by the first group — provided context really was the bottleneck. If instead the number barely
moves, the read would be that the backbone's own receptive field was already supplying most of the usable
context and I have mostly paid compute for a summary the net already had.

The risk to keep honest about is double-counting context the backbone already half-captures, or that the
bilinear-broadcast of a coarse summary smears context across a real boundary — a 2×2 cell's summary is a
smooth blanket over its whole quadrant and knows nothing about where the boundaries inside the quadrant
are, so if the fusion trusts it blindly it could pull a pixel's label toward its region's average and
soften a true edge. The hedge is structural: the fusion bottleneck sees the *sharp local feature*
alongside the broadcast context and can learn to down-weight the latter near boundaries, taking the
region prior where the local evidence is ambiguous and ignoring it where the local evidence is decisive.
If that balance fails I would expect it to show as over-smoothing of small objects sitting inside a
differently-classed region, not a global drop — but I expect the context gain on the ambiguous planar
classes to dominate. One more thing I carry over rather than reinvent: the auxiliary head. The dilated backbone is deep and I
have just made the main head heavier, so I keep the intermediate-stage auxiliary cross-entropy head from
the last rung — a lightweight FCN-style head on the stage-3 feature with its own loss — to give the
middle of the network a short gradient path while it adapts. It is a training-only scaffold, dropped at
inference, and it is doubly worth keeping here because the pyramid module concentrates the new capacity
at the very top of the net, so the stages below it benefit from a supervision signal that does not have
to survive the whole climb through the pyramid and back. It does not change the architecture whose mIoU
I report.

A placement decision worth stating: I compute the pyramid once, on the *deepest* OS-8 feature, and
nowhere else. The alternative would be to attach pooling pyramids at several backbone stages and fuse
them, but that misreads what the module is for. The deepest feature is where the semantics are richest, so
a region summary computed there is a summary of *classes*, which is the prior I want to broadcast; a
pyramid on a shallow stage would summarize edges and textures, which is not a useful scene-level cue and
would mostly add cost. And global context is a property of the whole scene, so it only needs to be gathered
once, at the level that best understands the scene, and then handed to every pixel. So one PPM on the top
feature is not a simplification I am settling for — it is where the module belongs, and it also keeps the
extra compute confined to a single module over the OS-8 map rather than smeared across the backbone, which
is what let me price the whole change as roughly one doubled bottleneck a moment ago.

The concrete change is the pyramid pooling module feeding multi-scale context into
the fusion bottleneck; the full head is in the answer.
