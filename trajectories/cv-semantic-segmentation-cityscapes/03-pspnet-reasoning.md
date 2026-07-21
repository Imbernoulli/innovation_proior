72.25 with the pretrained dilated backbone and the local conv head, up +3.15 from the U-Net's 69.10 — and
the metric pair says more. The ms+flip gap narrowed from +1.95 (69.10 → 71.05) to +1.11 here (72.25 →
73.36): the confirmation I wanted last time that the pretrained backbone really is more scale-robust, so
test-time multi-scale augmentation has less jitter left to fix. The semantics gambit paid. Now find *where*
it is still wrong, because that tells me what to add.

Compute the head's receptive field rather than wave at it. The head is a couple of 3×3 convs over the OS-8
feature map — two stacked 3×3s see a 5×5 window at OS-8, and since each step is 8 input pixels, that is a
40×40-pixel window in the original image. The dilated backbone underneath adds more reach in principle, but
a deep CNN's *effective* receptive field is much smaller than its theoretical one: the influence of far
inputs on a unit falls off toward the window edges roughly like a Gaussian, so even where the nominal
window covers the whole frame the mass that actually drives the unit is a central blob a fraction of that
size. Net effect: each output pixel's decision is informed by a few tens of pixels of context. For much of
Cityscapes that is plenty — but on a 1024-wide crop, 40-odd pixels is a few percent of the frame, and the
errors that survive are exactly the ones where the local evidence is genuinely ambiguous and only the whole
scene resolves it.

Make it concrete with the classes it bites. A gray planar surface, through a 40-pixel window, can be road
or sidewalk — identical texture, disambiguated only by the global layout, where it sits relative to the
buildings and the curb. A tall narrow vertical can be a pole, a tree trunk, or the leg of a sign post —
locally identical, separable only from what surrounds it across the frame. The classic failure of a
purely-local segmenter is a patch of "boat" floating in a parking lot, or "building" on a stretch the rest
of the scene makes obviously a wall: each pixel decided from too small a window, never consulting the
global context that would veto the mismatch. So the deficiency left, now that pretraining fixed the raw
semantics, is *global context with real weight* — a summary of the whole map that reaches every pixel at
full strength, not attenuated through depth. The backbone *can* in principle route a far signal to a unit,
but after passing through dozens of small kernels the weight it puts on that far signal is tiny, so "the
backbone already sees everything" is true on paper and false in effect.

How to summarize the whole map and feed it to every location. The bluntest is a single global-average-pool:
collapse the OS-8 map to one C-dimensional scene descriptor and broadcast it back, concatenated with the
local features, so every pixel's decision can be conditioned on "what kind of scene is this." Real and
cheap — but too blunt, because a single global vector discards all spatial structure of the context. A
driving scene has gross organization — sky at the top, road at the bottom, buildings flanking — and the
global average throws it away, so it can say "this is a street scene" but not "you are in the lower band
where road lives." Fighting locality within the head instead — stacking more or larger-dilated 3×3 convs —
would need enormous dilation rates to cover a 1024-wide frame at OS-8, bringing gridding and cost back, and
would still bake in *one* window size. What I want is context that is global in reach but not structureless,
at more than one scale at once.

That points to pooling at *several* region scales. Pool the feature map into a 1×1 grid — the single global
vector, the coarsest context — and *also* a 2×2 grid (four quadrant summaries), a 3×3, and a 6×6. Each cell
is the average of the features over its region, a context summary at that region's scale, and the stack is
context at a pyramid of scales, from the whole image down to a sixth of it. A pixel needing the whole-scene
cue gets it from the 1×1 level; a pixel needing "I am in the lower-left region where road usually is" gets
it from the 2×2 or 3×3 level. And unlike the single global vector, the 2×2/3×3/6×6 levels *keep* the gross
spatial layout — a 2×2 summary literally distinguishes top-left from bottom-right. On a horizon-organized
Cityscapes frame, average-pooling to 2×2 makes the two bottom cells summarize the lower half — road,
sidewalk, car — and the two top cells the upper half — sky, building, vegetation; so a lower-band pixel is
handed a context vector encoding "your region reads as road-and-vehicles," a usable prior for exactly the
road/sidewalk and wall/building confusions, and information a single global average cannot carry because it
has already averaged the road band and the sky band into one number.

The pooled context has to be made dense again before per-pixel fusion: for each scale, adaptive-average-pool
the OS-8 feature to that grid (1×1, 2×2, 3×3, 6×6), run a 1×1 conv to reduce channels, then bilinearly
upsample back to full feature-map resolution — so a 2×2 summary becomes a full-resolution map where every
pixel carries its quadrant's summary — then concatenate all four with the original feature and fuse with a
3×3 bottleneck before the classifier and ×8 upsample. Check what each level pools over on the real 64×128
map: 1×1 averages all ≈8200 locations (whole scene); 2×2, a 32×64 block each (a quadrant); 3×3, ≈21×43 (a
band by a third of the width); 6×6, ≈11×21 (about a sixth each way). In *image* pixels those are 8× larger —
the 6×6 cell summarizes about an 88×168-pixel patch, far wider than the head's ~40-pixel window yet still
local enough to tell "the road region" from "the sidewalk region" a few cells over. The four levels tile
the scale axis from ~90 image pixels up to the whole frame.

The pyramid pooling module and the context-fusion head are in the answer; the channel budget is what keeps
the concat from exploding. The four levels together hold 1 + 4 + 9 + 36 = 50 grid cells, but the cost is
set by channels, not cell count, because each pooled map is 1×1-reduced to a common width — 512 — before
being upsampled. So the concatenation is the 2048-channel feature stacked with four 512-channel context
maps: 2048 + 4·512 = 4096 channels, and the bottleneck 3×3 folds that back to 512. Without the per-level
reduction the concat would be five copies of 2048 = 10240 channels and the bottleneck four times heavier;
the reduction is why a four-level pyramid is affordable. The pooling itself is nearly free — adaptive-pool
to 6×6 is cheap and the 1×1 convs run on tiny 1×1…6×6 maps — so almost all the module's cost is the single
3×3 bottleneck over the fused 4096-channel map: ≈18.9M MACs per location against the FCN head's 9.4M,
roughly doubling the head compute while leaving the (much larger) backbone untouched. And I upsample the
pooled grids *bilinearly*, not nearest-neighbor, so the region summaries blend smoothly across grid-cell
borders instead of laying a hard 2×2 seam down the middle of the image — a nearest-neighbor broadcast would
inject a spurious edge at every pooling boundary that the classifier would have to learn to ignore.

Two design choices carry this. Why *these* four scales and not one, or a dozen? One level is structureless,
as argued. Too many fine levels approach the local conv head — a 12×12 grid pools over regions barely wider
than the head's own 40-pixel window, adding cost without new information. The 1/2/3/6 ladder spans the
useful range — whole image, halves, thirds, sixths — geometrically spaced so each roughly doubles the
region count of the last. And average pooling, not max, because I want each region's *typical* content as a
descriptor — the prior of what the region usually is — not its single most-activated location, which is the
wrong statistic for "what class does this region tend to be."

The prediction, against 72.25, and I think it is large. The errors I expect to clear are the
context-starved ones the local head left — road/sidewalk, the floating mislabeled patches, the
locally-ambiguous verticals — anywhere the right answer is set by the scene rather than the neighborhood.
Crucially that is a *different* deficiency from the previous steps': the U-Net fixed localization, FCN
fixed raw semantics, neither touched global context, so if context-starvation is a big share of the
remaining error, piping multi-scale context into every pixel should move the class-averaged metric more
than the +3.15 the backbone swap bought. Because mIoU weights all 19 equally, the gain should concentrate
on the confusable area classes — road vs sidewalk, wall vs building vs fence, terrain vs vegetation, the
floating-patch errors — and stay roughly flat on the thin, boundary-dominated classes (pole, traffic sign,
rider) whose IoU is set by edge localization, which pooled context does not sharpen and may slightly smear.
So the shape of the improvement is large gains on the confusable area classes and roughly flat on the thin
ones, a big net move on the equal-weighted average driven by the first group — provided context really was
the bottleneck. On the metric pair, since this gain is scene-level rather than boundary sharpness, I expect
the +1.11-ish ms+flip gap to hold rather than blow up — the improvement should already be present at single
scale. I will not put a number on the jump; the val mIoU will tell whether context was the dominant
remaining error.

The risk to keep honest: double-counting context the backbone half-captures, or the bilinear broadcast of a
coarse summary smearing context across a real boundary — a 2×2 cell's summary is a smooth blanket knowing
nothing about where the boundaries inside the quadrant are, so blind fusion could pull a pixel's label
toward its region's average and soften a true edge. The hedge is structural: the fusion bottleneck sees the
*sharp local feature* alongside the broadcast context and can learn to down-weight the latter near
boundaries, taking the region prior where the local evidence is ambiguous and ignoring it where it is
decisive. If the balance fails it should show as over-smoothing of small objects sitting inside a
differently-classed region, not a global drop. And I compute the pyramid once, on the *deepest* OS-8 feature
and nowhere else: that is where the semantics are richest, so a region summary there is a summary of
*classes* — the prior I want to broadcast — whereas a pyramid on a shallow stage would summarize edges and
textures and mostly add cost. Global context is a property of the whole scene, gathered once at the level
that best understands it, then handed to every pixel. I keep the intermediate-stage auxiliary head from
last time — a lightweight FCN-style head on the stage-3 feature with its own loss, giving the deep middle a
short gradient path — doubly worth it here because the pyramid concentrates the new capacity at the very
top of the net. It is a training-only scaffold, dropped at inference. The concrete change is the pyramid
pooling module feeding multi-scale context into the fusion bottleneck; the full head is in the answer.
