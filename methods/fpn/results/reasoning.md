OK, let me think this through from scratch. The thing I keep getting wrong is small objects. A detector built on a single deep feature map handles big, centered objects fine, but a 20-pixel object on a stride-16 map occupies barely a pixel of activation, and there's just not enough there to recognize or localize it. So the real problem isn't "detection," it's *scale*: the same class shows up at radically different sizes and I need features that are good for all of them.

The oldest answer to scale is the image pyramid. Resize the image to a geometric ladder of scales, run the whole feature extractor at each scale, detect on each. It works because a change in an object's size is exactly canceled by moving up or down a level — so a fixed-size model, scanned over positions and over levels, sees every object at roughly the same canonical size. And there's a property I want to hold onto: *every level of that pyramid is semantically strong*, including the high-resolution levels, because I actually ran the full network at that scale. A small object, upsampled and pushed through the whole net, gets the same depth of processing a large object gets. That's why a single shared head works at every scale of an image pyramid.

So why not just do that. Because it's brutally expensive. Running the backbone at, say, four scales is roughly four times the inference cost. And worse, I can't train on it: holding the activations of a deep network at several image scales in memory, end to end, blows up. So in practice people only featurize the image pyramid at test time, train on a single scale, and live with the train/test mismatch. That's an ugly compromise — and the inference cost alone makes it impractical for anything real-time-ish. The single-scale region detectors dropped image pyramids by default for exactly this reason. So the image pyramid gives me the accuracy property I want (strong at all scales) but fails the cost and the end-to-end-trainability requirements. Hold that thought: the *goal* is to keep "strong at all scales" and "one shared head works everywhere," but pay almost nothing and train end to end.

Now, is there multi-scale structure I'm already paying for? Yes — the network itself. A ConvNet with subsampling computes a feature *hierarchy*: a stack of maps at decreasing resolution and increasing depth. For a ResNet the stage outputs — call them C2, C3, C4, C5 — sit at strides 4, 8, 16, 32. That's a pyramid for free, in a single forward pass. So the lazy idea is: treat {C2,C3,C4,C5} as if it were a featurized image pyramid, attach a head to each, done. Zero extra cost.

Let me actually try that in my head and see where it breaks. The problem is the levels are not equivalent. C5 is deep, semantically strong, but coarse — great for big objects, useless for localizing small ones. C2 is high-resolution, perfect for small objects spatially, but it's *shallow* — it's edges and blobs, not object semantics. There's a big semantic gap between the levels, growing with how deep the network is. So if I just predict off the raw hierarchy, the high-res levels — the very ones I need for small objects — are too weak to recognize anything. That's the crux: the image pyramid is uniformly strong; the in-network hierarchy is strong only at the bottom (deep end) and weak at the top (high-res end). Reusing it naively trades my accuracy property away.

This is basically what the single-shot multi-box approach ran into. It predicts from several layers of the hierarchy — good instinct, reuse the forward pass — but to dodge the weak shallow maps it starts the pyramid up high (around conv4_3 of a VGG) and bolts on *new* downsampling layers below that for the coarser levels. So it never reuses the high-resolution early maps at all. Which means it abandons exactly the resolution I need for small objects. The maps it does use still differ in semantic strength top to bottom. So that's not it either — it sidesteps the semantic gap by refusing to use the high-res maps, rather than fixing it.

So let me state the constraint sharply. I have, at high resolution, maps that are *spatially* what I want but semantically poor (C2, C3). I have, at low resolution, a map that is *semantically* what I want but spatially coarse (C5). I want a high-resolution map that is *also* semantically strong. Where would the strong semantics at high resolution come from? They can't be conjured from C2 itself — C2 just hasn't been through enough network. The only place strong semantics exist is C5. So I have to *move* the semantics from the coarse map up to the finer resolutions.

Upsample, then. Take C5, which is strong and coarse, and upsample it by 2× so it now lives at C4's resolution. Nearest-neighbor is fine — I'm not trying to be clever about interpolation, I just want the strong feature values present at the finer grid. Now I have, at C4's resolution, a semantically strong map. Good — but is upsampled-C5 enough on its own? No. C5 was subsampled five times to get there; its activations are spatially smeared, the localization is mushy. If I keep upsampling C5 all the way down to C2's resolution and predict there, I'd have strong semantics but lousy localization — every position is a blur of where things roughly are. Let me hold that as a candidate and come back to test it, because it's tempting: pure top-down, no use of the bottom-up maps except their shapes.

But wait — I *have* a map at C4's resolution whose localization is excellent: C4 itself, fresh from the bottom-up pass, subsampled fewer times, spatially crisp. It's just semantically weaker. And the upsampled C5 is semantically strong but spatially smeared. These two are complementary at the *same resolution*. So merge them: add the upsampled-C5 (strong, smeared) to C4 (weak, crisp), element-wise. The sum has the strong semantics from above *and* the precise localization from the side. The semantics flow down through the network's depth via the top-down path; the spatial precision is injected sideways from the bottom-up map at each level — a lateral connection.

There's a dimension mismatch to fix before I can add. The bottom-up maps have stage-dependent channel counts (256, 512, 1024, 2048 for a ResNet), and the top-down stream I want to keep at a single uniform width. So put a 1×1 convolution on the lateral connection to project the bottom-up map to the common channel count. And notice why that operation is enough: a 1×1 conv mixes channels at each spatial location independently, so it changes representation width without changing the spatial grid. I can leave local spatial mixing to the separate 3×3 output convolution after the merge.

Now iterate it. This is a recursion: I have a merged map at level k; to get level k−1 I upsample the merged map 2×, lateral-1×1 the bottom-up map at k−1, add. C5→merge with C4→merge with C3→merge with C2. Each step pushes the strong semantics one resolution finer and re-anchors the localization with the bottom-up map at that resolution. To *start* the recursion I need a top map: just put a 1×1 conv on C5 (no map above it to add) to get the coarsest level at the common width. So the whole top-down stream is seeded from C5 and walks downward.

One artifact to clean up. Nearest-neighbor upsampling replicates each value into a 2×2 block — blocky, aliased. When I then add the bottom-up map and use this for prediction, that aliasing is still there. So after each merge, run a single 3×3 convolution over the merged map. It smooths the upsampling blockiness and lets the network reconcile the added streams locally. Call the outputs after these 3×3 convs P2, P3, P4, P5 — one per bottom-up level C2..C5, same resolutions. C2 has a large memory footprint and conv1 even more, so start the pyramid at C2 and don't pull conv1 in.

Let me settle the width. I want a single head to run on every level — that's the image-pyramid property I refused to give up, and it only holds if every level looks the same to the head. So fix the channel count d to one value across all P-levels; I'll use d = 256. All the lateral 1×1 convs and the 3×3 convs output 256 channels. Do I need nonlinearities between these extra convs? I tried reasoning about it and it's a wash — these are light glue layers, not the representational workhorse (that's the backbone), and leaving them linear keeps the design clean with negligible difference. So no ReLUs in the extra layers.

Now I should actually test the two halves against each other, because I have two tempting simplifications sitting around and I should know which parts are load-bearing.

If I drop the top-down path and just put the 1×1 lateral plus 3×3 convs on the raw bottom-up maps, I have only dressed up the "reuse the hierarchy directly" idea. C2 is still shallow no matter how I reshape its channels, so the semantic gap is still sitting there. The point of the top-down stream is precisely to import C5's semantics into every level; without it the maps can be channel-matched and smoothed, but not made equally semantic.

If I drop the lateral connections instead, I get pure top-down features: upsample C5 down through all the levels, 3×3 each, then predict. Strong semantics at every resolution, yes. But the localization is the problem: those maps have been downsampled to C5 and then upsampled back several times, so feature positions are smeared and boxes will not sit precisely on objects. The lateral connections are the fix for exactly this — they pipe the crisply-localized bottom-up activations straight into each level. Both halves earn their place: top-down supplies semantics, lateral supplies localization, and I need both.

So the construction is fixed. Now wire it into detector front-ends with as few new rules as possible.

The region proposal network originally slides a tiny head — a 3×3 conv then two sibling 1×1 convs, one for object/not-object, one for box regression — over a single feature map, and it covers scale by having *multiple anchor scales* at each location. But now I have a whole pyramid of equally-strong maps. So I don't need multiple scales stacked on one map; I can let the pyramid carry the scale axis. Attach the *same* head to every level, and assign a *single* anchor scale per level: areas 32², 64², 128², 256², 512² on P2, P3, P4, P5, and one more level P6. Three aspect ratios {1:2, 1:1, 2:1} at each, so 15 anchor types over the pyramid instead of a tall stack on one map. Where does P6 come from — there's no C6. I only need it to host the 512² anchor, so just stride-2 subsample P5; cheap, and it extends the scale range by one octave. (P6 is only for that largest RPN anchor; the region detector below won't use it.)

How do I label anchors with ground truth? I deliberately add no new rule. Keep the original IoU criterion — an anchor is positive if it has the highest IoU with some ground-truth box or IoU > 0.7 with any, negative if its IoU is below 0.3 with all. Notice I do *not* explicitly route ground-truth boxes to levels by their size; the levels are determined by which anchors live where, and ground-truth attaches to anchors. So the scale-to-level routing for RPN is implicit in the anchor placement — no extra machinery.

Should the head share weights across levels? I'll share. And it's a real test of the whole premise: if a single set of head weights works on every level, that's evidence the levels are genuinely at the same semantic level — which is the entire point of the top-down enrichment, and the same reason a shared classifier works across an image pyramid. (I'd expect un-shared, level-specific heads to do about the same, which would confirm it isn't head capacity doing the work.)

Now the region-based detector — Fast R-CNN style. Here it's different: instead of sliding a head densely, I have a set of RoIs (boxes), and for each I pool a fixed 7×7 grid of features and run a small head. The question that didn't exist for the single-map detector: *which* pyramid level do I pool each RoI from? A big RoI obviously wants the coarse, strong, low-res level; a small RoI wants a fine, high-res level. I want a rule that mimics what an image pyramid would do — in an image pyramid, an object lands on whichever level rescales it to the canonical size the network was trained on.

Let me derive that rule. The backbone was pretrained on ImageNet at a canonical size of 224, so a region whose size is ~224 is exactly what the "default" feature map is tuned for. In the single-scale ResNet detector that default map is C4 — stride 16, the natural level for a canonical object. So set a canonical level k0 = 4: a 224-sized RoI should be pooled from P4. Now, if an RoI is smaller, it should move to a finer (higher-resolution, lower-index) level, and by how much should scale logarithmically, because pyramid levels are spaced by factors of 2. Measure RoI size by √(wh) (geometric mean of width and height — the side of an equal-area square). I need to be consistent about box coordinates: the old detector code treats pixel-index boxes as inclusive and computes w = x2 − x1 + 1, h = y2 − y1 + 1; a continuous-coordinate implementation can use x2 − x1 and y2 − y1 instead, but it must train and test with the same convention. Then the offset from the canonical level is log2(√(wh)/224): when √(wh)=224 the offset is 0 (level k0); when the RoI is half-size, 112, the ratio is 1/2, log2 is −1, so it drops one level finer to k=3; double size goes one level coarser. Floor it to land on an integer level:

  k = ⌊ k0 + log2(√(wh)/224) ⌋,  with k0 = 4.

Let me sanity-check the direction once more, because it's easy to flip. Smaller RoI → ratio < 1 → log2 negative → smaller k → finer, higher-resolution level. Yes — small objects get pulled to the high-res maps, which is the whole reason I built strong high-res maps. And clamp k to the available level range so an enormous or tiny box doesn't index off the pyramid. That's the one new rule the region detector needs.

The head itself: in the ResNet single-scale detector, the conv5 stack was repurposed as a heavy 9-layer head on top of C4. But I've already spent conv5 — its semantics are inside the pyramid now via C5. So I can't reuse it as the head. Instead, after RoI-pooling to 7×7, run a light head: two 1024-d fully-connected layers each with ReLU, then the classification and box-regression layers. These fc layers are randomly initialized (there are no pretrained fc layers in a ResNet). This 2-fc head is lighter and faster than the conv5 head, which is a nice side effect of having consumed conv5 into the pyramid. Share these head weights across all levels too, for the same reason as in RPN.

This also ought to replace image pyramids for dense mask proposals. DeepMask and SharpMask already have the right local prediction idea: take a crop-like window, predict an objectness score and a mask. The painful part is scale. They need a dense image pyramid at inference, often with two scales per octave, because a fixed crop-size predictor only covers a narrow object-size band. If my feature maps are already a strong pyramid, I can turn the dense image-pyramid axis into a feature-level axis the same way I did for anchors.

For masks, the natural scale is not √(wh) but max(width, height), because the mask window has to cover the object's larger side. Integer octaves are easy: scales 32, 64, 128, 256, 512 map to P2, P3, P4, P5, P6. A 5×5 sliding MLP on each level can then play the role of a crop predictor for that level's canonical scale. But the old mask systems use two scales per octave, and if I only handle integer octaves, objects halfway between 64 and 128 are always a little mismatched. The simplest way to add the half-octave without another feature level is a second, slightly larger window on the same map: 7×7 is close to 5√2, so it covers the scale halfway to the next octave. These two MLPs play the same role anchors played for boxes: a small set of scale templates tied to each level.

The window needs context around the canonical object, not just the tight object extent. If I use 25% padding, then the 5×5 mask output regions over P2..P6 correspond to image regions 40, 80, 160, 320, 640 pixels wide, and the 7×7 head covers √2 times those sizes. Each spatial location at level k should be responsible for masks whose center falls within 2^k input pixels of that location — one cell in either direction on that feature map. If no object center falls there, the location is a negative for the objectness branch and should not train the mask branch. The head can stay simple: a 5×5 convolution with 512 outputs, then sibling 1×1 layers for a 14×14 mask and one objectness score; the 7×7 branch is identical except for the input window. A lighter width, d = 128, is enough for this proposal head because the dense mask predictor itself is small and runs at every location. During training I can sample positives and negatives, weight the mask loss more heavily than the score loss, and at inference just score every location and scale and keep the top mask proposals without extra image-pyramid post-processing.

Let me make sure I haven't lost the original goals. Cost: one backbone forward pass plus a handful of 1×1 and 3×3 convs at fixed width — marginal over single-scale, and the region head is actually lighter, so overall it can come out *faster* than the conv5-headed single-scale detector. End-to-end trainable: it's all one feedforward graph on a single input image, no image resizing ladder, so train and test do the identical computation — no memory blowup, no train/test mismatch. Strong at all scales: every P-level carries C5's semantics, so the shared heads are legitimate at every level. That's all three requirements, which the image pyramid and the single map each missed.

Let me write the pyramid builder. The forward pass seeds from the deepest map and walks up in resolution, inserting each finished level at the front of the result list.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFeatureBuilder(nn.Module):
    def __init__(self, in_channels_per_stage, out_channels=256, add_extra_coarse_level=True):
        # in_channels_per_stage: channels of [C2, C3, C4, C5] from the backbone,
        # ordered high-resolution -> low-resolution.
        super().__init__()
        lateral_convs = []   # 1x1: project each bottom-up map to out_channels
        output_convs = []    # 3x3: de-alias each merged map
        for in_ch in in_channels_per_stage:
            lateral = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            output = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            nn.init.xavier_uniform_(lateral.weight)
            nn.init.zeros_(lateral.bias)
            nn.init.xavier_uniform_(output.weight)
            nn.init.zeros_(output.bias)
            lateral_convs.append(lateral)
            output_convs.append(output)
        # store low->high-resolution order so the top-down loop reads naturally
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])
        self.add_extra_coarse_level = add_extra_coarse_level

    def forward(self, bottom_up_maps):
        # bottom_up_maps: list [C2, C3, C4, C5], high-res -> low-res
        # seed the top-down stream from the deepest (coarsest, strongest) map: 1x1 on C5
        prev = self.lateral_convs[0](bottom_up_maps[-1])
        results = [self.output_convs[0](prev)]      # P5: 3x3 on the seed
        for idx in range(1, len(self.lateral_convs)):
            c = bottom_up_maps[-idx - 1]             # the next finer bottom-up map
            top_down = F.interpolate(prev, scale_factor=2.0, mode="nearest")  # bring semantics up 2x
            lateral = self.lateral_convs[idx](c)     # 1x1: match channels, keep localization
            prev = lateral + top_down                # merge: strong semantics + crisp localization
            results.insert(0, self.output_convs[idx](prev))   # 3x3 de-alias -> P_k
        if self.add_extra_coarse_level:
            # P6 for the largest RPN anchor: stride-2 subsample of P5
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results  # [P2, P3, P4, P5, optional P6]
```

And the RoI-to-level routing, which is the one new rule the region detector needs:

```python
def assign_roi_to_level(
    boxes,
    min_level=2,
    max_level=5,
    canonical_level=4,
    canonical_box_size=224.0,
    eps=1e-6,
    legacy_plus_one=True,
):
    # boxes: (N, 4) as (x1, y1, x2, y2) in input-image pixels.
    # legacy_plus_one matches the original Detectron/Faster-R-CNN pixel-index convention.
    offset = 1.0 if legacy_plus_one else 0.0
    w = boxes[:, 2] - boxes[:, 0] + offset
    h = boxes[:, 3] - boxes[:, 1] + offset
    if ((w <= 0) | (h <= 0)).any():
        raise ValueError("boxes must have positive width and height")
    s = torch.sqrt(w * h)                                   # sqrt(area) = sqrt(wh)
    k = torch.floor(canonical_level + torch.log2(s / canonical_box_size + eps))
    k = torch.clamp(k, min=min_level, max=max_level)        # stay on the available levels
    return k.to(torch.int64)
```

So the chain is: scale is the real problem; the image pyramid solves it with uniformly-strong levels but is too slow and not end-to-end trainable; the network already gives a multi-scale hierarchy for free but its high-res levels are semantically weak; the fix is to carry C5's strong semantics down through the resolutions with a top-down upsampling pathway, and re-inject precise localization at each level with 1×1 lateral connections from the same-resolution bottom-up map, then a 3×3 to de-alias — giving uniformly-strong maps {P2..P5} at marginal cost; a shared head then runs on every level, with one anchor scale per level for proposals, the region detector pools each RoI from the level k = ⌊k0 + log2(√(wh)/224)⌋ that an image pyramid would have placed it on, and dense mask proposals reuse the same scale axis with integer-octave and half-octave windows instead of a resized image pyramid.
