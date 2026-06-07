# FPN synthesis notes

## Pain point
Detect objects across a huge range of scales. Two classic answers, each broken:
- **Featurized image pyramid**: resize image to many scales, run the whole feature extractor on each. All levels semantically strong (the high-res levels too), so a single shared head works at every scale. But: inference ~4x slower; training end-to-end on a pyramid is memory-infeasible, so it's test-time only -> train/test inconsistency. Faster R-CNN drops it by default.
- **Single feature map** (Fast/Faster R-CNN): run head on one deep map (ResNet C4). Fast, ConvNet features are somewhat scale-robust, but weak on small objects because the single map is coarse (stride 16/32) — small objects vanish.

## The half-answer that already exists for free
A ConvNet already computes a **feature hierarchy** {C2,C3,C4,C5} with strides {4,8,16,32}. Pyramidal shape for free. But large **semantic gap**: C2 is high-res but shallow/weak; C5 is deep/strong but coarse. Using the shallow high-res maps directly hurts.

SSD: predicts from multiple layers, but to avoid the weak shallow maps it starts the pyramid high (conv4_3 of VGG) and adds new layers downward — so it never reuses the high-res early maps. Misses exactly what helps small objects.

## Goal
Strong semantics at ALL scales, at marginal cost, from a single input image, trainable end-to-end.

## Derivation of top-down + lateral
- Strong semantics live only in the deep, coarse map. To get strong semantics at high res, propagate them downward: take the deepest map, **upsample 2x** (nearest-neighbor for simplicity), so the strong semantics now sit at higher resolution.
- Upsampling alone loses precise localization (the map was subsampled many times). The same-resolution bottom-up map C_i has accurate localization but weak semantics. **Merge** them: 1x1 conv on C_i (lateral connection) to match channel dim d=256, then element-wise add. The 1x1 conv is channel-projection only — does not mix spatial neighbors, so it preserves localization.
- Repeat down the levels (C5->C4->C3->C2). Start the iteration with a 1x1 conv on C5 to make the coarsest map.
- After each merge, a **3x3 conv** to reduce aliasing from nearest-neighbor upsampling -> P_i.
- Fixed d=256 channels everywhere so one shared head works on all levels (echoes the image-pyramid property). No nonlinearity in the extra convs (empirically minor).

Ablations confirm both halves matter: top-down without lateral (e) loses localization -> AR drops ~10; lateral on bottom-up without top-down (d) keeps the semantic gap -> on par with baseline only.

## Level assignment (Eqn.1)
Region-based detector pools each RoI from ONE pyramid level. Mimic the image-pyramid intuition: a box of canonical size 224 (ImageNet pretrain size) goes to canonical level k0=4 (the level a single-scale ResNet detector would use, C4). A smaller box -> finer (higher-res, lower-index) level.
  k = floor(k0 + log2(sqrt(wh)/224)), k0=4.
Check: wh=224^2 -> k=4. wh=112^2 -> ratio 1/2 -> log2=-1 -> k=3. Correct. Clamp to [min,max].
Note: ground-truth scales are NOT used to assign anchors to levels in RPN — anchors of a single scale per level ({32^2,...,512^2} on {P2..P6}), gt assigned to anchors by IoU as usual.

## Heads
- RPN: same head (3x3 conv + two sibling 1x1 for cls/reg) attached to every level; one anchor scale per level, 3 aspect ratios -> 15 anchors total. Shared params across levels (works -> levels share semantic level). P6 = stride-2 subsample of P5 for the 512^2 anchor.
- Fast R-CNN: RoIPool 7x7 + two 1024-d fc + ReLU -> cls/reg. Lighter than ResNet conv5 head (conv5 already consumed by the pyramid). Shared across levels.

## Code grounding
detectron2 modeling/backbone/fpn.py: FPN.forward = lateral_convs[0](C5) -> output_convs[0] -> P5; loop finer: interpolate prev 2x nearest + lateral_conv(C_i) -> output_conv -> insert front. LastLevelMaxPool -> P6.
poolers.py assign_boxes_to_levels: floor(canonical_level + log2(sqrt(area)/canonical_box_size)); clamp.
