# Feature Pyramid Networks (FPN)

## Problem

Detect objects across a wide range of scales with features that are strong at *every* scale, at marginal cost over a single backbone forward pass, and trainable end to end. Featurizing an image pyramid is uniformly strong but several times slower and memory-infeasible to train end to end (so it is test-time only). A single deep feature map is fast but coarse, so small objects are weakly represented. A ConvNet already computes a multi-scale feature hierarchy {C2,C3,C4,C5} (strides {4,8,16,32}) for free, but its high-resolution early maps are semantically weak — a large semantic gap across levels.

## Key idea

Build a feature pyramid {P2,P3,P4,P5} that is semantically strong at all scales by combining two pathways:

- **Bottom-up pathway**: the backbone's feedforward stage outputs {C2,C3,C4,C5}.
- **Top-down pathway + lateral connections**: propagate the strong semantics of the deepest map upward in resolution and re-inject precise localization at each level.
  - Seed: a 1×1 conv on C5 gives the coarsest map.
  - At each finer level: upsample the current top-down map 2× (nearest-neighbor), apply a **1×1 lateral conv** to the same-resolution bottom-up map C_k (to match channel dimension d=256 while preserving its crisp localization — 1×1 does not blend spatial neighbors), and **add** element-wise.
  - Apply a **3×3 conv** to each merged map to reduce nearest-neighbor upsampling aliasing → P_k.
- All P-levels have the same channel count d = 256, so a **single shared head** runs on every level (the image-pyramid property). No nonlinearities are added in the extra convs.

Both halves are necessary: without the top-down path the semantic gap across bottom-up levels remains; without the lateral connections localization is smeared by repeated down/up-sampling.

## Final construction

**Merge (per level, from coarse to fine):**

    P5 = conv3x3( conv1x1(C5) )
    M_k = conv1x1(C_k) + upsample_2x_nearest(M_{k+1})
    P_k = conv3x3(M_k)        for k = 4, 3, 2

**RPN front-end:** the same 3×3-conv + two sibling 1×1-conv head on every level; a single anchor scale per level — areas {32², 64², 128², 256², 512²} on {P2,P3,P4,P5,P6} — with aspect ratios {1:2, 1:1, 2:1} (15 anchors total). P6 = stride-2 subsample of P5 (only to host the 512² anchor; not used by the region detector). Anchor labels by IoU as usual (>0.7 positive, <0.3 negative); ground-truth scale is not used to route anchors to levels.

**Region-detector (Fast R-CNN) front-end:** RoIPool to 7×7, then a light 2×1024-d fc head with ReLU + cls/reg layers (conv5 is consumed by the pyramid, so it is not reused as the head). Each RoI of width w, height h is pooled from level

    k = floor( k0 + log2( sqrt(w*h) / 224 ) ),   k0 = 4

(224 = canonical ImageNet pretrain size; k0=4 = the level a single-scale ResNet detector uses, C4). A smaller RoI → finer/higher-resolution level. Clamp k to the available levels. Heads shared across all levels.

**Dense segmentation-proposal front-end:** use the same pyramid with P2–P6 and a lighter width d=128. Define mask scale as max(width, height): scales {32, 64, 128, 256, 512} map to {P2, P3, P4, P5, P6}. A 5×5 fully convolutional MLP predicts a 14×14 mask and objectness score for integer octaves; a second 7×7 MLP handles half-octaves because 7 ≈ 5√2. With 25% padding, the 5×5 mask windows correspond to image regions {40, 80, 160, 320, 640}; the 7×7 windows cover √2 larger regions. A location at level k is responsible for masks whose center lies within 2^k input pixels of that location; negatives train only the score branch.

## Code

Pyramid builder:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureBuilder(nn.Module):
    """Top-down pathway with lateral connections over backbone stage maps.
    in_channels_per_stage: channels of [C2, C3, C4, C5], high-res -> low-res."""

    def __init__(self, in_channels_per_stage, out_channels=256, add_extra_coarse_level=True):
        super().__init__()
        lateral_convs = []
        output_convs = []
        for in_ch in in_channels_per_stage:
            lateral = nn.Conv2d(in_ch, out_channels, kernel_size=1)        # match channels
            output = nn.Conv2d(out_channels, out_channels, 3, padding=1)   # de-alias
            nn.init.xavier_uniform_(lateral.weight); nn.init.zeros_(lateral.bias)
            nn.init.xavier_uniform_(output.weight); nn.init.zeros_(output.bias)
            lateral_convs.append(lateral)
            output_convs.append(output)
        # store in top-down (low-res -> high-res) order
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])
        self.add_extra_coarse_level = add_extra_coarse_level

    def forward(self, bottom_up_maps):
        # bottom_up_maps: list [C2, C3, C4, C5], high-res -> low-res
        prev = self.lateral_convs[0](bottom_up_maps[-1])     # 1x1 on C5: seed top-down stream
        results = [self.output_convs[0](prev)]               # P5
        for idx in range(1, len(self.lateral_convs)):
            c = bottom_up_maps[-idx - 1]
            top_down = F.interpolate(prev, scale_factor=2.0, mode="nearest")
            lateral = self.lateral_convs[idx](c)
            prev = lateral + top_down                         # merge by element-wise add
            results.insert(0, self.output_convs[idx](prev))   # P_k
        if self.add_extra_coarse_level:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2))  # P6 from P5
        return results   # [P2, P3, P4, P5, (P6)]
```

RoI-to-level assignment (Eqn. above):

```python
def assign_roi_to_level(
    boxes,
    min_level=2,
    max_level=5,
    canonical_level=4,
    canonical_box_size=224.0,
):
    """boxes: (N,4) as (x1,y1,x2,y2) in input-image pixels. Returns level index per box."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    box_sizes = torch.sqrt(w * h)
    level = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8))
    level = torch.clamp(level, min=min_level, max=max_level)
    return level.to(torch.int64)
```

Heads (shared across levels): the RPN head is a 3×3 conv + two sibling 1×1 convs (objectness, box regression) applied to every P-level; the Fast R-CNN head is RoIPool(7×7) → fc(1024)→ReLU → fc(1024)→ReLU → (class scores, box deltas), with each RoI pooled from the level returned by `assign_roi_to_level`. Dense mask proposals use per-level 5×5 and 7×7 fully convolutional MLP heads as described above.
