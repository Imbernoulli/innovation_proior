The central difficulty in object detection is scale. The same object can appear as a tiny region of a few dozen pixels or as a large object filling most of the frame, and a detector must recognize and localize it accurately at every size. The classic solution is an image pyramid: resize the input to many scales, run the network on each, and combine the results. This gives strong, semantically rich features at every scale because the full network is evaluated at each resolution. But it multiplies inference cost and is too memory-intensive to train end-to-end, so in practice it is used only at test time. A single-scale detector is fast and trainable end-to-end, but it relies on one deep feature map that is coarse and weak for small objects. What is needed is the scale coverage of the image pyramid at nearly the cost of a single forward pass.

The network already computes a multi-scale structure for free. A deep ConvNet with subsampling produces a hierarchy of feature maps {C2, C3, C4, C5} at strides {4, 8, 16, 32} relative to the input. The problem is a semantic gap across these levels: the high-resolution maps are shallow and contain only low-level edges and textures, while the low-resolution maps are semantically strong but too coarse to localize small objects. So the hierarchy is pyramidal in shape but not in usefulness. The key insight is that the strong semantics from the deepest map can be carried upward in resolution, while the precise localization of the bottom-up maps can be preserved at each level.

The new method is the Feature Pyramid Network, or FPN. It constructs a feature pyramid {P2, P3, P4, P5} that is semantically strong at every scale by combining a top-down pathway with lateral connections. The construction starts from the coarsest map C5: a 1x1 convolution projects it to a fixed channel width of 256. Then, for each finer level, the current top-down map is upsampled by a factor of two using nearest-neighbor interpolation, and a 1x1 convolution projects the same-resolution bottom-up map to 256 channels. The two are added element-wise, merging the strong semantics from above with the crisp localization from the side. A final 3x3 convolution is applied after each merge to smooth aliasing from the upsampling. This gives a set of uniformly strong maps at all resolutions, all with the same channel count, so the same detection head can be applied at every level. The result is end-to-end trainable on a single input scale and adds only marginal computation to the backbone forward pass.

Both the top-down and lateral components are essential. Without the top-down path, the high-resolution maps remain semantically weak and the semantic gap persists. Without the lateral connections, repeated upsampling from the coarsest map would produce strong but spatially blurred features, degrading localization. The 1x1 lateral convolutions are chosen specifically because they preserve the spatial grid of the bottom-up map while only changing the channel dimension, and the 3x3 output convolutions allow the network to reconcile the merged streams locally.

FPN plugs into standard detector front-ends with minimal changes. For region proposal generation, the same lightweight head is attached to every level of the pyramid, and a single anchor scale is assigned to each level, corresponding to areas {32^2, 64^2, 128^2, 256^2, 512^2} on {P2, P3, P4, P5, P6}. Here P6 is obtained cheaply by stride-2 subsampling P5 and is used only for the largest RPN anchor. For Fast R-CNN style detection, each region of interest is pooled from the pyramid level given by k = floor(4 + log2(sqrt(w*h)/224)), where 224 is the canonical ImageNet pretraining size and level 4 is the natural level for a canonical object. Smaller regions are routed to finer, higher-resolution levels. The detection head consists of two 1024-dimensional fully-connected layers followed by classification and box-regression outputs, with weights shared across levels. Because conv5 is consumed inside the pyramid, it is not reused as part of the head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureBuilder(nn.Module):
    """Top-down pathway with lateral connections over backbone stage maps.

    in_channels_per_stage: channels of [C2, C3, C4, C5], high-res -> low-res.
    Returns: list [P2, P3, P4, P5, (P6)] of feature maps with 256 channels.
    """

    def __init__(self, in_channels_per_stage, out_channels=256, add_extra_coarse_level=True):
        super().__init__()
        lateral_convs = []
        output_convs = []
        for in_ch in in_channels_per_stage:
            lateral = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            output = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            nn.init.xavier_uniform_(lateral.weight)
            nn.init.zeros_(lateral.bias)
            nn.init.xavier_uniform_(output.weight)
            nn.init.zeros_(output.bias)
            lateral_convs.append(lateral)
            output_convs.append(output)
        # store in low-res -> high-res order for the top-down loop
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])
        self.add_extra_coarse_level = add_extra_coarse_level

    def forward(self, bottom_up_maps):
        # bottom_up_maps: list [C2, C3, C4, C5], high-res -> low-res
        prev = self.lateral_convs[0](bottom_up_maps[-1])   # 1x1 on C5
        results = [self.output_convs[0](prev)]             # P5
        for idx in range(1, len(self.lateral_convs)):
            c = bottom_up_maps[-idx - 1]                   # next finer bottom-up map
            top_down = F.interpolate(prev, scale_factor=2.0, mode="nearest")
            lateral = self.lateral_convs[idx](c)
            prev = lateral + top_down                       # merge semantics + localization
            results.insert(0, self.output_convs[idx](prev)) # P_k
        if self.add_extra_coarse_level:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2))  # P6
        return results


def assign_roi_to_level(
    boxes,
    min_level=2,
    max_level=5,
    canonical_level=4,
    canonical_box_size=224.0,
    eps=1e-6,
    legacy_plus_one=True,
):
    """Return the FPN level k for each RoI of size sqrt(w * h)."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    offset = 1.0 if legacy_plus_one else 0.0
    w = boxes[:, 2] - boxes[:, 0] + offset
    h = boxes[:, 3] - boxes[:, 1] + offset
    if ((w <= 0) | (h <= 0)).any():
        raise ValueError("boxes must have positive width and height")
    box_size = torch.sqrt(w * h)
    level = torch.floor(canonical_level + torch.log2(box_size / canonical_box_size + eps))
    level = torch.clamp(level, min=min_level, max=max_level)
    return level.to(torch.int64)
```
