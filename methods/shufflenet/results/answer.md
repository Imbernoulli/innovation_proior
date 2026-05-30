# ShuffleNet, distilled

ShuffleNet is an extremely computation-efficient CNN for mobile devices (≈10–150 MFLOPs). Its key insight: at tiny budgets the dense 1×1 (pointwise) convolutions dominate the FLOPs, so it makes them *grouped* (pointwise group convolution) to free budget for more channels, and adds a cheap, parameter-free *channel shuffle* to restore the cross-group information flow that grouping otherwise breaks.

## The problem

State-of-the-art efficient blocks (grouped 3×3, depthwise separable) make the *spatial* convolution cheap but leave the *pointwise* 1×1 convolutions dense — and in a small network those dominate (≈93% of a grouped-3×3 residual unit's multiply-adds at cardinality 32). Expensive pointwise convs force a small channel count, and thin feature maps cannot carry enough information. Cut the pointwise cost → afford more channels → more accuracy at fixed budget.

## The key ideas

- **Pointwise group convolution.** Group the 1×1 layers (as cardinality grouped the 3×3): each 1×1 connects only within a channel group, cutting its cost by ~`g`, freeing FLOPs for wider feature maps.
- **Channel shuffle.** Stacking group convs blocks cross-group information flow (each output channel depends only on its group's inputs). Fix with a parameter-free permutation: reshape the `g·n` channels to `(g, n)`, transpose to `(n, g)`, flatten back — so each next group receives a subgroup from every previous group. It's nearly free, works even when the two convs have different group counts, and is differentiable (trains end-to-end).
- **Depthwise spatial conv on the bottleneck only.** The 3×3 is depthwise (cheapest spatial conv), applied only on the narrow bottleneck to limit depthwise's poor compute/memory-access overhead on mobile. No ReLU after the depthwise conv (Xception).

## The ShuffleNet unit (from the ResNet bottleneck)

- **Stride 1 (residual add):** `1×1 group conv (reduce) → BN → ReLU → channel shuffle → 3×3 depthwise (stride 1) → BN → 1×1 group conv (expand) → BN`, then `+ identity`, then ReLU. Bottleneck = ¼ of output channels.
- **Stride 2 (downsample):** 3×3 average pool (stride 2) on the shortcut; depthwise conv at stride 2; combine by *channel concatenation* (cheaply doubles channels between stages) instead of addition.
- No second channel shuffle after the expand conv (comparable scores).

**FLOPs** for input `c×h×w`, bottleneck `m`: ResNet `hw(2cm + 9m²)`, ResNeXt `hw(2cm + 9m²/g)`, **ShuffleNet `hw(2cm/g + 9m)`** — pointwise term cut by `g`, spatial term linear in `m`. So a fixed budget buys wider feature maps.

## The architecture

Stem: 3×3/stride-2 → 24 channels; 3×3/stride-2 maxpool. Three stages of ShuffleNet units (first unit per stage stride-2/concat, rest stride-1/add); output channels double per stage; bottleneck = ¼ output. Stage repeats: 4 / 8 / 4 units (i.e. 1+3, 1+7, 1+3). Global 7×7 avg pool → 1000-FC.

Group count `g` controls pointwise sparsity; widths are adapted to `g` to hold ~140 MFLOPs (stage-2 output channels 144/200/240/272/384 for g=1/2/3/4/8). Larger `g` → more channels (more info) but fewer input channels per filter; smallest models favor largest `g`. Stage-2's first pointwise conv is *not* grouped (its input is only the 24 stem channels). A scale factor `s` ("ShuffleNet s×") scales widths → ~`s²` complexity (1×≈140M, 0.5×≈38M, 0.25×≈13M).

## Training recipe

Inherited from the grouped-residual recipe with two small-net adjustments (tiny nets underfit, not overfit): weight decay 4e-5 (not 1e-4), linearly decayed LR from 0.5 to 0, less aggressive scale augmentation. SGD, batch 1024 on 4 GPUs, ~3×10⁵ iterations. Eval: single 224×224 center crop from 256× resize.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    N, C, H, W = x.size()
    n = C // groups
    x = x.view(N, groups, n, H, W)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(N, C, H, W)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):
        super().__init__()
        self.groups = groups
        self.combine = combine
        self.bottleneck = out_channels // 4
        if combine == 'add':
            self.depthwise_stride = 1
        else:
            self.depthwise_stride = 2
            out_channels = out_channels - in_channels      # concat adds in_channels back
        first_groups = groups if grouped_conv else 1
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, self.bottleneck, 1, groups=first_groups, bias=False),
            nn.BatchNorm2d(self.bottleneck), nn.ReLU(inplace=True))
        self.dwconv = nn.Conv2d(self.bottleneck, self.bottleneck, 3,
                                stride=self.depthwise_stride, padding=1,
                                groups=self.bottleneck, bias=False)
        self.dwbn = nn.BatchNorm2d(self.bottleneck)
        self.expand = nn.Sequential(
            nn.Conv2d(self.bottleneck, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
        out = self.compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.dwbn(self.dwconv(out))                  # no ReLU after depthwise
        out = self.expand(out)
        if self.combine == 'add':
            out = out + residual
        else:
            out = torch.cat((residual, out), 1)
        return F.relu(out)


class ShuffleNet(nn.Module):
    _stage_out = {1: [144, 288, 576], 2: [200, 400, 800], 3: [240, 480, 960],
                  4: [272, 544, 1088], 8: [384, 768, 1536]}

    def __init__(self, groups=3, num_classes=1000, scale=1.0):
        super().__init__()
        self.groups = groups
        stage_out = [int(scale * c) for c in self._stage_out[groups]]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage2 = self._make_stage(24, stage_out[0], repeats=3, first_grouped=False)
        self.stage3 = self._make_stage(stage_out[0], stage_out[1], repeats=7, first_grouped=True)
        self.stage4 = self._make_stage(stage_out[1], stage_out[2], repeats=3, first_grouped=True)
        self.fc = nn.Linear(stage_out[2], num_classes)

    def _make_stage(self, in_ch, out_ch, repeats, first_grouped):
        layers = [ShuffleUnit(in_ch, out_ch, self.groups,
                              grouped_conv=first_grouped, combine='concat')]
        for _ in range(repeats):
            layers.append(ShuffleUnit(out_ch, out_ch, self.groups,
                                      grouped_conv=True, combine='add'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


model = ShuffleNet(groups=3, num_classes=1000, scale=1.0)    # ~140 MFLOPs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=4e-5)
# linear LR 0.5 -> 0 over ~3e5 iterations, batch 1024 on 4 GPUs, light scale augmentation.
```
