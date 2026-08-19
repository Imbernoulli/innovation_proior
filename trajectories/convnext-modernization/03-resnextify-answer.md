**Problem.** The macro-design skeleton (79.51%, 4.42G) is settled; block
internals are still an unmodified ResNet bottleneck (full 3x3 conv mixing
space and channel simultaneously). Replace the spatial-mixing operator with
depthwise convolution — grouped convolution taken to groups = channels —
which factors space-mixing and channel-mixing into separate operations, the
same factorization self-attention (spatial mixing) plus feedforward
(channel mixing) already has. Compensate the resulting capacity loss by
widening, following the standard "more groups, expand width" strategy.

**Changes (against the rung 2 macro-design baseline).**

- Replace the bottleneck's 3x3 conv with a depthwise 3x3 conv (`groups =
  channels`): each output channel only sees its own channel's spatial
  neighborhood; all cross-channel mixing is pushed entirely onto the
  surrounding 1x1 convs.
- Widen the base channel count from 64 to 96 (matching Swin-T's width, since
  the macro skeleton already matches Swin-T's stage ratio and patch size),
  to reinvest the FLOPs depthwise conv frees up and restore the 1x1 layers'
  channel-mixing capacity.

```python
# rung 3: ResNeXt-ify -- depthwise 3x3 conv + widen 64 -> 96.
import torch.nn as nn

class DepthwiseBottleneck(nn.Module):
    """ResNet bottleneck with the 3x3 conv replaced by a depthwise 3x3 conv."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # groups = planes: each output channel sees only its own input channel
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1,
                                groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))     # depthwise: no cross-channel mixing here
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

def make_stage(in_planes, planes, blocks, stride=1):
    layers = [DepthwiseBottleneck(in_planes, planes, stride=stride,
                                   downsample=nn.Sequential(
                                       nn.Conv2d(in_planes, planes * 4, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes * 4)))]
    for _ in range(1, blocks):
        layers.append(DepthwiseBottleneck(planes * 4, planes))
    return nn.Sequential(*layers)

# Stage ratio (3, 3, 9, 3) and patchify stem from rung 2 unchanged.
# Base width 64 -> 96, matching Swin-T's first-stage channel count.
base_width = 96  # was 64 in rungs 1-2
stage1 = make_stage(base_width, base_width, 3, stride=1)
stage2 = make_stage(base_width * 4, base_width * 2, 3, stride=2)
stage3 = make_stage(base_width * 8, base_width * 4, 9, stride=2)
stage4 = make_stage(base_width * 16, base_width * 8, 3, stride=2)
```

**Test.** Two measured sub-points against the rung 2 baseline: (a) depthwise
conv alone, width still 64 (the predicted-regression point, isolating the
capacity loss); (b) depthwise conv + width 96 (the compensated point).
