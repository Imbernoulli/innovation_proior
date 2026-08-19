**Problem.** With the recipe frozen (78.82% at 4.09G), every further gain is
attributable to architecture. Before touching what a block computes
internally (operator, kernel, expansion ratio), settle the network's outer
skeleton — how compute is distributed across stages, and how the stem
processes the raw image — since those are structural, low-risk edits that
are orthogonal to any block-internal redesign.

**Changes (both against the unmodified-ResNet-50 baseline from rung 1,
training recipe otherwise identical and frozen).**

- **Stage compute ratio**: (3, 4, 6, 3) -> (3, 3, 9, 3), matching Swin-T's
  1:1:3:1 distribution as closely as integer block counts allow, moving away
  from the disproportionately heavy res4 stage.
- **Patchify stem**: replace the 7x7-stride-2-conv + maxpool stem (4x
  downsampling via two chained ops) with a single 4x4, stride-4 convolution
  (4x downsampling via one non-overlapping op), matching Swin-T's patch
  size. BatchNorm + ReLU immediately after, unchanged convention.

Block internals (bottleneck structure, 3x3 conv, BatchNorm, ReLU) are
otherwise untouched — this rung only redistributes and re-enters, it does
not redesign the block.

```python
# rung 2: macro design -- stage ratio + patchify stem.
# Block internals unchanged from rung 1 (BN-based bottleneck + LayerScale).
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

class PatchifyStem(nn.Module):
    """Single non-overlapping 4x4 s4 conv, replacing 7x7 s2 conv + maxpool."""
    def __init__(self, in_chans=3, out_chans=64):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=4)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def make_stage(in_planes, planes, blocks, stride=1):
    layers = [Bottleneck(in_planes, planes, stride=stride,
                          downsample=nn.Sequential(
                              nn.Conv2d(in_planes, planes * 4, 1, stride, bias=False),
                              nn.BatchNorm2d(planes * 4)))]
    for _ in range(1, blocks):
        layers.append(Bottleneck(planes * 4, planes))
    return nn.Sequential(*layers)

class ResNet50MacroDesign(nn.Module):
    """ResNet-50 bottleneck blocks, Swin-T stage ratio (3,3,9,3), patchify stem."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = PatchifyStem(3, 64)
        # depths: (3, 4, 6, 3) -> (3, 3, 9, 3); channel plan unchanged (64,128,256,512 x4)
        self.stage1 = make_stage(64, 64, 3, stride=1)
        self.stage2 = make_stage(256, 128, 3, stride=2)
        self.stage3 = make_stage(512, 256, 9, stride=2)   # was 6 blocks, now 9
        self.stage4 = make_stage(1024, 512, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
```

**Test.** Two measured sub-points against the frozen recipe: (a) stage ratio
alone, ResNet-style stem unchanged; (b) stage ratio + patchify stem
together. Compare both to rung 1's 78.82% / 4.09G.
