The recipe swap alone moved ResNet-50 from 76.13% to 78.82% at identical
FLOPs, with the architecture completely untouched — a real gain, and it
confirms training procedure was carrying weight in the original comparison,
but it still leaves a meaningful gap to Swin-T's 81.30% with zero
architectural credit claimed. From here every further gain is attributable
to the network design alone, since the recipe is now frozen. I don't want to
open with the convolution operator itself — grouped or depthwise
convolution, kernel size, expansion ratio all change what a block computes
and cost, and I'd rather not conflate that with a coarser, orthogonal
question first: how compute is distributed across the network, independent
of what any individual block does.

Two structural facts about Swin-T answer that question without touching a
block's internals. First, stage compute ratio: ResNet-50's depths (3, 4, 6,
3) load disproportionately onto the third stage, a choice driven mainly by
downstream detection-head compatibility rather than an ImageNet-optimal
search, while Swin-T spreads compute more evenly at roughly 1:1:3:1. I'll
move ResNet-50 to (3, 3, 9, 3), the closest integer analogue, which also
happens to land FLOPs near Swin-T's 4.5G budget as a side effect. Second,
the stem: a ResNet stem downsamples 4x via a 7x7 stride-2 conv followed by a
max pool — two chained operations doing what is fundamentally one job,
reducing the heavy spatial redundancy of a raw image. Every Transformer-family
architecture does this with a single non-overlapping "patchify" convolution
instead; Swin-T's is 4x4 stride-4, matching its stage resolutions exactly.
I'll replace ResNet's two-op stem with that single conv, keeping BatchNorm
and ReLU immediately after since normalization itself isn't up for revision
yet.

I'm proposing both together as one macro rung because neither is a
hypothesis about capacity or a new computational primitive — both ask
whether the network's outer skeleton matters, independent of block
internals — and bundling them lets one training run settle both, while I
can still read each sub-step's individual contribution since they're
reported separately. My prior here is modest rather than large: neither
edit touches the operator a block computes, the parameter count barely
moves, and the stem in particular touches only the very first layer of a
fifty-layer network, where whatever the patchify-versus-strided-conv choice
loses or preserves is a rounding error against everything downstream. The
stage-ratio change has a bit more room, since it moves where roughly half
the network's depth sits, but redistributing existing blocks without
changing what any block computes is a coarser lever than redesigning the
block. I'd read a large swing here, positive or negative, as a signal that
ResNet's original skeleton was doing more (or less) work than credited, and
either way this settles the outer shape — (3, 3, 9, 3), patchify stem —
that every later, block-level rung will sit inside.

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
