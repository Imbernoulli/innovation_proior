The macro-design rung landed almost exactly where I expected — 78.82% to
79.51%, a real but modest +0.69 — confirming that redistributing existing
blocks and simplifying the stem is a small lever next to whatever lives
inside each block. Both sub-steps moved the same small-positive direction
with no surprises, so the outer skeleton I inherited from Swin-T (stage
depths (3, 3, 9, 3), patchify stem) is worth building on rather than
revisiting, and FLOPs at 4.42G leave essentially no headroom before I'd be
comparing an unfairly larger network. Time to open the block itself.

ResNet-50's bottleneck mixes space and channel simultaneously in one full
3x3 convolution. Grouped convolution — split channels into groups, convolve
only within each group — cuts FLOPs, and the standard move that comes with
it is to reinvest those savings into width rather than bank them. Taken to
its extreme, groups equal to channel count, grouped convolution becomes
depthwise convolution: each output channel depends on exactly one input
channel's spatial neighborhood, nothing else. That extreme is worth trying
first for a reason beyond compute savings: a depthwise conv factors "mix
space" and "mix channels" into two completely separate operations — the
surrounding 1x1 convs handle channel mixing, the depthwise conv handles
spatial mixing only — which is exactly the structure a self-attention block
already has (attention mixes spatially per-channel, the feedforward
sublayer mixes channels, and neither does both). If part of what makes
Transformer-style networks work is keeping those two kinds of mixing
cleanly separated, depthwise convolution is the direct convolutional
analogue of the spatial-mixing half, and it's worth testing as a hypothesis,
not only as a FLOPs trick.

I expect a real cost before I expect a gain. A depthwise conv at the
network's *current* width has far less capacity per parameter than the full
3x3 conv it replaces — it cannot combine information across channels at
all, and that job gets pushed onto 1x1 layers still sized for a
full-mixing conv, not compensated for a channel-mixing bottleneck that just
lost a big chunk of its job. So swapping to depthwise convolution alone, at
unchanged width, should cost accuracy relative to 79.51% even as FLOPs drop
substantially. That's not a reason to skip the change; it's the predicted
first half of a two-part move, worth measuring on its own so the size of
the capacity hole is visible before deciding how much width to add back.

The second half is the standard fix: widen, spending the freed FLOPs on
more channels rather than banking them, since every layer — including the
1x1s now carrying the entire channel-mixing burden — benefits from more
capacity. I need a concrete target, and there's a natural one already in
this exploration: Swin-T's width. I've already matched Swin-T's stage ratio
and patch size, so matching its channel count too — 96 at the first stage,
versus ResNet-50's current 64 — keeps every structural comparison to it
apples-to-apples and removes a free parameter I'd otherwise be choosing by
taste. This should restore, and plausibly exceed, the FLOPs cut by going
depthwise, landing back near the 4.4-4.5G budget I've been tracking. I
expect the width increase to at least recover the macro-design point,
because 64-to-96 is a fairly modest multiple against how much compute the
grouping extreme frees up, and the extra capacity lands exactly where the
loss occurred — in the 1x1 layers doing the channel mixing. If widening
doesn't recover the loss, that would say the depthwise-plus-1x1
factorization is a net capacity loss even after paying for it with width,
and not worth carrying into the rest of the ladder — but that is not what I
expect.

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
