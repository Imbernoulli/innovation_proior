**Problem.** Every convolution in this ladder has stayed 3x3 since rung 0 —
the block-shape edits so far (macro layout, depthwise+width, inverted
bottleneck) never touched receptive field, the one property that most
obviously separates a conv from (even windowed) self-attention. Growing the
kernel on the current rung-4 block is expensive because the depthwise conv
sits at the 4x-expanded channel width; reorder first so the expensive
spatial op sits at the narrow, cheap width, mirroring attention-before-MLP
in a Transformer block, then sweep kernel size.

**Changes (against the rung 4 inverted-bottleneck baseline).**

1. Move the depthwise conv from between the two 1x1 convs to *before* the
   first one: narrow input -> depthwise conv (at narrow width) -> 1x1
   expand 4x -> 1x1 project back to narrow. Kernel size still 3x3 for this
   sub-step, isolating the reorder's own effect.
2. With the reorder in place, sweep kernel size in {3, 5, 7, 9, 11},
   adjusting only `kernel_size` / `padding` on the (now cheap) depthwise
   conv; everything else fixed.

```python
# rung 5: move depthwise conv above the 1x1 expansion; sweep kernel size.
import torch.nn as nn

class LargeKernelBlock(nn.Module):
    """narrow -> depthwise(k) [narrow width] -> 1x1 expand 4x -> 1x1 project -> narrow."""
    expand_ratio = 4

    def __init__(self, dim, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        pad = kernel_size // 2
        # depthwise now runs at the NARROW boundary width, not the 4x-expanded width
        self.dw = nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=pad,
                             groups=dim, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        hidden = dim * self.expand_ratio
        self.pw_expand = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw_project = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.dw(x)))              # spatial mixing first, narrow width
        out = self.relu(self.bn2(self.pw_expand(out)))      # dense mixing does the heavy lifting
        out = self.bn3(self.pw_project(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

# sub-step 1: reorder only, kernel_size=3 (compare to rung 4's 80.64% / 4.64G)
# sub-step 2: sweep kernel_size in (3, 5, 7, 9, 11) with the reorder in place
KERNEL_SIZES_TO_SWEEP = (3, 5, 7, 9, 11)
```

**Test.** Six measured points: (a) reorder-only at kernel=3, against rung
4's 80.64% / 4.64G; (b)-(f) the five-way kernel sweep. Looking for a
climb-then-plateau shape, checked against Swin-T's minimum local-window
size of 7x7 as the candidate saturation point; adopt whichever kernel size
the plateau settles at.
