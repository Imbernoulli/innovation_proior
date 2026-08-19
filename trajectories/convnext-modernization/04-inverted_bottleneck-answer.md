**Problem.** Depthwise conv + width 96 (rung 3) recovered accuracy to
80.50% but pushed FLOPs to 5.27G, above the 4.5G Swin-T reference. The
current block is still wide-narrow-wide (bottleneck): boundary width is
wide, middle (depthwise) width is narrow, so every stage-transition
shortcut 1x1 conv connects at the wide boundary width — the single most
expensive 1x1 conv per stage. Invert the shape: narrow-wide-narrow, matching
both mobile-efficiency ConvNet precedent and the Transformer feedforward
sublayer's expand-4x-then-project shape.

**Change (against the rung 3 depthwise+width-96 baseline).**

- Block boundary (input/output, and what the shortcut connects) stays at
  the current narrow "planes" width.
- First 1x1 conv expands 4x from the boundary width.
- Depthwise conv operates at the expanded (4x) width.
- Second 1x1 conv projects back down to the boundary width.
- Downsampling shortcuts now connect narrow-to-narrow instead of
  wide-to-wide.

```python
# rung 4: inverted bottleneck -- narrow -> wide(4x) -> narrow, depthwise at the wide point.
import torch.nn as nn

class InvertedBottleneck(nn.Module):
    """narrow -> 1x1 expand 4x -> depthwise -> 1x1 project -> narrow (+ shortcut)."""
    expand_ratio = 4

    def __init__(self, dim, stride=1, downsample=None):
        super().__init__()
        hidden = dim * self.expand_ratio
        self.pw_expand = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                             groups=hidden, bias=False)          # depthwise, now at 4x width
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw_project = nn.Conv2d(hidden, dim, 1, bias=False)  # back to narrow boundary
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample     # now narrow -> narrow, cheaper than rung 3's shortcut

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.pw_expand(x)))
        out = self.relu(self.bn2(self.dw(out)))
        out = self.bn3(self.pw_project(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

def make_stage(dim_in, dim_out, blocks, stride=1):
    downsample = None
    if stride != 1 or dim_in != dim_out:
        downsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, stride, bias=False),   # narrow -> narrow now
            nn.BatchNorm2d(dim_out),
        )
    layers = [InvertedBottleneck(dim_out, stride=stride, downsample=downsample)
              if dim_in == dim_out else
              InvertedBottleneck(dim_in, stride=stride, downsample=downsample)]
    for _ in range(1, blocks):
        layers.append(InvertedBottleneck(dim_out))
    return nn.Sequential(*layers)

# Boundary widths are the stage dims directly now (no separate "planes x4"
# expansion factor at the block-IO level -- the 4x expansion is internal).
dims = [96, 192, 384, 768]      # matches rung 3's width-96 base, no more x4 IO expansion
depths = [3, 3, 9, 3]           # unchanged since rung 2
```

**Test.** Same frozen recipe. Compare against rung 3's 80.50% / 5.27G — the
primary question is whether FLOPs come back down toward the 4.5G band
without giving back accuracy.
