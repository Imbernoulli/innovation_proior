# CBAM (Convolutional Block Attention Module)

## Problem

Depth, width, and cardinality scale a CNN's capacity but apply it uniformly across all channels and spatial locations, even though much of a feature map is irrelevant for a given input. CBAM is a lightweight, plug-and-play module that, given an intermediate feature map, learns input-adaptive attention along two complementary axes — channel ("what") and spatial ("where") — and refines the feature map in place, with negligible parameter/compute overhead, insertable at every convolutional block.

## Key idea

Given F ∈ R^{C×H×W}, apply channel attention then spatial attention sequentially:
- F' = M_c(F) ⊗ F (channel gate, broadcast over space)
- F'' = M_s(F') ⊗ F' (spatial gate, broadcast over channels)

**Channel attention** (eq):
M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) ) = σ( W_1·ReLU(W_0·F^c_avg) + W_1·ReLU(W_0·F^c_max) ),
W_0 ∈ R^{C/r×C}, W_1 ∈ R^{C×C/r}, MLP **shared** between the two descriptors, reduction ratio r=16. Using both avg-pool (smooth global summary, as in SE) and max-pool (distinctive-peak cue) yields finer attention than either alone; avg-only reduces to the SE channel gate.

**Spatial attention** (eq), symmetric on the channel axis:
M_s(F) = σ( f^{7×7}( [AvgPool_chan(F); MaxPool_chan(F)] ) ),
i.e. avg- and max-pool along channels → two 1×H×W maps → concat → one 7×7 conv → 1×H×W → sigmoid.

**Arrangement:** sequential (channel → spatial) beats parallel; channel-first beats spatial-first.

CBAM is inserted inside each residual block, refining the conv output before the residual addition.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pooled = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pooled = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            att = self.mlp(pooled)
            channel_att_sum = att if channel_att_sum is None else channel_att_sum + att
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        scale = torch.sigmoid(self.spatial(x_compress))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
```

Integrated by inserting `CBAM(planes)` inside a residual block, applied to the block's conv output before adding the shortcut. Validated on ImageNet-1K (ResNet/WideResNet/ResNeXt/MobileNet), MS COCO, and VOC 2007; r=16 default.
