# Squeeze-and-Excitation Networks (SENet)

## Problem

A convolution fuses spatial and channel information within a *local* receptive field, and its channel mixing is fixed after training and identical for every input. So channel interdependencies are modelled only implicitly and locally, with no mechanism to use *global, image-wide* context to emphasise the channels that are informative for the current image. The Squeeze-and-Excitation (SE) block adds exactly that: a lightweight, input-dependent recalibration of channel responses.

## Key idea

Given the feature maps U in R^{H x W x C} produced by any transformation (a conv, an Inception module, or a residual branch), three operations:

1. **Squeeze** — global average pooling over space gives a per-channel descriptor with a global receptive field:

       z_c = (1 / (H·W)) · Σ_{i=1..H} Σ_{j=1..W} u_c(i, j),   z ∈ R^C.

2. **Excitation** — a bottleneck gating MLP maps z to per-channel weights:

       s = σ( W₂ · δ( W₁ z ) ),   δ = ReLU,  σ = sigmoid,
       W₁ ∈ R^{(C/r) x C},  W₂ ∈ R^{C x (C/r)}.

   The C→C/r→C bottleneck limits capacity and aids generalisation; the reduction ratio r (default 16) trades cost against capacity. **Sigmoid, not softmax**, so the gates are non-mutually-exclusive — several channels may be emphasised at once rather than competing for a fixed budget.

3. **Scale** — channel-wise multiply by the gates:

       x̃_c = s_c · u_c.

Since s_c ∈ (0,1), the block attenuates uninformative channels and preserves informative ones. It is a self-attention over channels with global reach.

## Cost

The only added parameters are the two FC layers. Per block, with no biases: (C/r)·C + C·(C/r) = **2C²/r**. Over a network with stages s of N_s blocks at width C_s:

    (2/r) · Σ_s N_s · C_s²

For SE-ResNet-50 at r = 16 this is ~2.5M extra parameters on ~25M (~10%), and FLOPs rise from ~3.86 to ~3.87 GFLOPs (~0.26%), because the FC layers act on length-C vectors, not on H x W maps. Most of the cost is in the final stage (largest C); dropping SE there reduces the increase to ~4% at negligible accuracy cost.

## Integration

The SE block is a drop-in: insert after the nonlinearity of a conv; wrap an entire Inception module; or, for residual networks, recalibrate the **residual branch output before the identity addition** (so the skip-connection highway is left ungated). Stacking SE blocks compounds the recalibration through depth.

## Code

```python
import torch
import torch.nn as nn


class SELayer(nn.Module):
    """Squeeze-and-Excitation: recalibrate channels using global context."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)               # squeeze: GAP -> z in R^C
        self.fc = nn.Sequential(                              # excitation
            nn.Linear(channel, channel // reduction, bias=False),  # C -> C/r
            nn.ReLU(inplace=True),                                 # delta
            nn.Linear(channel // reduction, channel, bias=False),  # C/r -> C
            nn.Sigmoid(),                                          # sigma: gates in (0,1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)        # squeeze
        s = self.fc(z).view(b, c, 1, 1)        # excitation
        return x * s.expand_as(x)              # scale


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                     # recalibrate residual branch
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual                        # before the identity add
        return self.relu(out)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)   # on expanded width
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)
```
