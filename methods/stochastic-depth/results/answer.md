# Stochastic Depth

## Problem

Depth makes networks expressive but hard to train: vanishing gradients, diminishing feature reuse, and training time that scales linearly with depth. The wish is contradictory on its face — a *short* network during training (fast, strong gradient flow) but a *deep* network at test time (full capacity). Stochastic Depth resolves it by randomly dropping entire residual blocks during training while keeping the full network at test time.

## Key idea

In a ResNet, a block is H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + s_ℓ(H_{ℓ-1})), where s_ℓ is the shortcut: identity for same-shape blocks, average-pool plus zero-pad for CIFAR/SVHN transition blocks, or a projection shortcut in ImageNet-style transitions. Gate only the transformation with a per-mini-batch Bernoulli b_ℓ ∈ {0,1}, Pr(b_ℓ=1) = p_ℓ (survival probability):

**Training:** H_ℓ = ReLU(b_ℓ · f_ℓ(H_{ℓ-1}) + s_ℓ(H_{ℓ-1})).
- b_ℓ=1 → original block.
- b_ℓ=0 → residual branch removed. For same-shape identity shortcuts, H_ℓ = H_{ℓ-1} exactly because the block input is non-negative, so ReLU acts as identity. For transition blocks, the output is the shortcut alone (downsample/pad or projection), not a literal same-tensor identity. In all cases the expensive f_ℓ branch has no forward/backward computation and no parameter update.

**Survival schedule (linear decay):** anchor p_0=1, decay to p_L at the last block:
p_ℓ = 1 − (ℓ/L)(1 − p_L).
Early layers extract low-level features reused everywhere, so they should survive more often. Set p_L = 0.5 (training is insensitive to p_L).

**Expected training depth:** E(L̃) = Σ_{ℓ=1}^L p_ℓ. Under linear decay with p_L=0.5,
E(L̃) = Σ_{ℓ=1}^L [1 − ℓ/(2L)] = L − (L+1)/4 = (3L − 1)/4 ≈ 3L/4.
For a 110-layer CIFAR ResNet (L=54 blocks), E(L̃) ≈ 40 — train ~40 blocks, test with 54. Roughly 25% of training time is saved (more with smaller p_L).

**Test:** all blocks active, with each transformation recalibrated by its survival probability (Dropout-style scaling):
H_ℓ^Test = ReLU(p_ℓ · f_ℓ(H_{ℓ-1}^Test) + s_ℓ(H_{ℓ-1}^Test)).

**Why test error also drops:** (1) shorter expected training depth ⇒ shorter gradient chains ⇒ stronger gradients in early layers; (2) the 2^L on/off block subsets form an implicit ensemble of depth-varying networks with shared weights, combined at test by the survival-weighted forward rule. It regularizes even with Batch Normalization (unlike Dropout, which gives little benefit on deep BN-ResNets) because it shortens the network rather than thinning it.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDropBlock(nn.Module):
    """H = ReLU(skip(x) + gate * f(x)); one gate is sampled per mini-batch."""
    def __init__(self, in_ch, out_ch, stride=1, death_rate=0.0):
        super().__init__()
        self.death_rate = death_rate
        self.stride, self.in_ch, self.out_ch = stride, in_ch, out_ch
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def skip(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride, self.stride)
        if self.out_ch > self.in_ch:
            zeros = x.new_zeros(x.size(0), self.out_ch - self.in_ch, x.size(2), x.size(3))
            x = torch.cat([x, zeros], dim=1)           # zero-pad channels (option A)
        return x

    def forward(self, x):
        skip = self.skip(x)
        if self.training:
            if torch.rand((), device=x.device).item() >= self.death_rate:
                return self.relu(self.f(x) + skip)
            return self.relu(skip)                      # dropped block: shortcut only, no f compute
        return self.relu((1.0 - self.death_rate) * self.f(x) + skip)

class StochasticDepthResNet(nn.Module):
    def __init__(self, block_chs, num_classes=10, p_L=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, block_chs[0][0], 3, padding=1, bias=False),
            nn.BatchNorm2d(block_chs[0][0]), nn.ReLU(inplace=True),
        )
        L = len(block_chs)
        blocks = []
        for ell, (in_ch, out_ch, stride) in enumerate(block_chs, start=1):
            death_rate = (ell / L) * (1.0 - p_L)        # deathRate = 1 - p_ell
            blocks.append(ResidualDropBlock(in_ch, out_ch, stride, death_rate=death_rate))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_chs[-1][1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
```

The basic two-conv block is used on CIFAR/SVHN; the bottleneck block on ImageNet. A 110-layer CIFAR net has L=54 blocks; the method also trains nets beyond 1000 layers on CIFAR. The original Torch implementation stores `deathRate = 1 - p_ℓ`, sets linear decay as `deathRate_ell = (ell/L) * opt.deathRate`, closes a block when `rand < deathRate`, skips residual-branch gradient computation when closed, and scales by `1 - deathRate = p_ℓ` at test.
