# Wide Residual Networks (WRN), distilled

A wide residual network keeps the residual block but decreases depth and increases width: instead of a thousand thin, under-used residual blocks, it stacks a few dozen *wide* ones. A WRN matches or beats much deeper thin ResNets at comparable parameter count while training several times faster, and it regularizes the larger blocks with dropout inserted *between* the convolutions.

## The problem

Very deep residual networks improve with depth, but at a brutal exchange rate (≈doubling layers per fractional accuracy gain) and slow training. Worse, the identity shortcut that makes depth trainable also lets gradient bypass a block's weights, so many blocks learn almost nothing — *diminishing feature reuse*. Stochastic depth (randomly dropping whole blocks) improving very deep nets is direct evidence the depth is under-used. So: how wide should residual networks be, and is widening a more effective axis than depth?

## The key findings

- **Block layout barely matters at fixed parameters.** Comparing `B(3,3)`, `B(3,1)`, `B(1,3,1)`, `B(3,1,3)`, `B(1,3)`, `B(3,1,1)` at matched parameter count, all land in the same place; `B(3,3)` wins by a hair. Use the basic two-3×3 block.
- **Don't deepen the block.** At fixed total convolutions and parameters, deepening factor `l=2` beats `l=1,3,4`. More convs per block → fewer blocks → fewer residual connections → harder optimization. Keep `l=2`.
- **Width is the leverage.** Widen by a factor `k` (feature-plane multiplier). At fixed depth, accuracy rises monotonically with `k` (tested to 12×). At fixed `k`, depth helps up to ~28 then hurts (WRN-40-8 < WRN-22-8). A wide WRN-40-4 (8.9M) matches thin ResNet-1001 (10.2M) on CIFAR but trains ~8× faster; depth adds no regularization that width can't match. Width is GPU-friendly (big tensors parallelize), so it can be faster in wall-clock despite quadratic FLOPs.
- **Dropout between the convolutions.** Widening adds parameters → need regularization beyond BN (whose regularization needs heavy augmentation). Put dropout inside the residual branch *between* the two convolutions (after ReLU), never on the identity path (that hurts). It perturbs the next block's BN and fights diminishing feature reuse. p≈0.3 CIFAR, 0.4 SVHN; biggest gain on SVHN where, without augmentation, BN overfits.

## The architecture

- Pre-activation basic block: `BN → ReLU → conv3×3 → [dropout] → BN → ReLU → conv3×3`, added onto identity (or 1×1 projection when width/stride changes).
- Stem 3×3 at width 16; three groups of `N` blocks at widths `16k`, `32k`, `64k`; downsample (stride 2) at the first block of groups 2 and 3; final BN–ReLU → 8×8 global average pool → FC → softmax.
- Total conv layers `n = 6N + 4`. Notation **WRN-n-k** (e.g. WRN-28-10 = 28 layers, 10× width). `k=1` recovers the original thin ResNet.

## Training recipe

SGD with Nesterov momentum, cross-entropy. Init LR 0.1, weight decay 5e-4, dampening 0, momentum 0.9, minibatch 128. CIFAR: LR ×0.2 at epochs 60/120/160, 200 epochs total; light augmentation (horizontal flip + random crop from 4-pixel reflect-pad), mean/std normalization. SVHN: init LR 0.01, ×0.1 at 80/120, 160 epochs, no augmentation. (A weight-decay-induced loss oscillation appears after the first LR drop; lowering weight decay hurts accuracy, so keep it; dropout partially mitigates it.)

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = dropout
        self.equal = (in_planes == out_planes and stride == 1)
        if not self.equal:
            self.convdim = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                     stride=stride, bias=False)

    def forward(self, x):
        o1 = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(o1)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)   # between convs
        o2 = F.relu(self.bn2(y), inplace=True)
        z = self.conv2(o2)
        shortcut = x if self.equal else self.convdim(o1)
        return z + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth, k, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        widths = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], n, 1, dropout)
        self.group2 = self._make_group(widths[1], widths[2], n, 2, dropout)
        self.group3 = self._make_group(widths[2], widths[3], n, 2, dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_group(self, in_planes, out_planes, n, stride, dropout):
        layers = [BasicBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_planes, out_planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x); x = self.group2(x); x = self.group3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.avg_pool2d(x, 8).flatten(1)
        return self.fc(x)


def wrn_28_10(num_classes=10): return WideResNet(28, 10, num_classes, dropout=0.3)
def wrn_16_8(num_classes=10):  return WideResNet(16, 8,  num_classes, dropout=0.0)


model = wrn_28_10(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True, dampening=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
```
