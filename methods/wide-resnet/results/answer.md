# Wide Residual Networks

The method is to keep the residual unit simple, spend capacity on channel width instead of extreme depth, and regularize the enlarged residual branch without corrupting the identity shortcut.

Core architecture:

- Residual block: pre-activation `B(3,3)`, i.e. `BN -> ReLU -> 3x3 conv -> BN -> ReLU -> dropout -> 3x3 conv`, added to the raw identity when input/output planes match or to a stride-matched `1x1` projection of the first preactivation when they do not.
- Stages: fixed `3x3` stem with 16 channels, then three groups of `N` blocks at `16k`, `32k`, and `64k` channels.
- Downsampling: first block of groups 2 and 3 uses stride 2.
- Depth convention: `N = (depth - 4) / 6`, so valid CIFAR/SVHN depths satisfy `(depth - 4) % 6 == 0`.
- Name: `WRN-depth-k`, e.g. `WRN-28-10`.
- Dropout: inside the residual branch only, after the second BN/ReLU and before the second convolution; use about `0.3` on CIFAR and `0.4` on SVHN.
- Training: SGD with Nesterov momentum, cross-entropy, LR `0.1`, momentum `0.9`, dampening `0`, weight decay `5e-4`, batch size `128`; CIFAR uses horizontal flips and random crops from 4-pixel reflected padding, with LR multiplied by `0.2` at epochs `60/120/160` for 200 epochs. SVHN uses LR `0.01`, LR multiplied by `0.1` at epochs `80/120`, 160 epochs, no augmentation.

PyTorch transliteration of the official Torch/Lua CIFAR/SVHN model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout=0.0):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.dropout = dropout

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False
            )

    def forward(self, x):
        pre = F.relu(self.bn1(x), inplace=True)
        residual = self.conv1(pre)
        residual = F.relu(self.bn2(residual), inplace=True)
        if self.dropout > 0:
            residual = F.dropout(
                residual, p=self.dropout, training=self.training
            )
        residual = self.conv2(residual)
        shortcut = x if self.equal_in_out else self.shortcut(pre)
        return shortcut + residual


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        blocks_per_group = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(
            widths[0], widths[1], blocks_per_group, stride=1, dropout=dropout
        )
        self.group2 = self._make_group(
            widths[1], widths[2], blocks_per_group, stride=2, dropout=dropout
        )
        self.group3 = self._make_group(
            widths[2], widths[3], blocks_per_group, stride=2, dropout=dropout
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        self._init_like_reference()

    def _make_group(self, in_planes, out_planes, count, stride, dropout):
        layers = [WideBasicBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(1, count):
            layers.append(WideBasicBlock(out_planes, out_planes, 1, dropout))
        return nn.Sequential(*layers)

    def _init_like_reference(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.avg_pool2d(x, 8, 1, 0).flatten(1)
        return self.fc(x)


def wrn_28_10(num_classes=10):
    return WideResNet(28, 10, num_classes=num_classes, dropout=0.3)


def wrn_16_8_svhn():
    return WideResNet(16, 8, num_classes=10, dropout=0.4)


model = wrn_28_10(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    dampening=0,
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[60, 120, 160], gamma=0.2
)
```
