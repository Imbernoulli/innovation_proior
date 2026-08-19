**Question.** Re-measure the strongest configurations found so far (40-4, 16-8, 28-10) under mean/std
normalization — the same preprocessing convention every thin-ResNet baseline reports under — for a
protocol-matched comparison, and re-test dropout under that corrected protocol across a wider span of
configurations (16-4, 28-10, 52-1) and datasets (CIFAR-10, CIFAR-100, SVHN).

**Design.** Preprocessing switched from ZCA whitening to simple mean/std normalization for CIFAR;
SVHN uses no image preprocessing beyond scaling to `[0,1]`, unchanged from the initial protocol. No
architectural change from the settled recipe: `B(3,3)` block (rung 1), `l=2` (rung 2), stage widths
`16, 16k, 32k, 64k` (rung 3), dropout inside the residual branch only, between the two convolutions
(rung 4's placement argument, unaffected by the preprocessing question). Dropout probability
cross-validated per dataset as in rung 4's methodology: `0.3` on CIFAR, `0.4` on SVHN (no augmentation
there, so a different overfitting regime). Two comparisons run side by side: (a) no-dropout numbers for
40-4, 16-8, 28-10 against the thin pre-act-ResNet / original-ResNet / stochastic-depth family, all now
on the same mean/std footing; (b) a dropout ablation at 16-4, 28-10, and 52-1 (the last one close to
the thin end of this family, `k=1`), each measured with and without dropout, on CIFAR-10, CIFAR-100,
and (where applicable) SVHN.

Final architecture and training recipe — the full transliteration used for every configuration in this
rung:

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


def wrn_28_10(num_classes=10, dropout=0.3):
    return WideResNet(28, 10, num_classes=num_classes, dropout=dropout)


def wrn_16_8_svhn(num_classes=10, dropout=0.4):
    return WideResNet(16, 8, num_classes=num_classes, dropout=dropout)


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

**Protocol.** CIFAR-10/CIFAR-100: mean/std normalization, flip + 4px reflected-pad crop, SGD Nesterov
lr 0.1, momentum 0.9, weight decay 5e-4, batch 128, 200 epochs, lr x0.2 at 60/120/160. SVHN: no
preprocessing beyond `[0,1]` scaling, no augmentation, lr 0.01, lr x0.1 at 80/120, 160 epochs total.
Median test error over 5 runs where applicable. Run the no-dropout comparison (40-4, 16-8, 28-10)
against the mean/std-reported thin baselines, and the dropout ablation (16-4, 28-10, 52-1; CIFAR-10,
CIFAR-100, SVHN where applicable) with and without dropout at the cross-validated probability.
