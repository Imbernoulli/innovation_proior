# Wide Residual Networks

Residual shortcuts let me train networks that would otherwise be impossible to optimize: a block computes $x_{l+1} = x_l + F(x_l, W_l)$, so the signal has an exact additive path around the residual weights, and the degradation that made plain very deep nets untrainable largely disappears. But the empirical exchange rate is poor. Once I am near the best thin residual models, a small accuracy gain demands a very large increase in layers, with the extreme CIFAR variants pushing past a thousand layers at high training cost. That is a warning that depth is no longer being converted into useful representation efficiently. The identity shortcut explains why: it is the optimization gift, but it is also an escape hatch. In backpropagation the gradient can travel through the shortcut even when the residual branch is weak, so there is no force compelling every extra block to learn a genuinely new transformation. Stochastic depth makes the suspicion concrete — if whole residual blocks can be randomly dropped during training and a very deep net still improves, then many of those blocks are not individually indispensable. The thin deep baseline and its pre-activation refinement are the strong reference points, but both treat depth as the one scaling axis to maximize, and that is exactly the habit I want to test.

I propose Wide Residual Networks (WRN): keep the residual unit simple, spend the capacity budget on channel width rather than extreme depth, and regularize the enlarged residual branch without ever touching the identity shortcut. The first thing to settle is what the block should contain. There are three honest ways to add power inside a block — more convolutions, a different mix of $3\times3$ and $1\times1$ filters, or more feature planes — and I deliberately exclude larger spatial kernels, since the VGG/Inception evidence already favors stacked small filters and a bigger kernel would only confound the question with an older design choice. Comparing block structures at matched parameter count (the basic $B(3,3)$ against $B(3,1)$, $B(1,3)$, $B(1,3,1)$, $B(3,1,3)$, $B(3,1,1)$) gives close results: $B(3,3)$ wins by a small margin while several mixed variants sit just behind it and cheaper, which tells me the internal filter mix is not where the leverage lives. So I keep the cleanest unit, the pre-activation two-convolution block in the $\texttt{BN}\!\to\!\texttt{ReLU}\!\to\!\texttt{conv}$ order, because the identity-mapping work has already shown that this ordering preserves a cleaner identity path after the addition. I also have to resist the tempting idea that, since the block is the useful unit, making each block deeper should help. Holding the total convolution count and parameter budget fixed and varying the deepening factor $l$, the experiment says no: $l=2$ wins, $l=1$ is too weak, and $l=3$ or $l=4$ is worse. The reason is structural — with the convolution budget fixed, deeper blocks mean fewer blocks and therefore fewer shortcuts, and since the shortcut is part of the optimization machinery rather than a passive wire, removing addition points makes the network harder to train. The bottleneck block is wrong here for the same reason: it exists to thin a block so one can stack more layers, which would bake the old depth-first answer right back into the design.

That leaves channel count, the very axis the thin designs minimized to buy depth. Width costs quadratically in parameters and FLOPs while extra blocks cost only linearly, so on paper width looks expensive — but wall-clock training is not paper arithmetic. A thousand small sequential convolutions underuse a GPU, whereas fewer, wider convolutions hand the hardware more parallel work, so the metric that matters is accuracy per unit of training time, not parameters alone. I therefore introduce a single width multiplier $k$ applied to the feature-plane counts of the three residual groups, giving stage widths $16$, $16k$, $32k$, $64k$ on top of a fixed $3\times3$ stem of $16$ channels. Sweeping $k$ and depth together reveals the pattern that is the answer: at fixed modest depth, increasing $k$ keeps improving CIFAR results across the tested range, while at fixed large $k$, increasing depth helps at first and then stalls or hurts. Concretely, a 40-layer model at $k=4$ reaches roughly the parameter scale of the 1001-layer pre-activation baseline and compares favorably on CIFAR while training far faster, and a 28-layer model at $k=10$, though much larger, trains cleanly and beats that thin 1001-layer reference. The old depth-to-width ratio was never sacred; it was one expensive point in the design space. The depth is parameterized so each group holds an equal number of blocks: I require $(\text{depth}-4)\bmod 6 = 0$ and set $N = (\text{depth}-4)/6$ blocks per group (the $4$ accounting for the stem convolution, the final classifier convolution-equivalent, and the per-group bookkeeping), with downsampling by stride $2$ at the first block of groups 2 and 3 and the classifier ending in a final BN, ReLU, $8\times8$ global average pool, and a linear layer. I name instances $\texttt{WRN-}d\texttt{-}k$, for example WRN-28-10.

Widening multiplies the parameter count, and batch normalization alone is not a complete regularizer — especially when augmentation is weak. SVHN is the clean stress test: no augmentation, only scaling to $[0,1]$, and the curves can show the model driving training loss very low while test error stalls, which is exactly the regime that needs extra stochastic regularization. The crucial design choice is where the dropout goes. It must not sit on the shortcut: the shortcut is the stable path that carries information forward and gradients backward, and randomly damaging it fights the very mechanism that makes residual nets trainable. Dropout therefore lives strictly inside the residual branch, after the second BN/ReLU and immediately before the second convolution — first BN/ReLU and conv, then second BN/ReLU, then dropout, then the second conv. This perturbs the residual transformation while leaving the identity path exact, and it hands the next block's batch normalization a less stale activation distribution; I use about $0.3$ on CIFAR and $0.4$ on SVHN. The shortcut semantics have to be precise for this to hold: when input and output planes match, the addition uses the raw identity input; when they change, the shortcut is a $1\times1$ projection carrying the block's stride, applied to the first preactivated tensor $\texttt{pre} = \texttt{ReLU}(\texttt{BN}(x))$ so that the projected shortcut and the residual align. Convolutions carry no bias, and weights use He/MSR fan-in initialization with standard deviation $\sqrt{2/(k_W k_H\, n_{\text{in}})}$, which in PyTorch means $\texttt{mode="fan\_in"}$ rather than fan-out. Training is SGD with Nesterov momentum on cross-entropy, learning rate $0.1$, momentum $0.9$, dampening $0$, weight decay $5\times10^{-4}$, batch size $128$; CIFAR uses horizontal flips and random crops from 4-pixel reflected padding with the learning rate multiplied by $0.2$ at epochs $60/120/160$ over 200 epochs, while SVHN starts at learning rate $0.01$ and multiplies by $0.1$ at epochs $80/120$ over 160 epochs with no augmentation.

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
