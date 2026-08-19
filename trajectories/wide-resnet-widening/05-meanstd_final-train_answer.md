The dropout result from the last rung is genuinely mixed, and I don't want to smooth over that with a
tidy story I haven't earned. CIFAR-100 moved the way I hoped, 20.5% down to 20.0%; CIFAR-10 moved the
other way, 4.17% up to 4.39%, a small but real regression on the identical architecture and the
identical regularizer. If branch-placed dropout were simply "correct" the way ruling it off the
shortcut was correct, I'd expect it to help on both datasets even at different magnitudes — CIFAR-100's
ten-times-fewer-images-per-class regime giving it more room to help there specifically. One dataset
improving while the other regresses on the same run is not that clean story. Before I draw any
conclusion about dropout's value, I want to rule out that the *evaluation protocol* itself, rather than
the architecture, is the confound — because if it is, it's been sitting underneath every comparison
since the width-depth grid, not just this one.

There's a real asymmetry I inherited as a default and haven't examined carefully until now. Every
measurement since the block-type rung has used ZCA whitening, but the baselines I keep comparing
against — the thin pre-act-ResNet family, original ResNet, stochastic depth — all report their numbers
under simple mean/std normalization. Every comparison I've drawn against those baselines has quietly
been ZCA-versus-mean/std, treating the gap as purely architectural when part of it could be a
preprocessing artifact running in either direction. I was already careful about a training-protocol
mismatch once before — flagging that pre-act-ResNet-1001's headline 4.92% used the same batch size
(128) I use throughout, while a parenthetical 4.64% at batch size 64 wasn't directly comparable. I
should be at least as careful about preprocessing, since it touches every comparison rather than one
baseline's footnote.

Two separate reasons point at the same next move. The direct one: if I want any WRN-versus-thin-ResNet
comparison to be trustworthy, my own numbers need to be measured under the same mean/std convention the
baselines use — this is a correctness requirement independent of what dropout did, and it's overdue
regardless. The more speculative one: switching preprocessing gives me a way to check whether the
CIFAR-10 dropout regression was protocol-dependent. ZCA whitening decorrelates and rescales input
channels in a way mean/std normalization doesn't, and if dropout's interaction with batch normalization
is even mildly sensitive to what statistics the network's first layers see, a preprocessing change
could plausibly move where dropout's cost/benefit line sits. I don't have a confident mechanistic
argument for exactly how — this is a real unknown I'm measuring, not a conclusion I've pre-decided —
but re-measuring under the corrected protocol lets me find out instead of building further on an
unexamined confound.

So: switch CIFAR preprocessing to mean/std normalization, and re-measure two things properly. First,
the no-dropout comparison — CIFAR-10 and CIFAR-100 for 40-4, 16-8, and 28-10 under the corrected
protocol, giving a clean, like-for-like head-to-head against the thin pre-act-ResNet family,
stochastic depth, and the original ResNet, instead of the cross-preprocessing comparison I've been
running. This also finally lets me check the parameter-matched depth-versus-width question properly:
40-4 sits at 8.9M parameters, genuinely close to the 1001-layer reference's 10.2M, in a way 28-10's
36.5M never was — that's the clean matched-budget test the grid rung wanted and couldn't quite deliver,
since 28-10 turned out to be the grid's accuracy winner at a much larger budget, not a parameter-matched
one. Second, dropout itself needs re-running under the new protocol rather than assumed unchanged —
that's the whole reason I'm suspecting the protocol in the first place. I re-test it on 28-10 again for
direct comparability with its own ZCA-preprocessed predecessor, and I extend the comparison to two
points I haven't tested dropout on at all: 16-4, to see whether the effect depends on how much spare
capacity there is to regularize, and 52-1, near the thin end of this family, to check whether
branch-placed dropout helps even where width isn't the source of the extra capacity. And since
dropout's whole motivating case was overfitting risk under light augmentation, this is also where SVHN
belongs — no augmentation at all, the sharpest stress test for any regularization gap, and one I
haven't touched since the initial context flagged it as the edge case to watch.

This rung is really doing one thing under two headings: fixing a protocol mismatch that's been sitting
under every comparison since the grid, and using the fix as the occasion to close out the dropout
question properly, across more configurations and one more dataset than the single mixed result I had
going in. I'm not assuming the mean/std switch resolves CIFAR-10's regression in dropout's favor — I
don't have the mechanistic confidence for that specific prediction — but the no-dropout baseline
comparisons become trustworthy the moment they're measured on the same footing as what they're being
compared against, and that alone justifies the switch independent of what happens to dropout.

The recipe this rung fixes, holding every earlier design decision in place — `B(3,3)` block, `l=2`,
stage widths `16, 16k, 32k, 64k`, dropout inside the branch only, between the two convolutions,
cross-validated to `0.3` on CIFAR and `0.4` on SVHN — is the full transliteration below:

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

I run the no-dropout comparison (40-4, 16-8, 28-10) against the mean/std-reported thin baselines, and
the widened dropout ablation (16-4, 28-10, 52-1; CIFAR-10, CIFAR-100, and SVHN where applicable) under
this corrected protocol, and read off whether fixing the preprocessing mismatch changes the picture
the ZCA-preprocessed dropout comparison left unresolved.
