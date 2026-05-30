Let me start from the exchange rate, because it's absurd and it's the thing nagging me. Residual networks scale to a thousand layers and keep getting better — that's the headline. But look at the *cost* of each gain: to buy another fraction of a percent of accuracy you roughly *double* the number of layers. A 1001-layer net is the state of the art, and it trains for an eternity. That can't be the right way to spend compute. So I want to ask a question the field hasn't really asked: is all that depth actually being *used*, and if not, is there a cheaper axis that buys the same accuracy?

There's a structural reason to suspect the depth is mostly idle, and it comes straight out of what makes residual nets trainable in the first place. A residual block computes `x_{l+1} = x_l + F(x_l, W_l)` — an identity shortcut plus a learned residual. The identity is exactly what lets gradient reach the bottom of a very deep net. But stare at the backward pass: the gradient has a free path straight through the identity, and *nothing forces it to flow through the block's residual weights* `W_l`. So a block can sit there passing its input along essentially unchanged, learning almost nothing, and the network is perfectly happy because the identity carries the signal anyway. The likely picture, then, is that in a thousand-layer net only a handful of blocks learn anything substantial, and the rest contribute tiny refinements or nothing — feature reuse diminishes as you stack more blocks. Highway networks named this. And there's a sharp piece of evidence that it's real: stochastic depth trains very deep nets by *randomly dropping entire residual blocks* during training — which is a per-block dropout, each block scaled by a Bernoulli identity weight. If randomly deleting whole blocks not only doesn't break the net but *improves* it, then those blocks were largely redundant. So the depth is real but under-exploited. That reframes my question: instead of fighting to *use* a thousand thin blocks, what if I build *fewer, fatter* blocks?

Before I jump to "make it wider," let me actually study the block, because the field's attention has been narrow — almost everything published is about the order of activations inside a block and about depth. There's a much bigger design space. A residual block's representational power can be raised three ways: add more convolution layers per block, widen the convolution layers with more feature planes, or use bigger filters. The third I'll cut immediately — small 3×3 filters have been shown over and over (VGG, the Inception line) to be the right primitive, so I won't go above 3×3. That leaves two factors to turn: a deepening factor `l`, the number of convolutions in a block, and a widening factor `k`, the multiplier on the number of feature planes. The plain basic block — two stacked 3×3s — is `l = 2, k = 1`.

A couple of setup decisions first. I'll use the pre-activation ordering, `BN → ReLU → conv`, inside each block rather than the original `conv → BN → ReLU`, because pre-activation has been shown to train faster and reach better accuracy, so there's no reason to study the worse order. And I'll drop the bottleneck block entirely. The bottleneck — `1×1 reduce → 3×3 → 1×1 expand` — exists precisely to make a block *thinner* so you can afford more layers; it's a tool for the depth-maximizing strategy I'm questioning. Since I want to study *widening*, a block whose whole purpose is thinning is the wrong object. So I keep the feature-plane count constant across the block and work with the basic 3×3 form.

Now let me probe what's inside the block, the convolution layout, because I shouldn't assume two 3×3s is best just because it's the default. Let me write `B(M)` for a block whose convolution kernels are the list `M` — so `B(3,3)` is the basic block, and I can compare it against variants that swap some 3×3s for cheaper 1×1s or rearrange them: `B(3,1,3)` with an extra 1×1 in the middle, `B(1,3,1)` (a "straightened" bottleneck where all convs keep the same width), `B(1,3)` and `B(3,1)` alternating, `B(3,1,1)` in the Network-in-Network style. To make this a fair fight I hold the *parameter count* roughly constant across the variants — comparing at matched capacity, so any accuracy difference is about the structure, not the size. Run them at `k = 2` with depth adjusted per variant to equalize parameters. What comes back is almost flat: `B(3,3)` is best, but only by a hair; `B(3,1)` and `B(3,1,3)` are right behind it with fewer parameters and fewer layers, and `B(3,1,3)` is even slightly faster. The honest reading is that *at comparable parameter count the block's internal layout barely matters* — they all land in the same place. So I'll just use `B(3,3)`, both because it's marginally best and because it keeps me consistent with everyone else's 3×3-only nets. The layout isn't where the leverage is.

Next, how many convolutions per block — the deepening factor `l`? I have to be careful to compare at fixed total work: hold the parameter count fixed (2.2M) *and* the total number of convolution layers fixed, then vary `l ∈ {1,2,3,4}`, which forces the number of blocks `d` to shrink as `l` grows (more convs per block means fewer blocks for the same total). Run it: `l = 2`, the plain `B(3,3)`, wins. `l = 1` is clearly worse, and `l = 3` and `l = 4` are worse than `l = 2`, with `l = 4` the worst of all. Why would *deeper blocks* hurt when the total depth and parameters are held fixed? Because for a fixed total number of convolutions, packing more convs into each block means *fewer blocks*, and fewer blocks means *fewer residual connections*. Each skip is an optimization aid; cut their number and the net gets harder to optimize. So the residual connections themselves are doing real work, and I don't want to thin them out. `l = 2` it is — two 3×3s per block, and don't deepen the block.

So both interior knobs — the conv layout and the per-block depth — point back to the simple `B(3,3)`, `l = 2` block. That leaves the one axis I haven't yet turned, and the one the whole field skipped: the width `k`. This is where I expect the leverage to be, and let me reason about why before I run it. First, an existence argument: nearly every successful architecture *before* residual nets — VGG, Inception — was far wider than the thin residual nets, and nobody thought those were over-parameterized to uselessness. The residual designers went thin on purpose, chasing depth, partly on circuit-complexity intuitions that depth is exponentially more expressive than width. But the diminishing-feature-reuse evidence says their extreme thinness left a lot of those deep blocks idle. Second, a hardware argument: parameters and compute grow *quadratically* in the width factor `k` but only *linearly* in the number of blocks — so naively width looks expensive. But a thin, deep net is a long *sequential* chain of tiny convolutions, and that's exactly what a GPU is bad at; the GPU wants big tensors it can parallelize over. So widening, even though it adds quadratic FLOPs, can run *faster* in wall-clock than an equally-accurate thin-deep net, because it uses the hardware properly. There should be an optimal depth-to-width ratio, and I doubt it's "as thin as possible."

Sweep it: widening factor `k` from 2 up to 12, depth from 16 to 40. Two clean trends. Holding depth fixed and increasing `k` from 1× toward 12×, the 40-, 22-, and 16-layer nets all improve steadily — width buys accuracy. And holding `k` fixed (say 8 or 10) while increasing depth, accuracy improves from 16 to 28 layers but then *decreases* going to 40 — a wide-40 loses to a wide-22 at the same width. So past a point, more depth at fixed width hurts, while more width keeps helping. The headline comparison I care about: a wide WRN-40-4 (8.9M params) matches the thin ResNet-1001 (10.2M params) on accuracy at comparable parameter count — but trains about eight times faster. Same accuracy, same rough size, a fraction of the layers, a fraction of the wall-clock. That tells me two things at once. The depth-to-width ratio in the thin nets was far from optimal. And depth was adding no special regularization that width can't match — at equal parameters the wide net learns the same or better representation. I can even push past ResNet-1001's parameter count: wide nets with two or more times its parameters train fine and beat it, where matching that capacity with a thin net would mean doubling its depth into the infeasible.

I'll fix the network family from this. A 3×3 stem at width 16; then three groups of `N` basic blocks each, at widths `16k`, `32k`, `64k`, halving the spatial map at the start of the second and third groups (and widening there); a final BN–ReLU, global average pool, and a classification layer. The total number of convolution layers is `n = 6N + 4` (three groups times `N` blocks times two convs, plus the stem and the FC's input conv), and `k` scales the three groups' widths; `k = 1` recovers the original thin net. I'll name a net WRN-`n`-`k` for `n` total conv layers and width `k` — so WRN-28-10 is 28 layers, ten times wider than the original. WRN-22-8 and WRN-16-10, incidentally, come out comparable to VGG in width, depth, and parameter count, which is a nice sanity check that I'm back in the regime that worked before.

Now the regularization problem I created. Widening multiplies the parameters, so I need to guard against overfitting. Batch norm already regularizes, but its regularization leans on heavy data augmentation, and I want to handle the case where there isn't much — SVHN, for instance, where I do no augmentation at all and I can watch the training loss collapse to near zero while the test error stalls, which is BN overfitting in plain sight. So I want to bring dropout back. The question is *where* to put it in a residual block, and there's a known landmine: putting dropout in the *identity* path of the block was tried and reported to hurt — which makes sense, because the identity path is the thing carrying the signal and the gradient, and randomly zeroing it is sabotaging the very mechanism that makes residual nets work. So not there. Instead I'll insert the dropout *inside* the residual branch, between the two convolutions (after the ReLU). Two reasons it should help here specifically. It perturbs the activations going into the next block's batch norm, which keeps that BN from overfitting to a stale activation distribution. And — connecting back to where I started — by injecting noise into the residual branch it forces different blocks to actually learn, pushing against diminishing feature reuse rather than letting blocks coast on the identity. I cross-validate the drop probability: about 0.3 on CIFAR, 0.4 on SVHN, and I don't need extra epochs. On CIFAR with its augmentation the gain is small; on SVHN, where BN was overfitting, the gain is large.

One training wrinkle I should record because it nearly looks like a bug. With the standard recipe, after the first learning-rate drop the loss and validation error suddenly jump up and oscillate at high values until the next drop. Tracking it down, it's the weight decay causing it. The reflex is to lower the weight decay — but doing that significantly hurts final accuracy, so I keep it where it is. Interestingly the between-convolution dropout partially smooths this oscillation away, which is a small bonus on top of its regularization role.

Let me pin down the recipe so the comparisons are clean and reproducible. SGD with Nesterov momentum and cross-entropy loss; initial learning rate 0.1, weight decay 0.0005, dampening 0, momentum 0.9, minibatch 128. On CIFAR, drop the learning rate by 0.2 at epochs 60, 120, and 160, training 200 epochs total. On SVHN, start at 0.01 and drop by 0.1 at epochs 80 and 120, for 160 epochs. CIFAR augmentation is light — horizontal flips and random crops from a 4-pixel reflect-padded image — with mean/std normalization; SVHN gets nothing but a divide by 255. The depth has to satisfy `n = 6N + 4` so the blocks divide evenly among the three groups.

What I'd want to validate, stated sharply: at comparable parameter count, a wide shallow net should match or beat a thin very-deep one on test error while training several times faster; widening at fixed depth should improve accuracy monotonically up to large `k`; and the between-convolution dropout should give consistent gains, largest exactly where BN was overfitting (SVHN, no augmentation). If those hold, the conclusion is that the power of deep residual nets lives mostly in the residual *blocks*, and depth beyond a modest amount is supplementary.

Let me write it as code, filling the one open slot — the residual block — with the pre-activation basic block plus the between-convolution dropout, and leaving the width/depth-parameterized harness as it was.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # pre-activation basic block B(3,3): BN-ReLU-conv -> dropout -> BN-ReLU-conv,
    # added onto an identity (or 1x1 projection when width/stride changes) shortcut.
    def __init__(self, in_planes, out_planes, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = dropout
        # projection shortcut only when channels or resolution change
        self.equal = (in_planes == out_planes and stride == 1)
        if not self.equal:
            self.convdim = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                     stride=stride, bias=False)

    def forward(self, x):
        o1 = F.relu(self.bn1(x), inplace=True)               # pre-activation
        y = self.conv1(o1)
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)  # between convs
        o2 = F.relu(self.bn2(y), inplace=True)
        z = self.conv2(o2)
        shortcut = x if self.equal else self.convdim(o1)     # project from the pre-act signal
        return z + shortcut                                  # residual add (no final ReLU: pre-act)


class WideResNet(nn.Module):
    def __init__(self, depth, k, num_classes=10, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        widths = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], n, stride=1, dropout=dropout)
        self.group2 = self._make_group(widths[1], widths[2], n, stride=2, dropout=dropout)
        self.group3 = self._make_group(widths[2], widths[3], n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_group(self, in_planes, out_planes, n, stride, dropout):
        layers = [BasicBlock(in_planes, out_planes, stride, dropout)]   # first block: stride/width
        for _ in range(1, n):
            layers.append(BasicBlock(out_planes, out_planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x); x = self.group2(x); x = self.group3(x)
        x = F.relu(self.bn(x), inplace=True)                 # final pre-activation
        x = F.avg_pool2d(x, 8).flatten(1)
        return self.fc(x)


def wrn_28_10(num_classes=10): return WideResNet(28, 10, num_classes, dropout=0.3)
def wrn_16_8(num_classes=10):  return WideResNet(16, 8,  num_classes, dropout=0.0)

model = wrn_28_10(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True, dampening=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
# CIFAR: flip + random crop from 4-px reflect-pad, mean/std norm, 200 epochs, batch 128.
```

The whole thing is one redirection of effort: the trainability of deep residual nets comes from the identity shortcut, but that same shortcut lets most blocks coast and under-contribute, so instead of stacking a thousand thin under-used blocks, build a few dozen *wide* ones — matched or better accuracy at comparable parameters, several times faster on the GPU — and regularize the now-larger blocks with dropout placed between the convolutions, never on the identity path.
