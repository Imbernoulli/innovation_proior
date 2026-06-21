# Inception-v3, distilled

Inception-v3 is a 42-layer convolutional image classifier that scales up the parallel-branch Inception architecture efficiently — spending each multiply-add where a set of design principles says it pays. It reaches strong ILSVRC-2012 accuracy at a fraction of the compute and parameters of denser networks, by factorizing convolutions, reducing grids without representational bottlenecks, and regularizing aggressively (dropout, batch-normalized auxiliary head, and label smoothing).

## The problem

Better ImageNet backbones transfer to every downstream vision task, so scaling up the backbone is the highest-leverage move — but naive scaling is wasteful: doubling all filter widths in a complex Inception net costs 4× compute and parameters. The goal is general principles for spending compute efficiently, and a re-engineering of the expensive pieces (large filters, grid reduction, the head, the objective).

## Design principles

1. Avoid representational bottlenecks, especially early; representation size should decrease gently.
2. Higher-dimensional representations are easier to process locally (more disentangled features, faster training).
3. Spatial aggregation can be done over lower-dimensional embeddings with little loss — reduce dimension before a spatial conv (adjacent units are correlated).
4. Balance width and depth; grow both together for a fixed compute budget.

## The key ideas

- **Factorize large filters into stacked 3×3.** Two stacked 3×3s = one 5×5 receptive field at (9+9)/25 → 28% less compute and parameters. Keep ReLU in both layers — a linear first layer is worse (76.2% vs 77.2% top-1 in a controlled run); the extra nonlinearity enlarges the function space, especially with BN. The 7×7 stem is likewise factorized into three 3×3s.
- **Asymmetric factorization.** n×n → 1×n then n×1. A 3×1 + 1×3 is 33% cheaper than a 3×3 (vs only 11% for 3×3→two 2×2). Use on medium grids (12–20), n=7 (1×7 then 7×1); it works poorly on early layers.
- **Efficient grid reduction.** Conv-then-pool avoids a bottleneck but is ~3× expensive; pool-then-conv is cheap but creates a representational bottleneck. Instead run a stride-2 conv branch and a stride-2 pool branch in parallel and concatenate — cheap and bottleneck-free.
- **Expanded modules on the coarsest 8×8 grid.** Put the 1×3 and 3×1 in parallel (not stacked) to widen the output — principle 2: high-dimensional local representation matters most where spatial extent is smallest.
- **Auxiliary classifier = regularizer, not gradient pump.** Training curves are identical with/without it until late; removing the lower of two heads is harmless. Batch-normalizing the side head improves the *main* classifier (+0.4% top-1). Keep one BN'd auxiliary head.
- **Label smoothing.** Replace the one-hot target with `q'(k) = (1−ε)δ_{k,y} + ε/K`. The cross-entropy decomposes as `H(q',p) = (1−ε)H(q,p) + ε H(u,p)`: the second term is (up to a constant) `D_KL(u‖p)`, pulling predictions toward the uniform prior. Fixes over-confidence; ε=0.1 gives ~0.2% top-1/top-5. (Gradient `∂ℓ/∂z_k = p(k)−q(k)`, bounded in [−1,1].)

## The architecture (input 299×299×3)

- **Stem:** conv 3×3/2 (→149²×32) → conv 3×3 (→147²×32) → conv 3×3 pad (→147²×64) → maxpool 3×3/2 → conv 1×1 (80) → conv 3×3 (192) → maxpool 3×3/2 → **35×35×192**.
- **3× InceptionA** (cheap reduced 5×5 branch plus a separate double-3×3 branch) → 35×35×288.
- **GridReductionB** → 17×17×768.
- **5× InceptionC** (1×7 / 7×1, c7 = 128/160/160/192) → 17×17×768. BN auxiliary head on the last one.
- **GridReductionD** → 8×8×1280.
- **2× InceptionE** (expanded parallel 1×3 / 3×1) → 8×8×2048.
- AvgPool → 2048 → dropout keep probability 0.8 → linear → softmax 1000. 42 layers, ~2.5× the BN-Inception predecessor's compute.

"Inception-v3" = this architecture + RMSProp + label smoothing + factorized-7×7 stem + batch-normalized auxiliary head.

## Training recipe

BN (eps 0.001) after every conv. Dropout with keep probability 0.8 before the main logits. RMSProp, decay 0.9, ε=1.0. LR 0.045, ×0.94 every 2 epochs. Gradient clipping at 2.0. Evaluate on an EMA of parameters. ~50 GPU replicas, batch 32, ~100 epochs. Total loss = label-smoothed CE on the main head + 0.4 × label-smoothed CE on the auxiliary head.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):                   # 35x35: reduced 5x5 + double 3x3
    def __init__(self, in_ch, pool_features):
        super().__init__()
        self.b1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b5_1 = ConvBN(in_ch, 48, kernel_size=1)
        self.b5_2 = ConvBN(48, 64, kernel_size=5, padding=2)
        self.b3_1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b3_2 = ConvBN(64, 96, kernel_size=3, padding=1)
        self.b3_3 = ConvBN(96, 96, kernel_size=3, padding=1)
        self.bp = ConvBN(in_ch, pool_features, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b5 = self.b5_2(self.b5_1(x))
        b3 = self.b3_3(self.b3_2(self.b3_1(x)))
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b5, b3, bp], 1)


class GridReductionB(nn.Module):               # 35x35 -> 17x17
    def __init__(self, in_ch):
        super().__init__()
        self.b3 = ConvBN(in_ch, 384, kernel_size=3, stride=2)
        self.b3d_1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b3d_2 = ConvBN(64, 96, kernel_size=3, padding=1)
        self.b3d_3 = ConvBN(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        b3 = self.b3(x)
        b3d = self.b3d_3(self.b3d_2(self.b3d_1(x)))
        bp = F.max_pool2d(x, 3, stride=2)
        return torch.cat([b3, b3d, bp], 1)


class InceptionC(nn.Module):                   # 17x17: 1x7 + 7x1
    def __init__(self, in_ch, c7):
        super().__init__()
        self.b1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b7_1 = ConvBN(in_ch, c7, kernel_size=1)
        self.b7_2 = ConvBN(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.b7_3 = ConvBN(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.b7d_1 = ConvBN(in_ch, c7, kernel_size=1)
        self.b7d_2 = ConvBN(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.b7d_3 = ConvBN(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.b7d_4 = ConvBN(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.b7d_5 = ConvBN(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.bp = ConvBN(in_ch, 192, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b7 = self.b7_3(self.b7_2(self.b7_1(x)))
        b7d = self.b7d_5(self.b7d_4(self.b7d_3(self.b7d_2(self.b7d_1(x)))))
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b7, b7d, bp], 1)


class GridReductionD(nn.Module):               # 17x17 -> 8x8
    def __init__(self, in_ch):
        super().__init__()
        self.b3_1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b3_2 = ConvBN(192, 320, kernel_size=3, stride=2)
        self.b7_1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b7_2 = ConvBN(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.b7_3 = ConvBN(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.b7_4 = ConvBN(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        b3 = self.b3_2(self.b3_1(x))
        b7 = self.b7_4(self.b7_3(self.b7_2(self.b7_1(x))))
        bp = F.max_pool2d(x, 3, stride=2)
        return torch.cat([b3, b7, bp], 1)


class InceptionE(nn.Module):                   # 8x8: expanded parallel 1x3 / 3x1
    def __init__(self, in_ch):
        super().__init__()
        self.b1 = ConvBN(in_ch, 320, kernel_size=1)
        self.b3_1 = ConvBN(in_ch, 384, kernel_size=1)
        self.b3_2a = ConvBN(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.b3_2b = ConvBN(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.b3d_1 = ConvBN(in_ch, 448, kernel_size=1)
        self.b3d_2 = ConvBN(448, 384, kernel_size=3, padding=1)
        self.b3d_3a = ConvBN(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.b3d_3b = ConvBN(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.bp = ConvBN(in_ch, 192, kernel_size=1)

    def forward(self, x):
        b1 = self.b1(x)
        b3 = self.b3_1(x); b3 = torch.cat([self.b3_2a(b3), self.b3_2b(b3)], 1)
        b3d = self.b3d_2(self.b3d_1(x)); b3d = torch.cat([self.b3d_3a(b3d), self.b3d_3b(b3d)], 1)
        bp = self.bp(F.avg_pool2d(x, 3, stride=1, padding=1))
        return torch.cat([b1, b3, b3d, bp], 1)


class AuxHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv0 = ConvBN(in_ch, 128, kernel_size=1)
        self.conv1 = ConvBN(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv1(self.conv0(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.fc(torch.flatten(x, 1))


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.Conv2d_1a = ConvBN(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a = ConvBN(32, 32, kernel_size=3)
        self.Conv2d_2b = ConvBN(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b = ConvBN(64, 80, kernel_size=1)
        self.Conv2d_4a = ConvBN(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = GridReductionB(288)
        self.Mixed_6b = InceptionC(768, c7=128)
        self.Mixed_6c = InceptionC(768, c7=160)
        self.Mixed_6d = InceptionC(768, c7=160)
        self.Mixed_6e = InceptionC(768, c7=192)
        self.aux = AuxHead(768, num_classes) if aux_logits else None
        self.Mixed_7a = GridReductionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)       # keep probability 0.8
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.Conv2d_1a(x); x = self.Conv2d_2a(x); x = self.Conv2d_2b(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.Conv2d_3b(x); x = self.Conv2d_4a(x)
        x = F.max_pool2d(x, 3, stride=2)
        x = self.Mixed_5b(x); x = self.Mixed_5c(x); x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x); x = self.Mixed_6c(x); x = self.Mixed_6d(x); x = self.Mixed_6e(x)
        aux = self.aux(x) if (self.aux is not None and self.training) else None
        x = self.Mixed_7a(x); x = self.Mixed_7b(x); x = self.Mixed_7c(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.dropout(x)
        return self.fc(x), aux


def label_smoothed_ce(logits, target, num_classes, eps=0.1):
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
    smooth = -logp.mean(dim=1)
    return ((1 - eps) * nll + eps * smooth).mean()


model = InceptionV3(num_classes=1000, aux_logits=True)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, alpha=0.9, eps=1.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
# total loss = label_smoothed_ce(main) + 0.4 * label_smoothed_ce(aux); clip grads at 2.0; EMA eval.
```
