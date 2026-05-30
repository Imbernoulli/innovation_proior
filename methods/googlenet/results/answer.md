# GoogLeNet / Inception

## Problem

Deep convolutional networks were winning ImageNet by getting bigger — deeper and wider. But "just make it bigger" has two costs that scale badly: more parameters invite overfitting (and labeled data for fine-grained categories is expensive), and chaining wide convolutions makes compute grow *quadratically* in the number of filters, much of it wasted on weights that end up near zero. The goal: get the accuracy of a much larger, deeper, wider network while holding the inference budget near **1.5 billion multiply-adds**, so the model is deployable rather than an academic curiosity.

## Key idea

The principled fix to both problems is **sparse connectivity** — and theory (correlation-clustering of co-activating units, echoing the Hebbian "fire together, wire together") says a good sparse topology can be built layer by layer. But real hardware is terrible at non-uniform sparse arithmetic; tuned dense matrix multiply is far faster even at 100× the FLOPs. The sparse-matrix literature offers the escape: a sparse matrix clustered into **dense submatrices** runs fast. So **approximate the optimal local sparse structure with dense building blocks**.

Concretely, at each location let the network cover correlated-unit clusters at several scales *in parallel* — a 1×1, a 3×3, a 5×5 convolution, plus a 3×3 max-pool path — and concatenate their outputs along channels. That is the **Inception module**. The naive version blows up in cost (a 5×5 over many channels is enormous, and the pool path keeps growing the channel count), so insert **1×1 convolutions as dimension reducers** *before* the 3×3 and 5×5 convs and *after* the pool, compressing channels exactly where signals must be aggregated en masse. The 1×1s are cheap, carry their own ReLU, and cut the expensive convs' cost by an order of magnitude.

Two more pieces make a 22-layer net of these trainable and lean:
- **Global average pooling** instead of a fully-connected head: average each final feature map into one number. Parameter-free, a structural regularizer, and it removes the bulk of the parameters a big FC head would add.
- **Auxiliary classifiers** on two intermediate modules during training: small side heads whose discounted (×0.3) loss is added to the total, injecting gradient deep in the net, encouraging discriminative mid-level features, and regularizing. Discarded at inference.

## Final architecture (GoogLeNet incarnation)

Stem (plain convs + pooling + local response normalization), then 9 Inception modules in three stages (3a–3b / 4a–4e / 5a–5b) with stride-2 max-pools between stages, then GAP → dropout(0.4) → one 1000-way linear + softmax. Two auxiliary heads hang off modules 4a and 4d during training. The inference network has about 7M parameters, roughly 12× fewer than AlexNet.

Each Inception module has four branches concatenated on the channel axis:
1. `1×1 conv`
2. `1×1 conv (reduce) → 3×3 conv`
3. `1×1 conv (reduce) → 5×5 conv`
4. `3×3 max-pool (stride 1) → 1×1 conv (project)`

Per-module channel widths `(in, #1×1, #3×3reduce, #3×3, #5×5reduce, #5×5, poolproj)`:

```
3a (192,  64,  96, 128, 16,  32,  32)   -> 256
3b (256, 128, 128, 192, 32,  96,  64)   -> 480
4a (480, 192,  96, 208, 16,  48,  64)   -> 512
4b (512, 160, 112, 224, 24,  64,  64)   -> 512
4c (512, 128, 128, 256, 24,  64,  64)   -> 512
4d (512, 112, 144, 288, 32,  64,  64)   -> 528
4e (528, 256, 160, 320, 32, 128, 128)   -> 832
5a (832, 256, 160, 320, 32, 128, 128)   -> 832
5b (832, 384, 192, 384, 48, 128, 128)   -> 1024
```

Training loss: `L = L_main + 0.3·L_aux1 + 0.3·L_aux2`, each a softmax cross-entropy over the same 1000 classes.

## Code

True 5×5 branches, LRN in the stem, and 3×3/stride-2 stage pools.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """Conv -> ReLU. Every conv in the network (including the 1x1 reducers
    and projections) is followed by a rectified linear activation."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class Inception(nn.Module):
    """One Inception module: four parallel branches covering several scales,
    concatenated on the channel axis. The 1x1 convs in branches 2/3 and after
    the pool are the dimension reducers that keep the cost bounded."""
    def __init__(self, in_ch, ch1, ch3red, ch3, ch5red, ch5, poolproj):
        super().__init__()
        # branch 1: a single 1x1 conv (the tightest, most local clusters)
        self.branch1 = BasicConv2d(in_ch, ch1, kernel_size=1)
        # branch 2: 1x1 reduce -> 3x3 (cut channels before the bigger kernel)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, ch3red, kernel_size=1),
            BasicConv2d(ch3red, ch3, kernel_size=3, padding=1),
        )
        # branch 3: 1x1 reduce -> 5x5 (the expensive kernel, run at low width)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_ch, ch5red, kernel_size=1),
            BasicConv2d(ch5red, ch5, kernel_size=5, padding=2),
        )
        # branch 4: 3x3 max-pool -> 1x1 project (cap the pool's channel count)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_ch, poolproj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class InceptionAux(nn.Module):
    """Auxiliary classifier hung off an intermediate module during training.
    Adds gradient signal deep in the net and acts as a regularizer; discarded
    at inference."""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_ch, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.avgpool(x)                    # avg pool 5x5/3 -> 4x4
        x = self.conv(x)                       # 1x1 reduce to 128
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # Stem: plain convs + pooling + local response normalization.
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Stage 3
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # Stage 4
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # Stage 5
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)   # off 4a
            self.aux2 = InceptionAux(528, num_classes)   # off 4d

        # Head: global average pooling -> dropout -> single linear.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrn1(self.maxpool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(self.lrn2(x))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if (self.aux_logits and self.training) else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if (self.aux_logits and self.training) else None
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux2, aux1
        return x


def total_loss(outputs, target, aux_weight=0.3):
    """L = L_main + 0.3 * (L_aux1 + L_aux2) during training."""
    if isinstance(outputs, tuple):
        main, aux2, aux1 = outputs
        return (F.cross_entropy(main, target)
                + aux_weight * F.cross_entropy(aux1, target)
                + aux_weight * F.cross_entropy(aux2, target))
    return F.cross_entropy(outputs, target)
```

### Why each piece is there
- **1×1 reducers before 3×3/5×5 and after the pool** — a naive 5×5 over a 28×28×192 map to 32 channels costs `28·28·192·32·5·5 ≈ 120M` MACs; reducing 192→16 first costs `28·28·192·16 ≈ 2.4M` plus `28·28·16·32·5·5 ≈ 10M`, about **10× less**, for the same output shape. This is what keeps the whole net inside ~1.5B MACs.
- **Parallel multi-scale branches concatenated** — cover correlated-unit clusters at several spatial extents at once; the next module abstracts across scales.
- **Global average pooling head** — parameter-free, removes the overfitting-prone FC head, acts as a structural regularizer; the single trailing linear exists only to retarget the 1024-vector to any label set.
- **Dropout 0.4 on the pooled vector** — regularizes the remaining pooled features and final readout after the large FC head has been removed.
- **Auxiliary heads (×0.3, training-only)** — counter the weak gradient through a plain 22-layer stack, push mid-level features to be discriminative, and add regularization.
- **LRN in the stem** — local response normalization is the available activation-stabilization primitive inherited from earlier large convnets.
