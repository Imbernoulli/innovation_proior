Nearly all of the convolutional-network work around me chases a single number — ImageNet accuracy — and lets parameter count be whatever happens to fall out. That is leaving a lot on the table, because for any given accuracy there are many architectures that reach it and they differ wildly in size. Parameter count, specifically, is the axis I want to exploit, and not because of FLOPs: the gradient communication in distributed data-parallel training scales with the number of parameters, so fewer parameters means faster training; shipping a model over the air to a deployed device (a vision model pushed to a car) means moving the parameters, and a 240MB AlexNet is a painful update; and FPGAs and similar accelerators often have under 10MB of on-chip memory and no off-chip storage, so a model small enough to live entirely on-chip can stream video through it with zero external-memory traffic. The obvious competing approach is model compression — SVD on the weight matrices, pruning sub-threshold weights to a sparse matrix and retraining, quantizing to a few bits with a codebook — and these reach 9x to 35x on AlexNet. But they all *start* from the heavy architecture; I want to attack the problem from the other end and design something small from scratch. The two aren't rivals — a small dense model can still be compressed afterward, so they stack. The target is therefore sharp: match AlexNet-level accuracy (57.2% top-1 / 80.3% top-5 on ImageNet) with an order of magnitude or two fewer parameters, in a principled way that explains *which* architectural choices drive size rather than relying on black-box search.

I propose SqueezeNet. The whole design follows from writing down where parameters live in a convolution. A layer of 3×3 filters has parameter count $(\text{input channels}) \times (\text{number of filters}) \times (3 \times 3)$ — three multiplicative factors — and I attack all three. The first factor is the $3\times 3 = 9$ spatial footprint: a $1\times1$ filter has exactly $9\times$ fewer parameters and still does useful work, mixing channels at each pixel in the Network-in-Network sense, so I replace $3\times3$ filters with $1\times1$ filters wherever I can. The one thing a $1\times1$ filter cannot do is capture spatial structure — it has no receptive field beyond a single pixel — so I keep just enough $3\times3$ filters to see space, treating "how few" as a knob to sweep rather than a number to assume. The third factor is the input-channel count feeding the $3\times3$ filters: even after minimizing the *number* of $3\times3$ filters, each one still costs $9 \times (\text{input channels})$, so a $3\times3$ filter fed a fat 512-channel input is expensive no matter what. I thin that input cheaply with a $1\times1$ convolution — squeeze the incoming channels down to a small number, then let the $3\times3$ filters operate on the thinner map. The squeeze itself is cheap ($1\times1$ filters) and it shrinks the dominant input-channel factor of the expensive layer. Those two strategies both *cut* parameters, which costs accuracy somewhere, so I add a third lever that *buys accuracy back* for free: the spatial size of each layer's output is set by where the stride-2 downsampling sits, and delaying downsampling so most layers keep large activation maps empirically raises accuracy at equal parameters — and it touches zero parameters.

The first two strategies fold into a single reusable building block I call the Fire module, so the whole network is describable by a few numbers the way Inception modules describe GoogLeNet. A Fire module begins with a *squeeze* convolution layer — $1\times1$ filters only — that reduces the incoming channels to a small count $s_{1\times1}$. That squeezed map then feeds an *expand* layer running two banks in parallel: a $1\times1$ bank of $e_{1\times1}$ filters and a $3\times3$ bank of $e_{3\times3}$ filters, whose outputs are concatenated along the channel axis to give $e_{1\times1} + e_{3\times3}$ output channels. The squeeze feeding both banks is the lever: each $3\times3$ filter now costs $9 \times s_{1\times1}$, and $s_{1\times1}$ is small. The constraint that makes this genuinely save parameters is $s_{1\times1} < e_{1\times1} + e_{3\times3}$ — if the squeeze were as wide as the output it would not be limiting the $3\times3$ input at all and Strategy 2 would be doing nothing. Two implementation details matter. A conv framework does not natively give one layer two kernel sizes, so I implement the expand as two separate conv layers and concatenate them on the channel dimension, which is numerically identical to a single mixed-kernel layer; and the $3\times3$ bank gets 1 pixel of zero-padding so its output height and width match the $1\times1$ bank's. ReLU is applied to the squeeze activations and to each expand bank. The arithmetic confirms the design does what I claimed: for fire2, with 96 input channels, $s_{1\times1} = 16$, $e_{1\times1} = e_{3\times3} = 64$, the squeeze costs $96 \times 16 = 1{,}536$, the expand-$1\times1$ costs $16 \times 64 = 1{,}024$, and the expand-$3\times3$ costs $16 \times 64 \times 9 = 9{,}216$, totaling about 11.8k for a 128-channel output. Feeding 96 channels straight into 64 $3\times3$ filters with no squeeze would cost $96 \times 64 \times 9 = 55{,}296$ for the $3\times3$ bank alone — roughly $5\times$ more — and the gap widens in later modules where the input is 256 or 512 channels, which is exactly the heavy lifting the squeeze is designed for.

So that the whole architecture collapses to a handful of metaparameters rather than 24 hand-set numbers, I grow the per-module dimensions on a schedule. Let $\text{base\_e}$ be the expand width of the first module and bump it by $\text{incr\_e}$ every $\text{freq}$ modules: module $i$ gets

$$e_i = \text{base\_e} + \text{incr\_e} \times \left\lfloor \frac{i}{\text{freq}} \right\rfloor,$$

then $e_{i,3\times3} = e_i \times \text{pct\_3x3}$, $e_{i,1\times1} = e_i \times (1 - \text{pct\_3x3})$, and $s_{i,1\times1} = SR \times e_i$. The entire network is then five numbers: $\text{base\_e} = 128$, $\text{incr\_e} = 128$, $\text{freq} = 2$, $\text{pct\_3x3} = 0.5$ (half $1\times1$, half $3\times3$ in each expand, a balanced default), and a squeeze ratio $SR = 0.125$. That aggressively low squeeze ratio — each squeeze has $8\times$ fewer channels than its accompanying expand — is where the name comes from, and it is the most parameter-frugal point of a sweep that would otherwise relax the bottleneck if accuracy fell. These yield the per-module $(s_{1\times1}, e_{1\times1}, e_{3\times3})$ triples (16, 64, 64), (32, 128, 128), (48, 192, 192), (64, 256, 256) across fire2–fire9.

The macroarchitecture applies Strategy 3 around that stack. It opens with one standalone convolution layer (conv1) — the input has only 3 channels, nothing to squeeze yet, so a plain $7\times7$ stride-2 stem makes sense — then eight Fire modules (fire2–fire9) with expand width stepping up on the schedule, and ends with a final convolution (conv10) mapping to the 1000 class channels. Stride-2 max-pooling sits only after conv1, after fire4, and after fire8, with a global pool after conv10, so pooling is sparse and pushed late and most Fire modules operate on large activation maps. Crucially there are *no fully-connected layers anywhere*: following the NiN lesson that the FC classifier is where AlexNet and VGG hide most of their weights, the head is a $1\times1$ conv to $\text{num\_classes}$ followed by global average pooling, which deletes the field's single biggest parameter sink. A 50% dropout goes after fire9, just before conv10. The input channels of each Fire module equal the previous module's $e_{1\times1} + e_{3\times3}$ output, so the channel schedule runs 96 → 128 → 128 → 256 → 256 → 384 → 384 → 512 → 512. The whole network lands at about 1.2M parameters — roughly $50\times$ fewer than AlexNet's ~60M — at 4.8MB in 32-bit, at AlexNet-level top-5 accuracy. Because it is dense and small it is itself still compressible, and the two approaches stack: Deep Compression at 8-bit quantization with 33% sparsity reaches 0.66MB ($363\times$ smaller than 32-bit AlexNet) and at 6-bit with 33% sparsity reaches 0.47MB ($510\times$ smaller), both at equivalent accuracy. There is also a near-free accuracy lever at the macro level. Because the squeeze layers are a deliberate $8\times$ bottleneck, a bypass routing information *around* a Fire module can help: a simple bypass is just an identity shortcut that elementwise-adds the input to the output, adding zero parameters but requiring matched input and output channel counts — which holds only for the modules where the expand width did not just step up, so bypasses go around fire3, fire5, fire7, fire9 — while a complex bypass puts a $1\times1$ conv on the shortcut to fix mismatched channels at some parameter cost. The default stays bypass-free for minimal size, with simple bypass as a candidate upgrade.

Trained with SGD in Caffe, starting at learning rate 0.04 with linear decay, the implementation mirrors how it goes together in a conv framework.

```python
import torch
import torch.nn as nn


class FeatureModule(nn.Module):
    """Squeeze then expand. The squeeze 1x1 thins the channels so the 3x3
    expand bank is cheap (Strategy 2); the expand is a mix of 1x1 and 3x3
    filters (Strategy 1). Output = channel-wise concat of the two banks."""
    def __init__(self, in_channels, squeeze, expand1x1, expand3x3):
        super().__init__()
        # Squeeze: 1x1 conv reducing channels to `squeeze` (< expand1x1 + expand3x3).
        self.squeeze = nn.Conv2d(in_channels, squeeze, kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        # Expand bank A: 1x1 filters (9x cheaper than 3x3).
        self.expand1x1 = nn.Conv2d(squeeze, expand1x1, kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        # Expand bank B: 3x3 filters, padded 1px so output H,W match the 1x1 bank.
        self.expand3x3 = nn.Conv2d(squeeze, expand3x3, kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        a = self.expand1x1_act(self.expand1x1(x))
        b = self.expand3x3_act(self.expand3x3(x))
        return torch.cat([a, b], dim=1)   # concatenate on the channel axis


class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem: standalone conv (3-channel input, nothing to squeeze yet),
        # 7x7 stride 2. Pooling is sparse and pushed late (Strategy 3): after
        # conv1, fire4, fire8 -> most Fire modules see large activation maps.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.features = nn.Sequential(
            FeatureModule(96, 16, 64, 64),      # fire2  -> 128 out
            FeatureModule(128, 16, 64, 64),     # fire3  -> 128
            FeatureModule(128, 32, 128, 128),   # fire4  -> 256
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FeatureModule(256, 32, 128, 128),   # fire5  -> 256
            FeatureModule(256, 48, 192, 192),   # fire6  -> 384
            FeatureModule(384, 48, 192, 192),   # fire7  -> 384
            FeatureModule(384, 64, 256, 256),   # fire8  -> 512
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FeatureModule(512, 64, 256, 256),   # fire9  -> 512
        )
        # No fully-connected layers (the NiN lesson): a 1x1 conv classifier
        # (conv10) plus global average pooling replaces AlexNet's huge FC head.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                              # dropout after fire9
            nn.Conv2d(512, num_classes, kernel_size=1),     # conv10: 1x1 -> classes
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                   # global average pool
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)                          # logits: (N, num_classes)


if __name__ == "__main__":
    net = Net(num_classes=1000)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params:,}")          # ~1.2M, ~50x fewer than AlexNet
    y = net(torch.randn(1, 3, 224, 224))
    print(y.shape)                              # torch.Size([1, 1000])
```
