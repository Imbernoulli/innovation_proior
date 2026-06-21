I propose the canonical method name MobileNetV1, also referred to as MobileNet, the efficient convolutional network introduced by Howard et al. in "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." The constraint of on-device deployment decides the architecture, so I start there. Image recognition is moving off the server and onto phones, embedded cameras, robots, and augmented-reality headsets, where a model has to run quickly under tight memory, power, and latency budgets and often without a network round trip. The usual path to accuracy — deeper, wider, more branchy convolutional stacks like VGG and Inception — produces models with billions of multiply-adds per image and tens or hundreds of millions of weights, and a high ImageNet number is useless on-device if a single forward pass misses the latency target. Post-hoc compression such as pruning, quantization, hashing, product quantization, low-rank tensor expansion, and distillation can accelerate one large trained model but does not give a clean family of architectures that can be trained from scratch at different budgets. Aggressive factorization in the other direction, flattened networks that replace a 3-D filter with a sequence of one-dimensional filters, can speed feedforward inference but imposes a strong rank-style structure that risks too much representational loss. And optimizing parameter count alone, the SqueezeNet route, misses the point: a model can be tiny in parameters and still slow, because latency tracks multiply-adds and whether the remaining operations map to efficient dense kernels, not file size. What I need is a structured reduction in arithmetic that leaves behind operations a real library can execute as dense kernels, plus a predictable knob for trading accuracy against budget.

The leverage is in a single convolutional layer, so I look there first. A standard square convolution maps an input of spatial size $D_F \times D_F$ with $M$ channels to an output of the same spatial size with $N$ channels through a kernel of shape $D_K \times D_K \times M \times N$, $G_{k,l,n} = \sum_{i,j,m} K_{i,j,m,n}\, F_{k+i-1,\,l+j-1,\,m}$, at a multiply-add cost of $D_K^2\, M\, N\, D_F^2$. What this expression exposes is multiplicative coupling: the spatial support $D_K^2$, the input depth $M$, the output depth $N$, and the feature-map area $D_F^2$ all multiply one another. The reason is that one convolution is quietly doing two jobs at once — it filters local spatial neighborhoods and it mixes input channels into new output features — and because those jobs are fused into a single weight tensor, the spatial factor $D_K^2$ and the output-channel factor $N$ multiply. MobileNetV1 is built on splitting those two jobs apart into a depthwise separable convolution and then scaling the result with two budget multipliers.

The split goes in two steps. First I keep only spatial filtering, applying one $D_K \times D_K$ filter to each input channel independently, $\hat{G}_{k,l,m} = \sum_{i,j} \hat{K}_{i,j,m}\, F_{k+i-1,\,l+j-1,\,m}$, at cost $D_K^2\, M\, D_F^2$. This depthwise step drops the factor $N$ entirely, but it can no longer create new mixtures of channels — each output channel still corresponds to exactly one input channel — so on its own it cannot replace a real convolution. The second step restores channel mixing with no spatial footprint: a $1 \times 1$ convolution, which at each spatial location is just an $M \times N$ matrix, at cost $M\, N\, D_F^2$. Together they cost $D_K^2\, M\, D_F^2 + M\, N\, D_F^2$, and dividing by the standard cost gives the load-bearing cancellation, $(D_K^2\, M\, D_F^2 + M\, N\, D_F^2) / (D_K^2\, M\, N\, D_F^2) = 1/N + 1/D_K^2$. There is no hidden constant and no sign ambiguity in that ratio. The first term is the price of keeping per-channel spatial filtering, the second the price of keeping channel mixing. With $3 \times 3$ kernels, $1/D_K^2 = 1/9$, and once $N \ge 64$ the term $1/N$ is small beside $1/9$, so the layer runs at roughly one eighth to one ninth of a full convolution. On a representative internal layer with $D_K = 3$, $M = N = 512$, $D_F = 14$, the full convolution costs $3\cdot 3\cdot 512\cdot 512\cdot 14\cdot 14 = 462.4\text{M}$ multiply-adds with $2.36\text{M}$ weights, while the split costs $3\cdot 3\cdot 512\cdot 14\cdot 14 + 512\cdot 512\cdot 14\cdot 14 = 52.3\text{M}$ multiply-adds with $0.267\text{M}$ weights, matching $1/512 + 1/9$ exactly.

The split also dictates where the work now lives, and that is what makes it deployable rather than merely cheap. The depthwise-to-pointwise cost ratio is $D_K^2 : N$, so with $3 \times 3$ filters and hundreds of channels almost all the remaining compute is in the $1 \times 1$ convolution, which is precisely a dense matrix multiply over the channel dimension at each spatial position. It needs no im2col lowering the way a general spatial convolution does; the saved arithmetic is not bought with irregular or sparse computation, so the dominant operation is a clean GEMM-like channel mixer that maps straight onto fast dense kernels. This is exactly why I stop the factorization here and do not push the spatial part down to one-dimensional pieces: once the depthwise term is already a small fraction of the total, making the spatial filter rank-one can only attack a sliver of the remaining cost while risking a larger representational loss. Keeping full two-dimensional per-channel filters and a full dense pointwise mixer is the better compromise.

The repeating block is therefore fixed: a depthwise $3 \times 3$ convolution, normalization, activation, then a pointwise $1 \times 1$ convolution, normalization, activation. The first layer is the one special case, because with only three RGB input channels a depthwise stem would have just three spatial filters before any expansion; I keep the stem as a full $3 \times 3$ convolution from 3 to 32 channels at stride 2 and use the split block everywhere after. The whole network follows the usual pyramid logic — as spatial resolution shrinks, channel count grows. The stem takes $224 \times 224$ to $112 \times 112$, and strided depthwise steps then walk $112 \to 56 \to 28 \to 14 \to 7$ while the channels run $32 \to 64 \to 128 \to 128 \to 256 \to 256 \to 512$, then five more $512 \to 512$ blocks, then $512 \to 1024$, then $1024 \to 1024$; stride is placed in the depthwise convolution, and the final $1024$ block stays at stride 1 to match the retained $7 \times 7$ map. Global average pooling collapses the spatial map to $1 \times 1$, and a linear classifier emits logits. Counting the stem, every depthwise and pointwise convolution separately, and the classifier gives 28 layers — a branch-free family of thirteen depthwise-separable blocks. I deliberately choose thin over shallow as the way to shrink the model: deleting whole nonlinear stages gives up representational depth, whereas uniformly thinning keeps the full sequence of transformations and attacks the dominant pointwise terms, so at equal compute the deep-thin version retains accuracy better.

That thinning is the first budget knob, the width multiplier $\alpha$. Scaling every channel count by $\alpha$ turns the block cost $D_K^2\, M\, D_F^2 + M\, N\, D_F^2$ into $D_K^2\, \alpha M\, D_F^2 + \alpha M\, \alpha N\, D_F^2$, where the depthwise term is only linear in $\alpha$ while the pointwise term is quadratic. The total is not purely $\alpha^2$, but since the pointwise term dominates, compute and parameters fall roughly as $\alpha^2$. Settings of $1.0$, $0.75$, $0.5$, and $0.25$ define separate thinner networks trained from scratch, not pruned copies; the single-layer check is consistent, since at $\alpha = 0.75$ the representative layer drops from $52.3\text{M}$ to $29.6\text{M}$ multiply-adds and from $0.267\text{M}$ to $0.151\text{M}$ weights. The second knob is the resolution multiplier $\rho$: scaling the input resolution makes every $D_F^2$ become $(\rho D_F)^2$, giving the combined cost $D_K^2\, \alpha M\, (\rho D_F)^2 + \alpha M\, \alpha N\, (\rho D_F)^2$. So $\rho$ scales compute by $\rho^2$ and leaves parameter count untouched, since weights do not depend on feature-map size — shrinking $14$ to $10$ ($\rho = 10/14 = 0.714$) takes the $\alpha = 0.75$ layer from $29.6\text{M}$ to $15.1\text{M}$ multiply-adds while the $0.151\text{M}$ weights stay fixed. Two independent, predictable controls over width and resolution come from one architecture.

A few training and implementation choices follow from the same capacity argument. These models have far less capacity than large Inception-like systems, and the depthwise filters carry very few parameters next to the pointwise ones, so heavy regularization aimed at overparameterized models becomes counterproductive: I use lighter augmentation, no auxiliary classifier heads, no label smoothing, and little or no weight decay on the depthwise filters. In a faithful build I also preserve the code-level details that change behavior — channel counts are floored as $\max(\lfloor d\,\alpha\rfloor, 8)$ so thinning never collapses a layer to zero, depthwise convolutions are expressed as per-channel grouped convolutions, and the reference activation is the clipped ReLU6 even though the architectural description often just says ReLU. The mathematical reason it works is the cancellation from $D_K^2 M N D_F^2$ to $D_K^2 M D_F^2 + M N D_F^2$; the engineering reason it deploys is that the remaining dominant computation is dense $1 \times 1$ channel mixing; and the practical reason it is useful is that $\alpha$ and $\rho$ give independent, predictable budget controls.

```python
import torch
import torch.nn as nn


def conv_bn_relu6(in_ch, out_ch, kernel, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = conv_bn_relu6(
            in_ch, in_ch, kernel=3, stride=stride, padding=1, groups=in_ch
        )
        self.pointwise = conv_bn_relu6(
            in_ch, out_ch, kernel=1, stride=1, padding=0
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNetV1(nn.Module):
    cfg = [
        (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
        (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
        (1024, 2), (1024, 1),
    ]

    def __init__(self, num_classes=1000, width_mult=1.0, min_depth=8,
                 dropout_keep_prob=0.999):
        super().__init__()

        def depth(ch):
            return max(int(ch * width_mult), min_depth)

        in_ch = depth(32)
        self.stem = conv_bn_relu6(3, in_ch, kernel=3, stride=2, padding=1)

        blocks = []
        for out_ch, stride in self.cfg:
            out_ch = depth(out_ch)
            blocks.append(DepthwiseSeparableBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=1.0 - dropout_keep_prob)
        self.logits = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":
    model = MobileNetV1(num_classes=1000, width_mult=1.0)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("output shape:", y.shape)
    params = sum(p.numel() for p in model.parameters())
    print("parameters:", params)
```
