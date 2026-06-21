The unit everything in these convolutional networks is built from is the convolution, and writing out what it computes exposes the limitation I want to fix. The $c$-th output map is $u_c = v_c * X = \sum_{s} v_c^{\,s} * x^s$: each input channel $x^s$ is convolved with its own small 2D kernel and the results are summed across all input channels. The "channel relationship" — how much output channel $c$ listens to input channel $s$ — is exactly the kernel $v_c^{\,s}$, but it is two things tangled together: the same weights that decide *which* channels matter also decide *what local spatial pattern* to look for. Two consequences follow. The mixing is *local* — $v_c^{\,s}$ is a $3\times3$ window, so away from the very top of the network each unit decides how to combine channels while looking through a keyhole, with no image-wide view to tell it "this is a dog, so the fur-texture channels matter and the sky-blue ones do not." And it is *static and instance-agnostic* — once trained, $v_c$ is fixed, so every image gets the same channel mixing, with no knob to turn channel 37 up and channel 112 down for *this particular* input.

The field's effort has mostly gone the other way. Depth (VGG, Inception), batch normalization, and residual shortcuts strengthen optimisation and the spatial encoding; grouped convolutions, $1\times1$ convolutions, and depthwise-separable factorisations reshape the channel axis but only ever as instance-agnostic functions over local receptive fields, usually to *save* computation. Spatial-transformer and trunk-and-mask attention learn *where* to look, but they target the spatial axis and the mask branches add substantial parameters and compute. Highway networks show that data-dependent multiplicative gates are useful, but they gate flow along the *depth* dimension rather than reweighting channels with global context. The gap nobody fills is exactly an explicit, dynamic, global handle on *which channels* matter for the current image. A solution must (a) see beyond the local receptive field — global, image-wide; (b) be conditioned on the input so it adapts per image; (c) be cheap enough to drop into a strong backbone without re-tuning the architecture; and (d) modulate the features the backbone already computes rather than replace them.

I propose the Squeeze-and-Excitation block, and the network built from it is SENet. The block sits on top of the feature maps $U \in \mathbb{R}^{H\times W\times C}$ produced by any transformation $F_{tr}$ — a conv, an Inception module, or a residual branch — and recalibrates them in three operations: squeeze, excite, scale. The squeeze solves the global-context sub-problem. The trouble with a channel map $u_c$ is that it is spatial and has only ever been touched by local filters, so any single location carries only local information; I do not want a location, I want one number per channel that summarises the channel over the *whole* spatial extent. The simplest collapse that uses every location equally is the mean, so
$$z_c = \frac{1}{H\cdot W}\sum_{i=1}^{H}\sum_{j=1}^{W} u_c(i,j), \qquad z \in \mathbb{R}^C.$$
This is global average pooling, and it earns its keep because that scalar's receptive field is the entire feature map: it is a channel descriptor with, by construction, the global context the local convolutions could never supply mid-network, and it costs no parameters. Averaging discards spatial layout, but that is fine — the question is "to what extent is this channel active across the image," and the mean is a clean, smooth, differentiable answer. Max pooling would track only the single strongest location rather than overall presence, so the mean is the principled default for a global-presence summary.

The excitation is the interesting part: turn the descriptor $z$ into per-channel modulation weights $s \in \mathbb{R}^C$. I let the requirements force the form. The map must be *learned*, because "which channels matter given this global summary" is precisely the relationship the network should discover; it must be *nonlinear*, since a linear $z\mapsto Wz$ composed with the linear pooling and linear scaling would collapse back to a fixed-ish reweighting and could not capture interactions like "channel A matters only when channel B is active"; and it must couple *across* channels, so $s_c$ may depend on all of $z$. The simplest object that is learned, nonlinear, and fully cross-coupled is a small MLP on $z$ — a fully-connected layer, a nonlinearity, a fully-connected layer back. The output nonlinearity is decided first because it interacts with how $s$ is used: $s$ will *scale* channels, which wants a bounded, well-behaved range, so I squash each $s_c$ into $(0,1)$. The tempting choice is softmax, the canonical attention nonlinearity, but softmax makes the gates *compete* — they sum to one, so emphasising one channel necessarily suppresses the rest, pushing toward pick-one-channel behaviour. That is exactly wrong: an image is full of many simultaneously-useful channels — edges *and* texture *and* colour — and I want several turned up at once. So softmax is out and a per-channel logistic sigmoid is in: $s_c = \sigma(g(z)_c)$, each gate independently in $(0,1)$, non-competing. (A tanh would let gates flip channel signs, breaking the clean attenuate-or-keep semantics; an output ReLU would be unbounded above and zero below, a recipe for amplifying channels arbitrarily and killing others — both worse, so the bounded smooth sigmoid is not incidental.) The inner nonlinearity is ReLU, the cheap default that supplies the needed nonlinearity without saturating on the positive side.

The hidden width is where cost enters. The richest map would be a full $C\times C$ matrix, $C^2$ parameters per block; in ResNet-50 the later stages have $C$ in the thousands across many blocks, so $C^2$ per block summed over the network would balloon the model and overfit, defeating the lightweight goal. So I put a bottleneck in the middle — reduce $C \to C/r$, ReLU, expand $C/r \to C$ — with reduction ratio $r$:
$$s = \sigma\big(W_2\,\delta(W_1 z)\big), \qquad \delta = \mathrm{ReLU}, \quad W_1 \in \mathbb{R}^{(C/r)\times C}, \quad W_2 \in \mathbb{R}^{C\times(C/r)}.$$
Dropping the FC biases (they hinder the channel modelling and are negligible in count), the per-block parameter cost is
$$\frac{C}{r}\cdot C + C\cdot\frac{C}{r} = \frac{2C^2}{r},$$
and over stages $s$ of $N_s$ blocks at width $C_s$ the network's extra parameters are $\frac{2}{r}\sum_s N_s C_s^2$. So $r$ is a direct dial on cost, cutting the quadratic-in-$C$ term by a factor $r$ while the ReLU keeps the map nonlinear and the shared $C/r$ code still lets every output channel depend on every input channel. Too large an $r$ starves the bottleneck of capacity; too small returns to the $C\times C$ blow-up and overfitting, so performance is not monotone in $1/r$ and there is a sweet spot — $r=16$ is the right order of magnitude. For SE-ResNet-50 this is about $2.5$M extra on $\sim25$M ($\sim10\%$), with FLOPs rising only from $\sim3.86$ to $\sim3.87$ GFLOPs ($\sim0.26\%$) because the FC layers act on length-$C$ vectors, not on $H\times W$ maps; most of the mass is in the final (widest) stage, and dropping SE there alone cuts the increase to $\sim4\%$ at negligible accuracy cost.

The third operation uses $s$. Respecting "modulate, don't replace," I scale each channel's whole map by its gate,
$$\tilde{x}_c = s_c \cdot u_c,$$
a channel-wise multiplication of the scalar $s_c \in (0,1)$ into $u_c$. Because the gate cannot exceed one, the block can only attenuate a channel toward zero or keep it near full strength — emphasising the channels the global summary deemed informative and suppressing the rest. The whole pipeline $z \to s \to$ scale is an input-dependent reweighting computed from global context: a self-attention over channels whose reach is the entire image, answering (a) global, (b) input-dependent, (c) cheap, (d) modulatory in turn. One check pins the benefit on the squeeze rather than the added parameters: a counterfactual that keeps both FC layers but skips the pooling — realising them as $1\times1$ convolutions that preserve $H\times W$ — computes each gate only from that location's channel vector, a purely local operator again; if that variant does worse at the same parameter budget, the gap is exactly the value of the global average.

Placement matters for residual backbones, the case I care about most. A residual block computes $y = x + F(x)$ with $F$ the non-identity branch (for deep nets the $1\times1$-reduce $\to 3\times3 \to 1\times1$-expand bottleneck with BN+ReLU between). I take $F_{tr}$ to be that branch, squeeze and excite its output, scale it, and *then* add the identity — so recalibration happens on the residual contribution *before* the summation. The reason is that the identity branch is the highway that carries the input through and makes deep training work; I do not want $(0,1)$ gates multiplying *that* and choking the gradient on the skip path. It is $F(x)$ whose channels I want to weigh, and putting the gate after the addition would scale the merged signal including the identity, attenuating the carried-through input and sitting on the path the next block's gradient must traverse. Applying it before the branches aggregate is the safe default. For a basic block the SE acts on the block's width; for a bottleneck it acts on the expanded width $\text{planes}\times4$, the residual-branch output. Stacking these blocks everywhere makes the recalibrations compound through depth.

```python
import torch
import torch.nn as nn


class SELayer(nn.Module):
    """Squeeze-and-Excitation: recalibrate channels using global context."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)               # squeeze: GAP -> z in R^C
        self.fc = nn.Sequential(                              # excitation
            nn.Linear(channel, channel // reduction, bias=False),  # C -> C/r
            nn.ReLU(inplace=True),                                 # delta
            nn.Linear(channel // reduction, channel, bias=False),  # C/r -> C
            nn.Sigmoid(),                                          # sigma: gates in (0,1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)        # squeeze
        s = self.fc(z).view(b, c, 1, 1)        # excitation
        return x * s.expand_as(x)              # scale


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                     # recalibrate residual branch
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual                        # before the identity add
        return self.relu(out)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)   # on expanded width
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)
```
