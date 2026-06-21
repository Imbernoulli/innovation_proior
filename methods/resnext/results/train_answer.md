We want an image-classification network that gains accuracy without paying for it in FLOPs or parameters, and the obstacle is that the two design philosophies we trust pull against each other. The VGG/ResNet way is to pick one building block and stack it — same shape over and over, doubling the width whenever the spatial map is halved. This collapses the hyper-parameter space to almost nothing, exposes depth as the single clean dimension to turn, and, because the rule is so simple, resists over-fitting the *architecture* to one dataset, which is exactly why such nets transfer everywhere. The Inception way instead hand-designs multi-branch modules that split the input into low-dimensional embeddings with $1\times1$ convolutions, transform each branch with its own filters, and merge by concatenation; this buys excellent accuracy at low theoretical complexity, but every branch's filter count and size is tailored by hand and the modules differ stage by stage, so a new dataset means redesigning from scratch — there is no clean knob to turn. The question I want to settle is whether one can keep the repeat-one-block simplicity *and* the split-transform-merge efficiency, in a form where the multi-branch structure is governed by a single turnable parameter, and do so under the hard constraint of *fixed complexity*: same FLOPs, same parameter count. Buying accuracy by going deeper or wider is easy and increasingly unrewarding — depth and width are giving diminishing returns — so what is interesting and rare is a *third* axis of the design space, distinct from depth and width, where the same compute budget buys more.

I propose ResNeXt, and its core is a single move that I trace back to the most elementary operation in the network. The simplest "neuron" — what a fully-connected or convolutional layer does per output — is an inner product $\sum_{i=1}^{D} w_i x_i$ over a $D$-channel input $x = [x_1, \dots, x_D]$. Read as a *process* rather than a formula, it splits $x$ into $D$ single-channel pieces $x_i$, transforms each (trivially, by a scalar multiply $w_i x_i$), and aggregates by summation. That is exactly the split-transform-merge shape of an Inception module, only with the splitting into single dimensions, the transforming a scalar scale, and the merging a plain sum. So split-transform-merge is not an exotic invention; it is the deep structure of the inner product itself. The generalization then writes itself: keep the split-and-sum skeleton but replace the elementary scalar transform $w_i x_i$ with a generic function $T_i(x)$ that is itself a small network — project $x$ into a low-dimensional embedding and transform it — and aggregate $C$ of them. The block's residual function becomes

$$F(x) = \sum_{i=1}^{C} T_i(x), \qquad y = x + \sum_{i=1}^{C} T_i(x),$$

added onto the identity (or, at stage transitions, projection) shortcut, so the residual conditioning that makes very deep nets trainable carries over untouched — I have only restructured what lives *inside* the residual function. The count $C$ sits in the same structural slot where the inner product's sum ran to $D$, but it is free: it need not equal the input channel count and can be any value. Width counts the number of *simple* transformations (the channels of an inner product); this new axis counts the number of *complex* ones. I call it **cardinality**.

The decision that makes cardinality a usable knob, and the place where I refuse to repeat Inception's mistake, is to give all the $T_i$ the *same topology*. Inception's un-transferability comes precisely from letting every branch differ; forcing one shared shape is just the VGG repeat-the-same-block philosophy applied at the sub-block level, and it makes the number of paths a single isolated factor I can dial to any value with no per-path design. For the topology of each $T_i$ I reuse the bottleneck I already trust: $1\times1\,(256\to d)\to 3\times3\,(d\to d)\to 1\times1\,(d\to 256)$, where the first $1\times1$ *is* the "project into a low-dimensional embedding" step. So one path is $1\times1 \to 3\times3 \to 1\times1$, and I lay down $C$ of them in parallel and sum.

Before trusting this, I check whether the block has equivalent forms, because the equivalences reveal what it really is. Take $C=2$: each path ends in a $1\times1$ layer with weight $A_i$, fed by the response $B_i$ from the second-to-last layer, so the summed contribution is $A_1 B_1 + A_2 B_2$. The algebraic identity $A_1 B_1 + A_2 B_2 = [A_1, A_2]\cdot[B_1; B_2]$ — horizontal concatenation of the last-layer weights times vertical concatenation (stacking) of the second-last responses — says that summing two separate last-$1\times1$ layers is identical to concatenating the two paths' outputs into one wider tensor and applying a single shared last $1\times1$. That is a second, *concatenation* form (it resembles Inception-ResNet, except all branches share a shape, so the cardinality knob survives). Pushing once more: in the concatenation form the first $1\times1$ of every path maps the same $x$ to a $d$-dimensional embedding, and $C$ independent linear maps on the same input *are* one wider $1\times1\,(256\to C\!\cdot\! d)$. The $C\!\cdot\! d$ channels must then be processed by $C$ separate $3\times3$ convolutions, each acting only on its own $d$-channel slice with no cross-slice mixing — which is exactly a **grouped convolution** with $C$ groups, followed by a final $1\times1\,(C\!\cdot\! d\to 256)$. This is the satisfying part: grouped convolution has been in every library since the two-GPU AlexNet, where it existed only as an engineering compromise to split a model across GPUs and was never used to *improve* accuracy; here it falls out as the most succinct implementation of "aggregate $C$ same-topology transformations." For this homogeneous bottleneck template the three forms — $C$ summed bottleneck paths, concatenate-then-restore, and grouped convolution — are *strictly* equivalent when batch norm and ReLU are placed consistently, so I can reason about the block as "summing $C$ transformations" and implement the grouped-convolution one because it is most compact and fastest. One caveat: these reformulations only yield a nontrivial topology when each path has depth $\ge 3$, like the bottleneck; a depth-2 basic block would collapse to a trivially wide dense layer. So the bottleneck depth is not just for efficiency — it is what makes cardinality a real, distinct axis. And this is *not* ensembling: the recent reading of a plain ResNet as an implicit ensemble of shallower paths leans on the additive $y = x + \sum T_i$, but ensemble members are trained *independently* and averaged, whereas my $C$ transformations are trained *jointly* inside one block, sharing one input and contributing to one residual; the aggregation is a representational device, not an averaging of separate models.

The fixed-complexity constraint is what turns cardinality into a controlled experiment. Raising $C$ adds paths, hence parameters and FLOPs, so I give something back by shrinking each path's bottleneck width $d$ — the cleanest lever, because $d$ is isolated from the block's $256$-channel input/output and from the network depth. A path costs $256\!\cdot\! d + 9 d^2 + 256\!\cdot\! d$ parameters and I have $C$ of them, so the block costs $C\cdot(256\!\cdot\! d + 9 d^2 + 256\!\cdot\! d)$, with FLOPs proportional at fixed feature-map size. The ResNet bottleneck baseline on this stage is $256\!\cdot\!64 + 9\!\cdot\!64^2 + 64\!\cdot\!256 \approx 70\text{k}$, recovered exactly at $C=1, d=64$. Matching $\approx 70\text{k}$ at higher cardinality gives the series $C=2\to d=40$, $C=4\to d=24$, $C=8\to d=14$, $C=32\to d=4$ (check $C=32,d=4$: $32\cdot(1024+144+1024)=32\cdot2192\approx 70\text{k}$). In the grouped form the middle width $C\!\cdot\! d$ runs $64, 80, 96, 112, 128$ along that same series — as I trade width for cardinality, the grouped $3\times3$ actually widens even as each group thins. My default is $C=32, d=4$, the network ResNeXt-50 (32$\times$4d); I adopt no smaller per-path width than $4d$, because the preserved-complexity sweep should saturate once $d$ is tiny and each transformation is too thin to be expressive. The decisive validation is *training* error: increasing $C$ at matched complexity should lower validation error monotonically, but the claim that the block genuinely represents the data better stands only if it lowers *training* error too — otherwise I could not rule out a mere regularization effect.

Building the full net is then just the VGG/ResNet template with this block: same map size $\to$ same width, halve the map $\to$ double the width. A $7\times7$, $64$, stride-2 stem and a $3\times3$ max-pool, then four stages of the block stacked $[3,4,6,3]$ (mirroring ResNet-50); with the 32$\times$4d template the grouped-$3\times3$ widths are $128/256/512/1024$ and block outputs $256/512/1024/2048$; global average pool, a single 1000-way FC, softmax. This lands at $\approx 25.0$M parameters and $4.2$ GFLOPs against ResNet-50's $25.5$M and $4.1$ GFLOPs — a genuinely matched-complexity twin that differs *only* in that its blocks aggregate 32 same-shape transformations instead of computing one. The implementation details keep the comparison clean: identity shortcuts except the dimension-increasing $1\times1$ projections at stage transitions (the type-B choice); downsampling by stride 2 in the $3\times3$ (grouped) layer of each stage's first block; batch norm after each convolution and ReLU after each BN, except at the block output where ReLU is applied *after* the shortcut add so the residual branch can carry negative corrections; and the inherited training recipe untouched so any accuracy difference is attributable to the block — $224\times224$ random crops with scale/aspect-ratio augmentation, SGD with mini-batch 256 across 8 GPUs, momentum 0.9, weight decay $10^{-4}$, learning rate 0.1 divided by 10 three times, He initialization.

```python
import torch
import torch.nn as nn
import math


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=32, base_width=4):
        super().__init__()
        D = int(planes * (base_width / 64.0))
        width = D * cardinality                          # C * d
        self.conv1 = conv1x1(inplanes, width)            # 1x1 (in -> C*d)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)   # grouped 3x3
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1 (C*d -> out)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))       # aggregates C same-shape transforms
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity                                  # y = x + sum_i T_i(x)
        return self.relu(out)


class ResNeXt(nn.Module):
    def __init__(self, layers, num_classes=1000, cardinality=32, base_width=4):
        super().__init__()
        self.cardinality = cardinality
        self.base_width = base_width
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtBottleneck.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        block = ResNeXtBottleneck
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.cardinality, self.base_width)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                cardinality=self.cardinality, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnext50_32x4d():  return ResNeXt([3, 4, 6, 3], cardinality=32, base_width=4)
def resnext101_32x4d(): return ResNeXt([3, 4, 23, 3], cardinality=32, base_width=4)
```
