Let me start from a tension I can't reconcile, because it's the whole problem. I have two ways of building an image classifier that both work, and they pull in opposite directions. One is the VGG/ResNet way: pick a building block, then stack it — same shape, over and over, doubling the width whenever you halve the spatial map. The appeal is that this collapses the hyper-parameter space to almost nothing; you design one block and the rule writes the rest of the net, so depth becomes the one clean dimension you turn. And because the rule is so simple, it's hard to over-fit your *architecture* to one dataset, which is exactly why these nets transfer everywhere. The other way is the Inception way: hand-design multi-branch modules — split the input into low-dimensional embeddings with 1×1 convolutions, transform each with its own filters, merge by concatenation — and you get excellent accuracy at low theoretical complexity. But the price is that every branch's filter count and size is tailored by hand, and the modules differ stage by stage. So when a new dataset or task shows up, you're back to redesigning, because there's no clean knob to turn.

So the question I want to answer: can I keep the repeat-one-block simplicity *and* get the split-transform-merge efficiency, in a form simple enough that the multi-branch structure is governed by a single, turnable parameter? And the harder constraint I'm going to hold myself to: I want accuracy gains *at fixed complexity* — same FLOPs, same parameter count. It's easy to buy accuracy by going deeper or wider; what's rare, and what I think is more interesting, is buying accuracy while the compute budget doesn't move. Because for the existing models, depth and width are starting to give diminishing returns — each additional layer or channel costs more and returns less. If there's a *third* axis of the design space, distinct from depth and width, it might be the more efficient place to spend.

Let me look for that third axis by going all the way down to the most elementary operation in the network and asking what structure is already hiding in it. The simplest "neuron" — the thing a fully-connected or convolutional layer does per output — is an inner product: take a D-channel input `x = [x_1, …, x_D]` and compute `Σ_{i=1}^{D} w_i x_i`. Stare at that sum as a *process* rather than a formula. It splits the input vector into D single-channel pieces `x_i`. It transforms each piece — here trivially, just scaling, `w_i x_i`. And it aggregates the transformed pieces by summing them. Split, transform, aggregate. That's the same three-step shape as an Inception module — except the splitting is into single dimensions, the transforming is just a scalar multiply, and the aggregating is a plain sum. So split-transform-merge isn't an exotic Inception invention; it's the deep structure of the inner product itself.

That reframing hands me the move. If the neuron is "split into D pieces, scale each, sum," I can generalize the *transform* step while keeping the split-and-sum skeleton: replace the elementary `w_i x_i` with a generic function `T_i(x)` that is itself a small network. So the new aggregating transformation is

  `F(x) = Σ_{i=1}^{C} T_i(x)`,

where each `T_i` does what `w_i x_i` did in spirit — project `x` into some (optionally low-dimensional) embedding and then transform it — but now `T_i` can be a richer, multi-layer function. This is a "Network-in-Neuron": where Network-in-Network put a small net in place of a filter and ended up adding to the *depth* dimension, putting a small net inside the *neuron* and aggregating C of them expands along a different axis. That axis — `C`, the number of transformations I'm summing — is the new dimension. I'll call it *cardinality*. Notice where `C` sits: in the inner product the sum ran to `D`, the input channel count; here the sum runs to `C`, the number of aggregated transformations. They occupy the same structural slot, but `C` is free — it need not equal `D` and can be anything. Width is about the number of *simple* transformations (the channels in an inner product); cardinality is about the number of *complex* ones.

Now I have to fix what the `T_i` are, and here's where I refuse to repeat Inception's mistake. Inception lets each branch be different, and that's the source of its un-transferability. I'll make all the `T_i` have the *same topology*. That's just the VGG repeat-the-same-shape philosophy applied at the sub-block level, and it buys me exactly what I wanted: the number of paths becomes a single isolated factor I can dial up to any value, with no per-path design. For the topology of each `T_i`, I reuse the thing I already trust — the bottleneck shape: a 1×1 that reduces to a low-dimensional embedding, a 3×3 that does the spatial work at that reduced width, a 1×1 that restores. The first 1×1 in each `T_i` *is* the "project into a low-dimensional embedding" step, the split-transform part of split-transform-merge. So one path is `1×1 (256→d) → 3×3 (d→d) → 1×1 (d→256)`, and I lay down `C` of them in parallel.

And the aggregation is a sum, matching the inner product's `Σ`. But I'm building residual blocks, so the natural thing is to make this aggregated transformation the *residual function* itself, added onto the identity shortcut:

  `y = x + Σ_{i=1}^{C} T_i(x)`.

So the block is: take `x`, fan it out into `C` same-shaped bottleneck paths, sum their outputs, add the result back to `x`. The residual conditioning from the identity shortcut carries over unchanged — I haven't touched what makes deep nets trainable, I've only restructured what's *inside* the residual function.

Now I want to be honest about whether this is actually new or whether I've just drawn the same thing twice, so let me see if the block has equivalent forms — if it does, the equivalences will tell me what it really is. Take the simplest nontrivial case, `C = 2`. Each path ends in a 1×1 layer; call the weight of that last layer in path `i` the matrix `A_i`, and the response coming out of the second-to-last layer of path `i` the vector `B_i`. The two paths' contributions before the residual add are `A_1 B_1` and `A_2 B_2`, and I sum them: `A_1 B_1 + A_2 B_2`. But there's an algebraic identity sitting right there: `A_1 B_1 + A_2 B_2 = [A_1, A_2] · [B_1; B_2]`, where `[A_1, A_2]` is the horizontal concatenation of the two weight matrices and `[B_1; B_2]` is the vertical concatenation (stacking) of the two response vectors. Read the right-hand side as a network: `[B_1; B_2]` is what you get if you *concatenate* the two paths' second-last outputs into one wider tensor, and `[A_1, A_2]` is a *single* 1×1 layer applied to that concatenation. So summing two separate last-1×1 layers is identical to concatenating the paths first and then applying one shared last-1×1 layer. That gives me a second, equivalent form of the block: instead of `C` full bottleneck paths summed, do `C` paths of `1×1 → 3×3`, *concatenate* their outputs, and run a single 1×1 to restore. That form looks like Inception-ResNet — branch and concatenate inside the residual function — except all my branches are the same shape, so I still have the clean cardinality knob that Inception-ResNet lacks.

Push the equivalence one step further, because the concatenated form is suggestive. In that form, the first 1×1 of every path takes the same `x` and produces a `d`-dimensional embedding; `C` of them produce `C·d` channels total. But `C` separate 1×1 layers each mapping `256 → d`, taken together, are just one wider 1×1 layer mapping `256 → C·d` — they're independent linear maps on the same input, which is exactly what one fat 1×1 layer is. So replace the `C` little input-1×1s with a single `1×1 (256 → C·d)`. Then I have `C·d` channels that need to be processed by `C` separate 3×3 convolutions, each acting only on its own `d`-channel slice and not mixing across slices. That is *precisely* a grouped convolution: a 3×3 conv on `C·d` channels with `C` groups, where the input and output channels are partitioned into `C` groups and each group is convolved independently. Then one final `1×1 (C·d → 256)` to restore. So the third equivalent form is: `1×1 (256 → C·d) → grouped 3×3 with C groups → 1×1 (C·d → 256)`, plus the identity shortcut. It looks almost exactly like an ordinary bottleneck block — except the middle 3×3 is wider (`C·d` channels instead of the bottleneck's narrow waist) but *sparsely connected*, because the grouping forbids cross-group mixing.

This is a small thrill, because grouped convolution has been sitting in every library since the two-GPU AlexNet, where it existed purely as an *engineering compromise* to fit a model across two GPUs — nobody had used it to *improve* accuracy. And here it falls out, not as a trick I reached for, but as the most succinct implementation of "aggregate `C` same-topology transformations." All three forms — the `C` summed bottleneck paths, the concatenate-then-restore form, the grouped-convolution form — are *strictly* equivalent as long as I place batch norm and ReLU consistently. So I can reason about the block as "summing C transformations," and implement it as one grouped convolution, which is more compact and faster. One caveat I should note: these reformulations only produce a nontrivial topology when each path has depth ≥ 3, like the bottleneck. If the paths were depth-2 (a basic block), concatenating and regrouping would collapse to a trivially wide dense layer — no interesting structure. So the bottleneck depth isn't just for efficiency here; it's what makes cardinality a real, distinct axis.

Now the constraint I promised myself: study `C` *at fixed complexity*. If I increase `C`, I add paths, which adds parameters and FLOPs — so to keep the budget fixed I have to give something back, and the cleanest thing to give back is the bottleneck width `d` of each path, because `d` is isolated: it's the internal width of a path and doesn't touch the block's input or output width (fixed at 256-d) or the network depth. So changing `d` lets me trade cardinality against per-path width without disturbing anything else, which is exactly the controlled knob I need. Let me count the parameters so I can balance the trade. A path is `1×1 (256→d)`, `3×3 (d→d)`, `1×1 (d→256)`, costing `256·d + 3·3·d·d + d·256` parameters, and I have `C` of them:

  `C · (256·d + 9·d² + 256·d)` parameters,

with FLOPs proportional (same feature-map size). The baseline I'm matching is the ordinary ResNet bottleneck on this stage: `256·64 + 3·3·64·64 + 64·256 ≈ 70k` parameters. So I want to choose `(C, d)` pairs that land near 70k. With `C=1, d=64` I recover the baseline exactly. Solve for matched complexity at higher `C`: `C=2 → d=40`, `C=4 → d=24`, `C=8 → d=14`, and `C=32 → d=4`, each giving ≈ 70k parameters. (Plug in `C=32, d=4`: `32·(256·4 + 9·16 + 4·256) = 32·(1024 + 144 + 1024) = 32·2192 ≈ 70k`. Good.) In the grouped-convolution form, the width of the grouped 3×3 is `C·d`, which goes `64, 80, 96, 112, 128` along that same series — so as I trade width for cardinality, the grouped layer actually gets a bit wider even as each group gets thinner.

So my default template is `C = 32, d = 4` — call the network built from it ResNeXt-50 (32×4d). One thing I'd watch for: as I push `C` up and `d` down at fixed complexity, the per-path width `d` eventually gets so small that the trade stops paying — accuracy should start to saturate when the bottleneck width is tiny, because each transformation has too few channels to be expressive. So I won't shrink `d` below 4; the trade is worthwhile down to about there and no further.

Building the full net is now just the VGG/ResNet template applied with this block. Two rules: blocks producing the same spatial-map size share the same width and filter sizes, and each time the map is downsampled by 2 the block width doubles (which keeps per-block FLOPs roughly equal across stages). So: a 7×7, 64, stride-2 stem and a 3×3 max-pool; then four stages of the new block. With the 32×4d template the stages carry grouped-3×3 widths of 128, 256, 512, 1024 and block outputs of 256, 512, 1024, 2048, stacked [3, 4, 6, 3] times (mirroring ResNet-50). Global average pool, a single 1000-way FC, softmax. Count it up and ResNeXt-50 (32×4d) lands at about 25.0M parameters and 4.2 GFLOPs, essentially equal to ResNet-50's 25.5M and 4.1 GFLOPs — the tiny differences come only from the subsampling blocks where the map size changes, and they don't bias the comparison. So I have a genuinely matched-complexity twin of ResNet-50, differing *only* in that its blocks aggregate 32 same-shape transformations instead of computing one.

Let me also be precise about what this is *not*, because there's a tempting wrong reading. Someone could look at the additive `y = x + ΣT_i` and call it an ensemble — there's a recent reading of a plain ResNet as an implicit ensemble of shallower paths, which leans on exactly this additive behavior. But that reading would be imprecise here: the members of an ensemble are trained *independently* and then averaged, whereas my `C` transformations are trained *jointly* inside one block, sharing the same input and contributing to one residual. The aggregation is a representational device, not an averaging of separately-learned models.

A few implementation details to pin down so the comparison stays clean and the block trains. I follow the standard ResNet recipe and the public Torch training code. Shortcuts are identity, except the dimension-increasing ones at stage transitions, which are 1×1 projections (the type-B choice from the residual work). Downsampling for stages 3, 4, 5 happens by putting stride 2 in the 3×3 (grouped) layer of the first block of each stage. Batch norm goes right after each convolution in the grouped-conv form, ReLU right after each BN — except at the block output, where the ReLU is applied *after* adding to the shortcut, so the residual branch can still carry negative corrections and only the summed result is rectified. (If I'd implemented the summed-paths form instead, BN would go after aggregating the transformations and before the shortcut add — but since all three forms are equivalent, I implement the grouped-conv one because it's the most succinct and the fastest.) Training is the inherited recipe, untouched, so any accuracy difference is attributable to the block: 224×224 random crops with scale and aspect-ratio augmentation, SGD with mini-batch 256 across 8 GPUs, momentum 0.9, weight decay 1e-4, learning rate starting at 0.1 and divided by 10 three times, He initialization, BN after every conv.

What I'd want to validate, stated as a falsifiable prediction: at matched complexity, increasing `C` from 1 upward should lower validation error monotonically (until `d` gets too small), and — crucially — it should lower *training* error too. That last part is the decisive test, because if cardinality only lowered validation error I couldn't rule out a regularization effect; if it lowers *training* error, the block is genuinely representing the data better, which is the claim. And I'd expect cardinality to beat spending the same extra budget on depth or width, precisely in the regime where depth and width have hit diminishing returns.

Let me write it as code, filling the one open slot — the block — with the grouped-convolution form, and leaving the VGG/ResNet template harness exactly as it was.

```python
import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=32, base_width=4):
        super().__init__()
        # width of the grouped 3x3 = C * d, where d scales with `planes`:
        # d = planes * (base_width / 64), and C*d is the grouped-conv width.
        D = int(planes * (base_width / 64.0))
        width = D * cardinality                  # = C * d  (e.g. 32*4 = 128 at stage 1)

        self.conv1 = conv1x1(inplanes, width)            # 1x1 (in -> C*d): the fused split
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)  # grouped 3x3: C transforms
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1 (C*d -> out): the merge
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample                      # identity, or 1x1 projection at transitions
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))        # grouped: aggregates C same-shape paths
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity                                   # y = x + sum_i T_i(x)
        return self.relu(out)                             # ReLU after the add


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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

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

The whole method is one move, traced back to its root: the inner product is already a split-transform-merge over its channels, so generalize the per-channel scalar transform into a same-topology bottleneck sub-network, aggregate `C` of them by summation as the residual function, and trade per-path width for `C` to keep complexity fixed — which exposes *cardinality* as a third design axis beside depth and width, implemented for free as a grouped convolution.
