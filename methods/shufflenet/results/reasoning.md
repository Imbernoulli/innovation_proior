Let me start from the regime I actually care about, because it's the opposite of where everyone is pushing. The field is racing toward bigger — hundreds of layers, thousands of channels, billions of FLOPs. I want the other end: the best accuracy I can get in tens to a couple hundred MFLOPs, the budget of a phone or a drone or a small robot. And I want to do it the hard way — not by taking some big "basic" network and pruning or quantizing it down, but by *designing a basic architecture from scratch* for this tiny regime. Those are different problems; compression accepts an architecture and trims it, while I want the architecture itself to be right for tens of MFLOPs.

So let me take the best efficient building blocks I have and ask where they break at this scale. The state-of-the-art efficient units make the *spatial* convolution cheap — depthwise separable convolutions, or grouped 3×3 convolutions like the cardinality blocks. Let me account for the FLOPs in a residual bottleneck unit to see exactly what they save and what they don't. Input is `c×h×w`, the bottleneck has `m` channels. A dense residual unit is `1×1 (c→m) → 3×3 (m→m) → 1×1 (m→c)`, costing `hw·(2cm + 9m²)`: the two pointwise 1×1 convolutions are the `2cm` term, the dense 3×3 is the `9m²` term (nine spatial taps times `m` in times `m` out). Now group the 3×3 into `g` groups — the cardinality trick. The grouped 3×3 costs `9m²/g`, so the unit becomes `hw·(2cm + 9m²/g)`. The spatial term dropped by a factor of `g`. Good. But look at what *didn't* move: the `2cm` pointwise term is untouched.

And here is the thing that decides the whole design. At this tiny scale, with the 3×3 already grouped down, the `2cm` term isn't a minor leftover — it's the *dominant* cost. Work it through for a grouped-3×3 unit with cardinality 32: the pointwise 1×1 convolutions end up being about 93% of the unit's multiply-adds. Ninety-three percent. So all the cleverness went into the 9% and the 91%... the 93% sits there dense. The depthwise-separable units have the same disease from the other direction — they make the spatial part nearly free with a depthwise conv, but the pointwise 1×1 that does all the channel mixing stays dense and dominates. So at tens of MFLOPs, *the pointwise convolutions are the bottleneck*, and because they're so expensive, they force the channel count to stay small to fit the budget. That's the real damage: a tiny network with too few channels has thin feature maps that simply can't carry enough information, and accuracy suffers. If I could cut the pointwise cost, I could spend the freed budget on *more channels*, which is exactly what a small network is starved for.

The obvious move is to do to the 1×1 layers what cardinality did to the 3×3: make them *grouped* too — pointwise group convolution. If each 1×1 only connects within a channel group, its cost drops by the group factor, and now the freed FLOPs can buy wider feature maps. But there's a side effect, and I have to face it squarely or the whole thing falls apart. Stack two group convolutions in a row, and trace a single output channel back: it depends only on the inputs *within its own group* at the previous layer, which in turn depended only on inputs within that group, and so on. The groups never talk. So the network fractures into `g` parallel, non-communicating lanes — outputs of a group are derived only from a small fixed fraction of the input channels. That blocks information flow across channels and badly weakens the representation. Grouping the 1×1s naively trades accuracy for the FLOPs I wanted to save.

So I need cross-group communication, but cheaply — I can't undo the savings by adding a dense mixing layer. What does cross-group flow actually require? After a group conv produces, say, `g` groups of `n` channels each, the next group conv would ideally receive, in each of its groups, a *mix* drawn from all the previous groups rather than one whole previous group. Concretely: take each previous group, split it into subgroups, and hand each next-layer group a different subgroup from each of the previous groups — so every next group sees a little of every previous group. That's a permutation of channels, nothing more. And a permutation is free. Make it concrete as a tensor reshape: a layer with `g` groups outputs `g·n` channels; reshape that channel axis into a `(g, n)` grid, *transpose* it to `(n, g)`, and flatten back to `g·n`. After the transpose-and-flatten, what used to be "the i-th channel of group j" is interleaved so that each new group of `n` channels is composed of one channel from each old group. That's exactly "feed each next group a subgroup from every previous group." I'll call it a channel shuffle. It costs no parameters and almost no compute, it works even if the two convolutions it sits between have *different* numbers of groups, and — because it's just a deterministic permutation — it's differentiable, so I can drop it into the network and train end-to-end straight through it. So pointwise group convolution buys the FLOP savings, and channel shuffle pays back the side effect they create.

Now build the unit. Start from the residual bottleneck I trust: `1×1 reduce → 3×3 → 1×1 expand`, added onto the shortcut. I'll make it as cheap as possible everywhere. The first 1×1 becomes a *pointwise group convolution* that reduces to the bottleneck width — and right after it I insert the channel shuffle, so the groups get mixed before the spatial step. For the 3×3, I go all the way to *depthwise*: one filter per channel, the cheapest spatial convolution there is, run on the bottleneck feature map. Then the second 1×1 is another pointwise group convolution that expands back up to match the shortcut's channel count. Add the identity shortcut, then ReLU.

A few details I want to get right rather than copy by reflex. After the depthwise 3×3 I put BN but *no ReLU* — rectifying right after a depthwise convolution is known to hurt, because a depthwise filter has so little capacity per channel that clamping its output throws away information it can't afford to lose; the nonlinearities live after the pointwise convs instead. I also do *not* add a second channel shuffle after the expand-1×1 — I tried imagining it and it gives comparable scores, so I leave it out for simplicity; one shuffle, right after the first group conv, is enough to restore cross-group flow before the spatial step. And the depthwise conv I keep *only* on the narrow bottleneck. That's deliberate and it's a hardware point, not a theory point: depthwise convolution looks almost free in FLOPs, but on a low-power mobile device its compute-to-memory-access ratio is poor — it moves a lot of data per multiply — so it's actually slow to run. Keeping it on the thin bottleneck limits how much of that overhead I pay.

Now the downsampling unit, where the spatial map halves and channels grow. Two changes. On the shortcut path I add a 3×3 average pooling with stride 2, so the shortcut lands on the same smaller spatial grid as the residual branch (whose depthwise conv now runs at stride 2). And instead of *adding* the shortcut to the branch, I *concatenate* them along the channel axis. The reason is economy: between stages I want to roughly double the channel count, and concatenation enlarges the channel dimension essentially for free — I just glue the pooled shortcut's channels onto the branch's output — whereas an add would force the branch to produce the full doubled width on its own. So the stride-2 unit concatenates, the stride-1 unit adds.

Let me check the FLOPs of the finished unit against the baselines, because the whole justification was cost. Same input `c×h×w`, bottleneck `m`. The two pointwise convs are now grouped by `g`, so together they cost `2cm/g` instead of `2cm`. The 3×3 is depthwise, costing `9m` (nine taps per channel, `m` channels) instead of `9m²` or `9m²/g`. So the ShuffleNet unit is `hw·(2cm/g + 9m)`. Set it beside the others: dense ResNet `hw(2cm + 9m²)`, grouped-3×3 ResNeXt `hw(2cm + 9m²/g)`, and mine `hw(2cm/g + 9m)`. Both of my terms are smaller — the pointwise term is cut by `g`, and the spatial term went from quadratic in `m` to linear. For a fixed FLOP budget, that means I can afford a substantially wider network — more channels — which is the entire point, because the tiny-network failure mode was too few channels to carry the information.

Assemble the network. A 3×3 stride-2 stem to 24 channels, then a 3×3 stride-2 max-pool, then three stages built from ShuffleNet units. Each stage's first unit is the stride-2 concatenating kind; the rest are stride-1 adding units, with the same hyper-parameters across a stage, and the output channels double at each new stage. Following the bottleneck convention, each unit's bottleneck width is one quarter of its output channels. Stage 2 repeats the unit 4 times (one stride-2 + 3 stride-1) at 28×28, stage 3 repeats it 8 times at 14×14, stage 4 repeats it 4 times at 7×7. Then a global 7×7 average pool and a 1000-way FC.

The group count `g` is a real knob, and I want to think about what it trades. `g` sets the sparsity of the pointwise convolutions: bigger `g` makes each 1×1 cheaper, so under a fixed FLOP budget I can afford *more output channels*, more filters, more capacity to encode information. So I'll *adapt the channel widths to `g`* so that every choice of `g` lands at roughly the same ~140 MFLOPs — that way I'm comparing the *effect of grouping* at constant complexity, not just spending more. Concretely the stage widths grow with `g`: at the 1× scale, stage-2 output channels run 144, 200, 240, 272, 384 as `g` goes 1, 2, 3, 4, 8, and the later stages double from there. But more `g` isn't unconditionally better: as `g` grows and channels widen, each individual filter sees *fewer input channels* (its group is a thinner slice), which can degrade what that filter can represent. So I'd expect a sweet spot — and I'd expect it to depend on model size, with the *smallest* models favoring *larger* `g`, because the thinner a network's feature maps are to begin with, the more it gains from the extra channels that grouping buys. One special case: in stage 2, the very first pointwise layer takes only the 24 stem channels as input, which is already tiny, so grouping it there would over-fragment an already-thin input for almost no saving — I leave that one pointwise conv ungrouped.

And to hit any target budget, a simple width knob: scale the number of filters everywhere by a factor `s`. Since FLOPs are roughly quadratic in width, "ShuffleNet `s×`" costs about `s²` times the 1× model — so the 1× model at ~140 MFLOPs becomes ~38 MFLOPs at 0.5× and ~13 MFLOPs at 0.25×.

The training recipe I mostly inherit from the grouped-residual setup, with two adjustments that both come from the same fact: very small networks *underfit* rather than overfit. So I dial regularization *down* — weight decay 4e-5 instead of the usual 1e-4 — and use a linearly decayed learning rate from 0.5 down to 0, with less aggressive scale augmentation than a big-network recipe would use. Batch size 1024 across 4 GPUs, about 3×10⁵ iterations.

What I'd want to validate, stated as falsifiable predictions. First, pointwise group convolution with `g > 1` should beat the `g = 1` version (which is essentially a depthwise-separable / Xception-style unit) at matched complexity — and the gain should be *larger* for smaller models, because they're the ones starved for channels. Second, turning the channel shuffle *off* should hurt, consistently, and hurt *more* when `g` is large — because large `g` means more fractured groups that need cross-group flow the most. If both hold, the story is exactly the one I reasoned to: cheapen the pointwise convolutions with grouping, spend the savings on width, and restore the cross-group communication that grouping breaks with a free channel permutation.

Let me write it as code, filling the two slots — the mixing operation becomes the channel shuffle, and the unit becomes the grouped-pointwise / depthwise / shuffle bottleneck.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    # restore cross-group flow: reshape (g, n) -> transpose -> flatten
    N, C, H, W = x.size()
    n = C // groups
    x = x.view(N, groups, n, H, W)            # (g, n)
    x = torch.transpose(x, 1, 2).contiguous()  # -> (n, g)
    return x.view(N, C, H, W)                   # flatten back


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):
        super().__init__()
        self.groups = groups
        self.combine = combine
        self.bottleneck = out_channels // 4               # ResNet 1/4 ratio
        if combine == 'add':                              # stride-1 unit
            self.depthwise_stride = 1
        else:                                             # 'concat': stride-2 unit
            self.depthwise_stride = 2
            out_channels = out_channels - in_channels     # concat adds in_channels back

        # stage-2's first unit takes the tiny 24-channel stem -> don't group it
        first_groups = groups if grouped_conv else 1
        # 1x1 pointwise GROUP conv: reduce to bottleneck
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, self.bottleneck, 1, groups=first_groups, bias=False),
            nn.BatchNorm2d(self.bottleneck), nn.ReLU(inplace=True))
        # 3x3 DEPTHWISE conv on the bottleneck (no ReLU after it)
        self.dwconv = nn.Conv2d(self.bottleneck, self.bottleneck, 3,
                                stride=self.depthwise_stride, padding=1,
                                groups=self.bottleneck, bias=False)
        self.dwbn = nn.BatchNorm2d(self.bottleneck)
        # 1x1 pointwise GROUP conv: expand back to match the shortcut
        self.expand = nn.Sequential(
            nn.Conv2d(self.bottleneck, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)
        out = self.compress(x)
        out = channel_shuffle(out, self.groups)           # restore cross-group flow
        out = self.dwbn(self.dwconv(out))                 # depthwise spatial, no ReLU
        out = self.expand(out)
        if self.combine == 'add':
            out = out + residual                          # stride-1: residual add
        else:
            out = torch.cat((residual, out), 1)           # stride-2: concat to grow channels
        return F.relu(out)


class ShuffleNet(nn.Module):
    # stage output channels per g, at 1x scale (adapted to hold ~140 MFLOPs):
    _stage_out = {1: [144, 288, 576], 2: [200, 400, 800], 3: [240, 480, 960],
                  4: [272, 544, 1088], 8: [384, 768, 1536]}

    def __init__(self, groups=3, num_classes=1000, scale=1.0):
        super().__init__()
        self.groups = groups
        stage_out = [int(scale * c) for c in self._stage_out[groups]]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage2 = self._make_stage(24, stage_out[0], repeats=3, first_grouped=False)
        self.stage3 = self._make_stage(stage_out[0], stage_out[1], repeats=7, first_grouped=True)
        self.stage4 = self._make_stage(stage_out[1], stage_out[2], repeats=3, first_grouped=True)
        self.fc = nn.Linear(stage_out[2], num_classes)

    def _make_stage(self, in_ch, out_ch, repeats, first_grouped):
        # first unit: stride-2, concatenating, channel-growing
        layers = [ShuffleUnit(in_ch, out_ch, self.groups,
                              grouped_conv=first_grouped, combine='concat')]
        # remaining units: stride-1, residual-add
        for _ in range(repeats):
            layers.append(ShuffleUnit(out_ch, out_ch, self.groups,
                                      grouped_conv=True, combine='add'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


model = ShuffleNet(groups=3, num_classes=1000, scale=1.0)            # ~140 MFLOPs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=4e-5)
# linear LR 0.5 -> 0 over ~3e5 iters, batch 1024 on 4 GPUs, light scale augmentation.
```

The whole method is one chain: at a tiny budget the dense pointwise 1×1 convolutions dominate the FLOPs and choke the channel count, so group them to make them cheap, restore the cross-group information flow they break with a free channel-shuffle permutation, run the spatial step as a cheap depthwise convolution on the bottleneck, and spend every freed FLOP on wider feature maps — which is exactly what a small network needs to carry enough information.
