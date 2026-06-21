The problem I want to solve is to design a convolutional image-recognition block that is accurate inside a very small compute envelope — roughly tens to a couple hundred MFLOPs, the range where a phone, a drone, or a small robot can run the model repeatedly. This is not a post-training compression problem; pruning, low-rank factorization, quantization, and distillation all start from a large trained model and ask how to shrink it, and none of them answers the prior question of which basic unit should be built in the first place. So I want to choose the building block itself, and I want to be judged honestly: by multiply-adds at matched budgets, and by actual single-thread latency on an ARM-class mobile CPU, because a layer that is cheap in FLOPs can still be slow if its memory access pattern is poor.

The instructive way to see why the standard options fall short is to do the accounting on a residual bottleneck. On an $h \times w$ feature map, with input and output width $c$ and bottleneck width $m$, the branch is $1\times1$ reduce, $3\times3$ spatial, $1\times1$ expand, and the cost is $hw(2cm + 9m^2)$: the $2cm$ term is the two pointwise channel-mixing layers, the $9m^2$ term is the spatial layer. ResNeXt-style grouping attacks the spatial term by splitting the $3\times3$ into $g$ groups, giving $hw(2cm + 9m^2/g)$. That is the right move when the spatial term dominates, but in the tiny-network regime it has already been squeezed, and the two dense pointwise layers are now what is expensive — with cardinality 32 the pointwise convolutions occupy $93.4\%$ of the unit's multiply-adds. MobileNet's depthwise-separable block goes to the other extreme on spatial filtering — one $3\times3$ filter per channel — but its pointwise convolution that does all the cross-channel mixing stays dense, so at very small widths that pointwise layer is again the bottleneck. The deeper point is that a small model is not just a large model with fewer operations: it is starved for channels, with too little width to carry information forward. Every multiply-add wasted on dense channel mixing is width I cannot afford. So I need to make the $1\times1$ mixing cheap precisely so that I can spend the freed budget on more channels at the same FLOP count.

I propose ShuffleNet. The decisive move is to make the expensive $1\times1$ pointwise convolutions grouped, just as ResNeXt groups the $3\times3$. A grouped $1\times1$ from $c$ to $m$ costs $cm/g$ instead of $cm$, and the two pointwise layers together drop from $2cm$ to $2cm/g$. But grouping the pointwise layers naively has a real failure mode: if I stack grouped $1\times1$ convolutions, each output group only ever sees the input channels of its own group, so the groups become isolated lanes — a channel in group 0 depends only on group 0 before it, which depended only on group 0 before that. I would have bought the FLOP savings by cutting exactly the cross-group communication that representation learning needs, which is a bad bargain. The repair is not another dense layer — that would give back the savings — it is to change *which channels sit together* before the next grouped convolution. If a grouped layer outputs $g \times n$ channels, I view the channel axis as a $g \times n$ grid, transpose it to $n \times g$, and flatten: now each downstream group receives one slice from every upstream group. This is a parameter-free permutation, the channel shuffle, with no weights and negligible arithmetic cost, and it lets the next grouped pointwise layer mix information that originated in different earlier groups. That single permutation is what makes grouping the pointwise layers safe.

So the unit is a bottleneck residual branch with three cheap choices stitched together. The first $1\times1$ is grouped and reduces to the bottleneck width $m$. The $3\times3$ is depthwise — one filter per bottleneck channel — so it costs $9m$ rather than $9m^2$ or $9m^2/g$, linear in $m$ because there is exactly one $3\times3$ filter per channel. The channel shuffle is applied before the second grouped $1\times1$, which then expands back to the output width, also grouped. For the stride-1 unit with input/output width $c$, bottleneck width $m$, map $h\times w$, and $g$ pointwise groups, the cost is

$$hw\!\left(\frac{2cm}{g} + 9m\right),$$

where the $2$ is the two pointwise layers and the $9$ is the $3\times3$ kernel. Set against the dense bottleneck $hw(2cm + 9m^2)$ and the grouped-$3\times3$ bottleneck $hw(2cm + 9m^2/g)$, I have reduced both the channel-mixing term and the spatial term. Two implementation details earn their place. There is no ReLU immediately after the depthwise convolution: a depthwise filter carries only one channel's worth of capacity, so clamping it there is a needless information loss — batch norm follows the depthwise, ReLU does not. And the very first pointwise layer in the first unit of stage 2 is left dense rather than grouped, because its input is only the stem width and grouping an already tiny set of channels would split it too finely.

Downsampling is handled as a separate case, and the choice there is concatenation rather than addition. When a stage changes resolution, the depthwise convolution takes stride 2, and the shortcut must also reach the smaller grid, so I average-pool the shortcut with a stride-2 $3\times3$ pool. If I added the two branches, the learned residual branch would have to manufacture the entire wider output tensor by itself; instead I concatenate the pooled shortcut with the learned branch and let the branch produce only the remaining $\text{out\_channels} - \text{in\_channels}$ channels. That lets a stage grow its width while spending less compute. So stride-1 units use residual addition, stride-2 units use average-pool shortcut plus channel concatenation.

The group count $g$ becomes a genuine architectural knob with a predictable tradeoff. Larger $g$ makes the pointwise layers cheaper, so at fixed complexity I can allocate more output channels — but each individual grouped filter then sees fewer input channels, so its local view thins out. Grouped pointwise layers should help at fixed MFLOPs because they buy width, yet very large $g$ can saturate or hurt because each filter's view becomes too narrow; the smallest models, being most starved for channels, should benefit most from larger $g$. The full network follows the residual-stage template: a $3\times3$ stride-2 stem, a $3\times3$ stride-2 max-pool, then three stages with repeat counts $[4, 8, 4]$ where the first unit of each stage downsamples and the rest run at stride 1, the bottleneck width fixed at one quarter of the stage output width, ending in global average pooling and a linear classifier. In the released reference, the channel shuffle is placed after the depthwise batch norm and immediately before the second grouped pointwise layer — the placement that matters, since that is exactly where cross-group information would otherwise be blocked.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    batch, channels, height, width = x.size()
    assert channels % groups == 0
    group_channels = channels // groups
    x = x.reshape(batch, group_channels, groups, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(batch, channels, height, width)


class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, stride):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.group = group
        outputs = oup - inp if stride == 2 else oup

        self.branch_main_1 = nn.Sequential(
            nn.Conv2d(inp, mid_channels, 1, 1, 0,
                      groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.branch_main_2 = nn.Sequential(
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        )
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        x = self.branch_main_1(old_x)
        if self.group > 1:
            x = channel_shuffle(x, self.group)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return F.relu(x + old_x)
        return torch.cat((self.branch_proj(old_x), F.relu(x)), 1)


class ShuffleNetV1(nn.Module):
    stage_repeats = [4, 8, 4]
    stage_out = {
        3: {
            "0.5x": [-1, 12, 120, 240, 480],
            "1.0x": [-1, 24, 240, 480, 960],
            "1.5x": [-1, 24, 360, 720, 1440],
            "2.0x": [-1, 48, 480, 960, 1920],
        },
        8: {
            "0.5x": [-1, 16, 192, 384, 768],
            "1.0x": [-1, 24, 384, 768, 1536],
            "1.5x": [-1, 24, 576, 1152, 2304],
            "2.0x": [-1, 48, 768, 1536, 3072],
        },
    }

    def __init__(self, n_class=1000, model_size="1.0x", group=3):
        super().__init__()
        if group not in self.stage_out:
            raise ValueError("Megvii reference supports group 3 or 8")
        if model_size not in self.stage_out[group]:
            raise ValueError("model_size must be 0.5x, 1.0x, 1.5x, or 2.0x")

        self.stage_out_channels = self.stage_out[group][model_size]
        input_channel = self.stage_out_channels[1]

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        features = []
        for idxstage, numrepeat in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                features.append(ShuffleV1Block(
                    input_channel, output_channel,
                    group=group,
                    first_group=first_group,
                    mid_channels=output_channel // 4,
                    stride=stride,
                ))
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.globalpool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(self.stage_out_channels[-1], n_class, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.globalpool(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        return self.classifier(x)

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(module.weight, 0, 0.01)
                else:
                    nn.init.normal_(module.weight, 0, 1.0 / module.weight.shape[1])
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


model = ShuffleNetV1(n_class=1000, model_size="1.0x", group=3)
```
