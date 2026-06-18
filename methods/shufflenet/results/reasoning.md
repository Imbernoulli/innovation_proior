I start from the budget, because that is where the ordinary architecture intuition misleads me. If I had billions of multiply-adds, I could deepen or widen a residual network and let the usual bottleneck block carry me. Here I have something like 10 to 150 MFLOPs. At that scale, the question is not "how do I shrink a good large model after the fact?" It is "what unit should I have built in the first place?"

The bottleneck residual unit is the clean accounting baseline. On an `h x w` map, with input and output width `c` and bottleneck width `m`, the dense residual branch costs `cm` for the first `1x1`, `9m^2` for the `3x3`, and `mc` for the last `1x1`, all multiplied by `hw`. So the cost is `hw(2cm + 9m^2)`. ResNeXt attacks the `9m^2` term by grouping the `3x3`, giving `hw(2cm + 9m^2/g)`. That is a good move if the spatial term is the problem. But in a tiny network the spatial term has already been squeezed, and the two pointwise convolutions are now the expensive part. With cardinality 32, the pointwise layers occupy 93.4% of the unit's multiply-adds. So I cannot get a small accurate network by only making the `3x3` cheaper. The dense `1x1` layers are choking the width.

The width point matters. A small model is not just a large model with fewer operations; it has too few channels to carry information through the network. If I save compute from the pointwise layers, I can spend it on more channels at the same MFLOP budget. That is exactly the trade I want. So the natural idea is to apply grouping to the `1x1` layers too. A grouped `1x1` from `c` to `m` costs `cm/g` instead of `cm`, and a grouped `1x1` from `m` back to `c` costs another `cm/g`. That would change the pointwise term from `2cm` to `2cm/g`.

But that move has a serious failure mode. If I stack grouped pointwise convolutions naively, each output group only sees the input channels from its own group. The groups become separate lanes. A channel in group 0 at this layer depends on group 0 before it, which depended on group 0 before that, and so on. I have saved compute by cutting cross-group communication, but representation learning needs cross-group communication. A cheap unit that partitions the feature map into isolated subnets is not a good unit.

What I need is not another dense layer; that would give back the FLOP savings. I only need to change which channels are placed together before the next grouped convolution. Suppose a grouped layer outputs `g * n` channels. I can view the channel axis as a grid: `g` groups, each with `n` channels. If I transpose that grid and flatten it, the next group no longer receives one old group; it receives one slice from each old group. That is only a permutation. It has no weights and no meaningful arithmetic cost, but it lets the next grouped pointwise layer combine information that came from different previous groups.

So the unit becomes a bottleneck residual branch with three cheap choices. The first `1x1` is grouped and reduces to the bottleneck width. Somewhere before the second grouped `1x1`, I apply the channel permutation so the next groups are mixed. The `3x3` spatial layer is depthwise, one filter per bottleneck channel, so it costs `9m` rather than `9m^2` or `9m^2/g`. The second `1x1` is grouped and expands back to the required output width. Batch norm and ReLU follow the pointwise reduction; batch norm follows the depthwise convolution, but no ReLU goes immediately after that depthwise operation. A depthwise filter has only one channel's worth of capacity, so clamping it there is an unnecessary information loss, and Xception already gives me that warning.

Now I check the arithmetic. For a stride-1 unit with input/output width `c`, bottleneck width `m`, feature map `h x w`, and `g` pointwise groups, the cost is `hw(2cm/g + 9m)`. The signs and constants are all positive because this is multiply-add counting. The `2` is exactly the two pointwise layers. The `9` is exactly the `3 x 3` kernel. The spatial term is linear in `m` because depthwise convolution has one `3 x 3` filter per channel, not `m` filters per channel. Compared with the dense bottleneck `hw(2cm + 9m^2)` and grouped-`3x3` bottleneck `hw(2cm + 9m^2/g)`, I have reduced both the channel-mixing term and the spatial term.

Downsampling is a separate case. If the stage changes resolution, the residual branch should use stride 2 in the depthwise convolution. The shortcut also has to land on the smaller grid, so I put a `3x3` average pool with stride 2 on the shortcut. If I add the two branches, the residual branch has to produce the whole wider output tensor by itself. Concatenation is cheaper: preserve the pooled shortcut channels and make the learned branch produce only the remaining channels. That lets the stage grow its width while spending less compute. So stride 1 uses residual addition; stride 2 uses average-pool shortcut plus channel concatenation.

The full network follows the residual-stage template. A `3x3` stride-2 stem feeds a `3x3` stride-2 max-pool, then three stages repeat the unit with repeat counts `[4, 8, 4]`; the first unit in each stage downsamples, and the rest use stride 1. The bottleneck width is one quarter of the stage output width. The first pointwise layer in the first unit of stage 2 is a special case: its input has only the stem width, so grouping it would split an already tiny set of channels. I leave that one first pointwise convolution dense.

The group count is now a real architectural knob. Larger `g` makes the pointwise layers cheaper, so at fixed complexity I can allocate more output channels. But larger `g` also means each individual grouped filter sees fewer input channels. That predicts a tradeoff: grouped pointwise layers should help at fixed MFLOPs because they buy width, but very large `g` can saturate or hurt because each filter's local view is too thin. The smallest models should benefit most from larger `g`, because they are most starved for channels.

For the implementation I put the channel permutation after the depthwise batch norm and before the second grouped pointwise layer. That is the placement that matters for communication: it occurs before the next grouped `1x1`, where cross-group information would otherwise be blocked.

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


class MobileUnit(nn.Module):
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


class MobileBackbone(nn.Module):
    STAGE_REPEATS = [4, 8, 4]
    STAGE_OUT = {
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
        if group not in self.STAGE_OUT or model_size not in self.STAGE_OUT[group]:
            raise ValueError("reference table supports group in {3, 8} and model_size in {0.5x,1.0x,1.5x,2.0x}")
        self.stage_out_channels = self.STAGE_OUT[group][model_size]
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        features = []
        for idxstage, numrepeat in enumerate(self.STAGE_REPEATS):
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                features.append(MobileUnit(
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
```

This lands on the architecture I was looking for: make the `1x1` channel mixing sparse enough to free budget for width, repair the information-flow damage with a channel permutation before the next grouped pointwise layer, keep the `3x3` spatial operation depthwise and narrow, and use concatenation only for the stride-2 channel-growing case. The method is not just "use depthwise convolution"; MobileNet already does that. The decisive move is to attack the pointwise bottleneck without permanently isolating the channel groups.
