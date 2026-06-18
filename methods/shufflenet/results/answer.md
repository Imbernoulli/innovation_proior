# ShuffleNet

ShuffleNet is a mobile-budget CNN block built from three choices:

- group the expensive `1x1` pointwise convolutions;
- insert a parameter-free channel permutation so stacked grouped pointwise layers do not isolate channel groups;
- use a depthwise `3x3` only on the narrow bottleneck.

For a stride-1 bottleneck with input/output width `c`, bottleneck width `m`, feature map `h x w`, and `g` pointwise groups:

```text
ResNet bottleneck:      hw(2cm + 9m^2)
ResNeXt bottleneck:     hw(2cm + 9m^2/g)
ShuffleNet bottleneck:  hw(2cm/g + 9m)
```

The stride-2 unit average-pools the shortcut and concatenates it with the learned branch, so the branch produces only `out_channels - in_channels`.

Reference-implementation note: the paper describes the channel shuffle immediately after the first grouped pointwise layer. The Megvii ShuffleNetV1 PyTorch reference applies the shuffle after the depthwise BN and before the second grouped pointwise layer. The code below follows that released reference.

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

Paper architecture table at `1.0x`: stage outputs are `[144, 288, 576]`, `[200, 400, 800]`, `[240, 480, 960]`, `[272, 544, 1088]`, and `[384, 768, 1536]` for `g = 1, 2, 3, 4, 8`, with complexities about `143M, 140M, 137M, 133M, 137M`. The released Megvii code specializes pretrained-friendly tables for `g=3` and `g=8`.
