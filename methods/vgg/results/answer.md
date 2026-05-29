# VGG — very deep ConvNets with small 3×3 filters

## Problem

Isolate **network depth** as a controlled variable in large-scale image classification and find out
whether increasing it — holding the rest of the recipe fixed — drives accuracy. The obstacle is that the
ConvNets of the day build early layers from large filters at large strides, which cannot be stacked deep
without exploding the parameter count and collapsing spatial resolution.

## Key idea

Use **only 3×3 convolutions** (stride 1, padding 1) throughout the network, plus a few 1×1 convolutions
in one variant. Stacking small filters is what unlocks depth:

- **Effective receptive field.** A stack of two 3×3 stride-1 convs sees a 5×5 input patch; three see 7×7;
  in general k stacked 3×3 layers see a (2k+1)×(2k+1) patch. Any large filter's reach can be assembled
  from a stack of 3×3 layers.
- **Fewer parameters.** For C input/output channels, one 7×7 layer costs 7²C² = 49C² weights; three
  stacked 3×3 layers cost 3·(3²C²) = 27C² — the single 7×7 needs ~81% more for the same receptive field.
  (Two 3×3 = 18C² vs one 5×5 = 25C², ~39% more.)
- **More nonlinearity.** The three-layer stack applies three ReLUs over that 7×7 field instead of one,
  giving a more discriminative decision function — a factored 7×7 filter with nonlinearity between factors.

This makes depth cheap: each added 3×3 layer barely moves the parameter count, because most weights live
in the first fully-connected layer: 512·7·7·4096 = 102,760,448 weights, about 102.8M. The network can be
grown from 11 to 19 weight layers as a clean single knob.

## Architecture

Input 224×224 RGB (mean-RGB subtracted). Conv body = stacks of 3×3 conv (stride 1, pad 1) + ReLU,
separated by five 2×2 stride-2 max-pools, so spatial size follows 224 → 112 → 56 → 28 → 14 → 7; channel
width 64 → 128 → 256 → 512, doubling after each pool, capped at 512. Head = FC-4096, FC-4096, FC-1000,
softmax, with dropout 0.5 on the first two FC layers. Local Response Normalization is kept only as a
shallow control variant, not as part of the deeper family.

The configuration family (identical except depth):

| Config | Weight layers | Notes |
|--------|---------------|-------|
| A | 11 | baseline, trainable from scratch |
| A-LRN | 11 | A + LRN control |
| B | 13 | extra 3×3 in first two blocks |
| C | 16 | extra layers are 1×1 (adds nonlinearity, not receptive field) |
| D | 16 | extra layers are 3×3 (adds nonlinearity AND spatial context) |
| E | 19 | a fourth 3×3 in each of the last three blocks |

Parameters: A/A-LRN 133M, B 133M, C 134M, D 138M, E 144M. C vs D isolates nonlinearity from spatial
context.

## Training and evaluation

- **Objective/optimizer:** softmax cross-entropy, mini-batch SGD, batch 256, momentum 0.9, L2 weight
  decay 5×10⁻⁴, dropout 0.5 on first two FC. LR 10⁻² → ÷10 on validation plateau (3 times), ~370K iters.
- **Initialization:** train shallow A from random init (zero-mean Gaussian, std 10⁻², biases 0); warm-start
  deeper nets from A's first four conv and last three FC layers (rest random) to keep deep-net gradients
  stable. The standalone module below uses fan-scaled conv initialization when no warm-start weights are
  loaded.
- **Multi-scale training (scale jittering):** crop 224×224 from images rescaled to smallest side S; draw S
  uniformly from [256, 512] per image so one model is robust across object scales. Trained by fine-tuning a
  fixed-S model (S=384, itself warm-started from S=256 at LR 10⁻³).
- **Dense evaluation:** re-express the FC head as convolutions (first FC → 7×7 conv, last two FC → 1×1
  convs) so the net is fully convolutional; apply it to the whole image at test scale Q, producing a
  spatial class-score map, and spatially average it (plus the horizontally-flipped image). Average over
  Q ∈ {S−32, S, S+32} for fixed-scale training, or Q ∈ {S_min, 0.5(S_min+S_max), S_max} for scale-jittered
  training. Optionally average with multi-crop evaluation (complementary border handling).

## Code

The reusable implementation keeps the family in a config table, builds the body with one layer builder,
and keeps pooling/classification inside the module.

```python
from typing import Union, cast

import torch
import torch.nn as nn


cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
          512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
          512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg: list[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            out_channels = cast(int, v)
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_vgg(cfg: str = "D", batch_norm: bool = False, **kwargs) -> VGG:
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)


def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

loss_fn = nn.CrossEntropyLoss()


def train_step(model, images, labels, optimizer):
    optimizer.zero_grad()
    loss = loss_fn(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def to_fully_convolutional(model, num_classes=1000):
    """Re-express the FC head as convolutions for dense whole-image evaluation."""
    fc1, fc2, fc3 = [m for m in model.classifier if isinstance(m, nn.Linear)]
    kwargs = {"device": fc1.weight.device, "dtype": fc1.weight.dtype}
    conv1 = nn.Conv2d(512, 4096, kernel_size=7, **kwargs)
    conv2 = nn.Conv2d(4096, 4096, kernel_size=1, **kwargs)
    conv3 = nn.Conv2d(4096, num_classes, kernel_size=1, **kwargs)
    with torch.no_grad():
        conv1.weight.copy_(fc1.weight.view(4096, 512, 7, 7))
        conv1.bias.copy_(fc1.bias)
        conv2.weight.copy_(fc2.weight.view(4096, 4096, 1, 1))
        conv2.bias.copy_(fc2.bias)
        conv3.weight.copy_(fc3.weight.view(num_classes, 4096, 1, 1))
        conv3.bias.copy_(fc3.bias)
    return nn.Sequential(model.features, conv1, nn.ReLU(inplace=True),
                         conv2, nn.ReLU(inplace=True), conv3)


def predict_dense(model, image, num_classes=1000):
    net = to_fully_convolutional(model, num_classes)
    net.eval()
    with torch.no_grad():
        score_map = net(image)
    return score_map.mean(dim=(2, 3))     # spatial average of the class-score map -> one vector
```
