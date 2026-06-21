Convolutional networks have just taken over large-scale image classification — the 2012 net buried the hand-crafted-feature pipelines on ImageNet, and the 2013 winners pushed the error down further — and everyone is now tweaking that same recipe. But when I look at what the successful tweaks actually changed, they all touch the early *spatial sampling* and the *test-time evaluation*: the 2013 winners shrank the first convolutional layer from an 11×11 filter at stride 4 to a 7×7 at stride 2, and they evaluated densely over the whole image at multiple scales. Nobody touched the one variable I keep coming back to. All of these networks are about the same depth — eight weight layers, give or take. The clean question has gone unasked: if I hold the entire recipe fixed and just keep stacking more layers, does accuracy keep climbing? Is depth, by itself, the lever?

The reason that question is hard to answer is that "make it deeper" is not a free knob with the layer design of the day. An 11×11×3 filter at stride 4 already throws away three of every four pixels in each direction in the very first operation, and a single such layer carries $11^2 C^2 = 121\,C^2$ weights for $C$ channels in and out. If I try to build depth out of layers like that, the parameter count detonates and the spatial map collapses within a couple of pools. So depth and filter design are tangled together, and I cannot study the first without first fixing the second. I want depth as a single controlled variable — a family of nets identical in everything but the number of layers — and the existing big-filter, big-stride substrate simply cannot be stacked to depth 16 or 19 to provide it.

I propose VGG: a family of very deep ConvNets built from nothing larger than a $3\times 3$ convolution. The move that unlocks everything is to refuse any filter bigger than $3\times 3$ (stride 1, padding 1 so each conv preserves spatial size and only pooling ever downsamples), because $3\times 3$ is the smallest filter that still captures left/right, up/down and center — a $1\times 1$ filter only mixes channels at a point. The first worry is that a $3\times 3$ filter sees almost nothing, so I trace the receptive field through a stack instead of hand-waving. With stride-1 convs and no pooling between them, a unit in the first layer sees a $3\times 3$ patch; a unit in the second sees a $3\times 3$ neighborhood of first-layer units whose outer members each reach one pixel further out, giving $3 + (3-1) = 5$, a $5\times 5$ patch; a third layer gives $5 + (3-1) = 7$. In general $k$ stacked $3\times 3$ layers see a $(2k+1)\times(2k+1)$ patch, so three give the reach of a $7\times 7$ filter and five give $11\times 11$. The reach is not lost; it is assembled.

That assembly is strictly the better deal, and two counts show why. Counting parameters for $C$ channels in and out, one $7\times 7$ layer costs $7^2 C^2 = 49\,C^2$ weights, while three stacked $3\times 3$ layers cost $3\cdot(3^2 C^2) = 27\,C^2$ — the single big filter needs $49/27 \approx 1.81$ times as many, about 81% more, for the *same* receptive field. The same arithmetic gives one $5\times 5$ at $25\,C^2$ against two $3\times 3$ at $18\,C^2$, roughly 39% more. So stacking small filters *saves* parameters, the opposite of what I feared. The second gain is the one that excites me: the single $7\times 7$ layer applies one ReLU to its field, while the three-layer stack applies three ReLUs over the same field. A $7\times 7$ filter is forced to be one linear map of its patch; the three-$3\times 3$ stack is a *factored* $7\times 7$ with a nonlinearity injected between the factors — a structural regularizer (fewer, factored parameters) that simultaneously buys a more discriminative, more nonlinear decision function. More expressiveness and fewer parameters at once; not a tradeoff. This dissolves the original tangle: each added $3\times 3$ stride-1 pad-1 layer is cheap and harmless to resolution, so depth finally becomes a free knob.

The skeleton is then fixed and identical across the family so that only depth varies. Input is a $224\times 224$ RGB crop with the training-set mean RGB subtracted. The body is stacks of $3\times 3$ conv (stride 1, pad 1, ReLU after each), separated by five $2\times 2$ stride-2 max-pools — the minimal pool that halves resolution — so the spatial size follows $224 \to 112 \to 56 \to 28 \to 14 \to 7$, a $2^5 = 32$ reduction to a $7\times 7$ map. Channel width starts at 64 and doubles after each pool, $64 \to 128 \to 256 \to 512$, capped at 512 so the doubling roughly compensates the halving resolution and per-layer compute stays balanced. On top sit three fully-connected layers — 4096, 4096, and 1000 — then softmax, with ReLU on every hidden layer. The head never changes; only the conv body's depth does. The family climbs by adding $3\times 3$ layers and nothing else: A at 11 weight layers (8 conv + 3 FC), B at 13 (a second $3\times 3$ in the first two blocks), D at 16 (a third $3\times 3$ in the last three blocks), E at 19 (a fourth in each of those blocks). To confirm depth does not blow up the count, note where the weights live — not in the cheap conv stacks but in the first FC layer that flattens the $7\times 7\times 512$ map into 4096: $512\cdot 7\cdot 7\cdot 4096 = 102{,}760{,}448$ weights, about 102.8M, the bulk of the model. Against that, the whole family runs 133M, 133M, 138M, 144M from A to E; eight extra layers move the total by about 8%, nearly all of it irrelevant to depth.

Two orthogonal probes ride along inside the family by construction. First, a net C that matches D's 16 layers but makes its three extra late-block layers $1\times 1$ instead of $3\times 3$: a $1\times 1$ conv keeps the receptive field at a single point — a linear channel projection — but followed by a ReLU it adds a nonlinear stage *without touching spatial context*. C and D have identical depth and identical nonlinearity count, so they differ only in whether those extra layers see a $3\times 3$ neighborhood or a point; if D beats C, spatial context matters beyond raw nonlinearity. Second, I keep one shallow variant A-LRN with Local Response Normalization, the lateral inhibition from the 2012 net, purely to check whether it earns its memory and compute; my expectation is it will not help on this data, so I drop it from the deeper nets and keep them clean.

The remaining wall is whether a 19-layer net can be trained at all, since deep nets are notoriously sensitive to initialization — bad starting weights make the signal vanish or explode through that many layers and learning stalls. The family supplies its own ladder: A is shallow enough to train from a zero-mean Gaussian with std $10^{-2}$ and zero biases, and once A is trained I warm-start each deeper net by initializing its first four conv layers and its last three FC layers from A's learned weights, leaving the middle random and unfrozen. The well-conditioned ends keep gradients sane while the random middle catches up. In standalone code, when no warm-start is loaded, I encode the same stability principle as a fan-scaled (Kaiming) conv initializer so activation and gradient magnitudes neither shrink nor grow systematically through the stack. The optimization is otherwise deliberately conventional, because the architecture is the only experimental variable: minimize softmax cross-entropy (multinomial logistic regression) by mini-batch SGD with batch 256 and momentum 0.9, regularize the FC-heavy model with L2 weight decay $5\times 10^{-4}$ and dropout 0.5 on the first two FC layers, and start the learning rate at $10^{-2}$, dividing it by 10 whenever validation accuracy plateaus (three times over roughly 370K iterations, about 74 epochs).

Scale is handled at both ends. Let $S$ be the smallest side of the isotropically rescaled training image; the $224\times 224$ crop spans the whole image when $S=224$ and a small object-part window when $S \gg 224$, so $S$ sets the apparent object scale. Rather than commit to one scale, I jitter it — drawing $S$ uniformly from $[256,512]$ per image (plus random horizontal flip and RGB color shift) — so a single net learns objects across a span of sizes; to get there cheaply I fine-tune all layers of a fixed-$S$ model ($S=384$, itself warm-started from the $S=256$ net at the lower rate $10^{-3}$). At test time the only thing demanding a fixed $224\times 224$ input is the FC head, and a fully-connected layer over a fixed spatial extent *is* a convolution: the first FC consumes the $7\times 7$ map so it is a $7\times 7$ conv, and the next two consume $1\times 1$ inputs so they are $1\times 1$ convs. Rewriting the head this way makes the net fully convolutional, so I apply it to the whole uncropped image at test scale $Q$ in a single shared-computation pass and get a spatial *map* of class scores — one 1000-vector per position — instead of re-running the net per crop. Spatially averaging that map (and the flipped image) collapses it to one prediction. Dense evaluation and multi-crop evaluation are complementary rather than rival: a cropped window is zero-padded at its edges, whereas the dense pass fills border positions with real neighboring content, so averaging the two can help. And because the net was trained with scale jitter, I average posteriors over several test scales — $Q \in \{S-32,\,S,\,S+32\}$ for a fixed-$S$ model, or $Q \in \{S_\text{min},\,0.5(S_\text{min}+S_\text{max}),\,S_\text{max}\}$ for the jittered one.

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
