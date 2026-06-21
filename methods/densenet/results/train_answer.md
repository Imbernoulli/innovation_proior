The central difficulty in training very deep convolutional networks is that both the forward signal and the backward gradient attenuate as they pass through long chains of nonlinear transformations. A plain stack of many layers exhibits the degradation problem: its training error is higher than that of a shallower counterpart, which means the failure is one of optimization rather than capacity. Residual networks address this by adding an identity shortcut around each block, giving gradients a clean additive path and making hundred-layer networks trainable. However, the shortcut output and the residual transformation are merged by summation, which forces old and new information into the same channels where they can interfere or overwrite each other. Moreover, every residual block owns a full-width set of weights, much of which is redundant because deep residual networks are highly over-parameterized; stochastic depth shows that whole layers can be dropped at random without harm, which tells us that direct non-adjacent connections are what help training, not the depth itself.

What is needed is a connectivity pattern that keeps short paths from every early layer to every late layer, preserves information explicitly rather than mixing it additively, and avoids devoting full-width parameters at each step to re-learning features that already exist. The way to do this is to make each layer read all preceding feature-maps and append its own small contribution to that stack. Formally, layer ℓ computes x_ℓ = H_ℓ([x_0, x_1, …, x_{ℓ−1}]), where [·] denotes channel-wise concatenation. This is the Dense Convolutional Network, or DenseNet.

Because the input to every layer is the concatenation of every earlier feature-map in the same block, preserved information is held intact and directly readable; it is never overwritten or summed away. Each layer therefore does not need to re-carry the entire state forward. Instead, it only adds a small fixed number k of new feature-maps, called the growth rate, to a shared collective state. With k kept small, the quadratic growth in connections is offset by the tiny per-layer output width, and the network becomes parameter-efficient through feature reuse rather than redundancy. A 3×3 convolution from c_in to c_out costs roughly 9 c_in c_out parameters, so an L-layer block costs on the order of k squared L squared parameters, but for k = 12 versus a conventional width of 256 the coefficient is about 1/455, so the total parameter count can still be modest despite the dense connectivity.

Concatenation requires equal spatial size, so the network is divided into dense blocks, each operating at one resolution with full dense connectivity inside, and transition layers that perform batch normalization, 1×1 convolution, and 2×2 average pooling to move from one resolution to the next. To prevent the growing input width from making the 3×3 convolutions expensive, each layer first applies a 1×1 bottleneck convolution that squeezes the concatenated input down to 4k channels before the 3×3 convolution emits k new maps; this is the bottleneck variant. The transitions can also compress the channel count by a factor θ, typically 0.5, so that the next block starts from a narrower state after the 1×1 mixing layer selects and recombines the most useful information. The combination of bottleneck layers and compressed transitions is DenseNet-BC, the most efficient standard form.

Training remains easy because an early feature-map is literally a channel slice of every later layer's input within the same block, so gradients reach early layers without passing through a long chain of convolutions. Across blocks, only the transition layers separate resolutions, keeping the supervision path short. The final batch-normalized features are globally average-pooled and fed to a linear classifier, with every layer enjoying near-direct supervision through the dense block connections.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """H_ell: 1x1 bottleneck (-> 4k) then 3x3 conv (-> k new feature-maps)."""
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, prev_features):
        x = torch.cat(prev_features, 1)               # read all preceding maps
        x = self.conv1(self.relu1(self.norm1(x)))     # 1x1 -> 4k
        x = self.conv2(self.relu2(self.norm2(x)))     # 3x3 -> k
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


class DenseBlock(nn.ModuleDict):
    """Dense connectivity at one resolution; output width = num_input + num_layers*k."""
    def __init__(self, num_layers, num_input_features, growth_rate,
                 bn_size=4, drop_rate=0.0):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            features.append(layer(features))
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    """1x1 conv (compress by theta) + 2x2 average pool (downsample)."""
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    """DenseNet-BC. block_config (6,12,24,16) with growth_rate 32 = DenseNet-121."""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0,
                 compression=0.5, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv0",
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.features.add_module("denseblock%d" % (i + 1),
                DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                out = int(num_features * compression)
                self.features.add_module("transition%d" % (i + 1),
                    Transition(num_features, out))
                num_features = out
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def densenet121(**kwargs):
    return DenseNet(32, (6, 12, 24, 16), 64, **kwargs)
```
