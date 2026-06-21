The question is whether image recognition networks can keep gaining accuracy simply by adding many more layers. In practice they cannot: once a plain convolutional stack passes about twenty to thirty layers, its training error stops falling and begins to rise with depth. That rise is not overfitting, because the test error is not the only thing moving upward; the training error itself gets worse. It is also not vanishing gradients, because He initialization and Batch Normalization already keep both forward activations and backward gradient norms at healthy levels. The failure is therefore an optimization or conditioning failure. A deeper network can always, by construction, reproduce a shallower one by copying its weights into the lower layers and setting every added layer to the identity, so a solution at least as good provably exists inside the parameter space of the deeper model. The puzzle is that stochastic gradient descent cannot find that solution, or anything close to it, in feasible time.

The reason identity mappings are hard to learn in a plain stack is that a sequence of convolutions, batch normalizations, and ReLUs must conspire to reproduce its input exactly. The identity sits at a specific, non-trivial point in weight space, while small random initialization and weight decay naturally pull weights toward zero, which produces a small random transform rather than the identity. So the parameterization itself makes the most sensible deep solutions difficult to reach. The fix is a reparameterization: instead of asking a block to compute the desired mapping H(x) directly, ask it to compute the residual F(x) = H(x) − x and recover the output by adding the input back through a shortcut. This turns the hard identity-mapping problem into the easy zero-function problem, because zero weights and zero batch-normalization shifts make the residual branch output zero. When the optimal mapping is near the identity, the optimizer only has to learn a small perturbation on top of x, which is much better conditioned than learning the whole mapping from scratch.

I propose ResNet. Each ResNet block is defined by one equation: y = F(x) + x. The shortcut that carries x is a parameter-free identity connection whenever the input and output shapes match, so it adds no parameters and no meaningful computation. That makes the comparison with a plain network perfectly controlled at the same depth, width, parameter count, and FLOPs. When shapes do not match, for example at stage transitions where the spatial map is halved and the channel count is doubled, a small 1×1 convolution with batch normalization is used on the shortcut only to align dimensions. The final ReLU is applied after the addition, so the residual branch can still represent negative corrections; putting the ReLU before the addition would clamp the residual to be non-negative and break the residual idea.

This identity shortcut is deliberately different from a gated shortcut such as the one used in Highway Networks. A data-dependent gate can close the shortcut path, in which case that layer reverts to an ordinary non-residual transform and again faces the original hard problem of learning identity-like mappings. The ResNet identity shortcut is always open, always passes all information, and never introduces extra parameters or computation. The fixed identity constraint is the source of the conditioning benefit, not a limitation.

To build very deep networks affordably, ResNet uses a bottleneck block for the 50-, 101-, and 152-layer variants. The bottleneck consists of three convolutions: a 1×1 convolution that reduces channels, a 3×3 convolution that performs spatial processing at the reduced width, and a 1×1 convolution that restores the channels with an expansion factor of four. Because the expensive 3×3 convolution runs at the reduced channel count, one three-layer bottleneck block costs about the same as one older two-layer 3×3 block while adding an extra weight layer. The identity shortcut then connects the two high-dimensional ends of the bottleneck; using a projection shortcut there would roughly double the block's cost, so the free identity shortcut is essential for making extreme depth economical. The 152-layer ResNet ends up lighter in FLOPs than VGG-19 while being eight times deeper.

The architecture keeps the VGG-style stage rule: the same spatial map size uses the same number of filters, and halving the spatial size doubles the channels. The network begins with a 7×7 convolution of 64 channels with stride 2, followed by a 3×3 max-pooling layer with stride 2, then four stages with block counts chosen to give 18, 34, 50, 101, or 152 layers. The head is a global average pooling layer followed by a single fully connected classification layer, avoiding the large fully connected head that dominates VGG's parameter budget. The training recipe is the standard mature recipe: batch normalization after every convolution and before activation, He initialization, SGD with momentum 0.9 and weight decay 1e-4, learning rate starting at 0.1 and divided by 10 when error plateaus, and the usual ImageNet scale jitter, crop, flip, and color augmentation. Dropout is omitted because batch normalization already provides regularization.

The expected experimental signature is that the deeper plain network shows degradation, while the deeper ResNet shows lower training error than its shallower residual counterpart and therefore better validation accuracy. The learned residual responses should also be small on average, confirming that the blocks are acting as mild refinements around the identity reference.

```python
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
```
