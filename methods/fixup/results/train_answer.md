A deep residual network can be written as x_l = x_0 + sum_{i<l} F_i(x_i), so every block adds its output to a running sum carried forward by the skip connection. That additive structure makes the activation variance grow monotonically with depth: at initialization E[F_l(x_l)|x_l] = 0, so Var[x_{l+1}] = E[Var[F_l(x_l)|x_l]] + Var[x_l]. With a standard He-style variance-preserving initialization, each branch is tuned so that Var[F_l(x_l)|x_l] is roughly Var[x_l], which gives Var[x_{l+1}] ~ 2 Var[x_l]. The variance therefore doubles every block and the logits at the top of a deep net are exponentially large. Because a normalization-free residual net built from bias-free convolutions, linear layers and ReLUs is positively homogeneous of degree one, that logit blowup can be turned into a lower bound on the gradient norm of the network's homogeneous weight sets: the first SGD step is catastrophic at any useful learning rate. Prior attempts mostly fight the symptom rather than the cause. Scaling the recurrence by sqrt(1/2) keeps activation variance at O(1), but it multiplies every earlier branch by a geometrically decaying factor and effectively turns off the deep residual functions. LSUV orthogonalizes and rescales each layer using a calibration batch, so it controls per-layer scale but remains data-dependent and still misses how the L branches act together. Near-zero residual initialization helps empirically, but zeroing the last layer without rescaling the others leaves the post-update branch contribution at O(1), so L of them again blow up the output. What is missing is a constraint on the update to the whole network function, not just on per-layer activation variance.

The method I propose is Fixup, short for fixed-update initialization. The goal is to make one SGD step change the network function by Theta(eta), independent of depth. To first order in eta, the function update across all residual branches is Delta f(x_0) = -eta J (partial l / partial z) + O(eta^2), where J is a sum over branches and layers of symmetric positive-semidefinite matrices. Because the matrices are PSD, their traces add, and the size of the update scales linearly with the number of residual branches L. The branches update the network in sync, not in opposition. Therefore each branch must contribute only Theta(eta/L) to the total function update. Studying a scalar caricature of an m-layer branch, F(x) = (prod_i a_i) x, one gradient step changes the branch output by Delta F(x) = -eta (partial l / partial F) F(x)^2 sum_i 1/a_i^2. The sum is dominated by the smallest layer scale A = min_k a_k, so Delta F is Theta(eta F(x)^2 / A^2). Setting this to Theta(eta/L) gives the constraint (prod_{k != j} a_k) x = Theta(1/sqrt(L)) for j = argmin_k a_k. The clean symmetric solution is to scale every layer in the branch by the same factor a = L^{-1/(2m-2)}. For the common two-conv residual branch, m = 2, the scaling is L^{-1/2}; for a three-conv bottleneck it is L^{-1/4}.

Fixup applies three rules. First, zero-initialize the last layer of every residual branch and the final classification layer. Zeroing the last conv makes each branch start as the zero function, so the entire residual stack initially behaves as a pure identity path: there is no variance explosion at init and no random branch output that SGD has to unlearn. Zeroing the classifier makes the initial logits exactly zero, which removes the positive logit-scale term in the activation-side gradient lower bound. Second, scale every other weight layer inside each branch by the L^{-1/(2m-2)} rule derived above, starting from a standard He initialization. This scaling is essential: without it, the first gradient step would revive the zeroed layer at O(1) scale and L branches would again add to an O(1) or larger output change. With the scaling, the post-update branch contribution is O(1/L), so L branches sum to O(1). Third, add a small number of scalar parameters to recover what normalization's shift and scale were doing: a scalar bias before each convolution, linear layer and element-wise activation restores the recentering that batch normalization's beta provided, and a scalar multiplier initialized to one on each branch restores the weight-norm dynamics that normalization's scale invariance gave for free. Under weight decay the weight magnitude shrinks while the multiplier absorbs the scale, giving an effective learning rate schedule close to that of a normalized net. These scalars cost O(1) per branch rather than O(channels) for per-channel normalization parameters, and they keep the convolutional weights themselves bias-free.

```python
import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
        out += identity
        out = self.relu(out)
        return out


class FixupResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                fan_out = m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:])
                nn.init.normal_(
                    m.conv1.weight, mean=0,
                    std=np.sqrt(2.0 / fan_out) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)
        seq = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            seq.append(block(planes, planes))
        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x + self.bias2)
        return x


def fixup_resnet56(**kwargs):
    return FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
```
