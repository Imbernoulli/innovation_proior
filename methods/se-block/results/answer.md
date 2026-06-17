# Squeeze-and-Excitation Block

An SE block recalibrates a convolutional feature tensor channel by channel. For
`U = [u_1, ..., u_C] in R^{H x W x C}`, it computes:

```text
z_c = (1 / (H W)) sum_{i=1}^H sum_{j=1}^W u_c(i, j)
s   = sigmoid(W_2 ReLU(W_1 z))
x_tilde_c = s_c u_c
```

with `W_1 in R^{C/r x C}` and `W_2 in R^{C x C/r}`. The reduction ratio `r = 16`
is the standard setting in the released models.

Key constraints:

- The squeeze constant is `1/(H W)`, not a raw spatial sum.
- The final gate is a componentwise sigmoid, not a softmax, so channels do not
  compete for a fixed total mass.
- The added weight count is `2C^2/r` per block, or `(2/r) sum_s N_s C_s^2`
  across stages. Optional FC biases add only `C/r + C` per block.
- In a residual block, the recalibration is applied to the residual branch before
  adding the shortcut: `out = SE(F(x)) + shortcut(x)`.

## Reference-Faithful PyTorch Fill

The official Caffe models implement the excitation MLP as two `1x1` convolutions
on the `1x1` globally pooled tensor and fuse scale-plus-shortcut-add with an
`Axpy` layer. The mathematical specification and the common PyTorch reference
omit FC biases; the released Caffe prototxt may use Caffe's default convolution
biases in those two `1x1` excitation layers. That affects only the optional bias
count, not the squeeze -> excite -> scale -> shortcut-add computation below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        s = self.fc(z).view(b, c, 1, 1)
        return x * s


class CustomBlock(nn.Module):
    """Basic residual block with channel recalibration."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.se = SELayer(planes, reduction=16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)
```

The provided CIFAR scaffold uses channel widths `16, 32, 64`, so
`channels // 16` gives the same hidden widths `1, 2, 4` implied by the
reference implementation.
