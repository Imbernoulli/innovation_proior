## Channel Mixing Is Entangled

A convolutional layer combines spatial and channel information. If the input is
`X = [x^1, ..., x^{C'}]` and the `c`-th learned filter is
`v_c = [v_c^1, ..., v_c^{C'}]`, then the `c`-th output map is

```text
u_c = v_c * X = sum_{s=1}^{C'} v_c^s * x^s.
```

The sum across `s` is a channel composition, but it is not exposed as a separate object.
The same filter coefficients specify both the spatial pattern and the cross-channel mixture.
Each response is produced from a finite receptive field, so the channel composition at any
location is determined by a local patch of the input.

Once training fixes the convolution weights, this channel composition is the same for every
input.

## Deep Backbones Protect Information Flow

By 2016, strong image classifiers were built by stacking many convolutional transformations.
VGG-style depth showed that deeper hierarchies can improve representations, Inception-style
modules diversified the local transformations inside a stage, and Batch Normalization made
deep stacks easier to optimize by controlling activation statistics.

Residual networks changed the reliability of very deep training by adding an identity path:

```text
y = F(x) + x
```

or, when dimensions change, a learned projection shortcut in place of the literal identity.
The useful property is that the through-path remains simple while the residual branch learns
the nontrivial transformation.

## Existing Channel And Attention Tools

Several pre-existing tools touch nearby problems. A `1x1` convolution recombines channels
cheaply at each spatial location using fixed learned weights. Grouped and multi-branch
convolutions, including Inception and ResNeXt-style designs, enlarge the set of local
transformations and channel partitions.

Batch Normalization has a learned per-channel affine scale and shift, so it is structurally
close to channel modulation. Its scale and shift are learned parameters fixed at inference.

Attention and gating mechanisms provide another precedent. Recurrent visual attention and
spatial transformer networks bias where a model samples or aligns information in space.
Highway networks use gates to regulate transform-versus-carry flow through depth. Residual
attention networks add trunk-and-mask modules that produce spatial masks via auxiliary
subnetworks.

## Evaluation Settings

The natural comparison setting is image classification with the backbone and training recipe
held fixed. Large-scale evaluation uses ImageNet 2012: about 1.28 million training images,
50 thousand validation images, and 1000 classes, with top-1 and top-5 validation error.
The usual controls are identical data augmentation, optimizer, learning-rate schedule,
initialization, and classifier head across the baseline and the modified backbone.

Small-image checks use CIFAR-10 and CIFAR-100: 50 thousand training and 10 thousand test
images at `32x32` resolution. A CIFAR ResNet keeps three residual stages, uses `3x3`
convolutions, downsamples by stride at stage boundaries, global-average-pools the last
feature map, and ends with a linear classifier. The block being compared must keep the
same constructor and output interface so the surrounding ResNet code does not change.

## Code Scaffold

The local scaffold is a post-activation CIFAR residual block. The constructor signature,
`expansion`, shortcut behavior, and final `forward(x)` interface are fixed; the open part is
the residual-branch computation before it is added to the shortcut.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
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
        # TODO: the architecture we will design for this residual block

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # TODO: apply the new residual-branch computation here
        out += self.shortcut(x)
        return F.relu(out)
```
