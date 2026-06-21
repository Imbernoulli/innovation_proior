# Context: building blocks for very deep image classifiers (circa 2015-2016)

## Research question

Stacking more layers is the lever that has driven the recent jump in image-classification
accuracy, and the natural hypothesis is "deeper is strictly better": a deeper network contains
a shallower one as a special case (set the extra layers to identity), so it should be able to
match the shallower net and then improve on it. Reality contradicts this. Past a point, adding
layers makes a plain feed-forward convolutional net *worse*, and crucially worse on the
*training* set, not just the test set — so it is not overfitting, it is an optimization
failure. Residual reformulation (below) pushes the usable depth dramatically — networks of
100+ layers train and generalize well — but on CIFAR-10, a 1202-layer residual net can fit the
training set extremely well yet test worse than a much smaller 110-layer residual net. The
question is how to design the internal structure of the repeating residual block to support
optimization and generalization as depth is pushed further, while adding no parameters and no
real computation and changing nothing about the data pipeline, optimizer, classifier head, or
initialization.

## Background

The field state is dominated by a few load-bearing facts and tools.

**The degradation phenomenon (the motivating diagnostic).** When a plain convolutional net is
made deeper, training error first saturates and then *rises*. He et al. (2016a) verify this
carefully on CIFAR-10 and ImageNet: a 56-layer plain net has higher training error than a
20-layer one; an ImageNet 34-layer plain net has higher training error than its 18-layer
counterpart throughout training. The argument that makes this surprising rather than expected:
there is an explicit solution by construction for the deeper net — copy the shallow net's
layers and set the extra ones to identity — that achieves the shallow net's training error, so
the deeper net's *representational* capacity is a strict superset. The solver simply fails to
find a solution at least as good.

**Residual reformulation.** Rather than ask a stack of layers to fit a target mapping
`H(x)` directly, let it fit the *residual* `F(x) := H(x) − x`, so the block computes
`F(x) + x` via a parameter-free shortcut that adds the input back. If the optimal local
mapping is near identity, the solver only has to drive `F` toward zero — easy — instead of
synthesizing identity out of nonlinear layers. He et al. (2016a) realize this with the unit

```
y = F(x, {W_i}) + x ,     (identity shortcut, dims match)
x_next = ReLU(y) ,
```

where `F` is `conv-BN-ReLU-conv-BN`, followed by addition and then the final ReLU. Empirically
the learned residuals have small responses, supporting the view that the identity reference is
good preconditioning. This is what unlocked 100+-layer nets. When the channel count or spatial
size changes, the shortcut cannot be a bare identity: either zero-pad the extra channels
(parameter-free, "option A") or use a 1x1 convolution to project (parameter-carrying, "option
B"); identity/occasional projection were found sufficient, while projection on every shortcut
was not treated as essential.

**Batch Normalization (Ioffe & Szegedy 2015).** For each scalar feature, normalize over the
mini-batch to zero mean and unit variance, then apply a learned per-channel scale `γ` and
shift `β`:

```
x_hat = (x − mean_B) / sqrt(var_B + eps) ,   y = γ * x_hat + β .
```

At inference, fixed population statistics (moving averages of mean/var collected in training)
replace the batch statistics. In the prevailing recipe BN is inserted immediately *before* the
nonlinearity, i.e. the layer computes `ReLU(BN(W x))`. Besides accelerating and stabilizing
training, BN is observed to *regularize*: it injects mini-batch-dependent noise into each
activation and can partly substitute for dropout.

**ReLU and the backprop chain rule.** The rectifier `σ(x)=max(0,x)` (Nair & Hinton 2010) has
derivative 1 for positive pre-activations and 0 for negative ones, so wherever a ReLU sits on
a path it can zero out the gradient flowing back through that path. Gradients through a deep
stack are computed by the chain rule (LeCun et al. 1989) as a product of per-layer Jacobians;
a product of many factors with magnitude consistently below or above 1 vanishes or explodes,
which is the classic obstruction to training depth.

## Baselines

**Plain deep convolutional nets (e.g. VGG-style stacks, Simonyan & Zisserman 2015).** Stack
many 3x3 conv-BN-ReLU layers and a classifier on top. Core idea: depth builds hierarchical
features.

**Original residual unit (He et al. 2016a).** As written above: `x_next = ReLU(F(x) + x)`
with `F` = conv-BN-ReLU-conv-BN and an identity shortcut, where the final ReLU is applied
*after* the element-wise addition. Core idea: the additive shortcut preconditions the
optimization toward identity, letting 100+-layer nets train. On CIFAR-10, a 1202-layer net
(19.4M params) reaches 7.93% — compared with 6.61% for a 110-layer net (1.7M params); He et
al. (2016a) attribute this to overfitting on the small dataset.

**Highway Networks (Srivastava, Greff & Schmidhuber 2015).** Replace the bare additive
shortcut with a learned, data-dependent gate. The block computes

```
y = H(x, W_H) · T(x, W_T) + x · (1 − T(x, W_T)) ,   T(x) = σ(W_T x + b_T) ,
```

with `b_T` initialized to a negative value so the network starts biased toward "carry"
(passing `x` through). Core idea: let the network *learn* how much of each unit to transform
vs. carry, inspired by LSTM gates.

## Evaluation settings

The natural yardsticks already in use for block/architecture design at this time:

- **CIFAR-10 and CIFAR-100** (Krizhevsky 2009): 32x32 natural images, 10 and 100 classes,
  50k train / 10k test. The standard recipe trains from scratch with SGD, mini-batch 128
  (split across 2 GPUs), momentum 0.9, weight decay 1e-4, He initialization; learning rate
  starts at 0.1 and is divided by 10 at fixed iteration milestones (32k and 48k iterations in
  the CIFAR residual-net recipe, with a 0.01 warmup for roughly the first 400 iterations in
  the very deep cases). Augmentation is the
  light standard: 4-pixel zero-pad then random 32x32 crop, plus random horizontal flip. The
  CIFAR ResNet family is built from `n` residual units per stage over three stages of feature
  widths {16, 32, 64} (basic) or with bottleneck units; depth is `6n+2` (basic) or `9n+2`
  (bottleneck), e.g. ResNet-20/56/110/164/1001.
- **ImageNet (ILSVRC 2012)** (Russakovsky et al. 2015): 1.28M training images, 1000 classes;
  standard scale/crop augmentation, mini-batch 256 over 8 GPUs, lr 0.1 divided by 10 at epoch
  milestones, same weight decay / momentum / initialization. ResNet-50/101/152/200 with
  bottleneck units; single-crop top-1 / top-5 validation error is the metric.
- **Protocol** for block comparisons: hold the entire training recipe, data pipeline,
  initialization, and depth fixed; swap only the building block; report the median of several
  runs on CIFAR to damp run-to-run variance. The metric is classification error / accuracy;
  the diagnostic curves are training loss and test error vs. iteration.

## Code framework

The block plugs into a fixed ResNet backbone and the fixed training recipe above; the only
thing being designed is the internal structure of the repeating residual block. The substrate
that already exists: convolution / batch-norm / ReLU primitives, a block abstraction with a
constructor `(in_planes, planes, stride)`, a class attribute `expansion` mapping `planes` to
output channels, a `forward(x)` that must return `planes * expansion` channels, and a backbone
that stacks these blocks into stages and appends global average pooling + a linear classifier.
A block must handle the case where `stride != 1` or the input/output channel counts differ
(the shortcut then cannot be a bare identity). The internal arrangement of convs, the
placement and type of normalization and activation, and the form of the shortcut are exactly
the open slots.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    """A residual block for the fixed ResNet backbone.

    Contract the backbone relies on:
      - constructor signature (in_planes, planes, stride)
      - class attribute `expansion` (output channels = planes * expansion)
      - forward(x) returns a tensor with planes * expansion channels
      - the shortcut must cope with stride != 1 or a channel-count change
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # Primitives that already exist: Conv2d, BatchNorm2d, ReLU.
        # TODO: the internal block design — how many convs, and where the
        #       normalization / activation sit relative to them and to the
        #       element-wise addition; and how the shortcut carries x when the
        #       dimensions change. This arrangement is what we will derive.
        pass

    def forward(self, x):
        # TODO: compute the residual branch and the shortcut, then combine them
        #       and return planes * expansion channels.
        pass


# Existing backbone the block plugs into (fixed; not what we are designing).
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.stage1 = self._make_stage(block, 16, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, 32, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_stage(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# Existing training loop (fixed): SGD, step learning-rate schedule, standard augmentation.
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
```
