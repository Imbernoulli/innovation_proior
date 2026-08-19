## Research question

Residual reformulation — fit `F(x) := H(x) - x` and output `F(x) + x` via a parameter-free
shortcut — has already unlocked networks over 100 layers deep, where plain feed-forward
convolutional nets used to hit a hard optimization wall (adding layers made *training* error
go up, not down, even though a deeper net strictly contains a shallower one as a special case:
copy the shallow net's layers and set the extra ones to identity). But the depth lever is not
yet clean past that point. A 1202-layer residual net (19.4M params) fits its training set
extremely well and still tests *worse* (7.93%) than a much smaller 110-layer residual net
(1.7M params, 6.61%) on CIFAR-10 — not an optimization failure this time, since training error
is low, so something about generalization at extreme depth. Separately, on the very deepest
residual nets tried so far, training loss is observed to fall painfully slowly at the start,
even though the shortcuts are supposed to make the identity solution trivial to reach. The
open question is how to design the internal structure of the repeating residual block — which
operations sit where, relative to the addition — to support optimization and generalization as
depth is pushed further, while adding no parameters and no real extra computation, and
changing nothing about the data pipeline, optimizer, classifier head, or initialization.

## Background

**The degradation phenomenon (the motivating diagnostic).** When a plain convolutional net is
made deeper, training error first saturates and then *rises*. This has been verified carefully
on CIFAR-10 and ImageNet: a 56-layer plain net has higher training error than a 20-layer one;
an ImageNet 34-layer plain net has higher training error than its 18-layer counterpart
throughout training. What makes this surprising rather than expected is that there is an
explicit solution by construction for the deeper net — copy the shallow net's layers and set
the extra ones to identity — that achieves the shallow net's training error, so the deeper
net's *representational* capacity is a strict superset of the shallow net's. The solver simply
fails to find a solution at least as good.

**Residual reformulation.** Rather than ask a stack of layers to fit a target mapping `H(x)`
directly, let it fit the residual `F(x) := H(x) - x`, so the block computes `F(x) + x` via a
parameter-free shortcut that adds the input back. If the optimal local mapping is near
identity, the solver only has to drive `F` toward zero — easy — instead of synthesizing
identity out of nonlinear layers. The unit as currently used is

```
y = F(x, {W_i}) + x ,     (identity shortcut, dims match)
x_next = ReLU(y) ,
```

where `F` is `conv-BN-ReLU-conv-BN`, followed by addition and then a final ReLU. Empirically
the learned residuals have small responses, supporting the view that the identity reference is
good preconditioning. This is what unlocked 100+-layer nets. When the channel count or spatial
size changes, the shortcut cannot be a bare identity: either zero-pad the extra channels
(parameter-free) or use a 1x1 convolution to project (parameter-carrying); identity shortcuts
with occasional 1x1 projection at shape-changing units are the standard recipe.

**Batch Normalization.** For each scalar feature, normalize over the mini-batch to zero mean
and unit variance, then apply a learned per-channel scale `gamma` and shift `beta`:

```
x_hat = (x - mean_B) / sqrt(var_B + eps) ,   y = gamma * x_hat + beta .
```

At inference, fixed population statistics (moving averages of mean/var collected in training)
replace the batch statistics. In the prevailing recipe BN is inserted immediately *before* the
nonlinearity, i.e. a weight layer's output is processed as `ReLU(BN(Wx))`. Besides accelerating
and stabilizing training, BN is observed to *regularize*: it injects mini-batch-dependent noise
into each activation and can partly substitute for dropout.

**ReLU and the backprop chain rule.** The rectifier `sigma(x) = max(0, x)` has derivative 1 for
positive pre-activations and 0 for negative ones, so wherever a ReLU sits on a path it can zero
out the gradient flowing back through that path. Gradients through a deep stack are computed by
the chain rule as a product of per-layer Jacobians; a product of many factors with magnitude
consistently below or above 1 vanishes or explodes, which is the classic obstruction to
training depth.

## Baselines

**Plain deep convolutional nets (VGG-style stacks).** Stack many 3x3 conv-BN-ReLU layers and a
classifier on top. Core idea: depth builds hierarchical features. Hits the degradation wall
described above past a certain depth.

**Original residual unit.** As written above: `x_next = ReLU(F(x) + x)` with `F` =
conv-BN-ReLU-conv-BN and an identity shortcut, the final ReLU applied *after* the element-wise
addition. Core idea: the additive shortcut preconditions the optimization toward identity,
letting 100+-layer nets train. On CIFAR-10, ResNet-110 reaches 6.61% test error; ResNet-1202
reaches 7.93%, worse despite fitting its training set well — attributed to overfitting on the
small dataset, since no other cause was identified.

**Highway Networks.** Replace the bare additive shortcut with a learned, data-dependent gate.
The block computes

```
y = H(x, W_H) . T(x, W_T) + x . (1 - T(x, W_T)) ,   T(x) = sigma(W_T x + b_T) ,
```

with `b_T` initialized to a negative value so the network starts biased toward "carry" (passing
`x` through). Core idea: let the network *learn* how much of each unit to transform vs. carry,
inspired by LSTM gates. On the datasets Highway was evaluated on, very deep highway networks
train successfully, though the gating mechanism has not yet been tested inside a residual-style
block at the depths reached by residual nets.

## Evaluation settings

- **CIFAR-10 and CIFAR-100**: 32x32 natural images, 10 and 100 classes, 50k train / 10k test.
  The standard recipe trains from scratch with SGD, mini-batch 128 (split across 2 GPUs unless
  noted), momentum 0.9, weight decay 1e-4, He initialization; learning rate starts at 0.1 and is
  divided by 10 at fixed iteration milestones, with a short low-lr warmup in the very deep
  cases. Augmentation is the light standard: 4-pixel zero-pad then random 32x32 crop, plus
  random horizontal flip. The CIFAR ResNet family is built from `n` residual units per stage
  over three stages of feature widths {16, 32, 64} (basic units, depth `6n+2`) or with
  bottleneck units (1x1-reduce, 3x3, 1x1-restore; depth `9n+2`), e.g. ResNet-20/56/110/164/1001.
- **Protocol for block comparisons**: hold the entire training recipe, data pipeline,
  initialization, and depth fixed; swap only the building block; report the **median of 5
  runs** on CIFAR to damp run-to-run variance. The metric is classification test error (%); the
  diagnostic curves are training loss and test error vs. training iteration.

## Code framework

The block plugs into a fixed ResNet backbone and the fixed training recipe above; the only
thing being designed is the internal structure of the repeating residual block. The substrate
that already exists: convolution / batch-norm / ReLU primitives, a block abstraction with a
constructor `(in_planes, planes, stride)`, a class attribute `expansion` mapping `planes` to
output channels, a `forward(x)` that must return `planes * expansion` channels, and a backbone
that stacks these blocks into stages and appends global average pooling + a linear classifier.
A block must handle the case where `stride != 1` or the input/output channel counts differ (the
shortcut then cannot be a bare identity). The internal arrangement of convs, the placement and
type of normalization and activation, and the form of the shortcut are exactly the open slots.

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
        # TODO: the internal block design -- how many convs, and where the
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
