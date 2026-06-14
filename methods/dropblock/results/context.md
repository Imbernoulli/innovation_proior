# Context: regularizing convolutional layers with structured noise (circa 2017-2018)

## Research question

Modern convolutional image classifiers are massively over-parameterized — tens to
hundreds of millions of weights — and they overfit unless trained with a lot of
regularization and injected noise. For fully connected layers the workhorse is dropout:
zero each unit independently with some probability, which behaves like averaging an
exponential ensemble of thinned sub-networks and breaks up co-adaptation between feature
detectors. The pain point is that this single most successful noise-injection trick stops
paying off when it is moved into the *convolutional* part of a network: empirically,
recent convolutional architectures (the residual and inception families and their
descendants) rarely place dropout on their convolutional feature maps, and when it is put
there it gives little improvement. So the question is what kind of injected noise actually
regularizes a convolutional feature map — a noise process that genuinely removes
information the network would otherwise lean on, rather than one the network can route
around, while staying general enough to drop into any convolutional layer, requiring no
change at inference beyond the standard rescaling, and not being so fragile that its
strength has to be hand-tuned per architecture.

## Background

The dominant recipe for training a large image classifier is heavy regularization layered
on top of a strong architecture: weight decay, data augmentation (random crops, flips,
photometric jitter), batch normalization, and — historically — dropout. Several facts
about how these interact with convolutional layers are established before any new method
is on the table.

**Spatial correlation of convolutional activations.** A convolutional feature map is not a
bag of independent units. Because each output is computed by sliding a kernel over a
locally smooth input, and because natural images themselves are spatially smooth, adjacent
activations in a feature map carry nearly the same information — a unit and its immediate
neighbors are strongly correlated. This is a measured, repeatedly-noted property: in the
output of a convolution, neighboring positions are "strongly correlated," and during
backpropagation the same kernel weight collects gradient from correlated neighbors. The
practical consequence has been observed directly: when independent-unit dropout is applied
before a convolution on correlated maps, it increases training time but does not prevent
over-training — "standard dropout fails" in that setting. The accepted explanation is
that if one activation is zeroed but its neighbors are not, the information it held still
reaches the next layer through those neighbors, so almost nothing is actually removed.

**The dropout mechanism and its inference-time bookkeeping.** Dropout (Srivastava, Hinton,
Krizhevsky, Sutskever & Salakhutdinov, JMLR 2014) keeps each unit with probability `p`
and zeros it with probability `1 - p` during training. At test time the full network is
used, with outgoing weights multiplied by `p`, so the test-time output equals the expected
training-time output. The equivalent and more convenient "inverted" form divides each
surviving activation by `p` during training, which keeps the expected activation magnitude
unchanged and lets the inference network run untouched. The interpretation that matters
downstream is the ensemble one: a single forward pass samples one thinned sub-network, and
the un-noised inference network approximates the average over that exponential family.

**Why structured noise keeps reappearing for convolutional nets.** A line of work built on
dropout injects noise that respects the structure of the architecture instead of zeroing
independent scalars: DropConnect (drop individual weights), maxout, stochastic depth (drop
whole residual blocks), DropPath / FractalNet (drop whole paths in a multi-branch cell),
shake-shake and ShakeDrop (perturb branch combinations). The recurring lesson across these
is that for convolutional and multi-branch architectures the *effective* noise has to be
structured — dropping a whole architectural object — rather than per-scalar. Each of these,
however, is tied to a particular structural element (a branch, a path, a residual block),
so it only applies where that element exists.

**Diagnostic observations about the obvious structured variants.** Two pre-existing
observations sharpen the design space. First, dropping at whole-channel granularity helps
on correlated maps but can be too aggressive: on high-resolution feature maps removing an
entire channel discards a great deal at once, and in networks with batch normalization
channel-level and max-activation drop schemes have been found to underperform plain
dropout in some cases. Second, when noise is injected as a *fixed* probability from the
first step of training it can hurt early learning — a strong drop rate applied before the
network has formed useful features is harmful — which motivated increasing the drop
probability gradually over the course of training rather than holding it constant.

## Baselines

These are the prior regularizers a new convolutional-layer noise method would be measured
against and is reacting to.

**Dropout (Srivastava et al., JMLR 2014).** Independent Bernoulli zeroing of units with
keep probability `p`; inverted form divides survivors by `p` to fix the expected
magnitude; test uses the plain network. Discourages co-adaptation, approximates an
ensemble of thinned networks. *Limitation:* it zeros independent scalars, so on a
convolutional feature map whose units are spatially correlated the information in a dropped
unit survives in its neighbors and reaches the next layer; the regularizing effect on
convolutional layers is small, which is why it is mostly relegated to the fully connected
head.

**SpatialDropout (Tompson, Goroshin, Jain, LeCun & Bregler, CVPR 2015).** For a
convolutional tensor of shape `nfeats × H × W`, perform only `nfeats` Bernoulli trials and
extend each outcome across an entire feature map, so a whole channel is either fully zeroed
or fully kept — "adjacent pixels in the dropped-out feature map are either all 0 or all
active." This was introduced precisely because independent dropout fails on correlated
maps. *Limitation:* the granularity is an entire channel — all-or-nothing per feature map,
with no control over *where* or *how large* a region within a map is removed. On
high-resolution maps this is harsh (it throws away a full channel of a large map at once),
and it offers no knob between "one scalar" and "one whole channel."

**Cutout (DeVries & Taylor, 2017).** A data-augmentation method: apply one fixed-size
square zero-mask at a random center on the *input image* during loading, so the occluded
region is propagated through all downstream feature maps and cannot be recovered from
context, forcing the model to use the whole image. Empirically the *size* of the square
matters more than its shape (so a square is used), and allowing the patch to extend past
the image border is important so that some training images stay mostly visible. Because it
is input augmentation, no test-time rescaling is applied. *Limitation:* it acts only on the
input and only as a single region; it is not a general mechanism for injecting structured
noise at arbitrary intermediate convolutional layers, where the correlation problem also
lives.

**ScheduledDropPath / DropPath (Larsson et al. 2017; Zoph, Vasudevan, Shlens & Le, NASNet,
CVPR 2018).** DropPath stochastically drops an entire path inside a multi-branch cell with
a fixed probability. The scheduled variant drops each path with a probability that is
"linearly increased over the course of training"; plain fixed-probability DropPath did not
help the searched cells, whereas the scheduled version did. *Limitation:* it requires
multiple branches/paths to drop — it is undefined for an ordinary branch-free convolutional
layer — so it cannot serve as a general convolutional regularizer.

## Evaluation settings

The natural pre-existing yardsticks for a convolutional-layer regularizer:

- **ImageNet (ILSVRC 2012) classification**: 1.2M training images, 50k validation, 1000
  classes; standard augmentation (horizontal flip, scale, aspect-ratio), single-crop
  evaluation; metric top-1 / top-5 validation accuracy. The canonical backbone is
  ResNet-50, with deeper/searched architectures (AmoebaNet) as a transfer check. Because
  longer training schedules overfit the baseline, the highest validation accuracy over the
  full schedule is the fair comparison point.
- **CIFAR-10 / CIFAR-100**: 32×32 images, the standard small-image benchmark where dropout
  variants and cutout are tuned, with the usual zero-pad-4 + random-crop + horizontal-flip
  augmentation and SGD with Nesterov momentum, weight decay `5e-4`.
- **COCO object detection** and **PASCAL VOC semantic segmentation** with a ResNet-FPN /
  RetinaNet backbone, metric AP (detection) / mIOU (segmentation) — used to check that a
  classification-layer regularizer transfers to dense prediction.
- Protocol for comparing noise schemes: sweep the keep probability for each method and
  report each method at its own best setting; sweep the structural size parameter; study
  where in the network (which residual groups, convolution branch vs. skip connection) the
  noise is applied.

## Code framework

A regularizer of this kind is a layer inserted into the convolutional stack: it leaves the
tensor shape unchanged, is a no-op at inference, and during training replaces the input
tensor with a noised version (and applies whatever rescaling keeps the expected magnitude
fixed). The substrate that already exists is the standard convolutional training harness -
`Conv2d`/`BatchNorm`/`ReLU` modules, a residual block with a convolution branch and a skip
connection, an SGD-with-momentum loop, and the data-augmentation pipeline. The new piece is
a single `nn.Module` whose `forward` computes the noised tensor; everything about the noise
itself is exactly what is to be designed. A generic training wrapper can adjust a layer's
strength over time if the final design needs that.

```python
import torch
from torch import nn
import torch.nn.functional as F


class ActivationNoise2D(nn.Module):
    """A noise-injection layer for a 4D conv activation tensor (N, C, H, W).
    Shape-preserving; identity at inference. During training it replaces the
    activations with a noised version. The noise process itself is the thing
    to be designed."""

    def __init__(self, drop_prob, **params):
        super().__init__()
        self.drop_prob = drop_prob          # target strength (0 = no regularization)
        self.params = params

    def forward(self, x):
        assert x.dim() == 4, "expected (N, C, H, W)"
        if not self.training or self.drop_prob == 0.0:
            return x                        # identity at inference / when off
        # TODO: the noise process we will design.
        raise NotImplementedError


class StrengthSchedule(nn.Module):
    """Wraps a noise layer and exposes a per-step strength update."""

    def __init__(self, noise_layer, values):
        super().__init__()
        self.noise_layer = noise_layer
        self.values = list(values)
        self.i = 0

    def forward(self, x):
        return self.noise_layer(x)

    def step(self):
        if self.i < len(self.values):
            self.noise_layer.drop_prob = self.values[self.i]
        self.i += 1


# the noise layer plugs into the existing residual conv stack
def residual_block(x, conv_branch, skip, noise=None):
    out = conv_branch(x)        # Conv2d -> BatchNorm -> ReLU -> ...
    if noise is not None:
        out = noise(out)
    return F.relu(out + skip(x))
```

The only empty slot is the neutral noise process inside `forward`; the final code fills
that slot while reusing the same layer and optional schedule-wrapper shape.
