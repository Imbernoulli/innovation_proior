# Context: normalization layers and image style in deep CNNs (circa 2016-2018)

## Research question

Real-world image recognition has to contend with *style* variation. By style I mean the cues
that are largely independent of an object's spatial configuration — texture, contrast, global
brightness, lighting and weather conditions, the color cast of a particular camera or filter.
These variations are usually irrelevant to what an object *is*, yet they change the pixel and
feature statistics. At the same time, style is not always a nuisance: sometimes the very same
statistic is the label-relevant signal — global brightness is irrelevant to which object is
present but is exactly the feature for predicting weather or time of day; the texture of a
fabric is irrelevant to "shirt vs. skirt" but *is* the answer for "spotted vs. striped".

The question pursued here is how to design the normalization layer inside a deep convolutional
network so that it handles style variation in recognition — operating as a drop-in replacement
for the normalization layer already used in modern architectures (same input/output shape
`[N,C,H,W]`, same affine parameters), adding little parameter or compute overhead, and leaving
the data pipeline and outer training loop essentially unchanged.

## Background

Normalization layers became a standard ingredient of deep networks because they control the
scale and distribution of internal activations, ease optimization, allow higher learning rates,
and act as a mild regularizer. The field also found, separately, that the *summary statistics*
of convolutional feature maps encode the *style* of an image. These are the two threads the
setting above sits on top of.

**Activation normalization for optimization.** Ioffe & Szegedy (2015) framed the difficulty of
training deep nets as *internal covariate shift*: as the parameters of lower layers change, the
distribution of inputs to higher layers keeps moving, so each layer must continually re-adapt,
which slows convergence and pushes saturating nonlinearities into flat regions. Their fix is to
normalize each layer's activations to a fixed scale as part of the architecture, computing the
normalizing statistics from each minibatch, followed by a learnable affine `γ, β` so the layer
can recover any scale/shift it actually needs.

**Feature statistics carry style.** A parallel line in image generation found that the *style*
or *texture* of an image is captured by statistics of deep features rather than their spatial
arrangement. Gatys et al. (2015, 2016) matched the second-order statistics (Gram matrices) of
feature maps to synthesize textures and transfer artistic style. Huang & Belongie (2017) then
showed that aligning just the per-channel **mean and variance** of features is enough to
control style: their adaptive instance normalization re-normalizes content features and then
re-scales them to the style features' mean and variance,
`AdaIN(x, y) = σ(y)·(x − μ(x))/σ(x) + μ(y)`. The reading they argued for is that per-instance
feature normalization *is* a form of **style normalization** — and that a batch-level
normalization, by contrast, effectively centers a whole batch of samples around a single style,
while each individual sample may have a different style. The operative premise from this thread:
the information in a convolutional feature map splits into **shape** (the spatial configuration
of activations) and **style** (the per-channel mean and variance of those activations).

**Reported observations about existing normalizers.** Two facts about per-instance
normalization are on record. First, replacing batch-level normalization with per-instance
normalization in a *generator* dramatically improves stylization, because it removes
instance-specific contrast/style that the network would otherwise have to learn to discard
(Ulyanov et al. 2016/2017). Second, directly substituting per-instance normalization into an
image *classifier* lowers accuracy relative to batch normalization (reported by Ulyanov et al.).

## Baselines

These are the normalization layers a new design would be measured against. Take a layer input
`x ∈ R^{N×C×H×W}`, with `n` the example index, `c` the channel, `h,w` the spatial location, and
a small `ε` for stability.

**Batch Normalization — BN (Ioffe & Szegedy 2015).** Normalize each channel using the mean and
variance pooled over the batch *and* the spatial dimensions, then apply a per-channel affine:

```
x̂^(B)_{nchw} = (x_{nchw} − μ^(B)_c) / sqrt(σ²^(B)_c + ε)
μ^(B)_c      = (1/NHW) Σ_n Σ_h Σ_w  x_{nchw}
σ²^(B)_c     = (1/NHW) Σ_n Σ_h Σ_w (x_{nchw} − μ^(B)_c)²
y            = γ_c · x̂^(B)_{nchw} + β_c
```

Because the statistics pool over the whole minibatch, BN keeps the *differences between
examples*: two images in the batch with different global brightness remain different after BN
(it subtracts one shared mean, not each image's own). In the style/shape reading, BN normalizes
a batch toward a single style while retaining the per-instance style variation.

**Instance Normalization — IN (Ulyanov, Vedaldi & Lempitsky 2016/2017).** Normalize each
example independently, per channel, over the spatial dimensions only:

```
x̂^(I)_{nchw} = (x_{nchw} − μ^(I)_{nc}) / sqrt(σ²^(I)_{nc} + ε)
μ^(I)_{nc}   = (1/HW) Σ_h Σ_w  x_{nchw}
σ²^(I)_{nc}  = (1/HW) Σ_h Σ_w (x_{nchw} − μ^(I)_{nc})²
```

By subtracting each image's *own* per-channel mean and dividing by its own standard deviation,
IN forces every example to the same per-channel feature statistics — i.e. it removes the
instance-specific style and keeps only the spatial shape. It is applied identically at train and
test time. This is the operation that helps generative stylization. Relative to BN, IN removes
the per-channel mean/variance — the style information — rather than retaining it.

**A static blend of the two (fixed-weight ensemble).** A midpoint between the two is to run both
BN and IN and average their outputs with a *fixed* weight (e.g. half BN, half IN), the same for
every channel.

**Minibatch-independence variants — LayerNorm (Ba et al. 2016), GroupNorm (Wu & He 2018).**
These keep BN's normalization idea but change *which* axes the statistics pool over to remove
the dependence on batch size, which is what makes BN unreliable at very small batches: LN pools
over all channels (and spatial locations) within a single sample; GN partitions channels into
groups and pools within each group per sample, commonly using 32 groups when the channel count
supports it. Its limiting cases are useful to keep straight: one group gives layer-style
normalization, while one channel per group gives instance-style normalization. GroupNorm's
reported motivation is that BN's error rises sharply as the batch shrinks because the batch
statistics become noisy. These variants address *batch dependence* — which axes the statistics
pool over — independently of the train/test affine behavior.

## Evaluation settings

The natural yardsticks already in place for a normalization layer in image recognition — the
pre-existing datasets, architectures, and protocols.

- **General object classification.** CIFAR-10 / CIFAR-100 (32×32 natural images, 10 / 100
  classes) and ImageNet (224×224, 1000 classes). Deep residual networks are the standard
  backbones — ResNet-110 on CIFAR, ResNet-18 on ImageNet — each with a normalization layer
  after every convolution; one swaps the normalization layer in place and leaves everything
  else unchanged. Metric: top-1 accuracy. A common stability check is to run CIFAR multiple
  times and report a mean with a confidence interval.
- **Scalability across architectures.** The same swap applied to a range of CNN families on
  CIFAR-100 — AlexNet, VGG, ResNet (and pre-activation ResNet), Wide ResNet, ResNeXt, DenseNet
  — each trained with that family's own published hyper-parameters, to check the layer is not
  tied to one backbone.
- **Multi-domain / domain-shift recognition.** The Office-Home dataset (four domains — Art,
  Clipart, Product, Real-World — sharing 65 categories) exhibits large style disparity across
  domains; used both for mixed-domain training (train on all four, test per domain) and for
  unsupervised domain adaptation with an adversarial domain-adaptation backbone on ResNet-18.
- **Image style transfer.** A feed-forward stylization network (an image-transformation net
  trained against a fixed VGG loss network producing content and style losses), trained on
  MS-COCO content images, with the normalization layer swapped — the setting where per-instance
  normalization is normally the right choice, used to check a layer can also serve generation.
- **Optimization protocol.** For the classification swaps, SGD with momentum 0.9, batch size
  128, initial learning rate 0.1, and weight decay `1e-4`; CIFAR training uses learning-rate
  drops at 32K and 48K iterations, while ImageNet uses drops at 30 and 60 epochs. Metric:
  top-1 accuracy for classification, domain-specific accuracy for multi-domain recognition,
  and the usual content/style losses for feed-forward stylization.

## Code framework

The new layer plugs into the existing CNN training harness as a drop-in replacement for the
batch-normalization module: same constructor signature `(num_features=C)`, same input/output
shape `[N,C,H,W]`, the same learnable per-channel affine `γ, β`, and numerically stable behavior
in both train and eval. The batch-normalization base class already supplies everything common —
the affine parameters `weight` (`γ`) and `bias` (`β`), the running statistics, `eps`, `momentum`,
and the per-dimensionality input check. The library `F.batch_norm` primitive is also available
and pools its statistics per channel over the batch and spatial axes (and can be redirected to
other axis groupings by reshaping its input). What computes the normalized activations inside
`forward` is exactly what is to be designed, so the substrate below inherits the base class and
leaves one empty slot for the normalization rule, plus a place for any extra parameter it needs
and the surrounding optimizer/training loop.

```python
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class _CustomNorm(_BatchNorm):
    """Drop-in replacement for the per-channel normalization layer used after every
    convolution. Inherits the affine params (weight=gamma, bias=beta), running stats,
    eps, momentum, and _check_input_dim from the batch-norm base. forward maps
    [N, C, H, W] to [N, C, H, W]; must be numerically stable in train and eval."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(num_features, eps, momentum, affine)
        # TODO: any extra learnable parameter the normalization rule we design needs
        #       (and, if it must stay in a range, tag it for a post-step constraint)

    def forward(self, input):
        self._check_input_dim(input)
        # Available primitive (pools per channel over batch+spatial axes):
        #   F.batch_norm(input, running_mean, running_var, weight, bias,
        #                training, momentum, eps)
        # reshaping input before the call redirects which axes the statistics pool over.
        # TODO: the normalization rule we will design — produce the normalized,
        #       affine-transformed output from `input` and the parameters above.
        raise NotImplementedError


# concrete 4D (image) variant; the base class fixes the common machinery
class CustomNorm2d(_CustomNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


# existing minibatch training loop the layer plugs into; a post-step constraint hook
# is left available in case the rule introduces a parameter that must stay in a range
def train(model, loss_fn, data_loader, optimizer):
    model.train()
    constrained = []  # TODO: collect any parameters the rule needs constrained post-step
    for inputs, targets in data_loader:
        loss = loss_fn(model(inputs), targets)   # forward; CustomNorm runs after each conv
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                         # SGD step on all parameters
        for p in constrained:                    # optional post-step parameter constraint
            pass
```

The outer loop is fixed; the contribution lives inside `forward` (and, if the rule adds a
constrained parameter, the optimizer grouping and the post-step hook).
