# Context: aggregating a convolutional score/feature map into one vector per image (circa 2013-2015)

## Research question

A convolutional network run on an image emits a 3D activation tensor: for the last
convolutional block we get `C` feature maps, each a `H × W` grid of activations,
`X ∈ ℝ^{C×H×W}`. Everything downstream — a classifier head, or a per-class score readout —
consumes a fixed-length vector, not a grid. So the spatial extent has to be collapsed: each
`H × W` feature map must be reduced to a single number, turning the `C × H × W` tensor into a
`C`-vector. The precise problem is to design that collapse — a map from one feature map's `N = H·W`
activations to a single scalar — so that the resulting `C`-vector is as discriminative as
possible for image classification.

Two things make the choice consequential rather than cosmetic. First, this single operation is
the entire bottleneck between the spatial evidence the convolutions found and the label the head
must predict: whatever the collapse discards is gone, whatever it over-weights the classifier
inherits. Second, the supervision is *image-level* — the label says only which categories are
present in the image, never *where* — yet a real training image is cluttered: the object of
interest occupies a few spatial locations against a large field of background and co-occurring
stuff, and the discriminative response for a class typically fires strongly at those few
locations and weakly or not at all elsewhere. A good collapse has to summarize that spatially
*localized*, *sparse* evidence under a label that gives no location, work across architectures
and feature-map sizes (the spatial grid ranges from `8×8` down to `1×1`), be cheap (it runs on
every image, every forward and backward pass), and ideally introduce no new parameters and no
overfitting at this layer. The activations entering it are non-negative when the last operation
is a ReLU.

## Background

By this time the convolutional activations of a network trained for large-scale image
classification (Krizhevsky, Sutskever & Hinton 2012; Russakovsky et al. 2015) are established as
strong general-purpose image features, including as off-the-shelf descriptors (Razavian et al.
2014), and fine-tuning the backbone for the target task improves them further. The standard
recipe for the head had been: flatten the last feature maps into one long
vector and run a stack of fully-connected (FC) layers into a softmax. That structure bridges the
convolutional feature extractor to a conventional classifier, but it carries two load-bearing
costs that the field had begun to diagnose. The FC layers hold the bulk of the parameters and are
the part most prone to overfitting (Hinton et al.'s dropout was introduced largely to regularize
exactly these layers). And flattening *fixes* the input geometry: a fully-connected layer expects
a fixed-length input, so the network is locked to a single input image size, and the spatial
layout is baked into the weights rather than handled translation-invariantly.

The diagnostic finding that reframes the head is that the last convolutional feature maps are
already spatially meaningful — a given channel responds to a particular pattern *at the location
where it occurs*. So the spatial grid does not need to be memorized by FC weights; it can be
*pooled out*. Lin, Chen & Yan (Network in Network, 2014) made this concrete: replace the FC
stack with a single global pooling of each final feature map into one number, producing a
`C`-vector fed straight to the classifier (or, with one map per class, straight to the softmax).
This removes the FC parameters ("no parameter to optimize ... overfitting is avoided at this
layer"), makes the feature maps interpretable as per-category confidence maps, and — because it
"sums out the spatial information" — is robust to spatial translation and works for any spatial
size. Global pooling, then, is the natural modern head; the whole question collapses onto *which*
pooling rule.

The structural fact a pooling rule must reckon with is what a real feature map actually looks
like under image-level supervision: it is *not* uniform. For a class-discriminative channel on a
cluttered image, a few spatial locations fire strongly (the object, or its distinctive parts)
against a large mostly-weak field (background, clutter, co-occurring objects). A pooling rule has
to decide how to weigh the rare strong locations against the many weak ones, and it has to do so
with no access to where the object is. There is also a closely related learning setting that
makes the same point sharply: weakly-supervised localization, where one wants the network to
*find* the object given only presence/absence labels — the per-location truth is hidden, only the
image-level bit is supplied. Earlier integrated segmentation-and-recognition networks (Keeler,
Rumelhart & Leow, NIPS 1990; the translation-invariant linked-receptive-field line of Lang,
Waibel & Hinton's TDNN) had already shown one can train a location-specific recognizer from
location-free targets by inserting a fixed "forward model" that maps the grid of per-location
scores to one location-independent output and backpropagating a non-specific error through it.
They aggregated the per-location evidence with a *sum* of exponential (softmax-like) units —
recovering position without position labels — but left open how a modern pooling head should
aggregate.

## Baselines

These are the global-pooling rules a new aggregation layer would be measured against and react
to. Each takes one feature map `X_k` (its `N = H·W` activations) and returns one scalar `f_k`;
the descriptor is `f = [f_1, ..., f_C]`.

**Global average pooling — GAP (Lin, Chen & Yan, Network in Network, 2014).** Average every
spatial location of each map:

```
f_k = (1/N) · Σ_{x ∈ X_k} x.
```

It uses the whole map, is smooth and stable, has no parameters, and makes the head
size-independent and translation-robust. It became the default modern classification head.
**Limitation:** every location is weighted identically at `1/N`. When the discriminative
response is concentrated at a few locations against a large mostly-dead field — the typical
cluttered-image case — averaging dilutes that sparse strong signal against the background, so the
per-channel summary is dominated by how much *area* responded rather than by the magnitude of the
sparse local evidence; and the gradient it sends back is spread uniformly across the grid,
crediting every location, including background, equally.

**Flatten + fully-connected head (Krizhevsky et al. 2012; the pre-NIN default).** Vectorize the
last feature maps and learn an FC stack into the softmax. It can in principle learn an arbitrary
spatial weighting. **Limitation:** it holds most of the network's parameters and overfits (hence
dropout on exactly these layers); it fixes the input size because the FC layer needs a
fixed-length input; and it bakes the spatial layout into weights rather than handling it
translation-invariantly — the opposite of what a cluttered, variable-position object wants.

**Sum-of-exponential / softmax-style spatial aggregation (Keeler, Rumelhart & Leow, NIPS 1990).**
Aggregate the grid of per-location class scores into one location-independent output by summing
exponential units (a smooth, soft combination), trained from presence/absence targets alone via
a fixed forward model. It established that location can be recovered from location-free labels.
**Limitation:** the soft sum mixes the contributions of *all* locations into the output, so on a
cluttered image the background locations still leak into the aggregated score and the error
signal is shared across the whole field; a temperature-weighted blend of every location carries
the same dilution character as the average, gentler but the same kind.

Common thread: the prior heads either weight every location equally (GAP, and in a softer sense
the exponential sum) — diluting localized evidence and spreading the learning signal over
background — or pay a large parametric/fixed-size price to learn the weighting (FC). None
matches the regime where the label is a single image-level bit and the evidence is sparse,
unknown-position structure in a sea of clutter.

## Evaluation settings

The natural yardsticks at the time. The pooling layer is dropped between a standard convolutional
backbone and the classifier head, inside an otherwise-fixed training harness. The backbone's
convolutional stack is the usual ImageNet-class architecture family (Krizhevsky et al. 2012;
VGG, Simonyan & Zisserman 2014; ResNet, He et al. 2016; and lightweight mobile backbones), with
the FC layers removed and the last activation a ReLU. Training is SGD with momentum `0.9` and
weight decay `5e-4` under a fixed cosine learning-rate schedule over a long run (200 epochs),
with standard `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` augmentation, on standard
small-image classification benchmarks (CIFAR-100 with ResNet and VGG backbones; FashionMNIST
with a MobileNet backbone). The metric is best test accuracy. Feature-map spatial sizes the
layer must handle range from `8×8` (ResNet on CIFAR) down to `1×1` (after VGG's max-pool stack,
or a MobileNet stem at low resolution), and the channel count `C` varies by architecture (64,
512, 1280). The pooling layer must accept `[B, C, H, W]`, return `[B, C]` with the channel
dimension unchanged, handle any spatial size, and use no access to data or labels inside the
layer.

## Code framework

The pooling layer is a single `nn.Module` slotted between the convolutional backbone and the
classifier head, inside an otherwise-fixed training harness (the optimizer, schedule, loss, and
data pipeline are given and not to be touched). All that exists before the method is the generic
contract — consume the conv feature tensor `[B, C, H, W]`, return a per-image channel vector
`[B, C]` — and the standard tensor primitives available to implement it (elementwise ops,
shape manipulation, generic reductions over the `H, W` axes, learnable `nn.Parameter`s if wanted).
The aggregation rule itself is the empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPool(nn.Module):
    """Global spatial pooling: collapse a conv feature map to one vector per image.

    Input : x of shape [B, C, H, W]  (activations are >= 0 after a ReLU)
    Output:     shape [B, C]         (channel dimension unchanged; any H, W)
    """

    def __init__(self):
        super().__init__()
        # TODO: any state the aggregation rule we design will need.
        pass

    def forward(self, x):
        # x: [B, C, H, W] -> collapse the H, W grid of each channel to one number.
        # TODO: the aggregation rule we will design; return [B, C].
        pass


# existing classification harness the pooling layer plugs into (fixed; not modified)
def train(backbone, pool, head, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        feats = backbone(inputs)          # [B, C, H, W] conv feature maps
        vec = pool(feats)                 # [B, C] descriptor  <-- the slot
        logits = head(vec)                # existing classifier head
        loss = loss_fn(logits, targets)   # existing loss (image-level labels only)
        loss.backward()
        optimizer.step()
```

The backbone, head, loss, optimizer, and schedule are all given; the one thing to design is the
body of `CustomPool`.
