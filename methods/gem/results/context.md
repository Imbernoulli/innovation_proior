# Context: global spatial pooling for CNN image descriptors (circa 2014-2017)

## Research question

A modern convolutional network, run on an image, emits a 3D activation tensor: for the last
convolutional block we get `K` feature maps, each a `W × H` grid of activations,
`X = {X_1, ..., X_K}` with `X_k` the set of `W·H` numbers in map `k`. To turn this into a
fixed-length image descriptor that downstream layers (a classifier head, or a metric for
retrieval) can consume, the spatial grid has to be collapsed: every feature map is summarized
into a single number, so the `W × H × K` tensor becomes a `K`-vector. This aggregation is
the *global pooling* step. The vector it produces is the entire image representation as far as
everything after it is concerned.

The precise problem: design that collapse — a map from one feature map's `W·H` activations to a
single scalar — so that the resulting `K`-vector is discriminative. It has to
work across architectures and feature-map sizes, and it runs on every image, on
every forward and backward pass. The activations entering it are non-negative (the last layer is
a ReLU), and the descriptor is trained end to end, so any free parameter inside the pooling rule could in
principle be learned by backpropagation rather than fixed by hand.

## Background

By this time it is established that the convolutional activations of a network trained for
large-scale image classification (Krizhevsky et al. 2012; Russakovsky et al. 2015), used as
off-the-shelf features (Razavian et al. 2014), make strong general-purpose image descriptors, and
that *fine-tuning* the network for the target task improves them further (Oquab et al. 2014). For
producing a *compact* global descriptor, the dominant recipe is: take the last convolutional
feature maps and apply a single global-pooling operation, yielding a vector whose dimensionality
equals the number of feature maps `K` — for common networks 256, 512, or 2048 — a small,
search-friendly representation. The descriptor is then `ℓ2`-normalized so that image similarity
is an inner product.

The spatial collapse is therefore governed by the choice of pooling function, and
the field has converged on two default rules, plus weighted and regional variants. A standing
observation about activation fields is that a feature map is *not* uniform: a few
spatial locations fire strongly (the discriminative object parts the channel detects) against a
large field of weak or zero responses. A pooling rule weighs the rare strong
locations against the many weak ones, and the two standard rules sit at the two extremes of that
decision — one keeps only the single strongest location, the other weighs every location
equally.

## Baselines

These are the pooling rules a new global-pooling layer would be measured against. Each takes one
feature map `X_k` (its `N = W·H` activations) and returns one scalar `f_k`; the
descriptor is `f = [f_1, ..., f_K]`.

**Global max pooling — MAC (Razavian et al. 2014; integral max-pooling, Tolias, Sicre & Jégou,
ICLR 2016).** Keep only the single strongest activation per map:

```
f_k = max_{x ∈ X_k} x.
```

Because each component is one activation, it corresponds to one receptive-field patch, and the
descriptor inner product implicitly matches the most-distinctive patches between two images.
Regional max-pooling variants (R-MAC) max over a rigid grid of rectangular regions and then sum the
region descriptors, injecting some locality through a region grid set by hand (number of scales,
sizes, strides, overlap).

**Sum / average pooling — SPoC (Babenko & Lempitsky, ICCV 2015).** Average every location:

```
f_k = (1/N) · Σ_{x ∈ X_k} x.
```

It uses the whole map, is smooth and stable, and was found to work especially well once followed
by descriptor whitening. Every spatial location is weighted identically.

**Weighted-sum pooling — CroW (Kalantidis, Mellina & Osindero, ECCVW 2016).** A sum in which each
spatial location and each channel is multiplied by a weight before accumulation, the weights
derived from spatial saliency and channel sparsity statistics computed from the activations. It
up-weights informative regions relative to plain SPoC. The weighting rules are hand-designed
heuristics, and the operation is at bottom a re-weighted arithmetic mean.

**Mixed / hybrid pooling (Mousavian & Kosecka 2015 for retrieval; Lee, Gallagher & Tu, AISTATS
2016, for recognition).** Take a convex combination of the two extremes,
`a·max + (1−a)·avg`, with `a` a mixing scalar. The output is a straight-line blend of the
arithmetic mean and the max of the pool, an additive mixture of two fixed rules with the mixing
weight as the free knob.

Each of these is a *fixed* rule (or a fixed rule with hand-set knobs), sitting at or between the
two endpoints of "one location" and "all locations equally."

## Evaluation settings

The natural yardsticks at the time. The pooling layer is dropped into a standard CNN backbone
whose convolutional stack is initialized from ImageNet classification (Krizhevsky et al. 2012;
VGG, Simonyan & Zisserman 2014; ResNet, He et al. 2016), fully-connected
layers discarded, last layer a ReLU. For the image-classification setting the descriptor feeds a
linear classifier head;
training is the usual SGD-with-momentum recipe (momentum 0.9, weight decay `5e-4`) under a fixed
cosine learning-rate schedule over a long run, with standard `RandomCrop`/`RandomHorizontalFlip`
augmentation, on standard small-image classification benchmarks (CIFAR-100 with ResNet and VGG
backbones; FashionMNIST with a MobileNet backbone). The metric is top test accuracy. Feature-map
spatial sizes the layer must handle range from `8×8` (ResNet on CIFAR) down to `1×1` (after
VGG's max-pool stack, or a MobileNet stem at low resolution), and the channel count `C` varies by
architecture (64, 512, 1280). The pooling layer must accept `[B, C, H, W]`, return `[B, C]` with
the channel dimension unchanged, handle any spatial size, and use no access to data or labels
inside the layer.

## Code framework

The pooling layer is a single `nn.Module` slotted between the convolutional backbone and the
classifier head, inside an otherwise-fixed training harness (the optimizer, schedule, loss, and
data pipeline are given and not to be touched). All that exists before the method is the generic
contract — consume the conv feature tensor `[B, C, H, W]`, return a per-image channel vector
`[B, C]` — and the standard tensor primitives available to implement it (elementwise ops, the
spatial mean/max reductions, learnable `nn.Parameter`s). The aggregation rule itself is the empty
slot.

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
        # TODO: any learnable state the aggregation rule we design will need.
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
        vec = pool(feats)                 # [B, C] global descriptor  <-- the slot
        logits = head(vec)                # existing classifier head
        loss = loss_fn(logits, targets)   # existing loss
        loss.backward()
        optimizer.step()
```

The backbone, head, loss, optimizer, and schedule are all given; the one thing to design is the
body of `CustomPool`.
