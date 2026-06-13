# Context: collapsing a convolutional feature map into one vector per image

## Research question

A convolutional network run on an image emits a 3D activation tensor: the last convolutional
block produces `C` feature maps, each an `H × W` grid of activations, `X ∈ ℝ^{C×H×W}`. The
classifier downstream consumes a fixed-length vector, not a grid, so the spatial extent has to be
collapsed — each `H × W` map reduced to a single number, turning `C × H × W` into a `C`-vector.
The problem is to design that per-channel spatial collapse — a map from one feature map's
`N = H·W` activations to a single scalar — so the resulting `C`-vector is as discriminative as
possible for image classification, across different backbone architectures and datasets.

The choice is consequential, not cosmetic. This single operation is the entire bottleneck between
the spatial evidence the convolutions found and the label the head must predict: whatever it
discards is gone, whatever it over-weights the classifier inherits. And the statistics of a real
feature map are uneven. A class-discriminative channel on a cluttered image fires strongly at a
few spatial locations (the object, or its distinctive parts) against a large, mostly-weak field
(background, clutter, co-occurring stuff); but a channel that responds to a diffuse, spread-out
pattern lights up broadly at moderate strength. The same collapse must serve both kinds of channel
at once, with no access to where the object is (the label is image-level only). A good collapse
also has to work across feature-map sizes (the spatial grid ranges from `8×8` down to `1×1`), be
cheap (it runs on every image, every forward and backward pass), keep the channel dimension exactly
`C` so the classifier interface is unchanged, and ideally introduce no new parameters and no
overfitting at this layer. The activations entering it are non-negative when the last operation is
a ReLU.

## Background

Spatial pooling — combining the responses of feature detectors over a neighborhood into one
statistic that summarizes their joint distribution — has been part of visual-recognition
architectures since Hubel and Wiesel's complex cells and Fukushima's Neocognitron, and it is built
into convolutional networks (LeCun et al. 1989, 1998) and the HMAX family (Serre et al. 2005). The
pooling operation is typically a commutative reduction over a region — a sum, an average, or a max.
Two have dominated practice. **Average pooling** (the classical CNN subsampling step) returns the
mean activation over the region. **Max pooling** (Ranzato et al. 2007; the HMAX line) returns the
single largest activation. By the early 2010s max pooling had become the common default inside
CNNs.

That default rested on direct empirical comparison. Scherer, Müller & Behnke (ICANN 2010) held a
CNN architecture fixed and varied only the aggregation function across object-recognition tasks,
and found max pooling significantly outperforms average subsampling — a clean, confound-free
demonstration that the choice of pooling function matters, with the parenthetical observation that
choosing from more than one type of pooling could help.

The deeper, *diagnostic* finding — why one pooling rule beats another, and when — comes from
Boureau, Ponce & LeCun (ICML 2010), who treated pooling as a class-separability problem. Model a
single feature pooled over `P` locations whose per-location activations are i.i.d. Bernoulli with
mean `α` (the feature's activation rate). View separability between two classes (activation rates
`α₁ > α₂`) as a signal-to-noise ratio: the distance between the two class-conditional means of the
pooled value, divided by the spread (standard deviations) of those distributions.

For **average pooling** `f_a = (1/P)Σ vᵢ`, the pooled value has mean `µ_a = α` (independent of `P`)
and variance `σ_a² = α(1-α)/P`. So the mean-separation `|α₁ - α₂|` is *fixed* in `P`, while the
spread shrinks like `1/√P`; the SNR therefore *grows* with pool cardinality, like `√P`. Averaging
over more locations buys a cleaner estimate of the mean activation rate.

For **max pooling** `f_m = maxᵢ vᵢ`, the pooled value is itself Bernoulli with mean
`µ_m = 1 - (1-α)^P` (equal to `α` at `P = 1` and approaching 1 as `P` grows) and variance
`σ_m² = (1-(1-α)^P)(1-α)^P`. The mean-separation is
`φ(P) = |(1-α₂)^P - (1-α₁)^P|`; at `P = 1` it equals `|α₁ - α₂|`, while in the large-`P`
limit it falls back toward zero as both means saturate at 1. For the usual sparse-feature
case it rises to a peak at an intermediate cardinality before falling; in dense cases that
peak can move to very small cardinality. Max-pool separability is thus non-monotone in `P`,
with the useful cardinality window depending on the activation rate — and for sparse, rare
features it peaks much later (larger `P`).

Comparing the two SNRs, Boureau et al. found there is a range of cardinalities for which the
max-pool mean-separation exceeds the average-pool one, and a regime — sparse features, especially
`α₂ ≪ α₁ ≪ 1` (a feature that is rare overall but much more frequent in one class) — where the
max-pool SNR is the larger. In other regimes (dense features, large pools) average pooling has the
better SNR. Which statistic separates the classes better depends on the feature's activation rate
and the effective pool size, both of which vary from channel to channel and from dataset to
dataset. The same analysis frames average and max as the two endpoints of a single continuum of
pooling behaviors.

Independently, the question of *how to collapse the whole feature map for classification* had been
reframed by Lin, Chen & Yan (Network in Network, 2014). The standard head had flattened the last
feature maps and run a stack of fully-connected layers into a softmax; that stack holds most of the
parameters, overfits (dropout was introduced largely to regularize exactly these layers), and
fixes the input size because a fully-connected layer needs a fixed-length input. NIN observed that
the last convolutional maps are already spatially meaningful and proposed *global average pooling*:
average each final feature map to one number, producing a `C`-vector fed straight to the
classifier. It has no parameters ("overfitting is avoided at this layer"), is interpretable as
per-category confidence maps, and "sums out the spatial information," making the head robust to
translation and usable at any spatial size. Global pooling thus became the natural modern head, and
the whole question collapses onto *which* global pooling rule to use.

## Baselines

The global-pooling rules a new aggregation layer would be measured against and react to. Each takes
one feature map `X_k` (its `N = H·W` activations) and returns one scalar `f_k`; the descriptor is
`f = [f_1, ..., f_C]`.

**Global average pooling — GAP (Lin, Chen & Yan, Network in Network, 2014).** Average every
location of each map:

```
f_k = (1/N) · Σ_{x ∈ X_k} x.
```

It uses the whole map, is smooth and stable, has no parameters, makes the head size-independent and
translation-robust, and became the default classification head. Its gradient is spread uniformly,
`∂f_k/∂x = 1/N` at every location. **Limitation:** every location is weighted identically at
`1/N`. When the discriminative response is concentrated at a few locations against a large
mostly-dead field, averaging divides the strong peak by `N`, so the per-channel summary is governed
by how much *area* responded rather than by the magnitude of the sparse local evidence; by
Boureau's analysis this is the right statistic in the dense/large-pool regime but the worse one in
the sparse-feature regime.

**Global max pooling — GMP (Ranzato et al. 2007 for the operation; the HMAX line; the CNN default
Scherer et al. 2010 measured).** Take the single largest activation of each map:

```
f_k = max_{x ∈ X_k} x.
```

It returns the peak response undiluted by area, has no parameters, and — per Boureau — has the
better SNR in the sparse-feature regime where a rare strong response carries the class signal. Its
gradient flows only to the argmax location, `∂f_k/∂x = 1[x = max]`. **Limitation:** it reads off a
single location and discards the other `N-1` — it cannot register how broadly or how consistently a
feature fired across the map, which is exactly the information that distinguishes a diffuse pattern
from a one-pixel spike; and it sends a learning signal to one location per channel per image,
leaving the rest of the field uncredited. By Boureau's analysis its separability is non-monotone
and degrades for dense features and large pools, the regime where averaging wins.

**Flatten + fully-connected head (Krizhevsky et al. 2012; the pre-NIN default).** Vectorize the
last feature maps and learn an FC stack into the softmax, which can in principle learn an arbitrary
spatial weighting. **Limitation:** it holds most of the network's parameters and overfits (hence
dropout on exactly these layers), fixes the input size because the FC layer needs a fixed-length
input, and bakes the spatial layout into weights rather than handling it translation-invariantly —
the opposite of what a cluttered, variable-position object wants.

Common thread: the two parameter-free global poolers each commit, irrevocably and identically for
every channel, to one end of Boureau's continuum — GAP to the mean-rate statistic, GMP to the
peak statistic — and so each is the worse choice precisely in the regime where the other wins; and
the parametric FC alternative pays a large parameter and fixed-size cost to weight locations at
all. Across channels, layers, architectures and datasets, the activation rates and effective pool
sizes that decide *which* endpoint is better are not the same, yet a single fixed parameter-free
pooler can sit at only one endpoint.

## Evaluation settings

The pooling layer is dropped between a standard convolutional
backbone and the classifier head, inside an otherwise-fixed training harness (optimizer, schedule,
loss, and data pipeline given and not modified). Backbones are the usual image-classification
families with the FC layers removed and the last activation a ReLU: ResNet-56 and VGG-16-BN on
CIFAR-100, and a MobileNetV2 backbone on FashionMNIST. Training is SGD with `lr=0.1`,
`momentum=0.9`, `weight_decay=5e-4` under a fixed cosine learning-rate schedule over 200 epochs,
with `RandomCrop(32, pad=4)` + `RandomHorizontalFlip` augmentation. The metric is best test
accuracy. Feature-map spatial sizes the layer must handle range from `8×8` (ResNet on CIFAR) down
to `1×1` (after VGG's max-pool stack, or a MobileNet stem at low resolution), and the channel
count `C` varies by architecture (`64`, `512`, `1280`). The pooling layer must accept
`[B, C, H, W]`, return `[B, C]` with the channel dimension unchanged, handle any spatial size, and
use no access to data or labels inside the layer.

## Code framework

The pooling layer is a single `nn.Module` slotted between the convolutional backbone and the
classifier head, inside an otherwise-fixed training harness. All that exists before the method is
the generic contract — consume the conv feature tensor `[B, C, H, W]`, return a per-image channel
vector `[B, C]` — and the standard tensor primitives available to implement it: elementwise ops,
shape manipulation, generic reductions over the `H, W` axes (e.g. `F.adaptive_avg_pool2d`,
`F.adaptive_max_pool2d` to size 1), and learnable `nn.Parameter`s if wanted. The aggregation rule
itself is the empty slot.

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
