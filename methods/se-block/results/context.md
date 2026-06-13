# Context: convolutional feature learning and the channel axis (circa 2016-2017)

## Research question

A convolutional layer is the workhorse of modern image recognition, and the thing it does at
every position is fuse two kinds of information at once: spatial pattern (a small `k×k`
neighbourhood) and channel composition (a weighted sum across the input channels). Formally
the `c`-th output feature map is

```
u_c = v_c * X = sum_{s=1}^{C'} v_c^s * x^s,
```

where each `v_c^s` is a 2-D spatial kernel acting on input channel `x^s` and the sum runs over
all `C'` input channels. So channel relationships *are* being modelled — they are baked into
the filter `v_c` — but in three constrained ways. First, they are **implicit**: the channel
weighting is entangled with the spatial pattern the filter is matching, never exposed as a
quantity the network can manipulate on its own. Second, they are **local**: every entry
`u_c(i,j)` was computed from a small receptive field, so no unit anywhere in the lower layers
can condition on information outside its window — there is no path for global context to
influence a response. Third, and most importantly, they are **static and
instance-agnostic**: once training fixes the filter weights, the same channel combination is
applied to every input image, so the network has no mechanism to say "for *this* image,
emphasise these channels and suppress those."

The precise goal is a unit that closes exactly that gap: an inexpensive, drop-in computational
unit that lets a network use **global** information to adaptively, and **per input**, modulate
the importance of its feature channels — retaining high-gain responses for useful channels and
damping less useful ones — without redesigning the backbone. To be worth adopting it must (1)
be generic enough to sit inside many architectures (plain conv nets, residual nets,
grouped/multi-branch nets) and at many depths; (2) add only a slight parameter and compute
overhead; and (3) preserve the gradient-flow and optimisation properties that already make deep
nets trainable. None of the units on the table below does all of this on the channel axis.
Closing that gap is the problem.

## Background

By this point depth is the dominant lever on representation quality. VGGNets and Inception
showed that stacking more layers buys sharply better features; Batch Normalization (Ioffe &
Szegedy 2015) stabilised the training of those deep stacks by normalising each layer's
pre-activations to a controlled distribution, smoothing the optimisation surface; and Residual
Networks (He et al. 2016) made very deep nets trainable at all by adding an identity shortcut
around each pair of layers, so a block learns a residual `F(x)` added back onto its input
`y = F(x) + x`. The identity path is parameter-free and always open, which is what keeps
gradients healthy through hundreds of layers. The residual branch `F`, however, is still an
ordinary stack of convolutions, so the channel mixing it performs is the same implicit, local,
static mixing described above.

A second, closely related line attacks the *functional form* of the computational element
rather than the depth. Network-in-Network (Lin et al. 2014) introduced `1×1` convolutions that
recombine channels at a single spatial location; grouped convolutions and multi-branch
modules (Inception; ResNeXt, Xie et al. 2017) compose channels along several parallel paths and
add cardinality as a capacity axis. Almost all of this work shares an unstated assumption:
that channel relationships are well captured by a *fixed* composition of operators with *local*
receptive fields — the same function applied to every input. Batch Normalization does give each
channel a learnable affine scale and shift, but those are learned from dataset-wide statistics
and are constant at inference; they cannot react to the content of an individual image.

There is also a body of work on letting a network allocate its "attention" — biasing
computation toward the most informative parts of a signal. Early models learned where to glance
in an image with hard, sequential, reinforcement-learned attention (Larochelle & Hinton 2010;
Mnih et al. 2014), which is powerful but expensive and non-differentiable; spatial transformer
networks (Jaderberg et al. 2015) learn to warp the spatial sampling grid. The common thread is
that these mechanisms answer *where to look in space*. The channel axis — *which feature
channels matter* — is largely left to the static filter weights.

Two empirical facts about trained deep nets are worth holding onto, because they bear on any
channel-modulation scheme. First, feature channels are not equally useful for every input or at
every depth: a large literature on feature visualisation and transfer (Lee et al. 2009; Yosinski
et al. 2014) finds that early-layer features are general and class-agnostic while later-layer
features are increasingly class-specific. Second, a unit's receptive field grows with depth, so
only near the top of a network does any single response see most of the image; lower down,
"global context" is simply unavailable to a local operator.

## Baselines

These are the prior units a new channel-modulation unit would be measured against and would
react to.

**Residual block (He et al. 2016).** The substrate. Each block computes `y = F(x) + x`, with `F`
a short stack of conv–BN–ReLU layers and an identity (or `1×1`-projection) shortcut. The
identity path is parameter-free and always open, which is what makes very deep nets trainable.
**Limitation on the channel axis:** the residual branch `F` mixes channels only through ordinary
convolution, so that mixing is the same implicit, local, instance-agnostic mixing as any conv;
the block has no way to use global context to reweight its own channels per input.

**`1×1` convolution / Network-in-Network (Lin et al. 2014).** A `1×1` conv recombines channels
as a learned linear (then nonlinear) map *at each spatial location independently*. It is the
cheapest available channel mixer and is used everywhere for projection and bottlenecking.
**Limitation:** the recombination is still local (it sees one spatial location at a time) and the
same weights apply to every input — it remaps channels with an instance-agnostic function of a
local operator, with no access to a global summary of the feature map.

**Multi-branch / grouped convolution (Inception, Szegedy et al.; ResNeXt, Xie et al. 2017).**
Channels are split across parallel paths of different receptive fields or into groups, then
concatenated/summed; cardinality becomes a third capacity knob beside depth and width.
**Limitation:** this enlarges and diversifies the *static* set of channel combinations the layer
can express, but the combination chosen is still fixed after training and computed from local
receptive fields — there is no input-conditioned, globally-informed emphasis of channels.

**Batch Normalization affine (Ioffe & Szegedy 2015).** After normalising each channel, BN
applies a learned per-channel scale `γ_c` and shift `β_c`. This is a per-channel multiplicative
modulation, structurally close to what we want. **Limitation:** `γ_c` is a single learned constant
per channel, estimated from dataset-wide statistics and frozen at inference; it is the same for
every image and carries no dependence on the current input's content.

**Highway gating (Srivastava et al. 2015).** Highway networks regulate information flow along a
deep stack with multiplicative gates: `y = H(x)·T(x) + x·(1 − T(x))`, where the transform gate
`T(x) = σ(W_T x + b_T)` is a learned, sigmoid-bounded function of the layer input. This shows
that learned multiplicative gating is trainable and useful inside deep nets.
**Limitation:** the gate is a *local* transform of `x` (it sees the same receptive field as the
layer) whose job is to trade off the carried input against the transformed input *for depth-wise
information flow*; it does not summarise global context, and it gates the shortcut/transform
balance rather than reweighting feature channels by their per-input importance.

**Trunk-and-mask residual attention (Wang et al. 2017).** Inserts an auxiliary "mask" branch —
an hourglass (encoder–decoder) sub-network — between stages of a residual net to produce a soft
3-D attention map that multiplies the trunk features. It demonstrates that a learned soft mask
can improve a deep net. **Limitation:** the mask branch is heavy (a whole extra encoder–decoder
per attention module), and the mask it produces is dominated by spatial structure; the cost and
complexity make it ill-suited as a lightweight unit to drop into every block, and the channel
dimension is not isolated as the thing being modulated.

## Evaluation settings

The natural yardsticks already in use for image-classification block design, all pre-existing:

- **ImageNet (ILSVRC 2012)** — 1.28M training / 50K validation images, 1000 classes; metric is
  top-1 and top-5 error on the validation set. The standard large-scale benchmark for backbone
  and block design. Standard recipe: SGD with momentum, scale/aspect-ratio random crop to
  `224×224`, random horizontal flip, mean-RGB subtraction, step learning-rate decay, He
  initialisation, train from scratch.
- **CIFAR-10 and CIFAR-100** — 50K train / 10K test `32×32` RGB images, 10 and 100 classes. The
  small-image testbed where a block design is checked across network depths. For a CIFAR ResNet
  the convention is the depth family ResNet-`6n+2` built from three stages of `n` basic blocks on
  feature maps of size `32×32 → 16×16 → 8×8` (e.g. ResNet-20 with `[3,3,3]`, ResNet-56 with
  `[9,9,9]`, ResNet-110 with `[18,18,18]`), global-average-pooled into a linear classifier.
  Augmentation is `RandomCrop(32, padding=4)` plus random horizontal flip, with per-channel
  mean/std normalisation.
- **Fixed training pipeline for block comparison.** To attribute any change purely to the block,
  everything outside the block is held constant: optimiser SGD with `lr=0.1`, `momentum=0.9`,
  `weight_decay=5e-4`; a cosine-annealed schedule over 200 epochs; the data pipeline, weight
  initialisation, global pooling, and linear classifier head unchanged. A block must keep the
  backbone's interface — a constructor `(in_planes, planes, stride)`, an `expansion` attribute,
  a `forward(x)` returning `planes * expansion` channels, and a shortcut that handles stride or
  channel-count mismatches.
- **Metric:** best test accuracy (%) reached during training, higher better; the only thing that
  varies between runs is the block.

## Code framework

The unit plugs into the residual block slot of an otherwise-fixed CIFAR ResNet. The backbone,
optimiser, schedule, data pipeline, pooling, and classifier all already exist; the only thing to
design is what happens *inside* one block. So the substrate is just the generic residual-block
skeleton — two `3×3` convolutions with batch-norm, a shortcut that matches dimensions when the
stride or channel count changes, and the identity addition — with one empty slot where the new
computation will go.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    """The residual block the ResNet backbone instantiates. Constructor signature,
    `expansion`, the (planes * expansion)-channel output, and the dimension-matching
    shortcut are fixed by the backbone; everything inside is open to design."""
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
        # TODO: any extra sub-module the block's computation will need

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # TODO: the computation we will design, applied to the residual-branch features
        out += self.shortcut(x)
        return F.relu(out)


# existing CIFAR-ResNet training loop the block plugs into (fixed; shown for context)
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:        # RandomCrop(32, pad=4) + flip already applied
        optimizer.zero_grad()
        outputs = model(inputs)                # forward through the ResNet built from CustomBlock
        loss = loss_fn(outputs, targets)       # cross-entropy
        loss.backward()
        optimizer.step()                       # SGD, lr=0.1, momentum=0.9, wd=5e-4, cosine 200ep
```

The residual addition and the dimension-matching shortcut are fixed; the single empty slot is
the computation applied to the residual-branch features before they are added back.
