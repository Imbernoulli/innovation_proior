## Research question

Capturing *long-range* dependencies — relating positions that are far apart in space, in time, or in
both — is central to deep networks for vision and sequence data. Yet the two workhorse operators both
act only on a *local* neighborhood: a convolution mixes a small spatial window, a recurrent step mixes
the current position with the immediately preceding one. The only way these operators reach distant
positions is to be *stacked deeply*, so that a signal propagates step by step across the data. The
precise problem is that this progressive, local-only propagation is a poor way to model long-range
dependence: it is computationally inefficient (you pay for many layers just to connect two faraway
points), it is hard to optimize (deep stacks have the well-known signal-propagation difficulties), and
it makes *multi-hop* interactions — where information must travel back and forth between distant
positions — awkward. The goal is a single, simple, generic operator that connects *any* two positions
*directly*, regardless of how far apart they are, that works for images, sequences, and video alike,
preserves the input's spatial/temporal layout and variable size, and can be dropped into existing
networks alongside convolutions.

## Background

**Non-local means and non-local image processing.** A classical denoising filter (Buades et al.)
computes the response at a pixel as a *weighted mean of all pixels in the image*, the weight between
two locations set by the similarity of their surrounding patches — so distant pixels with similar
appearance contribute to each other. This non-local idea is the engine of strong image processing
methods: block-matching denoising (BM3D, which filters groups of similar but non-adjacent patches and
remains a strong baseline even against deep networks), texture synthesis, super-resolution, and
inpainting. The lesson from this literature is that letting far-apart, *appearance-similar* locations
interact directly is powerful — and it had been largely overlooked in modern neural networks for
vision.

**Self-attention.** A self-attention layer (Vaswani et al.), introduced for machine translation,
computes the response at a sequence position by attending to all positions and taking their weighted
average in an embedding space — Attention(Q,K,V) = softmax(QKᵀ/√d)·V, with Q, K, V linear projections
of the input. It is an all-pairs operation over a sequence. It is, in effect, a learned, embedded,
softmax-normalized version of the same "weighted average over all positions by pairwise similarity"
idea as non-local means, but it had only been developed for 1-D sequences in language.

**Relation and interaction networks.** Other all-pairs models (Relation Networks, Interaction
Networks) compute a function over every pair of input elements, e.g. via a small network on the
concatenation of two elements' features, for visual reasoning and physical-system modeling. They
establish that processing *all pairs* is useful, though they frame it around attention/relation rather
than around non-locality per se.

**How this differs from existing operators.** A convolution sums a *learned*-weighted input over a
local window; a recurrent step uses only the current and latest positions. A fully-connected layer
does relate all positions, but with *fixed learned weights* — the relationship between two positions is
*not* a function of the input data, it requires a fixed input size, and it loses the correspondence
between an input position and its output position. The operator sought here differs from all three: its
pairwise weight is computed *from the data* (a function of the two positions' features), it spans *all*
positions (unlike conv/recurrent), and it preserves variable size and positional correspondence
(unlike fc).

**Residual learning.** Residual connections (He et al.) let a layer compute a residual added to its
input, y = F(x) + x; a block can be made an exact identity at initialization (by zeroing the residual
branch), which is what permits inserting a new block into a *pretrained* network without disturbing it.

**Backbones for video.** For video classification, a 2D-CNN-per-frame backbone (C2D, e.g. a ResNet
applied frame-by-frame with temporal information handled only by pooling) can be initialized from an
ImageNet-pretrained ResNet. A 3D ConvNet can be obtained by "inflating" 2D k×k kernels into 3D t×k×k
kernels spanning t frames, initialized from the 2D weights (each of the t temporal planes set to the
pretrained 2D kernel rescaled by 1/t, so a static repeated frame reproduces the 2D result) — the
inflated-3D (I3D) recipe. These 3D convolutions capture spacetime structure but are computationally
heavy and still local in spacetime.

## Baselines

**3D ConvNets / I3D (Tran et al.; Carreira & Zisserman).** Capture spatiotemporal dependence with 3D
convolutions over space and time, optionally inflated from 2D ImageNet-pretrained kernels. They are
more accurate than CNN+LSTM video models, but each kernel is still *local* in spacetime, so long-range
spacetime interactions again require deep stacking; and 3D convolutions are expensive in parameters and
FLOPs. The gap: locality and cost.

**CNN+RNN video models.** Combine a per-frame CNN with a recurrent network over time. They inherit the
recurrent operator's local, step-by-step temporal modeling and its optimization difficulties.

**Strong image-task baseline (Mask R-CNN).** For images, a strong region-based detector/segmenter built
on a convolutional backbone. Its receptive field is built up by local convolutions, so distant-context
interactions are again indirect. It serves as the baseline a generic non-local component would augment
to test generality beyond video.

## Evaluation settings

The primary benchmark is video action classification on Kinetics (~240k training videos, 400 action
classes) and Charades (longer, multi-label activity videos), with top-1 / top-5 accuracy, using RGB
frames only (no optical flow, no multi-scale testing). Backbones are ImageNet-pretrained ResNet-50 /
ResNet-101. Clips of 32 frames at 224×224 are sampled from each video; inference averages softmax
scores over several clips and uses fully-convolutional spatial testing. Models are also compared on
parameter count and FLOPs relative to the C2D baseline. As a generality check, image tasks on COCO
(object detection, instance segmentation, keypoint/pose) measure accuracy added on top of a strong
detector at small extra cost.

## Code framework

The available primitives are: standard convolutions (2D and 3D / 1×1×1), batch normalization, residual
blocks, matrix multiplication and softmax, and ImageNet-pretrained ResNet backbones (used directly as a
per-frame C2D model or inflated to I3D). The scaffold builds such a backbone and leaves empty the slot
for the new operator that relates all positions directly, and the wrapper that lets it be inserted into
a pretrained network without disturbing it.

```python
import torch
from torch import nn


def pairwise(x_i_embed, x_j_embed):
    # TODO: a data-dependent scalar relating every pair of positions (i, j)
    raise NotImplementedError


class AllPairsOp(nn.Module):
    """An operator whose output at position i is a normalized weighted combination of a
    content embedding at EVERY position j, the weights computed from the data."""
    def __init__(self, channels):
        super().__init__()
        self.g = nn.Conv3d(channels, channels, kernel_size=1)   # content embedding g(x_j)
        # TODO: embeddings for the pairwise function and the output projection

    def forward(self, x):
        # x: (B, C, T, H, W). Flatten positions, relate all pairs, recombine.
        # TODO: y_i = (1/C(x)) * sum_j pairwise(x_i, x_j) * g(x_j)
        raise NotImplementedError


class InsertableBlock(nn.Module):
    """Wrap the operator so it can be dropped into a PRETRAINED network unchanged."""
    def __init__(self, channels):
        super().__init__()
        self.op = AllPairsOp(channels)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)  # back to C channels
        # TODO: make the block an identity at initialization

    def forward(self, x):
        # TODO: residual combination of the operator's output with the input
        raise NotImplementedError


class VideoBackbone(nn.Module):
    """ImageNet-pretrained ResNet as per-frame C2D (or inflated to I3D)."""
    def __init__(self, depth=50, inflate=False):
        super().__init__()
        self.stages = build_resnet_video(depth, inflate)   # res2..res5, temporal pooling
        self.head = nn.Linear(2048, 400)

    def forward(self, clip):
        # clip: (B, 3, T, 224, 224)
        # TODO: run stages; where might the new blocks be inserted?
        raise NotImplementedError
```
