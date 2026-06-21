## Research question

Capturing *long-range* dependencies — relating positions that are far apart in space, in time, or in
both — is central to deep networks for vision and sequence data. The two workhorse operators, convolution
and recurrence, act on a *local* neighborhood; they reach distant positions by being stacked deeply so
that a signal propagates step by step. The question is how to directly relate positions at arbitrary
distance in a single, generic operator that works for images, sequences, and video alike.

## Background

**Non-local means and non-local image processing.** A classical denoising filter (Buades et al.)
computes the response at a pixel as a *weighted mean of all pixels in the image*, the weight between
two locations set by the similarity of their surrounding patches — so distant pixels with similar
appearance contribute to each other. This non-local idea is the engine of strong image processing
methods: block-matching denoising (BM3D, which filters groups of similar but non-adjacent patches and
remains a strong baseline even against deep networks), texture synthesis, super-resolution, and
inpainting.

**Self-attention.** A self-attention layer (Vaswani et al.), introduced for machine translation,
computes the response at a sequence position by attending to all positions and taking their weighted
average in an embedding space — Attention(Q,K,V) = softmax(QKᵀ/√d)·V, with Q, K, V linear projections
of the input. It is an all-pairs operation over a sequence, developed for 1-D sequences in language.

**Relation and interaction networks.** Other all-pairs models (Relation Networks, Interaction
Networks) compute a function over every pair of input elements, e.g. via a small network on the
concatenation of two elements' features, for visual reasoning and physical-system modeling. They
establish that processing *all pairs* is useful.

**Properties of existing operators.** A convolution sums a *learned*-weighted input over a
local window; a recurrent step uses only the current and latest positions. A fully-connected layer
relates all positions with *fixed learned weights* — the relationship between two positions is a
function of the model parameters rather than the input data, it requires a fixed input size, and it
loses the correspondence between an input position and its output position.

**Residual learning.** Residual connections (He et al.) let a layer compute a residual added to its
input, y = F(x) + x; a block can be made an exact identity at initialization (by zeroing the residual
branch), which permits inserting a new block into a *pretrained* network without disturbing it.

**Backbones for video.** For video classification, a 2D-CNN-per-frame backbone (C2D, e.g. a ResNet
applied frame-by-frame with temporal information handled only by pooling) can be initialized from an
ImageNet-pretrained ResNet. A 3D ConvNet can be obtained by "inflating" 2D k×k kernels into 3D t×k×k
kernels spanning t frames, initialized from the 2D weights (each of the t temporal planes set to the
pretrained 2D kernel rescaled by 1/t, so a static repeated frame reproduces the 2D result) — the
inflated-3D (I3D) recipe. These 3D convolutions capture spacetime structure.

## Baselines

**3D ConvNets / I3D (Tran et al.; Carreira & Zisserman).** Capture spatiotemporal dependence with 3D
convolutions over space and time, optionally inflated from 2D ImageNet-pretrained kernels. They are
more accurate than CNN+LSTM video models; each kernel spans a local spacetime neighborhood.

**CNN+RNN video models.** Combine a per-frame CNN with a recurrent network over time for temporal
modeling.

**Strong image-task baseline (Mask R-CNN).** A strong region-based detector/segmenter built on a
convolutional backbone, serving as the baseline to test generality beyond video.

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
for the new operator and its insertion into the backbone.

```python
import torch
from torch import nn


class Operator(nn.Module):
    """The new operator that addresses the long-range-dependency problem above."""
    def __init__(self, channels):
        super().__init__()
        # TODO: design the operator

    def forward(self, x):
        # x: (B, C, T, H, W)
        # TODO
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
