## Research question

We constantly need to answer "how different are these two images?" — to decide whether a
compressed image is good enough, to score a super-resolution or colorization output against
ground truth, or to use the distance itself as a *training loss* for an image-synthesis network.
The trouble is that the obvious distance, per-pixel Euclidean (and the monotone Peak
Signal-to-Noise Ratio derived from it), disagrees badly with what people see. Blurring an image
can leave the average pixel error deceptively small while visibly destroying high-frequency
detail; conversely, shifting an image by one pixel, or warping it slightly, changes pixels a lot
while a person sees essentially the same content. The per-pixel measure treats each pixel as
independent and so is blind to structure, to spatial ambiguity, and to the fact that some changes
are perceptually invisible while others are glaring.

The goal is a *perceptual distance*: a function `d(x, x0)` on pairs of images whose ordering
agrees with human judgments of similarity — small exactly when people say "these look the same,"
large when they say "these look different" — across the kinds of distortions that actually occur
(blur, noise, compression, color and contrast shifts, small geometric warps) and across the
artifacts produced by real algorithms (super-resolution, frame interpolation, deblurring,
colorization). It has to be cheap to evaluate and, ideally, differentiable, so it can double as a
loss. Building such a function is hard for three reasons: human similarity judgments depend on
high-order image structure, they are context-dependent (a red circle can be judged closer to a
red square or to a blue circle depending on what "respect of similarity" the viewer holds in
mind), and they may not even satisfy the axioms of a metric. The question is how to get a
distance that tracks perception despite all three.

## Background

The field state has two strands that have not been connected.

On one side is a long line of **hand-designed perceptual metrics**. The most widely used is the
Structural Similarity index, **SSIM** (Wang, Bovik, Sheikh & Simoncelli 2004), which abandons
per-pixel comparison for a comparison of *local windowed statistics*. Over a sliding window it
computes three terms — luminance `l = (2 μ_x μ_y + C1) / (μ_x² + μ_y² + C1)`, contrast
`c = (2 σ_x σ_y + C2) / (σ_x² + σ_y² + C2)`, and structure `s = (σ_xy + C3) / (σ_x σ_y + C3)` —
and combines them as `SSIM = l^α c^β s^γ`, with stabilizers `C1 = (k1 L)²`, `C2 = (k2 L)²`
(`k1 = 0.01`, `k2 = 0.03`, `L` the pixel dynamic range) and `C3 = C2/2`. Its multiscale variant
MS-SSIM (Wang, Simoncelli & Bovik 2003) evaluates SSIM across a pyramid; FSIM (Zhang et al. 2011)
uses phase congruency and gradient magnitude. These are shallow, fixed formulas — a handful of
local-statistic ratios — and they were not built to cope with situations where spatial ambiguity
or geometric distortion dominates, exactly the situations modern image algorithms create.

On the other side is the discovery, from the representation-learning community, that the
**internal activations of deep convolutional classifiers transfer broadly**. Networks trained for
ImageNet classification — AlexNet (Krizhevsky, Sutskever & Hinton 2012), the deeper VGG (Simonyan
& Zisserman 2014), the tiny SqueezeNet (Iandola et al. 2016) — learn intermediate feature maps
that are useful far beyond classification. Crucially, several image-synthesis methods began
*measuring distance in this feature space* and using it as a training objective, calling it a
"perceptual loss": Gatys, Ecker & Bethge (2016) for style transfer, Johnson, Alahi & Fei-Fei
(2016) for fast style transfer and super-resolution, and Dosovitskiy & Brox (2016, "DeePSiM")
for image generation. They reported results that look far better than per-pixel losses produce —
sharper textures, more plausible detail. There is also a parallel observation from neuroscience
(Yamins & DiCarlo 2016) that representations trained for vision tasks model primate visual-cortex
activity, and that the better the task representation, the better the cortical model.

The motivating, diagnostic facts that set up the problem are: (1) traditional metrics `ℓ2`, SSIM,
and FSIM agree with humans only modestly and break under geometric distortion; (2) the
feature-space "perceptual losses" are asserted to be perceptual *purely from qualitative output
quality*, with no measurement against a large body of human similarity judgments; and (3) it is
unknown whether VGG is special, whether ImageNet supervision is required, whether the network must
be trained at all, or how a large set of context-dependent pairwise judgments should be turned into
a reusable distance rather than a triplet classifier.

## Baselines

These are the prior measures a new perceptual distance would be compared against and would react
to.

**Per-pixel `ℓ2` / PSNR.** `d(x, x0) = ||x − x0||²` (PSNR is a log-rescaling of the per-pixel
mean squared error). Simple, differentiable, ubiquitous as a regression loss. **Gap:** assumes
pixel-wise independence, so it is blind to structure; blur is cheap under it though obviously
perceptual, and a one-pixel shift is expensive under it though imperceptible. It does not order
distortions the way people do.

**SSIM / MS-SSIM / FSIM (Wang et al. 2004; Wang et al. 2003; Zhang et al. 2011).** Compare local
windowed statistics (means, variances, covariances; phase and gradient for FSIM) rather than raw
pixels, capturing luminance/contrast/structure changes a person would notice. A real improvement
over `ℓ2` on photometric distortions. **Gap:** they are shallow, hand-built, fixed functions of a
few local statistics; they were not designed for spatial ambiguity or geometric distortion, where
they are observed to disagree sharply with humans, and they cannot be adapted to a body of human
judgments because there is nothing in them to learn.

**Feature-space "perceptual loss" on a fixed pretrained network (Johnson et al. 2016; Gatys et
al. 2016; Dosovitskiy & Brox 2016).** Take a network `φ` pretrained for classification (typically
16-layer VGG), freeze it, and define distance as the squared error between its feature maps at a
chosen layer `j`: `ℓ_feat = (1 / (C_j H_j W_j)) · ||φ_j(ŷ) − φ_j(y)||²₂`. Used as a training
objective, this produces visibly better synthesis than per-pixel losses. **Gap:** it is used and
asserted to be perceptual on the strength of output quality alone — never measured against a
large dataset of human similarity judgments; it compares *raw, un-normalized* feature activations
(so a few high-magnitude channels dominate the squared error), it picks the layer and any
weighting by hand, and it is uncalibrated to human data. Whether it actually orders distortions
the way people do, whether the choice of network or its training task matters, and whether the
network needs to be trained at all, are all open.

**Directly fitting a similarity predictor to human judgments.** Collect human pairwise
preferences and train a network to predict, for a triplet `(x, x0, x1)`, which of `x0, x1` is
closer to `x`. **Gap:** similarity judgments are context-dependent and pairwise, and may not obey
metric axioms; a predictor trained directly on triplets risks learning the quirks of that triplet
distribution rather than a reusable image distance.

## Evaluation settings

The natural yardsticks for a perceptual distance are datasets and protocols that test whether a
score orders images the way people do.

- **Full-Reference Image Quality Assessment datasets**: LIVE (Sheikh et al. 2006), TID2008
  (Ponomarenko et al. 2009), CSIQ (Larson & Chandler 2010), and TID2013 (Ponomarenko et al. 2015).
  TID2013 has ~500k judgments over 3000 distortions (25 images × 24 distortion types × 5 levels).
  These have been the de-facto benchmarks for similarity metrics; the standard score is the
  Spearman rank correlation between a metric's distances and the human mean-opinion scores.
- **A two-alternative forced-choice (2AFC) patch protocol**: take a reference patch `x`, apply two
  distortions to get `x0, x1`, and record which a human says is closer to `x` (`h ∈ {0,1}`).
  In the usual data-loader convention, `h = 0` means `x0` is chosen and `h = 1` means `x1` is
  chosen; aggregated judgments can be fractional.
  Patch-level judgments are a natural way to focus on low-level similarity while covering a large
  distortion space; an algorithm is scored by how often its ordering agrees with the human votes
  (with multiple votes per triplet, partial credit by vote fraction). The achievable
  human-consistency ceiling itself is below 1: if humans split `(p, 1−p)`, a human agent scores
  `p² + (1−p)²` in expectation.
- **A just-noticeable-difference (JND) protocol**: show a reference and a distorted patch briefly
  and ask "same or different"; a good metric should rank confusable pairs as close. Scored by area
  under the precision-recall curve (mAP). This is a *second, less subjective* perceptual task,
  used to check that a metric tuned on 2AFC generalizes to a different test of perception.
- **Distortion families and real algorithm outputs**: parametrized "traditional" distortions
  (photometric/contrast/saturation shifts, several noise types, blur, spatial shifts and
  warps, compression), distortions produced by passing patches through many small CNNs (to
  mimic deep-method artifacts), and real outputs from super-resolution, frame interpolation,
  video deblurring, and colorization systems — the actual deployment setting for a perceptual
  metric. Candidate learned feature spaces should be compared against controls so that any gain is
  not mistaken for a property of architecture alone.

## Code framework

A perceptual distance is computed by passing both images through a feature extractor and reducing
the per-layer feature differences to a scalar; if it is to be learnable, there is also a training
loop that consumes human triplet judgments. The pieces that already exist are the pretrained
backbone (any ImageNet classifier, frozen, exposed as a stack of intermediate feature maps), the
data pipeline that yields triplets `(x, x0, x1, h)`, and a standard optimizer and training loop.
What is *not* settled — and is exactly what is to be designed — is how the per-layer feature
differences are turned into a distance, and how (if at all) that reduction is fit to the human
judgments. Those are the empty slots.

```python
import torch
import torch.nn as nn


class PretrainedBackbone(nn.Module):
    """An ImageNet-pretrained classifier (e.g. AlexNet / VGG / SqueezeNet), frozen,
    exposed as a list of intermediate feature maps. Already exists; not the contribution."""

    def __init__(self):
        super().__init__()
        # load a pretrained classifier and slice it into L stages
        # self.slices = [...]; freeze all parameters
        pass

    def forward(self, x):
        feats = []
        # h = x; for each slice: h = slice(h); feats.append(h)
        return feats  # list of (N, C_l, H_l, W_l) feature maps, one per layer l


class PerceptualDistance(nn.Module):
    """Maps a pair of images to a scalar distance using the backbone's features.
    The reduction from per-layer feature differences to a scalar is the open problem."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # TODO: optional trainable components for the reduction

    def forward(self, x0, x1):
        feats0 = self.backbone(x0)
        feats1 = self.backbone(x1)
        # TODO: turn the feature differences into a single scalar distance.
        raise NotImplementedError


def train_on_human_judgments(distance, triplet_loader, optimizer):
    """Optional: fit whatever parameters the distance has to human preferences.
    Each batch yields (x, x0, x1, h), with h=0 for x0 and h=1 for x1."""
    for x, x0, x1, h in triplet_loader:
        d0 = distance(x, x0)
        d1 = distance(x, x1)
        # TODO: a loss that pushes the distances to agree with the human preference h.
        loss = None  # TODO
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The frozen backbone, the triplet data, and the optimizer are given; the reduction inside
`PerceptualDistance.forward` and the fitting rule in `train_on_human_judgments` remain open.
