# Context: measuring perceptual image similarity (circa 2016-2018)

## Research question

Comparing two data items is one of the most basic operations in computing, and for most data
types it is easy — Hamming distance for bit strings, edit distance for text, Euclidean distance
for vectors. Images are the exception. Visual patterns are extremely high-dimensional and highly
correlated, and the notion of "similar" we actually care about is *perceptual*: two images should
count as similar when a human looking at them judges them to be alike, not when their pixel arrays
happen to be close. The two notions come apart badly. Blurring can produce a large perceptual
change while leaving the average pixel error deceptively small; shifting an image by a single pixel
can produce a large per-pixel error while looking essentially identical. A useful "perceptual
distance" would assign small distance exactly when a human perceives the two images as similar, and
it would have to do this across the kinds of corruption that real systems produce — photometric and
geometric distortions, noise, compression, and the characteristic artifacts of neural-network image
generators.

The goal, then, is a function `d(x, x0)` on pairs of image patches whose ordering of pairs agrees
with human perceptual judgments. What makes this hard is not just engineering. Human similarity
judgments (1) depend on high-order image structure rather than local pixel statistics; (2) are
context-dependent — there are many "respects of similarity" a person can hold in mind at once (is a
red circle closer to a red square or a blue circle?), so the "right" answer shifts with context;
and (3) may not even satisfy the axioms of a distance metric — Tversky's work on similarity shows
human judgments can be asymmetric and violate the triangle inequality. So fitting a function
directly to raw human judgments is not obviously tractable, and a method has to find some way around
the context-dependence and the sheer dimensionality of the image space.

## Background

By this time the field state has two relevant strands.

**Hand-designed image-quality metrics.** The longstanding response to "pixel distance is not
perceptual" is a family of structural metrics. The Structural Similarity Index (SSIM; Wang, Bovik,
Sheikh & Simoncelli 2004) compares two images through local luminance, contrast, and structure
terms,

```
SSIM(x,y) = [(2 mu_x mu_y + C1)(2 sigma_xy + C2)] / [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)],
C1 = (K1 L)^2,  C2 = (K2 L)^2,  K1 = 0.01,  K2 = 0.03,  L = dynamic range,
```

computed in sliding windows and pooled. Multi-Scale SSIM (Wang, Simoncelli & Bovik 2003) applies
this across a scale pyramid; FSIM (Zhang et al. 2011) weights a phase-congruency feature map; HDR-VDP
(Mantiuk et al. 2011) models visibility. These are shallow, fixed functions of low-level statistics.
They were validated on full-reference IQA datasets (LIVE, TID2008, CSIQ, TID2013) built around a few
images and a handful of distortion types, and on those they beat plain pixel error. A known caveat:
they were not designed for situations where spatial/geometric ambiguity is a large factor (Sampat et
al. 2009 on complex-wavelet SSIM), and on geometric distortions they degrade.

**Deep features used as a "perceptual loss."** A separate, empirically driven discovery had recently
swept image synthesis: the *internal activations* of a deep convolutional network trained for
ImageNet classification, although never trained for it, turn out to be a powerful representational
space for many other tasks. In particular, several image-generation systems replaced the per-pixel
training loss with a distance computed in VGG feature space and got dramatically better-looking
outputs. The motivating empirical fact behind this is sharp and well documented: training a generator
under a per-pixel `l2` (or `l1`) loss yields *over-smoothed, blurry* images (Dosovitskiy & Brox 2016),
because squared pixel error averages over the many plausible high-frequency completions instead of
committing to one. Measuring the error in a deep feature space instead produces sharp, natural-looking
results. A diagnostic observation accompanies it: when you optimize an image to minimize the VGG
feature-reconstruction loss at *early* layers, the result is visually indistinguishable from the
target; minimize it at *deeper* layers and the image keeps the content and overall spatial structure
but not the exact color, texture, or shape (Johnson et al. 2016, reproducing Dosovitskiy & Brox
2016). And a cautionary one: minimizing feature distance *alone* as a generator objective produces
high-frequency artifacts, "because for each natural image there are many non-natural images mapped to
the same feature vector" (Dosovitskiy & Brox 2016) — so feature distance had been treated as one term
to be combined with others, never examined on its own as a candidate *perceptual distance*.

These deep features come from standard classification backbones — VGG (Simonyan & Zisserman 2014),
the shallower AlexNet (Krizhevsky 2012; Krizhevsky 2014), and the lightweight SqueezeNet (Iandola et
al. 2016) — and there is also a growing set of networks of the *same* architecture trained without
class labels: self-supervised objectives such as solving jigsaw puzzles (Noroozi & Favaro 2016),
cross-channel/colorization prediction and split-brain autoencoders (Zhang et al. 2016, 2017),
context prediction and inpainting (Doersch et al. 2015; Pathak et al. 2016), learning from video and
motion (Wang & Gupta 2015; Pathak et al. 2017; Agrawal et al. 2015), and adversarial feature
learning (BiGAN; Donahue et al. 2016); plus purely data-dependent unsupervised initializations
(stacked k-means; Krähenbühl et al. 2015). The prevailing wisdom is that representations good at one
high-level task tend to transfer to others, and there is a parallel finding in neuroscience that
representations trained on vision tasks also model macaque visual cortex (Yamins & DiCarlo 2016).
Whether any of this transfer extends to *low-level perceptual similarity*, and what about a network
(architecture, training task, or merely having weights at all) would be responsible, is unestablished.

## Baselines

A new perceptual distance would be measured against, and reacts to, the following.

**Per-pixel `l2` / PSNR.** `d(x,y) = mean_i (x_i - y_i)^2`, or its log-scaled form PSNR. Treats each
pixel independently. *Limitation:* a large pixel error need not correspond to a large perceptual
change and vice versa — blur produces a large perceptual change with small `l2`, a one-pixel shift
produces a large `l2` with almost no perceptual change. It is blind to structure because it never
looks beyond a single pixel.

**SSIM / MS-SSIM / FSIM (Wang 2004; Wang 2003; Zhang 2011).** Hand-built combinations of local
luminance/contrast/structure statistics (formula above). *Limitation:* fixed, shallow functions of
low-order local statistics with a small number of designer-chosen constants; they capture some
structure that pixel error misses but were not designed for geometric distortions and correlate only
moderately with human judgments on richer distortion sets; nothing in them is learned from human data.

**Feature-reconstruction "perceptual loss" (Johnson et al. 2016; Gatys et al. 2016; Dosovitskiy &
Brox 2016).** A distance computed as the (squared, normalized) Euclidean distance between deep feature
maps,

```
ell_feat^{phi,j}(y_hat, y) = (1 / (C_j H_j W_j)) || phi_j(y_hat) - phi_j(y) ||_2^2,
```

summed over chosen layers `j` of a fixed pretrained network `phi` (Gatys additionally matches Gram
matrices `G^phi_j(x)_{c,c'} = (1/(C_j H_j W_j)) sum_{h,w} phi_j(x)_{h,w,c} phi_j(x)_{h,w,c'}` for
style). *Limitation:* this object was constructed and used as a *training* signal for generators; it
was never validated as a *metric* against human perceptual judgments, and it is used raw — the
activations of different channels and layers enter at wildly different magnitudes with no calibration,
and there is no account of which channels or layers actually matter for human perception, nor any
human grounding at all. It is a plausible perceptual quantity that nobody had measured against people.

**Earlier deep-net image-assessment work (Kim & Lee 2017; Gao et al. 2017; Amirshahi et al. 2017;
Talebi & Milanfar 2017; Berardino et al. 2017).** Various uses of CNNs for image quality: training a
CNN on low-level differences, comparing internal activations with multi-scale post-processing, or
no-reference aesthetic scoring. *Limitation:* narrower datasets and mostly single-image *quality*
assessment rather than pairwise perceptual *similarity*, and no systematic study across architectures
and training signals.

**Existing IQA datasets (LIVE, TID2008, CSIQ, TID2013).** The de-facto yardsticks: many human
judgments collected on a few reference images and a handful of synthetic distortion types (TID2013:
~500k judgments over 3000 distortions from 25 images × 24 distortion types × 5 levels). *Limitation:*
they target full-image *quality*, contain few distortion types, include no CNN-generated artifacts and
few geometric distortions, and cover the image space thinly (few images, many ratings each).

## Evaluation settings

The natural way to test a candidate perceptual distance is whether its ordering of pairs agrees with
people, over a wide enough space of corruptions.

- **Two-alternative forced choice (2AFC).** Take a reference patch `x`, produce two distorted versions
  `x0, x1`, and ask a human which is closer to `x`, recording `h in {0,1}`. A dataset is a set of
  triplets-with-judgment `(x, x0, x1, h)`. A metric scores a triplet correctly when it ranks the
  human-preferred patch as closer. Because each triplet gets several judgments, scoring is fractional:
  if a fraction `p` of humans pick one side, an oracle predicting the majority gets `max(p, 1-p)`, and
  a human agent (choosing with the same distribution `p`) scores `p^2 + (1-p)^2` in expectation — the
  natural ceiling.
- **Just-noticeable-difference (JND).** Show a reference then a distorted patch and ask "same or
  different"; a good metric should rank confusable (low-distance) pairs as more likely "same". Scored
  by precision-recall / area under the curve (mAP). Serves as a second, less cognitively-penetrable
  perceptual test to check that 2AFC is capturing something objective.
- **Distortion space to cover.** *Traditional* distortions (photometric: contrast/saturation; noise;
  blur; spatial shifts and corruptions; compression), composed in pairs to enlarge the space; and
  *CNN-based* distortions, generated by training many small denoising-autoencoder-style networks
  across tasks/architectures/losses (autoencoding, denoising, colorization, super-resolution) to
  reproduce the artifacts real generators make. Plus held-out *real-algorithm* outputs —
  super-resolution (NTIRE 2017 / Div2K), frame interpolation, video deblurring, colorization — sampled
  as `64x64` patch triplets, the real-world use case.
- **Patches, not full images**, at `64x64`, drawn from large image collections (MIT-Adobe 5k for
  train, RAISE1k for validation), collected in-the-wild on Amazon Mechanical Turk with sentinel
  checks. Patches focus the test on low-level similarity and dodge the high-level "respects of
  similarity" ambiguity, and the `64x64` color-patch space is already a ~12k-dimensional domain.
- **Networks to probe**, applied as candidate distances out of the box: SqueezeNet, AlexNet, VGG
  (supervised); the self-supervised and unsupervised nets listed above; and randomly-initialized
  networks of the same architecture as a control. Also cross-checked against the standard TID2013 IQA
  benchmark via Spearman correlation, at several input resolutions.

## Code framework

A perceptual distance is a function of two image patches built from primitives that already exist:
pretrained convolutional backbones that expose intermediate activations, tensor operations for
pairwise comparisons and reductions, and a generic training loop that can fit a small binary
predictor to 2AFC labels if the distance ends up needing learned parameters. What is not settled is
how the stack of feature maps for two patches becomes a single scalar that tracks human perception,
and what, if anything, should be fitted from the judgment data.

```python
import torch
import torch.nn as nn


class FeatureTrunk(nn.Module):
    """A fixed, pretrained convolutional backbone (e.g. VGG / AlexNet / SqueezeNet).
    Frozen by default. Exposes the activations of a chosen list of L layers."""

    def __init__(self, pretrained=True, requires_grad=False):
        super().__init__()
        # load a standard backbone; pick out L intermediate layers; freeze if requires_grad=False
        ...

    def forward(self, x):
        # return a list [f^0, ..., f^{L-1}] of feature maps, f^l of shape (N, C_l, H_l, W_l)
        ...


class PerceptualDistance(nn.Module):
    """Maps a pair of image patches to a scalar distance, on top of a frozen FeatureTrunk.
    How the per-layer feature stacks become one perceptually-meaningful number — and what,
    if anything, is learned from human judgments — is exactly what we have to design."""

    def __init__(self, trunk):
        super().__init__()
        self.trunk = trunk
        # TODO: any components the distance needs (the object we will define here)

    def forward(self, x0, x1):
        feats0 = self.trunk(x0)            # [f0^0, ..., f0^{L-1}]
        feats1 = self.trunk(x1)            # [f1^0, ..., f1^{L-1}]
        # TODO: turn the two per-layer feature stacks into a single scalar distance d(x0, x1)
        raise NotImplementedError


# existing training loop, used only if the distance has any fitted component
def train_on_judgments(dist, judgment_predictor, data_loader, optimizer):
    bce = nn.BCELoss()
    for x, x0, x1, h in data_loader:          # triplet + human judgment h in {0,1}
        d0 = dist(x, x0)
        d1 = dist(x, x1)
        h_hat = judgment_predictor(d0, d1)     # map the distance pair to a probability in (0,1)
        loss = bce(h_hat, h)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: any constraint the fitted parameters must satisfy is enforced here
```

The feature-to-scalar rule and any fitted parts remain undefined.
