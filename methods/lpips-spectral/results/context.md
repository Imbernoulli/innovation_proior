# Context: measuring and supervising image similarity beyond per-pixel error (circa 2015-2018)

## Research question

We constantly need to answer "how different are these two images?" — to score a compression
codec, to train an image-synthesis network, to rank super-resolution outputs. The honest
target is *perceptual* difference: two images should count as close exactly when a human
sees them as close. But the metric that everyone actually optimizes is per-pixel squared
error (equivalently PSNR), `||x - y||_2^2`, and it answers a different question. It treats
every pixel as independent and every spatial location as rigidly aligned, so a one-pixel
shift, a small warp, or a faint blur — all nearly invisible to a person — can register as a
large error, while a structured corruption a person would flag immediately can register as
small. The mismatch is not academic: when a generator is *trained* under per-pixel error it
inherits the metric's blind spots. If many plausible outputs are consistent with the input
(the target is multi-modal), the squared-error-minimizing prediction is the *average* of
those modes, and the average of several sharp images is a blurry one — so per-pixel training
losses systematically produce soft, washed-out results that score well numerically and look
wrong.

The precise goal is twofold and coupled. (1) A *measurement* problem: define a distance
between images that tracks human similarity judgments far better than per-pixel error or the
existing hand-designed perceptual metrics, across a wide range of distortions including the
artifacts that modern neural networks actually produce. (2) A *supervision* problem: turn
such a perceptual notion into a differentiable training loss for an image-producing network,
so that what we optimize is finally aligned with what we want to see — sharp, structurally
faithful images — instead of with pixel-aligned squared error. The two are the same coin:
a good perceptual distance is exactly the loss we wish we could train under.

## Background

By the mid-2010s the prevailing wisdom on image quality was splitting in two directions.

The classic, hand-designed side. Per-pixel `L2` / PSNR remained the default regression
loss despite its known failure: blurring an image causes a *large* perceptual change but a
*small* `L2` change, the textbook demonstration that pixel distance and perception diverge.
The community's perceptually-motivated answer was a family of shallow, explicitly engineered
metrics — SSIM (structural similarity), its multiscale variant MS-SSIM, FSIM, HDR-VDP — that
compute local statistics (luminance, contrast, structure) over sliding windows and combine
them. These are demonstrably better than `L2` on many distortions, but they are *shallow
functions of the pixels*, built by hand, and they assume the two images are spatially
registered. Where the distortion involves geometric ambiguity — a small shift, a warp,
resampling — the windowed comparison breaks, because the structures it compares no longer sit
at the same coordinates. They were never designed for that regime.

The emergent, learned side. Separately, the image-synthesis community had stumbled onto
something striking: the *internal activations* of a deep convolutional network trained for
ImageNet classification form a representational space in which Euclidean distance behaves
much more like perceptual distance than raw pixels do. Gatys et al. (2015) showed that the
"content" of an image is well captured by the squared error between high-level VGG feature
maps of two images, and its "style/texture" by the Gram matrices (channel-wise correlation
matrices) of those features — the basis of neural style transfer. Johnson et al. (2016)
turned this into a *training loss*: their feature reconstruction loss is the normalized
squared distance in a fixed VGG layer,

```
ell_feat(yhat, y) = (1 / (C_j H_j W_j)) * || phi_j(yhat) - phi_j(y) ||_2^2 ,
```

where `phi_j` is the activation of VGG layer `j`. Networks trained to minimize this — for
super-resolution, style transfer, conditional synthesis — produced visibly sharper, more
semantically faithful images than pixel-loss networks. The name that stuck for any
feature-space distance used this way was "perceptual loss." But the name outran the
evidence. These losses were used because they *worked*, with hand-chosen layers and uniform
weighting of every channel and layer, and with no grounding in actual human judgments. It
was unknown how perceptual they really are, which architectures or layers matter, whether
ImageNet supervision is necessary, or whether the networks need to be trained at all.

Two more pre-method facts about the world matter here, both knowable before any new metric
exists. First, the multi-modality argument above is a property of `L2`, not of any model:
when the conditional target distribution has several modes `v1, v2`, the value minimizing
expected squared error is the mean `(v1+v2)/2`, which lies between the modes and is blurry —
so an `L2`-trained predictor blurs whenever the problem is genuinely ambiguous. Second,
images live equally in the frequency domain: the discrete Fourier transform decomposes an
image into amplitude and phase per spatial frequency, and the *high-frequency* band is
exactly the fine detail and sharp edges. Pixel and feature losses, computed locally, give
weak and indirect pressure on high frequencies, which is the mechanistic reason their outputs
lose detail — the missing detail is missing energy in the upper part of the spectrum.

A few representational facts that the learned side rests on. Channel activations within a
deep layer have wildly different magnitudes, so a raw Euclidean distance in feature space is
dominated by whichever channels happen to be large, not by whichever channels are
perceptually informative. And the existing perceptual-judgment datasets (LIVE, TID2008,
CSIQ, TID2013) collected many judgments over a *few* reference images and a handful of
parametric distortion types — useful, but too narrow in distortion space to settle questions
about deep features, and they contain none of the artifacts that deep generative/restoration
networks themselves produce.

## Baselines

These are the prior approaches a new perceptual distance — and a new perceptual training
loss — would be measured against and react to.

**Per-pixel `L2` / PSNR.** `d(x,y) = ||x-y||_2^2`; PSNR is its log-scaled monotone cousin.
Cheap, convex, the default regression loss. Core property: treats pixels as independent and
positions as fixed. *Limitation:* it is blind to structure and to spatial ambiguity — blur
is a small `L2` change but a large perceptual one — and as a *training* loss it drives the
predictor to the mean of plausible outputs, producing blur on any multi-modal target.

**SSIM / MS-SSIM / FSIM (Wang et al. 2004; Wang et al. 2003; Zhang et al. 2011).** Hand-
designed perceptual metrics. SSIM compares two images through local luminance, contrast, and
structure terms over a sliding window and multiplies them into a per-window score, averaged
spatially; MS-SSIM does this across a scale pyramid; FSIM weights by phase congruency and
gradient magnitude. They beat `L2`/PSNR on many distortions. *Limitation:* they are shallow,
fixed functions of pixel statistics with no learning, and they assume pixel-level alignment,
so they degrade badly when the distortion is geometric (small shift/warp/resample) — the
windowed statistics no longer line up. They also tend to be hard to use directly as a sharp
training objective.

**VGG feature-reconstruction / perceptual loss (Gatys et al. 2015; Johnson et al. 2016;
Dosovitskiy & Brox 2016).** Distance in the activation space of a fixed ImageNet-pretrained
network, `ell_feat = (1/CHW)||phi_j(yhat) - phi_j(y)||_2^2`, optionally with Gram-matrix
"style" terms. Empirically a far better *training* loss for synthesis than pixel loss —
sharper, more semantic. *Limitation:* every channel and layer is weighted equally and the
layers are picked by hand; the metric has never been calibrated against human judgments, so
it is unknown which of its components carry the perceptual signal, and equal channel
weighting lets large-magnitude but perceptually-irrelevant channels dominate the distance.

**Gradient / multi-scale image losses (Mathieu et al. 2016; Lai et al. 2017).** To fight the
blur of pixel losses directly, Mathieu et al. introduced the Gradient Difference Loss,
penalizing the difference between the *finite-difference image gradients* of prediction and
target,

```
L_gdl = sum_{i,j} | |Y_{i,j}-Y_{i-1,j}| - |Yhat_{i,j}-Yhat_{i-1,j}| |^alpha
                 + | |Y_{i,j-1}-Y_{i,j}| - |Yhat_{i,j-1}-Yhat_{i,j}| |^alpha ,
```

(with `alpha=1` this is an `L1` on horizontal/vertical neighbor differences, a finite-
difference edge term), together with a multi-scale architecture so coarse structure is
supervised too. Lai et al. (LapSRN) replaced `L2` with the robust Charbonnier penalty
`rho(x) = sqrt(x^2 + eps^2)` — a smooth `L1` that is less sensitive to outliers and over-
smoothing than `L2` — inside a Laplacian pyramid. *Limitation:* these are still local,
spatial-domain hand-crafted terms; each fixes one symptom (edge softness, scale, outliers)
without a principled, perceptually-grounded notion of overall similarity, and none of them
puts explicit, global pressure on the full frequency content.

**Frequency-domain losses (Fuoli et al. 2021; Jiang et al. 2021).** Since the detail that
spatial losses miss is high-frequency energy, supervise directly in Fourier space. Fuoli et
al. take the FFT of prediction and target and penalize the `L1` difference of their
amplitude spectra,

```
L_F = (2 / (U V)) * sum_{u=0}^{U/2-1} sum_v | |Yhat|_{u,v} - |Y|_{u,v} | ,
```

(summing over half the spectrum, since the FFT of a real image is Hermitian-symmetric so the
other half is redundant), optionally adding a phase term; Jiang et al.'s Focal Frequency Loss
is the weighted-complex-spectrum sibling. This gives *global* guidance — every coefficient
sees the whole image — and direct pressure on the high-frequency band. *Limitation:* a pure
frequency-amplitude loss discards spatial locality and phase (where things are), so on its
own it neither localizes errors nor captures the higher-order, semantic structure that the
feature-space metrics do; it is a complementary signal, not a complete perceptual distance.

## Evaluation settings

The yardsticks that exist for *measuring* a perceptual distance and for *training* under one.

- **Full-reference IQA datasets** — LIVE, TID2008, CSIQ, TID2013 (Ponomarenko et al. 2015):
  reference images with parametric distortions at several levels, plus human quality scores;
  the de-facto benchmark for similarity metrics. TID2013 is the largest, ~500k judgments over
  ~3000 distortions from 25 images. The natural protocol is rank correlation between a
  metric's distances and the human scores.
- **Two-alternative forced choice (2AFC)** — show a reference patch and two distorted versions
  and ask which is more similar; record the human's binary choice. A metric is scored by how
  often it agrees with the (noisy, multiply-judged) human preference; if a triplet is judged
  with fractions `p, 1-p`, a human achieves `p^2 + (1-p)^2` in expectation, the ceiling.
- **Just-noticeable-difference (JND)** — show a reference and a (possibly distorted) patch and
  ask "same or different"; order pairs by a metric and measure how well it ranks the
  confusable-as-same pairs (precision-recall / mAP). A second, less subjective perceptual test
  used to validate that 2AFC preferences reflect something objective.
- **Distortion space** — both *traditional* distortions (photometric and contrast/saturation
  changes, several noise types, blur, spatial shifts/warps, JPEG-style compression, and
  sequential compositions of these) and *CNN-based* distortions (outputs of autoencoding,
  denoising, colorization, super-resolution networks across architectures and losses) — the
  latter being the artifacts a perceptual metric will actually face in practice.
- **Real-algorithm outputs** — patch triplets sampled from real super-resolution (NTIRE'17 /
  Div2K), frame-interpolation, video-deblurring, and colorization systems; the true test of
  generalization beyond synthetic distortions. Patches are typically `64x64`.
- **Training under a perceptual loss** — image-synthesis / generation tasks with a fixed
  ImageNet-pretrained feature extractor available; data normalized to `[-1, 1]`. In a
  velocity-based generator on the linear path `z_t = (1-t)x + t*eps` with conditional velocity
  `v = eps - x`, a predicted velocity implies a denoised image estimate `x_hat = z_t - t*v_pred`
  on which image-space losses can act, while the velocity MSE stays the correctness anchor.

## Code framework

The pieces that already exist before any new perceptual distance is defined: an off-the-shelf
ImageNet-pretrained convolutional backbone that exposes its intermediate layer activations, a
spatial-domain regression / velocity-MSE loss, and the standard `torch.fft` primitive for
moving an image to the frequency domain. There is one empty slot — the image-space similarity
/ auxiliary-loss function itself, the object to be designed. The harness below is a generic
velocity-based image-generation training step (the network emits a velocity whose implied
denoised image should match the clean image) plus a stub for that loss.

```python
import torch
import torch.nn as nn
import torchvision


class PretrainedFeatures(nn.Module):
    """Off-the-shelf ImageNet-pretrained backbone, frozen, exposing the
    activations of a fixed set of intermediate layers. Channel magnitudes
    across these activations are known to differ widely."""

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList(_split_into_layer_blocks(backbone))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):                  # x in [-1, 1]
        feats, h = [], x
        for s in self.slices:
            h = s(h)
            feats.append(h)                # one activation map per chosen layer
        return feats


def velocity_mse(v_pred, v_target):
    return ((v_pred - v_target) ** 2).flatten(1).mean(1)        # per-sample [B]


def image_quality_loss(v_pred, v_target, x, x_t, t, features):
    """The object to design: a differentiable image-space similarity acting on the
    implied denoised image x_hat = x_t - t*v_pred vs the clean image x, returned
    per-sample. The primitives on hand are the pretrained feature maps, per-pixel
    comparison, and torch.fft.

    returns : per-sample auxiliary loss [B]
    """
    # TODO: the image-space similarity / auxiliary loss we will design.
    pass


# existing velocity-based generator training step the loss plugs into
def train_step(net, x, optimizer, features):
    eps = torch.randn_like(x)                                   # prior sample
    t = torch.rand(x.shape[0], 1, 1, 1, device=x.device)        # noise level
    x_t = (1 - t) * x + t * eps                                 # linear path
    v_target = eps - x                                          # conditional velocity
    v_pred = net(x_t, t)

    base = velocity_mse(v_pred, v_target)                       # correctness anchor
    aux = image_quality_loss(v_pred, v_target, x, x_t, t, features)
    loss = (base + aux).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

The frozen backbone, the velocity-MSE anchor, and `torch.fft` are the substrate; everything
about how to turn them into a perceptual auxiliary similarity lives in the one empty slot.
