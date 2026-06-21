# Context: a frequency-domain loss for image reconstruction and synthesis (circa 2020)

## Research question

Generative image models — autoencoders, VAEs, conditional and unconditional GANs — are trained to
produce an output image that should match a target (in reconstruction) or a target distribution (in
synthesis). The objective that drives this is almost always a *spatial* loss: a per-pixel `L1`/`L2`
term, often combined with a perceptual feature distance and/or an adversarial term. The standing
complaint about these objectives is that the generated images, however good they look globally, tend to
be missing or distorting fine detail — the high-frequency content. The research question is whether the
*frequency representation* of an image gives a handle on this gap that spatial losses structurally
cannot: can a loss defined directly on the 2D spectrum of the image close the frequency mismatch that
pixel- and feature-space losses leave behind, and do so as a drop-in complement to whatever spatial
loss a model already uses?

What makes this nontrivial is that not all frequencies are equally easy. Neural image generators have a
well-documented *spectral bias*: trained under spatial losses, they fit low-frequency, smooth content
quickly and struggle to reproduce high-frequency detail, so the error is concentrated in particular
bands of the spectrum that shift over the course of training and differ from image to image. A naive
frequency loss that weights every spectral component equally would spend most of its gradient on the
many easy, already-matched components and barely touch the few hard ones that actually account for the
perceptual gap. So the real question is not just "measure error in frequency space" but "measure it in a
way that adaptively concentrates on the *hard* frequencies."

## Background

By this time the field state has three relevant strands.

**Spatial reconstruction and perceptual losses.** The default training signal for image generation is a
per-pixel distance — squared error (the negative log-likelihood of a fixed-variance Gaussian decoder)
or absolute error (its Laplace sibling, sharper because it penalizes large residuals less). Both treat
pixels as independent and are minimized, under reconstruction uncertainty, by the blurry conditional
mean (Dosovitskiy & Brox 2016). The standard remedy is to add a distance computed in the feature space
of a pretrained classifier — VGG features — which behaves perceptually and penalizes blur that pixel
losses miss (Johnson et al. 2016; Zhang et al. 2018, LPIPS). These help perceptual quality but remain
*spatial-domain* objectives: they have no explicit notion of which frequency bands are being lost.

**Adversarial losses.** GANs (Goodfellow et al. 2014) add a discriminator that learns to tell real from
generated images and pushes the generator toward realistic high-frequency texture; PatchGAN
discriminators (Isola et al. 2017) concentrate this signal on local patches. Adversarial training does
improve sharpness, but it is unstable, hard to balance against reconstruction terms, and still does not
*directly target* the frequency content — it pressures realism, and frequency fidelity is at best an
indirect consequence.

**Spectral analysis of generators.** A growing line of work documents that generated images differ from
real ones in their frequency statistics in characteristic, measurable ways (Durall et al. 2020; Frank
et al. 2020; Wang et al. 2020 on CNN-generated-image detection), and that this spectral discrepancy is
robust enough to *detect* generated images. The natural reading is that if the discrepancy is
measurable post hoc, it could be penalized *during training* with a loss defined on the spectrum. Prior
attempts to add a frequency term mostly used a plain (unweighted) spectral distance, which inherits the
easy-frequency-domination problem above, or focused on specific bands by hand.

The 2D discrete Fourier transform is the tool that makes the spectrum available. For an `H×W` image
channel `f(x,y)`, `F(u,v) = Σ_x Σ_y f(x,y) · exp(-i2π(ux/H + vy/W))` is a complex spectrum; each
component `F(u,v) = a(u,v) + i·b(u,v)` carries the amplitude and phase of one spatial frequency. An
exact match of two images is equivalent to an exact match of their full complex spectra, so a distance
between spectra is a legitimate reconstruction objective. The open design questions are: what distance
on complex components; how to make it adaptively emphasize hard frequencies rather than easy ones; and
how to keep that emphasis from destabilizing the gradient. The method below — focal frequency loss
(Jiang, Dai, Wu & Loy, ICCV 2021; arXiv:2012.12821) — answers all three with a single
dynamically-reweighted spectral distance, demonstrated as a complement that improves VAE, pix2pix, and
SPADE reconstruction and synthesis.

## Code framework

```python
import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Minimal stub for focal frequency loss.

    Computes a weighted spectral distance between predicted and target
    images using their 2D FFT spectra. Use as a complement to a spatial
    reconstruction loss.
    """

    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        # TODO: form |F_pred - F_target|^2, weight by detached error^alpha,
        # normalize to [0, 1], and return the weighted mean.
        raise NotImplementedError
```
