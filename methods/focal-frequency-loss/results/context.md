# Context: a frequency-domain loss for image reconstruction and synthesis (circa 2020)

## Research question

Generative image models — autoencoders, VAEs, conditional and unconditional GANs — are trained to
produce an output image that should match a target (in reconstruction) or a target distribution (in
synthesis). The objective that drives this is almost always a *spatial* loss: a per-pixel `L1`/`L2`
term, often combined with a perceptual feature distance and/or an adversarial term. An image also has a
*frequency representation*: its 2D spectrum, in which low-frequency components carry smooth global
structure and high-frequency components carry fine detail. The research question is whether a loss
defined directly on the 2D spectrum of the image can serve as a training signal for image generation,
used as a drop-in complement to whatever spatial loss a model already uses.

Neural image generators trained under spatial losses show a *spectral bias*: they fit low-frequency,
smooth content quickly while high-frequency detail comes more slowly, so the error between generated and
target images is distributed unevenly across the spectrum and changes over the course of training and
from image to image. A loss defined on the spectrum therefore has to decide how to treat the different
frequency components.

## Background

By this time the field state has three relevant strands.

**Spatial reconstruction and perceptual losses.** The default training signal for image generation is a
per-pixel distance — squared error (the negative log-likelihood of a fixed-variance Gaussian decoder)
or absolute error (its Laplace sibling, which penalizes large residuals less). Both treat pixels as
independent and are minimized, under reconstruction uncertainty, by the conditional mean (Dosovitskiy &
Brox 2016). A common addition is a distance computed in the feature space of a pretrained classifier —
VGG features — which behaves perceptually (Johnson et al. 2016; Zhang et al. 2018, LPIPS). These are
*spatial-domain* objectives.

**Adversarial losses.** GANs (Goodfellow et al. 2014) add a discriminator that learns to tell real from
generated images and pushes the generator toward realistic high-frequency texture; PatchGAN
discriminators (Isola et al. 2017) concentrate this signal on local patches. Adversarial training
pressures overall realism rather than frequency content directly.

**Spectral analysis of generators.** A growing line of work documents that generated images differ from
real ones in their frequency statistics in characteristic, measurable ways (Durall et al. 2020; Frank
et al. 2020; Wang et al. 2020 on CNN-generated-image detection), and that this spectral discrepancy is
robust enough to *detect* generated images. Since the discrepancy is measurable post hoc, it can also be
penalized *during training* with a loss defined on the spectrum. Earlier frequency terms have used a
plain (unweighted) spectral distance, or focused on specific bands chosen by hand.

The 2D discrete Fourier transform is the tool that makes the spectrum available. For an `H×W` image
channel `f(x,y)`, `F(u,v) = Σ_x Σ_y f(x,y) · exp(-i2π(ux/H + vy/W))` is a complex spectrum; each
component `F(u,v) = a(u,v) + i·b(u,v)` carries the amplitude and phase of one spatial frequency. An
exact match of two images is equivalent to an exact match of their full complex spectra, so a distance
between spectra is a legitimate reconstruction objective. The open design questions are: what distance
to use on complex components, how to treat frequencies that differ in how well they are currently
matched, and how to keep a spectral term's gradient stable. A loss along these lines can be tested as a
complement on VAE, pix2pix, and SPADE reconstruction and synthesis.

## Code framework

```python
import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):
    """Minimal stub for a spectral training loss.

    Computes a distance between predicted and target images using their
    2D FFT spectra. Use as a complement to a spatial reconstruction loss.
    """

    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        # TODO: form the per-component spectral distance and return a scalar loss.
        raise NotImplementedError
```
