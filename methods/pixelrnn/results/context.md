# Context

## Research question

How can we estimate the probability distribution `p(x)` of natural images in a way that is simultaneously *tractable* — letting us compute the exact likelihood of an image and sample new ones — and *expressive* enough to capture the highly nonlinear, long-range, and multimodal correlations between pixels that make natural images what they are?

Natural images are high-dimensional and highly structured, so a good density model would unlock image compression, inpainting, deblurring, and generation (and, conditioned on side information, text-to-image or future-frame prediction). The central obstacle is the trade-off between expressiveness and tractability: powerful models tend to require intractable inference, while tractable models tend to be too weak to capture the dependencies. A satisfactory method must keep an *exact, tractable likelihood* while being expressive enough to model every pixel-to-pixel dependency, including the dependencies between the color channels within a single pixel, with no independence shortcuts.

## Background

**Latent-variable models.** Most work on generative image modeling uses stochastic latent-variable models such as the VAE (Kingma & Welling, 2013; Rezende, Mohamed & Wierstra, 2014), which aim to extract meaningful latent representations. They come with an intractable inference step — the likelihood is only bounded (the ELBO), not computed exactly — and they typically impose independence structure in the latents, which can limit how faithfully the full pixel dependency structure is modeled.

**Autoregressive / fully-visible models.** An alternative that is exactly tractable casts the joint distribution of the pixels as a product of conditionals: `p(x) = Π_i p(x_i | x_{<i})`. This is the approach of fully-visible neural networks (Neal, 1992; Bengio & Bengio, 1999) and NADE (Larochelle & Murray, 2011). There are no latent variables and no approximate inference: the likelihood is exact, and generation is ancestral. The factorization turns density estimation into a *sequence* problem — predict the next pixel given all previous ones — so the quality of the model hinges entirely on how expressive the conditional sequence model is.

**Recurrent and 2D-recurrent sequence models.** RNNs offer a compact, shared parametrization of a whole series of conditional distributions and excel at hard sequence problems — handwriting generation (Graves, 2013), character prediction (Sutskever et al., 2011), machine translation (Kalchbrenner & Blunsom, 2013). For images specifically, a two-dimensional LSTM that scans from the top-left pixel toward the bottom-right (Theis & Bethge, 2015) gave promising results on grayscale images and textures, because its two-dimensional state propagates signals in both the left-to-right and top-to-bottom directions and so handles the long-range dependencies central to object and scene structure. LSTM units (Hochreiter & Schmidhuber, 1997) carry a cell state through gated updates, which is what lets such a model retain information over long ranges.

**Residual learning.** Very deep networks are hard to train; residual connections (He et al., 2015) — adding a layer's input to its output — speed convergence and propagate signal directly through depth, and were the enabling trick for training networks of many layers.

**Modeling pixels as continuous vs discrete.** Prior pixel models used *continuous* distributions for pixel intensities (Theis & Bethge, 2015; Uria et al., 2013), which require a parametric assumption about the shape of the conditional density and make multimodality awkward. A discrete distribution over the 256 intensity values, by contrast, can be arbitrarily multimodal with no assumption about its shape.

**Masked weights.** To make a network output, at every spatial position simultaneously, a conditional that depends only on the *already-seen* pixels, one can zero out the weights that would connect a position to itself or to future positions — a masking idea also used in MADE (Germain et al., 2015) and in masked-weight variational autoencoders (Gregor et al., 2013).

## Baselines

**Fully-visible / NADE-style autoregressive models (Neal, 1992; Bengio & Bengio, 1999; Larochelle & Murray, 2011).** Factorize `p(x) = Π_i p(x_i | x_{<i})` and model each conditional with a (shallow) neural network. Strength: exact tractable likelihood, no latent variables. Gap: the conditional models are not expressive enough to capture the nonlinear long-range dependencies of natural images, and the original formulations did not scale to large color images.

**Two-dimensional LSTM for images (Theis & Bethge, 2015).** A 2D LSTM scanning top-left to bottom-right, with *continuous* pixel conditionals. Strength: handles long-range dependencies via the recurrent 2D state; promising on grayscale and textures. Gap: the continuous conditional limits multimodality and is harder to learn; the fully sequential 2D recurrence over every pixel is slow; demonstrated on grayscale rather than large-scale color images.

**Latent-variable models (VAE; Kingma & Welling, 2013; Rezende et al., 2014).** Encoder/decoder with a latent prior, trained on the ELBO. Strength: learns representations, scalable training. Gap: only a *lower bound* on the likelihood (intractable exact inference), and latent independence assumptions, so it does not deliver an exact density that captures the full pixel dependency structure.

## Evaluation settings

- **Datasets.** MNIST (28×28 binary handwritten digits); CIFAR-10 (32×32 natural color images); ImageNet resized to 32×32 and to 64×64 for large-scale natural-image density estimation.
- **Metrics.** Negative log-likelihood in bits per dimension (nats/bits per pixel sub-pixel) — the exact likelihood the model affords — used for quantitative comparison; qualitative inspection of samples and of image completions (condition on the top half / an occluded region, sample the rest).
- **Protocol.** Train by maximizing exact log-likelihood with the conditionals computed *in parallel* over all positions (the true pixels are known during training); generate by sampling pixels sequentially in raster order, feeding each sampled value back in. Optimize with a standard first-order optimizer; report bits/dim on held-out test images.

## Code framework

Pre-existing primitives: PyTorch `nn.Module`, `nn.Conv2d`, `nn.LSTM`-style gating, batch norm, ReLU, an Adam optimizer, and `F.cross_entropy`. A plain convolution that outputs, at every spatial position, a vector of logits over the 256 intensity values already exists. What does not yet exist is the mechanism that forbids each position from seeing itself or future pixels while still computing every position at once, and the recurrent layers that reach long-range context.

```python
import torch
from torch import nn
import torch.nn.functional as F

class CausalConv2d(nn.Conv2d):
    # TODO: a convolution whose receptive field excludes the current and future pixels,
    # so that position i depends only on x_{<i}. Two variants will be needed:
    # one for the first layer (stricter) and one for the rest.
    def __init__(self, variant, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: build a 0/1 mask over the kernel and apply it to the weights
        pass
    def forward(self, x):
        pass

class RecurrentImageLayer(nn.Module):
    # TODO: a layer that propagates an LSTM state across the spatial grid to reach
    # long-range context, computing a whole row/diagonal at once via convolution.
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass
    def forward(self, x):
        pass

class PixelModel(nn.Module):
    # outputs, per pixel, a distribution over the 256 intensity values
    def __init__(self, fm=128, n_layers=8):
        super().__init__()
        # TODO: first (strict) causal layer -> stack of causal / recurrent layers
        #       -> head that emits 256 logits per pixel position
        self.body = None     # TODO
        self.head = None     # TODO
    def forward(self, x):
        # returns logits of shape [B, 256, H, W]
        pass

# training: exact likelihood, conditionals in parallel (true pixels known)
def train_step(net, img, opt):
    target = (img[:, 0] * 255).long()           # discrete pixel value in {0..255}
    logits = net(img)                           # [B, 256, H, W]
    loss = F.cross_entropy(logits, target)      # = negative log-likelihood per pixel
    opt.zero_grad(); loss.backward(); opt.step()
    return loss

# generation: sequential, feed each sampled pixel back in (filled in after the model exists)
```
