# Context

## Research question

How can we estimate the probability distribution `p(x)` of natural images in a way that is simultaneously *tractable* — letting us compute the exact likelihood of an image and sample new ones — and *expressive* enough to capture the highly nonlinear, long-range, and multimodal correlations between pixels that make natural images what they are?

Natural images are high-dimensional and highly structured, so a good density model would unlock image compression, inpainting, deblurring, and generation (and, conditioned on side information, text-to-image or future-frame prediction).

## Background

**Latent-variable models.** Most work on generative image modeling uses stochastic latent-variable models such as the VAE (Kingma & Welling, 2013; Rezende, Mohamed & Wierstra, 2014), which aim to extract meaningful latent representations. They come with an intractable inference step — the likelihood is only bounded (the ELBO), not computed exactly — and they typically impose independence structure in the latents.

**Autoregressive / fully-visible models.** An alternative that is exactly tractable casts the joint distribution of the pixels as a product of conditionals: `p(x) = Π_i p(x_i | x_{<i})`. This is the approach of fully-visible neural networks (Neal, 1992; Bengio & Bengio, 1999) and NADE (Larochelle & Murray, 2011). There are no latent variables and no approximate inference: the likelihood is exact, and generation is ancestral. The factorization turns density estimation into a *sequence* problem — predict the next pixel given all previous ones — so the quality of the model hinges entirely on how expressive the conditional sequence model is.

**Recurrent and 2D-recurrent sequence models.** RNNs offer a compact, shared parametrization of a whole series of conditional distributions and excel at hard sequence problems — handwriting generation (Graves, 2013), character prediction (Sutskever et al., 2011), machine translation (Kalchbrenner & Blunsom, 2013). For images specifically, a two-dimensional LSTM that scans from the top-left pixel toward the bottom-right (Theis & Bethge, 2015) gave promising results on grayscale images and textures, because its two-dimensional state propagates signals in both the left-to-right and top-to-bottom directions and so handles the long-range dependencies central to object and scene structure. LSTM units (Hochreiter & Schmidhuber, 1997) carry a cell state through gated updates, which is what lets such a model retain information over long ranges.

**Residual learning.** Very deep networks are hard to train; residual connections (He et al., 2015) — adding a layer's input to its output — speed convergence and propagate signal directly through depth, and were the enabling trick for training networks of many layers.

**Modeling pixels as continuous vs discrete.** Prior pixel models used *continuous* distributions for pixel intensities (Theis & Bethge, 2015; Uria et al., 2013), which require a parametric assumption about the shape of the conditional density and make multimodality awkward, even though observed pixel conditionals in real images are often sharply multimodal. Pixel intensities are in fact taken from a fixed set of 256 values.

**Masked weights.** Zeroing out selected connections in a network's weights — constraining which inputs each output may depend on — is an established technique, used in MADE (Germain et al., 2015) and in masked-weight variational autoencoders (Gregor et al., 2013).

## Baselines

**Fully-visible / NADE-style autoregressive models (Neal, 1992; Bengio & Bengio, 1999; Larochelle & Murray, 2011).** Factorize `p(x) = Π_i p(x_i | x_{<i})` and model each conditional with a (shallow) neural network. Exact tractable likelihood, no latent variables.

**Two-dimensional LSTM for images (Theis & Bethge, 2015).** A 2D LSTM scanning top-left to bottom-right, with *continuous* pixel conditionals. Handles long-range dependencies via the recurrent 2D state; demonstrated on grayscale images and textures.

**Latent-variable models (VAE; Kingma & Welling, 2013; Rezende et al., 2014).** Encoder/decoder with a latent prior, trained on the ELBO. Learns representations and supports scalable training.

## Evaluation settings

- **Datasets.** MNIST (28×28 binary handwritten digits); CIFAR-10 (32×32 natural color images); ImageNet resized to 32×32 and to 64×64 for large-scale natural-image density estimation.
- **Metrics.** Negative log-likelihood in bits per dimension (nats/bits per pixel sub-pixel) — the exact likelihood the model affords — used for quantitative comparison; qualitative inspection of samples and of image completions (condition on the top half / an occluded region, sample the rest).
- **Protocol.** Train by maximizing exact log-likelihood with the conditionals computed *in parallel* over all positions (the true pixels are known during training); generate by sampling pixels sequentially in raster order, feeding each sampled value back in. Optimize with a standard first-order optimizer; report bits/dim on held-out test images.

## Code framework

Pre-existing primitives: PyTorch `nn.Module`, `nn.Conv2d`, `nn.LSTM`-style gating, batch norm, ReLU, an Adam optimizer, and `F.cross_entropy`. A plain convolution that outputs, at every spatial position, a vector of logits over the 256 intensity values already exists. The network components that turn this into a working conditional model for `p(x_i | x_{<i})` remain to be designed.

```python
import torch
from torch import nn
import torch.nn.functional as F

class PixelModel(nn.Module):
    # outputs, per pixel, a distribution over the 256 intensity values
    def __init__(self, fm=128, n_layers=8):
        super().__init__()
        # TODO: design the layers that map an input image to the per-pixel
        #       conditionals, then a head that emits 256 logits per pixel position
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
