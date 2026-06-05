## Research question

How can we model the distribution `p(x)` of natural images so that, all at once, we can (a) compute
the **exact** likelihood of any held-out image (a true tractable density, not a bound), (b) **sample**
new images from the model, and (c) **scale** the whole thing to large datasets and high-dimensional
images on modern hardware?

This is the expressive/tractable/scalable trilemma of generative image modeling. Images are
high-dimensional (a 32×32 RGB image already has 3072 strongly-correlated integer values) and highly
structured, so the model must capture long-range and highly nonlinear dependencies. Yet the most
expressive families of the time pay for that expressiveness with intractable likelihoods: you cannot
say how probable a given image is under the model, which makes them hard to compare on a common
yardstick and hard to use for tasks like inpainting, compression, or density estimation. A method
that delivered an exact, comparable likelihood **and** stayed expressive enough to produce sharp,
globally coherent samples — and that could benefit from larger capacity — would resolve the
trilemma. The question is what functional form `p(x)` should take, and what neural architecture can
realize it without smuggling in independence assumptions between pixels.

## Background

**The exact-likelihood handle: the chain rule.** For any joint distribution, repeated conditioning
gives an exact factorization with no approximation:

```
p(x) = Π_i p(x_i | x_1, ..., x_{i-1}).
```

Fixing an ordering of the variables turns density estimation into a sequence of prediction problems:
learn to predict each variable given all the earlier ones. For images, the natural ordering is the
**raster scan** — row by row from the top, left to right within a row — so an `n×n` image becomes a
sequence `x_1, ..., x_{n^2}` and

```
p(x) = Π_{i=1}^{n^2} p(x_i | x_1, ..., x_{i-1}).
```

This is the *fully-visible* directed model (Neal, 1992; Bengio & Bengio, 2000): every conditional is
explicit, so the likelihood is exact and the only modeling choice is how to parameterize the
conditionals. The training signal — `−log p(x)` summed over positions — is just a sum of per-position
prediction losses, and crucially all of these can be evaluated **in parallel** on a training image
(the "previous pixels" are the known input), even though *generating* a fresh image is inherently
**sequential** (each sampled value must be fed back before the next is predicted).

**Why latent-variable and undirected models fall short of exact likelihood.** A variational
autoencoder (Kingma & Welling, 2013; Rezende et al., 2014) defines `p(x) = ∫ p(x|z) p(z) dz`; this
marginal is intractable, so training maximizes an evidence lower bound rather than the true
likelihood, inference of `z` is only approximate, and the latent bottleneck typically forces the
pixels to be conditionally independent given `z`. Deep undirected models (RBM/DBM; Salakhutdinov &
Hinton, 2009) carry an intractable partition function, so exact likelihood is unavailable and must be
estimated. Recurrent latent generators such as DRAW (Gregor et al., 2015) and deep autoregressive
networks DARN (Gregor et al., 2014) also report bounds. None gives a clean exact number.

**Autoregressive density estimators that do give exact likelihood.** NADE (Larochelle & Murray, 2011)
parameterizes the chain-rule conditionals with a single shared hidden layer whose weights are tied
across positions, so it is compact and trainable, and EoNADE (Uria et al., 2014) averages over many
orderings. MADE (Germain et al., 2015) realizes the same idea inside an autoencoder by **masking** the
weight matrices so output unit `i` connects only to input units `< i`; one forward pass then produces
all conditionals at once. On the continuous side, RNADE (Uria et al., 2013) and the spatial-LSTM /
RIDE model (Theis & Bethge, 2015) use a recurrent net to emit, per pixel, the parameters of a
**continuous** density — a Gaussian or a mixture of Gaussian scale mixtures.

**Pixels are discrete values, not continuous measurements.** Pixel intensities
are genuinely discrete: integers in `{0, ..., 255}`. The prevailing autoregressive image models place
a *continuous* density on them, which forces a choice of parametric shape (a mixture is only as
multimodal as its number of components), leaks probability mass outside `[0, 255]`, and assumes
smoothness across neighboring intensities. The standard way to make continuous-density likelihoods
comparable across models is to dequantize by adding uniform noise in `[0, 1]` to the integer pixels
(Theis et al., 2015), after which a discrete distribution corresponds to a piecewise-uniform density
with the *same* likelihood — so discrete and continuous models can be put on one bits-per-dimension
scale.

**Recurrent networks as sequence models.** RNNs, and LSTMs in particular (Hochreiter & Schmidhuber,
1997), give a compact shared parameterization of a long series of conditionals and have excelled at
hard sequence tasks — handwriting generation (Graves, 2013), character-level text (Sutskever et al.,
2011), translation (Kalchbrenner & Blunsom, 2013). Multidimensional LSTMs (Graves & Schmidhuber,
2009) extend recurrence to 2-D grids, and a 2-D LSTM scanning an image top-left to bottom-right has
shown promise on grayscale textures (Theis & Bethge, 2015); a parallel multi-dimensional LSTM also
exists (Stollenga et al., 2015). Convolutional networks, by contrast, share weights spatially and run
fully in parallel, but a stack of convolutions has only a **bounded** receptive field that grows
linearly with depth.

**Residual learning.** Very deep nets are hard to optimize; residual connections (He et al., 2015) —
adding a layer's input to its output so the layer learns a residual — let gradients and signal
propagate directly and were the enabler for training networks far deeper than before. Depth-wise
gating in recurrent stacks (Kalchbrenner et al., 2015; Zhang et al., 2016) addresses the same
signal-propagation problem with extra gates.

## Baselines

- **NADE / EoNADE (Larochelle & Murray, 2011; Uria et al., 2014).** Autoregressive estimator: the
  chain-rule conditionals share one hidden layer with tied weights, trained by exact likelihood.
  Compact and tractable, but the shared single-hidden-layer parameterization limits expressiveness,
  and it was developed mainly on small binary data; modeling the highly nonlinear, long-range,
  full-color dependencies of natural images needs a far more expressive conditional model.

- **MADE (Germain et al., 2015).** A single autoencoder whose weight matrices are masked so output `i`
  depends only on inputs `< i`, giving all conditionals in one pass; averaging over masks/orderings
  improves it. The masking idea is exactly the right primitive for respecting an ordering, but MADE is
  a fully-connected autoencoder with a fixed input size and no spatial weight sharing, so it does not
  exploit the 2-D translation structure of images and does not scale to large images.

- **RIDE / spatial LSTM, and RNADE (Theis & Bethge, 2015; Uria et al., 2013).** A 2-D recurrent net
  emits, per pixel, the parameters of a **continuous** conditional density (a mixture of Gaussian
  scale mixtures, or a Gaussian mixture). Exact likelihood, genuinely recurrent, captures long-range
  context. The gaps: a continuous parametric head imposes a shape prior and leaks mass outside the
  valid intensity range, and the 2-D LSTM as formulated is expensive and does not obviously parallelize
  to large-scale training.

- **NICE / deep GMMs / deep diffusion (Dinh et al., 2014; van den Oord & Schrauwen, 2014;
  Sohl-Dickstein et al., 2015).** Other exact- or near-exact-likelihood models for natural images
  reported on CIFAR-10 in bits/dim. They establish likelihood-based image modeling as a meaningful
  benchmark, but they do not yet combine exact likelihood, strong local image structure, and a
  scalable conditional predictor.

- **Latent / undirected generators (VAE, DBM, DBN, DLGM, DRAW; Kingma & Welling 2013; Salakhutdinov &
  Hinton 2009; Rezende et al. 2014; Gregor et al. 2015).** Strong representational models and, for
  DRAW, attractive samples, but they report only bounds or estimates of the likelihood and impose
  conditional-independence structure through the latent, so they are not directly comparable on exact
  likelihood.

## Evaluation settings

- **Datasets.** Binarized MNIST (Salakhutdinov & Murray, 2008; LeCun et al., 1998) as a small,
  well-studied sanity check with much prior art; CIFAR-10 (Krizhevsky, 2009) as the standard 32×32 RGB
  natural-image benchmark; ImageNet (Russakovsky et al., 2015) resized to 32×32 and 64×64 as a large,
  diverse, high-volume natural-image source.
- **Metric.** Average negative log-likelihood. For MNIST, reported in **nats**; for CIFAR-10 and
  ImageNet, in **bits per dimension** — the total negative log-likelihood divided by the number of
  dimensions (e.g. `32·32·3 = 3072`), interpretable as the bits a likelihood-based compressor would
  spend per RGB value.
- **Comparability protocol.** To compare against continuous-density baselines, dequantize by adding
  uniform `[0, 1]` noise to the integer pixels; a discrete model then maps to a piecewise-uniform
  density with the same likelihood on the noised data (Theis et al., 2015).
- **Optimization protocol.** Train by minimizing the exact NLL with a stochastic optimizer; small
  batches for small datasets and the largest batch GPU memory allows for ImageNet; input scaled and
  centered, no other augmentation.

## Code framework

Available building blocks: a data pipeline that yields integer-valued image tensors, convolution
layers, LSTM-style gates, residual addition, categorical cross-entropy, and an optimizer plus a
training loop. The empty slot is the image-to-conditionals architecture: it must emit a categorical
distribution at every position while ensuring that position `i` never depends on positions `≥ i`.

```python
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# integer pixel targets in {0, ..., 255} (or {0,1} for binarized MNIST)
train_loader = DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=16, shuffle=True,
)

# A convolution whose receptive field must be restricted so position i never
# reads positions >= i. The mechanism that enforces this is the open problem.
class OrderedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: build and apply the connectivity restriction. The mask_type slot
        #       will distinguish the raw-input layer from later feature layers.
        pass

    def forward(self, x):
        # TODO: apply the restriction, then convolve.
        return super().forward(x)


class GatedActivation(nn.Module):
    def forward(self, x):
        # TODO: decide whether the conditional predictor needs a multiplicative gate.
        pass


class OrderedImageLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, first_layer=False):
        super().__init__()
        # TODO: fill the ordered layer. It may be a simple ordered convolution or a
        #       split spatial computation, but it must preserve the variable order.
        pass

    def forward(self, above_state, row_state):
        # TODO: return updated states plus any layer-to-output contribution.
        pass


# The full image model: ordered layers -> per-position distribution.
class ImageDensityModel(nn.Module):
    def __init__(self, in_ch=1, n_values=256, n_layers=10, channels=64,
                 split_state=True, head_channels=64):
        super().__init__()
        # TODO: choose and stack the ordered layers, then emit n_values logits per
        #       position for the categorical conditional.
        pass

    def forward(self, x):
        # TODO: return logits of shape (batch, n_values, H, W).
        pass


def nll_loss(logits, target_pixels):
    # TODO: exact negative log-likelihood = sum of per-position categorical losses.
    pass


def train_step(model, x, opt):
    # TODO: one stochastic update on a batch of images.
    pass


def train(model):
    opt = optim.RMSprop(model.parameters())
    for x, _ in train_loader:
        train_step(model, x, opt)


@torch.no_grad()
def sample(model, n, H, W, n_values=256):
    img = torch.zeros(n, 1, H, W)
    # TODO: sequential generation — for each position in raster order, run the
    #       model, read off p(x_i | x_{<i}), sample, write it back, continue.
    pass
```
