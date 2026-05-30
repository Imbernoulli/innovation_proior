# Context

## Research question

How can we generate images that are simultaneously high-fidelity, high-resolution, and *diverse* — covering the full variety of a complex dataset like ImageNet — using a model whose training objective gives a principled, measurable handle on generalization?

The tension is concrete. Adversarial models produce the sharpest, highest-resolution images available, but they are known to drop modes: a generator can win the minimax game while only ever producing a slice of the true distribution, and there is no agreed way to measure on a held-out set how much of the distribution has been lost. Likelihood-based models, by contrast, optimize negative log-likelihood, which is the forward KL between data and model and therefore assigns infinite penalty to leaving any data example with zero mass — they are *forced* to cover all modes, and their objective is comparable across runs and measurable on test data. But directly maximizing likelihood in pixel space is unsatisfying: pixel NLL is not a reliable proxy for perceptual quality, the model has no intrinsic pressure to spend capacity on global structure rather than imperceptible local texture, and an autoregressive model over a quarter-million pixels is painfully slow to sample.

A satisfactory method would keep the mode-coverage and measurability of likelihood training, but (a) relieve the generative model from spending capacity on negligible perceptual detail, (b) make training and sampling fast enough to reach high resolution, and (c) still produce coherent global structure.

## Background

**Two families of generative models.** Likelihood-based models — VAEs (Kingma & Welling, 2013; Rezende, Mohamed & Wierstra, 2014), flow-based models (Dinh et al., 2014, 2016; Rezende & Mohamed, 2015; Kingma & Dhariwal, 2018), and autoregressive models (Larochelle & Murray, 2011; van den Oord et al., 2016) — optimize NLL. Because NLL is the forward KL, these models in principle cover all modes of the data and offer a generalization measure on held-out data. Implicit models, chiefly GANs (Goodfellow et al., 2014), optimize a minimax objective; scaled up (BigGAN, Brock et al., 2018; StyleGAN, Karras et al., 2018) they produce high-quality high-resolution images but are known not to fully capture the diversity of the data, and are hard to evaluate — researchers fall back on Inception Score (Salimans et al., 2016) and Fréchet Inception Distance (Heusel et al., 2017) as proxies.

**The pixel-space difficulty.** NLL in pixel space is not always a good measure of sample quality (Theis et al., 2016); a model can have excellent bits/dim and produce poor samples. There is also no intrinsic incentive for a pixel-space model to allocate capacity to global structure. Partial fixes introduce inductive biases — multi-scale modeling (Theis & Bethge, 2015; van den Oord et al., 2016; Reed et al., 2017; Subscale Pixel Networks, Menick & Kalchbrenner, 2018) or modeling only the dominant bit planes (Kolesnikov & Lampert, 2017; Kingma & Dhariwal, 2018).

**Lossy compression as a lens.** Classical image compression (JPEG; Wallace, 1992) shows that one can discard well over 80% of an image's data without a perceptible change. This suggests a generative model should not waste itself on the discardable part: compress the image first into a compact code that preserves what matters perceptually, and model the *distribution over codes* instead of over pixels.

**Vector-quantized autoencoders.** A way to obtain such a code with discrete symbols: an encoder maps the input to a continuous vector, which is then replaced by the index of its nearest entry in a learned codebook of prototype vectors (vector quantization). This produces a small grid of discrete symbols — a representation tens of times smaller than the image — from which a decoder can reconstruct with little distortion. Discrete codes are also a natural input to an autoregressive categorical model.

**Autoregressive models over discrete grids.** PixelRNN / PixelCNN (van den Oord et al., 2016) and Gated PixelCNN (van den Oord et al., 2016) factorize a joint distribution over a discrete grid as `p(x) = Π_i p(x_i | x_{<i})` using masked convolutions in a fixed raster order; each conditional is a categorical softmax. PixelSNAIL (Chen et al., 2017) interleaves gated residual convolutional blocks with causal multi-head self-attention, giving an effectively unbounded receptive field for long-range dependencies that convolutions alone miss, and reaching state-of-the-art bits/dim on CIFAR-10 and 32×32 ImageNet. These are exactly the models one would reach for to put a distribution over a grid of discrete codes.

**Learned priors close the posterior/prior gap.** Fitting a prior over the latents *after* the autoencoder is trained — rather than fixing it — reduces the gap between the aggregate (marginal) posterior and the prior (variational-lossy-autoencoder analysis, Chen et al., 2016). Latents drawn from such a learned prior at test time look like what the decoder saw during training, so decoded samples are more coherent; information-theoretically it is a near-lossless re-encoding to a bit rate close to the latents' true entropy.

**Hierarchical latents.** Multi-level latent-variable models go back at least to Rezende et al. (2014). For vector-quantized codes specifically, Dieleman et al. (2018) used a hierarchy of codes to model music with a WaveNet decoder, where the lower levels *refine* the information in the upper levels; that scheme suffered hierarchy-collapse problems requiring mitigation.

## Baselines

**Vector-Quantized VAE (van den Oord, Vinyals & Kavukcuoglu, 2017).** The direct foundation. An encoder `E` maps `x` to a continuous vector; quantization replaces it by the nearest codebook prototype, `Quantize(E(x)) = e_k` with `k = argmin_j ||E(x) − e_j||`. The decoder `D` reconstructs from `e_k`. Gradients flow to the encoder through the non-differentiable argmin via the straight-through estimator (forward = `e_k`, backward = identity onto `E(x)`). The objective is
`L = ||x − D(e)||₂² + ||sg[E(x)] − e||₂² + β·||sg[e] − E(x)||₂²`,
where `sg[·]` is stop-gradient: the first term reconstructs, the second (codebook loss) pulls the chosen prototype toward the encoder output, the third (commitment loss, weight `β`) keeps the encoder output from drifting away from its chosen code. The codebook loss can be replaced by an exponential-moving-average update of each prototype toward the mean of the encoder outputs assigned to it (an online k-means step), with decay `γ`. Because the posterior is a deterministic one-hot and the prior over codes is fixed uniform, the KL term is the constant `log K`, so there is no pressure to silence the latent — the discrete bottleneck stays in use. After training, a PixelCNN is fit over the single grid of codes to enable sampling. Gap: a *single* flat code grid forces one resolution to carry both global shape and local texture; at high image resolution this compromise limits fidelity, and a single PixelCNN over a large code grid must model global and local correlations at once.

**Pixel-space autoregressive models (PixelCNN / Gated PixelCNN, PixelSNAIL).** Model `p(x)` directly over the pixel grid. Strength: exact tractable likelihood, full mode coverage, strong density estimates. Gap: sampling is sequential over every pixel (and color channel) — an order of magnitude too slow at high resolution — and capacity is spent modeling imperceptible local detail rather than global structure.

**GANs at scale (BigGAN, Brock et al., 2018).** State-of-the-art FID/IS, high-resolution, high-quality samples, via self-attention, stabilization tricks, large-batch TPU training, and a *truncation* knob that trades sample diversity for quality at test time. Gap: incomplete mode coverage / limited diversity, no test-set generalization measure, and reliance on proxy metrics for model selection.

**Hierarchical VQ for audio (Dieleman et al., 2018).** A hierarchy of vector-quantized codes with a WaveNet decoder, where lower levels refine the upper level's information. Gap: because lower levels only refine, and because of the autoregressive decoder, it is subject to hierarchy-collapse failure modes that need explicit mitigation.

## Evaluation settings

- **Datasets.** ImageNet at 256×256 for class-conditional generation; FFHQ at 1024×1024 for high-resolution faces. (Diagnostic comparisons in the literature also use CIFAR-10 and 32×32 ImageNet for bits/dim.)
- **Metrics.** Negative log-likelihood in bits/dim for the priors over codes; reconstruction fidelity of the autoencoder; for whole-model sample assessment, Fréchet Inception Distance and Inception Score, plus diversity/coverage and qualitative coherence.
- **Protocol.** Stage 1: train encoder/decoder + codebook(s) by gradient descent with Adam, batch 128, MSE reconstruction in pixels, commitment weight `β = 0.25`, EMA codebook decay `γ = 0.99`, codebook size 512, code dimension 64. Stage 2: extract the discrete codes for all training images and fit one autoregressive prior per level on them. Sampling: draw codes ancestrally from the priors (top first, then bottom conditioned on top) and run the feed-forward decoder once. Optional sample selection re-scores generated images with a pretrained ImageNet classifier.

## Code framework

Pre-existing primitives: PyTorch `nn.Module`, strided `Conv2d` / `ConvTranspose2d`, the Adam optimizer, an `MSELoss`, an `ImageFolder` data pipeline with resize/center-crop/normalize, and `F.embedding` / `F.one_hot`. A residual block is a standard primitive. What does not yet exist is the discrete bottleneck and the way the encoder/decoder are arranged around it.

```python
import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    # standard pre-activation residual block: ReLU, 3x3 conv, ReLU, 1x1 conv, + input
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )
    def forward(self, input):
        return self.conv(input) + input

class Encoder(nn.Module):
    # strided convs downsample, then residual blocks; generic conv stack
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        # TODO: build strided conv stack (stride 4 or 2) + n_res_block ResBlocks
        pass
    def forward(self, input):
        pass

class Decoder(nn.Module):
    # residual blocks then transposed convs upsample back; generic conv stack
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        # TODO: conv-in + n_res_block ResBlocks + transposed-conv upsampling
        pass
    def forward(self, input):
        pass

class Bottleneck(nn.Module):
    # TODO: the discrete bottleneck we will design.
    # Takes a continuous feature map, returns a (discretized) feature map to decode from,
    # an auxiliary loss term, and the chosen symbol indices.
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass
    def forward(self, input):
        # returns quantized, aux_loss, indices
        pass

class AutoEncoder(nn.Module):
    # TODO: how to arrange encoder(s), bottleneck(s), decoder(s) for a large image.
    def __init__(self):
        super().__init__()
        self.enc = None      # TODO
        self.bottleneck = None  # TODO
        self.dec = None      # TODO
    def forward(self, input):
        # encode -> bottleneck -> decode; return reconstruction + aux loss
        pass

class CodePrior(nn.Module):
    # TODO: autoregressive model p(codes) fit AFTER the autoencoder, for sampling.
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass
    def forward(self, codes, condition=None):
        pass

# training loop (stage 1): reconstruction + bottleneck aux loss
def train_stage1(model, loader, opt):
    criterion = nn.MSELoss()
    aux_weight = None   # TODO: weight on the bottleneck's auxiliary loss
    for img in loader:
        out, aux_loss = model(img)
        recon_loss = criterion(out, img)
        loss = recon_loss + aux_weight * aux_loss   # TODO: assemble
        opt.zero_grad(); loss.backward(); opt.step()
```
