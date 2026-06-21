# Context

## Research Question

How can a generative image model keep the measurable, mode-covering benefits of likelihood training while producing high-resolution samples that look sharp and varied?

Adversarial models can produce sharp images at high resolution, but they are not trained with a likelihood and can drop parts of the data distribution without a direct held-out likelihood penalty. Likelihood models minimize the forward KL from data to model up to the data entropy, so assigning negligible probability to real data is heavily penalized; they give a comparable test-set objective and, in principle, pressure the model to cover all modes.

## Background

**Likelihood versus implicit generation.** VAEs, flows, and autoregressive models train by likelihood or a likelihood bound; GANs train an implicit generator through a discriminator. The first family offers held-out objectives and mode-coverage pressure, while the second family historically produced very sharp samples but needed proxy metrics such as Inception Score and FID for comparison.

**Pixel-space likelihood mismatch.** Pixel NLL is not always aligned with perceptual quality. A model can improve bits/dim by modeling local texture, sensor noise, or bit-plane detail while doing little for object shape and scene layout. Autoregressive pixel models sample one location at a time.

**Lossy compression.** Image compression suggests that much of an image can be discarded without a visible change. A generative model that operates on a compact representation rather than raw pixels can spend probability mass on the information the decoder actually needs for reconstruction.

**Vector quantization.** A vector-quantized autoencoder maps an input to continuous encoder outputs and replaces each vector by the nearest entry in a learned codebook. The transmitted latent is a grid of discrete indices. The straight-through estimator lets reconstruction gradients reach the encoder despite the nearest-neighbor argmin, while codebook and commitment terms align encoder outputs with the selected prototypes.

**Autoregressive priors over discrete grids.** PixelRNN, Gated PixelCNN, and PixelSNAIL factor a discrete grid as `p(c) = product_i p(c_i | c_<i)`. PixelSNAIL combines causal convolution with causal self-attention, which is useful when long-range dependencies matter but expensive on large grids.

**Learned priors over learned latents.** Fitting a prior after the representation is learned reduces the gap between the aggregate posterior codes and the prior used for sampling. Information-theoretically, it re-encodes the learned codes with a distribution closer to their empirical entropy.

## Baselines

**Flat VQ-VAE.** For an encoder output `z_e(x)` and codebook vectors `e_j`, quantization chooses `k = argmin_j ||z_e(x) - e_j||_2` and decodes from `z_q(x)=e_k`. A minimization loss is

`||x - D(z_q(x))||_2^2 + ||sg[z_e(x)] - e_k||_2^2 + beta ||z_e(x) - sg[e_k]||_2^2`.

With a deterministic one-hot posterior and a uniform `K`-way prior, the KL is `log K` per latent position, or `N log K` for `N` independent positions, so it is constant during autoencoder training. The codebook term can be replaced by an EMA update, equivalent to online k-means.

**Pixel-space autoregressive models.** PixelCNN-style models give exact likelihood and strong density estimates.

**GANs at scale.** BigGAN-style generators provide strong sample quality and a truncation knob for a quality-diversity tradeoff.

**Hierarchical latent-variable models.** Multi-level latents and hierarchical quantized codes have been used before, including for raw audio.

## Evaluation Settings

- **Datasets.** A useful benchmark should include complex class-conditional natural images at `256x256` and a higher-resolution face dataset such as `1024x1024`.
- **Reconstruction diagnostics.** If the method compresses images, report reconstruction error and inspect whether the decoder preserves global layout and local detail.
- **Prior diagnostics.** If the method learns a prior over codes, report train and validation NLL for that prior, and avoid comparing code-prior NLLs across different encoders or decoders.
- **Sample diagnostics.** Use FID, Inception Score, precision/recall-style coverage measures, classification-accuracy-style tests for class-conditional generation, and visual inspection for long-range coherence.
- **Speed and memory.** Sampling cost should be measured in the compressed latent space and in the final pixel decoder, since pixel-space autoregression is the main computational baseline.

## Code Framework

```python
import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
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
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        # TODO: build strided convolutions and residual blocks.
        pass

    def forward(self, input):
        pass


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        # TODO: build residual blocks and transposed-convolution upsampling.
        pass

    def forward(self, input):
        pass


class Bottleneck(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: map continuous features to discrete symbols and back to vectors.
        pass

    def forward(self, input):
        # Return quantized features, auxiliary bottleneck loss, and symbol indices.
        pass


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.bottleneck = None
        self.decoder = None

    def forward(self, input):
        pass


class CodePrior(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # TODO: autoregressively model a grid of discrete code indices.
        pass

    def forward(self, codes, condition=None):
        pass
```
