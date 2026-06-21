# Context

The goal is a real-time, high-fidelity *neural audio codec*: an end-to-end trained encoder-decoder that compresses speech and music to a small, fixed bitstream and reconstructs it with minimal perceptual distortion, while running faster than real time on a single CPU core across a range of target bitrates. The surrounding field already has most of the ingredients: convolutional audio transforms, learned discrete latents, adversarial perceptual losses, and entropy coding.

## Research question

Lossy audio compression trades bitrate against distortion, where the distortion that matters is the one humans hear. Classical codecs (Opus, EVS) achieve this with hand-engineered signal transforms and psychoacoustic models. The question: can a single neural network, trained end to end, encode waveforms into a compact discrete bitstream and decode back to perceptually faithful audio, meeting the quality and latency bar across speech and music, multiple target bitrates (1.5, 3, 6, 12, 24 kbps), and both 24 kHz mono and 48 kHz stereo, while running faster than real time on a single CPU core?

Training such a model couples several loss terms — a time-domain reconstruction term, a multi-scale frequency term, an adversarial term, a feature-matching term, and a quantization-commitment term — that are combined with weights into a single objective.

## Background

By 2022 the pieces for a neural codec exist; the task is to assemble and improve them.

**Convolutional encoder–decoder audio models.** Fully convolutional symmetric encoder–decoder architectures (SEANet and the source-separation/enhancement and vocoder lineages) reliably map waveforms to latents and back: an encoder of strided convolutions downsamples and doubles channels, a mirror decoder of transposed convolutions upsamples and halves channels. Adding a recurrent (LSTM) layer over the latent sequence captures temporal structure. These run efficiently and can be made streaming/causal by putting all convolution padding before the current timestep.

**Vector quantization and its discrete-latent lineage.** VQ-VAE established that a continuous encoder output can be mapped to the nearest entry of a learned codebook, training the encoder through the non-differentiable `argmin` with a **straight-through estimator** (copy the gradient from the decoder input straight back to the encoder output) plus a **commitment loss** that pulls the encoder output toward its chosen code. The number of bits per frame a single codebook provides is fixed by its size.

**Residual vector quantization (RVQ).** Introduced for neural codecs by SoundStream (and used in Jukebox-style settings): quantize the latent with a first codebook, compute the **residual** between the latent and its quantization, quantize *that* with a second codebook, and so on for `N_q` stages; the final code is the sum of the `N_q` chosen entries. This is a coarse-to-fine, multi-stage quantizer — each stage refines what the previous stages missed — and crucially, since the stages are ordered by importance, you can **drop trailing codebooks at inference to lower the bitrate**, so one trained RVQ supports many bitrates by varying `N_q`. Codebooks are maintained with exponential-moving-average updates and dead-entry replacement (re-seeding unused codes from the current batch).

**Adversarial perceptual losses for audio.** Multi-scale waveform discriminators (MelGAN) and STFT-based discriminators, trained adversarially with a feature-matching term, act as learned perceptual losses and sharpen generated audio far beyond what ℓ₁/ℓ₂ reconstruction alone achieves. The hinge form of the adversarial loss is a stable, standard choice.

**Entropy coding.** Given a probabilistic model of a discrete source, a range/arithmetic coder compresses it to near its entropy. If a learned model predicts the distribution of the next code, its output can be entropy-coded for further lossless savings; this requires encoder and decoder to agree bit-for-bit on the probabilities, while floating-point evaluation can differ across hardware and batching.

## Baselines

**Opus / EVS (classical codecs).** Carefully engineered transform-plus-psychoacoustic pipelines. Opus spans wideband speech and music from a few kbps up; EVS is a speech-oriented standard. They are fast, mature, and the bar to beat.

**SoundStream.** A prior neural codec: a fully convolutional encoder–decoder with **residual vector quantization** and a combination of adversarial and reconstruction losses, operating at 24 kHz and supporting a range of bitrates with a single model via the RVQ codebook-dropping trick.

**VQ-VAE / DiffQ and other quantization schemes.** General learned-quantization approaches for compressing neural representations.

**SEANet encoder–decoder.** The convolutional backbone (strided/transposed conv blocks, channel doubling, residual units) reused here as the encoder/decoder. *Role:* the architecture skeleton, not a competitor.

## Evaluation settings

- **Audio configurations:** 24 kHz monophonic and 48 kHz stereophonic.
- **Domains/datasets:** clean speech (DNS Challenge, Common Voice), and general audio / music (AudioSet, FSD50K, Jamendo), with an on-the-fly mixing strategy (single source, or mixtures of two or three sources).
- **Target bitrates:** 1.5, 3, 6, 12, 24 kbps at 24 kHz; 3, 6, 12, 24 kbps at 48 kHz.
- **Subjective metric:** MUSHRA — human listeners rate compressed excerpts against the uncompressed reference and competing codecs.
- **Objective metrics:** standard signal/perceptual reconstruction measures, reported in ablations across bandwidths.
- **Streaming/latency constraint:** must run faster than real time on a single CPU core; algorithmic latency tied to the frame size (e.g. 13 ms at 24 kHz).
- **Training protocol:** Adam, batch 64 one-second clips, learning rate 3×10⁻⁴, β₁=0.5, β₂=0.9, 300 epochs of 2,000 updates each; one-second clips with variable bitrate sampled per batch.

## Code framework

The starting scaffold is a convolutional encoder-decoder codec harness with a discrete-bottleneck interface, a learned-perceptual-feedback interface, spectral reconstruction helpers, and a trainer that can combine independent loss terms.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class ConvEncoder(nn.Module):
    """Strided-conv downsampling backbone: waveform -> latent sequence [B, D, T]."""
    def __init__(self, channels=1, n_filters=32, strides=(2, 4, 5, 8), dimension=128):
        super().__init__()
        # TODO: initial conv (k=7); residual+downsample blocks doubling channels; LSTM; final conv (k=7)
        pass
    def forward(self, x):  # -> z  [B, D, T]
        raise NotImplementedError

class ConvDecoder(nn.Module):
    """Mirror of the encoder with transposed convolutions: latent -> waveform."""
    def __init__(self, *a, **k):
        super().__init__()
        # TODO: mirror ConvEncoder, transposed convs, reversed strides
        pass
    def forward(self, z):  # -> x_hat
        raise NotImplementedError

class Quantizer(nn.Module):
    """Maps continuous latents to discrete codes and back; trainable through the codes."""
    def __init__(self, dim):
        super().__init__()
        # TODO: choose the discrete bottleneck and any auxiliary training loss.
        pass
    def forward(self, z, n_q=None):
        # TODO: return quantized latent, code indices, and auxiliary loss.
        raise NotImplementedError

class Discriminator(nn.Module):
    """Learned perceptual loss: real vs reconstructed audio + intermediate features."""
    def __init__(self):
        super().__init__()
        # TODO: choose the discriminator input representation and scales.
        pass
    def forward(self, x):
        raise NotImplementedError

def reconstruction_loss(x, x_hat):
    # TODO: time-domain term + multi-scale frequency term
    raise NotImplementedError

def train_step(enc, quant, dec, disc, x, opt_g, opt_d):
    # encode -> quantize -> decode; combine the active reconstruction and perceptual losses
    ...
```
