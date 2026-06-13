# Context

The goal is a *second-stage* neural vocoder: a model that takes a mel-spectrogram (the low-resolution acoustic representation a text-to-speech front-end produces) and synthesizes the corresponding 22.05 kHz raw waveform, fast enough to be useful in production and with fidelity in the range occupied by slow autoregressive vocoders. This is the landscape as it stands in 2020.

## Research question

Modern neural speech synthesis is a two-stage pipeline. Stage one predicts a compact intermediate representation (a mel-spectrogram) from text; stage two — the vocoder — turns that representation into the actual waveform, up to 24,000 samples per second at 16-bit fidelity. Stage two is the bottleneck for both quality and speed.

The precise problem: invert a mel-spectrogram to a raw waveform such that the output is **both** (a) perceptually indistinguishable from a real recording and (b) generated far faster than real time, ideally on commodity hardware. The two goals pull against each other in the existing toolkit. Autoregressive models hit (a) but fail (b) catastrophically — they emit one sample per forward pass, so a one-second clip is tens of thousands of sequential network evaluations. Flow-based models recover (b) by generating all samples in parallel, but pay for it in enormous parameter counts and still trail on fidelity. Adversarial models are small and fast but, so far, audibly worse.

A solution would have to: synthesize in parallel (no per-sample recurrence), stay small enough to run faster than real time on a single GPU and ideally on CPU, and close the perceptual gap to autoregressive output. The open clue is a property of the signal itself: speech audio is, physically, a superposition of sinusoidal components at many different periods (pitch harmonics, formants), and a model that fails to reproduce those periodic structures will sound buzzy or noisy no matter how good its average spectral error is.

## Background

By 2020 the second-stage vocoder problem has a rich set of partial answers, each illuminating a different constraint.

**The autoregressive ceiling.** WaveNet established that a neural network can synthesize raw audio that surpasses concatenative and parametric vocoders in quality. It is a stack of dilated causal convolutions modeling `p(x_t | x_{<t})`. The cost is structural: sampling is inherently sequential, one sample per network evaluation, which is prohibitively slow at audio rates. The lesson kept: convolutions with large dilated receptive fields can capture the long-range structure of speech (a single phoneme spans >100 ms, i.e. thousands of adjacent samples that must stay coherent).

**Parallelizing via flows.** Parallel WaveNet trains an inverse-autoregressive-flow student to match a pretrained WaveNet teacher via probability-density distillation, buying a >1000× speedup. WaveGlow removes the teacher: it is a Glow-style normalizing flow trained by plain maximum likelihood, transforming a Gaussian noise sequence of the same length into the waveform in one parallel pass, conditioned on the mel-spectrogram. It produces high-quality audio but needs a very deep architecture (>90 coupling layers) and a correspondingly large parameter budget.

**The signal-processing facts that constrain any solution.** Two facts about waveforms matter. First, *long-range coherence*: because a phoneme can last >100 ms, samples thousands apart are correlated, so the receptive field of both generator and discriminator must be large. Second, *periodicity*: voiced speech is quasi-periodic — a sum of sinusoids at the fundamental frequency and its harmonics. Average pooling (downsampling by averaging) acts as a low-pass filter: it attenuates high frequencies and smears phase, so a discriminator that only ever sees average-pooled views of the waveform can be fooled by a generator that gets the low-frequency envelope right while corrupting fine periodic structure. That leaves a weakness in evaluating audio only at progressively smoothed scales.

## Baselines

**MelGAN.** A GAN vocoder built for speed. Its generator is a fully convolutional stack of transposed convolutions that upsample the mel-spectrogram to waveform resolution, with residual blocks using dilated convolutions to enlarge the receptive field; no noise input. Its key contribution is the **multi-scale discriminator (MSD)**: three structurally identical sub-discriminators operating on the raw waveform, a 2×-average-pooled version, and a 4×-average-pooled version, each a stack of strided (and grouped) 1D convolutions with leaky ReLU. It also uses a **feature-matching loss** — the L1 distance between the discriminator's intermediate feature maps on real vs. generated audio — to stabilize and guide the generator. MelGAN runs faster than real time on CPU. *Gap:* every view the discriminator sees is either the raw signal or an average-pooled (low-pass-filtered) signal, so fine periodic structure at specific phases is poorly policed; sample quality trails autoregressive/flow models.

**Parallel WaveGAN.** A small GAN vocoder trained with a **multi-resolution STFT loss** — the sum of spectral-convergence and log-magnitude losses computed at several STFT resolutions — jointly with the adversarial loss. This explicitly forces the generator to match the time-frequency distribution and improves stability and parameter efficiency over an IAF baseline. *Gap:* still a quality gap to autoregressive output; the discriminator is a single waveform-domain network with no explicit handle on periodicity.

**GAN-TTS.** Generates waveforms from linguistic features using an ensemble of **random-window discriminators (RWD)**: multiple discriminators each operating on randomly sampled windows of different sizes, some conditional on the linguistic features. It demonstrates that a *mixture of discriminators* viewing the signal differently is powerful and that GAN vocoders can be FLOP-efficient. *Gap:* the windows overlap heavily in what they cover, the discriminators are conditional and average their outputs, and the ensemble is organized around window *size* rather than any property tied to the signal's periodicity; quality still trails autoregressive output.

**LSGAN (the loss to adopt).** Standard GANs minimize a binary-cross-entropy objective whose gradient vanishes as the discriminator saturates. Least-squares GAN replaces the BCE terms with squared-error terms: the discriminator regresses real samples to 1 and fakes to 0, the generator pushes fakes toward 1. The squared-error form keeps gradients non-vanishing for samples on the correct side of the decision boundary but still far from the target, which stabilizes adversarial training of a high-capacity generator.

**Reconstruction-loss-assisted GANs.** A recurring lesson (from image-to-image translation and from STFT-loss vocoders) is that pairing the adversarial objective with an explicit reconstruction loss to the ground truth produces sharper, more faithful output and stabilizes early training. For audio, a loss in a perceptually weighted frequency domain (mel) is the natural choice.

**Normalization tools.** Weight normalization reparameterizes each weight vector by a separate magnitude and direction, accelerating convergence. Spectral normalization bounds the largest singular value of each weight matrix, capping the discriminator's Lipschitz constant and stabilizing training. Both are standard, drop-in.

## Evaluation settings

The natural yardsticks already in place:

- **LJSpeech** — 13,100 single-speaker English clips, ~24 hours total, 16-bit PCM, used here at a 22.05 kHz sample rate, with no manipulation. The standard single-speaker vocoder benchmark.
- **VCTK** — ~44,200 clips from 109 speakers (~44 hours), downsampled to 22.05 kHz, with several speakers held out entirely, to test mel-inversion generalization to *unseen* speakers.
- **Input features:** 80-band mel-spectrograms, computed with FFT size 1024, window 1024, hop 256.
- **Quality metric:** 5-scale Mean Opinion Score (MOS) crowd-sourced from human raters, reported with 95% confidence intervals; all clips volume-normalized.
- **Speed metric:** synthesis throughput relative to real time, measured on a single V100 GPU and on a laptop CPU, at 32-bit float with no inference-time optimization.
- **Optimizer protocol:** AdamW with β₁=0.8, β₂=0.99, weight decay 0.01; initial learning rate 2×10⁻⁴ with a 0.999 per-epoch decay.

## Code framework

A generic GAN-vocoder harness is available. The mel front-end, the optimizers, the leaky-ReLU/weight-norm primitives, the L1 distance, and the adversarial training loop already exist; the generator architecture and the discriminator architecture are unresolved.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1

def mel_spectrogram(wav, n_fft=1024, hop=256, win=1024, n_mels=80):
    # STFT -> mel filterbank -> log. Used both as model input and as a loss target.
    ...

class Generator(nn.Module):
    """Mel-spectrogram -> raw waveform, fully parallel (no autoregression)."""
    def __init__(self, cfg):
        super().__init__()
        # TODO: audio upsampling stack
        pass
    def forward(self, mel):
        # TODO: upsample mel to waveform resolution; return waveform in [-1, 1]
        raise NotImplementedError

class Discriminator(nn.Module):
    """Distinguishes real from generated waveforms; also exposes intermediate features."""
    def __init__(self):
        super().__init__()
        # TODO: waveform discriminator stack
        pass
    def forward(self, y, y_hat):
        # TODO: return per-sub-discriminator scores for real & fake, plus feature maps
        raise NotImplementedError

# --- generic loss slots ---
def discriminator_loss(real_scores, fake_scores):
    # TODO: the adversarial loss form
    raise NotImplementedError

def generator_loss(fake_scores):
    # TODO
    raise NotImplementedError

def feature_loss(fmap_real, fmap_fake):
    # TODO: a learned similarity between real/fake via discriminator features
    raise NotImplementedError

def train_step(G, D, mel, wav, optG, optD):
    # standard alternating GAN updates; spectral reconstruction can be added to G loss
    ...
```
