# Context

## Research question

Can a single neural network generate *raw* audio waveforms — the actual sequence of amplitude samples, sampled at wideband rates of at least 16,000 samples per second — directly, as a fully probabilistic, autoregressive model trained by maximum likelihood, with no vocoder and no hand-built signal model in between?

The difficulty is one of scale and memory at once. Audio has enormous temporal resolution: one second of speech is already 16,000 samples, and a few seconds is on the order of 10^5 samples. Yet the structure a listener cares about lives at many timescales simultaneously — the fine periodicity of a vowel (sub-millisecond), the identity of a phoneme (tens of milliseconds), the rhythm and intonation of a phrase (hundreds of milliseconds to seconds), and the harmonic continuity of music (seconds). A model that predicts the next sample from its past therefore has to *see* a very long stretch of past — thousands to tens of thousands of samples — to predict each single one well. A solution would have to (a) be a tractable likelihood over the raw samples so it can be trained and validated by log-likelihood, (b) reach a receptive field of thousands of samples, and (c) remain feasible to train on sequences this long.

## Background

**Autoregressive neural models of high-dimensional data.** A general and effective way to build a generative model over a high-dimensional object `x = {x_1,…,x_T}` is to factorize the joint distribution by the chain rule into a product of conditionals,

`p(x) = Π_{t=1}^{T} p(x_t | x_1,…,x_{t-1})`,

and to model each conditional with a neural network sharing parameters across positions. This had been pushed hard on images and text: PixelRNN and PixelCNN (van den Oord et al., 2016) model an image as a product over pixels `p(x) = Π_i p(x_i | x_{<i})`, scanning the grid in raster order, and neural language models do the same over word/character tokens (Jozefowicz et al., 2016). These models are trained by maximizing log-likelihood, which is tractable, so one can tune hyperparameters on a validation set and directly detect over/underfitting. Remarkably, they model joint distributions over thousands of strongly-dependent variables (e.g. 64×64 = 4096 pixels) and reach state-of-the-art likelihoods.

Two design findings from that line are load-bearing. First, **the output distribution.** Even when the data is implicitly continuous (pixel intensities, audio amplitudes), a *categorical* softmax over a discretized value works better than a parametric continuous density such as a mixture density network (Bishop, 1994) or a mixture of conditional Gaussian scale mixtures (Theis & Bethge, 2015): the categorical makes no assumption about the shape of the distribution and can model arbitrary, multimodal shapes (van den Oord et al., 2016). Second, **the nonlinearity.** Gated PixelCNN (van den Oord et al., 2016) replaced the rectified-linear activation inside the model with a *gated* unit — a tanh "content" path multiplied elementwise by a sigmoid "gate" path — and found it modeled images significantly better.

**Ordering constraints in convolutional autoregressive models.** For the chain-rule factorization to be valid, the network computing `p(x_t | x_{<t})` must not peek at `x_t` or any later sample. In a convolutional model this is enforced by *masking* the convolution so each output position only depends on earlier input positions; for images this is a masked 2-D convolution (van den Oord et al., 2016). The convenience of the convolutional form is that, at training time, all positions' conditionals can be computed *in parallel* in a single forward pass over the ground-truth sequence (the future is masked out), whereas generation is unavoidably sequential — each sample is drawn and fed back before the next.

**Recurrent models and the cost of sequential training.** Recurrent networks (LSTMs; Hochreiter & Schmidhuber, 1997) are the standard tool for sequence modeling and carry, in principle, unbounded memory through their hidden state. But two problems surface on raw audio. Training is inherently *sequential*: backpropagation through time unrolls step by step, which is very slow when the sequence is hundreds of thousands of steps long. And in practice the effective memory is limited — gradients struggle to carry dependencies across more than a few hundred steps — so the long-range context audio needs is hard to retain.

**Receptive field of stacked convolutions, and dilation.** A stack of ordinary convolutions grows its receptive field only *linearly* with depth: with `L` layers of filter width `k`, the receptive field is `L·(k-1)+1`. Reaching a receptive field of a thousand samples this way would require on the order of a thousand layers. *Dilated convolutions* (also "à trous", convolution with holes), known from signal processing (Holschneider et al., 1989; Dutilleux, 1989) and used for dense image prediction / segmentation (Chen et al., 2015; Yu & Koltun, 2016), apply a filter over a region larger than its length by skipping input positions with a fixed step (the dilation). They aggregate context at multiple scales while keeping the output at the same resolution as the input, and a stack of dilated convolutions with exponentially increasing dilation expands the receptive field exponentially with depth (Yu & Koltun, 2016).

**Residual and skip connections.** Very deep stacks are hard to optimize. Residual connections (He et al., 2015) — adding a layer's input to its output — let gradients flow and make deep stacks trainable. Skip connections that route each layer's output directly to the final read-out give the output access to features at every depth.

**Statistical parametric and concatenative speech synthesis (the prior art for TTS).** The dominant ways to synthesize a waveform from text were two. *Concatenative* synthesis (Hunt & Black, 1996; Moulines & Charpentier, 1990) stores a large database of recorded speech units and stitches the best-matching units together; it gives high segmental quality but is inflexible (it cannot smoothly change voice or prosody) and has a large footprint. *Statistical parametric* synthesis (Zen et al., 2009) instead trains a generative model — an HMM, a feed-forward DNN, or an LSTM-RNN (Zen & Sak, 2015) — over a sequence of *vocoder parameters* (cepstra or line spectral pairs for the vocal tract; fundamental frequency `F0` and aperiodicity for the source), extracted every ~5 ms, then reconstructs the waveform from the predicted parameters with a vocoder. It is small and flexible but its naturalness is reliably worse: the speech sounds muffled and has artifacts. Three causes are documented (Zen et al., 2009): the vocoder itself caps quality, the acoustic model is imperfect, and the predicted parameter trajectories are oversmoothed. Underlying all of these are strong, often-violated assumptions baked into the classical raw-audio models — e.g. linear predictive analysis (Itakura & Saito, 1970) models speech as a *linear* autoregressive *zero-mean Gaussian* process, `x_t = Σ_p a_p x_{t-p} + ε_t`, `ε_t ~ N(0, G²)`, estimated within a fixed-length window — whereas real speech is non-stationary on sub-20-ms scales, the sample-to-sample relationship is highly nonlinear, and the amplitude distribution is far from Gaussian.

**Companding of audio for coarse quantization.** Audio is usually stored as 16-bit integers (65,536 levels per sample). Telephony has long used *μ-law companding* (ITU-T G.711, 1988) — a logarithmic amplitude transform applied before coarse quantization — because the ear's sensitivity to amplitude is roughly logarithmic, so allocating quantization levels logarithmically (more resolution at small amplitudes) preserves perceived quality far better than uniform/linear quantization at the same bit depth.

## Baselines

**LSTM-RNN statistical parametric synthesizer (Zen & Sak, 2015).** An LSTM-RNN predicts vocoder parameter trajectories (spectral envelope, `F0`, aperiodicity) from linguistic features derived from text; a vocoder reconstructs the waveform. Core gap: the waveform is only ever produced by the vocoder, which imposes a quality ceiling; the predicted trajectories tend to be oversmoothed, producing muffled, buzzy speech. It also trains and generates step-by-step over the parameter sequence.

**HMM-driven unit-selection concatenative synthesizer (Gonzalvo et al., 2016).** Selects and concatenates recorded speech units guided by an HMM cost. Core gap: no generative model of the waveform at all — it can only replay recordings, so it cannot flexibly change speaker or prosody, joins can be audible, and it needs a large stored database.

**Classical linear AR / source-filter models of raw audio (LPC and relatives; Itakura & Saito, 1970).** Model the waveform as a linear autoregressive Gaussian process within a short analysis window. Core gap: the linearity, stationarity-within-window, and Gaussianity assumptions are all violated by real speech, so samples are noisy and lose the detail that makes audio sound natural; designing a good *nonlinear* causal filter by hand is hard.

**Image/text autoregressive models as the architectural reference point (PixelRNN/PixelCNN; neural LMs).** The chain-rule + masked-convolution + categorical-softmax recipe that succeeded on images and text. Core gap *for audio*: those sequences are short (thousands of variables) relative to audio (10^4–10^5 samples per second), and the receptive field needed for a single image pixel is small compared with what a single audio sample needs; it is an open question whether the same recipe can be made to cover thousands of samples of context cheaply.

## Evaluation settings

- **Datasets.** A multi-speaker English corpus (CSTR VCTK, 109 speakers, ~44 h) for free-form (text-unconditioned) speech generation; single-speaker professional North American English (~24.6 h) and Mandarin Chinese (~34.8 h) corpora — the same data used to build production TTS systems — for text-to-speech; music corpora (the MagnaTagATune set of tagged ~29 s clips, ~200 h; a ~60 h solo-piano set from public videos); and the TIMIT corpus for a phoneme-recognition probe.
- **Metrics.** Tractable log-likelihood on held-out audio for model selection and over/underfitting checks. For TTS quality, subjective human listening tests: paired-comparison preference tests (listeners choose which of two samples they prefer, or "neutral") and mean-opinion-score tests (naturalness rated 1–5), run blind and crowdsourced over sentences not seen in training. For the recognition probe, phone error rate.
- **Protocol.** Train the autoregressive model by maximum likelihood with a gradient optimizer; conditionals for all timesteps computed in parallel over ground-truth audio at training time, sampling sequential at generation time. For TTS, the model is conditioned on linguistic features (and optionally `log F0`) derived from text, with external models predicting phone durations and `F0`; baselines (LSTM parametric, HMM concatenative) are built from the *same* datasets and linguistic features so the comparison is fair. Audio is μ-law companded to 8 bits before modeling.

## Code framework

Before the method exists, the pieces below are already standard: a μ-law companding transform with coarse quantization, a one-hot encoding of the quantized samples, a causal (mask-shifted) 1-D convolution primitive that forbids dependence on future samples, an optimizer, a categorical cross-entropy loss, and a training loop that computes all timesteps' conditionals in parallel over a ground-truth waveform. What does not yet exist is the network that maps the (causally-masked) sample history to the parameters of the next-sample distribution with a large enough receptive field — that is the one empty slot.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT = 256  # quantization levels after companding (8-bit)


def mu_law_encode(audio, mu=QUANT - 1):
    # logarithmic companding then quantize to `mu+1` integer levels in [0, mu]
    audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def mu_law_decode(quantized, mu=QUANT - 1):
    signal = 2 * (quantized.astype(np.float32) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


def causal_conv1d(x, weight, dilation):
    # 1-D convolution that depends only on current and PAST inputs:
    # left-pad by (k-1)*dilation and drop the tail so output[t] sees x[<=t] only.
    pass  # known primitive


class SequenceModel(nn.Module):
    """The architecture we will design: maps a causally-masked waveform history
    to the parameters of p(x_t | x_{<t}). It must reach a receptive field of
    thousands of samples while staying cheap to train. This is the empty slot."""

    def __init__(self):
        super().__init__()
        # TODO: the architecture is the contribution

    def forward(self, x_onehot):
        # TODO: returns per-timestep logits over QUANT classes
        pass


def train_step(model, waveform, optimizer):
    # waveform: int waveform in [0, QUANT); predict each sample from its past
    x = F.one_hot(waveform[:, :-1], QUANT).float().transpose(1, 2)
    target = waveform[:, 1:]
    logits = model(x)                       # (B, QUANT, T-1)
    loss = F.cross_entropy(logits, target)  # maximize Σ log p(x_t | x_{<t})
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
