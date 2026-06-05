# Context: neural text-to-speech synthesis

## Research question

Generate natural-sounding speech from text. For decades the field traded off
along one axis: pipelines that sound least unnatural require the most hand-built
linguistic machinery, and the systems that simplify the front end sound worse.
The precise problem is to synthesize speech that is hard to distinguish from a
human, while learning as much of the pipeline as possible *from data* rather than
from domain expertise — no hand-engineered text analysis, no separate duration and
pitch models, no lossy intermediate features that cap audio quality. A solution
would map characters end-to-end to a waveform of near-human naturalness.

## Background

**The historical pipeline and its trade-off.** Concatenative synthesis with unit
selection stitches together small pre-recorded waveform units; it was the
long-standing state of the art but suffers boundary artifacts and is inflexible.
Statistical parametric synthesis instead generates smooth trajectories of speech
features (e.g. spectral envelope, fundamental frequency, durations) that a
*vocoder* turns into a waveform; it removed concatenation artifacts but the audio
sounds muffled and buzzy, because the features are a lossy, hand-designed summary
and the vocoder is a fixed signal-processing model.

**WaveNet (van den Oord et al. 2016).** An autoregressive generative model of the
raw waveform: a stack of dilated causal convolutions models p(x_t | x_{<t},
conditioning) directly in the time domain, producing audio that begins to rival
real human speech. The catch is its *inputs*: WaveNet is conditioned on
linguistic features, predicted log-F₀, and phoneme durations — exactly the
hand-engineered front end (text analysis, a pronunciation lexicon, a duration
model) that the field wants to be rid of. So WaveNet solves the *vocoding* half
beautifully while still demanding the expensive linguistic half.

**Tacotron (Wang et al. 2017).** A sequence-to-sequence model with attention that
maps a character sequence directly to a *magnitude spectrogram*, replacing the
linguistic/acoustic feature front end with one network trained from data. To get a
waveform it uses Griffin-Lim phase estimation followed by inverse STFT — explicitly
a placeholder, since Griffin-Lim leaves characteristic artifacts and lower quality
than a neural vocoder. So Tacotron solves the *front-end* half (text → acoustic
features, learned) while leaving the vocoding half to a weak algorithm.

**Sequence-to-sequence with attention.** The encoder-decoder-with-attention
framework (Sutskever et al. 2014; Bahdanau et al. 2014) lets a decoder generate a
variable-length output while, at each step, computing a context vector as a
soft-weighted sum of encoder states. **Additive (Bahdanau) attention** scores each
encoder state h_j against the decoder state s_{i-1} via
e_{ij} = vᵀ tanh(W s_{i-1} + V h_j), softmaxed to weights α_{ij}. For a monotonic,
strictly-advancing alignment like text→speech, content-only attention can stall,
skip, or repeat. **Location-sensitive attention** (Chorowski et al. 2015) extends
the additive form with features derived from the *cumulative* attention weights of
prior steps, encouraging the alignment to move forward consistently.

**Mel-frequency spectrograms.** A mel spectrogram applies a nonlinear (auditory)
warping to the STFT magnitude's frequency axis, emphasizing lower frequencies
critical to intelligibility and compressing high frequencies; ~80 channels suffice
per frame. It is smooth, easily computed from the waveform (so the acoustic model
and vocoder can be trained separately), phase-invariant within a frame (so a
squared-error loss is well-behaved), and far lower-dimensional and lower-level than
hand-built linguistic features.

**Regularizers and output modeling.** Dropout (Srivastava et al. 2014) and
zoneout (Krueger et al. 2016, which stochastically keeps an RNN's previous hidden
state) regularize the recurrent network. For continuous waveform-sample outputs, a
mixture of logistics (PixelCNN++; Salimans et al. 2017; Parallel WaveNet) replaces
a 256-way softmax over quantized amplitudes, giving a smooth distribution over
16-bit samples.

## Baselines

**Concatenative / statistical parametric synthesis.** Core idea: select-and-stitch
units, or generate hand-designed acoustic features for a vocoder. Gaps:
concatenation has boundary artifacts and no flexibility; parametric output sounds
muffled and unnatural because the features and vocoder are lossy and fixed.

**WaveNet conditioned on linguistic features (van den Oord et al. 2016).** Core
idea: autoregressive raw-waveform generation, p(x_t | x_{<t}, c), via dilated
causal convolutions. Gap: requires a full hand-engineered front end (text
analysis, lexicon, predicted F₀ and durations) to produce its conditioning
inputs — exactly the expertise the field wants to eliminate.

**Tacotron (Wang et al. 2017).** Core idea: char→magnitude-spectrogram seq2seq with
attention, learned end-to-end. Gap: vocodes with Griffin-Lim + inverse STFT, a
phase-estimation placeholder that injects artifacts and caps naturalness; uses
heavier "CBHG" blocks and GRUs.

**Deep Voice 3 / Char2Wav (Ping et al. 2017; Sotelo et al. 2017).** Core idea:
similar end-to-end neural TTS with a neural vocoder. Gaps: naturalness not shown to
rival human speech; Char2Wav uses traditional vocoder features and a quite
different architecture.

## Evaluation settings

The standard yardstick is a **mean opinion score (MOS)**: human raters score the
naturalness of synthesized utterances on a 1–5 scale, typically collected via
crowd-sourcing on a held-out test set, and compared against ground-truth recorded
speech (and side-by-side preference tests against baselines). A single-speaker
US-English studio recording corpus (tens of hours of (text, audio) pairs) is the
natural training/eval material; audio at 24 kHz. These protocols and the MOS
metric predate the method.

## Code framework

The pre-method toolkit: an STFT/mel front end, the standard recurrent
seq2seq-with-attention primitives (character embedding, bidirectional-LSTM
encoder, an attention mechanism, an autoregressive LSTM decoder), convolutional
blocks with batch norm, and an autoregressive raw-waveform generative model
(WaveNet) to act as a neural vocoder. The empty slots: which attention keeps the
alignment monotonic, the exact decoder/output structure that predicts acoustic
frames and knows when to stop, and what acoustic representation bridges the two
stages.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Mel front end (known): STFT magnitude -> mel filterbank -> log compression.
def wav_to_mel(wav, n_mels=80):
    # 50ms window, 12.5ms hop, Hann; mel 125Hz-7.6kHz; clip to 0.01 then log.
    ...

# Encoder (known seq2seq primitive): char embedding -> conv stack -> BiLSTM.
class Encoder(nn.Module):
    def __init__(self, n_chars, d=512):
        super().__init__()
        self.embed = nn.Embedding(n_chars, d)
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.BatchNorm1d(d),
                          nn.ReLU(), nn.Dropout(0.5)) for _ in range(3))
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
    def forward(self, chars):
        x = self.embed(chars).transpose(1, 2)
        for c in self.convs:
            x = c(x)
        return self.lstm(x.transpose(1, 2))[0]

# Attention that summarizes the encoder into a per-step context vector.
class Attention(nn.Module):
    def __init__(self, attn_dim=128):
        super().__init__()
        # TODO: which attention mechanism keeps a monotonic, advancing alignment?
        pass
    def forward(self, query, memory, prev_weights):
        pass

# Autoregressive acoustic decoder: predict the acoustic frames and a stop signal.
class Decoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        # TODO: prenet bottleneck, recurrence, frame projection, stop-token, postnet.
        pass
    def forward(self, memory, targets=None):
        pass

# Neural vocoder (known: WaveNet) conditioned on the predicted acoustic frames.
class WaveNetVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: condition WaveNet on the chosen acoustic representation; output dist.
        pass
    def forward(self, frames):
        pass
```
