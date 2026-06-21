# Research question

Generate natural-sounding speech from text. The field has explored pipelines
that vary in how much hand-built linguistic machinery they require and how much
they learn from paired text and audio. The question is how to synthesize speech
that is hard to distinguish from a human recording, learning as much of the
pipeline as possible from data.

# Background

Concatenative synthesis with unit selection stitches together small
pre-recorded waveform units. Statistical parametric synthesis instead generates
smooth trajectories of speech features, such as spectral envelope, fundamental
frequency, and durations, that a vocoder turns into a waveform.

WaveNet models the raw waveform autoregressively: a stack of dilated causal
convolutions models p(x_t | x_{<t}, conditioning) directly in the time domain.
A complete WaveNet TTS pipeline is conditioned on linguistic features, predicted
log-F0, and phoneme durations, which means a text-analysis system, a
pronunciation lexicon, and a duration model are part of the pipeline.

Tacotron maps a character sequence directly to a magnitude spectrogram with a
sequence-to-sequence attention model, replacing the linguistic and acoustic
feature front end with one network trained from data. To get a waveform it uses
Griffin-Lim phase estimation followed by inverse STFT.

The encoder-decoder-with-attention framework lets a decoder generate a
variable-length output while, at each step, computing a context vector as a
soft-weighted sum of encoder states. Additive attention scores each encoder
state h_j against the decoder state s_{i-1} with
e_ij = v^T tanh(W s_{i-1} + V h_j), then normalizes the scores into weights.
For text-to-speech, the alignment is mostly monotonic and advancing: speech
usually consumes text in order. Location-sensitive attention extends additive
attention with features computed from the cumulative attention weights of
previous decoder steps, which gives the scoring function a memory of how far
through the input it has moved.

A mel-frequency spectrogram applies an auditory frequency scale to the STFT
magnitude, warping the linear frequency axis onto a perceptual scale. Like the
linear-frequency STFT magnitude it is a standard signal-processing representation
computed directly from the waveform.

Dropout regularizes feed-forward and convolutional layers. Zoneout regularizes
recurrent layers by stochastically preserving previous hidden states. For
continuous waveform-sample outputs, a mixture of logistics can model 16-bit
audio samples without reducing the output to a 256-way categorical distribution.

# Baselines

Concatenative and statistical parametric synthesis either select and stitch
recorded units or generate hand-designed acoustic features for a vocoder.

WaveNet conditioned on linguistic features is an autoregressive raw-waveform
generator with dilated causal convolutions.

Tacotron is a character-to-spectrogram sequence model with attention that uses
Griffin-Lim plus inverse STFT to produce a waveform.

Deep Voice 3 and Char2Wav are related neural TTS systems. Deep Voice 3 uses a
different architecture; Char2Wav uses traditional vocoder features as the
acoustic representation.

# Evaluation settings

The usual naturalness yardstick is mean opinion score: human raters score
synthesized utterances on a 1-5 scale, often on a fixed held-out set, with
ground-truth recorded speech and established TTS systems as comparison points.
A single-speaker US-English corpus with normalized text and 24 kHz audio is a
natural training and evaluation setup for this problem.

# Code framework

The available toolkit is an STFT-derived acoustic front end, recurrent
sequence-to-sequence primitives, convolutional blocks with batch normalization,
attention over an encoded character sequence, autoregressive LSTM decoding,
sequence-level acoustic refinement, and an autoregressive raw-waveform generator
that can be used as a neural vocoder. The design is left open: it is for you to
work out how these pieces fit together to solve the problem.

```python
import torch
import torch.nn as nn

def acoustic_features(wav):
    raise NotImplementedError

class TextEncoder(nn.Module):
    def __init__(self, n_chars, d_model):
        super().__init__()
        self.embed = nn.Embedding(n_chars, d_model)
        self.local_context = nn.ModuleList()
        self.sequence_context = nn.Identity()

    def forward(self, chars):
        x = self.embed(chars)
        raise NotImplementedError

class Alignment(nn.Module):
    def forward(self, query, memory):
        raise NotImplementedError

class AcousticDecoder(nn.Module):
    def forward(self, memory, targets=None):
        raise NotImplementedError

class AcousticRefiner(nn.Module):
    def forward(self, frames):
        raise NotImplementedError

class NeuralVocoder(nn.Module):
    def forward(self, acoustic_frames, audio_prefix):
        raise NotImplementedError
```
