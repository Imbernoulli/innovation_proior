# Context: robust speech recognition without per-deployment fine-tuning

## Research question

A speech recognizer should work reliably "out of the box" across many recording
conditions, accents, domains, and languages — not just on the single distribution
it was tuned for. The dominant pipeline does the opposite: pre-train an audio
encoder, then *fine-tune* it on each target dataset. The precise problem is that
this fine-tuning step is both a usability burden (it needs a skilled practitioner
per deployment) and a robustness hazard (fitting to one dataset's idiosyncrasies
inflates in-distribution accuracy while the model still makes basic errors on
other distributions). The goal is a single system that generalizes broadly
*zero-shot* — usable directly, with no dataset-specific fine-tuning of a decoder —
across environments, tasks, and languages. The open question is what training
recipe yields that breadth.

## Background

**The unsupervised-encoder gap.** Self-supervised pre-training on raw audio
(wav2vec 2.0; Baevski et al. 2020) learns excellent *encoder* representations from
enormous unlabeled corpora — scaled to ~1,000,000 hours (Zhang et al. 2021) — far
beyond the ~1,000 hours of a typical academic supervised set. But because the
objective is purely unsupervised, these systems learn no equivalently capable
*decoder* mapping representations to usable text; they require a fine-tuning stage
to perform recognition at all. That fine-tuning is the source of both the
usability and the brittleness problems.

**Why fine-tuning hurts robustness.** Machine-learning models are very good at
exploiting dataset-specific patterns that boost held-out accuracy *within* a
dataset but do not transfer. In vision, fine-tuning on ImageNet was documented to
raise that dataset's accuracy by ~9% while *not* improving average accuracy on
seven other natural-image datasets of the same objects (Radford et al. 2021): a
model can look "superhuman" on one distribution and still fail basic cases on
another, precisely because it learned that distribution's quirks. The same risk
applies to a fine-tuned speech decoder.

**Supervised multi-dataset training generalizes better.** Recognizers pre-trained
in a *supervised* fashion across many datasets/domains are more robust and
transfer better to held-out data than single-source models (Narayanan et al. 2018;
Likhomanenko et al. 2020; Chan et al. 2021). SpeechStew (Chan et al. 2021) mixes
seven supervised datasets into 5,140 hours — robust, but tiny next to the million
unsupervised hours.

**Weak supervision trades quality for quantity.** Relaxing the demand for
gold-standard human transcripts lets automated pipelines harvest far more
(audio, transcript) pairs from the internet — 10,000 to 30,000 hours of noisier
data (Chen et al. 2021; Galvez et al. 2021). Vision saw that moving from curated
datasets to much larger weakly-supervised ones improves robustness and
generalization (Mahajan et al. 2018; Kolesnikov et al. 2020). Speech had not yet
pushed this scale.

**The sequence-to-sequence Transformer.** An encoder-decoder Transformer (Vaswani
et al. 2017) consumes a source sequence with a bidirectional encoder and generates
a target token sequence autoregressively with a causally-masked decoder that
cross-attends to the encoder; trained by next-token cross-entropy. It scales
reliably and, as an audio-conditional language model, can in principle emit *any*
text format directly — punctuation, casing — removing the separate inverse-text-
normalization stage. Byte-level BPE tokenization (GPT-2; Radford et al. 2019)
gives an open vocabulary over arbitrary text.

## Baselines

**Self-supervised encoders + fine-tuning (wav2vec 2.0; Baevski et al. 2020).**
Contrastive masked pre-training on unlabeled audio, then fine-tune for
recognition. Core idea: learn representations cheaply from unlabeled data. Gap: no
pre-trained decoder, so each deployment needs a fine-tuned head, with the
robustness hazard above; performance is reported *after* dataset-specific
fine-tuning, not zero-shot.

**Supervised multi-dataset training (SpeechStew; Chan et al. 2021).** Pool several
high-quality supervised corpora and train one model. Core idea: domain diversity
buys robustness. Gap: the total available gold-standard supervision (~5k hours) is
orders of magnitude smaller than unlabeled corpora, capping how far this can go.

**Weakly-supervised mid-scale datasets (GigaSpeech, People's Speech; Chen et al.
2021; Galvez et al. 2021).** Automated pipelines harvest 10k–30k hours of noisier
labeled audio. Core idea: relax transcript quality to gain quantity. Gap: only a
few times larger than the sum of clean datasets, still far below unsupervised
scale, and these systems are typically still evaluated with in-distribution
training splits rather than zero-shot.

## Evaluation settings

The yardstick is *zero-shot* generalization: evaluate on the test splits of many
existing speech datasets *without* using any of their training data, so the metric
measures broad transfer rather than in-distribution fit. Short-form English ASR
(e.g. LibriSpeech, TED-LIUM 3, Common Voice, CHiME, etc.), multilingual ASR
(e.g. Multilingual LibriSpeech, VoxPopuli, Fleurs, Common Voice in many
languages), and X→English speech *translation* (e.g. CoVoST 2, Fleurs) are the
task families. The primary metric is **word error rate (WER)** (BLEU for
translation); because WER over-penalizes innocuous formatting/style differences —
acute for a zero-shot model that has never seen a dataset's transcript
conventions — a careful text normalizer is applied before scoring. Audio is 16 kHz;
features are 80-channel log-mel spectrograms (25 ms window, 10 ms stride). These
datasets and metrics predate the method.

## Code framework

The pre-method toolkit: a log-mel front end, a standard encoder-decoder Transformer
(pre-activation residual blocks, sinusoidal/learned position embeddings, tied
input-output token embeddings), a byte-level BPE tokenizer, and an AdamW training
loop with warmup and gradient clipping. The empty slots are: how raw web data
becomes training examples, and how a *single* decoder is told which of many tasks
to perform on the same audio.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Log-mel front end (known): 80-channel spectrogram, 25ms window, 10ms stride.
def log_mel_spectrogram(wav, n_mels=80):
    # returns (n_mels, n_frames)
    ...

# Standard encoder-decoder Transformer primitives (known).
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d, heads, cross=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_x = nn.LayerNorm(d) if cross else None
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True) if cross else None
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, xa=None, mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask)[0]
        if self.cross is not None:
            h = self.ln_x(x)
            x = x + self.cross(h, xa, xa)[0]
        return x + self.mlp(self.ln2(x))

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, d, layers, heads):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList(ResidualAttentionBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        # sinusoidal position embedding added after the conv stem
    def forward(self, mel):
        x = F.gelu(self.conv1(mel)); x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)        # + sinusoids
        for b in self.blocks:
            x = b(x)
        return self.ln(x)

class TextDecoder(nn.Module):
    def __init__(self, vocab, d, layers, heads):
        super().__init__()
        self.token = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.empty(448, d))   # learned positions
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(d, heads, cross=True) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, tokens, audio):
        x = self.token(tokens) + self.pos[:tokens.size(1)]
        mask = torch.triu(torch.full((tokens.size(1),) * 2, float('-inf')), 1)
        for b in self.blocks:
            x = b(x, audio, mask)
        x = self.ln(x)
        return x @ self.token.weight.t()               # tied output projection

# Turn raw web (audio, transcript) data into training examples.
def build_example(audio, transcript, meta):
    # TODO: how raw internet pairs are filtered and segmented into model inputs/targets.
    pass

# Tell a single decoder which task to perform on the same audio.
def build_decoder_sequence(target_text, meta):
    # TODO: the token sequence the decoder is asked to predict, encoding the task
    #       and conditioning so one model covers many speech tasks.
    pass
```
