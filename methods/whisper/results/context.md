## Research question

A speech recognizer should work reliably "out of the box" across many recording
conditions, accents, domains, and languages — not just on the single distribution
it was tuned for. The dominant pipeline pre-trains an audio encoder, then
*fine-tunes* it on each target dataset. The open question is what training
recipe yields broad zero-shot generalization across environments, tasks, and
languages without dataset-specific fine-tuning of a decoder.

## Background

**Self-supervised audio pre-training.** Self-supervised pre-training on raw audio
(wav2vec 2.0; Baevski et al. 2020) learns *encoder* representations from
enormous unlabeled corpora — scaled to ~1,000,000 hours (Zhang et al. 2021) — far
beyond the ~1,000 hours of a typical academic supervised set. The objective is
contrastive masked prediction over quantized latent speech units; the result is an
encoder whose representations transfer well to downstream tasks via fine-tuning.

**Robustness and distribution shift.** Machine-learning models can exploit
dataset-specific patterns. In vision, fine-tuning on ImageNet raised that
dataset's accuracy by ~9% without improving average accuracy on seven other
natural-image datasets of the same objects (Radford et al. 2021): a model can
perform well on one distribution while behaving differently on another.

**Supervised multi-dataset training.** Recognizers pre-trained in a *supervised*
fashion across many datasets/domains transfer broadly to held-out data
(Narayanan et al. 2018; Likhomanenko et al. 2020; Chan et al. 2021). SpeechStew
(Chan et al. 2021) mixes seven supervised datasets into 5,140 hours of training
material.

**Weak supervision trades quality for quantity.** Relaxing the demand for
gold-standard human transcripts lets automated pipelines harvest far more
(audio, transcript) pairs from the internet — 10,000 to 30,000 hours of noisier
data (Chen et al. 2021; Galvez et al. 2021). In vision, moving from curated
datasets to larger weakly-supervised ones improves robustness and generalization
(Mahajan et al. 2018; Kolesnikov et al. 2020).

**The sequence-to-sequence Transformer.** An encoder-decoder Transformer (Vaswani
et al. 2017) consumes a source sequence with a bidirectional encoder and generates
a target token sequence autoregressively with a causally-masked decoder that
cross-attends to the encoder; trained by next-token cross-entropy. It scales
reliably across modalities. Byte-level BPE tokenization (GPT-2; Radford et al. 2019)
gives an open vocabulary over arbitrary text.

## Baselines

**Self-supervised encoders + fine-tuning (wav2vec 2.0; Baevski et al. 2020).**
Contrastive masked pre-training on unlabeled audio, then fine-tune for
recognition. Core idea: learn representations cheaply from unlabeled data.
Performance is reported after dataset-specific fine-tuning.

**Supervised multi-dataset training (SpeechStew; Chan et al. 2021).** Pool several
high-quality supervised corpora and train one model. Core idea: domain diversity
buys robustness. Total available gold-standard supervision is ~5k hours.

**Weakly-supervised mid-scale datasets (GigaSpeech, People's Speech; Chen et al.
2021; Galvez et al. 2021).** Automated pipelines harvest 10k–30k hours of noisier
labeled audio. Core idea: relax transcript quality to gain quantity. These systems
are typically evaluated with in-distribution training splits.

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
are the standing datasets, metrics, and input features for the evaluation setting.

## Code framework

The available toolkit is a log-mel front end, a standard encoder-decoder Transformer
(pre-activation residual blocks, sinusoidal/learned position embeddings, tied
input-output token embeddings), a byte-level BPE tokenizer, and an AdamW training
loop with warmup and gradient clipping.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Log-mel front end: 80-channel spectrogram, 25ms window, 10ms stride.
def log_mel_spectrogram(wav, n_mels=80):
    # returns (n_mels, n_frames)
    ...

# Standard encoder-decoder Transformer primitives.
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

def sinusoids(length, d, max_timescale=10000):
    inv = torch.exp(-torch.log(torch.tensor(max_timescale)) *
                    torch.arange(d // 2) / (d // 2 - 1))
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([t.sin(), t.cos()], dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, d, layers, heads, n_ctx=1500):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.register_buffer('pos', sinusoids(n_ctx, d))
        self.blocks = nn.ModuleList(ResidualAttentionBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, mel):
        x = F.gelu(self.conv1(mel)); x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.pos[:x.size(1)]
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

# Assemble training examples and the decoder target.
def build_training_example(audio, transcript, meta):
    # TODO: the training recipe — what data feeds the model and what the decoder
    #       is asked to predict.
    pass
```
