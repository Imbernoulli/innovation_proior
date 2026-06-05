# Context: self-supervised representations from raw speech

## Research question

Modern speech recognizers are accurate only because they are fed thousands of
hours of *transcribed* speech. That requirement is the bottleneck: of the nearly
7,000 languages spoken on Earth, only a handful have transcription budgets large
enough to train a competitive recognizer, while raw, untranscribed audio is
abundant for almost all of them. The precise problem is to learn a
representation of speech *from audio alone* — no transcripts — such that, once a
representation is learned, a recognizer can be fitted to it with a tiny amount of
labeled data (tens of minutes, not thousands of hours) and still reach accuracy
that today demands the full labeled corpus. A solution would have to (a) consume
raw waveform, (b) train with an objective that needs no labels, (c) produce
contextualized frame representations that capture the phonetic structure of
speech, and (d) transfer to recognition with a thin, cheaply-trained head.

## Background

**Self-supervised pre-training as the paradigm.** In language and vision, the
recipe that works is: define a *pretext* task whose targets come from the input
itself, train a large model on huge unlabeled data, then fine-tune on the small
labeled task. Masked language modeling (Devlin et al. 2018) hides a fraction of
tokens and predicts them from bidirectional context; the resulting
representations transfer broadly. In vision the dominant pretext family is
*contrastive*: pull together views of the same instance, push apart different
ones (He et al. 2019; Chen et al. 2020). The open question for speech is which of
these transfers, because speech is neither discrete tokens nor a single static
image — it is a continuous-valued sequence with no given segmentation into units.

**Contrastive predictive coding (van den Oord et al. 2018).** The load-bearing
idea behind self-supervised *sequence* learning: instead of reconstructing future
samples (expensive, and forces modeling of low-level detail that may be
irrelevant), learn by *prediction in latent space via classification*. Encode
the signal to latents, summarize the past into a context vector, and train the
context to identify the true future latent among randomly-drawn distractors with
a noise-contrastive (InfoNCE) loss. This maximizes a bound on the mutual
information between context and future and avoids ever generating raw audio.

**Product quantization and the Gumbel-softmax.** To turn a continuous vector into
a discrete code differentiably, two pieces exist. Product quantization (Jégou et
al. 2011) splits the vector into G sub-vectors and quantizes each against its own
small codebook of V entries, so G codebooks of size V express up to Vᴳ distinct
codes with few parameters. The Gumbel-softmax (Jang et al. 2016; Maddison et al.
2014) makes a categorical choice differentiable: add Gumbel noise to logits, take
a temperature-controlled softmax, and in the hard/straight-through variant pick
the argmax on the forward pass while letting gradients flow through the soft
distribution on the backward pass.

**CTC.** Connectionist Temporal Classification (Graves et al. 2006) trains a
frame-wise classifier to emit a label sequence without a frame-level alignment,
by summing over all alignments (including a blank symbol) that collapse to the
target. It is the standard way to attach a recognizer head to a frame-level
acoustic representation when no alignment is available.

**Empirical findings that motivate the design.** Two diagnostic observations
about *prior* self-supervised speech systems frame the work. First, systems that
learn a discretization of the audio in a *separate first step* and then learn a
contextual model on top of those frozen units underperform: freezing the units
early throws away information the context model could have used. Second, when one
quantizes the representation, *what* gets quantized matters — continuous,
high-information features carry detailed artifacts (speaker, background) that make
a self-supervised task too easy and yield representations that transfer worse to
recognition, whereas a discretized target removes some of that nuisance detail.

## Baselines

**wav2vec / CPC for speech (Schneider et al. 2019; van den Oord et al. 2018).**
A multi-layer convolutional encoder maps raw audio to latents; a context network
(also convolutional) predicts future latents via a contrastive loss against
distractors. Core idea and math: InfoNCE — maximize
log[exp(sim(c, z₊)) / Σ exp(sim(c, z))] over a positive future latent z₊ and
sampled negatives. Gap: the representations are continuous throughout and the
context network is convolutional, with a limited receptive field; there is no
discrete inventory of speech units and no global self-attention.

**vq-wav2vec / DiscreteBERT (Baevski et al. 2019, 2019).** A two-step pipeline:
first learn a *vector-quantized* discretization of audio (vq-wav2vec) so each
frame becomes a discrete token; then run a BERT-style masked-token model over the
discrete tokens to get contextual representations, and fine-tune for recognition.
Core idea: by discretizing first, the well-developed masked-language-model
machinery of NLP applies directly to speech. Gap: the discretization is learned
and *frozen* before the context model sees it, so the two stages cannot co-adapt;
the context model only ever ingests the already-lossy discrete tokens, never the
richer continuous features. This is exactly the "fixed units learned in a prior
step" weakness above.

**Semi-supervised self-training / pseudo-labeling (Park et al. 2020; Xu et al.
2020).** Train a teacher on the available labels, label the unlabeled audio with
it, and retrain a student on the union, iterating. Strong, but requires labels to
start, multiple rounds of labeling/filtering/retraining, and a recognizer that is
already good — it improves a supervised system rather than learning
representations from audio alone.

## Evaluation settings

The natural yardsticks already exist. **LibriSpeech** (Panayotov et al. 2015):
960 hours of read English audiobooks with transcripts; the standard dev/test
clean and other splits, scored by **word error rate (WER)**. **Libri-Light** (Kahn
et al. 2020): the same audio domain repackaged for low-resource study, with
limited-label training subsets of 10 minutes, 1 hour, and 10 hours, plus a 100-hour
clean subset, and a 53.2k-hour unlabeled set (LV-60k). **TIMIT** (Garofolo et al.
1993): ~5 hours of audio with phoneme labels, standard train/dev/test split with
phone labels collapsed to 39 classes, scored by **phoneme error rate (PER)**.
Decoding can optionally fuse an external language model (an n-gram or a
Transformer LM trained on the LibriSpeech LM corpus) via beam search. Inputs are
16 kHz raw waveform. These datasets and metrics predate the method and are the
fixed measuring sticks.

## Code framework

The pre-method toolkit is a convolutional front end over raw waveform, a
Transformer encoder, an Adam optimizer with warmup/decay, a differentiable
discretizer (Gumbel-softmax over product-quantization codebooks), and a CTC head
for fine-tuning. The pretext objective, the masking of the latent sequence, the
choice of contrastive targets, and the auxiliary regularizer are the empty slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Raw-waveform convolutional front end (known primitive): a stack of strided
# temporal convolutions + norm + GELU producing a latent frame sequence.
class ConvFeatureEncoder(nn.Module):
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, s),
                       nn.GroupNorm(1, c_out), nn.GELU()]   # layer-norm-like + GELU
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):                 # wav: (B, T_samples)
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)  # (B, T, C)

# Transformer encoder (known): self-attention is all-to-all; positions injected
# by a grouped conv rather than fixed sinusoids.
class TransformerContext(nn.Module):
    def __init__(self, d=768, layers=12, heads=8, ffn=3072):
        super().__init__()
        self.pos_conv = nn.Conv1d(d, d, 128, padding=64, groups=16)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d, heads, ffn, 0.1, F.gelu, batch_first=True)
            for _ in range(layers))
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        p = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)[:, :x.size(1)]
        x = self.ln(x + F.gelu(p))
        for l in self.layers:
            x = l(x)
        return x

# Differentiable discretizer (known primitive): product quantization chosen by a
# hard Gumbel-softmax. Returns a quantized vector per frame plus the per-group
# probabilities (used by whatever regularizer the objective needs).
class GumbelProductQuantizer(nn.Module):
    def __init__(self, in_dim, groups=2, vars=320, out_dim=256):
        super().__init__()
        self.G, self.V = groups, vars
        self.weight = nn.Linear(in_dim, groups * vars)
        self.codebook = nn.Parameter(torch.randn(1, groups * vars, out_dim // groups))
        self.proj = nn.Linear(out_dim, out_dim)
        self.tau = 2.0
    def forward(self, z):                   # z: (B, T, in_dim)
        # TODO: logits -> hard gumbel-softmax per group -> select & concat codes
        #       -> linear; also return averaged per-group probs for the regularizer.
        pass

# Define the self-supervised pretext signal over the latent sequence.
def mask_latents(z):
    # TODO: which frames become prediction targets, and what replaces them in the
    #       context-network input?
    pass

def pretext_loss(context_out, targets, probs):
    # TODO: the objective that makes the contextual representation predict the
    #       right target at each masked frame, plus any codebook regularizer.
    pass

# Fine-tuning head (known primitive): a linear projection trained with CTC.
class CTCHead(nn.Module):
    def __init__(self, d=768, n_vocab=32):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, x):
        return self.proj(x)
```
