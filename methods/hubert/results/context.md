# Context: self-supervised speech representation by predicting discovered units

## Research question

The aim is to learn speech representations from raw audio without transcripts,
such that downstream recognition needs very little labeled data. The sharp
difficulty, unique to speech, is that the self-supervision target is not given.
Text comes pre-segmented into a known lexicon of word-pieces, so a masked-token
objective has obvious discrete targets to predict. An image is one instance, so
the instance-discrimination objectives of vision apply. Speech is a
*continuous-valued sequence* with three problems at once: (1) each utterance
contains many sounds, breaking the one-instance-per-input assumption; (2) there is
no pre-existing inventory of discrete sound units to predict; and (3) the
boundaries between sound units are unknown, so even masked prediction has nothing
clean to predict. A solution must manufacture its own targets from audio,
robustly enough that a masked-prediction objective on them yields representations
that transfer to recognition with a thin head.

## Background

**Self-supervised pre-training, and why the targets are the crux.** The general
recipe — pretext task on unlabeled data, then fine-tune — is established across
NLP (masked-token prediction, BERT; Devlin et al. 2018) and vision (instance
discrimination, contrastive or not; He et al. 2020; Grill et al. 2020). The
pretext target in NLP is the masked token's identity, scored by cross-entropy
against the word-piece vocabulary. The whole question for speech is what plays
the role of that vocabulary when none is given.

**Acoustic unit discovery.** A long line of work shows that *unsupervised*
clustering of acoustic frames recovers structure correlated with phonetic units
even without labels. Simple discrete-latent models — k-means, Gaussian mixtures —
applied to frame features such as MFCCs yield cluster assignments that correlate
non-trivially with underlying phones (Lee & Glass 2012), and richer graphical or
neural latent-variable models do better. The clusters are noisy, but they are not
random: they carry phonetic signal. This is the raw material from which discrete
targets can be built.

**Masked prediction and the consistency insight.** Masking a span of the input
and predicting its content from the surviving context (as in BERT) forces a model
to learn both a representation of the visible audio and the long-range temporal
structure that lets it infer the hidden part. A key intuition for using *noisy*
discovered units as targets: what matters is not that each target label is
*correct*, but that the labeling is *consistent* — the same sound mapped to the
same unit — because consistency is what lets the model learn the sequential
structure even when the unit identities are imperfect.

**DeepCluster (Caron et al. 2018).** In vision, alternate between clustering the
current representations to produce pseudo-labels and training the network to
predict those pseudo-labels; the representation and the clustering bootstrap each
other across iterations. This is the template for *refining* targets over time:
once a network trained on crude targets gives better features, re-cluster those
features to get better targets, and repeat.

**Product quantization (Gray & Neuhoff 1998).** Partition a feature space into
subspaces and quantize each separately; the effective target space is the product
of the per-subspace codebooks. This lets several clusterings of different
granularity be combined cheaply.

**CTC (Graves et al. 2006).** The standard alignment-free recognizer head: sum
over all blank-augmented frame alignments that collapse to the target transcript.

## Baselines

**DiscreteBERT (Baevski et al. 2019).** Two stages: first learn a vector
quantization of audio (vq-wav2vec) so each frame is a discrete token, then run a
BERT masked-token model over those tokens and fine-tune for recognition. Core
idea: discretize first so the NLP masked-LM machinery applies. Gaps: the
Transformer ingests only the *quantized* tokens as input, discarding information
that the limited-capacity quantizer dropped; and the discretization is fixed
before the masked-prediction model is trained, so the teacher cannot improve.

**wav2vec 2.0 (Baevski et al. 2020).** A convolutional waveform encoder feeds a
Transformer; spans of the *continuous* latents are masked, and a contrastive loss
identifies the true quantized latent among same-utterance distractors, with a
jointly-learned Gumbel-softmax product quantizer and an auxiliary codebook
diversity loss. Core idea and math: InfoNCE,
−log[exp(sim(c_t,q_t)/κ)/Σ exp(sim(c_t,q̃)/κ)], with targets quantized but inputs
continuous. Gaps it leaves: the contrastive loss requires careful negative
sampling, a diversity loss to prevent codebook collapse, and a Gumbel-softmax
temperature-annealing schedule; the quantization is applied only to the
shallow convolutional encoder output, which may not be the best feature to
quantize; and target generation and representation learning are entangled in a
single end-to-end objective rather than separated.

**Pseudo-labeling / self-training (Kahn et al. 2020; Xu et al. 2020).** Train a
teacher on labels, label the unlabeled audio, retrain a student on the union,
iterate. Strong for semi-supervision, but it needs labels to begin, forces the
representation toward a single downstream task, and merely mimics a
supervised-data-limited teacher rather than compressing the full input signal.

## Evaluation settings

**LibriSpeech** (Panayotov et al. 2015): 960 hours of read English audiobooks,
standard dev/test clean and other splits, scored by **word error rate (WER)**.
**Libri-Light** (Kahn et al. 2020): a 60k-hour unlabeled set (LL-60k) plus
limited-label fine-tuning subsets of 10 minutes, 1 hour, 10 hours, and 100 hours,
for studying low-resource transfer. Audio is 16 kHz raw waveform. Optional
external language-model fusion at decoding. These datasets and the WER metric
predate the method and are the fixed yardsticks.

## Code framework

The pre-method toolkit: a convolutional waveform front end, a Transformer
(BERT-style) encoder, an offline clusterer (k-means / GMM) that can label frame
features, a projection-plus-codebook readout for classifying frames into units,
and a CTC head for fine-tuning. The empty slots are: what produces the discrete
targets, how the masked-prediction loss is shaped and where it is applied, and how
targets are improved over training.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Raw-waveform convolutional front end (known): strided temporal convolutions
# -> latent frames at ~20ms.
class ConvFeatureEncoder(nn.Module):
    def __init__(self, dims=(512,)*7,
                 kernels=(10,3,3,3,3,2,2), strides=(5,2,2,2,2,2,2)):
        super().__init__()
        layers, c_in = [], 1
        for c_out, k, s in zip(dims, kernels, strides):
            layers += [nn.Conv1d(c_in, c_out, k, s),
                       nn.GroupNorm(1, c_out), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
    def forward(self, wav):
        return self.conv(wav.unsqueeze(1)).transpose(1, 2)

# Transformer encoder (known); relative positions via a grouped conv.
class TransformerEncoder(nn.Module):
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

# Offline clusterer over frame features (known primitive: k-means).
def cluster_assignments(features, n_clusters):
    # TODO: which features get clustered, and how the resulting per-frame labels
    #       are used as prediction targets.
    pass

# Map an encoder output to a distribution over units (known: projection + codebook).
class UnitPredictor(nn.Module):
    def __init__(self, d=768, proj=256, n_units=100, tau=0.1):
        super().__init__()
        self.A = nn.Linear(d, proj)
        self.embed = nn.Embedding(n_units, proj)
        self.tau = tau
    def forward(self, o):
        # TODO: logits over units from cosine similarity between projected output
        #       and unit embeddings, scaled by 1/tau.
        pass

# Define the masked-prediction signal.
def mask_and_loss(encoder, x, targets):
    # TODO: how to mask, and where (masked frames / unmasked frames / both) the
    #       prediction loss is applied.
    pass

# Fine-tuning head (known): linear + CTC.
class CTCHead(nn.Module):
    def __init__(self, d=768, n_vocab=32):
        super().__init__()
        self.proj = nn.Linear(d, n_vocab)
    def forward(self, x):
        return self.proj(x).log_softmax(-1)
```
