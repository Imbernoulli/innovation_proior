# Context: encoder architectures for end-to-end speech recognition

## Research question

End-to-end ASR maps an audio feature sequence to a label sequence with a single
neural encoder (plus a small decoder). The encoder must model two very different
kinds of dependency at once: **global** structure — long-range relations across an
utterance, the kind that resolve a phoneme from distant context — and **local**
structure — the fine-grained, position-local feature patterns (formant
transitions, onsets) that distinguish nearby sounds. The precise question is how
to build an encoder that captures *both* well and is *parameter-efficient* about
it, because the two leading building blocks each handle only one of the two and
pay for the other in depth or parameters. A good answer would combine them so that
each does what it is best at, without simply stacking enough of either to brute-force
the missing capability.

## Background

**Why local and global are different problems.** A self-attention layer relates
every position to every other in one step, with weights computed from content; it
is excellent at long-range, content-based interactions but, being a global
weighted average, it is comparatively poor at extracting sharp, position-local
feature patterns. A convolution does the opposite: a kernel slides over a local
window, so it captures edges, onsets, and short patterns cheaply and with
translation equivariance, but its receptive field grows only one window per layer,
so reaching global context needs many layers or many parameters. This is the
core trade-off the field had hit: attention is globally strong / locally weak,
convolution is locally strong / globally weak.

**Self-attention and relative position.** The Transformer (Vaswani et al. 2017)
computes softmax(QKᵀ/√d_k)V per layer, multi-headed, with a position-wise
feed-forward network of inner width 4× the model dimension, wrapped in residual
connections and layer normalization. Self-attention is order-agnostic, so position
must be injected. Absolute sinusoidal or learned position encodings tie the
representation to absolute index; **relative positional encoding** (Transformer-XL;
Dai et al. 2019) instead makes attention depend on the *offset* between positions,
which generalizes better across the highly variable utterance lengths of speech
and makes the encoder robust to length.

**Convolutional building blocks.** Depthwise-separable convolution factorizes a
conv into a per-channel spatial (depthwise) convolution plus a 1×1 (pointwise)
channel mixing, drastically cutting parameters. The gated linear unit (GLU;
Dauphin et al. 2017) splits a projection in half and gates one half by a sigmoid
of the other, x = a ⊙ σ(b), giving the conv a learned multiplicative gate.
Batch normalization (Ioffe & Szegedy 2015) stabilizes training of deep conv
stacks. Swish, x·σ(βx) (Ramachandran et al. 2017), is a smooth activation that
tends to outperform ReLU in deep networks.

**Pre-norm residual units and the Macaron structure.** Placing layer
normalization *inside* the residual branch, before the sublayer (pre-norm; Wang et
al. 2019; Nguyen & Salazar 2019), eases optimization of deep stacks. Separately,
Macaron-Net (Lu et al. 2019) reinterprets a Transformer block through an
ordinary-differential-equation lens and argues the single post-attention
feed-forward layer is better replaced by *two half-step* feed-forward layers, one
before and one after the attention — sandwiching it — each contributing a
half-weighted residual.

**Combining attention and convolution.** Recent work shows the two are
complementary: augmenting self-attention with convolutional or relative-offset
information helps (Bello et al. 2019; Yang et al. 2019; Yu et al. 2018). A
multi-branch design (Wu et al. 2020) splits the input into a self-attention branch
and a convolution branch in parallel and concatenates their outputs, improving
machine translation. These establish that *content-based global* and
*position-based local* computation should coexist in a block; the open design
question is *how* to arrange them.

## Baselines

**RNN / LSTM encoders (Chiu et al. 2018; Graves 2012).** Recurrent encoders were
the de-facto choice; an LSTM threads a hidden state along time, modeling temporal
dependencies. Core idea: sequential state updates. Gaps: sequential computation
limits training efficiency, and very long-range dependencies are hard to carry
through the recurrence.

**Transformer ASR (Zhang et al. 2020; Karita et al. 2019).** Replace recurrence
with self-attention; parallel over time and strong at long-range context. The
Transformer Transducer (Zhang et al. 2020) is the strong prior published result on
LibriSpeech. Gap: self-attention is weaker at fine-grained local feature
extraction, so a pure-attention encoder leaves accuracy on the table where local
acoustic detail matters.

**Convolutional ASR — Jasper / QuartzNet / ContextNet (Li et al. 2019; Kriman et
al. 2019; Han et al. 2020).** Deep 1-D convolutional encoders capture local
context layer by layer. ContextNet specifically tries to inject global context by
adding a squeeze-and-excitation module to each block, which applies a *global
average* over the whole sequence and rescales channels. Core idea: progressive
local receptive fields plus a coarse global summary. Gap: squeeze-and-excitation
gives only a single averaged global vector — it cannot model *dynamic*,
position-dependent global interactions the way attention can.

## Evaluation settings

**LibriSpeech** (Panayotov et al. 2015): 970 hours of labeled read English speech
plus an 800M-word text-only corpus for language modeling; standard dev/test clean
and other splits, scored by **word error rate (WER)**, optionally with an external
LM via shallow fusion. Inputs are 80-channel log-mel filterbank features from a
25 ms window with a 10 ms stride; **SpecAugment** (Park et al. 2019) time/frequency
masking is the standard data augmentation. Encoders are typically trained within a
transducer or attention/CTC end-to-end framework and compared at matched parameter
budgets (e.g. ~10M, ~30M, ~118M). These datasets, features, and the WER metric
predate the method.

## Code framework

The pre-method toolkit: a convolutional subsampling front end over filterbank
features, the standard primitives (multi-head self-attention, position-wise
feed-forward, depthwise/pointwise convolution, GLU, BatchNorm, Swish, LayerNorm,
pre-norm residual wrappers), an Adam optimizer with the Transformer warmup
schedule, and a transducer/CTC training loop. The *block* — how attention,
convolution, and feed-forward are arranged into one encoder unit — is the empty
slot.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# Convolutional subsampling over filterbank features (known): two stride-2 convs
# reduce the time resolution before the encoder blocks.
class ConvSubsampling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c1 = nn.Conv2d(1, d, 3, stride=2)
        self.c2 = nn.Conv2d(d, d, 3, stride=2)
    def forward(self, x):                       # x: (B, T, n_mels)
        x = F.relu(self.c1(x.unsqueeze(1)))
        x = F.relu(self.c2(x))
        b, c, t, f = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, c * f)

# Multi-head self-attention with RELATIVE positional encoding (known primitive).
class RelMultiHeadSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        # relative positional encoding wired in here
    def forward(self, x):
        a, _ = self.mha(x, x, x)
        return a

# Position-wise feed-forward (known): two linears with a Swish in between.
class FeedForward(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, expansion * d), nn.SiLU(),
                                 nn.Dropout(0.1), nn.Linear(expansion * d, d),
                                 nn.Dropout(0.1))
    def forward(self, x):
        return self.net(x)

# The encoder block: how to arrange attention + convolution + feed-forward.
class EncoderBlock(nn.Module):
    def __init__(self, d, heads, kernel):
        super().__init__()
        # TODO: assemble the sublayers (attention, a convolution sublayer,
        #       feed-forward) and their residual/normalization structure.
        pass
    def forward(self, x):
        # TODO: the block's forward computation.
        pass

class Encoder(nn.Module):
    def __init__(self, d, layers, heads, kernel):
        super().__init__()
        self.subsample = ConvSubsampling(d)
        self.proj = nn.Linear(d, d)               # placeholder for subsampled-dim -> d
        self.blocks = nn.ModuleList(EncoderBlock(d, heads, kernel) for _ in range(layers))
    def forward(self, x):
        x = self.proj(self.subsample(x))
        for blk in self.blocks:
            x = blk(x)
        return x
```
