# Context: the ground a sequence-transduction model stands on

## Research question

The concrete problem is sequence transduction: map a variable-length source sequence of symbols `(x_1, …, x_n)` to a variable-length target sequence `(y_1, …, y_m)` — the canonical instance being machine translation, where the source is a sentence in one language and the target its translation. A solution is *conditional* and autoregressive — `p(y) = ∏_t p(y_t | y_{<t}, x)` — and must route information between positions, because translating one output word can depend on words arbitrarily far back in the source and in the already-generated output (subject–verb agreement, long-range reordering, coreference/anaphora).

By the established recipe this is done with a recurrent encoder-decoder. A recurrent net produces hidden states `h_t = f(h_{t-1}, x_t)`; step `t` begins after step `t-1` is done, so within a single training example the computation runs sequentially along the time axis. The accelerators of the day (GPUs, TPUs) are throughput machines that reward large parallel matrix multiplies.

The question is how to design the sequence model at the heart of this transduction system: what architecture maps the source representations and the partial target to the next-token features. A standard yardstick for comparing layer types tracks three quantities — total computation per layer, the minimum number of *sequential* operations, and the **maximum path length** a signal must traverse to connect two positions, since shorter forward/backward paths make long-range dependencies easier to learn (Hochreiter et al. 2001).

## Background

The field state going into this is dominated by recurrent encoder-decoders. Recurrent nets — and in particular gated cells, LSTM (Hochreiter & Schmidhuber 1997) and GRU (Cho et al. 2014) — are the established state of the art for language modeling and translation. The encoder-decoder framing itself (Sutskever et al. 2014; Cho et al. 2014) is the dominant structure: an encoder compresses the source into representations, a decoder generates the target conditioned on them. A large body of work (Wu et al. 2016; Luong et al. 2015; Jozefowicz et al. 2016) has pushed this line hard, and efficiency work — factorization tricks (Kuchaiev & Ginsburg 2017) and conditional computation / mixture-of-experts (Shazeer et al. 2017) — has shaved the constants.

A load-bearing concept underneath the recent gains is **attention** (Bahdanau et al. 2014; Luong et al. 2015): a mechanism that lets the decoder, at each step, form a content-based weighted average over *all* encoder positions, so dependencies can be modeled "without regard to their distance in the input or output sequences" — any output position can reach any input position in one hop. In nearly every system it is used alongside a recurrent net: the things being attended over, and the state doing the attending, are produced by the RNN. One known exception (Parikh et al. 2016, decomposable attention for natural-language inference) used attention without recurrence, on a sentence-pair classification task.

Two further threads are in the air. First, **self-attention** (intra-attention): relating different positions *of a single sequence* to compute a representation of that sequence, used for reading comprehension, summarization, entailment, and sentence embeddings (Cheng et al. 2016; Parikh et al. 2016; Paulus et al. 2017; Lin et al. 2017). Second, **end-to-end memory networks** (Sukhbaatar et al. 2015), which use a recurrent attention mechanism over a memory rather than sequence-aligned recurrence, demonstrated on question answering. Both show attention can do representational work on its own.

The supporting machinery that lets deep stacks train is also part of the background: residual connections (He et al. 2016) and layer normalization (Ba et al. 2016) for gradient flow; Adam (Kingma & Ba 2014) for optimization; dropout (Srivastava et al. 2014) and label smoothing (Szegedy et al. 2015) for regularization; subword vocabularies (Sennrich et al. 2015; Wu et al. 2016) for the input representation; and weight tying of input/output embeddings (Press & Wolf 2016). On the three-axis yardstick, a recurrent layer is `O(n·d²)` compute, `O(n)` sequential operations, and `O(n)` maximum path length.

## Baselines

**Seq2Seq with LSTMs (Sutskever et al. 2014).** Two multilayer LSTMs: the encoder reads the source and dumps everything into a single fixed-length vector (its final hidden state); the decoder LSTM generates the target from that vector. It established that a pure neural net can do competitive translation. They reversed the source word order to shorten the effective path between aligned words.

**Additive attention for NMT (Bahdanau et al. 2014).** Instead of one fixed vector, the decoder computes a fresh context at every step as a weighted sum over all encoder annotations. The alignment score between decoder state `s_{i-1}` and encoder annotation `h_j` is computed by a small one-hidden-layer feed-forward net, `e_ij = vᵀ tanh(W_s s_{i-1} + W_h h_j)`, then `α_ij = softmax_j(e_ij)`, and `c_i = Σ_j α_ij h_j`. The annotations `h_j` and the state `s_{i-1}` are produced by recurrent nets.

**Effective / multiplicative attention (Luong et al. 2015).** Introduced the global-vs-local distinction and the **dot-product (multiplicative) score** `q·k` (and `qᵀWk`) as an alternative to additive scoring. Same theoretical complexity as additive, and it maps onto a single dense matrix multiply. Britz et al. 2017 observed that unscaled dot products behave differently from additive scoring as the key dimension grows.

**ByteNet (Kalchbrenner et al. 2017).** Replaces recurrence with stacked *dilated convolutions*, decoder stacked over encoder, computing all positions in parallel. Dilation makes the number of layers needed to connect two positions grow logarithmically in their distance (`O(log_k n)` path length).

**ConvS2S (Gehring et al. 2017).** Fully convolutional encoder-decoder with gated linear units, *learned* positional embeddings, and separate attention per decoder layer, parallel across positions at training time. With kernel width `k`, connecting two positions distance `D` apart uses a stack of `O(D/k)` conv layers. It is the source of the learned-vs-fixed positional-encoding design choice.

## Evaluation settings

The yardstick for translation quality is the **WMT 2014 shared translation task**, in two language pairs that are the standard reporting points: English→German (the standard newstest2014 test set; the training corpus is roughly 4.5M sentence pairs) and the much larger English→French (≈36M sentence pairs). The accepted automatic metric is **BLEU**, computed against reference translations on the held-out test set (newstest2014); higher is better. The protocol also tracks training cost — wall-clock time and accelerator-hours to reach a given quality. Inputs are not raw words but **subword units**: byte-pair-encoding (Sennrich et al. 2015) with a shared source-target vocabulary of ~37k tokens for EN-DE, and word-piece (Wu et al. 2016) with a ~32k vocabulary for EN-FR. The subword choice keeps the sequence length `n` (a few dozen to low hundreds of subword tokens) below the representation width `d`.

## Code framework

A generic supervised-sequence-transduction harness fixes the parts that are not in question — how tokens become vectors, how vectors become a loss, which optimizer turns the loss into updates, how a variable-length batch is assembled and masked — and leaves exactly one empty slot: the function that maps the source representations and the partial target to the next-token features. Everything below is standard PyTorch; the single `# TODO` is the slot the work will fill.

**Token embedding.** A lookup table turns integer token ids into dense vectors.

```python
import torch, torch.nn as nn
from torch.nn.functional import log_softmax

class TokenEmbedding(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
    def forward(self, x):                 # x: (batch, seq) ids -> (batch, seq, d_model)
        return self.lut(x)
```

**The one open slot — the sequence model (TODO).** Some module has to consume the embedded source (with its padding mask) and the embedded partial target (with its mask) and emit, per target position, a `d_model` feature vector to be scored against the vocabulary. *What that module is* — that is the whole question; nothing about its internals is decided yet, so it is a single empty stub.

```python
class SequenceModel(nn.Module):
    """Map (embedded source, embedded target-so-far) -> per-target-position features.
    The architecture that goes in here is exactly what we have to design."""
    def __init__(self, d_model):
        super().__init__()
        # TODO: the architecture we'll design.
        pass
    def forward(self, src, src_mask, tgt, tgt_mask):
        # src: (batch, src_len, d_model)   tgt: (batch, tgt_len, d_model)
        # returns: (batch, tgt_len, d_model)
        raise NotImplementedError  # TODO
```

**Output projection + log-softmax.** A linear from `d_model` to vocabulary size, then a log-softmax, turns per-position features into next-token log-probabilities.

```python
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
```

**Loss.** Per-position cross-entropy (negative log-likelihood on the log-probabilities) of the gold next token, summed over the non-padding positions.

```python
def seq_loss(log_probs, gold, pad_idx):
    # log_probs: (batch, tgt_len, vocab); gold: (batch, tgt_len)
    loss = nn.functional.nll_loss(
        log_probs.reshape(-1, log_probs.size(-1)),
        gold.reshape(-1), ignore_index=pad_idx, reduction="sum")
    return loss
```

**Optimizer.** Adam over all parameters.

```python
def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)  # schedule TBD
```

**Data path: batching + padding masks.** Tokenized text is mapped to ids through a subword vocabulary, padded to a common length, and packed into a batch that also builds the masks. The mask known to be needed here is the **padding** mask (so the model and the loss ignore pad positions) on both source and target; whatever extra masking the target side may require is not yet decided and is left to the model.

```python
def make_batch(src_ids, tgt_ids, pad_idx):
    src = pad_sequences(src_ids, pad_idx)          # (batch, src_len)
    tgt = pad_sequences(tgt_ids, pad_idx)          # (batch, tgt_len)
    src_mask = (src != pad_idx).unsqueeze(-2)      # source padding mask
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)      # target padding mask (only)
    return src, tgt, src_mask, tgt_mask

def train_step(model, generator, batch, optimizer, pad_idx):
    src, tgt, src_mask, tgt_mask = batch
    feats = model(embed_src(src), src_mask, embed_tgt(tgt), tgt_mask)
    log_probs = generator(feats)
    loss = seq_loss(log_probs, gold_next_token(tgt), pad_idx)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    return loss
```

The harness is `embed → SequenceModel(# TODO) → generator → cross-entropy`, trained with Adam over padded, mask-carrying batches. Designing what fills the `SequenceModel` slot — and the few places the surrounding harness then needs adjusting (the loss, the optimizer's learning-rate schedule, the target mask) — is the work that follows.
