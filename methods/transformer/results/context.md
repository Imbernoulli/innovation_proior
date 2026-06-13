# Context: the ground a sequence-transduction model stands on

## Research question

The concrete problem is sequence transduction: map a variable-length source sequence of symbols `(x_1, …, x_n)` to a variable-length target sequence `(y_1, …, y_m)` — the canonical instance being machine translation, where the source is a sentence in one language and the target its translation. A solution must (a) be *conditional* and autoregressive — `p(y) = ∏_t p(y_t | y_{<t}, x)` — and (b) route information freely between any two positions, because translating one output word can depend on words arbitrarily far back in the source and in the already-generated output (subject–verb agreement, long-range reordering, coreference/anaphora).

By the established recipe this is done with a recurrent encoder-decoder, and that recipe carries one structural cost that has become the binding constraint: **its computation is inherently sequential in the sequence length.** A recurrent net produces hidden states `h_t = f(h_{t-1}, x_t)`; step `t` cannot begin until step `t-1` is done. Within a single training example there is therefore no parallelism along the time axis — and at long sequence lengths, where memory limits how many examples can be batched together, that lost intra-example parallelism is exactly what cannot be recovered by batching. The accelerators of the day (GPUs, TPUs) are throughput machines that reward large parallel matrix multiplies and are starved by long chains of small sequential ops.

So the goal a solution must meet is sharp: **achieve the unrestricted any-position-to-any-position information routing that attention already provides, while making the per-layer computation parallel across all positions** — turning the `O(n)` chain of sequential operations into `O(1)`, so that training throughput stops being throttled by sequence length, without sacrificing translation quality. The secondary criterion that decides between candidate solutions is the *path length* a signal must traverse to connect two positions: shorter forward/backward paths make long-range dependencies easier to learn.

## Background

The field state going into this is dominated by recurrent encoder-decoders. Recurrent nets — and in particular gated cells, LSTM (Hochreiter & Schmidhuber 1997) and GRU (Cho et al. 2014) — are the established state of the art for language modeling and translation. The encoder-decoder framing itself (Sutskever et al. 2014; Cho et al. 2014) is the dominant structure: an encoder compresses the source into representations, a decoder generates the target conditioned on them. A large body of work (Wu et al. 2016; Luong et al. 2015; Jozefowicz et al. 2016) has pushed this line hard, and efficiency work — factorization tricks (Kuchaiev & Ginsburg 2017) and conditional computation / mixture-of-experts (Shazeer et al. 2017) — has shaved the constants, but the `O(n)` sequential depth remains.

The load-bearing concept underneath the recent gains is **attention** (Bahdanau et al. 2014; Luong et al. 2015): a mechanism that lets the decoder, at each step, form a content-based weighted average over *all* encoder positions, so dependencies can be modeled "without regard to their distance in the input or output sequences." Attention dissolved the long-range-dependency problem at the routing level — any output position can reach any input position in one hop. But in nearly every system it is *bolted onto a recurrent net*: the things being attended over, and the state doing the attending, are still produced sequentially. The one well-known exception (Parikh et al. 2016, decomposable attention for natural-language inference) used attention without recurrence, but on a sentence-pair classification task, not autoregressive transduction.

Two further threads are in the air. First, **self-attention** (intra-attention): relating different positions *of a single sequence* to compute a representation of that sequence, used for reading comprehension, summarization, entailment, and sentence embeddings (Cheng et al. 2016; Parikh et al. 2016; Paulus et al. 2017; Lin et al. 2017). Second, **end-to-end memory networks** (Sukhbaatar et al. 2015), which replace sequence-aligned recurrence with a recurrent attention mechanism over a memory. Both show attention can do real representational work on its own — but each is a narrow demonstration: self-attention has so far been used only to compute representations for classification, comprehension, or embedding tasks, and memory networks for question answering, never as the generative engine of an autoregressive transduction system.

The conceptual yardstick for comparing layer types has three axes: total computation per layer; the minimum number of *sequential* operations (the parallelism); and the **maximum path length** that a signal must traverse to connect two positions, since shorter forward/backward paths make long-range dependencies easier to learn (Hochreiter et al. 2001). A recurrent layer is `O(n·d²)` compute, `O(n)` sequential, `O(n)` path length — bad on the last two axes. The supporting machinery that lets deep stacks train at all is also part of the background: residual connections (He et al. 2016) and layer normalization (Ba et al. 2016) for gradient flow; Adam (Kingma & Ba 2014) for optimization; dropout (Srivastava et al. 2014) and label smoothing (Szegedy et al. 2015) for regularization; subword vocabularies (Sennrich et al. 2015; Wu et al. 2016) for the input representation; and weight tying of input/output embeddings (Press & Wolf 2016).

## Baselines

**Seq2Seq with LSTMs (Sutskever et al. 2014).** Two multilayer LSTMs: the encoder reads the source and dumps everything into a single fixed-length vector (its final hidden state); the decoder LSTM generates the target from that vector. It established that a pure neural net can do competitive translation. Two gaps: (1) the fixed-length context vector is an information bottleneck — the entire source is squeezed into one vector, and quality degrades on long sentences (they even reversed the source word order as a hack to shorten the effective path between aligned words); (2) it is fully sequential in both encoder and decoder.

**Additive attention for NMT (Bahdanau et al. 2014).** Removes the bottleneck: instead of one fixed vector, the decoder computes a fresh context at every step as a weighted sum over all encoder annotations. The alignment score between decoder state `s_{i-1}` and encoder annotation `h_j` is computed by a small one-hidden-layer feed-forward net, `e_ij = vᵀ tanh(W_s s_{i-1} + W_h h_j)`, then `α_ij = softmax_j(e_ij)`, and `c_i = Σ_j α_ij h_j`. This is the move that made long-range routing cheap and is the direct ancestor of everything here. Gap: the annotations `h_j` and the state `s_{i-1}` are still produced by recurrent nets, so the sequential bottleneck remains; and additive scoring is a small MLP evaluated per query-key pair rather than a single fused matrix multiply.

**Effective / multiplicative attention (Luong et al. 2015).** Introduced the global-vs-local distinction and, crucially, the **dot-product (multiplicative) score** `q·k` (and `qᵀWk`) as a cheaper alternative to additive scoring. Same theoretical complexity as additive, but it maps onto a single dense matrix multiply, so it is much faster and more memory-efficient on real hardware. Gap: still RNN-based; and unscaled dot products were observed (Britz et al. 2017) to behave worse than additive scoring as the key dimension grows.

**ByteNet (Kalchbrenner et al. 2017).** Abandons recurrence for stacked *dilated convolutions*, decoder stacked over encoder, computing all positions in parallel. Dilation makes the number of layers needed to connect two positions grow only *logarithmically* in their distance. Gap: path length still grows with distance (`O(log_k n)`), so very long-range dependencies are still harder to learn than a single hop.

**ConvS2S (Gehring et al. 2017).** Fully convolutional encoder-decoder with gated linear units, *learned* positional embeddings, and separate attention per decoder layer. Fully parallel across positions at training time — the strongest "kill recurrence" baseline. Gap: with kernel width `k`, connecting two positions distance `D` apart needs a stack of `O(D/k)` conv layers, so the number of operations to relate two positions grows **linearly** with distance — the long-range-routing tax that attention is supposed to eliminate. It is also the source of the learned-vs-fixed positional-encoding design choice.

Across all of these, the recurring gap is the same: either routing between distant positions costs operations that grow with distance (Seq2Seq's bottleneck, ConvS2S linear, ByteNet log), or the per-layer computation is sequential in `n` (every RNN baseline) — never both `O(1)` path length *and* `O(1)` sequential operations at once.

## Evaluation settings

The natural yardstick for translation quality is the **WMT 2014 shared translation task**, in two language pairs that are the standard reporting points: English→German (the standard newstest2014 test set; the training corpus is roughly 4.5M sentence pairs) and the much larger English→French (≈36M sentence pairs). The accepted automatic metric is **BLEU**, computed against reference translations on the held-out test set (newstest2014); higher is better, and it is the number every translation system reports. Beyond quality, the protocol of the day also tracks training cost — wall-clock time and accelerator-hours to reach a given quality — because the whole motivation is throughput. Inputs are not raw words but **subword units**: byte-pair-encoding (Sennrich et al. 2015) for EN-DE with a shared source-target vocabulary of ~37k tokens, and word-piece (Wu et al. 2016) with a ~32k vocabulary for EN-FR. The subword choice matters for the analysis because it keeps the sequence length `n` (a few dozen to low hundreds of subword tokens) below the representation width `d`.

## Code framework

A generic supervised-sequence-transduction harness fixes the parts that are not in question — how tokens become vectors, how vectors become a loss, which optimizer turns the loss into updates, how a variable-length batch is assembled and masked — and leaves exactly one empty slot: the function that maps the source representations and the partial target to the next-token features. Everything below is standard PyTorch; the single `# TODO` is the slot the work will fill.

**Token embedding.** A lookup table turns integer token ids into dense vectors. Nothing here is in question.

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

**Data path: batching + padding masks.** Tokenized text is mapped to ids through a subword vocabulary, padded to a common length, and packed into a batch that also builds the masks. The only mask known to be needed here is the **padding** mask (so the model and the loss ignore pad positions) on both source and target; whatever extra masking the target side may require is not yet known and is left to the model.

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
