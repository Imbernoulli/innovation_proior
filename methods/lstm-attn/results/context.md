# Context: neural machine translation with encoder-decoder RNNs (circa 2014)

## Research question

We want a single neural network that reads a source sentence and emits its translation,
trained end to end to maximize the probability of the correct target, replacing the
many separately tuned sub-components of a phrase-based statistical system. The recently
established way to do this maps the whole variable-length source into one fixed-length vector
and then generates the target from that vector alone. The question is what architectural
choices govern translation quality — in particular across sentences of different lengths —
within this single, jointly trainable encoder-decoder framework.

## Background

Translation is cast probabilistically as finding the target sentence `y` that maximizes
`p(y | x)` for source `x`; a parameterized model is fit on a parallel corpus to maximize the
conditional probability of the observed sentence pairs, and at test time one searches for the
highest-probability target. The dominant neural realization is the *encoder-decoder*: an
encoder RNN consumes the source token by token, `h_t = f(x_t, h_{t-1})`, and a summary `c` is
read off the hidden states; a decoder RNN factorizes the target autoregressively,
`p(y) = Π_t p(y_t | y_1..y_{t-1}, c)`, with each conditional `p(y_t | y_<t, c) = g(y_{t-1},
s_t, c)` and `s_t` the decoder hidden state. The whole pair is trained jointly by maximizing
`(1/N) Σ_n log p_θ(y_n | x_n)`.

Two pieces of recurrent-network machinery make this trainable. First, gated recurrent units
keep gradients alive over long sequences. The vanilla RNN suffers vanishing gradients
(Hochreiter 1991; Bengio, Simard & Frasconi 1994; Pascanu, Mikolov & Bengio 2013): the
product of derivatives along the unrolled net shrinks toward zero, so distant inputs cannot
influence the loss. The long short-term memory unit (Hochreiter & Schmidhuber 1997) fixes this
with a gated memory cell whose temporal path has derivative product near one. A lighter gated
unit (Cho et al. 2014a) achieves the same with two gates — a reset gate `r` and an update gate
`z` — and a candidate state: `r = σ(W_r x + U_r h_{t-1})`, `z = σ(W_z x + U_z h_{t-1})`,
`h̃ = φ(W x + U(r ⊙ h_{t-1}))`, and `h_t = (1 − z) ⊙ h_{t-1} + z ⊙ h̃`. The update gate lets a
unit copy its previous activation unchanged (carrying long-term information), the reset gate
lets it drop the past and read mostly the current input.

Second, the decoder is a conditional RNN language model: `p(y_t | v, y_<t)` is a softmax over
the vocabulary, the sequence ends with a special end-of-sentence symbol so the model spans all
lengths, and at decode time one runs a left-to-right beam search keeping the `B` best partial
hypotheses (Graves 2012; Boulanger-Lewandowski et al. 2013).

An empirical observation about *length*: a basic encoder-decoder's BLEU falls as the source
sentence grows, with the drop most pronounced for sentences longer than those in the training
corpus (Cho et al. 2014b; Pouget-Abadie et al. 2014). A second observation comes from
sequence-to-sequence training: reversing the order of the source words (so the first source
words sit near the first target words) markedly improved results, dropping test perplexity from
5.8 to 4.7 and raising BLEU from 25.9 to 30.6 (Sutskever, Vinyals & Le 2014).

A related strand is differentiable, soft selection over an input sequence. In handwriting
synthesis, Graves (2013) let a generator consult a character string through a soft window
`w_t = Σ_{u=1}^U φ(t,u) c_u`, with `φ(t,u) = Σ_{k=1}^K α_t^k exp(−β_t^k (κ_t^k − u)²)` a
mixture of `K` Gaussians whose location `κ_t = κ_{t-1} + exp(κ̂_t)` advances by a positive,
learned offset at each step. The window is differentiable and trainable jointly with the
generator. The window is placed by *location* (a coordinate `κ` slid along the input) rather
than by matching content, and `κ` only ever increases, so the alignment is strictly monotonic
and one-directional.

## Baselines

These are the prior systems a new translation model would be measured against.

**Phrase-based statistical machine translation (Koehn, Och & Marcu 2003; Koehn 2010, "Moses").**
A log-linear model `log p(f | e) = Σ_n w_n f_n(f, e) + log Z(e)` combining a translation model
(phrase-pair probabilities) and a target language model, with weights tuned to maximize BLEU
on a development set. Strong and mature, and able to exploit large monolingual data for its
language model.

**RNN encoder-decoder for SMT rescoring (Cho et al. 2014a).** The encoder-decoder with gated
units, used as a feature: it scores phrase pairs that re-rank or augment an existing
phrase-based system, `log p(f|e) = g(h_t, y_{t-1}, c)` entering the log-linear model as one
more term.

**Sequence-to-sequence with deep LSTMs (Sutskever, Vinyals & Le 2014).** A standalone neural
translator: an LSTM encoder reads the source and its *final hidden state* `v = h_{T_x}` is the
fixed representation; a deep LSTM decoder is a language model whose initial state is `v`,
`p(y_1..y_{T_y} | x) = Π_t p(y_t | v, y_<t)`, trained by maximizing `(1/|S|) Σ log p(T|S)` and
decoded by beam search. It reached strong BLEU on English-to-French, and with source reversal
reported strong performance on longer sentences.

**Standalone RNN encoder-decoder translation (Cho et al. 2014a, used directly).** The same
gated encoder-decoder trained and used on its own as a translator (the natural same-family
comparison for any successor).

**Soft-window sequence generation (Graves 2013), as an alignment mechanism.** The Gaussian-mixture
soft window above provides a differentiable way to let a generator focus on parts of an
input sequence while training end to end.

## Evaluation settings

The natural yardstick for English-to-French translation:

- **Corpus:** the WMT '14 English-French parallel data (Europarl, news-commentary, UN, two
  crawled corpora; ~850M words), reduced to ~348M words by in-domain data selection (Axelrod
  et al. 2011); no monolingual data beyond the parallel corpora. Validation = news-test-2012 +
  news-test-2013; test = news-test-2014 (3003 sentences).
- **Preprocessing:** Moses tokenization; a shortlist of the 30,000 most frequent words per
  language, all others mapped to `[UNK]`; no lowercasing or stemming.
- **Length protocol:** train models twice — once on sentence pairs up to length 30, once up to
  length 50 — so that behavior can be read as a function of sentence length, and BLEU plotted
  against source length to expose length-dependent degradation.
- **Metric:** BLEU on the test set, reported both on all sentences and on the subset with no
  unknown words; conditional negative log-likelihood on train and development sets as the
  optimization-side measure.
- **Optimization / decoding protocol:** minibatch SGD (minibatch of 80 sentence pairs) with an
  adaptive per-parameter learning rate (Adadelta, ρ=0.95, ε=1e-6), gradient-norm clipping at a
  threshold of 1 (Pascanu et al. 2013); minibatches grouped by similar length to avoid wasted
  padding computation; beam search at decode.

## Code framework

The model plugs into the standard translation harness already in use: token embeddings for the
source and target vocabularies, a trainable sequence-to-sequence core, a vocabulary projection,
and a masked cross-entropy objective summed over target positions. What is *not* settled is the
core architecture: how the source sequence and the generated prefix should be represented before
the next-token softmax. The substrate below leaves that core as one neutral empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCore(nn.Module):
    """The source-to-target architecture still to be designed."""

    def __init__(self, emb_dim, hid_dim):
        super().__init__()
        # TODO: the architecture we will design.

    def forward(self, src_emb, tgt_emb, src_mask):
        # TODO: turn the source sequence and target prefix into target-side hidden states.
        pass


class TranslationModel(nn.Module):
    """Trainable conditional model p(target | source)."""

    def __init__(self, src_vocab, tgt_vocab, emb_dim, hid_dim, pad_id):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, emb_dim, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab, emb_dim, padding_idx=pad_id)
        self.core = SequenceCore(emb_dim, hid_dim)
        self.out = nn.Linear(hid_dim, tgt_vocab)
        self.pad_id = pad_id

    def forward(self, src, tgt_in):
        src_emb = self.src_embed(src)                 # [B, T_x, emb_dim]
        tgt_emb = self.tgt_embed(tgt_in)              # [B, T_y, emb_dim]
        hidden = self.core(src_emb, tgt_emb, src.ne(self.pad_id))
        return self.out(hidden)                       # [B, T_y, tgt_vocab]


def loss_fn(logits, target, pad_id):
    # masked cross-entropy over target positions
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target.reshape(-1), ignore_index=pad_id)
```

The embeddings, output softmax, and masked-cross-entropy loop all exist already; the single
empty slot is the sequence-to-sequence core.
