# Context: Translating with Neural Networks and the Fixed-Vector Bottleneck

## Research question

How can a single neural network read a source sentence in one language and emit a fluent, faithful translation in another — trained end to end to maximize the probability of correct translations, with no separately-tuned sub-components?

The probabilistic target is clean: translation is finding the target sentence y that maximizes p(y | x) for a source sentence x, and a neural model fits this conditional on a parallel corpus, then searches for a high-probability y at test time. The standard way to make one network handle variable-length input and variable-length output is to have the network read the whole source into a single fixed-length vector and then write the translation out of that vector, where that vector carries the content words, modifiers, long-distance dependencies and word order of a sentence whose length is not bounded, and the same vector is reused at every step of generation. The question is how to structure such a model so it remains end-to-end and jointly trained.

## Background

Neural machine translation is a young approach (Kalchbrenner & Blunsom 2013; Sutskever, Vinyals & Le 2014; Cho, van Merriënboer, Gulcehre et al. 2014). Unlike the traditional phrase-based statistical pipeline (Koehn, Och & Marcu 2003; Koehn 2010), which is assembled from many separately-tuned components — a translation model, a language model, a reordering model, a word-alignment step — neural MT trains one large network to map source to target directly.

The recurrent neural network is the workhorse. An RNN reads a sequence by maintaining a hidden state and updating it one symbol at a time, h_t = f(x_t, h_{t-1}). Plain RNNs struggle to carry information across many steps because gradients vanish or explode as they are propagated back through time (Hochreiter 1991; Bengio, Simard & Frasconi 1994; Pascanu, Mikolov & Bengio 2013). Two gated cells address this. The long short-term memory unit (Hochreiter & Schmidhuber 1997) introduces a memory cell and gates so that gradient can flow across long spans with a product of derivatives near one. The gated recurrent unit (Cho et al. 2014) is a lighter cell with the same goal: an update gate z decides how much of the candidate state to write, so 1 − z keeps the previous state, a reset gate r decides how much of the previous state to mix into the candidate state, and the new state is a gated interpolation,
  s̃ = tanh(W e(y) + U[r ∘ s_{prev}] + C c), z = σ(W_z e(y) + U_z s_{prev} + C_z c), s = (1 − z) ∘ s_{prev} + z ∘ s̃.

A second RNN variant is the bidirectional RNN (Schuster & Paliwal 1997), used in speech recognition (Graves, Jaitly & Mohamed 2013). A forward RNN reads left to right producing states →h_1 … →h_T, a backward RNN reads right to left producing ←h_1 … ←h_T, and the two are combined per position. An ordinary RNN state at position t summarizes only x_1 … x_t, whereas combining both directions gives a per-position representation reflecting context on both sides of a word.

Word alignment is a central concept inherited from statistical MT. The IBM models (Brown et al.) and the phrase-based systems that followed treat the correspondence between source and target words as an explicit alignment — a latent variable saying which source position generated which target word — and learn it from data with the EM algorithm. Because some words have no counterpart, these models add a special NULL token that words can be aligned to or from. Alignment there is discrete and hard, and it is a separate estimation problem from the translation model. Handwriting synthesis offers a differentiable variant: Graves (2013) lets a generator read from a character sequence through a mixture of Gaussian kernels whose locations are predicted and constrained to advance monotonically — a "learning to align" that moves in one direction only.

An empirical finding characterizes existing encoder–decoder systems. When the source sentence is read into one fixed-length vector and decoded from it, translation quality is good on short sentences and falls off as the sentence gets longer; past roughly twenty to thirty words the quality degrades, and it degrades further on sentences longer than those seen in training (Cho et al. 2014, "On the properties of neural machine translation"; Pouget-Abadie et al. 2014).

## Baselines

**RNN Encoder–Decoder (Cho et al. 2014; Sutskever et al. 2014).** An encoder RNN reads the source x = (x_1, …, x_Tx) and reduces it to a single vector, c = q({h_1, …, h_Tx}); in the simplest case c is just the final hidden state, c = h_Tx (Sutskever et al. use a deep LSTM and take the last state). A decoder RNN then defines the translation distribution by factorizing it left to right, p(y) = ∏_{t=1}^{T} p(y_t | y_1, …, y_{t-1}, c), and models each factor as p(y_t | y_{<t}, c) = g(y_{t-1}, s_t, c), where s_t is the decoder state and g is a nonlinear, possibly multi-layer readout. The whole pair is trained jointly to maximize the log-probability of correct translations. The encoder packs the entire source into c, and that same c feeds every decoding step. Sutskever et al. reach close to phrase-based quality on English–French and find that reversing the source word order helps; Cho et al. introduce the GRU cell in this framework and also use a neural sub-model to score phrase pairs.

**Phrase-based statistical MT (Koehn et al. 2003; Moses; Koehn 2010).** Translation is decomposed into a phrase table, a reordering model and an n-gram language model, with explicit word/phrase alignment estimated separately and combined in a log-linear model tuned on a development set. It is strong, especially with large monolingual data for the language model, and handles long sentences without a fixed-length representation. It is a pipeline of separately-tuned pieces, and its alignment is a discrete, separately-estimated object.

**Location-based monotonic reading for synthesis (Graves 2013).** A differentiable alignment in which the read location moves forward only. The mechanism is monotonic, advancing in one direction over the input.

## Evaluation settings

The natural task is English-to-French translation on the WMT '14 bilingual parallel corpora (Europarl, news commentary, UN, and two crawled corpora, roughly 850M words total), reduced to a working set with the data-selection method of Axelrod, He & Gao (2011). Development is news-test-2012 and news-test-2013 concatenated; the held-out test set is news-test-2014 (3003 sentences). Tokenization follows the Moses scripts; models use a shortlist of the 30,000 most frequent words per language with the rest mapped to an [UNK] token; no lowercasing or stemming. Translation quality is measured with BLEU on the test set, reported both over all sentences and over the subset with no unknown words, and additionally examined as a function of source-sentence length to probe behavior on long inputs. At decoding time, a translation that approximately maximizes the conditional probability is found by beam search. Optimization is minibatch SGD (80 sentences per update) with Adadelta. A natural yardstick at the time is the phrase-based Moses system, run with its usual monolingual language-model data.

## Code framework

The primitives that already exist: word embeddings, a gated RNN cell, a softmax output, an autoregressive decode loop with teacher forcing, cross-entropy loss with padding ignored, and an optimizer with gradient clipping. The contribution slot is deliberately empty: the scaffold only says where a proposed sequence-to-sequence mechanism must plug in, not what that mechanism is.

```python
import torch
import torch.nn as nn

class GenericEncoder(nn.Module):
    """Turns source tokens into the representation required by a seq2seq model."""
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.core = None  # TODO: choose the encoder computation

    def forward(self, src):
        embedded = self.embedding(src)
        # TODO: return the encoded source and any state needed for decoding
        raise NotImplementedError


class GenericDecoder(nn.Module):
    """Generates the target sentence one token at a time."""
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.core = None          # TODO: choose the decoder recurrence
        self.output_layer = None  # TODO: map decoder features to vocabulary logits

    def initial_state(self, encoded_source):
        # TODO: initialize the decoder from the encoder result
        raise NotImplementedError

    def forward(self, input_token, state, encoded_source):
        embedded = self.embedding(input_token)
        # TODO: update decoder state and emit logits
        raise NotImplementedError


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio):
        encoded_source = self.encoder(src)
        state = self.decoder.initial_state(encoded_source)
        # autoregressive loop with teacher forcing, collecting per-step logits
        raise NotImplementedError


# training loop primitives that already exist
optimizer = None                                   # e.g. Adam / Adadelta
criterion = nn.CrossEntropyLoss(ignore_index=0)    # ignore padding
# for each batch: logits = model(src, trg, tfr); loss = criterion(...);
# loss.backward(); clip gradient norm; optimizer.step()
```
