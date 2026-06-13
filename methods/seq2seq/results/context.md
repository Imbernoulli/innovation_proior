# Context: mapping variable-length sequences to variable-length sequences with neural networks

## Research question

How can a single, domain-independent neural network learn to map an input sequence to an output sequence when the two have different, a-priori-unknown lengths and a non-monotonic relationship between positions?

Deep feedforward networks have become the dominant tool for hard perception problems, but they share one structural limitation: they consume and emit fixed-dimensional vectors. The problems that matter most in language are not shaped that way. Machine translation maps a source sentence of `T` words to a target sentence of `T'` words, with `T' != T` in general and with word order that can shuffle across the languages. Speech recognition maps a variable-length acoustic stream to a variable-length transcript. Question answering maps a question sequence to an answer sequence. In each case the length of the output is not known until it is produced, and the alignment between input and output positions is neither one-to-one nor monotonic.

A satisfactory solution would have to accept inputs of arbitrary length, produce outputs of arbitrary self-determined length, make minimal assumptions about the structure of the alignment, and be trainable end-to-end by backpropagation from paired sequences alone. The standard toolbox supplies pieces of this, but not the whole mapping.

## Background

**Recurrent neural networks.** An RNN (Werbos 1990; Rumelhart, Hinton & Williams 1986) extends a feedforward net to sequences by carrying a hidden state across time. Given inputs `(x_1,...,x_T)` it computes
```
h_t = sigm(W_hx x_t + W_hh h_{t-1})
y_t = W_yh h_t
```
producing one output `y_t` per input step. This is the natural sequence model, and when the input and output are aligned and of equal length it maps sequences to sequences directly. Its shape forces a one-output-per-input-step correspondence: it does not say how to emit `T'` outputs from `T` inputs when `T' != T`, nor how to handle reordering.

**The long-range dependency problem.** Training RNNs over many timesteps is hard. Backpropagation-through-time multiplies many Jacobians, so gradients shrink or blow up exponentially with the time lag (Hochreiter 1991; Bengio, Simard & Frasconi 1994; Hochreiter, Bengio, Frasconi & Schmidhuber 2001; Pascanu, Mikolov & Bengio 2012). A plain RNN therefore struggles to connect events separated by long gaps, exactly the regime created when a network reads a whole input before writing any output.

**Long Short-Term Memory.** The LSTM (Hochreiter & Schmidhuber 1997) replaces the simple recurrence with a memory cell guarded by multiplicative input, forget, and output gates around a cell state with a near-linear self-recurrence. This lets information and gradients persist over hundreds of steps. Graves (2013) gives a clean LSTM formulation for sequence generation and uses gradient-norm clipping to control the exploding-gradient side of the problem. The LSTM is the recurrent model most plausibly able to survive a long lag between input and output.

**Neural language models.** A neural language model represents text autoregressively, `p(w_t | w_1,...,w_{t-1})`, via a softmax over the vocabulary at each step. Feedforward NNLMs (Bengio, Ducharme, Vincent & Jauvin 2003) and recurrent LMs (Mikolov et al. 2010; Sundermeyer, Schluter & Ney 2012) define probabilities for variable-length text by multiplying per-step factors until a termination symbol is produced. A language model is unconditional, though: it produces plausible text, not text tied to a particular source sequence.

**Fixed-vector memory bottlenecks.** Work that squeezes an entire source sentence into one fixed-length vector has observed degradation as the source grows longer (Pouget-Abadie, Bahdanau, van Merrienboer, Cho & Bengio 2014; Cho et al. 2014). Segmenting the source into shorter pieces was one response.

**Minimal time lag.** Hochreiter & Schmidhuber (1997) frame learning difficulty in terms of the minimal time lag: the smallest number of recurrent steps between a cause and the earliest effect that depends on it. The larger this smallest useful gap is, the harder it is for gradient descent to establish any dependency at all, because every useful error signal must survive a long chain before it can improve the relevant early state.

## Baselines

**Phrase-based statistical machine translation.** The dominant MT paradigm (Koehn et al.; Durrani, Haddow, Koehn & Heafield 2014; the LIUM/Schwenk WMT setup). A sentence is translated by searching over segmentations into phrases, each phrase translated via a learned phrase table, with reordering, and scored by a log-linear combination of features: phrase translation probabilities in both directions, an n-gram target language model, a reordering model, and word or phrase penalties. Decoding is a beam search over partial translations. Its gaps are the large pipeline of separately trained components, hard alignment and segmentation decisions, sparse symbolic phrase statistics, and the absence of a shared continuous representation learned end-to-end.

**Neural nets as features inside SMT.** A reliable way to use a neural net for translation is to take an SMT system's `n`-best list of candidate translations and rescore it with a neural language model (Mikolov 2012). Auli, Galley, Quirk & Zweig (2013) add a topic model of the source to the NNLM; Devlin et al. (2014) put a source-conditioned NNLM inside the decoder using decoder alignments to choose the relevant source words. The neural model re-judges candidates the symbolic system already produced. Its gap is that it does not translate on its own and inherits the SMT search space.

**Encode-to-a-vector neural translation.** Kalchbrenner & Blunsom (2013) map an entire source sentence into a single continuous vector and generate the target from it, but they build the source vector with a convolutional or bag-of-words-like map that discards word order. Cho, van Merrienboer, Gulcehre, Bougares, Schwenk & Bengio (2014) use an RNN encoder-decoder with gated recurrent units to map a sequence to a vector and back, aimed mainly at scoring phrase pairs within SMT rather than replacing the translation system. The gaps are loss of source order in some encoders, dependence on SMT in the strongest use cases, and poor behavior on long inputs.

**Monotonic-alignment sequence transducers.** Connectionist Temporal Classification (Graves, Fernandez, Gomez & Schmidhuber 2006) maps an input sequence to an output sequence with an RNN by marginalizing over alignments, but it assumes a monotonic alignment between input and output positions. Translation reorders words, so a strictly monotonic transducer is the wrong model class.

**Differentiable attention.** Graves (2013) introduces a differentiable attention mechanism that lets a network focus on parts of its input; Bahdanau, Cho & Bengio (2014) apply an attention variant to translation so the decoder reads from all source positions instead of relying on a single vector. This relaxes the memory bottleneck by changing the communication channel between input and output.

## Evaluation settings

The natural testbed is large-scale machine translation. The WMT English-to-French task provides a public training corpus, a standard test set, and phrase-based SMT candidate lists, making both direct translation and `n`-best rescoring measurable on the same data. Models use fixed vocabularies of the most frequent source and target words, with out-of-vocabulary tokens mapped to `UNK`, because embedding matrices and softmax layers scale with vocabulary size. Quality is measured by cased BLEU (Papineni, Roukos, Ward & Zhu 2002) on tokenized hypotheses and references, and optimization quality is tracked by held-out perplexity. Decoding an autoregressive model requires approximate search because the exact argmax over all finite output sequences is intractable.

## Code framework

The available building blocks are embedding lookup tables, multi-layer LSTM recurrences, a linear output layer followed by a softmax over the target vocabulary, cross-entropy training, SGD, gradient-norm clipping, minibatches of paired token sequences, and left-to-right search over partial output strings. The scaffold below leaves open how the input reader, output generator, and conditional sequence model are wired.

```python
import torch
import torch.nn as nn


def prepare_source(src_ids, eos_id):
    # TODO: turn raw source ids into the token sequence the reader consumes.
    pass


class SequenceReader(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # TODO: read the whole source and return the state used downstream.
        pass


class SequenceGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, state):
        # TODO: consume the previous target token and return next-token logits.
        pass


class ConditionalSequenceModel(nn.Module):
    def __init__(self, reader, generator, device):
        super().__init__()
        self.reader = reader
        self.generator = generator
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        # TODO: wire the reader and generator into one conditional model.
        pass


def init_weights(module):
    # TODO: initialize recurrent model parameters.
    pass


def train_step(model, src, trg, optimizer, criterion, clip):
    optimizer.zero_grad()
    logits = model(src, trg)
    logits = logits[1:].reshape(-1, logits.shape[-1])
    gold = trg[1:].reshape(-1)
    loss = criterion(logits, gold)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.item()


def beam_search(model, src, sos_id, eos_id, beam_width, max_len):
    # TODO: approximate the best output by extending, scoring, pruning,
    # and retiring completed hypotheses.
    pass
```
