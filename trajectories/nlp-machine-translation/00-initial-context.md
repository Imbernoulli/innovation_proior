# Initial context: neural machine translation

## Research question

How can a single neural network map a variable-length source sequence to a variable-length target sequence when the two lengths differ and the alignment between positions is neither one-to-one nor monotonic? Feedforward networks require fixed-size inputs and outputs, and plain recurrent networks emit one output per input step. The open problem is a domain-independent architecture that reads an arbitrary-length input and writes an arbitrary-length output, trained only from paired sequences.

## Prior art / Background / Baselines

**Phrase-based statistical machine translation.** Systems translate by segmenting the source into phrases, looking them up in a learned phrase table, and scoring candidates with a log-linear combination of phrase statistics, a target n-gram language model, and reordering and penalty features. They remain a pipeline of separately trained components, hard alignment and segmentation decisions, and sparse symbolic statistics, with no shared continuous representation learned end to end.

**Neural nets as features inside SMT.** A neural language model rescores the n-best list produced by an SMT decoder, sometimes conditioned on source-side topics or decoder alignments. It does not translate on its own and inherits the SMT search space.

**Encode-to-a-vector neural translation.** Neural models compress a whole source sentence into a single fixed vector and generate the target from it, using convolutional or recurrent encoders. They degrade as the source grows longer, and the strongest uses so far score phrase pairs inside SMT rather than replacing it.

**Monotonic-alignment sequence transducers.** Connectionist Temporal Classification trains an RNN to map an input sequence to an output sequence by marginalizing over monotonic alignments. The monotonicity assumption is too strong for translation, where word order can reorder across languages.

**Differentiable attention.** Mechanisms that let a model read from selected input positions rather than a single vector have been explored in handwriting synthesis and image captioning. They have not yet been shown to scale to the largest end-to-end translation tasks.

## Fixed substrate / Code framework

The available scaffold supplies embedding lookup tables, multi-layer LSTMs, a linear output layer with softmax over the target vocabulary, cross-entropy training, SGD, gradient-norm clipping, and left-to-right beam search. The wiring between the source reader, target generator, and conditional sequence model is left open.

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

## Editable interface

The design choices left to experiment with are: how the reader encodes a source sequence into a state; how the generator is conditioned on that state at each output step; how to preserve information from the start of a long source through to target generation; how to cap vocabulary size and handle unknown words; and how to implement and tune approximate decoding search.

## Evaluation settings

Large-scale machine translation, especially the WMT English-to-French task, is the natural testbed. Models use fixed vocabularies of the most frequent source and target words, with rare words mapped to `UNK`. Quality is measured by cased BLEU on tokenized hypotheses and references, and training progress by held-out perplexity. Decoding uses beam search because the exact argmax over output sequences is intractable.
