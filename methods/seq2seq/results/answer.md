# Seq2Seq: Sequence-to-Sequence Learning with LSTMs

## Problem

Seq2Seq maps an input sequence `x_1,...,x_T` to an output sequence `y_1,...,y_{T'}` where `T'` can differ from `T` and the alignment between positions can be non-monotonic. The goal is a single end-to-end neural model trained from paired sequences, without a hand-built alignment or segmentation pipeline.

## Key idea

Use one LSTM to read the whole source and another LSTM language model to write the target. The reader's final hidden and cell states are the fixed representation `v`; the generator is initialized from `v` and predicts one target token at a time.

The conditional probability is
```
p(y_1,...,y_{T'} | x_1,...,x_T) =
    product_{t=1}^{T'} p(y_t | v, y_1,...,y_{t-1})
```
with each factor a softmax over the target vocabulary. An `<EOS>` token lets the generator choose when the finite output ends.

The source is reversed before it is read. In the simple aligned case with source length `n`, natural order puts `x_i` at position `i` and `y_i` at `n+i`, so every corresponding lag is `n`. Reversing only the source puts `x_i` at `n-i+1`, giving lag `2i-1`: the average lag is still `n`, but the minimal lag drops from `n` to `1`. That short dependency gives SGD an early credit-assignment foothold without changing the model or objective.

## Algorithm

Training maximizes `(1/|D|) sum_{(x,y) in D} log p(y|x)`. The loss minimized in code is the negative of that objective: sum the teacher-forced token cross-entropies for each sequence, then average those sequence losses over the minibatch. Use SGD with learning rate `0.7`, halve it every half epoch after epoch 5, initialize weights uniformly in `[-0.08, 0.08]`, bucket minibatches by length, cap the vocabulary with `UNK`, and clip the averaged minibatch gradient by
```
g <- 5g / ||g||_2    if ||g||_2 > 5.
```

Decoding approximates `argmax_y p(y|x)` with left-to-right beam search: keep `B` live prefixes scored by summed log-probability, extend every live prefix by every vocabulary word, prune to the best `B`, and retire a hypothesis to the completed set when it emits `<EOS>`.

Large-scale configuration: 1000-dimensional embeddings, 1000 LSTM cells per layer, 4 layers, source vocabulary 160k, target vocabulary 80k.

## Code

```python
import random
import torch
import torch.nn as nn
import torch.optim as optim


def prepare_source(src_ids, eos_id):
    return list(reversed(src_ids)) + [eos_id]


class SequenceReader(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, state = self.rnn(embedded)
        return state


class SequenceGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, state):
        token = token.unsqueeze(0)
        embedded = self.dropout(self.embedding(token))
        output, state = self.rnn(embedded, state)
        logits = self.fc_out(output.squeeze(0))
        return logits, state


class ConditionalSequenceModel(nn.Module):
    def __init__(self, reader, generator, device):
        super().__init__()
        if reader.rnn.hidden_size != generator.rnn.hidden_size:
            raise ValueError("reader and generator hidden sizes must match")
        if reader.rnn.num_layers != generator.rnn.num_layers:
            raise ValueError("reader and generator layer counts must match")
        self.reader = reader
        self.generator = generator
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        trg_len, batch_size = trg.shape
        outputs = torch.zeros(trg_len, batch_size, self.generator.vocab_size, device=self.device)
        state = self.reader(src)
        token = trg[0]

        for t in range(1, trg_len):
            logits, state = self.generator(token, state)
            outputs[t] = logits
            top1 = logits.argmax(1)
            token = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


def init_weights(module):
    for _, parameter in module.named_parameters():
        nn.init.uniform_(parameter, -0.08, 0.08)


def train_step(model, src, trg, optimizer, criterion, clip=5.0):
    optimizer.zero_grad()
    logits = model(src, trg)
    batch_size = trg.shape[1]
    logits = logits[1:].reshape(-1, logits.shape[-1])
    gold = trg[1:].reshape(-1)
    loss = criterion(logits, gold) / batch_size
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.item()


def beam_search(model, src, sos_id, eos_id, beam_width=2, max_len=50):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = model.reader(src)
            beams = [(0.0, [sos_id], state)]
            completed = []

            for _ in range(max_len):
                candidates = []
                for logp, tokens, state in beams:
                    token = torch.tensor([tokens[-1]], device=src.device)
                    logits, next_state = model.generator(token, state)
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                    for word_id, token_logp in enumerate(log_probs.tolist()):
                        candidates.append((logp + token_logp, tokens + [word_id], next_state))

                candidates.sort(key=lambda item: item[0], reverse=True)
                beams = []
                for candidate in candidates[:beam_width]:
                    if candidate[1][-1] == eos_id:
                        completed.append(candidate)
                    else:
                        beams.append(candidate)

                if not beams:
                    break

            completed = completed or beams
            completed.sort(key=lambda item: item[0], reverse=True)
            return completed[0][1]
    finally:
        if was_training:
            model.train()


if __name__ == "__main__":
    src_vocab, trg_vocab = 5000, 5000
    emb_dim, hidden_dim, n_layers, dropout = 256, 512, 2, 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reader = SequenceReader(src_vocab, emb_dim, hidden_dim, n_layers, dropout)
    generator = SequenceGenerator(trg_vocab, emb_dim, hidden_dim, n_layers, dropout)
    model = ConditionalSequenceModel(reader, generator, device).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.7)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    # for src, trg in length_bucketed_loader:
    #     train_step(model, src, trg, optimizer, criterion, clip=5.0)
```
