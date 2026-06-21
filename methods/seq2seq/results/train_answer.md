The problem I want to solve is the shape mismatch that keeps blocking neural sequence tasks. A deep feedforward network wants a fixed-dimensional vector in and a fixed-dimensional vector out, but the problems that matter most in language are not shaped that way. Machine translation maps a source sentence of $T$ words to a target of $T'$ words, with $T' \neq T$ in general and with word order that can shuffle across the languages; speech recognition maps a variable-length acoustic stream to a variable-length transcript; question answering maps a question to an answer. In each case the output length is not known until it is produced, and the alignment between input and output positions is neither one-to-one nor monotonic. A plain recurrent net at least knows how to read a sequence, carrying $h_t = \mathrm{sigm}(W_{hx} x_t + W_{hh} h_{t-1})$ forward and emitting $y_t = W_{yh} h_t$, but that produces exactly one output per input step; it works only when the two sides are aligned and of equal length, and it has no clean place for "read all of this, then produce a different-length sequence." The existing alternatives each fall short of an end-to-end mapping. Phrase-based statistical MT is a large pipeline of separately trained components with hard alignment and segmentation decisions and sparse symbolic statistics; neural nets used to rescore an SMT system's $n$-best list never translate on their own and inherit the SMT search space; encoders that squeeze a sentence into one vector via convolution or bag-of-words discard word order; monotonic transducers like CTC assume a monotonic alignment, which is the wrong model class once words reorder. What we need is a single, domain-independent network that accepts arbitrary-length input, produces arbitrary self-determined-length output, assumes almost nothing about the alignment, and is trainable end-to-end from paired sequences alone.

I propose Seq2Seq: sequence-to-sequence learning with LSTMs. The defining move is to break the lockstep of one-output-per-input. I let one recurrent network — the reader — consume $x_1,\dots,x_T$ without emitting anything, and after the final input I keep its final LSTM hidden and cell states as a fixed representation $v$. That state is a function of the whole input. The remaining problem is then familiar: starting from a given state, generate a sequence one symbol at a time, which is exactly what a neural language model does by modeling $p(w_t \mid w_1,\dots,w_{t-1})$ with a softmax at each step. The only missing ingredient is the source, so I initialize a second LSTM — the generator — from $v$ instead of from a blank state, and every next-token distribution then depends on the source through $v$. The conditional model is the factorization
$$p(y_1,\dots,y_{T'} \mid x_1,\dots,x_T) = \prod_{t=1}^{T'} p(y_t \mid v,\, y_1,\dots,y_{t-1}),$$
each factor a softmax over the target vocabulary. This factorization is what frees the output length from $T$: the probability of a whole target is just the product of its left-to-right next-token probabilities, however long it runs. To make a finite output well defined I add an $\langle\mathrm{EOS}\rangle$ token to the vocabulary and train every target to end with it, so termination becomes just another softmax decision and the finite output is the prefix up to the first $\langle\mathrm{EOS}\rangle$; I also append $\langle\mathrm{EOS}\rangle$ to the source, because the reader needs a definite final input after which its state is handed to the generator.

A plain RNN for this is not enough, and the reason is credit assignment, not expressiveness. The first target word can depend on the first source word, but that source word was read $T$ steps before generation even begins, so the error signal for $y_1$ must travel back through the whole decoder start, the final reader state, and the reader's earlier states. A vanilla RNN multiplies a Jacobian at every step, so gradients shrink or explode roughly geometrically with the lag, and the optimizer gets almost no usable signal for the early source words. This is why I use LSTMs: the memory cell has a near-linear self-recurrence guarded by input, forget, and output gates, so information and gradients persist across long gaps instead of being squashed by the same nonlinear transition at every step. I keep the reader and the generator as two separate LSTMs rather than one shared network — they are doing different jobs, one compressing a source and one producing a target, and since each still runs once per token the extra parameters are cheap and useful capacity rather than forcing a single set of dynamics to do both jobs. Because the fixed vector must carry an entire sentence, I do not make the recurrent state shallow: a deeper LSTM gives more state across layers and more nonlinear processing per step (four layers at large scale, with `n_layers` a knob at smaller scale). The output stays a direct softmax over the target vocabulary, which is expensive but is the simplest exactly normalized next-token distribution and keeps the likelihood objective exact.

Training falls out of the factorization. For a dataset $D$ of paired sequences I maximize $\frac{1}{|D|} \sum_{(x,y)\in D} \log p(y \mid x)$, and because $\log p(y \mid x)$ is a sum of $\log p(y_t \mid v, y_{<t})$, maximizing this is the same as minimizing the teacher-forced negative log-likelihood, the cross-entropy: at each decoder step I feed the true previous target token and ask the softmax to put mass on the true next token. The loss sums token negative log-likelihoods inside each example and averages those sequence losses over the minibatch, matching the factorized likelihood. I initialize weights uniformly in $[-0.08, 0.08]$ so gates and recurrent activations start in a non-saturated range, and plain SGD suffices with a large fixed learning rate ($0.7$) that I begin halving every half epoch once progress slows after epoch 5. LSTMs tame vanishing gradients but not exploding ones, so I clip the norm of the whole minibatch gradient: with $g$ the averaged minibatch gradient and $s = \lVert g \rVert_2$, I replace $g$ by $5g/s$ when $s > 5$ and leave it otherwise. The direction is unchanged and the norm is capped at $5$, so the step length is the learning rate times the clipped norm. For efficiency I bucket sentences of similar length into the same minibatch — this changes no objective or model assumption, it only avoids paying for padding up to the longest sentence — and I cap the vocabulary with $\mathrm{UNK}$ because embeddings and the output softmax scale directly with vocabulary size.

The one non-obvious design choice is to reverse the source before reading it, and it is justified entirely by the minimal-time-lag argument. Take the simple aligned case where source length is $n$ and $x_i$ corresponds to $y_i$. In natural order the concatenated read-then-write stream puts $x_i$ at position $i$ and $y_i$ at position $n+i$, so every corresponding lag is $d_i = (n+i) - i = n$; the minimal useful dependency is $n$ steps long and there is no short dependency for SGD to learn first. I am free to choose the order in which the source is presented, so I reverse only the source: now $x_i$ sits at position $n-i+1$ while $y_i$ is still at $n+i$, giving $d_i = (n+i) - (n-i+1) = 2i-1$. The average lag is $\frac{1}{n}\sum_{i=1}^n (2i-1) = n$, identical to before — I have not shortened the dependencies on average — but the minimum drops from $n$ to $1$: $d_1 = 1$, $d_2 = 3$, and the first few source–target correspondences sit right at the boundary where generation starts. That short dependency is exactly the foothold the minimal-lag argument cares about; SGD can establish communication through the short paths first and then use the learned state dynamics to support the dependencies that remain long. Reversing the target would move the wrong side, and reversing both would restore the large boundary gap, so the useful transformation is to reverse the source and leave the target in natural order. This also reframes the fixed-vector worry: source reversal adds no memory and does not change the likelihood, it only shortens the shortest credit-assignment paths, which is reason enough to make it part of the data pipeline. At test time I need $\hat{y} = \arg\max_y p(y \mid x)$, but the search space is all finite strings over the vocabulary so exact maximization is intractable, and greedy decoding can commit to a word whose continuation is poor. I use left-to-right beam search instead: keep $B$ live prefixes each with an accumulated log-probability, extend every live prefix by every vocabulary item, score each extension by adding the new token's log-probability to its prefix score, sort and keep the best $B$, and retire any kept hypothesis ending in $\langle\mathrm{EOS}\rangle$ into a completed set rather than expanding it again; continue until the beam empties or a length cap is hit, then return the completed hypothesis with the highest summed log-probability. The scores must be summed log-probabilities because raw products underflow and log addition is exactly the product rule in the factorization.

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
