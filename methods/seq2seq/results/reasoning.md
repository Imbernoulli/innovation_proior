Let me start from the shape mismatch that keeps blocking neural sequence problems. A deep feedforward net wants a fixed-dimensional vector in and a fixed-dimensional vector out. Translation is not shaped that way: a source sentence has T words, the target has T' words, T' is not known in advance, and word order can move around. Speech recognition and question answering have the same problem. A plain recurrent net at least knows how to read a sequence, carrying h_t = sigm(W_hx x_t + W_hh h_{t-1}) forward and emitting y_t = W_yh h_t, but that gives one output per input step. It works when the alignment is known and the two sides have the same length. It has no clean place for "read all of this, then produce a different-length sequence."

The lockstep is the piece to break. I can let one recurrent net consume x_1,...,x_T without producing outputs, and after the final input keep its final LSTM hidden and cell states as the fixed representation v. That state is a function of the whole input. Now the remaining problem is familiar: starting from some state, generate a sequence one symbol at a time. A neural language model already does that by modeling p(w_t | w_1,...,w_{t-1}) with a softmax at each step. The only missing condition is the source. If the language model's initial state is v instead of a blank state, every next-word distribution can depend on the source through v.

So the conditional model should be

  p(y_1,...,y_{T'} | x_1,...,x_T) = product_{t=1}^{T'} p(y_t | v, y_1,...,y_{t-1}),

with each factor represented by a softmax over the target vocabulary. This factorization is important: the output length is not tied to T, and the probability of a whole target is just the product of the left-to-right next-token probabilities. I still need an explicit way to stop. If the decoder has no termination symbol, a "sequence probability" is not attached to a finite string in a useful way. Put an <EOS> token in the vocabulary and train every target to end with it. Then ending is just another softmax decision, and a finite output is the prefix up to the first <EOS>. I will also mark the source end with <EOS>, because the reader needs a definite final input after which its state is handed to the generator.

The first version with a plain RNN is not enough. The first target word can depend on the first source word, but that source word was read T recurrent steps before generation begins. Backpropagation has to push the error for y_1 back through the whole decoder start, through the final reader state, and across the reader's earlier states. A vanilla RNN multiplies a Jacobian at every one of those steps, so gradients shrink or explode roughly geometrically with the lag. The model may be expressive enough in principle, but the optimizer gets almost no usable signal for early source words.

This is the reason to use LSTMs rather than simple recurrent units. The memory cell has a near-linear self path guarded by input, forget, and output gates, so information and gradients can persist across long gaps instead of being repeatedly squashed by the same nonlinear transition. The reader and the generator are doing different jobs, too: one compresses a source, the other produces a target. Sharing one recurrent network would save parameters but would force one set of dynamics to do both jobs. Two LSTMs cost little more computationally because each still runs once per token, and the extra parameters are useful capacity.

The fixed vector has to carry an entire sentence, so I should not make the recurrent state shallow by default. A deeper LSTM gives more state across layers and more nonlinear processing at each timestep. Four layers is a sensible large-scale choice, while a smaller implementation can expose `n_layers` as a knob. The output side can remain a direct softmax over the target vocabulary; it is expensive, but it is the simplest normalized next-word distribution and it keeps the likelihood objective exact.

Training then falls out of the factorization. For a dataset D of paired source and target sequences, maximize

  (1 / |D|) sum_{(x,y) in D} log p(y | x).

Because log p(y | x) is a sum of log p(y_t | v, y_<t), maximizing this objective is the same as minimizing the teacher-forced negative log-likelihood, equivalently the cross-entropy loss: at decoder step t, feed the true previous target token and ask the softmax to put probability on the true next token. The loss should sum token negative log-likelihoods inside each example and average those sequence losses over the minibatch, matching the factorized likelihood. Backpropagation runs through the generator, through the initial generator state, and through the reader. I want small initial weights, say uniform in [-0.08, 0.08], so gates and recurrent activations start in a non-saturated range. Plain SGD is enough if I use a large fixed learning rate at first, then start halving it once progress slows after several passes through the data.

LSTMs reduce vanishing gradients, but exploding gradients are still possible. The clean guard is to clip the norm of the whole minibatch gradient. Compute g as the average gradient for the minibatch, s = ||g||_2. If s > 5, replace g by 5g/s; if s <= 5, leave it alone. The direction is unchanged, and the gradient norm is capped at 5; the SGD step length is then the learning rate times that clipped norm. With batches of 128 examples, the gradient being clipped is the summed batch gradient divided by 128.

There is also the basic efficiency problem of variable-length batches. If I mix very short and very long sentences in one minibatch, padding up to the longest sentence wastes most of the computation. Grouping sentences of roughly similar length into the same minibatch changes no objective or model assumption; it just avoids paying for padding. The vocabulary also has to be capped, because embeddings and the output softmax scale directly with vocabulary size, so rare words become `UNK`.

Now I need to stare at the time axis again, because the reader-then-generator arrangement creates a subtle optimization problem even with LSTMs. Imagine the easy monotone case first: source a,b,c translates roughly to target alpha,beta,gamma. If I feed a,b,c and then start producing alpha,beta,gamma, every corresponding pair is separated by the whole source length. For a source length n, x_i is at position i in the concatenated read-then-write stream, y_i is at position n+i, and the lag d_i is n+i-i = n for every i. The average lag is n and the minimal lag is also n. There is no short dependency for SGD to learn first.

I am free to choose the order in which the source is presented. If I reverse only the source, the stream becomes c,b,a, then alpha,beta,gamma. The original x_i now appears at position n-i+1, while y_i is still at n+i, so the lag becomes d_i = n+i-(n-i+1) = 2i-1. The average is (1/n) sum_{i=1}^n (2i-1) = n, exactly the same as before. I have not shortened the corresponding dependencies on average. What changed is the minimum: d_1 is now 1 instead of n, d_2 is 3, and the first few source-target correspondences are close to the boundary where generation starts. That is exactly the kind of foothold the minimal-time-lag argument cares about. SGD can establish communication through short dependencies first, then use the learned state dynamics to support the harder dependencies that remain long. Reversing the target would move the wrong side; reversing both would restore the large boundary gap. The useful transformation is to reverse the source and leave the target in its natural order.

This also changes how I should interpret the fixed-vector worry. If a single vector fails on long sentences, it may be because the vector is too small, but it may also be because the optimizer never learned how to fill it. Source reversal does not add memory or change the likelihood. It changes the shortest credit-assignment paths. That is enough reason to make it part of the data pipeline before adding a more elaborate communication mechanism.

At test time I need the best target under the model,

  y_hat = argmax_y p(y | x).

The exact search space is all finite strings over the target vocabulary, so exact maximization is out. Greedy decoding keeps one prefix and takes the locally highest-probability next token, but that can choose a word whose continuation is poor. A beam keeps B prefixes alive. Each prefix has an accumulated log-probability. At a step, extend every live prefix by every vocabulary item, score each extension by adding the new token's log-probability to the prefix score, sort all extensions, and keep only the B best. If one of the kept extensions ends in <EOS>, retire it into the completed set instead of expanding it again. Continue until the live beam is empty or a length cap is reached, then return the completed hypothesis with the highest summed log-probability. The scores must be summed log-probabilities: raw products underflow, and log addition is exactly the product rule in the factorization.

Here is the model in code form. The source preparation reverses the source and appends <EOS>. The reader folds that sequence into the final LSTM hidden and cell states. The generator is a one-step LSTM language model conditioned by that state. The wrapper trains by teacher-forced cross-entropy. Beam search follows the extend, sum-log-prob, prune, retire-on-<EOS> loop.

```python
import random
import torch
import torch.nn as nn


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
```

The chain is now tight: read the variable-length source into a single LSTM state, generate a variable-length target with an LSTM language model initialized from that state, make finite outputs explicit with <EOS>, train the conditional factorization by maximum likelihood, keep exploding gradients bounded by g <- 5g/||g|| when ||g|| > 5, reverse the source so the average corresponding distance stays the same while the minimal lag drops from n to 1 in the simple aligned case, and decode by beam search because exact sequence maximization is intractable.
