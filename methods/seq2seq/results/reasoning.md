Let me start from the shape mismatch that keeps blocking neural sequence problems. A deep feedforward net wants a fixed-dimensional vector in and a fixed-dimensional vector out. Translation is not shaped that way: a source sentence has T words, the target has T' words, T' is not known in advance, and word order can move around. Speech recognition and question answering have the same problem. A plain recurrent net at least knows how to read a sequence, carrying h_t = sigm(W_hx x_t + W_hh h_{t-1}) forward and emitting y_t = W_yh h_t, but that gives one output per input step. It works when the alignment is known and the two sides have the same length. It has no clean place for "read all of this, then produce a different-length sequence."

What forces the lockstep is that the same net both reads and writes on the same clock. So let me try splitting those jobs. One recurrent net can consume x_1,...,x_T without producing any output, and after the final input I keep its final LSTM hidden and cell states as a single representation v. That state is a function of the whole input — it has seen every source token before generation begins. Now the remaining problem is one I already know how to solve: starting from some state, generate a sequence one symbol at a time. A neural language model does exactly that, modeling p(w_t | w_1,...,w_{t-1}) with a softmax at each step. The only thing it is missing is the source. If I initialize the language model's state with v instead of a blank state, every next-word distribution can depend on the source through v.

Following that through, the conditional model is

  p(y_1,...,y_{T'} | x_1,...,x_T) = product_{t=1}^{T'} p(y_t | v, y_1,...,y_{t-1}),

with each factor a softmax over the target vocabulary. Two things fall out of this factorization that I should check are actually what I want. First, the output length T' appears nowhere on the right-hand side as a fixed quantity — the product runs until generation chooses to stop — so the output length is genuinely decoupled from T. Second, the probability of a whole target is the product of left-to-right next-token probabilities, which means I can score any candidate target by reading it off one factor at a time. Both are the properties the shape-mismatch problem demanded.

But "until generation chooses to stop" is doing unexamined work. If the decoder has no termination symbol, then the product above has no defined last factor, and a "sequence probability" is not attached to any particular finite string. So I add an <EOS> token to the vocabulary and train every target to end with it. Ending becomes just another softmax decision, and a finite output is the prefix up to the first <EOS>. I will also mark the source end with <EOS>, because the reader needs a definite final input after which its state is the v handed to the generator.

Suppose I instantiate this with a plain RNN as the recurrence. The first target word can depend on the first source word, but that source word was read T recurrent steps before generation begins. Backpropagation has to push the error for y_1 back through the decoder's first step, through the final reader state, and across all of the reader's earlier states — order T steps. A vanilla RNN multiplies a Jacobian at every one of those steps. If the typical singular value of that Jacobian is some r, the gradient that reaches the first source token scales like r^T. For r a little below 1, say r=0.9 and T=30, that is 0.9^30 ≈ 0.04 — a 25-fold attenuation; for a 50-word sentence, 0.9^50 ≈ 0.005. The model may be expressive enough in principle, but the optimizer gets almost no usable signal for the early source words. So the plain-RNN instantiation fails not on representation but on credit assignment.

That points at the recurrence, not the architecture, as the thing to replace. An LSTM cell has a near-linear self path through the cell state, guarded by input, forget, and output gates, so when the forget gate stays near 1 the per-step multiplier is near 1 rather than r<1, and information and gradients can persist across long gaps instead of being squashed geometrically. So I keep the read-then-write structure but make both recurrences LSTMs. There is also a question of whether the reader and generator should share one network. They are doing different jobs — one compresses a source into v, the other unrolls a target from v — so sharing would force one set of dynamics to do both. Two separate LSTMs cost little more at run time because each still runs once per token, and the extra parameters are capacity I expect to be useful rather than wasted; I'd confirm that empirically, but the asymmetry of the two jobs is reason enough to keep them separate by default.

The fixed vector has to carry an entire sentence, so I should not make the recurrent state shallow by default. A deeper LSTM gives more state across layers and more nonlinear processing at each timestep. Four layers is a sensible large-scale choice, while a smaller implementation can expose `n_layers` as a knob. The output side can remain a direct softmax over the target vocabulary; it is expensive, but it is the simplest normalized next-word distribution and it keeps the likelihood objective exact.

Training then falls out of the factorization. For a dataset D of paired source and target sequences, maximize

  (1 / |D|) sum_{(x,y) in D} log p(y | x).

Because log p(y | x) is a sum of log p(y_t | v, y_<t), maximizing this objective is the same as minimizing the teacher-forced negative log-likelihood, equivalently the cross-entropy loss: at decoder step t, feed the true previous target token and ask the softmax to put probability on the true next token. The loss should sum token negative log-likelihoods inside each example and average those sequence losses over the minibatch, matching the factorized likelihood. Backpropagation runs through the generator, through the initial generator state, and through the reader. I want small initial weights, say uniform in [-0.08, 0.08], so gates and recurrent activations start in a non-saturated range. Plain SGD is enough if I use a large fixed learning rate at first, then start halving it once progress slows after several passes through the data.

LSTMs reduce vanishing gradients, but exploding gradients are still possible. The clean guard is to clip the norm of the whole minibatch gradient. Compute g as the average gradient for the minibatch, s = ||g||_2. If s > 5, replace g by 5g/s; if s <= 5, leave it alone. The direction is unchanged, and the gradient norm is capped at 5; the SGD step length is then the learning rate times that clipped norm. With batches of 128 examples, the gradient being clipped is the summed batch gradient divided by 128.

There is also the basic efficiency problem of variable-length batches. If I mix very short and very long sentences in one minibatch, padding up to the longest sentence wastes most of the computation. Grouping sentences of roughly similar length into the same minibatch changes no objective or model assumption; it just avoids paying for padding. The vocabulary also has to be capped, because embeddings and the output softmax scale directly with vocabulary size, so rare words become `UNK`.

Even with LSTMs, though, I want to stare at the time axis again, because the read-then-write arrangement might still be making credit assignment harder than it needs to be. Take the easy monotone case: source a,b,c translates roughly to target alpha,beta,gamma. I feed a,b,c and then start producing alpha,beta,gamma. In the concatenated read-then-write stream, x_i sits at position i and y_i at position n+i, so the lag between a source token and the target token that most depends on it is d_i = (n+i)-i = n for every i. For n=3 that is d = [3,3,3]: every correspondence is separated by the full source length, and there is no short dependency for SGD to latch onto first.

Now, I am free to choose the order in which I present the source — the model has no built-in notion that input must arrive left-to-right. What if I reverse only the source, so the stream is c,b,a then alpha,beta,gamma? Then x_i appears at position n-i+1 while y_i is still at n+i, so d_i = (n+i)-(n-i+1) = 2i-1. Before I read anything into this, let me actually compute both vectors for a concrete length and check the averages, because the reversal could easily be making things worse, not better. For n=5:

  natural order:   d_i = (n+i)-i      = [5, 5, 5, 5, 5],  average 5, minimum 5
  reversed source: d_i = 2i-1         = [1, 3, 5, 7, 9],  average 5, minimum 1

The averages are equal — sum_{i=1}^n (2i-1) = n^2, so the mean is n either way — which is the first thing I had to rule out: reversal does not shorten the typical dependency, so it cannot be cheating by making every alignment short. What it does change is the minimum and the front of the distribution: d_1 drops from 5 to 1, d_2 to 3, so the first few source-target correspondences now sit right at the boundary where generation starts, while the cost is paid by the tail (d_5 went from 5 to 9). For SGD this is the favorable trade: it can establish communication through the now-short early dependencies first and use the state dynamics it learns there to carry the long tail. I check the alternatives the same way. Reversing the target instead would move the short link to the wrong side of the boundary; reversing both source and target restores d_i = n everywhere (the boundary gap is symmetric again). So the transformation that helps is specifically reverse-source, natural-target.

This also reframes the fixed-vector worry. If a single vector fails on long sentences, it could be that the vector is too small, but it could equally be that the optimizer never learned how to fill it — and the lag computation says the second failure mode is real and avoidable. Source reversal adds no memory and leaves the likelihood unchanged (it is a relabeling of input positions); it only shortens the front of the credit-assignment paths. That makes it cheap enough to put in the data pipeline first, before reaching for a more elaborate communication mechanism.

At test time I need the best target under the model,

  y_hat = argmax_y p(y | x).

The exact search space is all finite strings over the target vocabulary, so exact maximization is out. Greedy decoding keeps one prefix and takes the locally highest-probability next token. The worry is that a token that looks best locally can lead into a region with no good continuation. Let me make that worry concrete with a scripted three-symbol model (SOS, EOS, and one word 'a') so I can see whether greedy and beam actually diverge. Say after SOS the model assigns p(a)=0.9, p(EOS)=0.1, and after 'a' it assigns p(EOS)=0.6, p(a)=0.4. Greedy takes 'a' first (0.9 > 0.1), then EOS (0.6), giving the string [a] with log-prob log0.9 + log0.6 = -0.105 + -0.511 = -0.616. The one-token output [EOS] has log-prob log0.1 = -2.303. So here greedy lands on the genuinely better hypothesis, -0.616 > -2.303 — good, but only because I let it see the second factor. If after SOS the EOS probability had been 0.55 and 'a' 0.45, greedy would commit to EOS and never discover the -0.616 path. A beam of width B guards against exactly that by keeping B prefixes alive instead of one.

So beam search: each live prefix carries an accumulated score. At a step, extend every live prefix by every vocabulary item, score each extension by adding the new token's log-probability to the prefix score, sort all extensions, and keep the B best. If a kept extension ends in <EOS>, retire it into a completed set instead of expanding it again. Continue until the live beam is empty or a length cap is reached, then return the completed hypothesis with the highest score. Tracing the loop on the model above with B=2: step 0 produces [SOS,a] at -0.105 (kept live) and [SOS,EOS] at -2.303 (retired); step 1 expands [SOS,a] into [SOS,a,EOS] at -0.616 (retired) and [SOS,a,a] at -1.022 (kept live); the completed set already contains the eventual winner [SOS,a,EOS] at -0.616, which beats the early [SOS,EOS] at -2.303. So the retire-on-EOS rule correctly lets a short hypothesis that ends early compete against longer ones on total score rather than being expanded forever.

One detail the scoring forces: the score has to be the summed log-probability, not the raw product. This is not just convention. A long target is a product of many sub-1 factors; for a 400-token output with even a generous per-token probability of 0.1, the raw product is 0.1^400, which in float64 is exactly 0.0 — every long candidate underflows to the same value and the argmax becomes meaningless. The log of the same quantity is 400·log(0.1) = -921, a finite number I can compare. And summing logs is exactly the product rule applied to the factorization, so nothing about the objective changes — only the numerics.

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

Stepping back, the pieces hang together: read the variable-length source into a single LSTM state, generate a variable-length target with an LSTM language model initialized from that state, make finite outputs explicit with <EOS>, train the conditional factorization by maximum likelihood, keep exploding gradients bounded by g <- 5g/||g|| when ||g|| > 5, reverse the source — which I checked leaves the average lag at n while dropping the minimal lag from n to 1 in the aligned case — and decode by summed-log-prob beam search because exact sequence maximization is intractable. The parts I could pin down on paper here — the length-decoupling of the factorization, the reversal arithmetic, the beam trace, the underflow — hold. What I cannot settle without training is whether v is large enough to carry a long sentence and whether the early-lag foothold actually translates into better convergence; those are the first things I would watch on the WMT runs.
