The problem is neural machine translation with a single encoder-decoder network trained end to end. The standard approach reads the entire source sentence through an RNN and squeezes it into one fixed-length vector, then decodes the whole translation from that same vector. That design is clean, but it has a hard information bottleneck: a vector of fixed size must represent a sentence of unbounded length, carrying every content word, dependency, and reordering fact, and the decoder reuses it at every step. The symptom is empirical and sharp: translation quality is fine for short sentences but collapses once the source gets past twenty or thirty words, and it is worse still on longer sentences than the model saw during training. Simply widening the vector delays the problem without fixing it, because whatever size we choose can still be overwhelmed by a long enough sentence.

The reason the bottleneck bites is that the decoder never gets to go back to the source. At each step the decoder needs only the relevant part of the source for the word it is about to emit, but the fixed vector has already averaged that relevant piece together with everything else, once and for all. What we really need is a source representation the decoder can re-read, together with a mechanism that lets each decoding step pull out the parts it currently cares about.

The method I propose is Bahdanau attention, also called additive attention or RNNsearch. It keeps the source as a sequence of per-position vectors and computes a fresh context vector for every target position by taking a soft, learned weighted average of those source vectors. Because the weighting is differentiable, the alignment is trained jointly with the rest of the model by ordinary backpropagation, not estimated separately as a discrete latent variable.

Concretely, the encoder is a bidirectional RNN. A forward RNN reads the source left to right, giving states that summarize the prefix up to each word, and a backward RNN reads right to left, giving states that summarize the suffix from each word onward. For source position j we concatenate the two directions, so the annotation h_j is centered on x_j but aware of both left and right context. These annotations form the memory that the decoder will address.

At target step i the decoder has a hidden state s_{i-1} summarizing everything produced so far. It scores every source position j by comparing s_{i-1} to h_j with a small feedforward network: e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j). We use an MLP rather than a dot product because the query and annotation live in different dimensional spaces. The scores are normalized to a probability distribution over source positions with a softmax, alpha_ij = exp(e_ij) / sum_k exp(e_ik), and the context for this step is the expected annotation c_i = sum_j alpha_ij h_j. The previous decoder state is used as the query, not the current one, because the current state depends on c_i and using it would be circular. Since U_a h_j does not depend on the decoder step, it can be precomputed once per sentence to keep the alignment cost modest.

The context c_i is fed into a gated decoder, here a GRU, at every gate and at the candidate computation. The decoder update becomes s_i = (1 - z_i) s_{i-1} + z_i s_tilde_i, where the reset and update gates and the candidate state all condition on c_i alongside the previous word embedding and previous state. Emission then uses a small maxout readout over the previous state, the previous word, and the freshly read context. Training is straightforward: maximize log p(y | x) with minibatch SGD and Adadelta, clip gradients to norm one, initialize recurrent weights orthogonally, and start the alignment network with small near-zero weights so attention begins roughly uniform.

The fixed-context encoder-decoder is a special case of this model: if every c_i is forced to the same source summary, the alignment mechanism becomes a constant and we recover the older architecture. But freeing the context to change per step removes the fixed-vector bottleneck and lets the model keep quality up on long sentences. As a side benefit, the matrix of alpha weights is itself an interpretable soft alignment between source and target words.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional=True)
        self.init_state = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.init_state(hidden[-1]))
        return outputs, hidden


class AdditiveAlignment(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.key = nn.Linear(hidden_dim * 2, attn_dim, bias=False)
        self.energy = nn.Linear(attn_dim, 1, bias=False)

    def precompute(self, encoder_outputs):
        return self.key(encoder_outputs)

    def forward(self, hidden, projected_annotations):
        query = self.query(hidden).unsqueeze(0)
        e = self.energy(torch.tanh(query + projected_annotations)).squeeze(2)
        return torch.softmax(e.transpose(0, 1), dim=1)


class DecoderGRUCell(nn.Module):
    def __init__(self, emb_dim, context_dim, hidden_dim):
        super().__init__()
        self.candidate_x = nn.Linear(emb_dim, hidden_dim)
        self.candidate_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_c = nn.Linear(context_dim, hidden_dim, bias=False)
        self.update_x = nn.Linear(emb_dim, hidden_dim)
        self.update_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_c = nn.Linear(context_dim, hidden_dim, bias=False)
        self.reset_x = nn.Linear(emb_dim, hidden_dim)
        self.reset_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_c = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, embedded, prev_hidden, context):
        z = torch.sigmoid(
            self.update_x(embedded) + self.update_h(prev_hidden) + self.update_c(context)
        )
        r = torch.sigmoid(
            self.reset_x(embedded) + self.reset_h(prev_hidden) + self.reset_c(context)
        )
        candidate = torch.tanh(
            self.candidate_x(embedded) + self.candidate_h(r * prev_hidden) + self.candidate_c(context)
        )
        return (1 - z) * prev_hidden + z * candidate


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, maxout_dim, dropout, alignment):
        super().__init__()
        self.output_dim = output_dim
        self.maxout_dim = maxout_dim
        self.alignment = alignment
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn_cell = DecoderGRUCell(emb_dim, hidden_dim * 2, hidden_dim)
        self.readout = nn.Linear(hidden_dim + emb_dim + hidden_dim * 2, maxout_dim * 2)
        self.fc_out = nn.Linear(maxout_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, projected_annotations):
        embedded = self.dropout(self.embedding(input))
        alpha = self.alignment(hidden, projected_annotations)
        annotations = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)
        prev_hidden = hidden
        hidden = self.rnn_cell(embedded, prev_hidden, context)
        readout = self.readout(torch.cat((prev_hidden, embedded, context), dim=1))
        maxout = readout.view(readout.shape[0], self.maxout_dim, 2).max(dim=2).values
        prediction = self.fc_out(maxout)
        return prediction, hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        trg_len, batch = trg.shape
        outputs = torch.zeros(trg_len, batch, self.decoder.output_dim, device=trg.device)
        encoder_outputs, hidden = self.encoder(src)
        projected_annotations = self.decoder.alignment.precompute(encoder_outputs)
        input = trg[0]
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, projected_annotations)
            outputs[t] = output
            teacher = torch.rand((), device=trg.device).item() < teacher_forcing_ratio
            input = trg[t] if teacher else output.argmax(1)
        return outputs


# Training skeleton:
# alignment = AdditiveAlignment(hidden_dim, attn_dim)
# model = Seq2Seq(
#     Encoder(input_dim, emb_dim, hidden_dim, dropout),
#     Decoder(output_dim, emb_dim, hidden_dim, maxout_dim, dropout, alignment),
#     device
# ).to(device)
# optimizer = optim.Adadelta(model.parameters())
# criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
# for src, trg in loader:
#     optimizer.zero_grad()
#     output = model(src, trg, teacher_forcing_ratio=0.5)
#     loss = criterion(output[1:].reshape(-1, output.shape[-1]), trg[1:].reshape(-1))
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     optimizer.step()
```
