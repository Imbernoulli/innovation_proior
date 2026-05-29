# Soft Attention for Neural Machine Translation (Align-and-Translate / RNNsearch)

## Problem

Encoder–decoder neural translation compresses the entire source sentence into a single fixed-length vector and decodes from it. That vector is an information bottleneck: a fixed number of floats must represent a sentence of unbounded length, and the same vector conditions every decoding step. The visible symptom is that translation quality degrades sharply as source sentences get longer.

## Key idea

Do not summarize the source into one vector. Encode it into a *sequence* of per-position vectors (annotations), and let the decoder build a **distinct context vector at every output step** by taking a soft, learned, weighted average of those annotations — attending to the source positions relevant to the word it is about to emit. Because the weighting is a differentiable (softmax) function, the alignment is trained jointly with the rest of the network by backpropagation, rather than being a discrete latent variable estimated separately as in statistical MT.

## Method

**Bidirectional encoder.** A forward RNN produces →h_1…→h_Tx and a backward RNN produces ←h_1…←h_Tx; the annotation for source position j is the concatenation h_j = [→h_j ; ←h_j], summarizing the whole sentence with emphasis on the words around x_j.

**Per-step context (soft attention).** With s_{i-1} the decoder state just before emitting target word y_i:

- alignment score (additive MLP): e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j), with W_a ∈ R^{n'×n}, U_a ∈ R^{n'×2n}, v_a ∈ R^{n'}
- weights (softmax over source positions): α_ij = exp(e_ij) / Σ_k exp(e_ik)
- context (expected annotation): c_i = Σ_j α_ij h_j

An MLP is used rather than a dot product because the query s_{i-1} ∈ R^n and the key h_j ∈ R^{2n} have different dimensions; W_a and U_a project both into a common space. Since U_a h_j is independent of the decoder step i, it is precomputed once per sentence. The previous state s_{i-1} is used as the query (not s_i, which would be circular because s_i depends on c_i).

**Decoder (gated, context-conditioned).**

- s̃_i = tanh(W E y_{i-1} + U[r_i ∘ s_{i-1}] + C c_i)
- z_i = σ(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i)
- r_i = σ(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)
- s_i = (1 − z_i) ∘ s_{i-1} + z_i ∘ s̃_i,  with s_0 = tanh(W_s ←h_1)

**Output (deep maxout readout).** The detailed readout uses the pre-emission state: t̃_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i; t_i = [max(t̃_{i,2k-1}, t̃_{i,2k})]_k; p(y_i | y_{<i}, x) ∝ exp(y_i^T W_o t_i).

Fixing every c_i to the same source summary, with matching context dimensionality (the original reduction uses the final forward encoder state →h_Tx), recovers the plain RNN encoder–decoder.

**Training.** Maximize ∑ log p(y | x) by minibatch SGD with Adadelta; clip the global gradient norm to 1; orthogonal init for recurrent matrices, near-zero init for the alignment net. Decode with beam search. Reference sizes: hidden n = 1000, embeddings m = 620, maxout l = 500, alignment hidden n' = 1000.

## Code

A faithful, compact PyTorch implementation (bidirectional GRU encoder, additive soft alignment, an explicit context-conditioned decoder GRU cell, and the maxout readout):

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

    def forward(self, src):                               # src: [src_len, batch]
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)             # outputs = annotations h_j
        # s_0 = tanh(W_s backward_h_1); hidden[-1] is backward state at source position 1
        hidden = torch.tanh(self.init_state(hidden[-1]))
        return outputs, hidden


class AdditiveAlignment(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, attn_dim, bias=False)       # W_a
        self.key = nn.Linear(hidden_dim * 2, attn_dim, bias=False)     # U_a
        self.energy = nn.Linear(attn_dim, 1, bias=False)               # v_a

    def precompute(self, encoder_outputs):
        return self.key(encoder_outputs)                 # U_a h_j: [src_len, batch, attn]

    def forward(self, hidden, projected_annotations):    # hidden: [batch, n]
        query = self.query(hidden).unsqueeze(0)           # [1, batch, attn]
        e = self.energy(torch.tanh(query + projected_annotations)).squeeze(2)
        return torch.softmax(e.transpose(0, 1), dim=1)    # alpha_ij: [batch, src_len]


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

    def forward(self, input, hidden, encoder_outputs, projected_annotations):   # input: [batch]
        embedded = self.dropout(self.embedding(input))                          # E y_{i-1}
        alpha = self.alignment(hidden, projected_annotations)                   # [batch, src_len]
        annotations = encoder_outputs.permute(1, 0, 2)                          # [batch, src_len, 2*n]
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)          # c_i
        prev_hidden = hidden
        hidden = self.rnn_cell(embedded, prev_hidden, context)                  # s_i
        readout = self.readout(torch.cat((prev_hidden, embedded, context), dim=1))
        maxout = readout.view(readout.shape[0], self.maxout_dim, 2).max(dim=2).values
        prediction = self.fc_out(maxout)
        return prediction, hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

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


# --- training skeleton ---
# alignment = AdditiveAlignment(hidden_dim, attn_dim)
# model = Seq2Seq(Encoder(input_dim, emb_dim, hidden_dim, dropout),
#                 Decoder(output_dim, emb_dim, hidden_dim, maxout_dim, dropout, alignment),
#                 device).to(device)
# optimizer = optim.Adadelta(model.parameters())
# criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
# for batch in loader:
#     optimizer.zero_grad()
#     output = model(src, trg, teacher_forcing_ratio=0.5)   # [trg_len, batch, vocab]
#     loss = criterion(output[1:].reshape(-1, output.shape[-1]), trg[1:].reshape(-1))
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     optimizer.step()
```

A side benefit of the soft alignment is interpretability: the matrix of weights α_ij is itself a soft word-to-word alignment between source and translation, readable directly off the model.
