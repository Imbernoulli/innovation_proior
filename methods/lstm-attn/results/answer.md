# RNNsearch — additive (Bahdanau) attention, distilled

RNNsearch is an encoder-decoder translator that, instead of compressing the whole source into
one fixed-length vector, keeps every source position's encoder state and lets the decoder
**softly search** the source at each output step. At target step `i` it forms a *distinct*
context `c_i` as a softmax-weighted average of the encoder annotations — a differentiable soft
alignment, jointly trained with the rest of the network. This is the additive-attention
mechanism: a small feedforward "alignment model" scores how well each source position matches
the decoder's current state, softmax turns the scores into alignment weights, and the weighted
sum is the context the decoder reads.

## Problem it solves

The basic RNN encoder-decoder funnels a variable-length source into a single fixed-dimensional
vector `c`, from which the decoder generates the whole translation. A constant-size vector
cannot hold an arbitrarily long sentence, so BLEU degrades sharply with source length, worst
beyond training lengths. The goal: condition each target word directly on the relevant source
content, end to end and differentiable, with no length-dependent bottleneck.

## Key idea

Keep all encoder annotations `h_1..h_{T_x}` and compute a per-step context

```
c_i = sum_{j=1}^{T_x} alpha_ij * h_j          # expected annotation under soft alignment
alpha_ij = exp(e_ij) / sum_k exp(e_ik)        # softmax over source positions
e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)       # additive alignment model (a 1-hidden-layer MLP)
```

- **Soft, not hard, alignment.** A hard cursor (pick one source position) is
  non-differentiable (argmax has no gradient) and linguistically wrong (you must see "man" to
  translate "the" → "l'"). The convex combination `c_i = Σ_j α_ij h_j` is the differentiable
  relaxation — a hard pick is the corner where one `α` is 1 — and reads as the *expected*
  source annotation under alignment probabilities `α_ij`, so alignment becomes a deterministic,
  jointly trained function, not a discrete latent variable.
- **Softmax normalization** makes `c_i` a convex combination of the `h_j`: same scale and
  space regardless of `T_x`, so the context does not grow with sentence length and `α_ij` reads
  as an alignment probability.
- **Additive scorer.** Query `s_{i-1}` (dim `n`) and key `h_j` (dim `2n`) live in different
  spaces, so a raw dot product is not even defined; `W_a`, `U_a` project both into a shared
  `n'`-dim space, `tanh(W_a s + U_a h)` models the match, `v_a` reads off a scalar. One hidden
  layer is the cheapest scorer that can still learn the comparison — and it runs `T_x·T_y`
  times, so cost matters. `U_a h_j` is independent of `i`, so precompute it once per sentence.
- **Query is `s_{i-1}`** (the state before emitting `y_i`): `s_i` depends on `c_i`, so using
  `s_i` would be circular.
- **Bidirectional encoder.** `h_j = [⃗h_j ; ⃖h_j]` concatenates a forward and a backward RNN, so
  each annotation summarizes both the preceding and following words while staying focused near
  position `j` (RNN recency) — a position-tagged key that sees both sides, which a forward-only
  encoder cannot.
- **Content-based, non-monotonic.** Scoring any `(i, j)` pair by content lets `α_i` place mass
  anywhere, including behind where it just looked — needed for reordering (adjective/noun,
  verb-final), which a monotonic location window cannot do. Soft weights also handle fertility
  and source/target length mismatch with no `[NULL]` token.

The plain fixed-vector encoder-decoder is the special case obtained by freezing `c_i` to a
constant (e.g. `c_i = ⃗h_{T_x}` for all `i`).

## Architecture

- **Encoder:** bidirectional gated RNN; annotation `h_j = [⃗h_j ; ⃖h_j] ∈ R^{2n}`.
- **Alignment:** `e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)`, `W_a ∈ R^{n'×n}`,
  `U_a ∈ R^{n'×2n}`, `v_a ∈ R^{n'}`; `α_ij = softmax_j(e_ij)`; `c_i = Σ_j α_ij h_j`.
- **Decoder (gated unit):**
  `z_i = σ(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i)`,
  `r_i = σ(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)`,
  `s̃_i = tanh(W E y_{i-1} + U(r_i ⊙ s_{i-1}) + C c_i)`,
  `s_i = (1 − z_i) ⊙ s_{i-1} + z_i ⊙ s̃_i`, with `s_0 = tanh(W_s ⃖h_1)`.
- **Deep output (maxout):** `t̃_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i`,
  `t_{i,j} = max(t̃_{i,2j-1}, t̃_{i,2j})` for `j = 1..l`, `p(y_i | ·) ∝ exp(y_i^T W_o t_i)`.
- **Sizes:** `n = 1000` hidden, `m = 620` embedding, `l = 500` maxout, `n' = 1000` alignment.

## Training and decoding

Maximize conditional log-likelihood `(1/N) Σ_n log p_θ(y_n | x_n)` (masked cross-entropy over
target positions); gradients flow through the alignment model since it is fully differentiable.
Gradient-norm clipping at threshold 1 (Pascanu et al. 2013); minibatch SGD with Adadelta
(ρ=0.95, ε=1e-6), minibatch of 80 sentence pairs grouped by length. Init: recurrent matrices
orthogonal; `W_a, U_a ~ N(0, 0.001²)`; `v_a` and biases zero; other weights `N(0, 0.01²)`.
Decode with left-to-right beam search. The alignment matrix `[α_ij]` is directly inspectable as
a soft source-target alignment.

## Working code

The source-conditioning slot of the encoder-decoder harness is filled by the additive
soft-search; the decoder is a gated cell consuming `[E y_{i-1} ; c_i]` with a maxout deep
output.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRUEncoder(nn.Module):
    """Bidirectional gated encoder: h_j = [forward_j ; backward_j]."""

    def __init__(self, vocab, emb_dim, hid_dim, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.birnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)

    def forward(self, src):                       # src: [B, T_x]
        ann, _ = self.birnn(self.embed(src))      # ann: [B, T_x, 2*hid_dim]
        return ann


class AdditiveAlignment(nn.Module):
    """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j); alpha = softmax_j(e); c_i = sum_j alpha_ij h_j."""

    def __init__(self, dec_dim, ann_dim, align_dim):
        super().__init__()
        self.W_a = nn.Linear(dec_dim, align_dim, bias=False)   # query projection
        self.U_a = nn.Linear(ann_dim, align_dim, bias=False)   # key projection
        self.v_a = nn.Linear(align_dim, 1, bias=False)         # scalar score

    def precompute_keys(self, ann):               # U_a h_j is i-independent: compute once
        return self.U_a(ann)                      # [B, T_x, align_dim]

    def forward(self, s_prev, ann, keys, mask):   # s_prev = s_{i-1}: [B, dec_dim]
        e = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + keys)).squeeze(-1)  # [B, T_x]
        e = e.masked_fill(~mask, float('-inf'))   # ignore padding
        alpha = F.softmax(e, dim=-1)              # distribution over source positions
        c = torch.bmm(alpha.unsqueeze(1), ann).squeeze(1)      # c_i
        return c, alpha


class RNNsearchGRUCell(nn.Module):
    """s_i = (1-z_i) * s_{i-1} + z_i * s_tilde_i, with c_i in every gate."""

    def __init__(self, emb_dim, ann_dim, dec_dim):
        super().__init__()
        self.W_z = nn.Linear(emb_dim, dec_dim, bias=False)
        self.U_z = nn.Linear(dec_dim, dec_dim, bias=False)
        self.C_z = nn.Linear(ann_dim, dec_dim, bias=False)
        self.W_r = nn.Linear(emb_dim, dec_dim, bias=False)
        self.U_r = nn.Linear(dec_dim, dec_dim, bias=False)
        self.C_r = nn.Linear(ann_dim, dec_dim, bias=False)
        self.W = nn.Linear(emb_dim, dec_dim, bias=False)
        self.U = nn.Linear(dec_dim, dec_dim, bias=False)
        self.C = nn.Linear(ann_dim, dec_dim, bias=False)

    def forward(self, y_emb, c, s_prev):
        z = torch.sigmoid(self.W_z(y_emb) + self.U_z(s_prev) + self.C_z(c))
        r = torch.sigmoid(self.W_r(y_emb) + self.U_r(s_prev) + self.C_r(c))
        s_tilde = torch.tanh(self.W(y_emb) + self.U(r * s_prev) + self.C(c))
        return (1.0 - z) * s_prev + z * s_tilde


class AttnDecoder(nn.Module):
    """Gated decoder conditioned on the per-step context, with a maxout deep output."""

    def __init__(self, vocab, emb_dim, dec_dim, ann_dim, align_dim, maxout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.align = AdditiveAlignment(dec_dim, ann_dim, align_dim)
        self.cell = RNNsearchGRUCell(emb_dim, ann_dim, dec_dim)
        self.init_s = nn.Linear(ann_dim // 2, dec_dim)         # s_0 = tanh(W_s backward_1)
        self.U_o = nn.Linear(dec_dim, 2 * maxout, bias=False)
        self.V_o = nn.Linear(emb_dim, 2 * maxout, bias=False)
        self.C_o = nn.Linear(ann_dim, 2 * maxout, bias=False)
        self.W_o = nn.Linear(maxout, vocab)

    def forward(self, ann, mask, tgt_in):         # tgt_in: [B, T_y] teacher-forced prev tokens
        keys = self.align.precompute_keys(ann)
        backward_1 = ann[:, 0, ann.size(-1) // 2:]            # backward annotation at pos 1
        s = torch.tanh(self.init_s(backward_1))              # s_0
        logits = []
        for i in range(tgt_in.size(1)):
            y_emb = self.embed(tgt_in[:, i])                 # E y_{i-1}
            s_prev = s
            c, _ = self.align(s_prev, ann, keys, mask)       # query = s_{i-1} (pre-step state)
            s = self.cell(y_emb, c, s_prev)
            t_tilde = self.U_o(s_prev) + self.V_o(y_emb) + self.C_o(c)
            t = t_tilde.view(t_tilde.size(0), -1, 2).max(dim=-1).values   # maxout
            logits.append(self.W_o(t))
        return torch.stack(logits, dim=1)                    # [B, T_y, vocab]


class RNNSearch(nn.Module):
    def __init__(self, vocab, emb_dim=620, hid=1000, align_dim=1000, maxout=500, pad_id=0):
        super().__init__()
        self.encoder = BiGRUEncoder(vocab, emb_dim, hid, pad_id)
        self.decoder = AttnDecoder(vocab, emb_dim, hid, 2 * hid, align_dim, maxout, pad_id)
        self.pad_id = pad_id

    def forward(self, src, tgt_in):
        ann = self.encoder(src)                              # annotations
        mask = src.ne(self.pad_id)
        return self.decoder(ann, mask, tgt_in)               # next-token logits


def loss_fn(logits, target, pad_id):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target.reshape(-1), ignore_index=pad_id)
```
