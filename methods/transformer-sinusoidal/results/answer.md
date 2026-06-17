# Sinusoidal Absolute Positional Encoding

A fixed, non-learned scheme for injecting token order into a recurrence-free attention
model. For each position `t` it builds a `d_model`-dimensional vector of sines and
cosines at geometrically spaced frequencies and **adds** it to the token embedding at the
bottom of the stack, so that the otherwise order-blind self-attention layers can recover
position.

## The problem it solves

A stack of self-attention and per-position feed-forward layers is permutation-equivariant:
`softmax(QKᵀ/√d)V` depends only on token *contents*, not on their indices, so permuting the
input permutes the output identically — the model sees a bag of tokens, not a sequence.
Order must be injected externally. A usable position code has to be (1) a fixed-width
`d_model` object addable to embeddings, (2) bounded as `t` grows, (3) unique per position,
(4) such that relative offsets are easy to use and mean the same thing everywhere, and
(5) defined beyond the longest training sequence so the model can run on longer inputs.

## Key idea

Encode position `t` as a vector of paired sinusoids at geometrically decreasing
frequencies and add it to the (scaled) token embedding:

$$
PE_{(t,\,2i)} = \sin\!\big(t \,/\, 10000^{2i/d_{\text{model}}}\big), \qquad
PE_{(t,\,2i+1)} = \cos\!\big(t \,/\, 10000^{2i/d_{\text{model}}}\big),
$$

for `i = 0, …, d_model/2 − 1`. Equivalently, each sin/cos pair uses angular
frequency

$$
\omega_i = 10000^{-2i/d_{\text{model}}}.
$$

The wavelengths form a geometric progression from `2π` (fastest, `i = 0`) to
approximately `10000·2π` (slowest), giving fine-to-coarse position resolution like a
continuous counter while every component remains in `[−1, 1]`.

**Why the sin/cos pair per frequency — the load-bearing property.** Store each frequency
in the source layout, `(\sin \omega t, \cos \omega t)`. A shift `t -> t + k` is then a
fixed orthogonal rotation in this `(sin, cos)` coordinate order. It depends only on the
offset `k`, not on the absolute position `t`:

$$
\begin{bmatrix} \sin ω(t+k) \\ \cos ω(t+k) \end{bmatrix}
=
\begin{bmatrix} \cos ωk & \sin ωk \\ -\sin ωk & \cos ωk \end{bmatrix}
\begin{bmatrix} \sin ωt \\ \cos ωt \end{bmatrix}.
$$

Stacking these 2x2 rotations block-diagonally gives `PE_{t+k} = M_k PE_t` with `M_k`
independent of `t`. For "k positions back" the same formula uses offset `-k`. A single
sinusoid alone cannot support this fixed shift map (from `\sin \omega t` alone the phase
is ambiguous), which is why each frequency carries both sine and cosine.

**Why it can extrapolate.** `\sin`/`\cos` are defined for every real `t`, so position
`L+1` is just the next point on the same curves — not a missing entry like a learned
table of size `L`. The rotation relation holds past training length with the same `M_k`.

**Other choices.** *Add, not concatenate*: a linear layer over a concatenation `[e; p]`
equals separate projections of `e` and `p` summed, so addition gets a linear readout of
position at no extra width. *Scale embeddings by `sqrt(d_model)`* before adding: this is
part of the embedding layer, not the positional formula. The scale makes learned token
vectors numerically comparable to the fixed `O(1)` sinusoidal components at the moment
they are summed. The encoding is computed in log space,
`10000^{-2i/d_model} = exp((2i) * (-log(10000) / d_model))`, stored as a fixed buffer, and
added every forward pass. If a sequence is longer than the buffer, recompute or extend the
same formula; never clamp later positions to the last row.

## Working code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lut(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Fixed sinusoidal absolute positional encoding, added before dropout."

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("sinusoidal positional encoding requires even d_model")
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = self._build(max_len, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("pe", pe.unsqueeze(0))     # [1, max_len, d_model]

    def _build(self, length: int, device, dtype) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)     # omega_i = 10000^{-2i/d_model}
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims:  cos partner
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T <= self.pe.size(1):
            pe = self.pe[:, :T].to(device=x.device, dtype=x.dtype)
        else:
            pe = self._build(T, device=x.device, dtype=x.dtype).unsqueeze(0)
        return self.dropout(x + pe.detach())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class SeqModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers, max_len, dropout=0.1):
        super().__init__()
        self.tok = ScaledEmbedding(d_model, vocab)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.lut.weight          # tied embeddings

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.pos(self.tok(tokens))                  # scaled embeddings + fixed PE
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal)
        return self.head(self.ln_f(x))
```
