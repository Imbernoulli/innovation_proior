# Set Transformer

## Problem

Build a neural network for **set-structured input** that is (1) **permutation invariant** — its output is unchanged when the input elements are reordered — and (2) accepts **any number of elements**, while being expressive enough to model the **interactions among elements** during computation. The standard set-pooling baseline `ρ(sum(φ(·)))` encodes each element independently with `φ` and aggregates with a fixed symmetric reduction, so it discards inter-element interactions and uses a content-independent aggregation, which under-fits interaction-heavy tasks like amortized clustering.

## Key idea

Use **attention** as the set primitive: attention is permutation-*invariant* in its keys/values and permutation-*equivariant* in its queries. Build the encoder out of **self-attention** so elements interact, make it cheap on large sets with **inducing points**, and replace fixed pooling with a **learnable attention pooling**.

## Building blocks

- **MAB (Multihead Attention Block)** — a Transformer encoder block without positional encoding or dropout (positions would break permutation invariance). For sets `X` (queries) and `Y` (keys/values):
  `H = LayerNorm(X + Multihead(X, Y, Y; ω))`,  `MAB(X, Y) = LayerNorm(H + rFF(H))`,
  with scaled-softmax attention `ω(·) = softmax(·/√d)`. Equivariant in `X`, invariant in `Y`.
- **SAB (Set Attention Block)** = `MAB(X, X)`: self-attention within the set; encodes pairwise interactions, stack for higher-order. Cost `O(n²)`. Permutation equivariant.
- **ISAB (Induced Set Attention Block)** — `m` trainable **inducing points** `I ∈ R^{m×d}` give a low-rank bottleneck:
  `H = MAB(I, X) ∈ R^{m×d}`  (inducing points summarize the set, invariant in `X`),  `ISAB_m(X) = MAB(X, H) ∈ R^{n×d}`  (set reads back, equivariant in `X`). Cost `O(nm)`.
- **PMA (Pooling by Multihead Attention)** — `k` trainable **seed** vectors `S ∈ R^{k×d}` attend over the encoded set: `PMA_k(Z) = MAB(S, rFF(Z))`, a learnable content-dependent pooling producing `k` vectors. Use `k=1` usually; for `k` correlated outputs (e.g. clustering) follow with `SAB(PMA_k(Z))` to model explaining-away among them.

## Architecture

`Encoder(X) = SAB(SAB(X))` (small `n`) or `ISAB_m(ISAB_m(X))` (large `n`); `ℓ` blocks cost `O(ℓn²)` or `O(ℓnm)`.
`Decoder(Z) = rFF(SAB(PMA_k(Z)))`, with `PMA_k(Z) = MAB(S, rFF(Z))`. Encoder equivariant + PMA invariant ⇒ the model is permutation invariant.

## Properties

- **Permutation invariance:** equivariant encoder composed with invariant PMA decoder.
- **Universal approximation of permutation-invariant functions.** Mean is softmax attention with a zero query (`softmax(0)` is uniform); the decoder can express power-means `((1/n)Σ z_i^p)^{1/p}` (front/back `rFF` realize `z↦z^p`, `z↦z^{1/p}`, `h=d` one-dim heads each compute a mean); PMA can express sum-pooling (seed `s=0`, `ω = 1+f` with `f(0)=0` ⇒ all weights `1`). Setting `W^O = 0` in every SAB/ISAB collapses the encoder to instance-wise `rFF(X)`; then the decoder realizes `rFF(sum(rFF(·)))`, which is universal (Zaheer et al. 2017). Attention is not needed for universality but is what makes interaction-heavy tasks learnable in practice.

**Defaults (canonical implementation):** hidden dim 64–256, `num_heads = 4` (8 for the counting task), `num_inds = 16/32/64`, `num_seeds = k` (1 for classification/max; 4–6 for clustering), Adam lr `10⁻³`–`10⁻⁴` with decay. rFF = a single `Linear`+ReLU residual; the leading `rFF(Z)` inside the decoder's PMA is often dropped since the previous block already ends in a feed-forward.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_in, num_outputs, dim_out,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_out))
    def forward(self, X):
        return self.dec(self.enc(X))
```
