# Set Transformer pooling for variable aggregation (SAB + PMA), distilled

Aggregate the `V` per-variable tokens at each spatial location with the **Set Transformer** (Lee et al.,
ICML 2019) pooling: a **Set Attention Block (SAB)** lets the variable tokens attend to one another, then
**Pooling by Multihead Attention (PMA)** with one learnable seed summarizes the context-aware tokens into a
single vector. This is the "SAB + PMA" architecture restricted to one encoder block and one seed — the
minimal form that adds, over the single-query cross-attention baseline, the two levers it lacks: multihead
attention pooling and, decisively, *self-attention among the variables before pooling*, so inter-variable
correlations are modelled.

## Problem it solves

At each spatial location, reduce a set of `V` variable tokens (`x: [B, V, L, D]`) to one token
(`[B, L, D]`), permutation-invariant in the variables and accepting any `V`. The existing reductions —
uniform mean, learned weighted sum, single-query cross-attention — all score/encode each variable *in
isolation*: the pooling weight on variable `i` is decided without reference to the other variables'
contents. But the right combination of meteorological variables at a cell is *relational* (a wind token
matters only when the geopotential token is in a certain regime). The goal is a pooling that lets the
variables interact before being summarized.

## Key idea

Let the `V` variable tokens attend among themselves first (so they become context-aware), then pool with a
learnable seed query.

- **SAB(X) = MAB(X, X)** — self-attention within the variable set. Permutation-equivariant; mixes every
  variable's representation with every other's by content compatibility; the residual carries each token
  forward, so identity-collapse is not the cheap optimum. This
  is the inter-variable interaction the lower rungs lack. Cost `O(V²)` per location — trivial at `V = 48`,
  and does not touch the backbone's `h·w` sequence.
- **PMA₁(Z) = MAB(S, Z)** — one learnable seed `S ∈ R^{1×D}` cross-attends (multihead) over the encoded
  set, producing one summary token. Permutation-invariant in `Z`, content-dependent, multihead so it can
  pose several summary sub-questions. Composing equivariant SAB with invariant PMA ⇒ the whole aggregator
  is permutation-invariant.
- **MAB (the shared block):** `H = LayerNorm(X + Multihead(X, Y, Y))`, `MAB(X,Y) = LayerNorm(H + rFF(H))`,
  no positional encoding, no dropout (both would break the set symmetry/determinism), `1/√d` scaling to
  keep the softmax responsive.

## Why this and not the alternatives

- **vs. single-query cross-attention (ClimaX default):** by the Set Transformer's own analysis, PMA with
  `k=1` seed and single-head attention on raw tokens ≈ that cross-attention. This adds (i) multihead
  pooling and (ii) an SAB encoder so the variables read each other before pooling — modelling
  inter-variable correlations the single query over raw tokens cannot.
- **vs. learned weighted sum / mean:** those are the Deep-Sets skeleton with element-wise encoding and a
  fixed (global or uniform) reduction — no interaction among elements; under-fit relational structure.
- **vs. ISAB encoder:** ISAB's inducing-point bottleneck is for *large* sets where `O(V²)` hurts; with
  `V = 48` the full SAB is cheap, so no bottleneck is needed.
- **vs. `k > 1` seeds + trailing SAB:** multiple seeds (and the SAB among them) are for several correlated
  outputs (e.g. clustering); one summary token per location needs `k = 1`.

## Defaults

`embed_dim = D = 1024`, `num_heads = 16` (`d_k = 64`/head), one SAB block, one PMA seed,
LayerNorm on (`ln=True`, stable fine-tuning inside a deep ViT), seed Xavier-initialized, the leading
`rFF(Z)` of canonical PMA dropped since SAB already ends in a feed-forward. `V` read from the input shape;
every `Linear` is `D→D`, so any `V` is accepted.

## Working code

MAB faithful to the canonical Set Transformer reference (head-split stacked along the batch dim; residual
`Q_ + A V_`; LayerNorm; `O + ReLU(fc_o(O))`; LayerNorm). The aggregator is `SAB` then `PMA₁`.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, Lee et al. 2019, eqs. 6-7):
        H        = LayerNorm(X + Multihead(X, Y, Y))
        MAB(X,Y) = LayerNorm(H + rFF(H)),   rFF = row-wise Linear+ReLU residual.
    No positional encoding, no dropout (both would break the set symmetry).
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
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
        O = O if self.ln0 is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if self.ln1 is None else self.ln1(O)
        return O


class VariableAggregator(nn.Module):
    """Set Transformer pooling over the V variable tokens at each spatial location.

    SAB (self-attention among the variables) then PMA with one learnable seed (k=1).

    Input  x: [B, V, L, D]
    Output:   [B, L, D]
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # SAB encoder: variables attend to each other (MAB(X, X)).
        self.sab = MAB(embed_dim, embed_dim, embed_dim, num_heads, ln=True)
        # PMA pooling: one learnable seed cross-attends over the encoded set (MAB(S, Z)).
        self.seed = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.seed)
        self.pma = MAB(embed_dim, embed_dim, embed_dim, num_heads, ln=True)

    def forward(self, x):
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * l, v, d)   # [B*L, V, D]: each (ex, loc) is one set
        z = self.sab(x, x)                               # [B*L, V, D]: variables read each other
        s = self.seed.expand(b * l, -1, -1)              # [B*L, 1, D]: the shared learnable seed
        out = self.pma(s, z)                             # [B*L, 1, D]: seed pools the encoded set
        out = out.squeeze(1).reshape(b, l, d)            # [B, L, D]: one token per location
        return out
```
