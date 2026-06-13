**Problem.** The single-query cross-attention (255.85 / 2.1214 / 3.1455) scores each variable token
*independently* against one fixed query: the weight on the wind token cannot depend on what the
geopotential token says. But the right combination of meteorological variables at a cell is *relational*,
and at the longest lead (wind10m-7day, smallest residual headroom 3.1455) the informative combination is
the most state-coupled. The variables must attend to *each other* before being pooled.

**Key idea.** Set Transformer pooling (Lee et al., ICML 2019): a **SAB** lets the `V` variable tokens
self-attend (so each is represented in the context of the others — the inter-variable interaction every
lower rung lacks), then **PMA** with one learnable seed cross-attends (multihead) over the context-aware
tokens to return one summary. Built from one block, MAB(X,Y): `H = LayerNorm(X + Multihead(X,Y,Y))`,
`MAB = LayerNorm(H + rFF(H))`, no positional encoding, no dropout, `1/√d` scaling. SAB(X) = MAB(X,X)
(permutation-equivariant; the residual carries each token forward, so identity-collapse is not the cheap
optimum); PMA₁(Z) = MAB(S, Z) with one seed S
(permutation-invariant). Equivariant SAB ∘ invariant PMA ⇒ invariant aggregator.

**Why this over cross-attention.** PMA with k=1 seed and single head on raw tokens ≈ the ClimaX
cross-attention; this adds (i) multihead pooling and (ii) — decisively — the SAB encoder so variables read
each other before pooling, modelling correlations the independent-scoring query cannot. SAB is `O(V²)` per
location — trivial at `V = 48`, and does not touch the backbone's `h·w` sequence (so no `O((V·h·w)²)`
blowup). Full SAB (not ISAB's inducing-point bottleneck, which is for large sets); one seed (not k>1, which
is for several correlated outputs). LayerNorm on for stable fine-tuning; the leading `rFF(Z)` of canonical
PMA dropped since SAB already ends in a feed-forward.

**Hyperparameters.** `embed_dim = D = 1024`, `num_heads = 16` (`d_k = 64`/head), one SAB block, one PMA
seed (Xavier-initialized), `ln=True`. `V = 48` read from the input shape (every `Linear` is `D→D`); reduction
at every `L = 512` location via folding `B` and `L` together.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — finale: Set Transformer pooling (SAB + PMA)
import math


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, eqs. 6-7):
        H        = LayerNorm(X + Multihead(X, Y, Y))
        MAB(X,Y) = LayerNorm(H + rFF(H)),   rFF = row-wise Linear+ReLU residual.
    No positional encoding, no dropout (both break the set symmetry).
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
    """Set Transformer pooling: SAB over the V variable tokens, then PMA with one seed.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads.
        num_vars (int): Number of input variables V.
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
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * l, v, d)   # B*L, V, D
        z = self.sab(x, x)                               # B*L, V, D: variables read each other
        s = self.seed.expand(b * l, -1, -1)              # B*L, 1, D: the shared learnable seed
        out = self.pma(s, z)                             # B*L, 1, D: seed pools the encoded set
        out = out.squeeze(1).reshape(b, l, d)            # B, L, D
        return out
```
