# Cross-attention variable aggregation, distilled

Variable aggregation reduces a *set* of per-variable tokens at each spatial location to a
single token using a **learnable-query multi-head cross-attention**. Each input physical
variable is tokenized separately (so the variable set can vary across inputs); then, at each
spatial patch independently, one trainable query attends over the `V` variable tokens at that
location and emits one `D`-vector. This makes the model agnostic to how many variables it is
fed, removes the quadratic-in-`V` cost of running self-attention over the full
variable-and-space token sequence, and re-expresses the differently-grounded variable tokens
in one unified representation for the sequence backbone.

## Problem it solves

Geophysical inputs are a stack of `V` heterogeneous physical fields on an `H × W` grid, and
`V` differs across data sources (and across train/finetune/deploy). Stacking the variables as
input channels welds the first layer to a fixed `V` (no pretraining across heterogeneous
sources, no subset/unseen variables) and discards variable identity. Moving the variables onto
a token axis fixes the flexibility but multiplies the sequence length by `V`, making
self-attention cost `O((V·h·w)²)`, and leaves the backbone a sequence of semantically
incommensurable tokens. Variable aggregation collapses the `V` tokens per location to one,
restoring an `h·w`-length, unified-token sequence.

## Key idea

At each spatial location, the per-location reduction must produce a normalized, **content-
dependent** convex combination over the variable set and re-project the tokens into a shared
space — which is exactly attention with the `V` variable tokens as keys/values and a single
**learnable query** (the trainable-query-summarizes-a-set pattern of ViT's `[class]` token,
applied over the variable axis). For query `q` and variable tokens `x_1,…,x_V` at a location,
per head with `d_k = D / num_heads`:

```
score_{r,v} = (W^Q_r q) · (W^K_r x_v) / √d_k
α_{r,v}     = softmax_v(score_{r,1}, …, score_{r,V})   # convex combination over the V-set
out         = W^O · Concat_r( Σ_v α_{r,v} (W^V_r x_v) )
```

- **Set semantics:** the softmax runs over the `V` keys, so the output is permutation-invariant
  in the variables and defined for any `V` — the variable count is read from the input at
  runtime, never baked into a weight.
- **`1/√d_k`:** for unit-variance independent components, `q·k = Σ_{i=1}^{d_k} q_i k_i` has
  variance `d_k`; unscaled, large logits saturate the softmax (near-one-hot, tiny softmax
  Jacobian) and shrink the gradient through the attention weights. Dividing by `√d_k` keeps
  logits unit-variance and the softmax trainable.
- **Multi-head:** several heads keep distinct "which variables matter for *this* aspect"
  weightings separate instead of averaging them into one compromise; with `d_k = D / num_heads` the cost
  matches one full-width head, and `W^O` mixes the heads' subspaces back into one `D`-vector.
- **Single layer, single query:** one query already turns the set into exactly one token per
  location; no stacking needed.
- **Cost:** `O(V)` per location, `O(V·h·w)` total — linear in `V` — and the backbone sequence
  returns to `h·w`, deleting the `O(V²)` factor.

## Why a learnable query (and the degenerate cases)

- **Mean pooling** = this layer with the softmax forced to uniform weights (`softmax(0)`),
  and with identity value/output projections for a raw-token mean: content-blind, weighting
  every variable equally at every location/state.
- **Fixed learned per-variable scalar weights** (softmax over `V`) = this layer with logits
  that are learned constants rather than functions of the token contents: still
  content-independent, only rescales-and-adds raw tokens in its simplest form.
- **Full self-attention over the `V·h·w` sequence** = maximally expressive but `O((V·h·w)²)`.
- The learnable-query cross-attention is the general form whose weights depend on the data and
  whose `W^V` re-projects the tokens; mean and fixed-weight pooling are its degenerate cases.

**Zero-init the query:** with the attention biases initialized to zero, every score is 0 at
the start, so the softmax is uniform and the layer begins with the same equal-weighting
pattern as mean pooling, applied to the projected value tokens. Training then moves the query
toward content-dependent weights.

## Defaults

`embed_dim D = 1024`, `num_heads = 16` (`d_k = 64` per head, the standard head width),
single cross-attention layer, single query token, query initialized to zeros. The number of
variables (`V = 48` in the ERA5 default vocabulary: surface constants, surface fields, and
pressure-level geopotential / wind / temperature / humidity) is supplied at runtime and does
not enter the module's parameters. Spatial sequence length `L = h·w = 16×32 = 512` at 5.625°
(32×64 grid, patch size 2).

## Where it sits in the pipeline

Per-variable patch tokenization → add a learnable per-variable embedding (so tokens stay
identifiable) → **variable aggregation** (this module) → add spatial position embedding →
ViT backbone → prediction head. Trained/evaluated with the latitude-weighted MSE/RMSE: with
`lat(i)` the latitude of grid row `i`, weight `L(i) = cos(lat(i)) / ((1/H) Σ_{i'} cos(lat(i')))`
(weights average to one), and the score is `sqrt(mean over space of L·(pred−truth)²)`,
averaged over forecasts — accounting for equal-degree cells covering more area near the equator.

## Working code

The module is a single `nn.MultiheadAttention` plus one learnable query parameter; the
per-location set is formed by folding batch and space together.

```python
import torch
import torch.nn as nn


class VariableAggregator(nn.Module):
    """Learnable-query cross-attention variable aggregation.

    A single trainable query attends over the V variable tokens at each spatial
    location, producing one token per location.

    Args:
        embed_dim (int): embedding dimension D.
        num_heads (int): number of attention heads.
        num_vars (int):  number of variables V (informational; the set size is read
                         from the input at runtime, so any V is accepted).
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # one learnable query, zero-init -> uniform weights over the projected
        # variable tokens, then learns content-dependent weights
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation per spatial location.
        """
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3)   # [B, L, V, D]
        x = x.reshape(b * l, v, d)  # [B*L, V, D]: each (example, location) is one set

        query = self.var_query.expand(b * l, -1, -1)  # [B*L, 1, D]
        out, _ = self.var_agg(query, x, x)            # [B*L, 1, D]: softmax over the V tokens
        out = out.squeeze(1)                          # [B*L, D]

        out = out.reshape(b, l, d)  # [B, L, D]
        return out
```
