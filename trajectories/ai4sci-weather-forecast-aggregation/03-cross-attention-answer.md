**Problem.** The learned weighted sum (345.79 / 2.5934 / 3.4245) beat the mean on z500-3day and t850-5day
but *regressed* on wind10m-7day (3.4245 > the mean's 3.3991): a single global per-variable distribution is
one compromise weighting forced to serve all three targets, and it cannot react to the local atmospheric
state. The fix is a reduction whose weights depend on the *token contents* at each location, and which
re-expresses the differently-grounded variable tokens in a shared space rather than rescaling-and-adding
raw ones.

**Key idea.** "Normalized, data-dependent convex combination over a set, with re-projected values" is
exactly attention. Keys/values are the `V` variable tokens at a location (with learned projections, so
`W^V x_v` re-expresses them); the query is a single **learnable** vector reused at every location (the
trainable-query-summarizes-a-set move of ViT's class token, pointed at the variable axis). Per head with
`d_k = D/num_heads`: `α_v = softmax_v((W^Q q)·(W^K x_v)/√d_k)`, head output `Σ_v α_v (W^V x_v)`, heads
concatenated through `W^O`. The softmax over the `V` keys gives set semantics (permutation-invariant, any
`V`); `1/√d_k` keeps the softmax responsive (the very property the frozen-logit weighted sum lacked);
multi-head keeps several "which variables matter for *this* aspect" patterns distinct instead of forcing
the one compromise that sank wind10m. One layer, one query suffice (one query already emits one token per
location). Cost is `O(V)` per location — linear in `V`, and the backbone still sees only the `h·w`-length
sequence, so no `O(V²)` blowup.

**Why this and the degenerate cases.** The mean = this layer with the softmax forced uniform (`softmax(0)`)
and identity projections; the learned weighted sum = this layer with logits that are learned *constants*
rather than functions of the tokens; cross-attention is the general form with content-dependent weights and
re-projected values. The two lower rungs are its special cases — which is why each fixed the previous one's
exposed defect. **Zero-init the query:** every score is 0 at step 0, so the softmax is uniform — fine-tuning
starts at the mean (the safe prior the pretrained ClimaX backbone expects) and learns content-dependence
from there.

**Hyperparameters.** One `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)` + one learnable
query `var_query ∈ R^{1×1×D}`, **zero-initialized**. `embed_dim = D = 1024`, `num_heads = 16`
(`d_k = 64`/head), single layer, single query. `V = 48` is read from the input shape, never entering the
module's weights; reduction runs at every `L = 512` location via folding `B` and `L` together.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — step 3: ClimaX cross-attention
class VariableAggregator(nn.Module):
    """Cross-attention variable aggregation (ClimaX default).

    A learnable query token attends to all V variable tokens at each spatial
    location via multi-head cross-attention, producing one token per location.

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
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3)   # B, L, V, D
        x = x.reshape(b * l, v, d)  # B*L, V, D

        query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
        out, _ = self.var_agg(query, x, x)             # B*L, 1, D
        out = out.squeeze(1)                            # B*L, D

        out = out.reshape(b, l, d)  # B, L, D
        return out
```
