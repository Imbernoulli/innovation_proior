**Problem.** At each spatial location the per-variable tokenizer produces a set of `V = 48` tokens in
`R^D` that must be collapsed to one token (`x: [B, V, L, D] -> [B, L, D]`). I want the transparent floor:
the simplest reduction that is correct for the contract, adds no parameters, and that every learned
aggregator must beat to justify its cost.

**Key idea.** Order-invariance over the variable set forces a *symmetric* reduction (sum/max/mean — not
concatenation, not a recurrent reader, not a designated summary token, which the variable set does not
have). The bare sum leaks set scale (`E = Vμ`, std `√V·σ`), which would inflate the pooled token off the
operating point of the pretrained ClimaX backbone and drown the position embedding; dividing by `V` fixes
the expectation at `μ` independent of count, so the **uniform mean** preserves the single-token scale the
frozen backbone expects. The uniform weighting is also the unique permutation-symmetric / maximum-entropy
choice when no variable is privileged, and it is the degenerate limit of both the learned weighted sum
(equal logits → `1/V`) and cross-attention (uniform weights, identity projections) — i.e. the honest
floor those rungs must clear.

**Why mean, not the alternatives.** Sum leaks count into magnitude. Max keeps the per-dimension extreme
variable and is non-smooth — right when one outlier carries the signal, wrong for 48 comparably-informative
meteorological fields where the signal is spread across many; mean uses every variable equally and is
smooth. In the general set-pooling case sets are ragged and the reduction must mask
(`Σ m_v x_v / clamp(Σ m_v, ε)`); **here every location has the full `V` present (no padding)**, so the
masked mean collapses exactly to `x.mean(dim=1)` — the padding apparatus and the `√V` variance-stabilizing
sibling are dropped because the harness exposes a fixed-full variable set.

**Hyperparameters.** None — zero learnable parameters. The contract args `num_heads` and `num_vars` are
stored but unused (no attention; the mean is size-agnostic). `embed_dim = D = 1024`. Reduction is over the
variable axis at every one of `L = 512` locations.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — step 1: mean pooling
class VariableAggregator(nn.Module):
    """Mean pooling variable aggregation.

    Simply averages all V variable tokens at each spatial location.
    No additional learnable parameters.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads (unused).
        num_vars (int): Number of input variables V (unused).
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        # Average across variable dimension
        out = x.mean(dim=1)  # B, L, D
        return out
```
