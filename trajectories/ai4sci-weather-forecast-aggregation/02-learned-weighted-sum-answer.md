**Problem.** The mean (353.50 / 2.6032 / 3.3991 on z500-3day / t850-5day / wind10m-7day) weights all
`V = 48` variables identically at every location and state, diluting the few fields that actually drive a
target (mid-tropospheric geopotential and wind for z500) by averaging in near-inert ones (land-sea mask,
orography). I want to buy back exactly one degree of freedom — a *global* per-variable importance — as
cheaply as possible, before paying for content-dependent mixing.

**Key idea.** Combine by a weighted sum `O = Σ_v w_v x_v`, learn the weights, but constrain them to the
simplex so only the *relative* split is free, not the scale. Per-variable **scalar** weights suffice (one
number per variable = `V = 48` params, negligible). Free weights are homogeneous in `w`, so the overall
gain runs away and re-creates the scale-leak that would knock the pretrained ClimaX backbone off the
operating point the mean carefully preserved; **normalizing onto the simplex** (`w_v ≥ 0`, `Σ w_v = 1`)
makes `O` a convex combination — same single-token scale as the mean, no arbitrary gain — that reduces to
the mean at the uniform point. **Softmax** `w_v = e^{a_v}/Σ_j e^{a_j}` maps unconstrained raw params onto
that simplex, is smooth and differentiable so plain Adam trains it with no projection step, and reads as a
distribution over the 48 variables.

**Why this and not the alternatives.** vs. the mean: it is the uniform point of this same convex family;
the learned split lets the model leave it where the data rewards. vs. unbounded learnable weights: the
simplex removes the runaway scale (downstream-operating-point break + unbounded-gain instability) while
keeping all relative-importance expressiveness. vs. cross-attention (the rung above): that recomputes a
*content-* and *location-dependent* mixing per location with QKV projections and a per-location attention
computation; this uses a single global per-variable distribution (`V` params, no projections, no attention
matrix) — most of the benefit if the global split is most of the signal, at almost none of the cost. I keep
only the softmax form (not the ReLU fast-normalized detector sibling, not a per-channel weight).

**Hyperparameters.** One learnable vector `var_weights ∈ R^V` (`V = 48`), **zero-initialized** so softmax
gives `1/V` at init — i.e. fine-tuning starts as the measured mean and climbs. `embed_dim = D = 1024`,
`num_heads` stored but unused (no attention). Softmax taken over the variable axis; broadcast-multiply and
sum over `V` at every `L = 512` location.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — step 2: learned weighted sum
class VariableAggregator(nn.Module):
    """Learned weighted sum variable aggregation.

    Learns a scalar weight per variable, applies softmax normalization,
    then computes a weighted sum across variable tokens.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads (unused).
        num_vars (int): Number of input variables V.
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # Learnable weight per variable
        self.var_weights = nn.Parameter(torch.zeros(num_vars), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        # Softmax-normalized variable weights
        w = F.softmax(self.var_weights, dim=0)  # V
        w = w.view(1, -1, 1, 1)                # 1, V, 1, 1
        # Weighted sum across variables
        out = (x * w).sum(dim=1)  # B, L, D
        return out
```
