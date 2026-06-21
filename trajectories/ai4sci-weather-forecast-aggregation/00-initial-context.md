## Research question

A weather forecasting model tokenizes each input physical field independently, producing per-variable patch embeddings `x: [B, V, L, D]`. Before the ViT backbone runs, these `V` tokens at each of `L` spatial locations must collapse into **one** token per location, `[B, L, D]`. The only editable piece is the `VariableAggregator` that performs this reduction. Everything else — per-variable tokenization, the 8-block ViT, the ERA5 data pipeline, the fine-tuning recipe from pretrained ClimaX weights, the optimizer/schedule, and the latitude-weighted RMSE metric — is fixed. The question is: how should information be aggregated across the heterogeneous variable set at each location?

## Prior art / Background / Baselines

The standard image-model approach treats the `V` variables as input channels of a single field; the first projection welds its input dimension to exactly `V`. This is fast but hard-wires the variable set and destroys variable identity in the first layer.

ClimaX instead tokenizes each variable's `H × W` map separately and adds a learnable per-variable embedding, so the variable set becomes a runtime argument. This makes the backbone sequence `V` times longer, so self-attention costs `O((V·h·w)²)`, and the sequence mixes geopotential, humidity, wind, and surface constants. The remaining task is to reduce those `V` tokens at each location to one token before the backbone.

The baselines are three set-to-vector reductions:

- **Mean pooling.** Take the uniform average over the `V` tokens at each location. It preserves scale and needs no parameters, but it gives equal weight to every variable everywhere, diluting informative fields with near-inert ones.
- **Learned weighted sum.** Learn one scalar weight per variable and combine by softmax. It beats uniform weighting where a fixed global split helps, but a single global weighting must serve every target and every location; it cannot react to the local state and regresses on targets where the right variables differ.
- **Cross-attention (ClimaX default).** A single learned query attends to the `V` variable tokens with content-dependent multi-head attention. The weight assigned to one variable at a location is unaffected by the values of the other variables at that same location.

## Fixed substrate / Code framework

A ClimaX forecasting pipeline is frozen and must not be touched. Per-variable patch tokenization produces, for each of `L = h·w` spatial patches, one `D`-dimensional token per input variable (`x: [B, V, L, D]`); a learnable per-variable embedding is added so tokens stay identifiable. The aggregator output (`[B, L, D]`) receives a spatial position embedding and a lead-time embedding, then passes through an 8-block ViT backbone (`embed_dim D = 1024`, `num_heads = 16`, `mlp_ratio = 4`) and a linear prediction head. Fine-tuning starts from pretrained ClimaX weights on ERA5 reanalysis at 5.625° (a 32×64 grid, patch size 2, so `L = 16×32 = 512`), using latitude-weighted MSE as the training loss. The variable vocabulary has `V = 48`: 3 surface constants, 3 surface fields, and 42 pressure-level fields.

## Editable interface

Exactly one region is editable — the `VariableAggregator` class in `custom_forecast.py` (the block at lines 310–351 the harness replaces from `edits/<baseline>.edit.py`). The contract is fixed:

- `__init__(self, embed_dim, num_heads, num_vars)` — `embed_dim = D = 1024`, `num_heads = 16`, `num_vars = V = 48`. The module may allocate any parameters here.
- `forward(self, x)` — `x: [B, V, L, D]` → returns `[B, L, D]`. The reduction is over the variable axis `V`, performed independently at every spatial location `L`. `V` is read from the input shape at runtime; the module must accept any `V`.

The FIXED section imports `torch`, `torch.nn as nn`, and `torch.nn.functional as F`, and standard PyTorch modules are available. The scaffold default is the ClimaX learnable-query cross-attention:

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — default fill: ClimaX cross-attention
class VariableAggregator(nn.Module):
    """Aggregates per-variable patch embeddings into a unified representation.

    Default: learnable query with single-layer cross-attention (ClimaX default).
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # Learnable query token for cross-attention aggregation
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        b, v, l, d = x.shape
        # Reshape to treat each spatial location independently
        x = x.permute(0, 2, 1, 3)   # B, L, V, D
        x = x.reshape(b * l, v, d)  # B*L, V, D

        # Cross-attention: query attends to all variable tokens
        query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
        out, _ = self.var_agg(query, x, x)             # B*L, 1, D
        out = out.squeeze(1)                            # B*L, D

        out = out.reshape(b, l, d)  # B, L, D
        return out
```

## Evaluation settings

The model is fine-tuned from pretrained ClimaX weights on ERA5 at 5.625° and evaluated on three forecasting targets at increasing lead times: **z500-3day** (500 hPa geopotential height, 3-day lead), **t850-5day** (850 hPa temperature, 5-day lead), and **wind10m-7day** (10 m wind speed, 7-day lead), under seed 42. The metric on every target is **latitude-weighted RMSE — lower is better**. Equal-degree grid cells cover more area near the equator than near the poles, so each row's squared error is weighted by `L(i) = cos(lat(i)) / ((1/H) Σ_{i'} cos(lat(i')))` (weights average to one), and the score is the square root of the latitude-weighted mean squared error, averaged over forecasts. The same weighting defines the training loss.
