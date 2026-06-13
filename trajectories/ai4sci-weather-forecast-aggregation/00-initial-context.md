## Research question

A weather forecasting model has tokenized each input physical field independently: for every spatial
patch it holds one `D`-dimensional token per meteorological variable. Before the Vision-Transformer
backbone runs, these per-variable tokens must be collapsed into **one** token per spatial location. The
single thing being designed is that reduction — the `VariableAggregator`: it receives per-variable patch
embeddings `x: [B, V, L, D]` and must return `[B, L, D]`, one aggregated representation per location.
Everything else — the per-variable tokenization, the ViT backbone, the ERA5 data pipeline, the fine-tuning
recipe from pretrained ClimaX weights, the optimizer/schedule, and the latitude-weighted RMSE metric — is
fixed. The question is purely: *how should information be aggregated across the heterogeneous variable set
at each location?*

## Prior art before the first rung

The first rung reacts to how every standard image model ingests a stack of `V` physical fields, and to
the design that made a single weather/climate foundation model possible.

- **Channel-stacking image models (ResNet/UNet/ViT over `V × H × W`; Rasp & Thuerey 2021, Pathak et al.
  2022, Bi et al. 2022).** Treat the `V` variables as input channels; the first conv/patch projection has
  its input dimension welded to exactly `V`. Strong on a fixed variable set. **Gap:** the variable set is
  hard-wired — no pretraining across sources with different variables, no subset/unseen variables at
  finetune time — and variable identity is destroyed the instant the first layer sums the channels.
- **Per-variable tokenization (the ClimaX move; Nguyen et al. 2023).** Give each variable its own tokens
  by tokenizing its `H × W` map separately, with a learnable per-variable embedding so tokens stay
  identifiable. The variable set becomes a runtime argument. **Gap:** this multiplies the sequence length
  by `V`, so backbone self-attention costs `O((V·h·w)²)` — quadratic in the variable count — and leaves
  the backbone a sequence of semantically incommensurable tokens (geopotential, humidity, wind, a
  land-sea mask all interleaved). Both pathologies reduce to one need: collapse the `V` variable tokens
  at each location to a single unified token *before* the backbone. That collapse is the editable slot.
- **The set-pooling lineage the slot inherits.** Once variables are a set of tokens at each location, the
  reduction is a set-to-vector map. The cheapest is uniform mean (parameter-free, content-blind); a step
  up is a fixed learned per-variable weighting (content-independent); the ClimaX default is a
  learnable-query cross-attention (content-dependent). These are the rungs of this ladder.

## The fixed substrate

A ClimaX forecasting pipeline is frozen and must not be touched. Per-variable patch tokenization
produces, for each of `L = h·w` spatial patches, one `D`-dimensional token per input variable
(`x: [B, V, L, D]`); a learnable per-variable embedding is added so the tokens stay identifiable. The
output of the aggregator (`[B, L, D]`) then receives a spatial position embedding and a lead-time
embedding, and is processed by an 8-block ViT backbone (`embed_dim D = 1024`, `num_heads = 16`,
`mlp_ratio = 4`) and a linear prediction head. The model is fine-tuned from pretrained ClimaX weights on
ERA5 reanalysis at 5.625° (a 32×64 grid, patch size 2, so `L = 16×32 = 512`), with the
latitude-weighted MSE as the training loss. The variable vocabulary has `V = 48`: 3 surface constants
(land-sea mask, orography, latitude), 3 surface fields (2 m temperature, 10 m wind u/v), and 42
pressure-level fields (geopotential, u/v wind, temperature, relative/specific humidity at 50–925 hPa).

## The editable interface

Exactly one region is editable — the `VariableAggregator` class in `custom_forecast.py` (the block at
lines 310–351 the harness replaces from `edits/<baseline>.edit.py`). The contract is fixed:

- `__init__(self, embed_dim, num_heads, num_vars)` — `embed_dim = D = 1024`, `num_heads = 16`,
  `num_vars = V = 48`. The module may allocate any parameters here.
- `forward(self, x)` — `x: [B, V, L, D]` → returns `[B, L, D]`. The reduction is over the variable axis
  `V`, performed independently at every spatial location `L` (and every example `B`). `V` is read from
  the input shape at runtime; the module must accept any `V`.

The FIXED section imports `torch`, `torch.nn as nn`, and `torch.nn.functional as F`, and standard
PyTorch modules (`nn.Linear`, `nn.MultiheadAttention`, `nn.LayerNorm`, …) are available. Every method on
the ladder is a fill of this same contract. The scaffold default is the ClimaX learnable-query
cross-attention:

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

The model is fine-tuned from pretrained ClimaX weights on ERA5 at 5.625° and evaluated on three
forecasting targets at increasing lead times: **z500-3day** (500 hPa geopotential height, 3-day lead),
**t850-5day** (850 hPa temperature, 5-day lead), and **wind10m-7day** (10 m wind speed, 7-day lead),
under seed 42. The metric on every target is **latitude-weighted RMSE — lower is better**. Equal-degree
grid cells cover more area near the equator than near the poles, so each row's squared error is weighted
by `L(i) = cos(lat(i)) / ((1/H) Σ_{i'} cos(lat(i')))` (weights average to one), and the score is the
square root of the latitude-weighted mean squared error, averaged over forecasts. The same weighting
defines the training loss.
