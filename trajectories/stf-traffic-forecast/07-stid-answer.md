**Problem.** The ladder ends in a standoff: no single model wins all three datasets. TimesNet owns flow
(explicit 2D *temporal* period modeling), SOFTS and iTransformer own the speed grids (explicit *cross-node*
modeling) — each rung built one explicit axis and let a generic operator carry the other, losing the
dataset where the neglected axis dominates. Strong spatial-temporal forecasting needs both axes modeled
explicitly at once.

**Key idea.** Re-read what the expensive structures were *doing*: graph/attention let the model tell *which
sensor* a sample is; spectral/2D-period cells let it tell *what time regime* it is. Both are mechanisms for
making samples **distinguishable**. A plain channel-shared MLP fails because identical histories from
different sensors, or at different times, have divergent futures yet get one averaged continuation. So buy
distinguishability directly, for almost nothing:

- **Spatial identity** — a learnable `N × h` table, one vector per node, attached to its representation.
  Sensor-to-sensor structure lives in the embedding geometry (no adjacency, no Laplacian, no `O(N²)`).
- **Temporal identities** — learnable time-in-day (288 slots) and day-in-week (7) tables, looked up from
  the *current* step's timestamps (which every prior rung ignored). Each regime gets a learned signature a
  raw scalar cannot express.
- **Time-series embedding** — a shared `Linear(input_len, h)` of each node's own history.

Concatenate the four (`4h`), encode with residual MLP blocks, regress to the horizon. No graph, attention,
recurrence, or 2D reshape — the disambiguation is the whole mechanism.

**Faithfulness to the reference (finale).** STID (Shao et al., CIKM 2022, arXiv 2208.05233). The canonical
arch uses `1×1` Conv2d over a `[B, C, N, 1]` layout with per-block dropout; this scaffold re-expresses the
identical map with `nn.Linear` + `nn.Parameter` tables over `[B, N, h]` (a `1×1` conv over a singleton
spatial axis *is* a per-node linear). Index derivation (`×288`, `×7`, last step) matches the canonical lookup.

**Bar to clear (no feedback — this is the endpoint).** Beat all three winners *at once* and by a clear
margin: METR-LA 3.8321 (iTransformer), PEMS-BAY 1.9621 (SOFTS), PEMS04 21.8436 (TimesNet). The sharpest
test is METR-LA; the most nervous watch is PEMS04 (whose winner used explicit temporal modeling). If it
wins, the ablations to run: drop spatial identity (should collapse toward the floor's averaging on dense
grids); drop temporal identities (should err most where the same curve continues differently by time).

**Hyperparameters.** `hidden_size = 32` (per-component embedding dim), `num_layers = 3` residual MLP
blocks, `dropout = 0.0`; `CONFIG_OVERRIDES = {'lr': 2e-3}` (the harness default — the shallow MLP is stable).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=32)
    num_layers: int = field(default=3)
    dropout: float = field(default=0.0)


class ResMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))) + x


class Custom(nn.Module):
    """STID: Spatial-Temporal Identity baseline.

    Per-node MLP over historical steps, augmented with learnable
    spatial embeddings and temporal (time-of-day, day-of-week) embeddings.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        h = config.hidden_size  # embedding dim for each component

        # Time series embedding: project input_len -> h per node
        self.ts_embed = nn.Linear(config.input_len, h)

        # Spatial embedding: learnable per-node identity [N, h]
        self.spatial_emb = nn.Parameter(torch.empty(config.num_features, h))
        nn.init.xavier_uniform_(self.spatial_emb)

        # Temporal embeddings
        self.tid_emb = nn.Parameter(torch.empty(288, h))  # time-in-day
        nn.init.xavier_uniform_(self.tid_emb)
        self.diw_emb = nn.Parameter(torch.empty(7, h))    # day-in-week
        nn.init.xavier_uniform_(self.diw_emb)

        # Encoder: stack of residual MLPs over concatenated embeddings
        total_h = h * 4  # ts + spatial + tid + diw
        self.encoder = nn.Sequential(
            *[ResMLP(total_h, total_h) for _ in range(config.num_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(total_h, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N], inputs_timestamps: [B, T, 2]
        B, T, N = inputs.shape

        # Time series embedding: [B, N, h]
        ts_emb = self.ts_embed(inputs.transpose(1, 2))

        # Temporal embeddings from last timestamp
        tid_idx = (inputs_timestamps[:, -1, 0] * 288).long()  # [B]
        diw_idx = (inputs_timestamps[:, -1, 1] * 7).long()    # [B]
        tid = self.tid_emb[tid_idx]  # [B, h]
        diw = self.diw_emb[diw_idx]  # [B, h]

        # Expand to [B, N, h]
        spatial = self.spatial_emb.unsqueeze(0).expand(B, -1, -1)
        tid = tid.unsqueeze(1).expand(-1, N, -1)
        diw = diw.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: [B, N, 4*h]
        hidden = torch.cat([ts_emb, spatial, tid, diw], dim=-1)

        # Encode and project: [B, N, output_len] -> [B, output_len, N]
        hidden = self.encoder(hidden)
        prediction = self.output_proj(hidden).transpose(1, 2)
        return prediction


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 2e-3}
```
