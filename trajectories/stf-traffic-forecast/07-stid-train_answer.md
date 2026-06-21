iTransformer landed best-on-average, and in landing there it exposed the one thing the whole ladder never achieved: no single model wins all three datasets. It took METR-LA to $3.8321$ MAE — the best on the ladder — and PEMS-BAY to $1.9637$, statistically tied with SOFTS's $1.9621$ for best on the densest grid, confirming that an explicit $N \times N$ correlation in the robust inverted view is the right cross-node mechanism. But on PEMS04 it came in at $25.0026$, trailing TimesNet's $21.8436$ by a wide margin: inverted attention models the per-node temporal axis with only a flat shared FFN, and flow's sharp multi-scale periodic shape is exactly what that under-models. The final tally is a standoff — TimesNet owns flow through explicit 2D *temporal* period modeling; iTransformer and SOFTS own the speed grids through explicit *cross-node* modeling; each strong rung built *one* explicit structure and let a generic operator carry the other, and loses the dataset where the neglected axis dominates.

The obvious resolution is a heavy composite — the graph of iTransformer plus the 2D conv of TimesNet, both axes modeled with full machinery at once. But before reaching for that, let me re-examine what the expensive structures were actually *doing*, because I suspect a much cheaper resolution. iTransformer's $N \times N$ attention exists so the model can tell *which sensor* a sample is — its whole content is letting node $i$'s update depend on which other nodes it correlates with, a way of conditioning on node identity. TimesNet's 2D-period machinery and StemGNN's spectral cell exist so the model can tell *what temporal regime* a sample is in. Strip it to the bone: every rung's expensive structure is a mechanism for making samples *distinguishable* — spatially (which sensor?) and temporally (what time?). That also explains the channel-independent floor's failure. DLinear and TimeMixer averaged over an equivalence class of inputs: two sensors with the same recent curve got the same prediction, even though a downtown loop and a suburban segment continue differently, and the same curve at 8am (rush building) and 8pm (rush clearing) continues in opposite directions. The samples were *indistinguishable*, so the shared map output one averaged continuation for futures that genuinely diverge. If distinguishability is the necessary ingredient, I can buy it directly, for almost nothing, and a plain MLP should suddenly become competitive with the whole ladder.

So I propose STID — spatial-temporal identity — which buys distinguishability directly with three learnable embedding tables on a plain residual MLP, and nothing else. Take the spatial dimension first. I want the model to know *which sensor* a sample comes from, so I give every sensor its own learnable vector: a spatial identity table $E_S \in \mathbb{R}^{N \times h}$, one row per node, learned end to end. Attach node $i$'s row to its representation, and two samples with identical histories from different sensors are no longer identical to the model — their identity vectors differ, so it learns a sensor-specific continuation for each. This is the model learning an *embedding of the network*: instead of being handed an adjacency, or learning a Laplacian and convolving, or computing $N \times N$ attention every forward pass, the sensor-to-sensor structure lives latent in the geometry of $N$ learnable vectors — sensors that behave alike drift to nearby embeddings because that minimizes the loss. I get the spatial distinguishability the whole ladder paid for, as a plain lookup table, with no message passing, no Laplacian, no attention, and no $O(N^2)$.

The temporal dimension mirrors the trick, and the harness hands me exactly the variables for it — `inputs_timestamps` carries time-of-day and day-of-week per step, the very features that disambiguate 8am from 8pm and Monday from Sunday, which every prior rung ignored. Traffic has 288 five-minute slots in a day, so a learnable table $E_{\text{TiD}} \in \mathbb{R}^{288 \times h}$ gives the time-in-day identity, looked up by the current slot; seven weekdays give a table $E_{\text{DiW}} \in \mathbb{R}^{7 \times h}$ for the day-in-week identity. Attach both. Now a sample is tagged with which sensor, which five-minute slot, and which weekday it sits in. Why learnable embeddings rather than the raw normalized scalars the timestamps already provide? Because the effect of "it is 8am" on the continuation is sharp and non-monotonic — rush hour is a regime, not a smooth function of $8/24$ — and a learned per-slot vector lets each slot carry an arbitrary signature that a single scalar through a shared MLP cannot express. The 288 slots and 7 days are few enough that dedicated tables are cheap.

The architecture is almost embarrassingly plain. Embed each node's own twelve-step history with a single shared $\text{Linear}(\text{input\_len}, h)$ into an $h$-dimensional time-series embedding — the per-node temporal-shape representation that was strong from the floor onward. Concatenate onto it the three identities — the spatial row for this node, the time-in-day vector for this sample's slot, the day-in-week vector for its weekday — giving a $4h$-wide vector per node that carries *what the recent history looks like*, *which sensor this is*, *what time of day*, and *what day of week*: exactly the quadruple a forecaster needs and exactly what every channel-independent rung was missing. Feed it through a short encoder of residual MLP blocks $\text{fc}_2(\text{relu}(\text{fc}_1(x))) + x$ — residual so depth trains stably and each block learns only a correction — three of them, then a regression head $4h \to \text{output\_len}$. No graph, no attention, no recurrence, no 2D reshape: an embedding layer, residual MLPs, a head, and three lookup tables doing all the disambiguation.

The index bookkeeping is where the model silently breaks, so I pin it to this harness. The timestamps arrive normalized to $[0,1]$. The time-in-day index is $\lfloor t_{\text{day}} \times 288 \rfloor \in \{0,\dots,287\}$; the day-in-week index is $\lfloor d \times 7 \rfloor \in \{0,\dots,6\}$. I read both at the *last* look-back step, because the forecast is anchored at "now" — the disambiguating context is what time it is at the moment I predict from. The spatial identity needs no index: it is the node itself, so the $N \times h$ table broadcasts across the batch. Each identity expands to $[B, N, h]$ to line up with the time-series embedding, the four concatenate to $[B, N, 4h]$, the encoder preserves $4h$ so the residual skips add cleanly, and the head maps to $[B, N, \text{output\_len}]$, transposed to $[B, \text{output\_len}, N]$.

One thing to name precisely against the canonical reference (Shao et al., CIKM 2022). The canonical STID implements the embeddings and encoder with $1\times1$ Conv2d over a $[B, C, N, 1]$ layout and a small per-block dropout; this scaffold re-expresses the identical computation with `nn.Linear` and `nn.Parameter` tables over a $[B, N, h]$ layout and no dropout — mathematically the same map, since a $1\times1$ conv over a singleton spatial axis *is* a per-node linear, and the index derivation ($\times 288$, $\times 7$, last step) matches the canonical lookup exactly. The shallow MLP is stable at the harness default learning rate, so `CONFIG_OVERRIDES = {'lr': 2e-3}` simply restates it. The move past the strongest baseline is to stop building one explicit axis and letting a generic operator carry the other, and instead make every sample distinguishable in both axes directly. The bar this must clear is the three different winners at once — METR-LA $3.8321$, PEMS-BAY $1.9621$, PEMS04 $21.8436$ — and by a clear margin, since if the graph, the attention, and the 2D-period machinery were all just expensive routes to distinguishability, resolving it directly should dominate every rung on every dataset.

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
