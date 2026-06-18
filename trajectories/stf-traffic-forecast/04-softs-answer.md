**Problem.** The spectral-graph rung proved cross-node coupling is the right axis (big PEMS04 win) but
its mechanism is heavy and fragile — it even lost to the channel-independent rung on METR-LA MAE. Keep
cross-node modeling, but through a mechanism that does not re-import the fragility and is cheap as `N`
grows into the hundreds.

**Key idea.** Invert the view — each node's whole 12-step lookback is one token (`N` tokens) — and route
cross-node interaction through a **single shared core**, not all pairs. All-pairs attention is `O(N²)` and
leans on per-pair correlations, exactly what overfits under non-stationarity. Instead: aggregate the `N`
tokens into one core (`MLP1` then a pool), redistribute it to every node, fuse with each node's own token
(`MLP2` + residual). A core aggregated over all nodes is a *statistic* — robust to a single anomalous
sensor — and the topology is star-shaped, so cost is `O(N)`. The aggregation is permutation-invariant over
nodes (a per-channel weight would spend capacity on the meaningless channel index). The pool is
**stochastic**: softmax node activations across the channel axis, sample one node per feature in training
(between mean and max, regularizes the shared core), take the expectation at test (a magnitude-weighted
average). RevIN brackets the whole thing against drift.

**Why.** Linear scaling *and* a robust aggregate instead of brittle pairwise weights — both StemGNN
problems fixed at once. Trades away explicit per-pair structure, which may under-model directed flow
propagation.

**Hyperparameters.** `hidden_size = 512`, `core_size = 128`, `num_layers = 2`, `dropout = 0.05`;
`CONFIG_OVERRIDES = {'lr': 0.0005}` (the wide inverted stack is unstable at the 2e-3 default),
`weight_decay` at default.

```python
import math
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
    hidden_size: int = field(default=512)
    core_size: int = field(default=128)
    num_layers: int = field(default=2)
    dropout: float = field(default=0.05)


class RevIN(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:
            return x * self.stdev + self.mean


class MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class STAR(nn.Module):
    """STar Aggregate-Redistribute module.

    Aggregates cross-variate info into a core representation via
    stochastic pooling (training) or weighted mean (inference),
    then redistributes back to each variate.
    """
    def __init__(self, hidden_size, core_size):
        super().__init__()
        self.ffn1 = MLP(hidden_size, hidden_size, core_size)
        self.ffn2 = MLP(hidden_size + core_size, hidden_size, hidden_size)

    def forward(self, x):
        B, N, D = x.shape
        combined = self.ffn1(x)  # [B, N, core_size]

        if self.training:
            # Stochastic pooling
            ratio = F.softmax(combined, dim=1)  # [B, N, core_size]
            ratio = ratio.transpose(1, 2).reshape(-1, N)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(B, -1, 1).transpose(1, 2)  # [B, 1, core_size]
            core = torch.gather(combined, 1, indices)  # [B, 1, core_size]
            core = core.repeat(1, N, 1)
        else:
            # Weighted mean
            weight = F.softmax(combined, dim=1)
            core = (combined * weight).sum(dim=1, keepdim=True).repeat(1, N, 1)

        return self.ffn2(torch.cat([x, core], dim=-1))


class SOFTSBlock(nn.Module):
    def __init__(self, hidden_size, core_size, dropout):
        super().__init__()
        self.star = STAR(hidden_size, core_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm1(x + self.star(x))
        x = self.norm2(x + self.ffn(x))
        return x


class Custom(nn.Module):
    """SOFTS: Series-Core Fusion baseline.

    Inverted architecture (nodes as tokens), using STAR modules
    instead of self-attention for O(N) cross-variate communication.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.revin = RevIN()

        # Sequence embedding: [B, T, N] -> transpose -> [B, N, T] -> [B, N, D]
        self.embed = nn.Linear(config.input_len, config.hidden_size)
        self.embed_drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            SOFTSBlock(config.hidden_size, config.core_size, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output: [B, N, D] -> [B, N, T'] -> [B, T', N]
        self.head = nn.Linear(config.hidden_size, config.output_len)

    def forward(self, inputs, inputs_timestamps):
        x = self.revin(inputs, "norm")
        N = x.size(-1)

        h = self.embed_drop(self.embed(x.transpose(1, 2)))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        pred = self.head(h).transpose(1, 2)[:, :, :N]
        pred = self.revin(pred, "denorm")
        return pred


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.0005}
```
