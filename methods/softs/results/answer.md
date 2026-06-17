# SOFTS — Series-cOre Fused Time Series forecaster

## Problem

Multivariate time series forecasting: from a lookback `X ∈ R^{C×L}` (`C` channels, `L` steps) predict
`Y ∈ R^{C×H}`. Cross-channel correlation is real signal, but on non-stationary data the per-pair correlations
that channel-mixing models extract overfit and break, and all-pairs cross-channel attention costs `O(C^2)` —
prohibitive when `C` is in the hundreds (traffic sensors, grid meters). SOFTS recovers cross-channel
information with **linear** cost in `C` and an interaction that depends on a robust aggregate rather than
fragile pairwise weights.

## Key idea

Use the **inverted (channel-as-token)** view: embed each channel's whole lookback into one `d`-dim token.
Replace all-pairs attention over the `C` tokens with the **STar Aggregate-Redistribute (STAR)** module — a
*centralized* (star-shaped) interaction:

1. **Aggregate** all channel tokens into a single global **core** via a shared per-channel MLP + a
   permutation-invariant pooling over the channel axis.
2. **Redistribute** the core back to every channel; concatenate it with each channel's own token and fuse
   with a second MLP plus a residual.

The only place all channels meet is the core (built in `O(C)`, broadcast in `O(C)`); no `C×C` matrix is ever
formed. The core, being a statistic over all channels, is far more stable than any single channel or pair, so
the interaction is robust to non-stationarity and to anomalous channels.

## Core representation (design rationale)

The core summarizes an unordered *set* of channels, so it must be **permutation-invariant** over channels
(DeepSets form `ρ(Σ_x φ(x))`), not the permutation-variant Kolmogorov–Arnold form with per-coordinate weights
`λ_m`: channel-specific parameters bind capacity to the channel index, which is meaningless under distribution
shift and hurts robustness. Let `d'` be the core width; `φ = MLP₁` (two linear layers with GELU), `ρ` is dropped
because it is redundant with the fusion MLP, and the sum is generalized to a learned pooling.

**Stochastic pooling** over the channel axis interpolates between mean and max and regularizes the shared
core. For activations `A ∈ R^{C×d'}`, per feature `j`:

- probabilities (softmax across channels): `p_{ij} = e^{A_{ij}} / Σ_{k=1}^{C} e^{A_{kj}}`
- **train:** sample `c ∼ Multinomial(p_{·j})`, set `o_j = A_{cj}` (stochastic; regularizes the core)
- **test:** expectation `o_j = Σ_{i=1}^{C} p_{ij} A_{ij}` (deterministic, magnitude-weighted average)

In batched code, `A: [B,C,d']` is softmaxed on `C`, permuted to `[B,d',C]`, flattened to `[B*d',C]` for one
sample per `(batch, feature)`, reshaped to `[B,1,d']`, then used as the channel index for `gather`.

## Full model

`RevIN normalize → series embedding (L→d per channel) → N × [STAR + per-channel FFN, residual + LayerNorm] →
linear head (d→H per channel) → RevIN denormalize`. Trained with MSE under Adam. Full encoding cost is
`O(CLd + C d^2 + C d H)`; when `d` is treated as a fixed model width this is `O(CL + CH)`, linear in `C`, `L`,
and `H`, versus `O(C^2 + CL + CH)` for inverted attention.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAR(nn.Module):
    """STar Aggregate-Redistribute: O(C) centralized cross-channel interaction."""

    def __init__(self, d_series, d_core):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)            # MLP_1 : R^d -> R^{d'}
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)   # MLP_2 : R^{d+d'} -> R^d
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, x, *args, **kwargs):                    # x: [B, C, d]
        B, C, d = x.shape
        a = self.gen2(F.gelu(self.gen1(x)))                  # A: [B, C, d']

        if self.training:                                    # sample one channel per feature
            ratio = F.softmax(a, dim=1).permute(0, 2, 1).reshape(-1, C)   # [B*d', C]
            idx = torch.multinomial(ratio, 1).view(B, -1, 1).permute(0, 2, 1)  # [B, 1, d']
            core = torch.gather(a, 1, idx).repeat(1, C, 1)   # o_j = A_{cj}
        else:                                                # expectation o_j = sum_i p_ij A_ij
            p = F.softmax(a, dim=1)
            core = (a * p).sum(dim=1, keepdim=True).repeat(1, C, 1)

        f = torch.cat([x, core], dim=-1)                     # redistribute + fuse
        return self.gen4(F.gelu(self.gen3(f))), None


class EncoderLayer(nn.Module):
    def __init__(self, interaction, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.interaction = interaction
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, _ = self.interaction(x, x, x)                 # STAR ignores extra args
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class SOFTS(nn.Module):
    def __init__(self, input_len, output_len, num_features,
                 hidden_size=512, core_size=128, num_layers=2, dropout=0.05):
        super().__init__()
        self.eps = 1e-5
        self.embed = nn.Linear(input_len, hidden_size)       # L -> d, per channel
        self.layers = nn.ModuleList([
            EncoderLayer(STAR(hidden_size, core_size), hidden_size, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_len)       # d -> H, per channel

    def forward(self, inputs):                               # inputs: [B, L, C]
        mean = inputs.mean(dim=1, keepdim=True).detach()     # RevIN normalize
        x = inputs - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x = x / stdev

        N = x.size(-1)
        h = self.embed(x.transpose(1, 2))                    # [B, C, d]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        pred = self.head(h).transpose(1, 2)[:, :, :N]        # [B, H, C]
        return pred * stdev + mean                           # RevIN denormalize
```

This is the method in executable form: STAR uses the `gen1..gen4` aggregate/fuse layers in the encoder's
interaction slot, the forecaster embeds each channel's lookback with `L→d`, the projection maps `d→H`, and the
prediction is returned after the inverse normalization.
