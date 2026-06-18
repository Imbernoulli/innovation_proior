## Research question

What modular spatial-temporal forecasting component generalizes across traffic-sensor networks of
different sizes and modalities — speed on METR-LA (207 sensors) and PEMS-BAY (325 sensors), flow on
PEMS04 (307 sensors) — under one fixed 12-step → 12-step horizon and one common evaluation protocol?
The single thing being designed is the architecture inside `Custom` — the map from a length-12 window
of all `N` nodes to the next 12 steps for every node. Everything else (the data pipeline, the loss, the
optimizer, the schedule) is frozen. The point is not to win one dataset but to find the component whose
forecasting skill survives the change of network size and the change of measured quantity.

## Prior art before the first rung (the forecasting lineage)

The first rung reacts to the recent history of *long-horizon* forecasters, which is a sequence of
ever-more-elaborate Transformers — and one embarrassing result that called the whole elaboration into
question. These are the methods the ladder climbs out of; each is named with the gap it leaves.

- **Vanilla Transformer for forecasting (Vaswani et al. 2017, applied to TSF).** A token per timestamp,
  self-attention over time, an autoregressive decoder emitting the horizon one step at a time. Gap:
  attention is `O(L²)` over the sequence length, and the autoregressive decoder accumulates error over a
  long horizon — exactly the regime it must handle.
- **Informer (Zhou et al., AAAI 2021).** ProbSparse attention (`O(L log L)`) plus a generative decoder
  that emits the whole horizon in one pass (direct multi-step). Gap: it changed *both* the attention and
  the forecasting strategy at once, so the reported gain is not attributable to either; still a
  high-capacity attention stack whose use of temporal order is unverified.
- **Autoformer (Xu et al., NeurIPS 2021).** A moving-average `series_decomp` block (trend = moving
  average, seasonal = residual) wired throughout the network, with an Auto-Correlation mechanism over
  the FFT-found dominant lags. Gap: it bundles two reusable ideas (decomposition, direct multi-step)
  with a still-complex sequence model and never isolates how much accuracy comes from which.
- **FEDformer / Pyraformer (Zhou et al. ICML 2022; Liu et al. ICLR 2022).** Frequency-domain attention
  with a mixture of decomposition kernels; pyramidal `O(L)` multi-scale attention. Gap: the same
  pattern — each more intricate, each a new best, each leaving open whether the intricate part is doing
  the work, and each not improving when handed a longer look-back, the fingerprint of a model that is
  not really using long-range temporal structure.

A common thread: the non-Transformer baselines these papers beat were iterated multi-step forecasters
that compound error over long horizons, so "Transformer wins" was confounded with "direct multi-step
wins." The first rung exists to strip that confound to the bone.

## The fixed substrate

A BasicTS training pipeline is frozen and must not be touched: per-dataset Z-score normalization,
sliding-window sampling of (12-in, 12-out) pairs, a masked loss and masked metrics (missing values are
encoded as `0.0` and excluded from both loss and metrics), inverse transform back to original scale
before scoring, Adam with `lr=2e-3`, `weight_decay=1e-4`, `MultiStepLR(milestones=[1,50,80], gamma=0.5)`
for 100 epochs at batch size 64, with gradient clipping. Only `lr` and `weight_decay` may be overridden
per method (epochs, batch size, scheduler, clipping are fixed). The loop also exposes a small toolbox a
method may import from `basicts.modules` — `MLPLayer`/`ResMLPLayer`, `RevIN`/`LayerNorm`, sequence
embeddings, a `MultiHeadAttention`/`Encoder`, and common activations — and six read-only reference
models in `basicts/models/` (SOFTS, DLinear, StemGNN, iTransformer, TimesNet, TimeMixer) as context.

## The editable interface

Exactly one region is editable — the `CustomConfig` dataclass and the `Custom` `nn.Module` in
`custom_model.py` (and the `CONFIG_OVERRIDES` dict at the bottom, for `lr`/`weight_decay` only). The
contract is one forward method:

```python
def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
    # inputs:            [batch_size, input_len=12, num_features]   # num_features = number of nodes
    # inputs_timestamps: [batch_size, input_len=12, 2]             # [time-of-day, day-of-week] in [0,1]
    # returns:           [batch_size, output_len=12, num_features] # next-hour prediction per node
```

`CustomConfig` extends `basicts.configs.BasicTSModelConfig` with at least `input_len`, `output_len`,
`num_features`. Every method on the ladder is a fill of this same contract. The starting point is the
scaffold default — a zero predictor that does nothing — and each method replaces exactly the
`CustomConfig` + `Custom` definitions (and optionally `CONFIG_OVERRIDES`).

```python
# EDITABLE region of custom_model.py — default fill (no model: predicts zeros)
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=64)
    num_layers: int = field(default=2)
    dropout: float = field(default=0.1)


class Custom(nn.Module):
    """Default baseline: no spatial-temporal model; returns zeros."""

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        return torch.zeros(batch_size, self.output_len, self.num_features, device=inputs.device)


# CONFIG_OVERRIDES: override training hyperparameters (allowed keys: lr, weight_decay).
CONFIG_OVERRIDES = {}
```

## Evaluation settings

Three datasets spanning size and modality — **METR-LA** (207 speed sensors, Los Angeles),
**PEMS-BAY** (325 speed sensors, Bay Area), **PEMS04** (307 flow sensors, Caltrans D4) — each trained
and tested by the fixed BasicTS pipeline at `input_len=12`, `output_len=12`. Three metrics, **all lower
is better**, computed in original scale after inverse transform with the missing-value mask applied:
**MAE**, **RMSE**, **MAPE**. The headline ranking metric is MAE.
