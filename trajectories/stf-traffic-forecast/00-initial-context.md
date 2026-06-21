## Research question

What spatial-temporal forecasting component generalizes across traffic-sensor networks of different sizes and modalities — speed on METR-LA (207 sensors) and PEMS-BAY (325 sensors), flow on PEMS04 (307 sensors) — under one fixed 12-step → 12-step horizon and one common evaluation protocol? The only trainable design choice is the architecture inside `Custom`, the map from a length-12 window of all `N` nodes to the next 12 steps for every node. Everything else in the pipeline is frozen. The aim is a component whose forecasting skill survives changes in network scale and measured quantity.

## Prior art / background / baselines

- **Vanilla Transformer for forecasting.** A token per timestamp, self-attention over time, and an autoregressive decoder that emits the horizon one step at a time.
- **Informer.** A sparse attention approximation that reduces complexity to `O(L log L)`, paired with a generative decoder that predicts all horizon steps at once.
- **Autoformer.** A moving-average decomposition of each series into trend and seasonal parts, wired through the network alongside an Auto-Correlation mechanism over FFT-identified dominant lags.
- **FEDformer / Pyraformer.** Frequency-domain attention with a mixture of decomposition kernels, and pyramidal `O(L)` multi-scale attention over time.

## The fixed substrate

The BasicTS training pipeline is frozen and must not be modified: per-dataset Z-score normalization, sliding-window sampling of (12-in, 12-out) pairs, masked loss and metrics with missing values encoded as `0.0` and excluded, inverse transform back to original scale before scoring, Adam(`lr=2e-3`, `weight_decay=1e-4`), `MultiStepLR(milestones=[1,50,80], gamma=0.5)`, 100 epochs at batch size 64, and gradient clipping. Only `lr` and `weight_decay` may be overridden per method. The pipeline exposes `basicts.modules` utilities — MLP/ResMLP layers, RevIN/LayerNorm, sequence embeddings, multi-head attention/encoder, activations — and six read-only reference models in `basicts/models/` for context.

## The editable interface

Exactly one region is editable: the `CustomConfig` dataclass and the `Custom` `nn.Module` in `custom_model.py`, plus the `CONFIG_OVERRIDES` dict at the bottom for `lr`/`weight_decay` only. The forward contract is:

```python
def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor) -> torch.Tensor:
    # inputs:            [batch_size, input_len=12, num_features]   # num_features = number of nodes
    # inputs_timestamps: [batch_size, input_len=12, 2]             # [time-of-day, day-of-week] in [0,1]
    # returns:           [batch_size, output_len=12, num_features] # next-hour prediction per node
```

`CustomConfig` extends `basicts.configs.BasicTSModelConfig` with at least `input_len`, `output_len`, `num_features`. The starting fill is the zero predictor below; each method replaces exactly the `CustomConfig` + `Custom` definitions and optionally `CONFIG_OVERRIDES`.

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

Three datasets spanning size and modality — **METR-LA** (207 speed sensors, Los Angeles), **PEMS-BAY** (325 speed sensors, Bay Area), **PEMS04** (307 flow sensors, Caltrans D4) — each trained and tested by the fixed BasicTS pipeline at `input_len=12`, `output_len=12`. Three metrics, **all lower is better**, computed in original scale after inverse transform with the missing-value mask applied: **MAE**, **RMSE**, **MAPE**. The headline ranking metric is MAE.
