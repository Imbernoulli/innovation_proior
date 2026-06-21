## Research question

Multivariate time series imputation under a fixed 25% random mask. A length-96 window arrives with a quarter of its (timestep, channel) entries deleted; the deleted positions are set to zero and a binary observation mask says which entries are real (`1`) and which were removed (`0`). The single design problem is the **reconstruction model**: the map from the masked window (plus the mask and the time-feature stamp) back to a full, dense window. Error is scored only at the masked positions, so the model must infer each missing value from the temporal neighbourhood of that channel and from the simultaneous values of correlated channels. Everything else — the masking protocol, standardisation, train/val/test split, optimiser loop — is fixed by the Time-Series-Library imputation pipeline and must not be touched.

## Prior art / Background / Baselines

- **Iterated / point-wise temporal models (RNNs, LSTMs).** They read the sequence one step at a time and carry a hidden state, filling gaps by rolling forward or bidirectionally through neighbours.

- **The Transformer forecasting stack (Informer, Autoformer, FEDformer).** Each variant introduces a different attention kernel — ProbSparse, auto-correlation with a decomposition block, a Fourier block — to reduce quadratic cost. The token is a single time step: one scalar or one channel-vector at time *t*.

- **Seasonal-trend decomposition (classical STL, Autoformer's moving-average block).** It writes a window additively as a slow trend plus a residual; each piece is more regular than the sum.

- **Non-stationary normalisation (Non-stationary Transformer).** It centres and scales each window before the model and undoes the transform after, so the network sees only the window shape while level and drift are handled outside.

## Fixed substrate / Code framework

The Time-Series-Library imputation loop is frozen. It samples a Bernoulli(0.75) observation mask per (timestep, channel), zeroes the masked entries of the standardised window, and hands the model `(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)`. It trains with Adam (`learning_rate=1e-3`, `batch_size=16`, `train_epochs=10`, `patience=3`) under an MSE loss applied only at the masked positions, and reports MSE and MAE on masked entries. The window is `seq_len=96`; for imputation `pred_len=seq_len`, so the model is a same-length sequence-to-sequence map. The loop also fixes the model config (`d_model`, `d_ff`, `e_layers`, `n_heads`, `dropout`, `embed='timeF'`, `freq='h'`, `moving_avg=25`, `top_k=5`, `num_kernels=6`, `factor=3`, `activation='gelu'`) and provides reusable layers (`series_decomp`, `DataEmbedding`, `Inception_Block_V1`, `PatchEmbedding`, the Transformer `Encoder`).

## Editable interface

Exactly one file is created and editable — `models/Custom.py` — and inside it exactly one class, `Model`, with the imputation contract below. Any solution fills this same contract: `__init__(self, configs)` builds the architecture from the frozen config; `imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask)` returns the dense reconstruction `[batch, seq_len, enc_in]`; and `forward` dispatches to it. `x_enc` already has masked entries zeroed; `mask` is `1` for observed and `0` for masked; `x_dec`/`x_mark_dec` are unused for imputation.

The starting point is the scaffold default: **identity** — return the masked input unchanged. A solution replaces only this `Model` body and nothing else.

```python
# EDITABLE file models/Custom.py — default fill (identity, no model)
import torch
import torch.nn as nn


class Model(nn.Module):
    """Custom model for time series imputation (default: identity)."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len      # imputation: pred_len = seq_len
        self.enc_in = configs.enc_in

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return x_enc                         # placeholder: return input as-is

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return None
```

## Evaluation settings

Three datasets spanning the channel-count range — **ETTh1** (7 variables, hourly transformer temperature), **Weather** (21 variables), and **ECL** (321 variables, hourly client electricity) — each at `seq_len=96`, `mask_rate=0.25`, seed 42. Two metrics on masked entries only, **lower is better**: `mse` and `mae`, reported per dataset.
