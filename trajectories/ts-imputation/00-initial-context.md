## Research question

Multivariate time series imputation under a fixed 25% random mask. A length-96 window arrives with a
quarter of its (timestep, channel) entries deleted; the deleted positions are set to zero and a binary
observation mask says which entries are real (`1`) and which were removed (`0`). The single thing being
designed is the **reconstruction model**: the map from the masked window (plus the mask and the
time-feature stamp) back to a full, dense window. Error is scored only at the masked positions, so the
model has to infer each missing value from two sources of context — the temporal neighbourhood of that
channel and the simultaneous values of the correlated channels. Everything else (the masking protocol,
the standardisation, the train/val/test split, the optimiser loop) is fixed by the Time-Series-Library
imputation pipeline and must not be touched.

## Prior art before the first rung (sequence-modelling lineage)

The first rung reacts to a specific tension in how temporal models had been built up to this point. The
ancestors below are the lineage the ladder argues with; each is a real line of work with a real gap.

- **Iterated / point-wise temporal models (RNNs, LSTMs).** Read the sequence one step at a time and
  carry a hidden state. For imputation they fill a gap by rolling forward (or bidirectionally) through
  neighbours. The signal path from a distant observed point to a masked one is long, so information is
  forgotten through the recurrence, and they are slow. Gap: long-range dependence is bottlenecked by the
  recurrence.
- **The Transformer forecasting stack (Informer, Vaswani et al. 2017 → Zhou et al. 2021; Autoformer,
  Wu et al. 2021; FEDformer, Zhou et al. 2022).** Each is a surgery on the attention kernel to tame its
  quadratic cost — ProbSparse, auto-correlation with a decomposition block, a Fourier block. The token
  is a single time step: one scalar (or one channel-vector) at time *t*. Gap: a single time step has
  almost no standalone meaning, so point-wise attention compares objects that carry little signal, and
  the elaborate kernels keep getting matched by far simpler maps.
- **Seasonal-trend decomposition (classical STL → Autoformer's moving-average block).** Write a window
  additively as a slow trend plus a residual; each piece is more regular and more predictable than the
  sum. A parameter-free preprocessing the ladder will reuse. Gap on its own: it is only a
  reparameterisation; it needs a predictor behind it.
- **Non-stationary normalisation (Kim et al. 2022; the Non-stationary Transformer, Liu et al. 2022).**
  Per-window centre-and-scale before the model and undo it after, so the network only sees the *shape*
  of the window and the drifting level/scale is handled outside it. The ladder adapts this to the masked
  setting (statistics computed over observed entries only).

## The fixed substrate

The Time-Series-Library imputation loop is frozen. It samples a Bernoulli(0.75) observation mask per
(timestep, channel), zeroes the masked entries of the standardised window, and hands the model
`(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)`. It trains with Adam (`learning_rate=1e-3`,
`batch_size=16`, `train_epochs=10`, `patience=3`) under an MSE loss **applied only at the masked
positions**, and reports MSE and MAE on masked entries. The window is `seq_len=96`; for imputation
`pred_len=seq_len`, so the model is a same-length sequence-to-sequence map. The loop also fixes the
config the model is built from (`d_model`, `d_ff`, `e_layers`, `n_heads`, `dropout`, `embed='timeF'`,
`freq='h'`, `moving_avg=25`, `top_k=5`, `num_kernels=6`, `factor=3`, `activation='gelu'`) and provides
the library's reusable layers (`series_decomp`, `DataEmbedding`, `Inception_Block_V1`,
`PatchEmbedding`, the Transformer `Encoder`).

## The editable interface

Exactly one file is created and editable — `models/Custom.py` — and inside it exactly one class, `Model`,
with the imputation contract below. Every method on the ladder is a fill of this same contract:
`__init__(self, configs)` builds the architecture from the frozen config; `imputation(self, x_enc,
x_mark_enc, x_dec, x_mark_dec, mask)` returns the dense reconstruction `[batch, seq_len, enc_in]`; and
`forward` dispatches to it. `x_enc` already has masked entries zeroed; `mask` is `1` for observed and
`0` for masked; `x_dec`/`x_mark_dec` are unused for imputation.

The starting point is the scaffold default: **identity** — return the masked input unchanged. Every later
method replaces exactly this `Model` body and nothing else.

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

Three datasets spanning the channel-count range — **ETTh1** (7 variables, hourly transformer
temperature), **Weather** (21 variables), and **ECL** (321 variables, hourly client electricity) — each
at `seq_len=96`, `mask_rate=0.25`, seed 42. Two metrics on masked entries only, **lower is better**:
`mse` and `mae`, reported per dataset.
