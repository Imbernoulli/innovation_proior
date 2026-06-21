## Research question

Univariate short-term forecasting on the M4 competition. The data is 100,000 short real-world series
split across seasonal regimes that behave very differently — Monthly (horizon 18, strong 12-month
seasonality), Quarterly (horizon 8, mild 4-quarter seasonality), Yearly (horizon 6, essentially no
seasonality, mostly trend). One forecasting component must deliver low SMAPE across all three under a
*single fixed* training and evaluation protocol. The design target is the `Model` class in
`models/Custom.py`: a map from a length-`seq_len` look-back of one channel to a length-`pred_len`
forecast of that channel. Everything around it — the data loader, the optimizer, the SMAPE training
loss, the M4 evaluator — is frozen.

## Prior art / Background / Baselines

The relevant baselines are the high-capacity sequence models that currently dominate forecasting
benchmarks.

- **Iterated RNN/seq2seq forecasters (e.g. DeepAR, Salinas et al. 2017).** Predict one step, feed it
  back, repeat to the horizon.
- **Transformer forecasters (Informer, Zhou et al. 2021; Autoformer, Wu et al. 2021; FEDformer, Zhou
  et al. 2022).** Replace recurrence with attention and predict the whole horizon at once. Autoformer
  adds an Auto-Correlation block and a moving-average **series decomposition** that splits a window
  into trend and seasonal parts; FEDformer moves the mixing into the frequency domain. These set the
  accuracy bar on long-horizon benchmarks.
- **The decomposition primitive (from Autoformer's `series_decomp`).** A length-preserving moving
  average gives the trend; the residual is the seasonal part. It is a reusable operation when a
  window's trend and seasonal balance matters.

## Fixed substrate / Code framework

The Time-Series-Library short-term-forecast harness is frozen and must not be touched. For M4 it sets
`seq_len = 2·pred_len` and `label_len = pred_len`, so the look-back is short: Monthly
`pred_len=18 → seq_len=36`, Quarterly `pred_len=8 → seq_len=16`, Yearly `pred_len=6 → seq_len=12`.
`features=M`, `enc_in = dec_in = c_out = 1` (univariate). The training loop builds a zero decoder
input, calls the model as `model(batch_x, None, dec_inp, None)` — so **no time-feature marks are
passed** (`x_mark_enc` and `x_mark_dec` are `None`) — takes the last `pred_len` steps of the output,
and optimizes the **SMAPE loss** the harness supplies. Adam is the optimizer; the M4 evaluator scores
SMAPE (primary) and MAPE on the official test horizon, lower is better.

The harness invokes the same `models/Custom.py` for all three regimes with **one fixed set of
training hyperparameters** — `d_model=512`, `d_ff=512`, `e_layers=2`, `learning_rate=1e-3`,
`batch_size=16`, `train_epochs=10`, `patience=3`. It does *not* pass the per-method tuned flags the
reference models' own scripts use (TimeMixer's `down_sampling_*`, its long
`train_epochs=50`/`lr=1e-2`, PatchTST's `e_layers=3`, etc.). So whatever is borrowed from a reference
architecture must run correctly and well under *these* fixed settings and a 36-step (or shorter)
window — any hyperparameter a method needs has to be set inside `Custom.py`, read from `configs`
with a safe default.

## Editable interface

Exactly one file is editable — `models/Custom.py` — and exactly one contract: a `Model(nn.Module)`
with `__init__(self, configs)` reading `configs.task_name`, `configs.seq_len`, `configs.pred_len`,
`configs.enc_in (=1)`, `configs.c_out (=1)`; a `forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)`
taking `x_enc : [batch, seq_len, 1]` and returning `[batch, pred_len, 1]`; and a `forward` that
dispatches the forecast task and returns the last `pred_len` steps.

The starting point is the scaffold default: a forecaster that returns zeros. Replace the body of
`__init__`/`forecast` and nothing else about the contract.

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Custom model for short-term time series forecasting (M4 dataset).

    forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
      x_enc:      [batch, seq_len, enc_in]   (enc_in = 1 for M4)
      x_mark_enc: passed as None by the M4 harness
      returns:    [batch, pred_len, c_out]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # default fill: no trainable forecaster

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # default: predict zeros of the right shape
        batch_size = x_enc.shape[0]
        return torch.zeros(batch_size, self.pred_len, self.c_out).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```

## Evaluation settings

Three M4 seasonal patterns — **Monthly** (`pred_len=18`, freq 12), **Quarterly** (`pred_len=8`,
freq 4), **Yearly** (`pred_len=6`, freq 1) — each run at the single seed {42} the config fixes, under
the one fixed Custom training protocol above (`d_model=512`, `e_layers=2`, `lr=1e-3`, batch 16, 10
epochs, patience 3, SMAPE loss). Two metrics, both **lower is better**: `smape` (primary) and `mape`,
computed by the M4 evaluator on the official test horizon for each pattern.
