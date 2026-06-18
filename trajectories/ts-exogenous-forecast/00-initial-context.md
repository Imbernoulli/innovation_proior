## Research question

A multivariate series arrives with a designated **target** channel plus a pile of **exogenous**
covariates — extra observed channels that genuinely drive the target (weather observations for a
weather station's wet-bulb, related sensors for an electricity-transformer temperature, the other 320
clients for one client's load). The Time-Series-Library `features=MS` mode formalizes the asymmetry:
all variables are fed in, but only the **last channel** (the target) is scored. The one thing being
designed is the **exogenous-fusion component** — how the side channels are folded into the target's
forecast — while the look-back window, the horizon, and the whole Time-Series-Library training/eval
pipeline stay fixed. Everything else about the run is frozen.

## Prior art before the first rung (long-horizon forecasting lineage)

The first rung reacts to a specific line of forecasters; the fixed substrate below is what that line
converged to. These are the ancestors, each with the gap that the ladder will press on.

- **Informer (Zhou et al., AAAI 2021).** A Transformer for long-horizon forecasting with ProbSparse
  attention (only the dominant queries attend), a self-attention distilling encoder, and a generative
  decoder that emits the whole horizon in one pass. Cut attention from $O(L^2)$ toward $O(L\log L)$ and
  made direct multi-step practical. Gap: still point-wise attention over single time steps, and the win
  over the baselines it beat conflated "Transformer" with "direct multi-step."
- **Autoformer (Wu et al., NeurIPS 2021).** Replaces dot-product attention with an Auto-Correlation
  block that aggregates by series periodicity, and bakes a moving-average **series decomposition**
  (trend + seasonal) into every layer. Gap: more machinery on the time axis, but the token is still a
  timestamp and the cross-channel structure is untouched.
- **FEDformer (Zhou et al., ICML 2022).** Attention in the frequency domain with a mixture of
  decomposition kernels — $O(L)$ and accurate on the headline benchmarks. Gap: the accuracy gains kept
  coming from temporal-attention surgery, and a plain linear map would soon match the whole family.
- **The MS asymmetry itself.** None of these is built around a *designated target with exogenous
  covariates*. They forecast every channel symmetrically; here only the last channel is scored, so the
  contribution space is specifically *how exogenous channels reach the target* — a question the
  temporal-attention lineage never had to answer.

## The fixed substrate

The Time-Series-Library training loop is frozen and must not be touched: the data loaders
(standardization, train/val/test splits, the fixed target column), `features=MS`, `seq_len=96`,
`label_len=48`, `pred_len=96`, Adam at `lr=1e-4`, `train_epochs=10`, `batch_size=32`, `patience=3`,
MSE training loss, and the scoring step that extracts `outputs[:, :, -1:]` (the target channel) before
computing MSE and MAE. The loop hands the model four tensors every step — `x_enc`
$(B,\text{seq\_len},\text{enc\_in})$ with **all** variables, `x_mark_enc` $(B,\text{seq\_len},\text{time\_feat})$
the calendar features, and the decoder-side `x_dec`/`x_mark_dec` — and reads back
$(B,\text{pred\_len},\text{c\_out})$ with `c_out == enc_in`. The standard Time-Series-Library layers
are importable read-only: `series_decomp` (moving-average trend/seasonal split), `PatchEmbedding`
(patch + value/positional embedding), `DataEmbedding_inverted` (whole-series-per-variate token),
`Encoder`/`EncoderLayer`/`FullAttention`/`AttentionLayer` (the stock Transformer blocks).

## The editable interface

Exactly one file is editable — `models/Custom.py`, the `Model` class. Every method on the ladder is a
fill of the same contract: `__init__(self, configs)` builds the architecture from `configs`
(`task_name`, `seq_len`, `pred_len`, `enc_in`, `c_out`, plus `d_model`, `d_ff`, `e_layers`, `n_heads`,
`dropout`, …); `forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)` maps the inputs to
$(B,\text{pred\_len},\text{c\_out})$; and `forward(...)` slices the last `pred_len` steps. The target is
the **last** channel of `x_enc`/`c_out`, and only it is scored, so a fill is free to forecast all
channels and let the harness slice, or to forecast the target alone and broadcast — the contract only
fixes the output shape.

The starting point is the scaffold default: a zero forecaster (returns zeros of the right shape).
Each method on the ladder replaces exactly this `Model` definition and nothing else.

```python
# EDITABLE region of models/Custom.py — scaffold default (zero forecaster)
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # TODO: define the architecture

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch, seq_len, enc_in] — all variables; target is the last channel
        batch_size = x_enc.shape[0]
        return torch.zeros(batch_size, self.pred_len, self.c_out).to(x_enc.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```

## Evaluation settings

Three datasets spanning the channel-count and signal range — **ETTh1** (7 channels, hourly
electricity-transformer temperature; target = oil temperature), **Weather** (21 channels, 21 weather
observations; target = wet-bulb-style last channel), and **ECL** (321 channels, hourly electricity
consumption of 321 clients; target = the last client) — all in `features=MS` mode with `seq_len=96`,
`label_len=48`, `pred_len=96`, one seed (42). Two metrics, **lower is better** on both: **MSE** and
**MAE**, computed on the target channel only after the harness slices `outputs[:, :, -1:]`.
