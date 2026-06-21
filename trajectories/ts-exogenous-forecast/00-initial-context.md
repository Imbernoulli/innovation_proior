## Research question

A multivariate series arrives with a designated **target** channel plus **exogenous** covariates — extra observed channels that drive the target. The Time-Series-Library `features=MS` mode formalizes the asymmetry: all variables are fed in, but only the **last channel** (the target) is scored. The design problem is the **exogenous-fusion component**: how side channels are folded into the target's forecast. The look-back window, horizon, and training/eval pipeline stay fixed.

## Prior art / Background / Baselines

These are the relevant forecasters at this point, each with the gap it leaves.

- **Informer.** A Transformer for long-horizon forecasting with ProbSparse attention and a generative decoder that emits the whole horizon in one pass. Gap: attention is still point-wise over single time steps, and all channels are treated with the same representation.
- **Autoformer.** Replaces dot-product attention with an Auto-Correlation block that aggregates by series periodicity, with series decomposition built into every layer. Gap: the machinery is on the time axis; cross-channel structure is not modeled explicitly.
- **FEDformer.** Attention in the frequency domain with a mixture of decomposition kernels. Gap: gains come from temporal-attention design, and all channels are still forecast symmetrically.
- **The MS asymmetry.** These methods forecast every channel; here only the last channel is scored. That leaves the exogenous-fusion question unresolved: how side channels should specifically inform the target forecast.

## Fixed substrate

The Time-Series-Library training loop is frozen: data loaders, `features=MS`, `seq_len=96`, `label_len=48`, `pred_len=96`, Adam at `lr=1e-4`, `train_epochs=10`, `batch_size=32`, `patience=3`, MSE training loss, and scoring on `outputs[:, :, -1:]`. The model receives `x_enc` $(B,\text{seq\_len},\text{enc\_in})$, `x_mark_enc`, `x_dec`, `x_mark_dec` and returns $(B,\text{pred\_len},\text{c\_out})$. Read-only layers are importable: `series_decomp`, `PatchEmbedding`, `DataEmbedding_inverted`, and the Transformer encoder blocks.

## The editable interface

Only `models/Custom.py` is editable — the `Model` class. The contract is `__init__(self, configs)` builds from `configs`; `forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)` maps inputs to $(B,\text{pred\_len},\text{c\_out})$; and `forward(...)` slices the last `pred_len` steps. The target is the **last** channel. A fill may forecast all channels or only the target; only the output shape is fixed.

The starting scaffold is a zero forecaster.

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

Three datasets — **ETTh1** (7 channels, target = oil temperature), **Weather** (21 channels, target = wet-bulb-style last channel), and **ECL** (321 channels, target = last client) — in `features=MS` mode with `seq_len=96`, `label_len=48`, `pred_len=96`, seed 42. Metrics: **MSE** and **MAE** on the target channel only, lower is better.
