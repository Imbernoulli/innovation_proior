**Problem.** The four baselines plateaued at mean SMAPE ~12.11 (TimesNet 12.803 / 10.089 / 13.442),
and the plateau is regime-split: each rung is *one fixed lens* (pooling ladder, period FFT, patch
attention) that fits periodic regimes but stalls on trend-dominated Yearly, where the SMAPE has barely
moved across the whole ladder. The unsolved problem is a single inductive structure forced to serve
trend-only and strong-seasonal series at once.

**Key idea (N-BEATS).** A pure deep stack of fully-connected **blocks** wired by a **double residual**
loop. Each block reads the residual look-back through a small FC stack and emits a *backcast* (its
reconstruction of the part of the input it explains) and a *forecast* (its contribution to the horizon)
via a basis expansion; the backcast is subtracted from the running residual (`r_b = r_{b-1} − x̂_b`)
and the forecasts accumulate (`y += ŷ_b`). The decomposition is *emergent and per-series* — each block
peels off whatever earlier blocks could not explain — so one architecture adapts to pure-trend Yearly
and seasonal Monthly without a regime-specific lens.

**Why.** Sequential residual refinement (boosting/ResNet on the look-back) makes a deep stack trainable
as small corrections and supplies an adaptive decomposition, the exact thing the per-regime baseline
split showed was missing. Direct-multi-step output keeps the no-error-accumulation property held since
DLinear. The **generic** identity basis (`hidden → seq_len` backcast, `hidden → pred_len` forecast) is
the raw-accuracy configuration; weight-sharing within the stack regularizes the short M4 series. The
forecast is **seeded at the last observed value** so blocks learn the deviation from persistence — a
sensible prior, strongest on the trend regime.

**Harness grounding.** Univariate (`enc_in=1`): squeeze the channel to feed `[B, seq_len]`, unsqueeze
`[B, pred_len]` back. No time marks (N-BEATS reads only the raw look-back). Repurpose the fixed
protocol's width as FC width `W=512` (the wide channels the previous rungs stranded now feed an FC
block that can use them), with weight-shared generic blocks, 4 FC layers each. Dense mask (all-ones).
**No ensembling** under the 10-epoch cap — the honest limitation: N-BEATS' published M4 result leans on
ensembling over look-back/init/loss, which this protocol withholds.

**Hyperparameters.** Generic config (ServiceNow generic factory): a flat stack of `n_blocks=6`
*distinct* generic blocks, `layers=4` FC layers each, `layer_size=512` (from `d_model`),
`theta_size = seq_len + pred_len`; forecast seeded at last value. Adam `lr=1e-3`, batch 16, 10 epochs,
patience 3, SMAPE loss.

**What it must clear.** TimesNet's 12.803 / 10.089 / 13.442. Strongest case on **Yearly** (trend
regime where every baseline plateaued — persistence-seeded refinement is built for it; expect to beat
13.44). **Quarterly** the hardest (generic FC must rediscover the sharp period TimesNet's FFT encodes —
competitive, possibly just short of 10.09). **Monthly** a tie-to-slight-win near 12.80. The endpoint
claim rests on a *mean* win driven by Yearly. If Yearly does not move, that falsifies the
"refinement, not capacity" bet and points back to ensembling as the withheld ingredient.

```python
import numpy as np
import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    """One block: FC stack -> theta -> identity (generic) basis split."""

    def __init__(self, input_size, theta_size, backcast_size, forecast_size, layers, layer_size):
        super(NBeatsBlock, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, layer_size)]
            + [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, x):                                # x: [B, input_size]
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
        theta = self.basis_parameters(h)                # [B, theta_size]
        backcast = theta[:, :self.backcast_size]        # generic identity basis
        forecast = theta[:, -self.forecast_size:]
        return backcast, forecast


class Model(nn.Module):
    """N-BEATS (generic): double-residual stack of FC blocks, direct multi-step."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        layer_size = getattr(configs, 'd_model', 512)
        layers = getattr(configs, 'nbeats_layers', 4)
        n_blocks = getattr(configs, 'nbeats_blocks', 6)
        theta_size = self.seq_len + self.pred_len       # generic: backcast (L) + forecast (H)

        # generic config: a flat stack of distinct generic blocks (ServiceNow generic factory)
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=self.seq_len,
                theta_size=theta_size,
                backcast_size=self.seq_len,
                forecast_size=self.pred_len,
                layers=layers,
                layer_size=layer_size,
            )
            for _ in range(n_blocks)
        ])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # univariate: [B, L, 1] -> [B, L]
        x = x_enc[:, :, 0]
        residuals = x.flip(dims=(1,))                   # most recent first
        forecast = x[:, -1:]                            # seed at last observed value (persistence)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast            # peel explained part (dense mask -> no scaling)
            forecast = forecast + block_forecast
        return forecast.unsqueeze(-1)                   # [B, H, 1]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]       # [B, H, 1]
        return None
```
