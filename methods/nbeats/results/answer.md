# N-BEATS, distilled

N-BEATS (Neural Basis Expansion Analysis) is a pure deep forecaster for univariate point
forecasting: a deep stack of fully-connected **blocks** wired by a **double residual** loop. Each
block reads the residual look-back through a small FC stack, emits coefficients for a **basis
expansion** of both a *backcast* (its reconstruction of the part of the look-back it explains) and a
*forecast* (its contribution to the horizon); the backcast is subtracted from the residual before
the next block, and the forecasts are summed. No recurrence, no statistical core, no hand-set
seasonality — the same architecture applies unchanged across every M4 frequency, and won M4 against
the statistical ensembles and the ES-RNN hybrid.

## Problem it solves

Short-horizon, many-series univariate forecasting (M4): map a length-`L` look-back of one series to
a length-`H` horizon, trained by direct multi-step regression on the percentage metric (sMAPE), with
one architecture across all frequency regimes. The open claim it refutes: that pure ML cannot beat
statistics on such data.

## Key ideas

- **Direct multi-step output.** A feed-forward map emits the whole horizon at once — no roll-out, so
  no error accumulation.
- **Double-residual stacking.** Block `b` gets residual `r_{b-1}`, outputs backcast `x̂_b` and
  forecast `ŷ_b`; the loop is `r_b = r_{b-1} − x̂_b` (peel the explained part off the input) and
  `y += ŷ_b` (accumulate the forecast). Each block models a small correction (trainable depth, like
  ResNet/boosting), and the look-back is sequentially decomposed.
- **Basis expansion (constrained output).** A block maps its hidden representation to a small
  coefficient vector `θ`, and the forecast is `V·θ` for a fixed basis `V`. **Generic** basis: `V`
  is identity (two linear maps `hidden → L` and `hidden → H`) — maximally flexible, the
  raw-accuracy configuration. **Interpretable** bases: a low-degree **polynomial** basis (trend
  blocks) and a **Fourier** sine/cosine basis (seasonality blocks) — each block can only produce
  its family, so stacking a trend stack then a seasonality stack yields an *emergent* learned
  trend/seasonality decomposition with no hand-set period.
- **Stacks + weight sharing.** Blocks are grouped into stacks (one basis type per stack); sharing
  weights across blocks within a stack is a strong regularizer for short series.
- **Input reversal + mask.** The look-back is reversed (most recent first) so recent points sit in a
  stable position; an optional input mask zeroes padded history in the backcast subtraction.
- **Ensembling (training procedure, not architecture).** Average the same model trained over several
  look-back lengths (`L = 2H..7H`), random inits, and loss functions (sMAPE/MASE/MAPE) for stability.

## Final algorithm (double-residual loop)

```
residuals = reverse(x); forecast = x[:, -1:]            # bias the running forecast to the last value
for block in blocks:
    backcast, block_forecast = block(residuals)
    residuals = (residuals - backcast) * input_mask     # peel explained part off the look-back
    forecast  = forecast + block_forecast               # accumulate
return forecast                                          # [batch, H]
```

## Default configuration (M4)

Generic: a few stacks (e.g. 30 blocks) of generic blocks, FC width `W = 512`, 4 FC layers per block,
weight-shared within stack. Interpretable: one trend stack (polynomial degree 2-3) + one seasonality
stack (Fourier harmonics), narrower. Trained DMS on sMAPE (and ensembled over `L`, init, loss) with
Adam; M4 evaluation reports sMAPE / OWA on the official horizon.

## Working code

Faithful to the canonical reference implementation (ServiceNow/N-BEATS): the block FC stack, the
generic / trend / seasonality bases, and the double-residual `NBeats` forward.

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class NBeatsBlock(nn.Module):
    """One block: FC stack -> theta -> basis expansion (backcast, forecast)."""

    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super(NBeatsBlock, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)]
            + [nn.Linear(in_features=layer_size, out_features=layer_size) for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class GenericBasis(nn.Module):
    """Identity basis: split theta into backcast and forecast halves."""

    def __init__(self, backcast_size, forecast_size):
        super(GenericBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    """Low-degree polynomial basis over normalized time (trend)."""

    def __init__(self, degree_of_polynomial, backcast_size, forecast_size):
        super(TrendBasis, self).__init__()
        self.polynomial_size = degree_of_polynomial + 1
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate([
                np.power(np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :]
                for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate([
                np.power(np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :]
                for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)

    def forward(self, theta):
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """Fourier sine/cosine basis over the horizon (seasonality)."""

    def __init__(self, harmonics, backcast_size, forecast_size):
        super(SeasonalityBasis, self).__init__()
        self.frequency = np.append(
            np.zeros(1, dtype=float),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=float) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=float)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=float)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32), requires_grad=False)
        self.backcast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32), requires_grad=False)
        self.forecast_cos_template = nn.Parameter(
            torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32), requires_grad=False)
        self.forecast_sin_template = nn.Parameter(
            torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32), requires_grad=False)

    def forward(self, theta):
        params_per_harmonic = theta.shape[1] // 4
        bc_h = params_per_harmonic
        backcast_harmonics_cos = torch.einsum(
            'bp,pt->bt', theta[:, 2 * bc_h:3 * bc_h], self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum(
            'bp,pt->bt', theta[:, 3 * bc_h:], self.backcast_sin_template)
        backcast = backcast_harmonics_cos + backcast_harmonics_sin
        forecast_harmonics_cos = torch.einsum(
            'bp,pt->bt', theta[:, :bc_h], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum(
            'bp,pt->bt', theta[:, bc_h:2 * bc_h], self.forecast_sin_template)
        forecast = forecast_harmonics_cos + forecast_harmonics_sin
        return backcast, forecast


class NBeats(nn.Module):
    """Double-residual stack of blocks."""

    def __init__(self, blocks: nn.ModuleList):
        super(NBeats, self).__init__()
        self.blocks = blocks

    def forward(self, x, input_mask):
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast
```

The generic stack builds `NBeats` from `NBeatsBlock`s with `GenericBasis(L, H)` and `theta_size = L + H`;
the interpretable stack uses `TrendBasis`/`SeasonalityBasis` with the matching `theta_size`. The same
`NBeats.forward` (reverse, peel backcast, accumulate forecast) drives both.
