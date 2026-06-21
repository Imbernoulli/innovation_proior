The problem I set out to solve is short-horizon, many-series univariate forecasting in the mold of the M4 competition: one hundred thousand short real-world series spanning Yearly through Hourly sampling, each its own univariate problem, each scored by a percentage error (sMAPE, combined with MASE into OWA against naive-2). The accepted wisdom on this kind of data is that pure machine learning loses to statistics. The competition's own winner is a hybrid that grafts an exponential-smoothing model onto a dilated-LSTM stack, and the pure-ML entrants came in behind the classical benchmarks. But I do not believe that conclusion follows from the evidence. The pure-ML entrants were off-the-shelf MLPs and RNNs with no inductive bias suited to short series, so they overfit. The hybrid wins not because the statistics are essential but because *some* structure — de-seasonalization, level normalization, residual handling — is essential, and the hybrid happens to supply it through a hand-built statistical core that is intricate and frequency-specific. The statistical benchmarks, for their part, are low-capacity per-series models with hand-set seasonality that share nothing across the hundred thousand series. So the real question is whether I can build those structural biases directly into a pure deep network — no hand-set seasonality, no statistical model bolted on — and have one architecture work unchanged across every frequency. That is the target: a single, frequency-agnostic, pure-deep, direct-multi-step forecaster that also, ideally, exposes a trend and a seasonality component as a free by-product rather than having that decomposition imposed by hand.

I propose N-BEATS (Neural Basis Expansion Analysis): a deep stack of fully-connected blocks wired together by a double-residual loop, where each block reads what the previous blocks left unexplained and emits coefficients for a basis expansion of both a backcast and a forecast. The output strategy comes first because it constrains everything else. A recurrent roll-out predicts one step and feeds it back, so a small per-step bias compounds over the horizon; I avoid this entirely by emitting the whole horizon at once — a direct-multi-step map from the length-$L$ look-back to the length-$H$ forecast, with no recursion and therefore no accumulation. A feed-forward network whose output spans all $H$ steps does exactly this, so the skeleton is feed-forward and the whole design problem is what to put between input and output so the network has the right biases for short, noisy single series instead of being a big MLP that overfits.

The first bias is sequential refinement. A single network forced to explain the entire look-back at once must model trend, seasonality, and noise simultaneously, and it will spend capacity wherever the loss is largest — the trend/season starvation that motivates classical decomposition. Boosting and deep residual learning both solve a version of this by fitting the target as a sum of successive corrections, each new learner handling what the running sum left unexplained. I carry that to forecasting: build the network as a stack of blocks where each block sees a *residual* input and produces two things — a partial forecast $\hat y_b$ (its contribution to the horizon) and a backcast $\hat x_b$ (its reconstruction of the part of the look-back it just used). The defining double-residual loop, with $r_0$ the look-back, is

$$r_b = r_{b-1} - \hat x_b, \qquad y = y + \hat y_b.$$

The backcast residual flows backward: each block strips its explained component out of the input, so downstream blocks face a cleaner, simpler signal. The forecast residual flows forward as a running sum. This double-residual stacking is what makes a very deep stack trainable — each block only models a small correction, exactly as in ResNet — and it is also what makes the decomposition emergent, since the look-back is peeled apart piece by piece by whatever each block learns rather than by a hand-set moving average.

What is inside one block is the second bias. I want each block to map its residual input to *coefficients*, not directly to forecast values, and then expand those coefficients through a fixed basis. If a block outputs the $H$ forecast values directly it is unconstrained and free to produce an overfit wiggle that fits training noise. If instead it outputs a small vector $\theta$ and the forecast is $\hat y = V\theta$ for a fixed basis matrix $V$, the forecast is confined to the span of $V$; choose $V$ to be a meaningful family and the block can only produce forecasts of that shape. So a block is a small fully-connected stack — a few $\text{Linear}+\text{ReLU}$ layers of width $W$ that read the residual look-back into a hidden representation — followed by two *separate* linear projections to a backcast coefficient vector and a forecast coefficient vector, then a basis expansion of each. The fully-connected stack is the learned part; the basis is the inductive bias on the output. Forecast and backcast share the hidden representation but use different basis matrices, because the backcast must span length $L$ and the forecast must span length $H$. The backcast's accuracy is a means, not the end — its only job is to clean the residual — but it must use the *same* basis family as the forecast, so that the kind of component a block removes from the input is the same kind it adds to the forecast.

I want two basis designs because they answer different needs, and crucially they share the exact same architecture, differing only in the basis matrices, so "one architecture across all frequencies" holds for both. The *generic* design puts no structure on the basis at all: $\theta$ itself is the output, so the basis is the identity and the block's two projections map the hidden vector straight to $L$ backcast values and $H$ forecast values. This is maximally flexible — the pure-deep, no-prior version — and it is what I reach for when raw accuracy across heterogeneous series matters more than interpretability, since the residual loop alone already supplies decomposition-by-refinement. The *interpretable* design constrains each block to a classical family. A trend block uses a low-degree polynomial basis over normalized time $t = [0, 1, \dots, H-1]/H$,

$$\hat y = \sum_{p=0}^{P} \theta_p\, t^p,$$

with $P$ small (2 or 3): a low-degree polynomial is smooth on the scale of the horizon, so the block can only produce a trend and its $\theta$ are directly interpretable as polynomial coefficients. A seasonality block uses a Fourier basis at the harmonics of the horizon,

$$\hat y = \sum_l \theta_l \cos(2\pi l t) + \theta'_l \sin(2\pi l t),$$

so it can only produce a periodic pattern, with $\theta$ the harmonic amplitudes. Stacking a trend stack then a seasonality stack gives a learned seasonal-trend decomposition — the same split STL imposes by hand, here emerging from the residual loop and the constrained bases with no per-frequency period supplied by me: the trend stack peels the trend off the look-back and contributes the trend forecast, and the seasonality stack works on the de-trended residual.

Two further structural choices matter. First, organize blocks into stacks, one basis type per stack, and let blocks within a stack *share weights*. Weight-sharing is a strong regularizer — it forces every block in, say, the trend stack to use one fully-connected map, so the stack learns a single trend operator applied residually rather than many independent ones — and on short M4 series, where overfitting is the enemy, that is precisely what the pure-ML entrants lacked. The generic configuration uses several stacks of generic blocks (around thirty blocks total, width $W = 512$, four FC layers per block); the interpretable configuration uses one trend stack of polynomial degree two or three followed by one seasonality stack. Second, the input orientation: the most recent look-back values matter most, so I reverse the look-back to put the most recent point first, which keeps recent points in a stable position regardless of $L$, and I bias the running forecast to start at the last observed value $x[:, -1:]$. If the harness supplies a mask for missing or padded history, the backcast subtraction is masked too, so padded positions never pollute the residual.

For the objective, M4 lives at wildly different magnitudes, so an MSE would let the few large series dominate; I train on the percentage metric I am judged by — sMAPE, $\frac{200}{H}\sum |y - \hat y| / (|y| + |\hat y|)$, which is scale-free per series and puts every series on equal footing — computed over the whole horizon at once, which is what the direct-multi-step output was for. Finally, for robustness across a hundred thousand short series, a single trained model is sensitive to initialization and to the chosen look-back length; the clean fix that does not touch the architecture is to ensemble, training the same model over several look-back lengths ($L = 2H, 3H, \dots, 7H$), several random initializations, and several losses (sMAPE, MASE, MAPE), and averaging the forecasts. Every member is the same pure-deep architecture, so the "pure deep, one architecture" claim holds while the ensemble averages away the variance a single short-series fit carries — the stability the statistical ensembles got from combining distinct methods, obtained here from a single architecture. The thesis closes on itself: every structural bias that made the hybrid work — direct-multi-step output, sequential residual refinement, basis-constrained outputs, weight-sharing, percentage-metric training, ensembling — is present, and none of it is statistics.

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
