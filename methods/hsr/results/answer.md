# Heteroskedastic Regression (HSR), distilled

HSR turns a point-prediction regression network into one that also reports an
**input-dependent variance**. The network outputs two quantities per target dimension — a
mean `μ(x)` and either a variance `σ²(x)` or, in the ClimSim convention, a log-precision
`ℓ(x)=ln(1/σ²(x))`. It warms up the mean with mean-squared error, then trains with the
**Gaussian negative log-likelihood** of the targets under an input-dependent-variance
Gaussian. At inference the mean is the prediction; the variance is the error bar.

## Problem it solves

Standard least-squares regression recovers only the conditional mean `⟨d|x⟩` and a single
global residual variance — wrong for *heteroscedastic* data, where the target noise level
varies across input space (e.g. calm vs. turbulent atmospheric regimes). HSR makes the error
bar a learned function of `x`, fit from the same feed-forward gradient-descent training, and
in doing so also sharpens the mean (high-noise outliers stop stealing capacity from
well-determined regions).

## Key idea

Squared error is the negative log-likelihood of a Gaussian with **constant** variance — which
is exactly why it only recovers the mean. So instead assume a Gaussian whose variance depends
on the input,

```
P(d_i | x_i) = [2π σ̂²(x_i)]^{-1/2} · exp{ −[d_i − ŷ(x_i)]² / (2 σ̂²(x_i)) },
```

let the network output both `ŷ(x)` and `σ̂²(x)`, and minimize the negative log-likelihood. The
total cost is

```
C = ½ Σ_i { [d_i − ŷ(x_i)]² / σ̂²(x_i)  +  ln σ̂²(x_i)  +  ln 2π }.
```

Two competing terms: an inverse-variance-weighted squared error and a `ln σ̂²` penalty on
claiming large variance. Holding `ŷ` fixed, the per-pattern minimizer is `σ̂² = [d − ŷ]²` — so
the variance head learns to predict the **squared residual**, the target the likelihood
invents (no separate variance label needed). If `σ̂²` is held constant, `C` reduces to ordinary
sum-squared-error backprop.

## Why each piece

- **Two heads (mean + variance), trained by NLL after a mean warmup.** MSE's optimum is the
  conditional mean only, with a single global residual variance. The input-dependent-variance
  Gaussian likelihood is the objective whose optimum fits both `μ(x)` and `σ²(x)`.
- **Automatic weighted regression.** Differentiating `C` gives the mean's update
  `Δw_{ŷj} = η · (1/σ̂²(x_i)) · [d_i − ŷ(x_i)] · h_j(x_i)` — the ordinary delta rule scaled by
  `1/σ̂²`. Low-noise patterns get a larger effective learning rate, high-noise patterns a
  smaller one; outliers from noisy regions stop dragging the fit. The variance update,
  `Δw_{σ²k} = η · (1/(2σ̂²)) · { [d_i − ŷ(x_i)]² − σ̂²(x_i) } · h_k(x_i)`, regresses `σ̂²` onto
  the squared errors.
- **Staged training (avoids the early-weighting trap).** While `ŷ` is still bad, residuals are
  large everywhere; the `1/σ̂²` weighting would tag accidentally-low-residual patterns as
  low-noise and freeze the high-residual ones out, mistaking underfitting for noise. The
  careful small-data schedule is: **(I)** mean only, plain squared error, until a held-out
  squared error bottoms out; **(II)** keep the mean fixed and train the variance head on
  honest residuals, initializing its bias to `ln(global MSE)`; **(III)** unfreeze and minimize
  the full `C` jointly. The ClimSim implementation uses the compact version of this logic:
  plain MSE for the first third of epochs, then direct joint training with the scaled NLL.
- **Exponential (log-variance) parameterization.** `σ̂²` must be positive. As `σ̂²→0`,
  `1/σ̂²` blows up while `ln σ̂²` runs to `−∞`, so a raw variance output is numerically unsafe.
  Output an unconstrained pre-activation and exponentiate:
  `σ̂²(x) = exp[Σ_k w_{σ²k} h_k(x) + β]`. So the head naturally predicts `ln σ̂²` (or, with a
  sign flip, the log-precision `ln(1/σ̂²)`); positivity is free, the configuration `σ̂²=0` is
  unreachable, and the chain-rule factor `dσ̂²/dz = σ̂²` turns the variance derivative into
  the log-scale likelihood derivative `½(1 − (d−ŷ)²/σ̂²)`. The homoscedastic init is one bias
  `β = ln(global MSE)`.
- **Two regularized MLPs.** The variance is generally a different function of `x` than the
  mean, so it needs its own capacity. The ClimSim baseline uses two MLPs with the same input:
  one for the mean and one for log-precision, both using linear layers, layer normalization,
  dropout, and ReLU.
- **Diagonal Gaussian for vector targets.** With many output dimensions, assume conditional
  independence: the joint NLL is the sum of per-dimension NLLs. (Full covariance with
  cross-correlations is the richer extension.)
- **Inference returns the mean.** Point metrics (NMSE/R²/RMSE) score the predicted value;
  the variance is the uncertainty channel (error bars / sampling / calibration).

## Final objective

Per output element, dropping the `ln 2π` constant and multiplying by 2, with the head emitting
log-precision `ℓ = ln(1/σ̂²)` (so `σ̂² = exp(−ℓ)`, precision `τ = exp(ℓ)`):

```
NLL_elem = (d − ŷ)² · τ − ℓ = (d − ŷ)² · exp(ℓ) − ℓ.
```

Equivalently, with the head emitting log-variance `v = ln σ̂²`:
`NLL_elem = (d − ŷ)² · exp(−v) + v` (same object; sign convention on the readout).

## Working code

Faithful to the canonical ClimSim HSR implementation (two MLPs, log-precision output,
MSE warmup then Gaussian NLL, mean used for point inference), filling the model and loss slots of
the regression harness.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Linear -> LayerNorm -> Dropout -> ReLU blocks, then a linear output."""
    def __init__(self, in_dim, out_dim, hidden_dim=512, layers=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(p=dropout),
            ))
        self.final_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for block in self.blocks:
            x = F.relu(block(x))
        return self.final_linear(x)


class HeteroskedasticRegression(nn.Module):
    """Two MLPs: per-dimension mean mu(x) and log-precision ln(1/sigma^2)."""

    def __init__(self, input_dim, output_dim, hidden_dim=512, layers=1, dropout=0.0):
        super().__init__()
        self.mean = MLP(input_dim, output_dim, hidden_dim, layers, dropout)
        self.logprec = MLP(input_dim, output_dim, hidden_dim, layers, dropout)

    def forward(self, x):
        return self.mean(x), self.logprec(x)

    def predict_mean(self, x):
        # point inference: the value NMSE/R2/RMSE metrics score
        mu, _ = self(x)
        return mu

    def predict(self, x):
        # mean and predictive std, for error bars / calibration
        mu, logprec = self(x)
        std = torch.exp(-0.5 * logprec)                              # sigma = exp(logprec)^(-1/2)
        return mu, std


def gaussian_nll(mu, logprec, target):
    # twice the per-element NLL, with ln 2pi dropped:
    #   (d - mu)^2 / sigma^2 + ln sigma^2  =  (d - mu)^2 * exp(logprec) - logprec
    prec = torch.exp(logprec)                                        # tau = 1/sigma^2
    return (prec * (target - mu) ** 2 - logprec).mean()


def training_loss(model, x, y, epoch, total_epochs):
    mu, logprec = model(x)
    if epoch < total_epochs / 3:
        # Phase-I warmup: mean only, plain squared error -- the 1/sigma^2
        # weighting must not reshape learning while mu is still bad.
        return ((y - mu) ** 2).mean()
    # scaled heteroskedastic objective: mean and precision co-adapt
    loss = gaussian_nll(mu, logprec, y)
    return torch.clamp(loss, min=-1e5, max=1e5)                      # coarse runaway guard
```

Two MLPs and a swap of the loss from MSE to the scaled Gaussian NLL after a mean-only warmup are
the whole implementation; everything else — the `1/σ²` weighted regression, the log-variance or
log-precision positivity convention, and the staged-training rationale — falls out of minimizing
the input-dependent-variance Gaussian's negative log-likelihood.
