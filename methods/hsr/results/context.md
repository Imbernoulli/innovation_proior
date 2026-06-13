# Context: predicting noisy continuous targets with feed-forward nets (early–mid 1990s)

## Research question

I have a feed-forward network doing nonlinear regression: it maps an input `x` to a
continuous target `d` and I train it by minimizing the sum of squared errors. It gives me a
single predicted value per input. The pain is that this single number is not enough for the
problems I actually have. The noise in the target is not the same everywhere in input space:
in some regions `d` is a tight, almost deterministic function of `x`, and in other regions
the same `x` produces a wide spread of `d` values — the data is *heteroscedastic*. A
least-squares network gives me one prediction and, at best, one global error bar computed
from the overall residual; it cannot tell me that my prediction is trustworthy here and
shaky there. What I want is a regressor that, for each input, reports not just a predicted
value but *how confident that value is at this particular input* — an error bar that is
itself a function of `x`. And I want to get it from the same feed-forward, gradient-descent
machinery I already use, without requiring enough data to estimate an entire conditional
density non-parametrically. The concrete instance driving this is a climate-physics emulator:
a network must map a 124-dimensional atmospheric column state to a 128-dimensional vector of
physics tendencies and fluxes, where different output channels and different atmospheric
regimes carry very different amounts of intrinsic spread, and the yardstick is a normalized
mean-squared error on the predicted values. The question is how to train a feed-forward net
so that it represents the *input-dependent uncertainty* of its targets while preserving the
point prediction that the emulator is scored on.

## Background

The field state. Feed-forward nets trained by back-propagation are the standard tool for
nonlinear regression, and there is a clean probabilistic understanding of what they compute.
The result everyone leans on (Bishop's 1994 analysis; the conditional-average property): a
network trained to minimize the sum-of-squares error

```
E^S(w) = (1/2) Σ_q Σ_k [ f_k(x_q; w) − t_k^q ]^2
```

converges, in the infinite-data limit, to the **conditional average** of the targets,
`f_k(x; w*) = ⟨t_k | x⟩`. You can see it two ways. Functional differentiation of the
infinite-data error sends the optimum to `⟨t_k|x⟩`. Or you add-and-subtract `⟨t_k|x⟩` inside
the square and the error splits into

```
E^S = (1/2) Σ_k ∫ [ f_k(x,w) − ⟨t_k|x⟩ ]^2 p(x) dx
    + (1/2) Σ_k ∫ [ ⟨t_k^2|x⟩ − ⟨t_k|x⟩^2 ] p(x) dx,
```

where only the first term depends on the weights and is killed when the net equals the
conditional average; the second, weight-independent term is exactly the *average variance of
the targets around that conditional average*. So least squares already, implicitly, has a
variance in it — but it is collapsed into a single residual number at the minimum, not a
function of `x`.

The maximum-likelihood reading of the same thing (Rumelhart, Durbin, Golden & Chauvin 1995;
Buntine & Weigend 1991). A network output need not be read as "a number" but as the *mean of
a target distribution*. If you assume the target is that mean plus zero-mean Gaussian noise,
`P(d_i | x_i, N) = K · exp(−Σ (y_i − d_i)^2 / (2σ^2))`, then doing gradient descent on the
**negative log-likelihood** `C = −Σ_i ln P(d_i | x_i, N)` of the data given the network is,
after taking the log, exactly squared-error training. This is the load-bearing principle:
the loss is a negative log-likelihood, and which loss you get depends on which target-noise
distribution you assume. Bishop's analysis makes the cost of the standard assumption
explicit: least squares corresponds to a Gaussian whose standard deviation `σ` is a
**single global constant**, and the maximum-likelihood value of the corresponding variance
is just the residual sum-of-squares divided by the number of scalar target observations,

```
σ^2 = (1/nc) Σ_q Σ_k [ f_k(x_q; w*) − t_k^q ]^2 .
```

One number for the whole input space.

A primitive worth having on the shelf. When a quantity that a network output must represent
is constrained to be positive — a scale, a variance — a clean way to honor the constraint is
to let the network produce an *unconstrained* output and pass it through an exponential:
`s = exp(z)`. The network's raw output `z` ranges over all of the reals, `exp(z)` is always
positive, and this also avoids the pathological configurations where the positive quantity is
driven to exactly zero (Bishop 1994 uses `σ_i = exp(z_i^σ)` for precisely this; means stay
linear outputs `μ = z^μ`). The exponential's derivative is itself, which keeps the
back-propagated gradient simple.

The contemporary alternatives for getting more than the mean. If you want the *shape* of the
target distribution rather than just its center, you can go fully general — model the whole
conditional density as a mixture of Gaussians whose mixing weights, centers, and widths are
all network outputs (Bishop's mixture-density networks, 1994), which can represent multimodal
and arbitrary conditional densities. Or estimate the density by binning / Monte-Carlo /
hidden-Markov methods (Srivastava & Weigend 1994; Fraser & Dimitriadis 1994). The trade is
data: a non-parametric or many-component estimate of the conditional density's *shape*
requires large quantities of data to pin down, which is exactly what is scarce in the regimes
I care about.

## Baselines

These are the prior approaches a new regressor would be measured against and would react to.

**Least-squares feed-forward regression (the default).** A multilayer net, linear output
unit(s), tanh hidden units, trained by gradient descent on `E^S`. Core idea: directly fit the
conditional mean `⟨t|x⟩`. It is simple, well-understood, and optimal *as a point predictor*
under squared-error loss. **Limitation:** the only uncertainty it exposes is the global
residual variance `σ^2 = residual SSE / (nc)` — a single number that is the same for every
input. In a heteroscedastic problem this is wrong everywhere except on average: it
over-states confidence in noisy regions and under-states it in clean ones, and it gives the
optimizer no way to treat a high-noise example differently from a low-noise one. All examples
push on the weights with equal force regardless of how informative they are.

**Mixture-density / full-conditional-density networks (Bishop 1994).** Core idea: make every
parameter of a Gaussian-mixture conditional density a network output —

```
p(t | x) = Σ_{i=1}^m α_i(x) φ_i(t | x),   φ_i = N(μ_i(x), σ_i(x)^2 I),
```

with mixing coefficients `α_i(x) = softmax(z_i^α)` summing to one, widths
`σ_i(x) = exp(z_i^σ)`, centers `μ_i(x) = z_i^μ`, and train by the negative log-likelihood
`E^q = −ln{ Σ_i α_i(x_q) φ_i(t_q | x_q) }`. This represents arbitrary, even multimodal,
conditional densities — far more than the mean. **Limitation:** the generality is paid for in
data and in optimization difficulty — fitting `m` competing components with softmax mixing
coefficients on a high-dimensional output, across all training budgets, is heavy when
what the problem actually exhibits is unimodal-but-input-dependent spread, not multimodality.
The full machinery is more than the situation calls for, and the extra components are hard to
identify from limited data.

**Non-parametric density estimates (binning / Monte-Carlo / HMM; Srivastava & Weigend 1994;
Fraser & Dimitriadis 1994).** Core idea: estimate the conditional density's shape directly
from data without assuming a parametric form. **Limitation:** data-hungry — pinning down the
shape of a high-dimensional conditional density empirically needs far more samples than
fitting a few summary statistics, and these methods do not slot into a single feed-forward
gradient-descent training run.

## Evaluation settings

The natural yardsticks for the climate-emulation instance are the dataset, metrics, and
protocol already fixed by the emulator harness:

- **Dataset:** ClimSim low-resolution (Yu et al., NeurIPS 2023 D&B), from
  the E3SM-MMF multi-scale climate model. The standard baseline setup maps a 124-dim
  atmospheric column state to a 128-dim target vector of vertically resolved tendencies and
  single-level fluxes. Inputs and outputs are normalized by the fixed pipeline.
- **Primary metric:** Normalized MSE, `NMSE = MSE / Var(target)`, lower is better, computed
  on the predicted value. **Secondary:** R² (higher better), RMSE, and separate multi-level
  (`ml_nmse`) and single-level (`sl_nmse`) breakdowns.
- **Protocol:** fixed dataset loading, normalization, train/val/test splits, optimizer and
  schedule, and the loss interface are all fixed by the harness; only the model architecture
  and the per-batch loss it computes are editable. The natural validation loop also tracks
  probabilistic scores such as likelihood-style losses and sample-based CRPS for stochastic
  regressors, while point scores are still computed from the predicted value.

## Code framework

The model plugs into a regression harness that already loads and normalizes the data, owns
the optimizer and schedule, runs a standard minibatch training loop, and computes the
evaluation metrics on the model's predictions. What is *not* settled is the model itself and
the per-example training objective — those are the empty slots. The substrate is the generic
PyTorch regression machinery that exists before any of this: linear layers, layer
normalization, dropout, ReLU, a feed-forward model exposing a training-time call and a point
prediction for evaluation, and a training step that computes a scalar loss from the model's
output and the target and back-propagates it.

```python
import torch
import torch.nn as nn


class Custom(nn.Module):
    """The regression model. Maps a (batch, input_dim) atmospheric state to a
    (batch, output_dim) prediction. Architecture is not yet decided."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # TODO: the model we will design.
        pass

    def forward(self, x):                  # x: (batch, input_dim)
        # TODO: produce the training-time output(s)
        raise NotImplementedError

    def predict_mean(self, x):             # x: (batch, input_dim)
        # TODO: return the point prediction (batch, output_dim)
        raise NotImplementedError


def training_loss(model, x, y, epoch, total_epochs):
    """Per-batch scalar loss the harness back-propagates. By default the harness
    scores predictions against targets with mean-squared error."""
    out = model(x)
    pred = model.predict_mean(x)
    # TODO: the objective we will train against, given the training output
    #       and the targets y. Default substrate:
    return nn.functional.mse_loss(pred, y)


# existing fixed training loop the model plugs into
def train(model, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        for batch in data_loader:
            x, y = batch['x'], batch['y']          # normalized state, normalized target
            optimizer.zero_grad()
            loss = training_loss(model, x, y, epoch, epochs)
            loss.backward()                        # back-prop fills gradients
            optimizer.step()


# existing fixed evaluation: point metrics on the model's prediction
def evaluate(model, data_loader):
    # NMSE / R2 / RMSE computed against model.predict_mean(x), the predicted value
    ...
```
