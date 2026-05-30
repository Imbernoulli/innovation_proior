# Context

## Research question

Deep learning is fluent at learning *functions* from data — a map from a finite vector to a label or a number. But many problems in science are about *operators*: maps that take an entire function as input and return another function. The solution map of a differential equation is the canonical example — feed it a forcing term or an initial condition (a function) and it returns the solution (a function); so are integral operators, derivative operators, and the input-output behavior of a dynamical system driven by an arbitrary signal. The question is whether a neural network can learn such an operator G: u ↦ G(u) directly from data — input-function/output-function examples — accurately and efficiently, from a relatively small dataset, and in a setting general enough to cover both integral (non-local) and differential operators without restricting the input or output to lie on a regular grid. The difficulty is not merely *can it be represented* but *can it be trained and made to generalize*: the existence of a good approximator is one thing; small optimization and generalization error on finite data is another, and in practice the latter dominate.

## Background

The universal approximation theorem most people know (Cybenko 1989; Hornik 1989) says a neural network with one hidden layer can approximate any continuous *function* to arbitrary accuracy given enough width. A less-known and stronger line of results (Chen & Chen 1995, building on functional-approximation work) extends this to *functionals* (function → real number) and to *operators* (function → function): a single-hidden-layer network can approximate any nonlinear continuous operator to arbitrary accuracy. Concretely, the operator theorem gives an explicit approximating form. Let G map an input function u to an output function G(u); for a query point y in the output domain, G(u)(y) is a real number. Represent u by its values at a fixed finite set of "sensor" locations {x₁, …, x_m}. Then for any ε > 0 there exist integers n, p, m and constants such that

|G(u)(y) − Σ_{k=1}^{p} [ Σ_{i=1}^{n} c_iᵏ σ( Σ_{j=1}^{m} ξ_{ij}ᵏ u(x_j) + θ_iᵏ ) ] · [ σ(w_k·y + ζ_k) ] | < ε

for all u in a compact set of input functions and all y in a compact output domain, where σ is any continuous non-polynomial activation. The two bracketed factors have a clear division of labor: the first depends only on the sampled input function values [u(x₁), …, u(x_m)] (a shallow sub-network producing a scalar), and the second depends only on the query location y (a single neuron σ(w_k·y + ζ_k)); the operator value is a sum of p products of these two pieces.

This theorem is suggestive but, by itself, only a representation result. A parallel from ordinary deep learning makes the gap concrete: the function-approximation theorem says a plain fully-connected network can represent the ideal image classifier, yet in practice convolutional networks vastly outperform fully-connected ones on images — because total error decomposes into approximation, optimization, and generalization error, and the universal theorem bounds only the first. A useful operator network therefore needs an *architecture* whose inductive bias keeps optimization and generalization error small, not just a wide shallow net that the theorem blesses.

The diagnostic phenomenon that motivates a specialized architecture comes from trying the obvious baseline. Since the network input is two pieces — the sensor values [u(x₁), …, u(x_m)] and the query y — the simplest thing is to concatenate them into one vector [u(x₁), …, u(x_m), y] and feed a fully-connected network. This concatenated input has no spatial/sequential structure, so there is no reason to reach for a CNN or RNN; a fully-connected network is the natural baseline. Empirically such a network *can* drive its training error down by growing wider, but its generalization error (test minus train) grows with size — exactly the regime where a better-structured architecture should help.

## Baselines

**Fully-connected network on concatenated input.** Treat [u(x₁), …, u(x_m), y] as a single input vector and regress G(u)(y) with an FNN. By the function-approximation theorem this is a universal approximator. Gap: it ignores that u and y are *different kinds of objects* playing *different roles* (and in d-dimensional problems y is a d-vector whose dimension does not even match the m sensor values), so it has no inductive bias toward the operator structure and generalizes poorly; training error shrinks with size but the generalization gap widens.

**Image-to-image CNN approaches (Winovich et al. 2019; Zhu & Zabaras 2019).** Treat the discretized input and output functions as images and learn an image-to-image map with a CNN. Gap: requires the input sensors *and* the output query points to lie on an equispaced grid, so it is restricted to that special case.

**Parametrized-PDE coefficient identification (Brunton et al. 2016; Rudy et al. 2017).** Assume a parametric PDE form and identify only the unknown coefficients from data. Gap: not a general operator learner — it recovers a few numbers, not an arbitrary function-to-function map.

**Generalized-moving-least-squares network (Trask et al. 2019).** Handles unstructured data but can only approximate *local* operators; cannot represent non-local maps like an integral operator. Gap: locality restriction.

## Evaluation settings

Pre-existing yardsticks and protocol: learn operators for dynamical systems (ODEs) and PDEs and measure mean-squared error on held-out functions. Input function spaces from which u is sampled: a mean-zero Gaussian random field with the RBF kernel k_l(x₁,x₂) = exp(−‖x₁−x₂‖²/2l²), the length-scale l controlling smoothness (larger l ⇒ smoother u); and orthogonal-polynomial spaces V_poly = {Σ_{i=0}^{N−1} a_i T_i(x) : |a_i| ≤ M} built from Chebyshev polynomials T_i, with coefficients a_i sampled uniformly from [−M, M]. Reference outputs G(u)(y) are computed with a Runge–Kutta (4,5) solver for ODEs and a second-order finite-difference solver for PDEs. A data point is a triplet (u, y, G(u)(y)); a single input function u contributes many data points, one per query location y (e.g. 100 trajectories × 100 query points = 10000 points). Test operators include the antiderivative (a linear operator), a nonlinear ODE with g = −s² + u, and a gravity pendulum. Knobs studied: number of sensors m, training-set size, sub-network architecture, optimizer.

## Code framework

The substrate is a standard deep-learning stack: a fully-connected network module (FNN) built from linear layers and an activation, an MSE loss, and a training loop over (input, query, target) batches. The two network inputs — the sensor-value vector and the query point — are available separately. The open slot is the model that maps these two inputs to the scalar operator value.

```python
import torch
import torch.nn as nn


class FNN(nn.Module):
    """A plain fully-connected network: the already-available primitive."""
    def __init__(self, layer_sizes, activation=nn.Tanh()):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.linears = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = self.activation(x)
        return x


class OperatorModel(nn.Module):
    """Maps (sensor values of u, query point y) -> G(u)(y).
    Pre-method skeleton: how the two inputs are encoded and combined is the open slot."""
    def __init__(self, m, ...):
        super().__init__()
        # TODO: the sub-network(s) / parameters the architecture will need,
        #       given inputs  u_sensors in R^m  and  y in R^d
        pass

    def forward(self, u_sensors, y):
        # u_sensors: [batch, m] sampled input-function values
        # y:        [batch, d] query locations
        # TODO: produce scalar G(u)(y) for each (u, y)
        pass


# training loop (already known):
# for u_sensors, y, target in loader:
#     pred = model(u_sensors, y)
#     loss = ((pred - target) ** 2).mean()
#     loss.backward(); opt.step(); opt.zero_grad()
```
