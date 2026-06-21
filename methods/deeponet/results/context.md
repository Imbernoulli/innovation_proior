# Context

## Research question

Deep learning is fluent at learning *functions* from data: a map from a finite vector to a label, a class score, or a number. Many scientific problems ask for something larger. They ask for an *operator*: a map that takes an entire function as input and returns another function. A differential-equation solution map is the canonical example: feed in a forcing term, initial condition, boundary condition, or material field, and the output is a solution function. Integral transforms, response maps of dynamical systems, and some nonlocal differential operators have the same shape.

The task is to learn such a map \(G:u\mapsto G(u)\) from input-function/output-function examples. The input function is observed at a fixed set of sensor locations, and for a query location \(y\) in the output domain the supervised scalar is \(G(u)(y)\). The learner is given a finite collection of input functions, each paired with output values at scattered query locations, and must predict \(G(u)(y)\) for new functions and new query points.

## Background

The familiar universal approximation theorem says that a sufficiently wide neural network can approximate continuous functions on compact finite-dimensional domains. That result is not enough for a function-to-function map. A less familiar line of work on nonlinear functionals and operators gives an operator analogue. If \(V\) is a compact family of input functions, \(K_2\subset\mathbb{R}^d\) is a compact output domain, \(G:V\to C(K_2)\) is continuous, and \(\sigma\) is continuous and non-polynomial, then for every \(\varepsilon>0\) there are sensor locations \(x_1,\ldots,x_m\), integers \(n,p\), and constants such that

\[
\left|G(u)(y)-\sum_{k=1}^{p}\left[\sum_{i=1}^{n}c_i^k\sigma\left(\sum_{j=1}^{m}\xi_{ij}^ku(x_j)+\theta_i^k\right)\right]\sigma(w_k\cdot y+\zeta_k)\right|<\varepsilon
\]

for every \(u\in V\) and \(y\in K_2\). The expression handles sampled information about the input function (through the inner sums over the sensor values \(u(x_j)\)) and the output query coordinate \(y\) separately, and recombines them through a finite sum of products.

This theorem controls approximation error for large enough networks. The overall performance of a neural model also depends on optimization error and generalization error, as with ordinary function-fitting networks: a fully connected network is a universal approximator, and convolutional structure generalizes well when the data have image structure.

## Baselines

**Fully connected network on concatenated input.** Treat \([u(x_1),\ldots,u(x_m),y]\) as one vector and regress \(G(u)(y)\). The input has no obvious grid or sequence layout in the most general setting, so a plain fully connected network is a natural baseline.

**Image-to-image convolutional models.** Discretize both the input and output functions on regular grids and learn an image-to-image map with convolutional networks.

**Parameterized-equation identification.** Assume a PDE form and identify unknown coefficients or forcing terms from data, when the governing family is already known.

**Local unstructured operators.** Generalized moving-least-squares networks handle point clouds and local stencils.

## Evaluation settings

Training data are triplets \((u,y,G(u)(y))\). One sampled input function can yield many triplets by pairing it with many query points. Input functions are drawn from spaces such as mean-zero Gaussian random fields with RBF kernels \(k_l(x_1,x_2)=\exp(-\|x_1-x_2\|^2/(2l^2))\), where larger \(l\) gives smoother functions, or from finite Chebyshev-polynomial families \(\sum_{i=0}^{N-1}a_iT_i(x)\) with bounded coefficients. Reference outputs for ODE examples come from Runge-Kutta solvers; PDE examples use finite-difference solvers.

Diagnostic operators include the antiderivative \(u\mapsto \int_0^x u(\tau)\,d\tau\), nonlinear ODE solution maps such as \(s'(x)=-s(x)^2+u(x)\), pendulum dynamics with forcing, and PDE solution maps. The measured quantities are training mean-squared error, test mean-squared error, and their gap under changes in sensor count, training-set size, input-function smoothness, and network size.

## Code framework

The substrate is a standard neural-network training loop with two already-separated inputs: the vector of input-function sensor values and the output query coordinate. The open slot is the model that maps those two inputs to one scalar prediction.

```python
import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh()):
        super().__init__()
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        )
        self.activation = activation

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.activation(x)
        return x


class OperatorModel(nn.Module):
    def __init__(self, m, query_dim):
        super().__init__()
        # TODO: define the trainable map for (u_sensors, y) -> scalar output.
        pass

    def forward(self, u_sensors, y):
        # u_sensors: [batch, m]
        # y:         [batch, query_dim]
        # TODO: return G(u)(y) for each paired input/query sample.
        pass


# for u_sensors, y, target in loader:
#     pred = model(u_sensors, y)
#     loss = ((pred - target) ** 2).mean()
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
```
