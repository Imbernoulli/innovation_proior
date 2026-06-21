The problem is to learn an operator, not merely a function. We are given examples of a map G that takes an entire input function u and returns an output function G(u), and we want to approximate it from finite data. Because a neural network cannot ingest a function directly, we represent the input by its values at a fixed set of sensor locations, [u(x_1), ..., u(x_m)], and we supervise the learner at arbitrary query points y by the scalar target G(u)(y). One sampled input function can therefore generate many training triples by pairing it with many query coordinates, and the model must be able to evaluate the predicted output field at any location without being tied to a fixed grid.

Existing approaches fall short for different reasons. A fully connected network that simply concatenates the sensor vector and the query coordinate into one flat input is universal in principle, but it destroys the structural distinction between "which input function am I transforming" and "where in the output domain am I querying." That missing inductive bias shows up as a large generalization gap even when training error is driven down. Image-to-image convolutional models assume regular grids for both input sensors and output locations, which is too restrictive for scattered measurements. PDE-identification methods only work when the governing equation family is already known, and local unstructured operators struggle with nonlocal maps such as integrals or global solution operators.

The right model is DeepONet. DeepONet explicitly separates the two roles required by the operator approximation problem. A branch network reads the sensor values of the input function and outputs a vector of coefficients b(u) in R^p. A trunk network reads the query coordinate y and outputs a vector of basis values t(y) in R^p. The predicted output value is the inner product of these two vectors plus a learned scalar bias: G_hat(u)(y) = sum_k b_k(u) t_k(y) + b_0. This mirrors the operator universal approximation theorem, which guarantees that continuous operators on compact function spaces can be represented by a finite sum of products of a function of the input measurements and a function of the output coordinate.

In practice, the unstacked variant is the standard choice. Instead of training p separate branch networks, one branch network has p output units. This shares hidden features across coefficients, reduces parameter count, and acts as a useful regularizer while preserving the same operator form. The trunk network is typically a standard feedforward network whose final linear output is passed through an activation before the dot product, matching the theorem's factorization. The branch output remains as raw coefficients. Once the coefficient vector for an input function has been computed, evaluating the output at new query locations is cheap because the trunk can be evaluated independently. The same architecture also supports a batched grid evaluation path: one coefficient vector per input function dotted against basis vectors for every query in a grid.

The finite-sensor representation is well behaved for solution operators. If the input class has interpolation error kappa(m,V) and the dynamics are Lipschitz, the error from replacing the true input by its sensor interpolation decays with kappa and is controlled by a Gronwall bound. Rougher input functions simply require more sensors, which matches the branch input width.

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


class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers, activation=nn.Tanh()):
        super().__init__()
        if branch_layers[-1] != trunk_layers[-1]:
            raise ValueError("branch and trunk output widths must match")
        self.branch = FNN(branch_layers, activation)
        self.trunk = FNN(trunk_layers, activation)
        self.activation = activation
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, u_sensors, y):
        b = self.branch(u_sensors)
        t = self.activation(self.trunk(y))
        out = torch.einsum("bi,bi->b", b, t).unsqueeze(1)
        return out + self.bias

    def forward_grid(self, u_sensors, y_grid):
        b = self.branch(u_sensors)
        t = self.activation(self.trunk(y_grid))
        return torch.einsum("bi,ni->bn", b, t) + self.bias
```
