# DeepONet

DeepONet learns an operator \(G:u\mapsto G(u)\) by representing the input function with fixed sensor values and evaluating the output at query points. For sensor values \([u(x_1),\ldots,u(x_m)]\) and query coordinate \(y\), it uses two networks:

\[
b(u)=\operatorname{branch}(u(x_1),\ldots,u(x_m))\in\mathbb{R}^p,\qquad
t(y)=\sigma(\operatorname{trunk}(y))\in\mathbb{R}^p.
\]

The scalar prediction is the inner product plus a learned bias:

\[
\widehat{G}(u)(y)=\sum_{k=1}^{p} b_k(u)t_k(y)+b_0.
\]

The stacked form uses \(p\) separate branch networks, one per coefficient. The unstacked form uses one branch network with \(p\) outputs; it is the practical default because it shares features, uses fewer parameters, and preserves the same inner-product operator form. The generalized approximation statement is:

\[
|G(u)(y)-\langle g(u(x_1),\ldots,u(x_m)),f(y)\rangle|<\varepsilon
\]

for continuous \(G:V\to C(K_2)\) on compact sets, with suitable sensors and continuous vector functions \(g:\mathbb{R}^m\to\mathbb{R}^p\), \(f:\mathbb{R}^d\to\mathbb{R}^p\). Neural networks satisfying ordinary universal approximation can instantiate \(g\) and \(f\).

For an ODE solution operator with input interpolation error \(\kappa(m,V)\) and Lipschitz constant \(c\), the finite-sensor error obeys

\[
\|G(u)(d)-G(u_m)(d)\|_2\le c(b-a)\kappa(m,V)e^{c(b-a)}.
\]

For squared-exponential Gaussian random fields, \(\kappa(m,V)=O(1/(m^2l^2))\), so rougher input functions need more sensors.

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
