I need to be precise about what is being learned. I am not fitting a map from one finite vector to one number. I want a map from a function to a function. If I hand over a forcing signal, the object I want back is a whole trajectory or field. To turn that into supervised learning, I evaluate the output function at a location \(y\), so the scalar target is \(G(u)(y)\). That means the network always sees two different things: information about the input function \(u\), and the output coordinate \(y\) where I am asking for the value.

I cannot feed an arbitrary function into a neural network directly, so I need finite measurements. The weakest representation that still keeps the problem general is to sample \(u\) at fixed sensors \(x_1,\ldots,x_m\) shared across training functions and feed \([u(x_1),\ldots,u(x_m)]\). I do not want to impose a grid on the output side, so \(y\) stays a query coordinate. A datum is then \(([u(x_1),\ldots,u(x_m)],y,G(u)(y))\), and one function \(u\) can contribute many query points.

The first question is whether a neural network can even represent this kind of map. The usual universal approximation theorem is about finite-dimensional functions, but the operator theorem gives a stronger object. For compact \(V\subset C(K_1)\), compact \(K_2\subset\mathbb{R}^d\), continuous \(G:V\to C(K_2)\), and continuous non-polynomial \(\sigma\), it gives

\[
G(u)(y)\approx \sum_{k=1}^{p}\left[\sum_{i=1}^{n}c_i^k\sigma\left(\sum_{j=1}^{m}\xi_{ij}^ku(x_j)+\theta_i^k\right)\right]\sigma(w_k\cdot y+\zeta_k).
\]

I should not read this as only "some huge network exists." The structure matters. Each term is a product of something that only sees the sampled input function and something that only sees the query coordinate. The first bracket is a scalar function of \([u(x_1),\ldots,u(x_m)]\). The second factor is a scalar function of \(y\). The operator value is a finite sum of these products.

The obvious baseline ignores that structure: concatenate \([u(x_1),\ldots,u(x_m)]\) and \(y\), then use a fully connected network. It is universal as an ordinary finite-dimensional function approximator, so it is not impossible. But it is a bad inductive bias. The sensor vector answers "which input function am I transforming?" while \(y\) answers "where do I evaluate the output?" In high dimension, the query coordinate and the sensor vector do not even have comparable dimensions. Treating them as one flat object asks a single network to discover that separation from data.

So I want the theorem's separation to be built into the model. Let one network read the sensor values and produce \(p\) numbers \(b_1(u),\ldots,b_p(u)\). Let another network read the query point and produce \(p\) numbers \(t_1(y),\ldots,t_p(y)\). Then merge by the theorem's finite product sum:

\[
\widehat{G}(u)(y)=\sum_{k=1}^p b_k(u)t_k(y).
\]

This has a useful interpretation. The query-side network is producing learned basis functions of \(y\), and the sensor-side network is producing the coefficients of the output function in that basis. In the literal shallow theorem, there are \(p\) separate coefficient-producing pieces. But once I see the coefficient vector as one object, it is natural to share hidden features and let one network emit all \(p\) coefficients at once.

The shallow theorem also should not force the final architecture to be shallow. The more general statement I need is: for a continuous operator on the compact sets, there exist sensor locations and continuous vector functions \(g:\mathbb{R}^m\to\mathbb{R}^p\), \(f:\mathbb{R}^d\to\mathbb{R}^p\) such that

\[
|G(u)(y)-\langle g(u(x_1),\ldots,u(x_m)),f(y)\rangle|<\varepsilon
\]

for all \(u\) and \(y\). If the chosen function classes for \(g\) and \(f\) are ordinary universal approximators, I can instantiate them as deep neural networks. That turns the theorem into an architecture rather than a one-hidden-layer fossil: a branch map \(g\) for the input function measurements, a trunk map \(f\) for the query coordinate, and an inner product.

There are two variants to consider. The stacked version keeps the literal picture: one trunk and \(p\) separate branch networks, each producing one coefficient. That is close to the theorem but expensive, because \(p\) is not tiny. The unstacked version uses one branch network with \(p\) outputs. This shares the coefficient features, uses fewer parameters and less memory, and gives a regularizing bias. If the test error improves even when the training error is a little larger, that is exactly the tradeoff I want from sharing.

Bias needs a careful statement. The theorem's product sum does not require a final scalar bias, and the branch pieces in the theorem do not need a final bias. But adding ordinary branch-layer biases and a final scalar \(b_0\) does not reduce expressivity, and it makes the optimizer's target easier:

\[
\widehat{G}(u)(y)=\sum_{k=1}^p b_k(u)t_k(y)+b_0.
\]

The query network also needs one special implementation detail. In the theorem, the query factor is \(\sigma(w_k\cdot y+\zeta_k)\), so the trunk output should be activated before the dot product. If the trunk is an ordinary FNN whose final layer is linear, I should apply the trunk activation after that final layer. The branch output stays as raw coefficients.

Now I need to check the sensor count story, because the whole construction depends on finite measurements of \(u\). For an ODE solution operator

\[
s'(x)=g(s(x),u(x),x),\qquad s(a)=s_0,
\]

let \(u_m\) be the piecewise-linear interpolation from the \(m+1\) uniform samples \(x_0,\ldots,x_m\). Suppose the compact input class \(V\) has interpolation error \(\max_x|u(x)-u_m(x)|\le \kappa(m,V)\) with \(\kappa(m,V)\to0\). Also suppose \(g\) is Lipschitz in \(s\) and \(u\) with constant \(c\) on the relevant compact range. Then for any \(d\in[a,b]\),

\[
\|G(u)(d)-G(u_m)(d)\|_2
\le c\int_a^d \|G(u)(t)-G(u_m)(t)\|_2\,dt+c(b-a)\kappa(m,V).
\]

Gronwall gives

\[
\|G(u)(d)-G(u_m)(d)\|_2\le c(b-a)\kappa(m,V)e^{c(b-a)}.
\]

So if I choose \(m\) so that this term is below the tolerance budget, the remaining map from the finite vector \((u(x_0),\ldots,u(x_m))\) to \(G(u_m)(d)\) is an ordinary continuous finite-dimensional function on a compact set, and a standard network can approximate it with the remaining error. I also need to keep the shapes straight: if \(W_1\in\mathbb{R}^{n\times(m+1)}\), then \(W_1[u(x_0)\cdots u(x_m)]^T\in\mathbb{R}^n\), so the hidden bias is in \(\mathbb{R}^n\). For Gaussian random fields with a squared-exponential kernel, the interpolation scale is \(O(1/(m^2l^2))\), so rougher functions need more sensors.

The architecture I end up with is therefore simple: branch coefficients from sampled input functions, activated trunk basis values from query locations, dot product plus scalar bias. The implementation should mirror that instead of hiding the merge inside another dense layer.

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

The paired-data path returns shape \([B,1]\), because the scalar is still a one-column supervised output. The grid path evaluates every input function against every query location by changing only the einsum indices. That is exactly the point of the split: once the coefficient vector for \(u\) and the basis vector for \(y\) are separated, new query locations are cheap.
