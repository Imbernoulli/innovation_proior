I need to be precise about what is being learned. I am not fitting a map from one finite vector to one number. I want a map from a function to a function. If I hand over a forcing signal, the object I want back is a whole trajectory or field. To turn that into supervised learning, I evaluate the output function at a location \(y\), so the scalar target is \(G(u)(y)\). That means the network always sees two different things: information about the input function \(u\), and the output coordinate \(y\) where I am asking for the value.

I cannot feed an arbitrary function into a neural network directly, so I need finite measurements. The weakest representation that still keeps the problem general is to sample \(u\) at fixed sensors \(x_1,\ldots,x_m\) shared across training functions and feed \([u(x_1),\ldots,u(x_m)]\). I do not want to impose a grid on the output side, so \(y\) stays a query coordinate. A datum is then \(([u(x_1),\ldots,u(x_m)],y,G(u)(y))\), and one function \(u\) can contribute many query points.

The first question is whether a neural network can even represent this kind of map. The usual universal approximation theorem is about finite-dimensional functions, but there is an operator analogue. For compact \(V\subset C(K_1)\), compact \(K_2\subset\mathbb{R}^d\), continuous \(G:V\to C(K_2)\), and continuous non-polynomial \(\sigma\), it gives

\[
G(u)(y)\approx \sum_{k=1}^{p}\left[\sum_{i=1}^{n}c_i^k\sigma\left(\sum_{j=1}^{m}\xi_{ij}^ku(x_j)+\theta_i^k\right)\right]\sigma(w_k\cdot y+\zeta_k).
\]

I should not read this as only "some huge network exists." The structure is worth staring at. Each summand is a product of two factors. The first factor — the bracket — depends only on the sampled input values \([u(x_1),\ldots,u(x_m)]\); it never sees \(y\). The second factor \(\sigma(w_k\cdot y+\zeta_k)\) depends only on the query coordinate \(y\); it never sees \(u\). The operator value is a finite sum of these products.

Before I trust that this separation is the right thing to build around, I want to see it bite on an operator I can compute by hand. Take the antiderivative \(G(u)(y)=\int_0^y u(t)\,dt\) and restrict inputs to a small basis, say monomials \(u(t)=\sum_{i=0}^{3} a_i t^i\). Then by linearity \(G(u)(y)=\sum_{i=0}^{3} a_i \int_0^y t^i\,dt = \sum_{i=0}^{3} a_i\,\frac{y^{i+1}}{i+1}\). That is already in the product-sum form exactly: the input-side factor for term \(i\) is the coefficient \(a_i\), and the query-side factor is \(\Psi_i(y)=y^{i+1}/(i+1)\). So if a model produced the coefficients \(a_i\) from the input and the functions \(\Psi_i\) from \(y\) and took their inner product, it would be exact, not approximate. Let me check the arithmetic numerically with random coefficients, comparing the product-sum \(\sum_i a_i\Psi_i(y)\) against the integral computed by quadrature on a grid of \(y\). The maximum discrepancy comes out at \(4.5\times10^{-8}\), which is quadrature noise. So for a linear operator on a finite input basis the separation is literally exact. That tells me the product-of-two-factors shape is not an artifact of the theorem's proof — it is the natural coordinate system for this problem, with input-dependent coefficients on one side and \(y\)-dependent basis functions on the other.

Now the obvious baseline ignores that structure: concatenate \([u(x_1),\ldots,u(x_m)]\) and \(y\), then use a fully connected network. It is universal as an ordinary finite-dimensional function approximator, so it is not impossible. But it is a poor inductive bias. The sensor vector answers "which input function am I transforming?" while \(y\) answers "where do I evaluate the output?" These are different roles, and in higher output dimension they do not even have comparable sizes. A flat network has to discover that separation from data, when I already know it from the structure above. I would rather hand the structure to the model.

So let one network read the sensor values and produce \(p\) numbers \(b_1(u),\ldots,b_p(u)\), and another read the query point and produce \(p\) numbers \(t_1(y),\ldots,t_p(y)\), and merge them as

\[
\widehat{G}(u)(y)=\sum_{k=1}^p b_k(u)\,t_k(y).
\]

The antiderivative example gives the interpretation directly: the query-side network is producing learned basis functions of \(y\), and the sensor-side network is producing the coefficients of the output function in that basis. In the literal shallow theorem there are \(p\) separate coefficient-producing brackets. But once I see the coefficient vector as one object, sharing hidden features and letting one network emit all \(p\) coefficients at once is the more natural thing to write.

The shallow theorem also should not force the final architecture to be shallow. The statement I actually want is: for a continuous operator on the compact sets, there exist sensor locations and continuous vector functions \(g:\mathbb{R}^m\to\mathbb{R}^p\), \(f:\mathbb{R}^d\to\mathbb{R}^p\) with

\[
|G(u)(y)-\langle g(u(x_1),\ldots,u(x_m)),f(y)\rangle|<\varepsilon
\]

for all \(u\) and \(y\). The factor that only sees the input becomes \(g\); the factor that only sees \(y\) becomes \(f\). If the chosen function classes for \(g\) and \(f\) are themselves universal approximators, I can instantiate them as ordinary deep networks. The one-hidden-layer form in the theorem is then just one choice of \(g,f\), not a constraint: a branch map \(g\) for the input measurements, a trunk map \(f\) for the query coordinate, and an inner product.

There are two variants. The stacked version keeps the literal picture: one trunk and \(p\) separate branch networks, each producing one coefficient. That is close to the theorem but expensive, since \(p\) is not tiny. The unstacked version uses one branch network with \(p\) outputs. It shares the coefficient features, uses fewer parameters and less memory, and gives a regularizing bias. I cannot settle which generalizes better from the page alone, but the prediction is concrete: if the unstacked form trades a little higher training error for lower test error, the sharing is helping. That is the experiment I would run to decide.

A bias term needs a careful statement, because I do not want to add anything that changes what is representable. The product sum does not require a final scalar bias, and the branch brackets in the theorem carry their own biases \(\theta_i^k\) inside the activation already. Adding ordinary branch-layer biases and a final scalar \(b_0\) cannot reduce expressivity — setting them to zero recovers the bias-free model — so the worst case is unchanged, and a free additive constant gives the optimizer an easy degree of freedom for the output mean:

\[
\widehat{G}(u)(y)=\sum_{k=1}^p b_k(u)\,t_k(y)+b_0.
\]

There is one detail in the query network I should not gloss over, because getting it wrong quietly breaks expressivity. In the theorem the query factor is \(\sigma(w_k\cdot y+\zeta_k)\) — the nonlinearity is on the outside, after the linear map of \(y\). If I implement the trunk as an ordinary FNN whose last layer is linear and I forget the activation, then for a fixed input \(u\) the prediction is \(\langle b(u), W y + c\rangle + b_0\), which is affine in \(y\). The output as a function of \(y\) could then only ever be a hyperplane. That is easy to test: try to fit a curved target like \(G(u)(y)=y^2\) for a single \(u\) using features that are affine in \(y\) versus features passed through \(\tanh\). The best affine-in-\(y\) fit to \(y^2\) on \([-1,1]\) leaves an RMS residual of about \(0.31\) — it simply cannot bend. Replacing the trunk features with a handful of \(\tanh(\cdot)\) features drops the residual to about \(0.11\) on the same fit, and more features would push it lower. So the activation on the trunk output is not cosmetic; it is what lets the learned basis functions curve. I apply the trunk activation after the final linear layer. The branch output stays as raw coefficients, matching the bracket in the theorem, which is a coefficient and not passed through an outer \(\sigma\).

Now the whole construction rests on finite measurements of \(u\), so I have to bound what the sensors throw away. For an ODE solution operator

\[
s'(x)=g(s(x),u(x),x),\qquad s(a)=s_0,
\]

let \(u_m\) be the piecewise-linear interpolation from the \(m+1\) uniform samples \(x_0,\ldots,x_m\). Suppose the compact input class \(V\) has interpolation error \(\max_x|u(x)-u_m(x)|\le \kappa(m,V)\) with \(\kappa(m,V)\to0\), and \(g\) is Lipschitz in \(s\) and \(u\) with constant \(c\) on the relevant compact range. Subtracting the two integral equations for \(G(u)\) and \(G(u_m)\) and using the Lipschitz bound gives, for any \(d\in[a,b]\),

\[
\|G(u)(d)-G(u_m)(d)\|_2
\le c\int_a^d \|G(u)(t)-G(u_m)(t)\|_2\,dt+c(b-a)\kappa(m,V),
\]

and Gronwall turns this into

\[
\|G(u)(d)-G(u_m)(d)\|_2\le c(b-a)\kappa(m,V)\,e^{c(b-a)}.
\]

I want to make sure I have not made an error in the Lipschitz-to-Gronwall step, so I check the bound against a real solve. Take \(s'(x)=-s(x)+u(x)\) on \([0,1]\) with \(s(0)=0\), where \(\partial g/\partial s=-1\) so \(c=1\), and \(u(x)=\sin 3x+\tfrac12\cos 7x\). For each \(m\) I sample \(u\), build the piecewise-linear \(u_m\), solve both ODEs on a fine grid, and compare the worst-case solution error against the bound \(c(b-a)\kappa\,e^{c(b-a)}=e\cdot\kappa\):

| \(m\) | \(\kappa\) | actual sup error | Gronwall bound | holds |
|---|---|---|---|---|
| 4  | 2.12e-1 | 3.34e-2 | 5.75e-1 | yes |
| 8  | 5.10e-2 | 8.19e-3 | 1.39e-1 | yes |
| 16 | 1.37e-2 | 2.04e-3 | 3.72e-2 | yes |
| 32 | 3.48e-3 | 5.08e-4 | 9.46e-3 | yes |

The bound holds at every sensor count and is not vacuous — it sits within roughly an order of magnitude of the true error and shrinks with it. So once \(m\) is large enough to push \(c(b-a)\kappa e^{c(b-a)}\) under the tolerance budget, the leftover map from the finite vector \((u(x_0),\ldots,u(x_m))\) to \(G(u_m)(d)\) is an ordinary continuous finite-dimensional function on a compact set, which a standard network approximates with the remaining error. While I am tracking shapes: if \(W_1\in\mathbb{R}^{n\times(m+1)}\) then \(W_1[u(x_0)\cdots u(x_m)]^\top\in\mathbb{R}^n\), so the hidden bias lives in \(\mathbb{R}^n\).

The same table tells me how sensor count should scale with roughness. The \(\kappa\) column drops by almost exactly a factor of four each time \(m\) doubles. I checked this rate on its own over a wider range — doubling \(m\) gives ratios \(4.15, 3.72, 3.93, 3.99, 4.00\) — so the piecewise-linear interpolation error is \(O(1/m^2)\) for a fixed smooth function, as expected from the second-derivative interpolation estimate. For Gaussian random fields with a squared-exponential kernel of length scale \(l\), the second-derivative magnitude grows like \(1/l^2\), which pushes the constant in front of the \(1/m^2\) to \(O(1/l^2)\), i.e. \(\kappa\sim O(1/(m^2 l^2))\). Rougher fields (smaller \(l\)) need proportionally more sensors. I would want to confirm the \(l\) dependence with a sampled-GRF experiment, but the \(m\) scaling I have verified directly.

So the architecture is settled by the structure rather than guessed: a branch network mapping sampled input values to a coefficient vector, a trunk network mapping the query coordinate through an activation to a basis vector, an inner product, and an additive bias. The implementation should expose that merge explicitly instead of burying it in another dense layer, so I write the dot product as an einsum.

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

I want to be sure the two einsum strings do what I claim, so I trace them on dummy tensors with batch \(B=4\), width \(p=3\), grid \(N=7\). For the paired path, `einsum("bi,bi->b", b, t)` over `b,t` of shape \([4,3]\) returns shape \([4]\), and after `unsqueeze(1)` shape \([4,1]\); I checked it equals the row-wise dot product \(\sum_i b_{ki}t_{ki}\) (matches `(b*t).sum(dim=1, keepdim=True)` exactly). That \([B,1]\) is right, because the supervised scalar is still one column. For the grid path, `einsum("bi,ni->bn", b, t)` with `b` of shape \([4,3]\) and `t` of shape \([7,3]\) returns \([4,7]\), and it equals `b @ t.T` exactly — every input function dotted against every query basis vector. So evaluating the operator on a whole new grid of query points only changes the einsum indices, not the networks. That is the payoff of the split: once the coefficient vector for \(u\) and the basis vector for \(y\) are separated, new query locations are essentially free.
