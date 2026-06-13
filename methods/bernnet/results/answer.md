# BernNet, distilled

BernNet is a spectral graph neural network that parameterizes its propagation as a
**non-negative Bernstein-basis polynomial of the normalized Laplacian**. By learning the
Bernstein coefficients through a non-negative clamp, it can approximate arbitrary continuous
filter responses over the graph spectrum `[0, 2]` while making the lower-bound validity constraint
direct and the coefficients interpretable as uniform spectral samples.

## Problem it solves

Design one polynomial graph filter that is (1) able to approximate any response shape — low-pass,
high-pass, band-pass, band-rejection, comb — as the order `K` grows; (2) respects the valid-filter
range `0 <= g(lambda) <= 1` after the usual scale/range control, with non-negativity enforced
directly; (3) is interpretable (the parameters tell you the learned shape); and (4) is end-to-end
learnable inside the standard MLP-then-propagate pipeline. Fixed filters (GCN, APPNP) are stuck on
one low-pass shape; learned ones (ChebNet, GPR-GNN) gain expressiveness only by letting coefficients
roam unconstrained, which sacrifices validity and interpretability.

## Key idea

The validity target, from the graph-optimization view of propagation, is a polynomial
`g(lambda) = sum_k w_k lambda^k` with `0 <= g(lambda) <= 1` for `lambda in [0, 2]` (a valid
smoothing energy needs `gamma(L)` positive semidefinite, which forces the induced response into
`(0, 1]`). The upper bound is a rescale; the lower bound `g >= 0` is the hard part. Forcing
*monomial* coefficients non-negative in the PageRank-style basis fails — by GPR-GNN's theorem that
yields a low-pass filter only. The fix is to change basis to one where non-negative coefficients
track non-negative interval polynomials after sufficient degree elevation: the **Bernstein basis**.

On `[0, 1]`, `b_k^K(t) = C(K, k)(1 - t)^{K-k} t^k`. Three properties make it the right basis:

- **Non-negative** on `[0, 1]`, and a **partition of unity** (`sum_k b_k^K(t) = 1`, by the
  binomial theorem). So if every coefficient `theta_k in [0, 1]`, the combination
  `sum_k theta_k b_k^K` is a convex combination, automatically in `[0, 1]` — validity for free.
- **Complete** for the constraint after sufficient degree elevation: a polynomial non-negative on
  the interval can be represented with non-negative Bernstein coefficients (Bernstein 1915; degree
  bounds by Powers & Reznick 2000), so constraining coefficients non-negative need not lose valid
  polynomial shapes. And `B_K(f) = sum_k f(k/K) b_k^K` converges uniformly to any continuous `f`
  (constructive Weierstrass), so continuous responses are reachable as `K` grows.
- **Interpretable bumps:** `b_k^K` peaks at `t = k/K`. Mapping the spectrum `[0, 2]` to `[0, 1]`
  via `t = lambda/2` lands the `k`-th bump at frequency `lambda = 2k/K`, so `theta_k` controls the
  response near `2k/K` — the coefficient vector is the filter, sampled at `K+1` uniform points.

Lifting `lambda -> L` gives the BernNet propagation
```
z = sum_{k=0}^{K} theta_k * (1 / 2^K) * C(K, k) * (2I - L)^{K-k} * L^k * x,   theta_k = ReLU(temp).
```

## Coefficient settings and operators

| Filter           | `h(lambda)`       | `theta_k`                        | BernNet operator                                  |
|------------------|-------------------|----------------------------------|---------------------------------------------------|
| All-pass         | `1`               | `theta_k = 1`                    | `I`                                               |
| Linear low-pass  | `1 - lambda/2`    | `theta_k = 1 - k/K`              | `I - L/2 = (I + P)/2`                              |
| Linear high-pass | `lambda/2`        | `theta_k = k/K`                  | `L/2`                                             |
| Impulse low-pass | `delta_0`         | `theta_0 = 1`, else `0`          | `(2I - L)^K / 2^K`                                 |
| Impulse high-pass| `delta_2`         | `theta_K = 1`, else `0`          | `L^K / 2^K`                                        |
| Impulse band-pass| `delta_1`         | `theta_{K/2} = 1`, else `0`      | `C(K, K/2)(2I - L)^{K/2} L^{K/2} / 2^K`            |

All-pass uses the partition of unity; linear low/high-pass use the Bernstein operator's exact
reproduction of affine functions. The impulse rows are exact operators for those coefficient
settings and act as localized polynomial approximants to ideal impulses; the middle row assumes
even `K`.

## Design choices and why

- **Bernstein over monomial:** non-negative monomial coefficients in the PageRank basis =
  low-pass only; non-negative Bernstein coefficients cover non-negative interval polynomials after
  sufficient degree elevation.
- **Bernstein over Chebyshev:** Chebyshev is `O(K)` and stable but its coefficients are
  unconstrained (filter can be ill-posed) and abstract (no per-frequency meaning); Bernstein trades
  `O(K)` for `O(K^2)` to buy a non-negative-response constraint and interpretable coefficients.
- **`ReLU(temp)` for non-negativity:** cheap, differentiable, and can drive a coefficient to
  exactly 0 (a hard rejection band) — which softplus could not. It enforces the lower-bound side of
  the `[0,1]` validity condition; the upper bound is a scale/range condition.
- **Init `temp = 1` (all-pass / identity):** an unbiased start that imposes no frequency
  preference, letting the data move the filter toward low-, high-, or band-pass.
- **`K = 10`:** high enough to shape comb / band-rejection on the target graphs while keeping the
  `O(K^2)` cost acceptable.
- **Separate learning rate / dropout (`dprate`) for the propagation:** the filter is a handful of
  coefficients and wants different regularization than the dense MLP weights (as in APPNP/GPR-GNN).

## Complexity

Evaluation caches `tmp[i] = (2I - L)^i x` for `i = 0..K` (`K` propagations), then for the
`k = i+1` term applies `L` to `tmp[K-i-1]` a total of `i+1` times. Total propagations
`K + sum_{i=0}^{K-1}(i+1) = K + K(K+1)/2 = O(K^2)` — quadratic in `K`, versus `O(K)` for ChebNet /
GPR-GNN. A linear-time corner-cutting evaluation exists for Bernstein polynomials but does not
transfer directly because the signal `x` must be multiplied through term by term.

## Working code

Propagation layer (non-negative Bernstein polynomial of `L`):

```python
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, add_self_loops
from scipy.special import comb


class Bern_prop(MessagePassing):
    """z = sum_k ReLU(temp)_k * C(K,k)/2^K * (2I - L)^{K-k} L^k x."""

    def __init__(self, K, bias=True, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)                      # all-pass start

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)                        # theta_k >= 0

        # L = I - D^{-1/2} A D^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim))
        # 2I - L
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0, num_nodes=x.size(self.node_dim))

        # tmp[i] = (2I - L)^i x ,  i = 0..K
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # k = 0 term
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        # k = i+1 terms:  L^{i+1} (2I - L)^{K-i-1} x
        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
```

Full model (MLP transform, then Bernstein propagation):

```python
import torch
import torch.nn.functional as F
from torch.nn import Linear


class BernNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = Bern_prop(K)
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
```
