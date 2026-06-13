# ChebNetII, distilled

ChebNetII is a learnable spectral graph filter that approximates an arbitrary frequency
response by **Chebyshev interpolation** rather than free-coefficient Chebyshev approximation.
It reparameterizes the filter by its *values* `γ_j = h(x_j)` at the Chebyshev nodes `x_j`
(the zeros of `T_{K+1}`), converts those values to Chebyshev coefficients by a discrete cosine
transform, and propagates with the stable three-term recurrence. Because the parameters are
filter values at near-minimax interpolation nodes, the coefficient vector is tied to a sampled
response instead of being learned freely; for smooth target responses this gives the expected
high-order decay without a hand-set `1/k` penalty. The Runge nodal-polynomial blowup is
suppressed, and side constraints on sampled response values, such as non-negativity at the
interpolation nodes, become box constraints on `γ_j` (a ReLU). It is linear in the polynomial
order `K`.

## Problem it solves

Learn a single graph filter that can take any frequency shape — low-pass for homophilic
graphs, high-pass/band-pass for heterophilic graphs — from labels alone, while (1) staying
linear in `K`, (2) not overfitting through wild high-frequency responses, and (3) admitting
constraints (notably non-negativity) cleanly. Unconstrained Chebyshev coefficients (ChebNet,
ChebBase) overfit and lose even to GCN; the monomial basis (GPR-GNN) is ill-conditioned; the
Bernstein basis (BernNet) is non-negative and interpretable but costs `O(K^2)`.

## Key idea

A filter `h` on `[-1,1]` is built not from free coefficients but by interpolating its values
at the **Chebyshev nodes** `x_j = cos((j + 1/2)π/(K+1))`, `j=0..K`. Two facts make this the
right choice:

- **Near-minimax and Runge-resistant.** The interpolation error is
  `R_K(λ̂) = h^{(K+1)}(ζ)/(K+1)! · π_{K+1}(λ̂)`, controlled by the nodal polynomial
  `π_{K+1}(λ̂) = prod_k(λ̂ - x_k)`. Among monic degree-`K+1` polynomials, the scaled
  Chebyshev polynomial `2^{-K} T_{K+1}` has the smallest uniform norm, `2^{-K}`; its roots are
  the Chebyshev nodes, so choosing them minimizes the error factor that drives the Runge blowup.
  The Lebesgue constant is `~log K` (versus `~2^K` for equispaced nodes), giving a near-best
  approximation `||h - P_K|| ≤ (1+ρ)||h - P_best||` with `ρ ~ log K`.
- **Coefficient control by parameterization.** An analytic filter's Chebyshev coefficients must
  decay like `1/k^q` (large-`k` = high-frequency). Parameterizing sampled filter values and then
  deriving the Chebyshev-interpolation coefficients ties the coefficients to the sampled
  response; when the sampled response is smooth, the expected decay comes from the response
  rather than from a heuristic `1/k` penalty.

**Reparameterize the values, derive the coefficients.** Let `γ_j = h(x_j)` be the trainable
parameters. By the discrete orthogonality of `T_k` at the Chebyshev nodes
(`sum_j T_m(x_j)T_l(x_j)` is `0` for `m≠l`, `(K+1)/2` for `m=l≠0`, `K+1` for `m=l=0`), the
interpolant's coefficients are the discrete cosine transform of the values:

```
c_k = (2/(K+1)) sum_{j=0}^K γ_j T_k(x_j),   P_K = (c_0/2)T_0 + sum_{k=1}^K c_k T_k.
```

Constraints on the sampled filter are now box constraints on `γ_j`: non-negative
interpolation-node values are `γ_j ≥ 0`, enforced by `ReLU(γ)` before forming the coefficients.
This is a sampled-value constraint, not a Bernstein-style global positivity certificate between
interpolation nodes.

## Final model

```
Y = (c_0/2)T_0(L_hat)f_θ(X) + sum_{k=1}^K c_k T_k(L_hat)f_θ(X),
c_k = (2/(K+1)) sum_{j=0}^K γ_j T_k(x_j).
```

with `x_j = cos((j + 1/2)π/(K+1))`, `f_θ` an MLP (transformation decoupled from propagation,
APPNP-style), and the spectrum rescaled by the a-priori bound `λ_max ≤ 2` (use `λ_max = 2`),
so `L_hat = 2L/2 - I = L - I` needs no eigen-computation. The single change from ChebNet's
`sum_k w_k T_k(L_hat)` is the reparameterization `c_k = (2/(K+1)) sum_j γ_j T_k(x_j)`,
with the constant term applied as `c_0/2`.

## Defaults and why

- `temp = γ` initialized to **all ones**: the constant filter `h ≡ 1`, i.e. the neutral
  all-pass/identity response, imposing no low- or high-pass bias so the response is learned
  from labels. (A specific seed shape would init `γ_j` to that filter's interpolation-node
  values, e.g. `x_j^2`.)
- **ReLU on `γ`** before the coefficient transform: enforces non-negative sampled values — the
  constraint that was awkward to impose on abstract coefficients.
- `L_hat = L - I`: maps the guaranteed normalized-Laplacian range `[0,2]` to `[-1,1]` using
  `λ_max = 2`, with no `λ_max` computation.
- `K = 10`, hidden `= 64`, dropout `= 0.5`, propagation dropout `dprate = 0.5`.

## Complexity

`O(K^2 + K m d)`: `O(K^2)` to form the coefficients (the `T_k(x_j)` are fixed and
precomputable), `O(K m d)` for the `K` sparse propagations (`m` edges, `d` channels). Linear
in `K` — same order as GPR-GNN and ChebNet, and strictly cheaper than BernNet's `O(K^2 m d)`.
Convergence is also faster than Bernstein: Chebyshev interpolation error `~ C ω(K^{-1}) log K`
versus Bernstein `~ (1 + (2K)^{-2}) ω(K^{-1/2})`.

## Working code

Filling the propagation slot of the decoupled harness; `temp` holds the filter values `γ_j`:

```python
import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from utils import cheby


class ChebnetII_prop(MessagePassing):
    """ChebNetII propagation: Chebyshev-interpolation filter.

    The K+1 parameters `temp` are the filter values gamma_j = h(x_j) at the Chebyshev
    nodes; they are converted to coefficients c_k = (2/(K+1)) sum_j gamma_j T_k(x_j).
    ReLU enforces non-negative sampled values. The applied filter is
    c_0/2 T_0(L_hat) + sum_{k=1}^K c_k T_k(L_hat), with L_hat = L - I.
    """

    def __init__(self, K, Init=False, bias=True, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.Init = Init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)                    # constant (all-pass) filter to start
        if self.Init:
            for j in range(self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                self.temp.data[j] = x_j ** 2         # optional value-shaped initialization

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)                  # gamma_j >= 0 at interpolation nodes
        coe = coe_tmp.clone()

        # c_i = (2/(K+1)) sum_j gamma_j T_i(x_j), with nodes enumerated in reverse order.
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L = I - D^{-1/2} A D^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim))
        # L_hat = L - I  (rescale [0,2] -> [-1,1] via lambda_max = 2)
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0, num_nodes=x.size(self.node_dim))

        # three-term recurrence: T_0=x, T_1=L_hat x, T_k = 2 L_hat T_{k-1} - T_{k-2}
        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1       # constant term halved
        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class ChebNetII(torch.nn.Module):
    """ChebNetII: MLP transform decoupled from Chebyshev-interpolation propagation."""

    def __init__(self, dataset, args):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = ChebnetII_prop(args.K)
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
```
