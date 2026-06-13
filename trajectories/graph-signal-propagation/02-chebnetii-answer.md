**Problem.** BernNet was soft on the homophilic citation graphs (Citeseer 0.7795, Cora 0.8554) — its
Bernstein basis converges only like `ω(K^{-1/2})` and costs `O(K^2 m d)`, so at the fixed `K=10` it
under-resolves a smooth low-pass and cannot raise `K` to compensate. I want the same near-best
approximation with *linear* cost and faster per-degree convergence, i.e. a Chebyshev construction —
while keeping BernNet's lesson that response constraints should be parameter constraints.

**Key idea (Chebyshev *interpolation*, not free-coefficient Chebyshev).** A free-coefficient Chebyshev
filter actually loses on these graphs, because nothing forces the high-`k` coefficients of a smooth
(analytic) response to decay, so gradient descent overfits a high-frequency response. Fix it by
parameterizing the filter by its **values** `γ_j = h(x_j)` at the Chebyshev nodes
`x_j = cos((j+1/2)π/(K+1))` (the roots of `T_{K+1}`) and *deriving* the coefficients:
`c_k = (2/(K+1)) sum_j γ_j T_k(x_j)` (a DCT), applying `h(L_hat) = (c_0/2)T_0 + sum_{k≥1} c_k T_k(L_hat)`.

**Why these nodes.** The interpolation error is `R_K = h^{(K+1)}(ζ)/(K+1)! · π_{K+1}`, controlled by the
nodal polynomial `π_{K+1} = prod_k(λ̂ - x_k)`. Among monic degree-`K+1` polynomials, `2^{-K}T_{K+1}` has
the smallest uniform norm `2^{-K}`; its roots are the Chebyshev nodes, so they minimize the Runge
blow-up and give a Lebesgue constant `~log K` (vs `~2^K` equispaced) — near-minimax. Because the
parameters are sampled response values, the coefficients are tied to a stable interpolant: for a smooth
target the decay comes from the response, not a `1/k` hack. `ReLU(γ)` enforces non-negative
sampled-values (a node-level constraint, weaker than BernNet's global certificate). Rescale with the
a-priori `λ_max = 2`, so `L_hat = L - I` needs no eigen-computation. Cost: `O(K^2 + K m d)` — **linear in
`K`**, strictly cheaper than BernNet; convergence `~ω(K^{-1})log K`, faster per degree.

**Hyperparameters.** `K=10`, `temp = γ` initialized **all-ones** (→ constant all-pass), `ReLU(γ)` before
the DCT, `hidden=64`, `dropout=0.5`, `dprate=0.0` (scaffold default; model passes `dprate` through). The
constant term is applied as `c_0/2`. Training overrides left at scaffold defaults.

```python
class CustomProp(MessagePassing):
    """ChebNetII propagation: Chebyshev interpolation filter.

    Learns filter values at Chebyshev interpolation nodes, then converts
    to Chebyshev polynomial coefficients. Uses ReLU to ensure non-negative
    interpolation values.

    Filter: h(L_tilde) = sum_{k=0}^{K} c_k * T_k(L_tilde)
    where L_tilde = L - I (shifted Laplacian), T_k is the k-th Chebyshev
    polynomial, and c_k are computed from interpolation values via DCT-like transform.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        # Convert interpolation values to Chebyshev coefficients
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # L_tilde = L - I (shifted to [-1, 1] range)
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        # Chebyshev recurrence: T_0(x)=x, T_1(x)=x, T_{k+1}=2xT_k - T_{k-1}
        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """ChebNetII: Chebyshev interpolation graph filter (He et al., 2022).

    MLP encoder + ChebNetII propagation with Chebyshev interpolation.
    """

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
```
