**Problem.** Both constrained rungs lost the heterophilic graphs: BernNet's global non-negativity and
ChebNetII's near-minimax interpolation each forbid the sharp, sign-changing high-pass response Texas and
Cornell need (ChebNetII fell to texas 0.8770), and ChebNetII's near-zero seed variance betrays a filter
too stiff to move. Drop the hard constraint: parameterize the filter with *unconstrained, sign-free*
coefficients and control overfitting *softly* instead.

**Key idea (learnable monomial / Generalized-PageRank filter).** Learn the hop-weights of a monomial
polynomial in the GCN-normalized adjacency, `h(P) = sum_k γ_k P^k`, `P = D^{-1/2} A D^{-1/2}`. Each `γ_k`
is the (signed, free) weight on `k`-hop information — directly interpretable, and free to subtract a hop
(build contrast) for heterophily, which the non-negative bases could not. Linear cost `O(K m d)`: `K`
sparse mat-vecs of `P`, no DCT, no Laplacian shift.

**Why these choices (the departures from the textbook monomial filter).**
- **Uniform init `γ_k = 1/(K+1)`**, *not* PPR `α(1-α)^k`. PPR bakes in a low-pass homophily prior the
  optimizer must climb out of to reach a heterophilic response — a headwind on 183-node WebKB graphs.
  The uniform hop-average is dataset-agnostic and flat, reachable to either low- or high-pass.
- **Same fast LR for filter and MLP (`0.05`)**, not the scaffold's throttled `PROP_LR=0.01`. The
  unconstrained hop-weights must move fast enough to find the heterophilic response; stiffness was
  ChebNetII's failure.
- **Weight decay on the encoder (`5e-4`), zero on the filter coefficients.** Spend the regularization
  budget where overfitting bites (the high-capacity MLP on a tiny graph); decaying the filter toward
  zero would re-impose an over-smooth prior.
- `dprate=0.0` (propagation dropout off), consistent with the task default for spectral filters on
  heterophilic data. No `alpha` is used by the filter (it sets only the unused PPR pattern).

The monomial basis is ill-conditioned for large `K`, but at `K=10` on these graphs the empirical record
is blunt: a learned monomial filter beats a free Chebyshev one — the conditioning cost is worth the
unconstrained, sign-free, hop-interpretable coefficients.

**Hyperparameters.** `K=10`, `temp = γ` init **all `1/(K+1)`**, `hidden=64`, `dropout=0.5`, `dprate=0.0`;
training overrides `custom_lr=0.05`, `custom_wd=5e-4`, `custom_prop_lr=0.05`, `custom_prop_wd=0.0`.

```python
class CustomProp(MessagePassing):
    """GPR propagation: learnable polynomial in the monomial basis.

    Filter: h(A) = sum_{k=0}^{K} gamma_k * A^k
    where A is the GCN-normalized adjacency and gamma_k are learnable.

    Initialized with uniform coefficients (1/(K+1)) so the filter starts
    as an equal-weight average of all hops. This is dataset-agnostic and
    lets the optimizer freely learn both low-pass (homophilic) and
    high-pass (heterophilic) filters.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Uniform initialization for dataset-agnostic starting point.
        nn.init.constant_(self.temp, 1.0 / (self.K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """GPRGNN: Generalized PageRank GNN (Chien et al., 2021).

    MLP encoder + learnable monomial polynomial filter.
    """

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha)
        self.dropout = dropout
        self.dprate = 0.0  # GPRGNN paper: no propagation dropout
        # Override training hyperparams (read by template's training loop)
        self.custom_lr = 0.05
        self.custom_wd = 0.0005
        self.custom_prop_lr = 0.05  # same lr for filter coefficients
        self.custom_prop_wd = 0.0

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
