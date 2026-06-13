**Problem.** The scaffold default is a *fixed* PPR monomial filter — a frozen low-pass that cannot
represent the high-/band-pass response heterophilic graphs need. Make the filter learnable, but choose a
basis whose coefficients control the response *value* interpretably, so the response can be constrained
to be **non-negative** — the family most valid graph filters live in — by a simple parameter constraint.

**Key idea (Bernstein-basis filter).** Parameterize the filter in the degree-`K` Bernstein basis on the
spectrum `[0,2]`: `h(λ) = sum_k θ_k · (1/2^K) C(K,k) (2-λ)^{K-k} λ^k`. The basis is non-negative and a
partition of unity, so with **all `θ_k ≥ 0`** (enforced by `ReLU`) the filter is a non-negative
combination of non-negative bumps — `h(λ) ≥ 0` certified *everywhere* on `[0,2]`, not just at sample
points — and `θ_k` controls a bump near frequency `2k/K`, making the learned response readable. Applied
as `h(L) = sum_k ReLU(θ_k) (1/2^K) C(K,k) (2I-L)^{K-k} L^k` with `L` the symmetric normalized Laplacian.

**Why.** Non-negativity becomes a box constraint (the monomial basis cannot express it); the
partition-of-unity makes all-ones the neutral all-pass start; the controllable bumps let the filter
become low-, high-, or band-pass from labels. The cost is the catch: assembling `(2I-L)^{K-k} L^k`
re-walks the graph per term, so the forward pass is `O(K^2 m d)` — **quadratic in `K`** — and Bernstein
approximation converges only like `ω(K^{-1/2})`, slower per degree than near-minimax bases, so at a
fixed `K=10` sharp responses can be under-resolved.

**Hyperparameters.** `K=10`, `temp = θ` initialized **all-ones** (→ constant all-pass), `ReLU(θ)` before
the basis sum, `hidden=64`, `dropout=0.5`, `dprate=0.0` (scaffold default; the model passes `dprate`
through). No teleport `alpha` is used. Training overrides left at the scaffold defaults.

```python
class CustomProp(MessagePassing):
    """Bernstein polynomial propagation layer.

    Filter: h(L) = sum_{k=0}^{K} theta_k * C(K,k)/2^K * L^k * (2I-L)^{K-k}
    where theta_k = ReLU(learnable), C(K,k) is binomial coefficient,
    and L is the symmetric normalized Laplacian.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        # L = I - D^{-1/2}AD^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim)
        )
        # 2I - L
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0,
            num_nodes=x.size(self.node_dim)
        )

        # Compute (2I-L)^k * x for k = 0, ..., K
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # Bernstein basis evaluation
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            # Apply L^{i+1}
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(nn.Module):
    """BernNet: Bernstein polynomial graph filter (He et al., 2021).

    MLP encoder + Bernstein polynomial propagation.
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
