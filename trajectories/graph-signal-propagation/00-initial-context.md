## Research question

A graph neural network for node classification pushes each node's features through the graph before reading them out, and the operator that does the pushing — a *graph filter* — decides what the network can learn. On a **homophilic** graph (linked nodes usually share labels, as in citation networks) a smoothing, low-pass operator is right: averaging a node with its neighbors sharpens the class signal. On a **heterophilic** graph (linked nodes tend to differ, as in webpage link graphs) smoothing is harmful; the discriminative signal lives in how a node contrasts with its neighborhood, a high-frequency component averaging destroys. The same fixed operator cannot serve both regimes.

The open design question is the **propagation filter**: a learnable polynomial of the graph Laplacian that can take any frequency shape — low-pass, high-pass, band-pass, or a mixture — from labels alone, and do it cheaply (linear, not quadratic, in the polynomial order `K`), without a high-order response becoming unstable, and ideally making simple side constraints (notably non-negativity) easy to impose. Everything else about the model is fixed: the MLP encoder, the dropouts, the optimizer, the data splits, and the early-stopping evaluation.

## Prior art / Background / Baselines

- **Spectral filtering via the Laplacian.** With the symmetric normalized Laplacian `L = I - D^{-1/2} A D^{-1/2} = U Λ U^T`, a filter applies a scalar `h` to each frequency, `y = U h(Λ) U^T x`. The eigendecomposition is `O(n^3)`, so `h` is restricted to a degree-`K` polynomial of `L`, `y ≈ sum_k w_k L^k x` — `K` sparse mat-vecs, no `U`. Gap: the choice of polynomial basis and coefficients is left open, and different choices give very different empirical behavior.
- **ChebNet.** Learns a degree-`K` Chebyshev polynomial of the shifted Laplacian `L_hat = 2L/λ_max - I` through a stable three-term recurrence, absorbing the coefficients into a full weight matrix `W_k` per order. Gap: on standard citation graphs it underperforms its own first-order special case (GCN), accuracy falls as `K` rises, and the per-order weight matrices make parameter count scale with `K`; the response is also learned freely, with no direct control over its shape.
- **GCN.** Collapses ChebNet to first order with a renormalization trick, `y = W · D̃^{-1/2} Ã D̃^{-1/2} x`. Gap: the filter is a fixed low-pass smoother; it cannot represent high-pass or band-pass responses and is not learnable as a filter.
- **APPNP.** Decouples the feature transform (an MLP) from graph propagation: run `f_θ(X)` first, then a fixed PPR polynomial `sum_k α(1-α)^k P̃^k`. Gap: the propagation polynomial is fixed (PPR), not learned from the data, so it cannot adapt its frequency response to the graph.

## Fixed substrate / Code framework

A single training/evaluation pipeline is frozen and must not be touched. Four datasets spanning the homophily range — **cora** (2,708 nodes, 7 classes, homophilic), **citeseer** (3,327, 6, homophilic), **texas** and **cornell** (183, 5, heterophilic WebKB) — each run over 10 random 60/20/20 train/val/test splits with early stopping (patience 200, up to 1000 epochs), features row-normalized.

The model is the decoupled `dropout → Linear → ReLU → dropout → Linear → (dprate) → CustomProp → log_softmax`, trained with Adam under NLL loss; the linear layers and the propagation parameters get their own learning rate / weight decay (a model may override `LR`, `WEIGHT_DECAY`, `PROP_LR`, `PROP_WD` through `custom_*` attributes). The loop also provides the operators a filter may use: `gcn_norm(edge_index)` → `D^{-1/2} A D^{-1/2}`; `get_laplacian(edge_index, normalization='sym')` → `L = I - D^{-1/2} A D^{-1/2}`; `add_self_loops(edge_index, weight, fill_value)` to shift the operator; `self.propagate(edge_index, x=x, norm=norm)` for one sparse mat-vec; `cheby(i, x)` for `T_i(x)` at a scalar; and `comb(n, k)` (binomial). Defaults: `K=10`, `HIDDEN=64`, `DROPOUT=0.5`, `DPRATE=0.0`, `ALPHA=0.1`.

## Editable interface

Exactly one region is editable — the `CustomProp` (propagation layer) and `CustomFilter` (full model) classes in `custom_filter.py`. `CustomProp` owns the `K+1` trainable filter parameters (`temp`), defines in `reset_parameters` how they are initialized, and in `forward(x, edge_index)` turns them into a degree-`K` polynomial of the (shifted, normalized) Laplacian applied to `x`; `CustomFilter` wraps it with the fixed MLP encoder and readout (and may set the `custom_*` training overrides). The starting point is the scaffold default — a PPR-like *monomial* polynomial with frozen PPR coefficients (APPNP, made parametric but not yet learned freely):

```python
# EDITABLE region of custom_filter.py — default fill (PPR-initialized monomial propagation)
class CustomProp(MessagePassing):
    def __init__(self, K, alpha=0.1, **kwargs):
        super(CustomProp, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha
        self.temp = Parameter(torch.Tensor(K + 1))   # K+1 polynomial coefficients
        self.reset_parameters()

    def reset_parameters(self):
        # PPR-like initialization: gamma_k = alpha (1-alpha)^k
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        # GCN-normalized adjacency P = D^{-1/2} A D^{-1/2}; monomial polynomial sum_k temp_k P^k x
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
    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super(CustomFilter, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop = CustomProp(K, alpha)
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

## Evaluation settings

Four datasets — **cora**, **citeseer** (homophilic citation), **texas**, **cornell** (heterophilic WebKB) — each over 10 random 60/20/20 splits with early stopping. Metric: **mean test accuracy over the 10 runs, higher is better**, reported per dataset (with the across-run standard deviation alongside). The contribution must remain a modular graph filter paired with this fixed pipeline; the data split and evaluation target are not to be changed.
