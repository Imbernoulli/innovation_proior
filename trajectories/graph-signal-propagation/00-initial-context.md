## Research question

A graph neural network for node classification has to push each node's features through the graph
before reading them out, and the operator that does the pushing — a *graph filter* — decides what the
network can learn. On a **homophilic** graph (connected nodes share labels: citation networks, where
two linked papers usually share a topic) a smoothing, low-pass operator is exactly right — averaging a
node with its neighbors sharpens the class signal. On a **heterophilic** graph (connected nodes tend to
*differ*: webpage link graphs where a hub links to pages of many categories) smoothing is actively
harmful — the discriminative signal lives in how a node *contrasts* with its neighborhood, a
high-frequency component averaging destroys. The same fixed operator cannot serve both regimes.

So the single thing being designed is the **propagation filter**: a learnable polynomial of the graph
Laplacian that can take *any* frequency shape — low-pass, high-pass, band-pass, or a mixture — from
labels alone, and do it cheaply (linear, not quadratic, in the polynomial order `K`), without
overfitting through a wild high-order response, and — for some designs — admitting side constraints
(notably non-negativity) cleanly. Everything else about the model is fixed: the MLP encoder, the
dropouts, the optimizer, the data splits, the early-stopping evaluation.

## Prior art before the first rung

The fixed harness below is the *decoupled* `MLP → polynomial propagation → readout` template, and the
first rung reacts to the spectral-GNN lineage that converged onto it.

- **Spectral filtering via the Laplacian.** With the symmetric normalized Laplacian
  `L = I - D^{-1/2} A D^{-1/2} = U Λ U^T`, a filter applies a scalar `h` to each frequency,
  `y = U h(Λ) U^T x`. The eigendecomposition is `O(n^3)`, so `h` is restricted to a degree-`K`
  polynomial of `L`, `y ≈ sum_k w_k L^k x` — `K` sparse mat-vecs, no `U`. The normalized-Laplacian
  spectrum satisfies `0 ≤ λ ≤ 2` a priori (Chung 1997), which later saves an eigen-computation. Gap:
  *which* polynomial — which basis, which coefficients — is left open, and that choice is the whole game.
- **ChebNet (Defferrard, Bresson & Vandergheynst 2016).** The original learnable spectral GNN:
  `y ≈ sum_k w_k T_k(L_hat) x`, `L_hat = 2L/λ_max - I`, via the stable Chebyshev three-term recurrence,
  with the coefficients absorbed into a full weight matrix `W_k` per order. Gap: on citation graphs it
  *loses* to GCN — its own first-order special case — and the gap *widens* as `K` rises (Cora 80.5 at
  `K=2`, 74.9 at `K=10` vs GCN 81.3); a `W_k` per order also blows the parameter count up with `K`. The
  `w_k` are fit freely, with nothing tying them to a well-behaved response.
- **GCN (Kipf & Welling 2017).** ChebNet collapsed to first order (`K=1`, `w_0=-w_1`, `λ_max=2`) plus a
  renormalization trick, `y = w · D̃^{-1/2} Ã D̃^{-1/2} x`. Gap: a *fixed* low-pass filter — it cannot
  represent the high-/band-pass responses heterophilic graphs need, and is not learnable as a filter.
- **APPNP / decoupling (Klicpera et al. 2019).** Observed the feature transform (an MLP) and the graph
  propagation need not be interleaved: run `f_θ(X)` first, then a *fixed* PPR polynomial
  `sum_k α(1-α)^k P̃^k`. Keeps the parameter count tied to the MLP, independent of `K`. Gap: the
  propagation is fixed (PPR), not learned — it cannot adapt its frequency response to the graph.

The ladder below keeps APPNP's decoupled harness and makes the propagation *learnable*; the rungs
differ only in **how the `K+1` trainable parameters become the polynomial filter** — which basis, what
constraint, what initialization.

## The fixed substrate

A single training/evaluation pipeline is frozen and must not be touched. Four datasets spanning the
homophily range — **cora** (2,708 nodes, 7 classes, homophilic), **citeseer** (3,327, 6, homophilic),
**texas** and **cornell** (183, 5, heterophilic WebKB) — each run over 10 random 60/20/20
train/val/test splits with early stopping (patience 200, up to 1000 epochs), features row-normalized.
The model is the decoupled `dropout → Linear → ReLU → dropout → Linear → (dprate) → CustomProp →
log_softmax`, trained with Adam under NLL loss; the linear layers and the propagation parameters get
their own learning rate / weight decay (a model may override `LR`, `WEIGHT_DECAY`, `PROP_LR`, `PROP_WD`
through `custom_*` attributes). The loop also provides the operators a filter may use:
`gcn_norm(edge_index)` → `D^{-1/2} A D^{-1/2}`; `get_laplacian(edge_index, normalization='sym')` →
`L = I - D^{-1/2} A D^{-1/2}`; `add_self_loops(edge_index, weight, fill_value)` to shift the operator;
`self.propagate(edge_index, x=x, norm=norm)` for one sparse mat-vec; `cheby(i, x)` for `T_i(x)` at a
scalar; and `comb(n, k)` (binomial). Defaults: `K=10`, `HIDDEN=64`, `DROPOUT=0.5`, `DPRATE=0.0`,
`ALPHA=0.1`.

## The editable interface

Exactly one region is editable — the `CustomProp` (propagation layer) and `CustomFilter` (full model)
classes in `custom_filter.py`. Every method on the ladder is a fill of this same contract: `CustomProp`
owns the `K+1` trainable filter parameters (`temp`), defines in `reset_parameters` how they are
initialized, and in `forward(x, edge_index)` turns them into a degree-`K` polynomial of the (shifted,
normalized) Laplacian applied to `x`; `CustomFilter` wraps it with the fixed MLP encoder and readout
(and may set the `custom_*` training overrides). The starting point is the scaffold default — a PPR-like
*monomial* polynomial with frozen PPR coefficients (APPNP, made parametric but not yet learned freely):

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

Each later method replaces exactly these two class bodies (and, for the strongest rung, the `custom_*`
training overrides) and nothing else.

## Evaluation settings

Four datasets — **cora**, **citeseer** (homophilic citation), **texas**, **cornell** (heterophilic
WebKB) — each over 10 random 60/20/20 splits with early stopping. Metric: **mean test accuracy over the
10 runs, higher is better**, reported per dataset (with the across-run standard deviation alongside).
The contribution must remain a modular graph filter paired with this fixed pipeline; the data split and
evaluation target are not to be changed.
