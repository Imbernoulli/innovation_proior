## Research question

Semi-supervised node classification on citation networks: each node is a document with a sparse
bag-of-words feature vector, edges are citations, and only ~20 nodes per class are labeled. The single
thing being designed is the **message-passing mechanism** — how a node constructs messages from its
neighbors, aggregates them, and updates its own representation — stacked into a model that labels every
node. Everything around it (the data, the splits, the optimizer, the training loop, the metrics) is
fixed; the contribution is the propagation/model design alone.

## Prior art before the first rung (pre-message-passing lineage)

The message-passing layer the first rung fills reacts to a line of attempts to put the graph into a
neural classifier. These precede the ladder; each leaves a gap the rungs close.

- **Label propagation (Zhu et al. 2003).** Solve for a harmonic function where each unlabeled node is the
  degree-weighted average of its neighbors' labels, in closed form. Uses the edges but ignores the node
  features entirely. Gap: no features, and "edge ⇒ same label" is baked in.
- **Graph-as-regularizer (Belkin et al. 2006; Weston et al. 2012).** Keep a feature-only predictor
  $f(\mathbf X)$ and add a Laplacian smoothness penalty $\sum_{ij}A_{ij}\|f_i-f_j\|^2$ to the loss. The
  graph only ever appears as a smoothness whip; the architecture itself carries no information across
  edges, and the penalty again assumes neighbors share a label. Gap: structure never enters the model.
- **Spectral graph convolution (Bruna et al. 2014).** Define convolution through the graph Fourier basis,
  $g_\theta\star\mathbf x=\mathbf U\,\mathrm{diag}(\theta)\,\mathbf U^\top\mathbf x$, with $\mathbf U$ the
  eigenvectors of the normalized Laplacian. Principled but needs an $O(N^3)$ eigendecomposition, has
  $O(N)$ non-localized parameters, and is welded to one graph's spectrum. Gap: expensive, non-local,
  non-transferable.
- **ChebNet (Defferrard et al. 2016).** Replace the free spectral filter with a degree-$K$ Chebyshev
  polynomial in the Laplacian, $\sum_k\theta_k T_k(\tilde{\mathbf L})\mathbf x$ — $K$-hop localized,
  $O(K|E|)$, no eigendecomposition. Gap: filter order and receptive field are still tangled, and the
  neighbor weights are determined by the supplied Laplacian, not learned from features.

The renormalized graph convolution (Kipf & Welling 2017) squeezed ChebNet to $K=1$ with the
renormalization trick — $\tilde{\mathbf P}=\tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\tilde{\mathbf
D}^{-1/2}$, $\tilde{\mathbf A}=\mathbf A+\mathbf I$ — and is exactly the **default fill** of the scaffold
below: a per-edge symmetric-normalized message. The ladder starts by replacing this default.

## The fixed substrate

The whole pipeline outside the editable region is frozen. PyTorch Geometric's `MessagePassing`
orchestrates `message → aggregate → update` via `self.propagate(edge_index, ...)`; the loop provides
`add_self_loops`, `degree`, `softmax`, and read-only reference convolutions (`GCNConv`, `GATConv`,
`SAGEConv`). Data is the Planetoid citation networks with `T.NormalizeFeatures()`. Training is
full-batch: cross-entropy on the labeled mask only, Adam (`lr=0.01`, `weight_decay=5e-4`), up to 200
epochs with early stopping on validation accuracy (patience 50). The model is built with
`hidden_channels=64`, `num_layers=2`, `dropout=0.5`. A **parameter budget** is enforced — the model must
stay within $1.05\times$ the largest baseline's parameter count (computed per dataset from GraphSAGE /
GPS / NAGphormer). The model may override `lr`/`weight_decay` by setting `custom_lr`/`custom_wd`
attributes; `hidden_channels`, `dropout`, and the `num_layers` argument value are passed in fixed.

## The editable interface

Exactly one region is editable — the `CustomMessagePassingLayer` (a `MessagePassing` subclass) and the
`CustomGNN` model in `custom_nodecls.py`. The layer defines `__init__`, `forward(x, edge_index)`, and
`message(...)`; the model defines `__init__(in_channels, hidden_channels, out_channels, num_layers,
dropout)` and `forward(x, edge_index)` returning logits. Every method on the ladder is a fill of this
same contract. The starting point is the scaffold **default**: a symmetric-normalized graph convolution.
Each later method replaces these two definitions.

```python
# EDITABLE region of custom_nodecls.py — default fill (symmetric-normalized convolution)
class CustomMessagePassingLayer(MessagePassing):
    """Default: standard graph convolutional layer (symmetric normalization)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.lin(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]            # 1/sqrt(d_i d_j) per edge
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    """Default: 2-layer convolution with ReLU and dropout."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```

## Evaluation settings

Three citation networks with the standard Planetoid semi-supervised splits — **Cora** (2,708 nodes, 7
classes, 1,433 features), **CiteSeer** (3,327 nodes, 6 classes, 3,703 features), **PubMed** (19,717
nodes, 3 classes, 500 features) — about 20 labeled nodes per class, fixed masks, evaluated at seed 42.
Two metrics per dataset, higher is better on both: **accuracy** and **macro-F1** on the test split.
