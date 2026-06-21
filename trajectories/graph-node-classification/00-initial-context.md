## Research question

Semi-supervised node classification on citation networks: each node is a document with a sparse bag-of-words feature vector, edges are citations, and only about 20 nodes per class are labeled. The design object is the **message-passing mechanism** — how a node builds messages from neighbors, aggregates them, and updates its representation — stacked into a model that labels every node. The data, splits, optimizer, training loop, and metrics are fixed; the contribution is the propagation rule alone.

## Prior art / Background / Baselines

These methods put graph structure into a classifier. Each offers a usable mechanism and leaves a gap.

- **Label propagation (Zhu et al. 2003).** Spread the few known labels across edges by solving for a harmonic function on the graph. Gap: node features are ignored, and the rule hard-codes that linked nodes share a label.
- **Graph-as-regularizer (Belkin et al. 2006; Weston et al. 2012).** Train a feature predictor and add a Laplacian smoothness penalty that discourages different predictions on adjacent nodes. Gap: the graph only enters the loss; the model itself never propagates information across edges.
- **Spectral graph convolution (Bruna et al. 2014).** Define convolution through the graph Fourier basis given by the Laplacian eigenvectors. Gap: it needs an O(N^3) eigendecomposition, uses O(N) non-localized parameters, and is tied to one graph's spectrum.
- **ChebNet (Defferrard et al. 2016).** Approximate the spectral filter with a degree-K Chebyshev polynomial of the Laplacian. Gap: the receptive field is controlled by a fixed polynomial order, and the neighbor weights come from the supplied Laplacian rather than being learned from features.

## Fixed substrate / Code framework

The pipeline outside the editable region is frozen. PyTorch Geometric's `MessagePassing` orchestrates `message → aggregate → update` via `self.propagate(edge_index, ...)`. The loop provides `add_self_loops`, `degree`, `softmax`, and read-only reference convolutions (`GCNConv`, `GATConv`, `SAGEConv`). Data is the Planetoid citation networks with `T.NormalizeFeatures()`. Training is full-batch: cross-entropy on the labeled mask, Adam (`lr=0.01`, `weight_decay=5e-4`), up to 200 epochs with early stopping on validation accuracy (patience 50). The model uses `hidden_channels=64`, `num_layers=2`, `dropout=0.5`. A parameter budget enforces the model to stay within 1.05× the largest baseline's parameter count (computed per dataset from GraphSAGE / GPS / NAGphormer). The model may override `lr`/`weight_decay` by setting `custom_lr`/`custom_wd`; `hidden_channels`, `dropout`, and the `num_layers` argument value are passed in fixed.

The default fill below is a symmetric-normalized graph convolution.

## Editable interface

Exactly one region is editable: the `CustomMessagePassingLayer` (`MessagePassing` subclass) and the `CustomGNN` model in `custom_nodecls.py`. The layer defines `__init__`, `forward(x, edge_index)`, and `message(...)`; the model defines `__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)` and `forward(x, edge_index)` returning logits. Each method fills this same contract. The starting point is the scaffold default: a symmetric-normalized graph convolution. Each method replaces these two definitions.

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

Three citation networks with the standard Planetoid semi-supervised splits — **Cora** (2,708 nodes, 7 classes, 1,433 features), **CiteSeer** (3,327 nodes, 6 classes, 3,703 features), **PubMed** (19,717 nodes, 3 classes, 500 features) — about 20 labeled nodes per class, fixed masks, evaluated at seed 42. Two metrics per dataset, higher is better: **accuracy** and **macro-F1** on the test split.
