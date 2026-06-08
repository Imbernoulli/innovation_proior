**Problem.** The scaffold default folds a node's own features into the same degree-normalized pot as its
neighbors. The simplest *separated* aggregator on the ladder gives the node a clean self channel and a
neighbor channel, with the most neutral neighbor weighting — a plain mean — so the floor reads what bare
two-hop aggregation is worth before any learned weighting.

**Key idea.** Mean-aggregation message passing with separate self/neighbor projections:
$\mathbf h_i'=\mathbf W_{\text{self}}\mathbf h_i+\mathbf W_{\text{neigh}}\,\mathrm{mean}_{j\in N(i)}\mathbf h_j$,
then $\ell_2$-normalize each node. The mean is permutation-invariant and assumption-free (every neighbor
counts equally); separating $\mathbf W_{\text{self}}$ from $\mathbf W_{\text{neigh}}$ lets the node weight
itself independently of the neighborhood; $\ell_2$ normalization keeps feature scale stable across the
two layers.

**Step-1 edit.** Replace the default symmetric-normalized layer with `aggr="mean"`, two linear maps, and
`F.normalize`. Two layers (hidden then logits), ReLU + dropout 0.5 between — a two-hop receptive field.

**What to watch.** The mean blurs mixed-class neighborhoods and applies no degree damping, so I expect
the weakest, highest-variance results on the sparsest, least-homophilous dataset (CiteSeer). That uniform
neighbor weighting is what forces a *learned* per-neighbor weight at step 2.

```python
# EDITABLE region of custom_nodecls.py — step 1: mean-aggregation message passing
class CustomMessagePassingLayer(MessagePassing):
    """Mean-aggregation layer with separate self/neighbor projections + L2 normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.zeros_(self.lin_self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        neigh_agg = self.propagate(edge_index, x=x)        # mean over neighbors, no self-loop
        out = self.lin_self(x) + self.lin_neigh(neigh_agg)  # separate self / neighbor channels
        out = F.normalize(out, p=2, dim=-1)                 # project each node onto the unit sphere
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class CustomGNN(nn.Module):
    """Mean-aggregation GNN with L2 normalization."""

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
