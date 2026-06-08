**Problem (from step 2).** Attention helped Cora (0.8260) but bought CiteSeer only an unstable lift
(0.7077, seeds {0.711, 0.696, 0.716} — still ~2-point swing) and nothing on PubMed (0.7777). On strongly
homophilous citation graphs, learning $\alpha_{ij}$ from ~20 labels per class overfits; the "right"
neighbor weight is mostly just a sensible degree-based discount, which a *fixed* operator can supply at no
parameter cost and no seed variance.

**Key idea.** Drop the learned weight; use the principled fixed weight — the renormalized symmetric graph
convolution. First-order spectral filter with tied coefficients gives $\mathbf I+\mathbf
D^{-1/2}\mathbf A\mathbf D^{-1/2}$ ("self + symmetric-normalized neighbors"); the $1/\sqrt{d_id_j}$ weight
damps high-degree neighbors, the damping the plain mean lacked and attention was paying to relearn. Fold
the self-loop in *before* normalizing to pull the spectral radius from ~2 back to 1:
$\tilde{\mathbf P}=\tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\tilde{\mathbf D}^{-1/2}$,
$\tilde{\mathbf A}=\mathbf A+\mathbf I$. One shared $\mathbf W$ per layer, no attention vectors, no per-edge
softmax.

**Why it works.** No learned edge weights means nothing to overfit the tiny label set, so the result is
low-variance; the principled degree normalization captures the useful part of attention on homophilous
graphs. Fewer parameters is itself a regularizer at 20 labels/class.

**Step-3 edit.** Replace the attention layer with the renormalized convolution — the scaffold default.
Two layers, ReLU + dropout 0.5.

**The bar to beat.** gat means: Cora 0.8260, CiteSeer 0.7077, PubMed 0.7777. Expect PubMed to hold or
slightly exceed (~0.785), CiteSeer's mean up and spread tightened (~0.715), Cora close to gat (~0.82,
possibly a touch under — the one place a fixed weight risks losing the learned sharpening). A CiteSeer as
noisy as gat's, or a PubMed below 0.7777, would falsify the "fixed weight matches attention on
homophilous graphs" thesis.

```python
# EDITABLE region of custom_nodecls.py — step 3: renormalized symmetric graph convolution
class CustomMessagePassingLayer(MessagePassing):
    """Standard graph convolutional layer (symmetric normalization, renormalization trick)."""

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
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))   # self-loop folded in first
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]            # 1/sqrt(d~_i d~_j): degree damping
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    """2-layer renormalized graph convolution with ReLU and dropout."""

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
