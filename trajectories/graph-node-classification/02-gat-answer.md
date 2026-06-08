**Problem (from step 1).** The mean aggregator collapsed on CiteSeer (0.6603, seeds {0.679, 0.674,
0.628}) while holding on Cora and PubMed — the signature of uniform neighbor weighting on a sparse,
mixed-class graph: no way to discount the off-topic citation, so a node's representation swings with
which neighbors it happens to have. The missing knob is a *learned* per-neighbor weight.

**Key idea.** Masked graph attention. A node attends over its real neighbors (plus a self-loop), scoring
each with a shared additive scorer and normalizing over the neighborhood:
$\alpha_{ij}=\mathrm{softmax}_j\big(\mathrm{LeakyReLU}(\vec a_s^\top\mathbf W\mathbf h_i+\vec a_d^\top\mathbf W\mathbf h_j)\big)$,
then $\mathbf h_i'=\sigma(\sum_{j\in N(i)}\alpha_{ij}\mathbf W\mathbf h_j)$. The mean is the constant-score
special case; attention is the generalization that can weight neighbors unequally. The source/destination
split makes the score a broadcast sum of two per-node scalars (no per-edge concatenation); LeakyReLU
preserves score ordering into the softmax where plain ReLU would zero it.

**Why it works.** Learned weights let the layer down-weight the off-topic neighbor that wrecked CiteSeer.
Multi-head attention (8 concatenated heads in layer 1, 1 averaged head at the output) replaces one
fragile weighting with a consensus of several — the variance reducer the mean lacked. Dropout 0.6 on the
attention coefficients samples a subset of each neighborhood per step — augmentation on the connectivity,
for the ~20-labels-per-class regime.

**Step-2 edit.** Replace the mean layer with multi-head additive attention; ELU activation, feature and
attention dropout 0.6.

**The bar to beat.** graphsage means: Cora 0.7923, CiteSeer 0.6603, PubMed 0.7693. Expect Cora clearly up
(past 0.82), CiteSeer's mean up and its seed spread tightened, PubMed to roughly hold (clean neighborhoods
left little room) — a flat/marginally-lower PubMed would not falsify; a Cora that fails to beat 0.7923 or
a CiteSeer that stays as noisy would.

```python
# EDITABLE region of custom_nodecls.py — step 2: multi-head graph attention
class CustomMessagePassingLayer(MessagePassing):
    """Graph attention layer with multi-head additive attention."""

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 8, concat: bool = True, negative_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        if concat:
            assert out_channels % heads == 0
            self.head_dim = out_channels // heads
        else:
            self.head_dim = out_channels
        self.lin = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))   # a_s (source score half)
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))   # a_d (destination half)
        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        H, D = self.heads, self.head_dim
        x = self.lin(x).view(-1, H, D)
        alpha_src = (x * self.att_src).sum(dim=-1)             # per-node source scalar (per head)
        alpha_dst = (x * self.att_dst).sum(dim=-1)             # per-node destination scalar (per head)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, alpha_src=alpha_src, alpha_dst=alpha_dst)
        out = out.view(-1, H * D) if self.concat else out.mean(dim=1)
        return out + self.bias

    def message(self, x_j: Tensor, alpha_src_i: Tensor, alpha_dst_j: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_src_i + alpha_dst_j                      # additive score = a_s.Wh_i + a_d.Wh_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)            # normalize over the neighborhood
        alpha = F.dropout(alpha, p=0.6, training=self.training)  # dropout on attention coefficients
        return x_j * alpha.unsqueeze(-1)


class CustomGNN(nn.Module):
    """Multi-head attention GNN: 8 concat heads then 1 averaged head."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels, heads=8, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels, heads=8, concat=True))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```
