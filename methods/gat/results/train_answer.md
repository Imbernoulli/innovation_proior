I need a neural layer that operates directly on graph-structured data and recovers what makes convolution so effective on grids: shared, learnable parameters applied at every position, with no dependence on a fixed neighborhood size or ordering. On a grid every pixel has four canonical neighbors, but on a graph a node may have two neighbors or two thousand, and there is no "up" or "left" to align against. Spectral graph convolutions define the operation through the graph Fourier transform, but that welds the learned filter to one graph's Laplacian eigenbasis: an O(N^3) eigendecomposition is required, and the model is simply not defined on a graph with a different structure. Chebyshev and GCN-style approximations drive the cost down to O(|E|), yet GCN pins every unweighted edge contribution to the fixed, untrainable coefficient 1/√(d̃_i d̃_j), so it cannot learn that one neighbor matters more than another. GraphSAGE escapes by learning feature-based aggregators, which makes the layer inductive, but it samples a fixed-size neighborhood, weights neighbors uniformly in its mean/GCN aggregators, and its best aggregator is an LSTM that must be fed neighbors in random permutations because the neighbor set has no natural order.

The missing primitive is one that takes a variable-sized, unordered set of neighbors and returns a learned weight for each. That is exactly attention. I propose the Graph Attention Network, or GAT. The core idea is masked self-attention over the local neighborhood: each node attends over its neighbors (including itself via a self-loop), scores each neighbor with a shared additive function of node features, normalizes the scores into a softmax distribution over the neighborhood, and aggregates the neighbors' transformed features with those learned coefficients. Because the scoring and transformation are shared functions of node features, the same parameters apply to any graph, making the layer inductive. Because the coefficients are learned from content rather than fixed by degrees, different neighbors can receive different importances. Because the aggregation is a weighted sum over a set, no neighborhood ordering is ever imposed.

Concretely, given node features h_i, a shared linear transform W is applied to every node. An attention vector a is split into source and destination halves so the score for edge i←j is the broadcast sum a_s^T W h_i + a_d^T W h_j, passed through LeakyReLU with negative slope 0.2. Softmax normalizes these scores over each node's neighborhood, and the updated feature is h_i' = σ(Σ_{j∈N(i)} α_ij W h_j). Multiple independent attention heads are run in parallel: their outputs are concatenated in hidden layers to enrich the representation, and averaged at the output layer so that class logits remain a proper score vector. Dropout on the attention coefficients provides a stochastic neighborhood sample at each training step, which is an effective regularizer in the small-label regimes typical of citation networks. The per-head cost is O(|V|FF' + |E|F'), on par with GCN, with no eigendecomposition or inversion.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch import Tensor
from typing import Optional


class GATLayer(MessagePassing):
    """Single graph attention layer with multi-head additive attention."""

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 8, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.6):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.head_dim = out_channels // heads if concat else out_channels
        self.lin = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index) -> Tensor:
        H, D = self.heads, self.head_dim
        x = self.lin(x).view(-1, H, D)
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, alpha_src=alpha_src, alpha_dst=alpha_dst)
        out = out.view(-1, H * D) if self.concat else out.mean(dim=1)
        return out + self.bias

    def message(self, x_j: Tensor, alpha_src_i: Tensor, alpha_dst_j: Tensor,
                index: Tensor, ptr, size_i: Optional[int]) -> Tensor:
        alpha = alpha_src_i + alpha_dst_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class GAT(nn.Module):
    """Two-layer GAT for transductive node classification."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATLayer(in_channels, hidden_channels * 8,
                              heads=8, concat=True, dropout=dropout)
        self.conv2 = GATLayer(hidden_channels * 8, out_channels,
                              heads=1, concat=False, dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```
