# SAGPool, distilled

SAGPool (Self-Attention Graph Pooling) is a learned, hierarchical graph-pooling layer for
graph classification. It scores every node with a single-output graph convolution — so the
importance score depends on both the node's features and its place in the graph — keeps the
top-ranked fraction of nodes, gates the kept features by their scores (which makes the score
parameters trainable through the otherwise non-differentiable top-k), and indexes the sparse
adjacency to the survivors. Stacking several conv-then-pool blocks with a per-block readout
gives a multi-scale graph embedding at sparse cost and with a parameter count independent of
graph size.

## Problem it solves

Map a variable-size graph to one fixed-size, permutation-invariant vector for classification,
while (a) using node features *and* graph topology to decide what to keep, (b) building a
hierarchical (multi-stage) coarsening rather than a single flat readout, (c) staying sparse —
O(|V|+|E|) storage — and (d) keeping the number of parameters independent of the number of
nodes, so one module fits graphs from tens to thousands of nodes.

## Key idea

Produce a per-node self-attention score with a graph convolution of output width 1:

```
Z = sigma( D̃^{-1/2} Ã D̃^{-1/2} X Θ_att ),   Ã = A + I_N,   Θ_att ∈ R^{F×1},   Z ∈ R^{N×1},
```

with `sigma = tanh`. Because the normalized-adjacency term `D̃^{-1/2} Ã D̃^{-1/2}` mixes each
node with its neighbors, `Z_i` depends on node `i`'s features *and* its neighborhood — unlike a
pure feature projection, the score sees the topology. Then select the top-`⌈kN⌉` nodes by `Z`,
gate the kept features, and index the adjacency:

```
idx     = top-rank(Z, ⌈kN⌉),   Z_mask = Z_idx,
X_out   = X_{idx,:} ⊙ Z_mask,             # gate: routes gradient to Θ_att
A_out   = A_{idx,idx}.                     # sparse indexing keeps the coarsened graph sparse
```

Two design points carry the method:

- **Topology-aware scoring at sparse cost.** Replacing a feature-only projection `y = Xp/||p||`
  with a single-output graph convolution makes the keep/drop decision depend on graph
  structure, while costing only one length-`F` weight vector per layer and reusing the
  normalized Laplacian already computed by the block's convolution. Indexing `A_{idx,idx}`
  (rather than a dense contraction `S^T A S`) keeps storage at O(|V|+|E|) and parameters
  independent of node count.
- **The gate is the differentiable bridge.** `top-rank` is a discrete `argsort` with zero
  gradient w.r.t. the scores, so selection alone would leave `Θ_att` untrained. Multiplying the
  kept features by `tanh(Z_idx)` makes the forward features depend continuously on `Z`, so
  gradient flows back to `Θ_att`. `tanh` is bounded (a wild score can't blow up a node),
  centered at zero (low-importance boundary nodes are suppressed toward zero, and sign is
  preserved), and monotone (ranking order and gate magnitude agree).

A **ratio** `k ∈ (0,1]` (keep `⌈kN⌉` nodes), not a fixed count, adapts the coarsening to each
graph's size and is what lets one module serve graphs of very different sizes.

## Variants

The score only needs a GNN that maps `(X, A)` to a per-node scalar, `Z = σ(GNN(X, A))`, so any
message-passing layer works (GCN default; also Chebyshev, GraphSAGE, GAT). Two-hop reach:
augment edges, `Z = σ(GNN(X, A + A^2))` (since `A^2` connects two-hop neighbors), or stack
scoring convs, `Z = σ(GNN_2(σ(GNN_1(X, A)), A))`. Multi-head: average `M` scoring GNNs,
`Z = (1/M) Σ_m σ(GNN_m(X, A))`.

## Architectures

- **Hierarchical:** three blocks, each a graph-convolution layer followed by a SAGPool layer;
  a per-block readout `s = (1/N Σ_i x_i) || (max_i x_i)` (mean concatenated with max); the
  block readouts are summed and fed to a linear classifier.
- **Global:** three graph-convolution layers, outputs concatenated, one SAGPool readout, linear
  classifier. ReLU activations; graph convolution fixed to the Kipf-Welling rule for fairness
  across methods.

## Complexity

Sparse pooling: O(|E|) compute, O(|V|+|E|) storage, versus dense O(|V|^2). Parameters per
pooling layer = one `F×1` convolution weight, independent of `N`.

## Working code

A SAGPool layer (scalar GCN score, `tanh` gate, sparse adjacency index) and a hierarchical
`GraphReadout` built from three conv-then-pool blocks with summed mean||max readouts:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool.topk_pool import topk, filter_adj  # per-graph top-k + sparse edge filter


class SAGPool(nn.Module):
    """Self-attention graph pooling layer (Lee et al., 2019). Scores nodes with a
    single-output graph convolution (features + topology), keeps the top
    ceil(ratio*N) per graph, gates kept features by tanh(score), indexes adjacency."""

    def __init__(self, in_channels, ratio=0.5, Conv=GCNConv, non_linearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        # the only parameter: Z = sigma(D^-1/2 A~ D^-1/2 X Theta_att), Theta_att in R^{F x 1}
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()                 # Z in R^N
        perm = topk(score, self.ratio, batch)                            # top-rank(Z, ceil(ratio*N))
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)        # gate -> trainable scores
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(                              # A[idx, idx], stays sparse
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm


class GraphReadout(nn.Module):
    """Hierarchical SAGPool readout: 3 x (GCN conv -> SAGPool), each block summarized
    by mean||max; block summaries summed into one graph-level vector."""

    def __init__(self, hidden_dim, num_layers, ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.pool1 = SAGPool(hidden_dim, ratio=ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPool(hidden_dim, ratio=ratio)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = SAGPool(hidden_dim, ratio=ratio)
        self.output_dim = hidden_dim * 2          # mean||max per block, summed across blocks

    def _readout(self, x, batch):
        # s = (1/N sum_i x_i) || (max_i x_i)
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)

    def forward(self, x, edge_index, batch, layer_outputs):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        s1 = self._readout(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        s2 = self._readout(x, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        s3 = self._readout(x, batch)

        return s1 + s2 + s3
```

For the modern PyTorch-Geometric library form, the same layer is `SAGPooling(in_channels,
ratio=0.5, GNN=GCNConv, nonlinearity='tanh')`: it computes `score = tanh((X·w)/||w||)` (or a
graph-conv score), `perm = topk(score, ratio, batch)`, `x = x[perm] * score[perm]`, and filters
edges to the retained nodes, returning `(x, edge_index, edge_attr, batch, perm, score)`.

## Relation to prior methods

- **DiffPool** (soft, dense assignment `S`, `A^{(l+1)} = S^T A^{(l)} S`): SAGPool replaces soft
  assignment with hard top-k node selection, so the coarsened adjacency is `A_{idx,idx}` (sparse,
  O(|V|+|E|)) instead of a dense O(|V|^2) contraction, and parameters no longer depend on a
  fixed cluster count / graph size.
- **gPool / Graph U-Nets** (`y = Xp/||p||`, top-k, sigmoid gate): SAGPool keeps the top-k +
  gate machinery but replaces the topology-blind feature projection with a graph-convolution
  score, so node importance depends on the graph structure, at the same parameter cost.
- **SortPool / Set2Set**: flat global readouts with no learned hierarchical coarsening; SAGPool
  coarsens in stages.
