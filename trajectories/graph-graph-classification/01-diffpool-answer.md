**Problem.** The backbone emits one embedding per node; graph classification needs one vector per
graph. A single global sum is *flat* — it collapses all nodes in one step and represents no scale
between a node's neighborhood and the whole graph, the functional-group scale where the label often
lives.

**Key idea.** Borrow CNN-style pooling: coarsen the graph by *clustering* nodes. A hard clustering is
non-differentiable, so learn a **soft assignment** `S = softmax(scores)` (each node a distribution over
clusters) and pool `X' = Sᵀ X` — permutation-invariant since for any node permutation P,
`(PS)ᵀ(PX) = SᵀX`. Then reduce the K cluster-nodes to one graph vector.

**Why it works (and its known weakness here).** The pool keeps a *learned*, soft grouping of nodes
rather than one flat reduction. But the task's edit surface is only the post-backbone `GraphReadout`:
there is no hook for the canonical method's separate pooling-GNN, coarsened-adjacency message passing,
or the link-prediction / entropy auxiliary losses (only the classification cross-entropy is
backpropagated). So this is the *stripped* DiffPool — a single coarsening level, the assignment from a
plain MLP over node features (adjacency-blind), no auxiliaries. With no structural prior to make
connected nodes share clusters and no entropy term to keep the softmax crisp, the assignment can stay
diffuse, in which case the pool degenerates toward a learned, noisier global *mean* — which discards the
counts a sum keeps. It is the bottom rung by construction.

**Scaffold edit / hyperparameters.** Dense-batch with `to_dense_batch` (pad to `max_nodes=150`),
`num_clusters=25`; assignment MLP `Linear(D,D)→ReLU→Linear(D,K)`; mask padded nodes to −∞ pre-softmax,
softmax over nodes, re-zero padded rows; pool `X' = Sᵀ X_dense`; final graph vector = mean over the K
clusters. `output_dim = hidden_dim`. (The coarsened adjacency `Sᵀ A S` is computed in the canonical
method; here only the embedding pool `Sᵀ X` is used for the graph vector.)

**What to watch.** Likely near, or just below, a plain global pool: high-variance on tiny MUTAG, and at
best mean-like on PROTEINS/NCI1. Its failure — discarded counts, no use of the per-layer
`layer_outputs`, no injective reduction — forces the move to an injective *sum read from every layer* at
the next rung.

```python
class GraphReadout(nn.Module):
    """DiffPool Readout (Ying et al., 2018).

    Uses a learned soft assignment matrix to cluster nodes into
    a fixed number of super-nodes, then reads out from the
    coarsened graph. Two-level hierarchy.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Assignment network: maps nodes to clusters
        self.max_nodes = 150  # Max nodes per graph (padded)
        self.num_clusters = 25
        self.assign_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_clusters),
        )
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Convert to dense batch format
        x_dense, mask = to_dense_batch(x, batch)  # [B, N_max, D]
        adj = to_dense_adj(edge_index, batch)  # [B, N_max, N_max]

        # Compute soft assignment
        s = self.assign_nn(x_dense)  # [B, N_max, K]
        s = s.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        s = torch.softmax(s, dim=1)
        s = s * mask.unsqueeze(-1).float()

        # Pool: X_coarse = S^T @ X, A_coarse = S^T @ A @ S
        x_coarse = torch.bmm(s.transpose(1, 2), x_dense)  # [B, K, D]

        # Global mean pool over clusters
        return x_coarse.mean(dim=1)  # [B, D]
```
