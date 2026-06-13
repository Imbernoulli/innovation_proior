**Problem (from the gin-sum rung).** The injective JK-sum is robust (PROTEINS 70.98→74.54, NCI1
78.25→79.52, MUTAG variance tightened) but pools *every node with equal weight*. On motif-driven graphs
the label is carried by a few substructures, and a uniform sum *dilutes* that signal across all nodes —
the readout has no notion that some nodes matter more.

**Key idea.** Make the readout *selective* and *hierarchical*. Score each node's importance with a single
graph convolution `Z = σ(GNN(X, A))` — structure-aware, so importance depends on connectivity, not
features alone (unlike DiffPool's adjacency-blind MLP). Keep the top `⌈k·N⌉` nodes and **gate** each
survivor's features by its own score (`X_idx ⊙ Z_idx`), which lets gradient flow back into the score net
through the hard top-k. Filter edges to the induced subgraph, then repeat to coarsen.

**Why it works (and its risk).** Selection concentrates the readout on the decisive nodes, undoing
gin-sum's dilution where the label is a local motif. But hard top-k at `ratio=0.5` *discards half the
nodes per level* (a quarter survive by level 2); where the decision is *distributed* across the graph
(large chemical sets), the score can drop exactly the nodes that mattered — and the discard is
irreversible.

**Scaffold edit / hyperparameters.** Two `SAGPooling(hidden_dim, ratio=0.5)` levels. Read out
`[sum-pool, mean-pool]` at three scales (original + 2 coarsened), concatenate (`6·hidden_dim`), project
to `hidden_dim`. `output_dim = hidden_dim`. The score convolution lives inside each pool and operates on
the final-layer GIN embeddings `x` (using `edge_index`/`batch`, which the surface exposes).

**What to watch.** Expect MUTAG and PROTEINS to clear gin-sum (84.02, 74.54) by concentrating on
decisive nodes — possibly a large MUTAG jump — but NCI1 to *regress* from 79.52, with high seed variance,
because halving the node set twice should destroy the distributed signal gin-sum kept by reading
everything. A big-MUTAG aggregate win bought by trading NCI1 away.

```python
class GraphReadout(nn.Module):
    """SAGPool Hierarchical Readout (Lee et al., 2019).

    Uses self-attention scores to hierarchically select top-k nodes,
    then applies sum+mean global readout on the coarsened graph.
    Two-level hierarchy: original -> coarsened.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        from torch_geometric.nn.pool import SAGPooling
        self.pool1 = SAGPooling(hidden_dim, ratio=0.5)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.5)
        # 3 levels (original + 2 coarsened), each with sum+mean
        self.output_dim = hidden_dim * 2 * 3
        self.proj = nn.Linear(self.output_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Level 0: readout on original graph
        r0 = torch.cat([global_add_pool(x, batch),
                         global_mean_pool(x, batch)], dim=-1)

        # Level 1: first coarsening
        x1, edge_index1, _, batch1, perm1, score1 = self.pool1(
            x, edge_index, batch=batch)
        r1 = torch.cat([global_add_pool(x1, batch1),
                         global_mean_pool(x1, batch1)], dim=-1)

        # Level 2: second coarsening
        x2, edge_index2, _, batch2, perm2, score2 = self.pool2(
            x1, edge_index1, batch=batch1)
        r2 = torch.cat([global_add_pool(x2, batch2),
                         global_mean_pool(x2, batch2)], dim=-1)

        return self.proj(torch.cat([r0, r1, r2], dim=-1))
```
