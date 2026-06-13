**Problem.** Predict molecular properties on a scaffold split, where train and test share no scaffold,
from a frozen atom/bond graph. The encoder must learn a graph-to-vector map; rung one asks only for the
simplest learned encoder whose power I can reason about, to set an honest floor.

**Key idea (the starter GIN).** Measure the encoder against the Weisfeiler-Lehman test: no message-
passing GNN beats WL, and it *reaches* WL only when neighbor aggregation, combine, and readout are all
injective on multisets — which forces **sum** over mean/max and an **MLP** over a single linear layer,
with a `(1+ε)` tag separating the center atom from its neighbors. The starter keeps the injective,
edge-aware sum *message* (`MLP((1+ε)·x + Σ_{N(v)}(x[u]+edge_proj(e))))`) but deliberately uses the
**lossy mean** *readout* off the last layer — the simplest, weakest honest fill.

**Why it is the floor.** Mean pooling discards the multiplicity/count signal sum keeps; the receptive
field (4 hops) is smaller than a drug-molecule's diameter, so the pooled vector misses global structure;
and there is no external chemical prior or pretrained weight. On a scaffold split, memorization buys
nothing, so these weaknesses bite directly — hardest on the single-target task most decided by global
physicochemistry.

**Hyperparameters.** hidden 256, 4 `GINConv` layers (each: edge-aware sum message, `(1+ε)` center tag,
2-layer MLP with BatchNorm), per-layer BatchNorm + ReLU + residual, dropout 0.1, mean pooling, 2-layer
FFN head to `num_tasks`.

**What to watch.** Best on Tox21 (most data, multi-task averaging), decent on BACE (local substructure),
near chance on BBBP (single binary target, global physicochemistry, hard scaffold shift). A BBBP
collapse toward 0.5 forces a sum readout that keeps counts plus a global descriptor prior at rung two.

```python
class GINConv(nn.Module):
    """Graph Isomorphism Network convolution layer."""

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        x: [total_atoms, in_dim]
        edge_index: [2, total_edges]
        edge_attr: [total_edges, edge_dim]
        batch_idx: [total_atoms]
        """
        src, dst = edge_index
        edge_msg = self.edge_proj(edge_attr)
        msg = x[src] + edge_msg

        # Aggregate messages to destination nodes
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        out = self.mlp((1 + self.eps) * x + agg)
        return out


class MoleculeModel(nn.Module):
    """Starter model: Graph Isomorphism Network (GIN) with mean pooling.

    Simple but effective baseline for molecular property prediction.
    Uses message passing on the molecular graph with learned edge features.
    """

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 256
        num_layers = 4

        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, batch):
        """
        Args:
            batch: MolBatch with sparse graph data.
        Returns:
            predictions: [B, num_tasks]
        """
        x = self.atom_embed(batch.x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + self.dropout(x_new)  # residual

        # Mean pooling per graph
        num_graphs = batch.batch_idx.max().item() + 1
        graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        graph_embed.index_add_(0, batch.batch_idx, x)
        counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
        graph_embed = graph_embed / counts.clamp(min=1)

        return self.readout(graph_embed)
```
