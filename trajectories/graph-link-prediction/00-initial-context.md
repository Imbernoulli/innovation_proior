## Research question

Link prediction on graphs: given a partially observed graph, score candidate node pairs so that true-but-held-out edges rank above non-edges. The design target is a single `LinkPredictor` — an **encoder** that maps nodes to embeddings from their features and observed connectivity, and a **decoder** that turns a pair of embeddings (plus, optionally, local structure around the pair) into an edge score. It must work without assuming a fixed feature dimension or graph size: the same class runs on small citation graphs with thousands of nodes and rich features and on a collaboration graph with a quarter-million nodes and 128-dim features. Data loading, the link split, negative sampling, the training loop, and the metrics are fixed; only the model is open to design.

## Prior art / Background / Baselines

Current approaches to link prediction each leave a concrete gap.

- **Heuristic structural scores (Adamic–Adar 2003; common neighbors; resource allocation).** Score a pair by a hand-designed function of its common neighborhood. No learning; ignores node features entirely; the formula is fixed and does not adapt to the data.
- **Random-walk embeddings (DeepWalk 2014; node2vec 2016).** Treat random walks as sentences and learn node embeddings with skip-gram; score a pair by the dot product of its node embeddings. Learns structure, but the embedding is transductive, ignores features, and the walk objective is decoupled from the link-prediction task.
- **GCN as an encoder (Kipf & Welling 2016).** A message-passing encoder folds features and structure into every node embedding in $O(|E|)$ and is end-to-end trainable. It has been applied to node classification, but it is not a link-prediction model: it defines no decoder for pairs and no objective for edge prediction.

## Fixed substrate / Code framework

A self-contained link-prediction harness (`custom_linkpred.py`) is frozen and must not be touched. It provides: dataset loading with an 85/5/10 random link split on the citation graphs (`RandomLinkSplit`, undirected, with negative train samples) and the official OGB split on `ogbl-collab`; per-epoch **negative sampling** that draws as many non-edges as positives; a training loop that optimizes **binary cross-entropy** on returned scores with Adam, gradient-norm clipping at 1.0, validation every `eval_every` epochs, and early stopping on validation AUC (citation) or validation Hits@50 (`ogbl-collab`); and metric computations (AUC, MRR, Hits@K). For `ogbl-collab` the loop folds validation edges into the adjacency at test time. A parameter budget is checked at startup. The loop only ever sees the scalar scores the model returns; any auxiliary objective must be injected into the scores' computation graph.

## Editable interface

Exactly one region is editable — the `LinkPredictor` class in `custom_linkpred.py`. The contract is three methods. `encode(x, edge_index) -> z` maps node features `[N, in_channels]` and training-graph connectivity `[2, E]` to a node-embedding table `[N, hidden_channels]`. `decode(edge_label_index, z, edge_index=None, num_nodes=None) -> scores` scores candidate edges `[2, M]` given `z`; crucially it receives the original node indices and the training `edge_index`, so a structure-aware decoder can read the live adjacency — train-only during validation, train+val at test, exactly as the loop hands it in. `forward(x, edge_index, edge_label_index)` chains encode→decode. Available building blocks: PyG's `GCNConv`, `SAGEConv`, `GATConv`, `GINConv`, `GraphConv`, `MessagePassing`, global pooling, and `torch_geometric.utils`.

The starting point is the scaffold below: a 2-layer GCN encoder with BatchNorm on intermediate layers and a pure dot-product decoder.

```python
# EDITABLE region of custom_linkpred.py — default fill (2-layer GCN encoder + dot-product decoder)
class LinkPredictor(nn.Module):
    """Default: 2-layer GCN encoder + dot-product decoder (simple baseline)."""

    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # BN only on intermediate layers — not the last layer, to preserve
        # embedding magnitude for dot-product scoring.
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
        ])

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        return (z_src * z_dst).sum(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z, edge_index=edge_index,
                           num_nodes=x.size(0))
```

## Evaluation settings

Three datasets. **Cora** (2,708 nodes, 10,556 edges, 1,433 features) and **CiteSeer** (3,327 nodes, 9,104 edges, 3,703 features) use the 85/5/10 random link split and report **AUC, MRR, Hits@20**, each over three seeds {42, 123, 456}. **ogbl-collab** (235,868 nodes, 1,285,465 edges, 128 features) uses the official OGB split and reports **Hits@50** and **MRR** (single seed 42). All metrics are higher-is-better. AUC measures separation of positives from a matched negative set; MRR and Hits@K are ranking metrics more sensitive to the top of the candidate list. Default hyperparameters: `hidden_channels=256`, `num_layers=2`, `dropout=0.0`, `lr=0.01`, `epochs=200`, early-stop `patience=20`, `eval_every=10`.
