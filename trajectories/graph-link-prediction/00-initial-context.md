## Research question

Link prediction on graphs: given a partially observed graph, score candidate node pairs so that the
true-but-held-out edges rank above non-edges. The thing being designed is a single `LinkPredictor` —
an **encoder** that maps nodes to embeddings from their features and the observed connectivity, and a
**decoder** that turns a pair of embeddings (plus, optionally, the graph structure around the pair)
into an edge score. It must work without assuming a fixed feature dimension or graph size: the same
class is instantiated on small citation graphs with thousands of nodes and rich features and on a
collaboration graph with a quarter-million nodes and 128-dim features. Everything around the model —
data loading, the link split, negative sampling, the training loop, the metrics — is fixed; only the
model is mine to design.

## Prior art before the first rung (the link-prediction lineage)

The first rung reacts to a line of unsupervised graph-representation methods. These are what the
ladder starts from; each had a gap the next move tried to close.

- **Heuristic structural scores (Adamic–Adar 2003; Liben-Nowell & Kleinberg 2007).** Score a pair by a
  hand-designed function of their neighborhoods — common neighbors, Adamic–Adar
  $\sum_{w\in N(u)\cap N(v)} 1/\log\deg(w)$, resource allocation $\sum 1/\deg(w)$. No learning, no node
  features, surprisingly strong on dense citation/collaboration graphs. Gap: a fixed formula ignores
  node attributes entirely and cannot adapt to the graph.
- **Random-walk embeddings (DeepWalk, Perozzi et al. 2014; node2vec, Grover & Leskovec 2016).** Treat
  truncated random walks as sentences and run skip-gram to embed nodes, then score a pair by the dot
  product of embeddings. Learns structure, but the embedding is *transductive* (one vector per node,
  no inductive map from features), it never reads node features, and the walk objective is decoupled
  from the link-prediction objective. Gap: ignores features; embeddings not learned end-to-end for the
  task.
- **GCN as an encoder (Kipf & Welling 2016, arXiv:1609.02907).** A message-passing encoder
  $H^{(l+1)}=\sigma(\hat A H^{(l)} W^{(l)})$ with $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ folds
  features *and* structure into every node embedding in $O(|E|)$ and is end-to-end trainable. It solved
  the "use features and structure together, inductively" problem for node classification — but it is an
  encoder, not a link-prediction model: it says nothing about how a *pair* of embeddings becomes an edge
  score, or whether the encoder should be trained as a generative model of the graph. Gap: needs a
  decoder and a training objective specialized to predicting edges.

That last gap is the substrate the first rung sits on: a GCN encoder is available, the question is how
to wire it into a predictor of links.

## The fixed substrate

A self-contained link-prediction harness (`custom_linkpred.py`) is frozen and must not be touched. It
provides: dataset loading with an 85/5/10 random link split on the citation graphs (`RandomLinkSplit`,
undirected, with negative train samples) and the official OGB split on `ogbl-collab`; per-epoch
**negative sampling** (`negative_sampling` draws as many non-edges as positives each epoch); a training
loop that optimizes **binary cross-entropy** on the returned scores (positives → 1, negatives → 0)
with Adam, gradient-norm clipping at 1.0, validation every `eval_every` epochs, and early stopping on
validation AUC (citation) or validation Hits@50 (`ogbl-collab`); and the metric computations
(`roc_auc_score`, `compute_mrr`, `compute_hits_at_k`). For `ogbl-collab` the loop follows the OGB
protocol of folding validation edges into the adjacency at test time. A **parameter budget** is checked
at startup (≈1.05× the largest reference model). The loop only ever sees the scalar scores my model
returns; any auxiliary objective a model wants (e.g. a KL term) has to be injected into the scores'
computation graph, because the loop computes only BCE on those scores.

## The editable interface

Exactly one region is editable — the `LinkPredictor` class (lines 127–210 of `custom_linkpred.py`).
The contract is three methods. `encode(x, edge_index) -> z` maps node features `[N, in_channels]` and
the training-graph connectivity `[2, E]` to a node-embedding table `[N, hidden_channels]`.
`decode(edge_label_index, z, edge_index=None, num_nodes=None) -> scores` scores candidate edges
`[2, M]` given the full embedding table `z`; crucially it receives the *original node indices* in
`edge_label_index` and the training `edge_index`, so a structure-aware decoder can compute neighborhood
features (common neighbors, Adamic–Adar, …) directly against the live adjacency — train-only during
validation, train+val at test, exactly as the loop hands it in. `forward(x, edge_index,
edge_label_index)` chains encode→decode. Available building blocks (pre-installed): PyG's `GCNConv`,
`SAGEConv`, `GATConv`, `GINConv`, `GraphConv`, `MessagePassing`, global pooling, and
`torch_geometric.utils` (`negative_sampling`, `to_undirected`, `degree`, `coalesce`).

The starting point is the scaffold default below: a 2-layer GCN encoder with BatchNorm on intermediate
layers and a **pure dot-product** decoder. Every method on the ladder replaces exactly this class.

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

Three datasets. **Cora** (2,708 nodes, 10,556 edges, 1,433 features) and **CiteSeer** (3,327 nodes,
9,104 edges, 3,703 features) use the 85/5/10 random link split and report **AUC, MRR, Hits@20**, each
over three seeds {42, 123, 456}. **ogbl-collab** (235,868 nodes, 1,285,465 edges, 128 features) uses
the official OGB split and reports **Hits@50** and **MRR** (single seed 42). All metrics are
higher-is-better. AUC measures separation of positives from a matched negative set; MRR and Hits@K are
ranking metrics that are far more sensitive to the *top* of the candidate list, where the hard
negatives sit. Default hyperparameters: `hidden_channels=256`, `num_layers=2`, `dropout=0.0`,
`lr=0.01`, `epochs=200`, early-stop `patience=20`, `eval_every=10`.
