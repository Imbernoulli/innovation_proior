## Research question

A message-passing GNN produces one embedding per node, but graph classification needs one label per *graph*. The remaining design question is the **graph-level readout (pooling)**: a permutation-invariant function that collapses a variable-size bag of node vectors into a single fixed-size graph vector for a classifier. The backbone that produces the node embeddings, the classifier head, the data pipeline, and the training loop are fixed.

## Prior art / Background / Baselines

The starting readouts are flat global reductions over the final-layer node embeddings.

- **Global mean / max pooling.** Reduce the final node embeddings elementwise by mean or max across each graph.
- **Global sum pooling.** Sum the final node embeddings across each graph.
- **Set-aggregation readouts (e.g., Set2Set).** Run a permutation-invariant recurrent or attention-based set network over all final node embeddings.

These methods share the same backbone, classifier, and training setup and mark the readout slot that is open for modification.

## Fixed substrate / Code framework

The backbone and training loop are frozen. The backbone is a **5-layer GIN** (`hidden_dim=64`, `train_eps=True`), each layer `GINConv(MLP) → BatchNorm1d → ReLU`, producing per-node embeddings at every layer. The full model is `GINBackbone → GraphReadout → MLP classifier`, where the classifier is `Linear(readout.output_dim, 64) → ReLU → Dropout(0.5) → Linear(64, 32) → ReLU → Dropout(0.5) → Linear(32, num_classes)`.

Datasets are loaded as PyG `TUDataset`s; graphs without node features receive one-hot degree encodings. Training uses Adam (`lr=0.01`, no weight decay) with cosine annealing over 350 epochs per fold. Each fold uses a stratified 10% validation split for model selection, and the test set is evaluated only when validation accuracy improves. There is a parameter budget: total model parameters must stay within `1.05×` the fixed backbone/classifier plus the larger of the GMT or Set2Set readout sizes (≈ `10·H² + 9·H` extra parameters).

## Editable interface

Only the `GraphReadout` class in `custom_graph_cls.py` (between the EDITABLE markers) may be changed. Its constructor takes `(hidden_dim, num_layers)` and must set `self.output_dim`. Its `forward` receives:

- `x` — final-layer node embeddings, shape `[N_total, hidden_dim]`.
- `edge_index` — batched edges, shape `[2, E_total]`.
- `batch` — graph-id of each node, shape `[N_total]`.
- `layer_outputs` — list of `[N_total, hidden_dim]` per-layer node embeddings, length `num_layers`.

It returns `[B, output_dim]`, one row per graph. Available imports are `torch`, `torch.nn`, `torch.nn.functional as F`, `torch_geometric.nn` (including `global_add_pool`, `global_mean_pool`), and `torch_geometric.utils` (including `to_dense_batch`, `to_dense_adj`, `degree`).

The default fill is plain global sum pooling on the last layer; every method replaces exactly this class and nothing else.

```python
# EDITABLE region of custom_graph_cls.py (lines 41-81) — default fill (global sum pool)
class GraphReadout(nn.Module):
    """Custom graph-level readout/pooling mechanism.

    Aggregates node-level representations into a single graph-level
    representation for classification. Receives node embeddings from
    a GIN backbone and must produce a fixed-size graph embedding.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Default: simple sum pooling over final-layer node embeddings
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Default: global sum pooling on last-layer embeddings
        return global_add_pool(x, batch)
```

## Evaluation settings

Three TU graph-classification collections are used: **MUTAG** (188 graphs, 2 classes, small molecular mutagenicity — high variance because tiny), **PROTEINS** (1113 graphs, 2 classes, protein enzyme classification — medium), and **NCI1** (4110 graphs, 2 classes, chemical-compound activity — the largest and hardest). Each is evaluated under **10-fold stratified cross-validation** with the fixed GIN backbone and 350 epochs per fold. Metrics are **test accuracy** and **macro F1**, both higher-is-better, reported as the cross-validation mean (and per-seed over seeds {42, 123, 456}). A useful readout must handle batches of variable-size graphs, stay permutation-invariant at the graph level, and generalize from the tiny molecular graphs of MUTAG to the larger, more numerous graphs of NCI1.
