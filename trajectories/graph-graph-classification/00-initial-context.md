## Research question

A message-passing GNN gives me one embedding per node, but graph classification wants one label per
*graph* — a molecule's mutagenicity, a protein's enzyme class, a compound's activity. So there is a
last mile that the backbone does not solve: collapse a variable-size bag of node vectors into a single
fixed-size graph vector that a classifier can eat. The single thing being designed here is the
**graph-level readout (pooling)** — the function that performs that collapse. Everything else is fixed:
the backbone that produces the node embeddings, the classifier head that consumes the graph vector, the
data pipeline, and the training loop. Whatever I build must take a batch of differently-sized graphs and
return one vector per graph, and it must be invariant to how the nodes happened to be numbered (a graph
label cannot depend on node ordering).

## Prior art before the first rung (readout lineage)

The first rung — a plain global readout over the final-layer node embeddings — is the resolution of a
short line of "how do you turn node vectors into a graph vector" ideas. These are the baselines the
ladder reacts to; the substrate below is what they share.

- **Global mean / max pooling (Duvenaud et al. 2015; Hamilton et al. 2017).** Average or take the
  elementwise max over all final node embeddings. Permutation-invariant by being a symmetric reduction,
  and trivially cheap. Gap: a mean throws away *counts* — it sees only the distribution of neighbor
  features, so a graph and an inflated copy of it collapse together — and a max sees only the *support*,
  which distinct elements are present, ignoring both counts and proportions. Neither is injective on the
  multiset of node features, so structurally different graphs can land on the same vector.
- **Global sum pooling (Xu, Hu, Leskovec & Jegelka 2019).** Sum the final node embeddings instead of
  averaging. The sum of learned features is *injective* on bounded multisets (it positionally encodes
  the multiplicity profile), so among the basic permutation-invariant reductions sum is the most
  expressive — it keeps the counts that mean discards and the multiplicities max discards. Gap: it is
  still a single *flat* reduction over the last layer only. It represents no scale between "one node's
  K-hop neighborhood" and "the whole graph," and it reads from one depth, discarding the more-local,
  often-better-generalizing representations the earlier layers hold.
- **Set-aggregation readouts (Vinyals et al. 2016, Set2Set; Gilmer et al. 2017).** Run a
  permutation-invariant set network (LSTM-attention over the node set) to produce the graph vector — more
  capacity than a plain sum. Gap: still one flat operation over all nodes at once, no notion of
  intermediate structure, and a heavier, order-sensitive recurrent module to fit on small datasets.

There is room to do better by reading from every depth (jumping knowledge), by attending over nodes, by
coarsening the graph hierarchically, or by combining several complementary aggregators — each is a
different fill of the one editable slot below.

## The fixed substrate

The backbone and the loop are frozen and must not be touched. The backbone is a **5-layer GIN**
(`hidden_dim=64`, `train_eps=True`) — each layer is `GINConv(MLP)` followed by `BatchNorm1d` and `ReLU`,
producing per-node embeddings at every layer. The full model is `GINBackbone → GraphReadout →
MLP classifier`, where the classifier is `Linear(readout.output_dim, 64) → ReLU → Dropout(0.5) →
Linear(64, 32) → ReLU → Dropout(0.5) → Linear(32, num_classes)`. Datasets are loaded as PyG
`TUDataset`s; graphs without node features get one-hot degree encodings. Training is Adam (`lr=0.01`,
no weight decay) with cosine annealing over 350 epochs per fold; epoch selection is by a stratified 10%
validation split carved from each training fold (the test set is only read when validation accuracy
improves). There is a parameter budget: total model parameters must stay within `1.05×` the fixed
backbone/classifier plus the larger of the GMT or Set2Set readout sizes (≈ `10·H² + 9·H` extra), so the
readout cannot be arbitrarily large.

## The editable interface

Exactly one region is editable — the `GraphReadout` class in `custom_graph_cls.py` (the lines between
the EDITABLE markers). Every method on the ladder is a fill of this same contract. The constructor takes
`(hidden_dim, num_layers)` and must set `self.output_dim` (the width the classifier head will expect).
`forward` receives:

- `x` — final-layer node embeddings, `[N_total, hidden_dim]` (the whole batch stacked).
- `edge_index` — `[2, E_total]`, the batched edges.
- `batch` — `[N_total]`, the graph-id of each node (this is what makes a pooling op operate
  per-graph within the batch).
- `layer_outputs` — a `list` of `[N_total, hidden_dim]` tensors, the per-layer node embeddings,
  `len == num_layers` (this is what a jumping-knowledge readout reads from).

It returns `[B, output_dim]`, one row per graph. The provided imports are `torch`, `torch.nn`,
`torch.nn.functional as F`, `torch_geometric.nn` (notably `global_add_pool`, `global_mean_pool`), and
`torch_geometric.utils` (notably `to_dense_batch`, `to_dense_adj`, `degree`). The starting point is the
scaffold default: **plain global sum pooling on the last layer**. Each later method replaces exactly this
class and nothing else.

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

Three TU graph-classification collections spanning the size/difficulty range: **MUTAG** (188 graphs, 2
classes, small molecular mutagenicity — high variance because tiny), **PROTEINS** (1113 graphs, 2
classes, protein enzyme classification — medium, larger graphs), and **NCI1** (4110 graphs, 2 classes,
chemical-compound activity — the largest and the held-out hard one). Each is evaluated under **10-fold
stratified cross-validation** with the fixed GIN backbone and 350 epochs/fold. Two metrics, **test
accuracy** and **macro F1**, both higher-is-better, reported as the cross-validation mean (and per-seed
over the leaderboard's seeds {42, 123, 456}). A useful readout must handle batches of variable-size
graphs, stay permutation-invariant at the graph level, and generalize from the tiny molecular graphs of
MUTAG to the larger, more numerous graphs of PROTEINS and NCI1.
