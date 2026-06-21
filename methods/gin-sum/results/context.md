# Context: representational power of neighborhood-aggregation GNNs (circa 2017-2018)

## Research question

We want to learn fixed-size vector representations of graphs — molecules, social ego-networks,
protein structures — that are good enough for graph classification. The dominant recipe is a
neighborhood-aggregation (message-passing) network: each node repeatedly updates its feature
vector by combining its own vector with an aggregate of its neighbors' vectors, and after some
number of rounds a permutation-invariant **readout** collapses all node vectors into one
graph-level vector. Dozens of variants exist with different neighbor-aggregators (mean, max,
weighted, LSTM) and different readouts (sum, sort, hierarchical pooling), each chosen by intuition
and trial-and-error.

The question we want to study: how expressive is an aggregation-based GNN — which non-isomorphic
graphs can it map to different embeddings, and which pairs can it not tell apart? Can we
characterize that ceiling, and on that basis design a network in this family — both the per-layer
aggregation/update and the **graph-level readout** that turns the multiset of node vectors into
the graph vector — that is permutation-invariant, handles variable-size graphs, and is end-to-end
learnable?

## Background

By this time, neighborhood-aggregation GNNs are the standard tool for graph representation
learning (Kipf & Welling 2017; Hamilton, Ying & Leskovec 2017; Gilmer et al. 2017; Velickovic
et al. 2018). The general form of the k-th layer is

```
a_v^(k) = AGGREGATE^(k)( { h_u^(k-1) : u in N(v) } )
h_v^(k) = COMBINE^(k)( h_v^(k-1), a_v^(k) )
```

initialized with h_v^(0) = X_v, where N(v) is the set of neighbors of v. After K rounds, a
node's vector summarizes the structure and features in its K-hop neighborhood — concretely, the
rooted subtree of height K hanging off that node. For graph classification, a readout collapses
the final node vectors,

```
h_G = READOUT( { h_v^(K) : v in G } ),
```

with READOUT a permutation-invariant function (summation, or a more elaborate pooling).

The load-bearing concepts the field already has on the table:

- **The Weisfeiler-Lehman (1-WL) test of graph isomorphism** (Weisfeiler & Lehman 1968). Its
  one-dimensional "naive vertex refinement" form is strikingly parallel to neighborhood
  aggregation: it iteratively relabels each node by applying a hash to the pair (own current
  label, the *multiset* of neighbors' current labels), where the hash assigns distinct new labels
  to distinct (label, neighbor-multiset) inputs, then declares two graphs non-isomorphic the first
  time their collections of node labels differ. It is known to distinguish a very broad class of
  graphs (Babai & Kucera 1979), with known failures only on some highly symmetric / regular
  families (Cai, Furer & Immerman 1992).

- **The WL subtree kernel** (Shervashidze et al. 2011). The label a node receives at iteration k
  of the WL test is, intuitively, an identifier for the rooted subtree of height k at that node;
  the kernel represents a graph by the *counts* of these subtree identifiers across iterations.
  It is a strong graph-classification baseline whose labels are one-hot identifiers, and the
  representation is fixed rather than learned jointly with the task.

- **The multiset.** A node's neighbors' feature vectors form a *multiset* (a set with
  multiplicities), because different neighbors can share the same vector. So a neighbor
  aggregator is precisely a function on multisets, and how expressive the aggregator is comes down
  to which distinct multisets it maps to distinct outputs.

- **Sum-decomposition of permutation-invariant set functions** (Zaheer et al. 2017, "Deep
  Sets"). For inputs drawn from a *countable* universe, every permutation-invariant function of a
  set decomposes exactly as f(X) = rho( sum_{x in X} phi(x) ) for some element-map phi and outer
  map rho. This is the structural template for building permutation-invariant networks on sets.
  The result is stated for sets; the inputs of interest here are multisets, where an element can
  recur with a multiplicity.

- **Universal approximation of MLPs** (Hornik, Stinchcombe & White 1989; Hornik 1991). A
  multilayer perceptron with a nonlinearity can approximate any continuous function on a compact
  set, whereas a one-layer perceptron (a linear map followed by a single nonlinearity) is a
  generalized linear model and is far more restricted. This is the tool one would reach for to
  *realize* an abstract element-map or outer-map by a learnable network.

- **Depth, receptive field, and the "washing out" of local information** (Xu et al. 2018,
  "Jumping Knowledge Networks"). The effective receptive field of a node after k aggregation
  rounds behaves like the spread of a k-step random walk, whose growth depends sharply on local
  graph structure: in expander-like (densely connected) regions it spreads to almost the whole
  graph within O(log |V|) steps, so deep node vectors there are dominated by the global graph,
  while in tree-like regions it stays local. A consequence already observed in the field is that
  the best results for some aggregation models (e.g. GCN) come at shallow depth (around two
  layers). Node vectors at *different* depths thus capture different scales of structure — early
  layers more local, later layers more global.

An observation that motivates the study below: on graph-classification benchmarks where structure
(not node features) carries the signal, aggregation-based GNNs built by intuition often do not fit
the training set well, and the achieved training accuracy differs systematically with the
aggregator used.

## Baselines

These are the prior aggregation-and-readout schemes a new design would be measured against.

**GCN, mean-aggregation (Kipf & Welling 2017).** Integrate aggregation and update into one
mean-then-transform step over the node and its neighbors,

```
h_v^(k) = ReLU( W * MEAN{ h_u^(k-1) : u in N(v) ∪ {v} } ).
```

Mean is permutation-invariant and works well where node features are rich and the neighborhood's
feature *distribution* is the signal (it is effective for node classification on citation
graphs). It captures the proportion of element types in a neighborhood.

**GraphSAGE, max-pooling (Hamilton, Ying & Leskovec 2017).** Transform each neighbor, then take
an element-wise maximum,

```
a_v^(k) = MAX( { ReLU( W * h_u^(k-1) ) : u in N(v) } ),
```

with COMBINE a concatenation-and-linear of the node vector and a_v. Max is robust and good at
picking out salient or "skeleton" elements (it identifies representative points in a point cloud
and is robust to outliers).

**One-layer-perceptron aggregators (used inside GCN, DCNN, DGCNN).** Many GNNs apply a single
linear map plus nonlinearity, sigma(W·), as the per-layer transform rather than a full MLP. This
is a generalized linear model: cheaper to fit and with fewer parameters than a multi-layer map.

**Sort / hierarchical pooling readouts (SortPooling — Zhang et al. 2018; DiffPool — Ying et al.
2018).** Beyond simple sum/mean/max readouts, these sort nodes by structural role (via WL
colors, then a 1-D convolution) or learn soft cluster assignments and coarsen the graph
hierarchically. They add learnable capacity and parameters to the graph-level pooling stage.

**The WL subtree kernel (Shervashidze et al. 2011).** A strong, theory-backed classifier whose
features are counts of WL subtree labels. Its features are one-hot subtree identifiers, computed
in a fixed pipeline rather than learned jointly with the task.

## Evaluation settings

The natural yardsticks already in use for graph classification:

- **Bioinformatics graphs with categorical node features**: MUTAG (188 mutagenic-compound
  graphs, 2 classes), PROTEINS (1113 graphs of secondary-structure elements, 2 classes), NCI1
  (4110 chemical-compound graphs, 2 classes), PTC (344 compounds). Here the node has a discrete
  chemical/structural label.
- **Social-network graphs** (IMDB-BINARY/MULTI, REDDIT-BINARY/MULTI5K, COLLAB), which come with
  *no* node features. The standard preprocessing makes structure the only usable signal: for
  REDDIT, set every node feature to a single shared constant (so features are uninformative); for
  the others, use a one-hot encoding of node degree.
- **Protocol**: 5 GNN layers including the input layer; 2-layer MLPs inside each layer; batch
  normalization on every hidden layer; the Adam optimizer with initial learning rate decayed over
  training; 10-fold cross-validation; classification accuracy as the metric. A complementary
  diagnostic is to compare *training* accuracy across aggregators (with all hyper-parameters
  fixed), as a proxy for the model's representational capacity. As a reference point, the WL
  subtree kernel's training accuracy is reported with its number of iterations matched to the GNN
  depth.
- A readout-focused diagnostic can hold the message-passing stack, hidden dimension, optimizer,
  folds, and datasets fixed, then swap only the permutation-invariant graph readout, isolating the
  effect of the readout choice from that of the per-layer aggregator.

## Code framework

The graph-classification harness has two separable pieces: a message-passing backbone that returns
node embeddings from each layer, and a graph readout that consumes the resulting node-vector
multisets. The existing substrate exposes batched node features, edge indices, a `batch` vector
assigning nodes to graphs, standard permutation-invariant pooling primitives, and a small MLP
classifier head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class NeighborhoodBackbone(nn.Module):
    """Produce final node embeddings and one node-embedding matrix per layer.

    Inputs:
        x          [N_total, input_dim]   batched node features
        edge_index [2, E_total]           batched edge index
        batch      [N_total]              graph id of each node (0..B-1)
    Output:
        node_emb      [N_total, hidden_dim]
        layer_outputs list of [N_total, hidden_dim], one per message-passing layer
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # TODO: choose the neighborhood aggregation/update rule for each layer.

    def forward(self, x, edge_index, batch):
        layer_outputs = []
        # TODO: iteratively update x and append each layer's node embeddings.
        return x, layer_outputs


class GraphReadout(nn.Module):
    """Collapse per-layer node embeddings into one fixed-size graph-level vector.

    Must be permutation-invariant over nodes and handle variable graph sizes within a
    batch. Set self.output_dim in __init__ so the downstream classifier knows the width.

    Inputs to forward():
        x             [N_total, hidden_dim]  final-layer node embeddings (batched)
        edge_index    [2, E_total]           batched edge index
        batch         [N_total]              graph id of each node (0..B-1)
        layer_outputs list of [N_total, hidden_dim], one per backbone layer
    Output:
        [B, output_dim]  one vector per graph
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # TODO: set the output width once the invariant readout is chosen.
        self.output_dim = hidden_dim
        # TODO: any parameters/sub-modules the readout needs.

    def forward(self, x, edge_index, batch, layer_outputs):
        # TODO: turn the per-layer multisets of node embeddings into one graph vector.
        pass


class GraphClassifier(nn.Module):
    """Message-passing backbone -> GraphReadout -> MLP classifier head."""

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout=0.5):
        super().__init__()
        self.backbone = NeighborhoodBackbone(input_dim, hidden_dim, num_layers)
        self.readout = GraphReadout(hidden_dim, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.readout.output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb, layer_outputs = self.backbone(x, edge_index, batch)
        graph_emb = self.readout(node_emb, edge_index, batch, layer_outputs)
        return self.classifier(graph_emb)
```

The backbone returns the node embeddings that the readout consumes; the classifier head only needs
the graph-vector width reported by the readout.
