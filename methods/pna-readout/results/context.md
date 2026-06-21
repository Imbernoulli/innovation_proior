# Context: how much can a single neighbor-aggregator retain, and what does it take to retain more

## Research question

A message-passing GNN updates each node by aggregating a *multiset* of its neighbors' feature vectors
and combining the result with the node's own vector. The aggregator is the heart of the layer: it is a
permutation-invariant function that must compress a variable-size multiset of vectors into one vector.
Almost every architecture picks *one* such aggregator — a mean, a sum, a max, an attention-weighted
average — by intuition, and the field's expressivity theory has so far been written for a *countable*
feature universe (discrete node labels), where a single carefully-chosen sum can be made injective on
multisets.

But the vectors an aggregator actually sees in a deep network are not from a countable alphabet: they
are *continuous* hidden representations in ℝ^d. In the continuous setting, how much of a neighbor
multiset can a single continuous aggregator retain, and what does a more complete aggregation
operator look like?

## Background

By this point neighborhood-aggregation (message-passing) GNNs are the standard family (Gilmer et al.
2017). One layer has the shape

```
m_{j->i} = M( x_i, e_{j->i}, x_j )                       (message)
x_i'     = U( x_i, AGG_{ j in N(i) } m_{j->i} )          (aggregate + update)
```

with AGG a permutation-invariant reduction over the neighbor multiset. The load-bearing prior results:

- **Expressivity on countable features (Xu, Hu, Leskovec & Jegelka 2019).** When node features come from
  a *countable* universe, a *sum* aggregator with a learned per-element map is injective on bounded
  multisets, which makes a GNN as discriminative as the 1-Weisfeiler-Lehman test — the ceiling for this
  family. This argument is built on countability (a place-value encoding of multiplicities).

- **Deep Sets / sum-decomposition (Zaheer et al. 2017).** Any permutation-invariant function of a set
  from a countable universe factors as ρ(Σ_x φ(x)).

- **The aggregators in use.** Mean keeps the *distribution* of neighbor features but divides out absolute
  counts (k copies of a multiset have the same mean as one copy). Max keeps the *support* — which distinct
  features are present — and discards counts and proportions. Sum keeps the count-weighted total; its
  magnitude scales with the number of neighbors. The standard deviation is itself permutation-invariant
  and captures the *spread* of the neighborhood.

- **Degree and signal scale.** A node's degree varies by orders of magnitude across a graph. A sum
  aggregator's output magnitude grows linearly with degree; a mean is degree-blind. Degree is structurally
  meaningful information in many graph tasks.

- **Borsuk–Ulam (topology).** A classical theorem: any continuous map from the n-sphere to ℝ^n sends some
  antipodal pair to the same point. It is the standard tool for proving that a continuous map *cannot* be
  injective on a space of a given dimension.

## Baselines

The aggregation schemes a new design is measured against and reacts to.

- **Mean aggregation (GCN, Kipf & Welling 2017).** AGG = average of neighbors. Degree-stable and captures
  the neighborhood's feature *distribution*.

- **Sum aggregation (GIN, Xu et al. 2019).** AGG = sum of (learned) neighbor features. Injective on
  *countable* multisets and WL-powerful there.

- **Max-pooling aggregation (GraphSAGE, Hamilton et al. 2017).** AGG = elementwise max. Picks out salient
  neighbors.

- **Attention-weighted mean (GAT, Veličković et al. 2018).** AGG = a learned convex combination of
  neighbors. More flexible than a plain mean; reweights the neighbor distribution.

## Evaluation settings

The natural yardsticks are tasks where the aggregator's information content is directly stressed:

- **Multi-task synthetic graph-theory benchmark.** A battery of node- and graph-level targets that demand
  faithful neighborhood aggregation — single-source shortest paths, node eccentricity, graph
  connectivity, diameter, spectral radius, Laplacian features — on random graphs of varied structure.
  An extra test is *extrapolation* to graphs larger than any seen in training.
- **Real-world molecular/chemical and vision graphs** (e.g. ZINC molecular regression, a molecular
  property classification set, superpixel-graph image classification). Structure-heavy chemical graphs
  and near-regular grid graphs (vision) provide complementary settings.
- **Protocol.** A fixed message-passing backbone with the aggregation operator swapped; a parameter-
  matched comparison (enlarge baselines so any gain is not just more parameters); accuracy / MAE per
  task, and the extrapolation gap to larger graphs.

## Code framework

The harness exposes a message-passing layer whose only open slot is the aggregation of the neighbor
multiset; messages, the per-node update MLP, and the training loop already exist. We write the scaffold
with the aggregation operator left as the missing piece.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class AggregationOperator(nn.Module):
    """Reduce a node's neighbor-message multiset into one vector.

    Inputs to forward():
        messages   [E, F]   one message per edge (already computed by M)
        index      [E]      destination node of each edge (for scatter)
        node_degree[N]      degree of each destination node
        num_nodes  int
    Output:
        [N, out_F]   one aggregated vector per node
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # TODO: choose the aggregation. set out_dim to the post-aggregation width.
        self.out_dim = out_dim

    def forward(self, messages, index, node_degree, num_nodes):
        # TODO: permutation-invariant reduction of the neighbor multiset per node.
        pass


class MPNNLayer(nn.Module):
    """message M -> AggregationOperator -> update U. The aggregator is the open slot."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.msg = nn.Linear(2 * in_dim, hidden_dim)
        self.aggr = AggregationOperator(hidden_dim, hidden_dim)
        self.upd = nn.Sequential(
            nn.Linear(in_dim + self.aggr.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        m = self.msg(torch.cat([x[dst], x[src]], dim=-1))
        deg = degree(dst, num_nodes=x.size(0), dtype=x.dtype)
        a = self.aggr(m, dst, deg, x.size(0))
        return self.upd(torch.cat([x, a], dim=-1))
```

The message and update are fixed; the question is only what reduction fills `AggregationOperator`.
