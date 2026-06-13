# Graph Isomorphism Network (GIN), distilled

GIN is the maximally expressive message-passing graph neural network under the
neighborhood-aggregation framework: it is provably as discriminative as the Weisfeiler-Lehman
(1-WL) graph isomorphism test, which is the upper bound for this entire family. It gets there by
making every stage of the network an **injective function of a multiset** — neighbor aggregation,
node update, and graph-level readout — using **sum** aggregation, an **MLP** update with a
**(1+epsilon)** center weighting, and a graph readout that **sum-pools each layer and concatenates
the layerwise graph vectors** (Jumping Knowledge). The `gin_sum` graph-classification baseline is
this readout: `global_add_pool` per layer, then concatenation across layers.

## Problem it solves

Message-passing GNNs (GCN, GraphSAGE, and many others) were designed by intuition with no theory
of their representational power: which non-isomorphic graphs can such a network map to different
embeddings, and which aggregator / update / readout choices decide that. GIN answers it by tying
the power of aggregation-based GNNs to the 1-WL test and constructing the maximally expressive
network in the family.

## Key idea

A node's neighbors' feature vectors form a *multiset*. A GNN reaches the WL ceiling exactly when
its per-step neighbor aggregation, its per-step update, and its graph-level readout are each
**injective on multisets** — three injective bottlenecks in series.

- **Upper bound.** Any aggregation-based GNN is *at most* as powerful as 1-WL: if 1-WL cannot
  separate two graphs, no such GNN can either (same functions applied everywhere + a
  permutation-invariant readout collapse identical node-vector multisets).
- **Reaching the ceiling.** If the neighbor aggregation f and update phi are injective and the
  readout is injective, an inductive argument shows the GNN node vectors are an injective image
  of the WL labels at every depth, so the GNN separates exactly what 1-WL separates.
- **Sum is the only standard injective multiset aggregator.** With a place-value element map, the
  sum is a *universal* multiset representation; **mean** keeps only the distribution (it maps a
  multiset and any k-times replication of it to the same vector), and **max** keeps only the
  underlying set. So sum strictly dominates and is forced.
- **(1+epsilon) center weighting** keeps the root node distinct from its neighbor multiset; it is
  injective over (center, neighbor-multiset) for all but countably many epsilon (every irrational
  works).
- **MLP, not one-layer perceptron**, because a one-layer perceptron behaves like a linear map and
  folds distinct equal-sum multisets together (e.g. {1,1,1,1,1} and {2,3}); an MLP is a universal
  approximator of the required maps.
- **Readout = sum per layer, then concatenate layers.** Sum at the node level is the injective readout the theory
  requires (mean/max would discard structure at the finish line). Combine across *all* layers
  (Jumping Knowledge) because depth is needed for discriminative power but the deepest layer's
  vectors get washed out in densely connected regions, while earlier, more local layers
  generalize better; concatenation is the lossless, low-parameter combine that suits small graphs.

## Final architecture

GIN layer (matches 1-WL):

```
h_v^(k) = MLP^(k)( (1 + epsilon^(k)) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1) )
```

- epsilon can be learnable (GIN-eps) or fixed to 0 (GIN-0). The irrational-epsilon construction
  gives the clean injectivity guarantee; the fixed-zero variant is simpler and slightly less
  expressive in contrived cases. First layer needs no pre-sum MLP if inputs are one-hot (summing
  one-hot vectors is already injective).

Graph-level readout (sum plus Jumping-Knowledge concat):

```
h_G = CONCAT( SUM({ h_v^(k) : v in G }) | k = 0, 1, ..., K )
```

## Why each choice

| Choice | Reason | Rejected alternative and its failure |
|---|---|---|
| Sum neighbor aggregation | only standard injective-on-multiset op; universal multiset rep | mean loses multiplicity; max loses everything but the set |
| MLP update | universal approximator of injective f, phi | 1-layer perceptron ≈ linear; folds equal-sum multisets |
| (1+epsilon) center | keep root distinct from neighbors; collision-avoiding epsilon is injective on (c, X) | merging center into neighbor sum conflates root vs neighbor |
| epsilon = 0 option (GIN-0) | fewer parameters and a simple center-plus-neighbor sum | a collision-avoiding epsilon is needed for the full guarantee; learnable epsilon exposes that degree of freedom |
| Sum readout | injective on the node-vector multiset (last bottleneck) | mean/max readout discards structure at the finish line |
| Concat over all layers | lossless multi-scale combine, few params, suits small graphs | last-layer-only loses local scale; max/LSTM select & overfit |
| output_dim = hidden_dim * num_layers | no bottleneck projection | projecting re-collapses what sum kept injective |

## Theory (why GIN = 1-WL)

- **Lemma (upper bound).** For non-isomorphic G1, G2, if a GNN maps them to different embeddings,
  1-WL also decides them non-isomorphic. Proof: induction shows that on any graph, equal WL labels
  imply equal GNN vectors (same AGGREGATE/COMBINE on identical inputs), so the GNN's node-vector
  multiset is a function of the WL-label multiset; a permutation-invariant readout then collapses
  any pair WL cannot separate.
- **Theorem (conditions for maximality).** With enough layers, a GNN separates every pair 1-WL
  separates if (a) it updates h_v^(k) = phi(h_v^(k-1), f({h_u^(k-1)})) with f and phi injective,
  and (b) its graph-level readout is injective. Proof: induction gives an injective varphi with
  h_v^(k) = varphi(l_v^(k)) at every k (compose injective maps; varphi = psi ∘ g^{-1} with g the
  injective WL hash), so differing WL label multisets give differing GNN node-vector multisets,
  which an injective readout maps apart.
- **Countability propagates (Lemma).** From countable input features, every layer's range is
  countable: finite products of countable sets are countable via (m,n) -> 2^(m-1)(2n-1); bounded
  multisets inject into a finite product; and the image of a countable multiset domain under a
  layer function is countable. Thus injectivity is well-defined at every depth.
- **Deep multisets (Lemma).** For countable X there is f with h(X) = sum_x f(x) injective over
  bounded multisets (take f(x) = N^{-Z(x)}, N above the max size, Z an enumeration: a no-carry
  N-ary place-value count), and any multiset function g = phi(sum_x f(x)).
- **Center injectivity (Corollary).** h(c, X) = (1+epsilon) f(c) + sum_x f(x) is injective over
  (c, X) for all irrational epsilon: a collision with c' ≠ c forces epsilon*(f(c) - f(c'))
  (irrational) to equal a sum of rationals — contradiction.
- **Aggregator ranking (Corollaries).** mean captures normalized multiplicity distributions and
  collapses uniform replications such as (S, m) and (S, k*m); max captures only the underlying set;
  one-layer-perceptron sum folds {1,1,1,1,1} and {2,3} (ReLU homogeneity + equal element-sum).

## Working code

The backbone uses PyG `GINConv` with explicit add aggregation. The readout sum-pools each layer's
node embeddings with `global_add_pool`, then applies Jumping-Knowledge concatenation:

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool


class GINBackbone(nn.Module):
    """h_v = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)."""

    def __init__(self, input_dim, hidden_dim, num_layers, train_eps=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            update = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(update, eps=0.0, train_eps=train_eps, aggr="add"))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(norm(x))
            layer_outputs.append(x)
        return x, layer_outputs


class SumJKReadout(nn.Module):
    """Sum-pool each layer, then concatenate the graph vectors."""

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.output_dim = hidden_dim * num_layers
        self.jk = JumpingKnowledge(mode="cat")

    def forward(self, x, edge_index, batch, layer_outputs):
        pooled = [global_add_pool(h, batch) for h in layer_outputs]
        return self.jk(pooled)


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout=0.5,
                 train_eps=False):
        super().__init__()
        self.backbone = GINBackbone(input_dim, hidden_dim, num_layers, train_eps)
        self.readout = SumJKReadout(hidden_dim, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.readout.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb, layer_outputs = self.backbone(x, edge_index, batch)
        graph_emb = self.readout(node_emb, edge_index, batch, layer_outputs)
        return self.classifier(graph_emb)
```

Implementation notes: the original graph-classification implementation pools every layer over
nodes and combines the layer contributions in the prediction head. PyG's `GINConv` computes
`nn((1 + eps) * x + sum_neighbors x)` with `aggr="add"`, while `global_add_pool` is the
batch-wise node sum used by the layerwise readout above.
