The problem is to turn a stack of node embeddings produced by a neighborhood-aggregation GNN into a single fixed-size vector that can be used for graph classification. The difficulty is not just pooling variable-sized sets; it is preserving enough structural information so that non-isomorphic graphs can be mapped to different embeddings. Earlier aggregation-based GNNs were built mostly by intuition, and when they underfit on structure-only tasks the cause is often representational rather than optimization. The key insight is that a node's neighbors form a multiset, and the message-passing layer is a function on that multiset. If the aggregation step collapses distinct multisets, the network loses information that can never be recovered downstream, no matter how deep or wide it becomes.

Existing choices fall short for exactly this reason. Mean aggregation keeps only the distribution of feature values, so a neighborhood with two copies of a feature is treated the same as a neighborhood with three. When every node shares the same feature, mean produces identical vectors everywhere and the network cannot see degree or structure at all. Max aggregation is even coarser: it reduces a multiset to its underlying set and discards multiplicity entirely, so green-red is indistinguishable from green-red-red. One-layer perceptron updates behave like linear maps and fold together distinct multisets that happen to have the same element-wise sum, such as five copies of one vector versus a pair whose sum matches. Hierarchical pooling schemes like DiffPool can help, but a stripped version that learns a soft assignment without structural priors or auxiliary losses often diffuses toward a noisy mean, again throwing away counts. The Weisfeiler-Lehman subtree kernel provides a strong non-learned baseline, yet its one-hot subtree identifiers cannot express similarity between related substructures and are not trained end-to-end.

The method I propose is the Graph Isomorphism Network with a sum-based Jumping Knowledge readout, abbreviated GIN-sum. It is the maximally expressive message-passing architecture in the sense that it is provably as discriminative as the one-dimensional Weisfeiler-Lehman graph-isomorphism test, which is the ceiling for any neighborhood-aggregation GNN. It reaches that ceiling by making three consecutive stages injective on multisets: the neighbor aggregator, the node update function, and the final graph-level readout. If any one of these stages collapses two different multisets into the same vector, expressive power is lost at that bottleneck.

At each layer the node update is MLP((1 + epsilon) * h_v + sum of h_u over neighbors u). The sum is the only standard permutation-invariant aggregator that can be injective over multisets: with a suitable element map, summing assigns a unique code to each possible multiset, preserving exact counts. The (1 + epsilon) weight on the center node keeps the root distinct from its surrounding multiset; it is injective for almost every choice of epsilon, including all irrational values. The MLP is essential because a single linear layer followed by a nonlinearity is not a universal approximator of multiset functions and can collapse different multisets with equal sums. The first layer can omit the pre-sum MLP when inputs are one-hot because summing one-hot vectors is already injective.

The graph-level readout must also be injective on the multiset of final node vectors, so it uses sum pooling rather than mean or max. Depth is another source of tension: later layers see larger receptive fields and can distinguish distant structures, but in densely connected regions deep embeddings get washed out, while earlier layers retain more local information. GIN-sum therefore reads every layer, not just the last one. Each layer's node embeddings are sum-pooled into a graph-level vector and the resulting vectors are concatenated side by side, a form of Jumping Knowledge. Concatenation is lossless and low-parameter, which is well suited to the small graph-classification benchmarks, and the downstream classifier learns how to weight each scale. A per-layer batch normalization on the pooled vectors can be added to keep different layers on a common optimization scale without affecting injectivity, since it is invertible.

Together these choices give a network that trains end-to-end, represents similarity in a learned embedding space, and matches the expressive power of the 1-WL test. Below is a compact PyTorch Geometric implementation that bundles the GIN backbone with the sum Jumping Knowledge readout and a small classifier head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool


class GINBackbone(nn.Module):
    """Sum neighbors, add the weighted center node, and update with an MLP."""

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
            self.convs.append(
                GINConv(update, eps=0.0, train_eps=train_eps, aggr="add")
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(norm(x))
            layer_outputs.append(x)
        return x, layer_outputs


class SumJKReadout(nn.Module):
    """Sum-pool each layer's node embeddings, then concatenate the graph vectors."""

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim * num_layers
        self.graph_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.jk = JumpingKnowledge(mode="cat")

    def forward(self, x, edge_index, batch, layer_outputs):
        pooled = [
            self.graph_bns[i](global_add_pool(h, batch))
            for i, h in enumerate(layer_outputs)
        ]
        return self.jk(pooled)


class GraphClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_layers,
        dropout=0.5,
        train_eps=False,
    ):
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
