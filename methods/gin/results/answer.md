# Graph Isomorphism Network (GIN)

## Problem

Message-passing / neighborhood-aggregation graph neural networks (GCN, GraphSAGE, and many others) are designed heuristically, with no theory of their *representational power*: which non-isomorphic graphs can such a network ever map to different embeddings, and which design choices (aggregator, per-layer transform, readout) actually determine that power. GIN answers this by tying the power of message-passing GNNs to the Weisfeiler-Lehman (WL) graph isomorphism test and building the maximally expressive network in this family.

## Key results

1. **Upper bound.** Any neighborhood-aggregation GNN is *at most* as powerful as the 1-dimensional WL test at distinguishing non-isomorphic graphs. (If the GNN separates two graphs, WL separates them too.)

2. **When the bound is met.** A GNN is *as powerful as* WL if its neighbor-aggregation function, its combine function, and its graph-level readout are all **injective on multisets**. Injectivity on a (countable) feature space is exactly the property WL's hash has and lossy aggregators lack; it propagates across layers because bounded multisets over a countable set form a countable set.

3. **Sum gives an injective multiset aggregator.** Over bounded multisets, `sum` can be made injective (in fact universal: any multiset function is `φ(Σ_x f(x))` for some `f, φ`), whereas:
   - **mean** keeps only the *distribution* — it cannot distinguish a multiset `(S, m)` from any inflated copy `(S, k·m)`;
   - **max** keeps only the *underlying set* (support) — it ignores all multiplicities.
   Concretely, for neighbor multisets: max confuses `{g,r}` with `{g,r,r}`; mean and max both confuse `{g,r}` with `{g,g,r,r}`; sum separates both. Power ranking: **sum ⊐ mean ⊐ max**.

4. **MLP, not a single linear layer.** A 1-layer perceptron `σ∘W` (no bias) collapses to a linear function of the neighbor sum (e.g. it cannot separate `{1,1,1,1,1}` from `{2,3}`, since `Σ ReLU(Wx) = ReLU(W Σx)` for positive inputs), so it is not a universal approximator of multiset functions. A real MLP (≥2 layers) is needed to realize the injective `f`.

5. **Tag the center node.** Merging center and neighbors into one flat multiset loses the root (e.g. middle of `a–b–b` vs `b–a–b` both become `{a,b,b}`). Weighting the center by `(1+ε)` keeps `h(c,X) = (1+ε)·f(c) + Σ_{x∈X} f(x)` injective over (center, multiset) pairs for any irrational `ε`. `ε` may be **learned** (GIN-ε) or **fixed to 0** (GIN-0, implemented by pooling self-loops with the neighbors).

## Final model

Per-layer update:

```
h_v^{(k)} = MLP^{(k)} ( (1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)} )
```

Graph readout (sum-pool each layer, concatenate across all depths — jumping-knowledge style; provably generalizes WL and the WL subtree kernel):

```
h_G = CONCAT( SUM({ h_v^{(k)} : v ∈ G }) : k = 0, 1, ..., K )
```

One MLP per layer models `f^{(k+1)}∘φ^{(k)}` (composition). With one-hot inputs, no MLP is needed before the first summation. Typical recipe: 5 layers (including input), 2-layer MLPs, BatchNorm on hidden layers, Adam, dropout on the final layer, sum or average graph pooling.

## Code

The implementation uses block-diagonal batched adjacency, neighbor pooling inside the model, optional `(1+ε)` center reweighting, and per-layer pooled scores whose sum realizes the all-depth readout.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """The transform that plays the role of f / phi.
    num_layers == 1 is the linear ablation; num_layers >= 2 is the MLP case.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_layers = num_layers
        self.linear_or_not = True

        if num_layers < 1:
            raise ValueError("number of layers should be positive")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, center_weighting,
                 graph_pooling_type, neighbor_pooling_type, device):
        super().__init__()
        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.center_weighting = center_weighting
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.device = device

        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers - 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, in_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(in_dim, output_dim))

    def preprocess_neighbors_for_maxpool(self, batch_graph):
        max_deg = max(graph.max_neighbor for graph in batch_graph)
        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            for j, neighbors in enumerate(graph.neighbors):
                pad = [n + start_idx[i] for n in neighbors]
                pad.extend([-1] * (max_deg - len(pad)))
                if not self.center_weighting:
                    pad.append(j + start_idx[i])
                padded_neighbor_list.append(pad)

        return torch.LongTensor(padded_neighbor_list).to(self.device)

    def preprocess_neighbors_for_matrix_pool(self, batch_graph):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])

        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        if not self.center_weighting:
            num_node = start_idx[-1]
            self_loop = torch.arange(num_node, dtype=torch.long)
            self_loop_edge = torch.stack([self_loop, self_loop])
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, torch.ones(num_node)], 0)

        Adj_block = torch.sparse.FloatTensor(
            Adj_block_idx, Adj_block_elem,
            torch.Size([start_idx[-1], start_idx[-1]]))
        return Adj_block.to(self.device)

    def preprocess_graph_pool(self, batch_graph):
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx, elem = [], []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1.0 / len(graph.g)] * len(graph.g))
            else:
                elem.extend([1.0] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1])])

        idx = torch.LongTensor(idx).transpose(0, 1)
        elem = torch.FloatTensor(elem)
        graph_pool = torch.sparse.FloatTensor(
            idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape(1, -1).to(self.device)])
        return torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]

    def next_layer_with_center_weighting(self, h, layer,
                                         padded_neighbor_list=None, Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(
                    Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        return F.relu(self.batch_norms[layer](pooled_rep))

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(
                    Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled_rep = self.mlps[layer](pooled)
        return F.relu(self.batch_norms[layer](pooled_rep))

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.preprocess_graph_pool(batch_graph)

        padded_neighbor_list, Adj_block = None, None
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.preprocess_neighbors_for_maxpool(batch_graph)
        else:
            Adj_block = self.preprocess_neighbors_for_matrix_pool(batch_graph)

        hidden_rep = [X_concat]
        h = X_concat
        for layer in range(self.num_layers - 1):
            if self.center_weighting:
                h = self.next_layer_with_center_weighting(
                    h, layer, padded_neighbor_list, Adj_block)
            else:
                h = self.next_layer(h, layer, padded_neighbor_list, Adj_block)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.final_dropout, training=self.training)
        return score_over_layer
```

In the maximally expressive setting, `neighbor_pooling_type="sum"`, `graph_pooling_type="sum"`, and `center_weighting=True` learn the `(1+ε)` center term. With `center_weighting=False`, the preprocessing adds self-loops, giving the fixed-`ε=0` variant. The same implementation keeps `average` and `max` neighbor pooling as controlled ablations.
