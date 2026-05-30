# Graph Isomorphism Network (GIN)

## Problem

Message-passing / neighborhood-aggregation graph neural networks (GCN, GraphSAGE, and many others) are designed heuristically, with no theory of their *representational power*: which non-isomorphic graphs can such a network ever map to different embeddings, and which design choices (aggregator, per-layer transform, readout) actually determine that power. GIN answers this by tying the power of message-passing GNNs to the Weisfeiler-Lehman (WL) graph isomorphism test and building the maximally expressive network in this family.

## Key results

1. **Upper bound.** Any neighborhood-aggregation GNN is *at most* as powerful as the 1-dimensional WL test at distinguishing non-isomorphic graphs. (If the GNN separates two graphs, WL separates them too.)

2. **When the bound is met.** A GNN is *as powerful as* WL if its neighbor-aggregation function, its combine function, and its graph-level readout are all **injective on multisets**. Injectivity on a (countable) feature space is exactly the property WL's hash has and lossy aggregators lack; it propagates across layers because bounded multisets over a countable set form a countable set.

3. **Sum is the injective aggregator.** Over multisets, `sum` is injective (in fact universal: any multiset function is `φ(Σ_x f(x))` for some `f, φ`), whereas:
   - **mean** keeps only the *distribution* — it cannot distinguish a multiset `(S, m)` from any inflated copy `(S, k·m)`;
   - **max** keeps only the *underlying set* (support) — it ignores all multiplicities.
   Concretely, for neighbor multisets: max confuses `{g,r}` with `{g,r,r}`; mean and max both confuse `{g,r}` with `{g,g,r,r}`; sum separates both. Power ranking: **sum ⊐ mean ⊐ max**.

4. **MLP, not a single linear layer.** A 1-layer perceptron `σ∘W` (no bias) collapses to a linear function of the neighbor sum (e.g. it cannot separate `{1,1,1,1,1}` from `{2,3}`, since `Σ ReLU(Wx) = ReLU(W Σx)` for positive inputs), so it is not a universal approximator of multiset functions. A real MLP (≥2 layers) is needed to realize the injective `f`.

5. **Tag the center node.** Merging center and neighbors into one flat multiset loses the root (e.g. middle of `a–b–b` vs `b–a–b` both become `{a,b,b}`). Weighting the center by `(1+ε)` keeps `h(c,X) = (1+ε)·f(c) + Σ_{x∈X} f(x)` injective over (center, multiset) pairs for any irrational `ε`. `ε` may be **learned** (GIN-ε) or **fixed to 0** (GIN-0, a self-loop; simpler, generalizes slightly better).

## Final model

Per-layer update:

```
h_v^{(k)} = MLP^{(k)} ( (1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)} )
```

Graph readout (sum-pool each layer, concatenate across all depths — jumping-knowledge style; provably generalizes WL and the WL subtree kernel):

```
h_G = CONCAT( SUM({ h_v^{(k)} : v ∈ G }) : k = 0, 1, ..., K )
```

One MLP per layer models `f^{(k+1)}∘φ^{(k)}` (composition). With one-hot inputs, no MLP is needed before the first summation. Typical recipe: 5 layers (incl. input), 2-layer MLPs, BatchNorm on hidden layers, Adam, dropout on the final layer, sum/mean graph pooling.

## Code

Faithful to the canonical implementation (block-diagonal batched adjacency; neighbor sum as a sparse matmul; `(1+ε)` center reweight; per-layer sum-pool with summed linear heads realizing the concatenate-across-layers readout).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """f / phi. num_layers==1 degrades to a 1-layer perceptron (the ablation);
    >=2 layers is the universal-approximator transform GIN needs."""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.linear_or_not = True
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


class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, device):
        super().__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps            # True: GIN-eps; False: GIN-0
        self.final_dropout = final_dropout
        self.device = device
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers - 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, in_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # per-layer linear head; summing their scores = concat-readout-across-depths
        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(in_dim, output_dim))

    def next_layer(self, h, layer, Adj_block):
        pooled = torch.spmm(Adj_block, h)                 # SUM over neighbors
        if self.learn_eps:
            pooled = pooled + (1 + self.eps[layer]) * h   # (1+eps) center tag
        # if not learn_eps: eps=0 and a self-loop is already in Adj_block
        pooled_rep = self.mlps[layer](pooled)             # MLP transform
        return F.relu(self.batch_norms[layer](pooled_rep))

    def forward(self, batch_graph, Adj_block, graph_pool):
        X_concat = torch.cat([g.node_features for g in batch_graph], 0).to(self.device)
        hidden_rep = [X_concat]
        h = X_concat
        for layer in range(self.num_layers - 1):
            h = self.next_layer(h, layer, Adj_block)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)          # SUM-pool per graph
            score_over_layer += F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.final_dropout, training=self.training)
        return score_over_layer
```

Here `Adj_block` is the batch's block-diagonal adjacency (with self-loops added when `ε` is fixed to 0), `graph_pool` is a sparse (num_graphs × num_nodes) pooling matrix (sum or mean), and each graph object carries `node_features` and neighbor structure built once per minibatch.
