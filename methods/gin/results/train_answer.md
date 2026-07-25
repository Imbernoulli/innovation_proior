Message-passing graph neural networks such as GCN and GraphSAGE have become the standard way to learn from graph-structured data, but their design has been largely empirical. Each layer follows the same template: a node aggregates the feature vectors of its neighbors, combines that aggregate with its own feature, and updates its representation. After $k$ rounds a node's vector is meant to summarize the rooted subtree of height $k$ around it, yet no prior work had pinned down what that summary actually preserves. The open questions are fundamental: which non-isomorphic graphs can such a network ever map to different embeddings, is there a hard upper bound on that power independent of width or training, and do choices like mean versus max versus sum aggregation or a single linear layer versus a deeper network change the representational ceiling rather than just making optimization easier?

Existing aggregators are provably lossy and therefore cap expressive power. Mean pooling preserves only the distribution of neighbor features, so it cannot distinguish a multiset from an inflated copy that has the same proportions; for instance it confuses the neighbor bags green-red and green-green-red-red. Max pooling preserves only the set of distinct elements and ignores multiplicities entirely, so it cannot tell green-red apart from green-red-red. A single linear layer followed by a nonlinearity is not a universal approximator of multiset functions; for positive inputs it degenerates to a linear function of the neighbor sum and can collapse structurally different multisets such as five copies of one feature versus two copies of different features. These are not mere optimization differences; they place a ceiling on what the network can distinguish no matter how it is trained or how wide it is made.

The method I propose is the Graph Isomorphism Network, or GIN. Its theoretical target is the one-dimensional Weisfeiler-Lehman graph isomorphism test, which repeatedly refines each node's color by hashing the pair consisting of its current color and the multiset of its neighbors' colors. I prove that no neighborhood-aggregation network can be more powerful than 1-WL, because whenever WL cannot separate two graphs, any network that uses the same aggregation and combine functions at every node must also produce identical feature multisets and therefore identical graph-level readouts. Conversely, I construct GIN so that it reaches this bound exactly. The key requirement is injectivity on multisets at every step: the neighbor aggregation, the combine function that merges the center node with its neighborhood, and the graph-level readout must all keep distinct inputs distinct.

The layer update is
$$h_v^{(k)} = \mathrm{MLP}^{(k)}\Big((1+\epsilon^{(k)})\cdot h_v^{(k-1)} + \sum_{u\in N(v)} h_u^{(k-1)}\Big).$$
The sum is the only aggregator among the common choices that keeps full multiset information; by summing learned per-element features it can encode exact counts, and any permutation-invariant multiset function can be written as a transformation of that sum. The MLP is essential because a one-layer transform cannot realize the injective encoding needed for arbitrary multisets; it lacks the universality required to separate inputs that happen to have the same total sum, so I build the transform as a genuine multi-layer perceptron and keep the single-linear-layer version around only as the ablation that the theory predicts should underperform. The $(1+\epsilon)$ factor distinguishes the center node from its neighbors; without it, pooling center and neighbors together would lose the root, for example confusing the middle atom of a-b-b with the middle atom of b-a-b, since both would collapse to the flat multiset a-b-b. I can either learn $\epsilon$ by gradient descent, which the injectivity argument requires, or fix $\epsilon=0$ and realize the same sum by adding self-loops to the neighborhood, which is simpler and slightly more forgiving in practice.

For graph-level prediction, I sum-pool the node features produced at every layer, since each per-layer sum is itself injective on the multiset of features at that depth, and I concatenate the per-layer readouts rather than committing to a single depth: shallower layers see smaller, more local rooted subtrees that may generalize better on unseen graphs, while deeper layers see larger subtrees with more discriminating power, and concatenation never merges what any one layer keeps apart. This all-depth readout is the learned analogue of the WL subtree kernel, which counts occurrences of each WL label — a node's height-$k$ feature is a learned embedding of exactly the subtree that the corresponding WL label encodes, so summing those embeddings over the graph is a continuous version of the same histogram, except similar subtrees can now land near each other in the embedding space instead of being forced into disjoint one-hot bins.

The implementation realizes this directly. The transform inside each layer is an MLP class that plays the role of the abstract $f$ and $\phi$ together: with `num_layers` set to one it degenerates to a single linear layer, giving me the underpowered ablation on the same code path, and with `num_layers` at least two it is the universal approximator the injectivity argument requires. The graph network, GraphCNN, stacks `num_layers - 1` of these transforms; neighbor pooling can be sum, average, or max, matching the three aggregators compared above, computed either through padded neighbor lists for max-pooling or through a sparse block-diagonal adjacency matrix for sum and average pooling; the center node is combined with its pooled neighbors either as the learned $(1+\epsilon)$ term when `center_weighting` is on, or by folding self-loops into the same adjacency when it is off, giving the fixed $\epsilon=0$ variant; and the graph readout sum- or average-pools node features at each depth through a sparse graph-pooling matrix and feeds each depth's pooled vector through its own linear head, summing the heads' outputs to realize the concatenate-then-classify readout as one accumulated score.

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

In the maximally expressive setting, `neighbor_pooling_type="sum"`, `graph_pooling_type="sum"`, and `center_weighting=True` learn the $(1+\epsilon)$ center term; with `center_weighting=False` the preprocessing adds self-loops instead, giving the fixed $\epsilon=0$ variant. The same implementation keeps average and max neighbor pooling in place as the controlled ablations the theory predicts should be strictly weaker.
