Message-passing graph neural networks such as GCN and GraphSAGE have become the standard way to learn from graph-structured data, but their design has been largely empirical. Each layer follows the same template: a node aggregates the feature vectors of its neighbors, combines that aggregate with its own feature, and updates its representation. After k rounds a node's vector is meant to summarize the rooted subtree of height k around it, yet no prior work had pinned down what that summary actually preserves. The open questions are fundamental: which non-isomorphic graphs can such a network ever map to different embeddings, is there a hard upper bound on that power independent of width or training, and do choices like mean versus max versus sum aggregation or a single linear layer versus a deeper network change the representational ceiling rather than just making optimization easier?

Existing aggregators are provably lossy and therefore cap expressive power. Mean pooling preserves only the distribution of neighbor features, so it cannot distinguish a multiset from an inflated copy that has the same proportions; for instance it confuses the neighbor bags green-red and green-green-red-red. Max pooling preserves only the set of distinct elements and ignores multiplicities entirely, so it cannot tell green-red apart from green-red-red. A single linear layer followed by a nonlinearity is not a universal approximator of multiset functions; for positive inputs it degenerates to a linear function of the neighbor sum and can collapse structurally different multisets such as five copies of one feature versus two copies of different features. These are not mere optimization differences; they place a ceiling on what the network can distinguish no matter how it is trained or how wide it is made.

The method I propose is the Graph Isomorphism Network, or GIN. Its theoretical target is the one-dimensional Weisfeiler-Lehman graph isomorphism test, which repeatedly refines each node's color by hashing the pair consisting of its current color and the multiset of its neighbors' colors. I prove that no neighborhood-aggregation network can be more powerful than 1-WL, because whenever WL cannot separate two graphs, any network that uses the same aggregation and combine functions at every node must also produce identical feature multisets and therefore identical graph-level readouts. Conversely, I construct GIN so that it reaches this bound exactly. The key requirement is injectivity on multisets at every step: the neighbor aggregation, the combine function that merges the center node with its neighborhood, and the graph-level readout must all keep distinct inputs distinct.

The layer update is h_v^{(k)} equals MLP^{(k)} applied to (1 plus epsilon^{(k)}) times h_v^{(k-1)} plus the sum over neighbors u of h_u^{(k-1)}. The sum is the only aggregator among the common choices that keeps full multiset information; by summing learned per-element features it can encode exact counts, and any permutation-invariant multiset function can be written as a transformation of that sum. The MLP is essential because a one-layer transform cannot realize the injective encoding needed for arbitrary multisets; it lacks the universality required to separate inputs that happen to have the same total sum. The (1 plus epsilon) factor distinguishes the center atom from its neighbors; without it, pooling center and neighbors together would lose the root, for example confusing the middle atom of a-b-b with the middle atom of b-a-b, since both would collapse to the flat multiset a-b-b.

For graph-level prediction, I sum-pool node features at every layer and concatenate or sum the resulting vectors, giving the classifier access to subtree representations from local to global scales. This mirrors the WL subtree kernel but with learned continuous embeddings, so similar substructures can map to nearby vectors while distinct subtrees remain separable. In the molecular setting, bond features are incorporated by projecting each edge vector into the atom feature space and adding it to the message from the source atom before summing into the destination atom. The full model is built by stacking these edge-aware GIN convolutions with batch normalization, ReLU, residual connections, and per-layer prediction heads whose outputs are summed to realize the multi-scale readout.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GINConv(nn.Module):
    """Graph Isomorphism Network convolution with edge-aware sum messages."""

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        edge_msg = self.edge_proj(edge_attr)
        msg = x[src] + edge_msg

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        out = self.mlp((1 + self.eps) * x + agg)
        return out


class GINMoleculeModel(nn.Module):
    """Graph Isomorphism Network for molecular property prediction."""

    def __init__(self, atom_dim, edge_dim, num_tasks, hidden_dim=256, num_layers=4):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_tasks),
            ) for _ in range(num_layers + 1)
        ])

    def forward(self, batch):
        x = self.atom_embed(batch.x)
        hidden_rep = [x]

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, batch.edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + self.dropout(x_new)
            hidden_rep.append(x)

        num_graphs = batch.batch_idx.max().item() + 1
        score = 0
        for h, head in zip(hidden_rep, self.prediction_heads):
            graph_embed = torch.zeros(num_graphs, h.size(-1), device=h.device)
            graph_embed.index_add_(0, batch.batch_idx, h)
            score = score + head(graph_embed)

        return score
```
