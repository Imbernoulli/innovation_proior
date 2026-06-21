The task is link prediction: given a partially observed undirected graph $G=(V,E)$ and a pair of nodes $(x,y)$ with no observed edge, output how likely the edge is real-but-missing or about to form. The instinct is to reach for a heuristic score — count common neighbors, sum Adamic-Adar, run Katz or rooted PageRank — and on the right network these are excellent, free, and need no training. But every one of them is a fixed function of the structure that bakes in a single hypothesis about how links come to exist. Common Neighbors says two nodes are more likely to connect the more friends they share; that is a genuine mechanism in social graphs, but in protein-protein interaction networks the data invert it — proteins sharing many partners are *less* likely to interact directly. The same score that wins on one network is worse than chance on another, and on power grids or router topologies essentially all the classic scores sit near chance because none of their assumptions describe how those networks wire up. The brittleness is structural, not a tuning issue: we are choosing a hypothesis by hand when we should be learning it from the network at hand. The direct predecessor, WLNM, does learn from local structure — it grows an enclosing subgraph around each pair, orders its vertices by a Weisfeiler-Lehman hashing, truncates to exactly $K$ vertices, and classifies the resulting fixed $K\times K$ adjacency matrix with a fully-connected net — and it beats the hand-crafted scores. But the fully-connected net demands a fixed-size input, which forces the truncation, so the model cannot read each pair's full $h$-hop neighborhood and discards exactly the structure it is trying to learn; and because its input is an adjacency matrix and nothing else, there is no slot for the learned embeddings and node attributes we know help when combined with topology. Both limits trace to one root: a fully-connected learner needing a fixed-size tensor.

I propose SEAL — learning from Subgraphs, Embeddings, and Attributes for Link prediction — which turns link prediction into *subgraph classification*: for each candidate pair, extract a small local enclosing subgraph, mark every node with a structural label encoding its role relative to the pair, attach optional latent embeddings and node attributes, and train a graph-level GNN to decide whether the subgraph has a link at its center. Before redesigning the learner I had to settle whether local subgraphs are even rich enough to learn the *strong* heuristics, since the strongest — Katz, rooted PageRank, SimRank — are high-order and by definition read the whole network. Define the $h$-hop enclosing subgraph $G^h_{x,y}$ as the subgraph induced by $\{i : d(i,x)\le h \text{ or } d(i,y)\le h\}$. Any heuristic of order $h$ depends only on nodes within $h$ hops of a center, so it is computable exactly from $G^h_{x,y}$; the low-order scores are trivially covered, and the whole fight is over the high-order ones. The clue is that each high-order heuristic, looked at structurally, is a damped sum over walks whose weight decays geometrically in walk length: Katz is $\sum_{l=1}^\infty \beta^l [A^l]_{x,y}$, rooted PageRank through the inverse-P-distance identity is $(1-\alpha)\sum_{w:x\rightsquigarrow y} P[w]\,\alpha^{\mathrm{len}(w)}$, and SimRank through its meeting-walk expansion is $\sum_w P[w]\,\gamma^{\mathrm{len}(w)}$. Abstract all three as a $\gamma$-decaying heuristic
$$H(x,y) = \eta \sum_{l=1}^\infty \gamma^l\, f(x,y,l), \qquad \gamma\in(0,1),\ \eta>0\ \text{bounded},\ f\ge 0.$$
A small enclosing subgraph cannot compute the infinite sum, but it does not need to: it computes the first chunk of terms and lets the tail go. If $f(x,y,l)$ is computable from $G^h_{x,y}$ for $l$ up to $g(h)$, approximate $H$ by $\tilde H := \eta\sum_{l=1}^{g(h)} \gamma^l f(x,y,l)$, with error equal to the dropped tail. Under the bound $f(x,y,l)\le \lambda^l$ with $\lambda < 1/\gamma$, each tail term is at most $(\gamma\lambda)^l$ with $\gamma\lambda<1$, so the tail is a convergent geometric series and, taking $g(h)=ah+b$,
$$|H-\tilde H| \le \eta \sum_{l=ah+b+1}^\infty (\gamma\lambda)^l = \frac{\eta\,(\gamma\lambda)^{ah+b+1}}{1-\gamma\lambda} = \frac{\eta\,(\gamma\lambda)^{b+1}}{1-\gamma\lambda}\big((\gamma\lambda)^a\big)^h,$$
a constant times $((\gamma\lambda)^a)^h$ — the approximation error decays *at least exponentially in $h$*. The cutoff $g(h)$ comes from a walk-locality lemma: for any walk $\langle x,v_1,\dots,v_{l-1},y\rangle$, if an intermediate $v_i$ had $d(v_i,x)\ge h+1$ and $d(v_i,y)\ge h+1$ then $l \ge d(v_i,x)+d(v_i,y)\ge 2h+2$, so for $l\le 2h+1$ every intermediate node lies within $h$ of a center and the whole walk lives in $G^h_{x,y}$; hence $g(h)=2h+1$ ($a=2,b=1$). The three heuristics satisfy the hypotheses: Katz has $f=[A^l]_{x,y}\le d^l$ (induction on $l$ with $d$ the max degree, base $A_{i,j}\le d$, step $[A^{l+1}]_{i,j}=\sum_k[A^l]_{i,k}A_{k,j}\le d^l\cdot d$), so $\lambda=d$ and the requirement $\lambda<1/\gamma$ becomes $d<1/\beta\approx 2000$, true for any realistic network; PageRank and SimRank have $f$ a sum of walk probabilities, hence $\le 1 < 1/\gamma$, so $\lambda=1$. Three for three, and it is not a coincidence: a heuristic that put non-vanishing weight on arbitrarily distant structure would be dominated by regions of the network irrelevant to whether $x$ and $y$ link, so the very exponential down-weighting that makes a global heuristic *good* is what makes it *local*. The local-subgraph program is therefore not capped below the high-order heuristics — a learner with enough capacity, fed small enclosing subgraphs, can match or beat any of them and can discover structural signals no named heuristic captures — and I get to use a small $h$.

With the program justified I replace the fixed-size learner with a graph-level GNN, which swallows arbitrary-size subgraphs and a continuous node-information matrix $X$ — so no truncation, and a natural slot for embeddings and attributes. But a standard GNN updates nodes with shared weights and pools by summing, which is exactly the symmetry I cannot afford: an enclosing subgraph has a center, the pair $(x,y)$ whose link is in question, and a symmetric, sum-pooling GNN cannot tell which two nodes that is — two subgraphs with completely different target pairs but the same overall shape would look identical. So I mark the nodes with a structural label and one-hot it into $X$. The label must satisfy two requirements derived from what it has to accomplish: the target pair $x,y$ gets a single distinctive label so the GNN can always locate the link, and two other nodes get the same label exactly when they share the same role with respect to the pair, where role is the pair of distances $(d(i,x),d(i,y))$ — the *double radius*. This is Double-Radius Node Labeling. Give $x,y$ the label $1$; order the remaining orbits by going outward in shells, ranking first by the distance sum $d(i,x)+d(i,y)$ (a larger sum is a farther shell, a larger label) and tie-breaking within a shell by the product $d(i,x)\cdot d(i,y)$ (smaller product, more central, smaller label), which is a total order on orbits. Rather than sort orbits per subgraph, I want a closed form, so with $d_x=d(i,x)$, $d_y=d(i,y)$, $d=d_x+d_y$ I count the labels consumed by all strictly smaller shells and add the within-shell offset $\min(d_x,d_y)$, yielding the perfect hash
$$f_l(i) = 1 + \min(d_x,d_y) + \frac{d}{2}\Big[\frac{d}{2} + (d\bmod 2) - 1\Big]$$
with integer division. It reproduces the hand-built labels exactly: $(1,1)\to 2$, $(1,2)\to 3$, $(1,3)\to 4$, $(2,2)\to 5$, $(1,4)\to 6$, $(2,3)\to 7$. Two subtleties matter. The distance $d(i,x)$ is measured with $y$ *removed* from the subgraph (and $x$ removed when measuring $d(i,y)$), because otherwise a path from $i$ to $x$ could shortcut through $y$ — $d(i,x)\le d(i,y)+d(x,y)$ — contaminating the pure radius; nodes unreachable from a center get the null label $0$. And the labels are one-hot, not used as magnitudes: WLNM needed maximally discriminating labels to define a fine vertex *ordering*, but here labels only mark structural *roles* for the GNN to consume, so coarse one-hot role indicators are exactly right.

The node-information matrix is then $X = [\text{one-hot DRNL label} \,\|\, \text{node embedding} \,\|\, \text{node attributes}]$, letting one GNN learn jointly from structure, latent, and explicit features, with two leakage fixes that are essential. Generating embeddings naively by running a network-embedding method on $G$ poisons them, because the positive training links are a subset of the observed edges $E_p\subseteq E$, so the embeddings encode the existence of exactly those edges; the GNN reads that off and overfits, learning nothing that transfers to held-out edges that were never in $G$. The fix — negative injection — is to embed on $G'=(V, E\cup E_n)$ with the negative training links $E_n$ temporarily injected, so positives and negatives both carry the same "these are connected" signal and the embeddings no longer separate them, forcing the GNN to learn from real structure. The same logic applies one level up: in a positive subgraph the center edge $(x,y)$ is literally present and *is* the label, absent at test time, so it must be deleted from every positive subgraph before classification. The GNN engine is DGCNN. The convolution is a localized, propagation-style layer,
$$Z = f\big(\tilde D^{-1}\tilde A\, X W\big),\qquad \tilde A = A + I,$$
so row $i$ becomes $f\big(\tfrac{1}{|\Gamma(i)|+1}[X_iW + \sum_{j\in\Gamma(i)} X_jW]\big)$ — each node averaged with its neighbors, a differentiable relaxation of Weisfeiler-Lehman color refinement; in code I instantiate this with `GCNConv` and keep its `edge_weight` path. Stacking several such layers and *concatenating* every layer's per-node output gives each node a descriptor spanning one-hop through $h$-hop substructure. Pooling is where ordinary GNNs lose too much: instead of summing node states (permutation-invariant but information-destroying), SortPooling sorts the nodes by their final convolutional states, truncates or zero-pads the sorted sequence to a fixed length $k$, and runs an ordinary 1-D CNN over it — sorting is permutation-invariant in the way that matters (isomorphic subgraphs sort to the same sequence) yet keeps each node's identity and arrangement, which a 1-D CNN can then read in a meaningful order, with $k$ chosen as a quantile of the training subgraph-size distribution floored at $10$. The radius $h$ is selected only from $\{1,2\}$, since the exponential-decay bound says each extra hop buys geometrically less while $h=3$ risks exploding the subgraph around a hub: use $h=2$ if Adamic-Adar beats Common Neighbors on a validation split, else $h=1$. As a sanity check, $h=0$ leaves only the two isolated target nodes, the propagation matrix becomes the identity, and the model reduces to a latent-feature link predictor — so SEAL is a strict generalization of the embedding methods with enclosing-subgraph structure layered on, not a fourth competitor.

```python
import math
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, Embedding, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_sort_pool


def neighbors(fringe, A):
    return set(A[list(fringe)].indices)


def drnl_node_labeling(adj, src, dst):
    """Double-Radius Node Labeling: integer role label per node from its
    distances to the two target nodes (src, dst)."""
    src, dst = (dst, src) if src > dst else (src, dst)

    # distance to one center measured with the OTHER center removed
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]
    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    # f_l = 1 + min(d_x,d_y) + (d/2)*[(d/2)+(d%2)-1]
    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.            # unreachable -> null label 0
    return z.to(torch.long)


def k_hop_subgraph(src, dst, num_hops, A, node_features=None, y=1):
    """Enclosing subgraph: BFS out num_hops hops from {src, dst}; drop target edge."""
    nodes, dists = [src, dst], [0, 0]
    visited = set([src, dst]); fringe = set([src, dst])
    for dist in range(1, num_hops + 1):
        fringe = neighbors(fringe, A) - visited
        visited |= fringe
        if len(fringe) == 0:
            break
        nodes += list(fringe); dists += [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]
    subgraph[0, 1] = 0; subgraph[1, 0] = 0       # remove the target edge (it is the label)
    if node_features is not None:
        node_features = node_features[nodes]
    return nodes, subgraph, dists, node_features, y


def construct_pyg_graph(node_ids, adj, dists, node_features, y):
    u, v, r = ssp.find(adj)
    edge_index = torch.stack([torch.LongTensor(u), torch.LongTensor(v)], 0)
    edge_weight = torch.LongTensor(r).to(torch.float)
    z = drnl_node_labeling(adj, 0, 1)            # targets at positions 0, 1
    return Data(node_features, edge_index, edge_weight=edge_weight, y=torch.tensor([y]), z=z,
                node_id=torch.LongTensor(node_ids), num_nodes=adj.shape[0])


class LinkPredictor(torch.nn.Module):
    """SEAL: DGCNN over each candidate edge's enclosing subgraph."""
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6,
                 train_dataset=None, dynamic_train=False,
                 use_feature=False, node_embedding=None):
        super().__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        if k <= 1:                                # k as a percentile of subgraph sizes
            if train_dataset is None:
                k = 30
            elif dynamic_train:
                num_nodes = sorted([g.num_nodes for g in train_dataset[:1000]])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
            else:
                num_nodes = sorted([g.num_nodes for g in train_dataset])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.z_embedding = Embedding(max_z, hidden_channels)
        initial_channels = hidden_channels
        if use_feature:
            initial_channels += train_dataset.num_features
        if node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs = ModuleList()
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))     # last channel drives the sort

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        # node information matrix X = [DRNL label || attrs || latent embedding]
        h = self.z_embedding(z)
        if h.ndim == 3:
            h = h.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([h, x.to(torch.float)], 1)
        if self.node_embedding is not None and node_id is not None:
            h = torch.cat([h, self.node_embedding(node_id)], 1)

        xs = [h]
        for conv in self.convs:                    # multi-hop substructure features
            xs.append(torch.tanh(conv(xs[-1], edge_index, edge_weight)))
        x = torch.cat(xs[1:], dim=-1)              # per-node concat of all layers

        x = global_sort_pool(x, batch, self.k)     # SortPooling
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)                        # logit: subgraph has a center link?
```
