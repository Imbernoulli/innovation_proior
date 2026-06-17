# SEAL, distilled

SEAL (learning from **S**ubgraphs, **E**mbeddings, and **A**ttributes for **L**ink prediction)
turns link prediction into *subgraph classification*: for each candidate pair `(x, y)`, extract a
local `h`-hop enclosing subgraph, mark every node with a structural label encoding its role
relative to the pair, attach optional latent embeddings and node attributes, and train a
graph-level GNN to classify whether the subgraph has a link in its center. It learns a
network-specific "heuristic" instead of assuming a fixed one, and is justified by a γ-decaying
theory showing local subgraphs already contain the information of the global high-order
heuristics.

## Problem it solves

Predict missing/future links in a partially observed graph. Predefined heuristics (Common
Neighbors, Adamic-Adar, Katz, rooted PageRank, SimRank) each hard-code one assumption about link
formation and fail when it does not match the network. SEAL instead learns the structural signal,
works across diverse networks, can fold in latent and explicit node features, and handles graphs
of arbitrary size.

## Key idea 1 — the γ-decaying heuristic theory (why local subgraphs suffice)

Define the `h`-hop **enclosing subgraph** `G^h_{x,y}` = subgraph induced by `{i : d(i,x) ≤ h or
d(i,y) ≤ h}`. Any `h`-order heuristic is computable exactly from `G^h_{x,y}`. The strong
heuristics are *high-order* (need the whole graph), but they all share a form:

A **γ-decaying heuristic** is `H(x,y) = η · Σ_{l=1}^∞ γ^l f(x,y,l)`, with `γ ∈ (0,1)`, `η > 0`
bounded, `f ≥ 0`.

**Theorem (exponential local approximability).** If (P1) `f(x,y,l) ≤ λ^l` with `λ < 1/γ`, and (P2)
`f(x,y,l)` is computable from `G^h_{x,y}` for `l = 1,…,g(h)` with `g(h) = ah+b`, `a>0`, then
`H̃ := η Σ_{l=1}^{g(h)} γ^l f` approximates `H` with

```
|H − H̃| = η Σ_{l=g(h)+1}^∞ γ^l f(x,y,l)
        ≤ η Σ_{l=ah+b+1}^∞ (γλ)^l
        = η (γλ)^{ah+b+1} / (1 − γλ),
```

which decays **at least exponentially in `h`**. (Geometric tail; `γλ < 1`.)

**Lemma (walk locality).** Any walk between `x` and `y` of length `l ≤ 2h+1` lies entirely in
`G^h_{x,y}`. *Proof:* for `⟨x,v_1,…,v_{l-1},y⟩`, if some `v_i` had `d(v_i,x) ≥ h+1` and
`d(v_i,y) ≥ h+1`, then `l ≥ d(v_i,x)+d(v_i,y) ≥ 2h+2`, contradicting `l ≤ 2h+1`; so every `v_i`
is within `h` of a center. Hence `g(h)=2h+1` (`a=2,b=1`) for walk-based heuristics.

**The three high-order heuristics are γ-decaying with P1, P2:**

- **Katz** `Σ β^l [A^l]_{x,y}`: `η=1, γ=β, f=[A^l]_{x,y}`. P1: `[A^l]_{i,j} ≤ d^l` (`d` = max
  degree) by induction — base `A_{i,j} ≤ d`; step `[A^{l+1}]_{i,j}=Σ_k[A^l]_{i,k}A_{k,j} ≤ d^l·d`.
  So `λ=d`; need `d < 1/β` (with `β ≈ 5e-4`, i.e. degree `< 2000`). P2: walk lemma.
- **rooted PageRank** `[π_x]_y = (1-α) Σ_{w:x⇝y} P[w] α^{len(w)}` (inverse P-distance): `η=1-α,
  γ=α, f(x,y,l)=Σ_{len(w)=l} P[w]` = Prob(walker from `x` is at `y` after `l` steps). `Σ_z
  f(x,z,l)=1 ⇒ f ≤ 1 < 1/α` (P1, `λ=1`). P2: walk lemma.
- **SimRank** `s(x,y)=Σ_w P[w] γ^{len(w)}` over simultaneous meeting walks: `f(x,y,l) ≤ 1 < 1/γ`
  (P1), computable from `G^h` for `l ≤ h` (P2).

`h=0` reduces SEAL to a latent-feature method (only the two isolated targets remain).

## Key idea 2 — Double-Radius Node Labeling (DRNL)

A symmetric GNN that sums node features cannot tell which two nodes are the target pair, losing
the link's location. So label each node by its role:

- target `x, y` get the distinctive label `1`;
- nodes `i, j` get the same label iff `(d(i,x), d(i,y)) = (d(j,x), d(j,y))` (same "double radius"
  / orbit), ordered by `d(i,x)+d(i,y)` (sum), tie-broken by `d(i,x)·d(i,y)` (product).

**Perfect-hash closed form** (`d_x=d(i,x)`, `d_y=d(i,y)`, `d=d_x+d_y`):

```
f_l(i) = 1 + min(d_x, d_y) + (d/2)·[(d/2) + (d%2) − 1]      (integer / and %)
```

Check: `(1,1)→2`, `(1,2)/(2,1)→3`, `(1,3)/(3,1)→4`, `(2,2)→5`, `(1,4)/(4,1)→6`, `(2,3)/(3,2)→7`.
Measure `d(i,x)` with `y` removed from the subgraph (and vice versa), so one center does not
shortcut the radius to the other; nodes unreachable from a center get null label `0`. One-hot the
labels into the node-information matrix `X` (only role-marking is needed, not magnitude).

## Key idea 3 — combining the three feature types + leakage fixes

`X` per node = `[ one-hot DRNL label ‖ node embedding ‖ node attributes ]`, letting one GNN learn
from structure, latent, and explicit features jointly. Two leak fixes:

- **Negative injection.** Embedding on `G` (which contains the positive train links `E_p ⊆ E`)
  leaks their existence into the embeddings; the GNN overfits this and fails to generalize. Fix:
  embed on `G' = (V, E ∪ E_n)` (inject the negative train links `E_n`), so positives and
  negatives carry the same existence signal and the GNN must learn from real structure.
- **Drop the target edge.** In a *positive* subgraph the edge `(x,y)` is the label and is absent
  at test time, so remove it before classification.

## GNN engine — propagation conv + sorting pool (DGCNN)

Paper-level DGCNN graph convolution (random-walk-normalized, a "soft Weisfeiler-Lehman"):

```
Z = f(D̃^{-1} Ã X W),   Ã = A + I,   row i: f( (1/(|Γ(i)|+1)) [X_i W + Σ_{j∈Γ(i)} X_j W] )
```

Stack conv layers (e.g. 4 layers, `32,32,32,1` channels), **concatenate** every layer's per-node
output into a multi-hop descriptor, then **SortPooling**: sort nodes by their final conv states
(the PyG reference path sorts by the last channel descending), truncate/pad to `k` (the
training-set quantile when `k <= 1`, with a floor of `10`), and run a 1-D CNN + dense head.
Sorting is permutation-invariant (isomorphic subgraphs → same representation) yet keeps
node/topology info that summing would wash out. `h` is selected from `{1,2}`: use `h=2` if
Adamic-Adar beats Common Neighbors on validation, else `h=1`, avoiding large subgraphs around hubs.

## Working code

Grounded in the canonical implementation (DRNL + enclosing-subgraph extraction + DGCNN). Fills
the `LinkPredictor` slot of the link-prediction harness.

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
