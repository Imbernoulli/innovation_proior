# GraphSAGE: inductive node embeddings by sampling and aggregating

## Problem

Produce embeddings for nodes — and for entire graphs — that were not present during training,
cheaply, at inference time, without per-node re-optimization. Prior node-embedding methods optimize
a free vector per node (a lookup table), which is transductive by construction: an unseen node has
no vector. GraphSAGE (SAmple and aggreGatE) instead learns a small set of functions that *generate*
a node's embedding from its features and its local neighborhood, so any unseen node simply runs the
functions.

## Key idea

Learn shared aggregator functions and weight matrices, not per-node embeddings. For a node v,
aggregate a fixed-size uniform sample of its neighbors' previous-layer representations, concatenate
with v's own representation (a skip connection across depths), transform, and ℓ2-normalize. Stack K
such layers so depth = number of hops reached.

## Forward pass (embedding generation)

Input: graph G(V,E), features {x_v}, depth K, weights {W^k}, nonlinearity σ, aggregators
{AGGREGATE_k}, neighbor sampler N.

```
h^0_v = x_v
for k = 1 … K:
    for v in V:
        h^k_{N(v)} = AGGREGATE_k({ h^{k-1}_u : u in N(v) })
        h^k_v      = σ( W^k · CONCAT( h^{k-1}_v , h^k_{N(v)} ) )
    h^k_v = h^k_v / ‖h^k_v‖_2   for all v
z_v = h^K_v
```

- **N(v):** a fixed-size uniform sample of v's neighbors, drawn fresh per depth k (sample with
  replacement if degree < sample size). For a fixed target batch size, per-batch cost is
  O(∏_{i=1}^K S_i), constant regardless of hub degrees; more exactly it scales with the batch size
  times that product. With K=2 and algorithm-layer sizes S_1=25, S_2=10, a target node samples S_2
  immediate neighbors and S_1·S_2 two-hop neighbors.
- **Minibatch:** from target batch B, expand backward through K hops of sampled neighbors to get the
  support sets B^K ⊆ … ⊆ B^0, then run the forward recursion up over exactly those nodes.

## Aggregator architectures (must be permutation-invariant)

- **Mean:** elementwise mean of neighbor vectors. Folding the self vector into the mean with a
  single shared weight matrix recovers an inductive variant of the GCN convolution (GraphSAGE-GCN),
  differing only by a normalization constant; it drops the concatenation. Cheap, low capacity.
- **Pool:** AGGREGATE^pool = max({ σ(W_pool h_u + b) : u ∈ N(v) }), elementwise max over a
  per-neighbor single-layer MLP. Symmetric, trainable, high capacity; the max acts as a soft
  existential over neighbor features.
- **LSTM:** higher capacity but not permutation-invariant; adapted by feeding a random permutation
  of neighbors. Strong but slowest.

## Losses

Unsupervised, task-agnostic graph-based loss applied to the *generated* embeddings z (not a lookup
table):

J(z_u) = − log σ(z_uᵀ z_v) − Q · E_{v_n ∼ P_n(v)} log σ(− z_uᵀ z_{v_n}),

where v co-occurs with u on a fixed-length random walk, P_n is a negative-sampling distribution
(degree^{3/4} smoothing), and Q is the number of negatives (≈20). The pull/push gradients flow into
the aggregator and weight parameters. For a single known task, replace J with supervised
cross-entropy on labels.

## Why it can capture structure

Setting K=|V|, identity weights, a hash aggregator, and no nonlinearity makes the recursion the
Weisfeiler–Lehman vertex-refinement isomorphism test; GraphSAGE is its continuous, trainable
relaxation. Concretely, if all node features are pairwise separated (‖x_v − x_{v'}‖₂ > C), the
pool-based model can approximate every node's clustering coefficient
c_v = 2(#edges among v's neighbors)/(d_v(d_v−1)) to arbitrary precision in K=4 iterations: coloring
the fourth graph power G⁴ gives locally unique one-hot indicators via pool MLPs (universal symmetric
approximation), summing/identity layers assemble [own indicator | own adjacency row | sum of
neighbors' rows] into h³_v, and for a simple undirected graph the blocks b=A_v and
c=Σ_{u∈N(v)} A_u satisfy bᵀc=2·(#edges among v's neighbors), so c_v=bᵀc/(d_v(d_v−1)) for d_v≥2
(with the usual zero convention for smaller degrees). This holds almost surely for continuous
random features.

## Working code

Mean and pool aggregators with sampling, an encoder doing concat + shared-weight + ReLU + ℓ2-norm,
stacked twice for K=2, plus the unsupervised graph loss and a supervised head.

```python
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _node_list(nodes):
    return [int(n) for n in nodes]


def _sample(neigh, fallback, num_sample):
    neigh = list(neigh)
    if not neigh:
        neigh = [fallback]
    if num_sample is None:
        return neigh
    if len(neigh) >= num_sample:
        return random.sample(neigh, num_sample)
    return random.choices(neigh, k=num_sample)


class MeanAggregator(nn.Module):
    """Mean of a uniform fixed-size sample of neighbor representations."""
    def __init__(self, features, gcn=False):
        super().__init__()
        self.features = features    # node ids -> previous-layer reprs
        self.gcn = gcn              # fold self into the mean (inductive GCN) vs. keep separate
        self.out_dim = None

    def forward(self, nodes, to_neighs, num_sample=10):
        nodes = _node_list(nodes)
        samp = [_sample(neigh, nodes[i], num_sample) for i, neigh in enumerate(to_neighs)]
        if self.gcn:
            samp = [s + [nodes[i]] for i, s in enumerate(samp)]
        uniq = sorted(set().union(*samp))
        idx = {n: i for i, n in enumerate(uniq)}
        mask = torch.zeros(len(samp), len(uniq))
        for row, neigh in enumerate(samp):
            for n in neigh:
                mask[row, idx[n]] += 1.0
        mask = mask / mask.sum(1, keepdim=True).clamp_min(1.0)
        embed = self.features(torch.as_tensor(uniq, dtype=torch.long))
        return mask.to(embed.device).mm(embed)

class PoolAggregator(nn.Module):
    """Per-neighbor MLP then elementwise max: symmetric, trainable, high capacity."""
    def __init__(self, features, in_dim, hidden=None):
        super().__init__()
        self.features = features
        self.out_dim = in_dim if hidden is None else hidden
        self.mlp = nn.Linear(in_dim, self.out_dim)

    def forward(self, nodes, to_neighs, num_sample=10):
        nodes = _node_list(nodes)
        out = []
        for i, neigh in enumerate(to_neighs):
            samp = _sample(neigh, nodes[i], num_sample)
            h = self.features(torch.as_tensor(samp, dtype=torch.long))
            out.append(F.relu(self.mlp(h)).max(0).values)
        return torch.stack(out, 0)

class Encoder(nn.Module):
    """One depth: aggregate neighbors, concat with self, shared W, ReLU, ell2-norm."""
    def __init__(self, features, feat_dim, embed_dim, adj_lists, aggregator,
                 num_sample=10, gcn=False):
        super().__init__()
        self.features, self.adj_lists = features, adj_lists
        self.aggregator, self.num_sample, self.gcn = aggregator, num_sample, gcn
        self.embed_dim = embed_dim
        neigh_dim = feat_dim if aggregator.out_dim is None else aggregator.out_dim
        in_dim = neigh_dim if gcn else feat_dim + neigh_dim
        self.weight = nn.Parameter(torch.empty(embed_dim, in_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        nodes = _node_list(nodes)
        neigh = self.aggregator(nodes, [self.adj_lists[n] for n in nodes],
                                self.num_sample)
        if not self.gcn:
            self_feats = self.features(torch.as_tensor(nodes, dtype=torch.long))
            combined = torch.cat([self_feats, neigh], dim=1)
        else:
            combined = neigh
        h = F.relu(self.weight.to(combined.device).mm(combined.t()))
        return F.normalize(h, p=2, dim=0, eps=1e-12)

def unsupervised_loss(z_u, z_v, z_neg, Q=None):
    """z_u/z_v are [batch, dim]; z_neg is [batch, Q, dim] or shared [Q, dim]."""
    pos = F.logsigmoid((z_u * z_v).sum(1))
    if z_neg.dim() == 2:
        neg_scores = z_u.mm(z_neg.t())
    else:
        neg_scores = (z_u.unsqueeze(1) * z_neg).sum(2)
    q = neg_scores.size(1) if Q is None else Q
    neg = F.logsigmoid(-neg_scores).mean(1)
    return -(pos + q * neg).mean()

class SupervisedGraphSAGE(nn.Module):
    def __init__(self, num_classes, enc):
        super().__init__()
        self.enc = enc
        self.weight = nn.Parameter(torch.empty(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return self.weight.to(embeds.device).mm(embeds).t()

    def loss(self, nodes, labels):
        return F.cross_entropy(self.forward(nodes), labels.squeeze())

def build(features, feat_dim, adj_lists, num_classes, gcn=False):
    agg1 = MeanAggregator(features, gcn=gcn)
    enc1 = Encoder(features, feat_dim, 256, adj_lists, agg1, num_sample=10, gcn=gcn)
    enc1_rows = lambda n: enc1(n).t()
    agg2 = MeanAggregator(enc1_rows, gcn=gcn)
    enc2 = Encoder(enc1_rows, enc1.embed_dim, 256, adj_lists, agg2,
                   num_sample=25, gcn=gcn)
    return SupervisedGraphSAGE(num_classes, enc2)
```
