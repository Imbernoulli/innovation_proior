The practical problem is embedding nodes that appear after training time. Production graphs keep growing: new papers are cited, new posts are made, and new protein networks are sequenced. Every useful downstream classifier expects a dense vector, but the dominant node-embedding methods — DeepWalk, node2vec, LINE — train a free lookup table Z with one row per node. That makes them transductive by construction: an unseen node has no row, so the only way to embed it is to add a row and run more SGD, which is orders of magnitude slower than a single forward pass and unacceptable at inference time. Worse, their objectives depend only on dot products z_i^T z_j, which are invariant under arbitrary orthogonal rotations of the whole embedding space, so two graphs trained separately land in unrelated coordinate systems and a classifier trained on one reads the other as noise. Graph convolutions such as GCN take a step in the right direction by learning shared weight matrices that act on node features, but they are implemented as a full-graph Laplacian multiply that requires the entire adjacency matrix in memory and has no mechanism to process a brand-new node from its local neighborhood alone.

I propose GraphSAGE, which stands for SAmple and aggreGatE. The core shift is to learn a set of aggregator functions and weight matrices that generate a node's embedding from the things observable about it — its raw features and the features of its local neighbors — rather than optimizing a table of embeddings directly. A trained GraphSAGE model can therefore embed a never-seen node, or even an entire unseen graph, by simply running the same forward functions on its local neighborhoods. The forward recursion is straightforward. Each node starts with h^0_v = x_v. For each depth k = 1 ... K, we sample a fixed-size uniform subset of the node's neighbors, aggregate their previous-layer representations into a single vector, concatenate that aggregated neighborhood vector with the node's own previous-layer representation, multiply by a shared weight matrix W^k, apply a nonlinearity, and l2-normalize. Formally, h^k_{N(v)} = AGGREGATE_k({h^{k-1}_u : u in N(v)}) and h^k_v = sigma(W^k . CONCAT(h^{k-1}_v, h^k_{N(v)})), followed by normalization. The separate self-channel acts as a skip connection across depths so the node's own signal is never diluted by repeated averaging. Stacking K layers reaches K hops into the neighborhood, so depth controls receptive field while the computation itself remains local and node-centric.

The aggregator must be permutation-invariant because neighbors have no canonical order. Three variants are natural. A mean aggregator simply averages neighbor vectors; if the self-vector is folded into that mean and a single weight matrix is used, this recovers an inductive variant of the GCN convolution up to a normalization constant, making GCN a special case of the framework. A pool aggregator pushes each neighbor vector through a shared single-layer MLP and then takes an elementwise max over the set, which is symmetric, trainable, and expressive enough to act like a soft existential test over neighbor features. An LSTM aggregator is higher-capacity but order-dependent, so it is fed random permutations of neighbors during training to encourage order-robustness at the cost of speed. To handle heavy-tailed degree distributions and hub nodes without letting a single batch explode, GraphSAGE draws a fixed-size uniform sample of neighbors at each depth, sampling with replacement when degree is smaller than the sample size. With K=2 and sample sizes such as 25 and 10, the per-batch cost is bounded regardless of how large the graph is, which makes the method scale to hundreds of thousands of nodes.

Training uses either an unsupervised graph-based loss or a supervised loss. For the unsupervised variant, generated embeddings are fed into the same SkipGram negative-sampling objective that powers DeepWalk: pull co-occurring nodes together and push random negatives apart, but now the gradients flow into the aggregator and weight parameters rather than into per-node lookup rows. When labels are available, the same generator is trained end-to-end with ordinary cross-entropy. Because the architecture repeatedly aggregates neighbor sets, it is connected to the Weisfeiler-Lehman vertex-refinement isomorphism test: with identity weights, a hashing aggregator, and no nonlinearity, the recursion exactly matches WL. For the pool variant, under mild separation assumptions on node features, one can show by universal approximation that the model can compute genuine structural quantities such as clustering coefficients to arbitrary precision. This means GraphSAGE learns about graph structure, not merely smooths features around the neighborhood.

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
        self.features = features
        self.gcn = gcn
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
    """One depth: aggregate neighbors, concat with self, shared W, ReLU, l2-norm."""
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
    """SkipGram negative-sampling loss on generated embeddings."""
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
