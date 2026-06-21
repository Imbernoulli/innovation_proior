The problem is link prediction on a single partially observed graph: score node pairs so that true held-out edges rank above non-edges. The challenge is to capture the structural signal that actually decides links without paying the cost of constructing a subgraph for every candidate pair. Plain message-passing GNNs fall short because they assign one embedding per node, so automorphic nodes receive identical embeddings and the model cannot even count common neighbors or triangles, which are the backbone of classical link-prediction heuristics. Subgraph GNNs such as SEAL fix this by extracting a k-hop enclosing subgraph around each candidate edge and labeling every node by its distances to the two endpoints, which restores expressiveness but is ruinously expensive: on power-law graphs the cost is roughly O(|E|) per edge, the subgraphs overlap massively, and precomputing them overflows memory. So the target is subgraph-GNN expressiveness at GCN-class cost, with work growing with |E| rather than with per-edge subgraph size.

The method I propose is BUDDY, the scalable variant of the ELPH subgraph-sketching framework. The key observation is that the load-bearing content of a labeled enclosing subgraph is the distance-label count table A_uv[d_u, d_v], the number of nodes at distance exactly d_u from u and exactly d_v from v, together with boundary counts B_uv[d]. A_uv[1, 1] is exactly the common-neighbor count, and the full table subsumes Adamic–Adar, resource allocation, Katz-like mixtures, and other overlap heuristics. These counts reduce to two per-node primitives: neighborhood cardinality |N_d(u)| and neighborhood intersection |N_{d_u}(u) ∩ N_{d_v}(v)|. BUDDY estimates both with compact sketches. HyperLogLog summarizes a set for cardinality, with the useful property that the sketch of a union is the elementwise max. MinHash summarizes a set so that Hamming similarity between two sketches estimates their Jaccard similarity, with the sketch of a union given by elementwise min. The intersection cardinality follows from |S ∩ T| = J(S, T) · |S ∪ T|.

The sketch propagation is itself a form of message passing. Give each node u initial MinHash and HyperLogLog sketches of the singleton {u}. Because the d-hop neighborhood satisfies N_d(u) = ∪_{v ∈ N(u)} N_{d-1}(v), the d-hop sketches are obtained by min-aggregating MinHash sketches and max-aggregating HyperLogLog sketches over neighbors. After k hops every node carries fixed-size sketches of all its neighborhoods up to distance k, in O(k|E|h) time with h the sketch size, independent of the total number of nodes. For any candidate pair (u, v) the model reads off |N_{d_u}(u)| from the HyperLogLog sketch, estimates the intersection from MinHash and HyperLogLog, and converts these into the count table A and boundary counts B by inclusion-exclusion arithmetic. Node features are handled separately: fixed sparse feature propagation x_u^{(l)} = mean_{v ∈ N(u)} x_v^{(l-1)} is precomputed for each hop, and the multi-scale features Z = [X^{(0)} || X^{(1)} || ... || X^{(k)}] are concatenated once.

The predictor is then a simple MLP. For a pair (u, v) it takes the Hadamard product z_u ⊙ z_v of the multi-scale node features and concatenates the structural count features {B_uv[d], A_uv[d_u, d_v] : d, d_u, d_v ≤ k}. An MLP ψ maps this concatenation to a single logit. The MLP learns how to weight the structural counts, recovering common neighbors, Adamic–Adar, resource allocation, or any mixture the data prefers, and how to fuse them with feature interactions. BUDDY specifically refers to the precomputed version: both sketches and propagated features are materialized ahead of time with scatter_min, scatter_max, and scatter_mean operations, so training and inference reduce to an MLP on cached edge features. This makes memory independent of graph size and avoids any per-edge subgraph construction. ELPH is the full-graph variant that computes sketches inside the message-passing loop; BUDDY is the scalable MLP realization and is the canonical method for large-scale link prediction.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElphHashes:
    """Per-node MinHash + HyperLogLog sketches, propagated by min/max for k hops."""

    def __init__(self, num_perm=128, hll_p=8, max_hops=3):
        self.num_perm = num_perm
        self.hll_p = hll_p
        self.max_hops = max_hops

    def initialise_minhash(self, n):
        a = torch.randint(1, (1 << 31) - 1, (self.num_perm,))
        b = torch.randint(0, (1 << 31) - 1, (self.num_perm,))
        ids = torch.arange(n).unsqueeze(1)
        return ((a * ids + b) % ((1 << 31) - 1)).float()

    def initialise_hll(self, n):
        m = 1 << self.hll_p
        reg = torch.zeros(n, m)
        h = (torch.arange(n) * 2654435761) % (1 << 31)
        idx = (h % m).long()
        rho = (32 - self.hll_p) - torch.floor(
            torch.log2((h >> self.hll_p).float() + 1)
        )
        reg[torch.arange(n), idx] = torch.maximum(
            reg[torch.arange(n), idx], rho
        )
        return reg

    def propagate(self, sketch, edge_index, op):
        src, dst = edge_index
        out = sketch.clone()
        if op == "min":
            out.index_reduce_(0, dst, sketch[src], "amin", include_self=True)
        else:
            out.index_reduce_(0, dst, sketch[src], "amax", include_self=True)
        return out

    def build(self, edge_index, n):
        m = [self.initialise_minhash(n)]
        h = [self.initialise_hll(n)]
        for _ in range(self.max_hops):
            m.append(self.propagate(m[-1], edge_index, "min"))
            h.append(self.propagate(h[-1], edge_index, "max"))
        return m, h

    def jaccard(self, m_u, m_v):
        return (m_u == m_v).float().mean(dim=-1)

    def cardinality(self, h):
        m = h.size(-1)
        z = (2.0 ** (-h)).sum(dim=-1)
        alpha = 0.7213 / (1 + 1.079 / m)
        return alpha * m * m / z

    def intersection(self, m_u, h_u, m_v, h_v):
        union = self.cardinality(torch.maximum(h_u, h_v))
        return self.jaccard(m_u, m_v) * union

    def structural_features(self, pairs, m, h):
        u, v = pairs
        k = self.max_hops
        inter = {}
        for du in range(1, k + 1):
            for dv in range(1, k + 1):
                inter[(du, dv)] = self.intersection(
                    m[du][u], h[du][u], m[dv][v], h[dv][v]
                )
        feats = []
        for du in range(1, k + 1):
            for dv in range(1, k + 1):
                a = inter[(du, dv)]
                for i in range(1, du + 1):
                    for j in range(1, dv + 1):
                        if (i, j) != (du, dv):
                            a = a - inter[(i, j)]
                feats.append(a.clamp(min=0))
        for d in range(1, k + 1):
            card_u = self.cardinality(h[d][u])
            feats.append((card_u - sum(feats)).clamp(min=0))
        return torch.stack(feats, dim=-1)


class BUDDY(nn.Module):
    """MLP on Hadamard node features + precomputed structural counts."""

    def __init__(self, in_channels, hidden=256, max_hops=3, num_struct=None):
        super().__init__()
        num_struct = num_struct or max_hops * (max_hops + 2)
        self.feat_lin = nn.Linear(in_channels * (max_hops + 1), hidden)
        self.struct_lin = nn.Linear(num_struct, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_u, z_v, struct_counts):
        zu = self.feat_lin(z_u)
        zv = self.feat_lin(z_v)
        had = zu * zv
        s = F.relu(self.struct_lin(struct_counts))
        return self.readout(torch.cat([had, s], dim=-1)).squeeze(-1)
```
