# BUDDY / ELPH — Link prediction with subgraph sketching

**Problem.** Plain message-passing GNNs cannot represent the neighborhood-overlap signal that decides
links (they give automorphic nodes identical embeddings and cannot count triangles / common
neighbors). Subgraph GNNs (SEAL) fix this by extracting and labeling a k-hop enclosing subgraph *per
candidate edge*, but that is O(|E|) per edge on power-law graphs and overflows memory. The goal is
subgraph-GNN expressiveness at GCN-class cost.

**Key idea.** The load-bearing content of a labeled enclosing subgraph is the *distance-label count
table* A_{uv}[d_u, d_v] = #{nodes at distance exactly d_u from u and d_v from v} (with A[1,1] = common
neighbors / triangles) plus boundary counts B_{uv}[d]. These reduce to two per-node primitives:
neighborhood cardinality |N_d(u)| and neighborhood intersection |N_{d_u}(u) ∩ N_{d_v}(v)|. Estimate
both with sketches: **HyperLogLog** gives cardinality (union sketch = elementwise max), **MinHash**
gives Jaccard (union sketch = elementwise min), and |S ∩ T| = J(S,T)·|S ∪ T| ≈
H(MinHash(S), MinHash(T)) · card(max(HLL(S), HLL(T))). Propagating sketches by elementwise min/max for
k hops (message passing with sketch messages) gives every node's d-hop neighborhood sketches once.

**Predictor.** p(u, v) = ψ( z_u ⊙ z_v , {B_{uv}[d], A_{uv}[d_u, d_v] : d, d_u, d_v ≤ k} ): Hadamard of
(multi-scale, precomputable) node features concatenated with the structural counts, through an MLP ψ.
**ELPH** computes sketches in the message-passing loop (full graph in memory). **BUDDY** precomputes
the sketches (scatter_min / scatter_max) and the fixed feature propagation (scatter_mean), caching the
per-edge features so training/inference is *just an MLP* — memory independent of graph size.

**Why it works.** The counts are pair-relative, so they separate automorphic-node links a GNN cannot;
with exact estimates the model provably avoids the automorphic-node problem and is strictly more
expressive than message-passing GNNs. Most predictive content is in the *low*-distance counts, so an
MLP directly on the counts (k = 2–3) matches propagating them, and fixed feature propagation ≈ learned.

**Hyperparameters.** receptive field k = 2–3; MinHash with ~128 permutations; HyperLogLog precision
p ≈ 8; hidden dim 256; multi-scale feature concat over hops; MLP readout with edge (Hadamard) pooling.

**Measured (HR@100 unless noted).** Cora 88.00, CiteSeer 92.93, Pubmed
74.10, ogbl-collab HR@50 65.94, ogbl-ppa HR@100 49.85, ogbl-citation2 MRR 87.56, ogbl-ddi HR@20 78.51
— best or second on five of seven benchmarks, with SEAL the closest competitor and BUDDY 200–1000×
faster.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

# Canonical BUDDY: precompute sketches + propagated features, then MLP on edge features.

class ElphHashes:
    """Per-node MinHash + HyperLogLog sketches, propagated by min/max for k hops."""
    def __init__(self, num_perm=128, hll_p=8, max_hops=3):
        self.num_perm = num_perm          # MinHash permutations
        self.hll_p = hll_p                # HyperLogLog precision (2**p registers)
        self.max_hops = max_hops

    def initialise_minhash(self, n):
        # random permutation hashes of singleton {node}
        a = torch.randint(1, (1 << 31) - 1, (self.num_perm,))
        b = torch.randint(0, (1 << 31) - 1, (self.num_perm,))
        ids = torch.arange(n).unsqueeze(1)
        return ((a * ids + b) % ((1 << 31) - 1)).float()      # [n, num_perm]

    def initialise_hll(self, n):
        m = 1 << self.hll_p
        reg = torch.zeros(n, m)
        # singleton register update from a hash of the node id (rho leading zeros)
        h = (torch.arange(n) * 2654435761) % (1 << 31)
        idx = (h % m).long()
        rho = (32 - self.hll_p) - torch.floor(torch.log2((h >> self.hll_p).float() + 1))
        reg[torch.arange(n), idx] = torch.maximum(reg[torch.arange(n), idx], rho)
        return reg

    def propagate(self, sketch, edge_index, op):
        # one hop of elementwise min (MinHash) or max (HLL) over neighbours
        src, dst = edge_index
        out = sketch.clone()
        if op == 'min':
            out.index_reduce_(0, dst, sketch[src], 'amin', include_self=True)
        else:
            out.index_reduce_(0, dst, sketch[src], 'amax', include_self=True)
        return out

    def build(self, edge_index, n):
        m = [self.initialise_minhash(n)]
        h = [self.initialise_hll(n)]
        for _ in range(self.max_hops):
            m.append(self.propagate(m[-1], edge_index, 'min'))
            h.append(self.propagate(h[-1], edge_index, 'max'))
        return m, h                                            # sketches per hop

    def jaccard(self, m_u, m_v):
        return (m_u == m_v).float().mean(dim=-1)               # Hamming sim ~ Jaccard

    def cardinality(self, h):
        m = h.size(-1)
        z = (2.0 ** (-h)).sum(dim=-1)
        alpha = 0.7213 / (1 + 1.079 / m)
        return alpha * m * m / z                               # HLL estimate

    def intersection(self, m_u, h_u, m_v, h_v):
        union = self.cardinality(torch.maximum(h_u, h_v))
        return self.jaccard(m_u, m_v) * union                  # |S∩T| = J*|S∪T|

    def structural_features(self, edge_index_pairs, m, h):
        """Distance-label counts A[d_u,d_v] and boundary B[d] for each pair.

        Schematic: the canonical impl carves k(k+2) features — the strict
        two-sided inclusion-exclusion A[d_u,d_v] = C[d_u,d_v] - C[d_u-1,d_v]
        - C[d_u,d_v-1] + C[d_u-1,d_v-1] plus two-sided boundary counts
        B[0,d] and B[d,0]. The loop below is the readable shell-subtraction
        sketch of that arithmetic, not the exact two-sided form."""
        u, v = edge_index_pairs
        k = self.max_hops
        feats = []
        # A[d_u,d_v] = |N_{d_u}(u) ∩ N_{d_v}(v)| minus nearer-shell counts (incl-excl)
        inter = {}
        for du in range(1, k + 1):
            for dv in range(1, k + 1):
                inter[(du, dv)] = self.intersection(m[du][u], h[du][u],
                                                    m[dv][v], h[dv][v])
        for du in range(1, k + 1):
            for dv in range(1, k + 1):
                a = inter[(du, dv)]
                for i in range(1, du + 1):
                    for j in range(1, dv + 1):
                        if (i, j) != (du, dv):
                            a = a - inter[(i, j)]
                feats.append(a.clamp(min=0))
        # boundary counts B[d] = |N_d(u)| - within-window mass (one-sided)
        for d in range(1, k + 1):
            card_u = self.cardinality(h[d][u])
            feats.append((card_u - sum(feats)).clamp(min=0))
        return torch.stack(feats, dim=-1)                      # [E, k(k+2)]


class BUDDY(nn.Module):
    """MLP on Hadamard node features + precomputed structural counts."""
    def __init__(self, in_channels, hidden=256, max_hops=3, num_struct=None):
        super().__init__()
        num_struct = num_struct or max_hops * (max_hops + 2)
        self.feat_lin = nn.Linear(in_channels * (max_hops + 1), hidden)
        self.struct_lin = nn.Linear(num_struct, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_u, z_v, struct_counts):
        # z_u, z_v: multi-scale propagated node features [E, in*(k+1)]
        zu = self.feat_lin(z_u)
        zv = self.feat_lin(z_v)
        had = zu * zv                                          # edge (Hadamard) pooling
        s = F.relu(self.struct_lin(struct_counts))
        return self.readout(torch.cat([had, s], dim=-1)).squeeze(-1)
```
