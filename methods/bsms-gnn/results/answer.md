# BSMS-GNN: Bi-Stride Multi-Scale Graph Neural Network

## What it is

BSMS-GNN is a hierarchical (U-Net-style) graph network for learning one step of a mesh-based physical simulation — mapping the nodal state at time *t* to the state at *t+Δt* — that scales to large, irregular, complex-geometry meshes. Its core is **bi-stride pooling**: a deterministic, topology-only coarsening that is provably second-order connection-conservative, plus a lightweight processor (one message-passing round per level and a non-parametric weighted transition) that the pooling makes possible.

## The problem it solves

A flat graph network must run on the order of *graph-diameter* message-passing rounds to couple distant parts of a large mesh. That is both super-linearly expensive in time/memory and over-smoothing (message passing is a low-pass filter; deep stacks erase high-frequency detail). A multi-scale hierarchy fixes both, but only if coarse levels can be built that (a) preserve connectivity, (b) never link across geometric boundaries, (c) work for any mesh type, and (d) are fully automatic. Prior coarsening — learnable top-k pooling, spatial-proximity grids, hand-drawn coarse meshes — each fails at least one. Bi-stride satisfies all four.

## Key idea: bi-stride pooling

Run BFS from a seed and label every node by its geodesic hop-distance. **Keep all nodes on every other BFS frontier** (all even depths, or all odd — keep the smaller set for a ~1/2 pooling ratio). For any edge (i,j), BFS gives depth(j) <= depth(i)+1 and depth(i) <= depth(j)+1, hence |depth(i)-depth(j)| <= 1. Same-depth edges may exist, so depth parity is not a proper graph coloring; the required guarantee is weaker: every dropped node has at least one kept material neighbor. If evens are kept, a dropped odd node has an even BFS parent. If odds are kept, a dropped even node with nonzero depth has an odd parent; a dropped seed has level-1 kept neighbors, and a singleton component keeps the even seed because the odd set is empty. Hence:

- **2-CC (second-order connection conservative) by construction.** Define the K-th-order outlier set O_K = { j : A^K(i,j)=0 for all kept i } (nodes more than K hops from every survivor). Bi-stride gives O_2 = empty because every dropped node is one hop from a survivor and every survivor is itself retained. K=2 is the first useful enhancement for reconnecting survivors across one removed node: K=1 leaves only original survivor-survivor edges, so a kept-dropped-kept path is severed; larger K drives A^K toward all-ones (a single convolution then averages all nodes: maximal over-smoothing).
- **Topology-only**, so it never draws a boundary-crossing edge; **general** across triangles/tetrahedra/surfaces; **automatic**.
- **Pooled and unpooled nodes stay directly connected**, so a single MP per level suffices.

**Coarse graph:** let \tilde A_l = A_l + I be the material adjacency with unit diagonal. The implemented rebuild is
A_{l+1} = clear_diag((\tilde A_l \tilde A_l)[𝓘, 𝓘]).
The diagonal before squaring keeps direct survivor-survivor material edges as well as two-hop pairs; the diagonal clear removes self-loops introduced by the square.

**Contact edges** (proximity edges for collision/self-contact, kept in a separate matrix A^C) are carried as
A^C_{l+1} = (\tilde A_l A^C_l \tilde A_l)[𝓘, 𝓘]. A coarse contact exists when a fine contact can be lifted to surviving material nodes: each endpoint either survives and uses its diagonal self-loop, or moves one material hop to a kept neighbor. The four cases are: both endpoints kept (both diagonal terms), only the left kept (left diagonal, right kept neighbor), only the right kept (left kept neighbor, right diagonal), and neither kept (kept neighbor on both sides). In each case the product contains a 1*1*1 term, so no fine contact edge is lost.

**Seeding:** one BFS seed per connected component. *MinAve* picks the node with minimum average geodesic distance (O(N^2)); *CloseCenter* picks the node nearest the spatial centroid (O(N), the linear fallback for very large meshes). Every seed preserves the 2-CC guarantee; the deterministic choice is for balanced layers and reproducible preprocessing.

## Architecture

- **Encode–process–decode**, encoder/decoder only at the finest level. MLPs are ReLU, residual, latent width 128, with LayerNorm on all outputs except the decoder.
- **One MP per level.** The edge offset is not separately encoded: prepend Δx_ij = x_i − x_j and ‖Δx_ij‖ to the stacked node latents.
  - e^s_ij ← f^s_l(Δx_ij, v_i, v_j) for each edge set s (material + contact)
  - v'_i ← v_i + f^V_l(v_i, Σ_j e^1_ij, …, Σ_j e^S_ij)
- **Non-parametric weighted transition.** Maintain a nodal weight w (1 at finest, aggregated down). Row-normalize sender edges Â_ij = A_ij/Σ_j A_ij; form ŵ_ij = w_i Â_ij; then normalize over incoming senders for each receiver, C_ij = ŵ_ij/Σ_i ŵ_ij, so Σ_i C_ij = 1 for fixed j. Down: v_j ← Σ_i v_i C_ij. Up: v_i ← Σ_j v_j C_ij, the transpose use of the same contribution table. This avoids plain unpooling's zero-filled mosaic artifacts and the over-smoothing of an unweighted graph convolution on irregular meshes without adding learnable transition modules.
- **U-Net loop:** per level (MP → save skip → weighted downsample → pool), bottom MP, then per level in reverse (unpool → weighted return → MP → add skip), decode the increment, integrate to the next state.

**Training:** single-step L2 loss with per-epoch Gaussian input noise; autoregressive rollout at inference. Multi-level structure is precomputed deterministically in one pass.

## Code

```python
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm

_INF = 1 + 1e10

# ----- utilities -----
def scatter_sum(src, index, dim, dim_size):
    shape = list(src.shape); shape[dim] = dim_size
    out = src.new_zeros(shape)
    return out.index_add_(dim, index, src)

def degree(index, num_nodes):
    out = torch.zeros(num_nodes, dtype=torch.float, device=index.device)
    return out.index_add_(0, index, torch.ones_like(index, dtype=torch.float))

class MLP(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, hidden_layers, layer_norm=True):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(hidden_layers):
            mods += [Linear(d, latent_dim), ReLU()]; d = latent_dim
        mods += [Linear(d, out_dim)]
        if layer_norm:
            mods += [LayerNorm(out_dim, elementwise_affine=False)]
        self.seq = Seq(*mods)
    def forward(self, x):
        return self.seq(x)

# ----- bi-stride pooling / multi-level preprocessing -----
def bstride_selection(graph, pos, num_nodes):
    """BFS-frontier-parity pooling -> 2-CC; rebuild coarse edges with (A+I)^2."""
    adj = graph.get_sparse_adj_mat(); adj.setdiag(1)      # self-looped material adjacency
    seeds = nearest_center_seed(pos, graph.clusters)      # one seed per connected component
    kept = set()
    for seed, c in zip(seeds, graph.clusters):
        dist = graph.bfs_dist(seed)
        even = {i for i, d in enumerate(dist) if d != _INF and d % 2 == 0}
        odd  = {i for i, d in enumerate(dist) if d != _INF and d % 2 == 1}
        kept |= even if (len(even) <= len(odd) or not odd) else odd   # smaller set -> balanced
    kept = sorted(kept)
    adj = adj.tocsr().astype(float)
    adj = adj @ adj; adj.setdiag(0)                       # direct and two-hop edges, no self-loops
    return kept, pool_edge(adj, num_nodes, kept)          # restrict to [kept, kept] and reindex

def nearest_center_seed(pos, clusters):
    seeds = []
    for c in clusters:
        center = pos[c].mean(0)
        seeds.append(c[int(np.argmin(np.linalg.norm(pos[c] - center, axis=-1)))])
    return seeds

# ----- one message-passing round -----
class GMP(nn.Module):
    def __init__(self, latent_dim, hidden_layer, pos_dim):
        super().__init__()
        self.mlp_edge = MLP(2*latent_dim + pos_dim + 1, latent_dim, latent_dim, hidden_layer)
        self.mlp_node = MLP(2*latent_dim, latent_dim, latent_dim, hidden_layer)
    def forward(self, x, g, pos):
        i, j = g[0], g[1]
        offset = pos[i] - pos[j]                                    # relative geometry only
        fiber  = torch.cat([offset, offset.norm(dim=-1, keepdim=True)], dim=-1)
        e = self.mlp_edge(torch.cat([fiber, x[i], x[j]], dim=-1))   # e_ij = f(dx, v_i, v_j)
        aggr = scatter_sum(e, j, dim=-2, dim_size=x.shape[-2])
        return x + self.mlp_node(torch.cat([x, aggr], dim=-1))      # residual node update

# ----- weighted, non-parametric transition -----
class WeightedEdgeConv(nn.Module):
    @torch.no_grad()
    def cal_ew(self, w, g):
        i, j = g[0], g[1]
        normed_w = w.squeeze(-1) / degree(i, w.shape[0])                # row-normalized sender weights
        w_to_send = normed_w[i]                                          # w_hat_ij
        aggr_w = scatter_sum(w_to_send, j, dim=-1, dim_size=normed_w.size(0)) + 1e-12
        return w_to_send / aggr_w[j], aggr_w                            # C_ij, aggregated weight
    def forward(self, x, g, ew, aggregating=True):
        i, j = g[0], g[1]
        src = x[i] if aggregating else x[j]                            # down vs up (C vs C^T)
        msg = src * ew.unsqueeze(-1)
        tgt = j if aggregating else i
        return scatter_sum(msg, tgt, dim=-2, dim_size=x.shape[-2])

class Unpool(nn.Module):
    def forward(self, h, pre_num_nodes, idx):
        new_h = h.new_zeros([pre_num_nodes, h.shape[-1]])
        new_h[idx] = h
        return new_h

# ----- hierarchical processor (the U-Net) -----
class BSGMP(nn.Module):
    def __init__(self, depth, latent_dim, hidden_layer, pos_dim):
        super().__init__()
        self.depth = depth
        self.down_gmps = nn.ModuleList(GMP(latent_dim, hidden_layer, pos_dim) for _ in range(depth))
        self.up_gmps   = nn.ModuleList(GMP(latent_dim, hidden_layer, pos_dim) for _ in range(depth))
        self.unpools   = nn.ModuleList(Unpool() for _ in range(depth))
        self.bottom_gmp = GMP(latent_dim, hidden_layer, pos_dim)
        self.edge_conv  = WeightedEdgeConv()
    def forward(self, h, m_ids, m_gs, pos):
        down_outs, down_ps, cts = [], [], []
        w = pos.new_ones((pos.shape[-2], 1))
        for i in range(self.depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h); down_ps.append(pos)
            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h   = self.edge_conv(h,   m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)
            h, pos, w = h[m_ids[i]], pos[m_ids[i]], w[m_ids[i]]        # pool survivors
        h = self.bottom_gmp(h, m_gs[self.depth], pos)
        for i in range(self.depth):
            d = self.depth - i - 1
            h = self.unpools[i](h, down_outs[d].shape[-2], m_ids[d])
            h = self.edge_conv(h, m_gs[d], cts[d], aggregating=False)  # weighted return (C^T)
            h = self.up_gmps[i](h, m_gs[d], down_ps[d])
            h = h + down_outs[d]                                        # skip
        return h

# ----- full simulator -----
class BSMS_Simulator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # encoder sees nodal fields + node type (cfg.out_dim + 1); mesh_pos is split off for the MP blocks
        self.encode  = MLP(cfg.out_dim + 1, cfg.latent_dim, cfg.latent_dim, cfg.hidden_layer, True)
        self.process = BSGMP(cfg.unet_depth, cfg.latent_dim, cfg.hidden_layer, cfg.pos_dim)
        self.decode  = MLP(cfg.latent_dim, cfg.latent_dim, cfg.out_dim,  cfg.hidden_layer, False)
    def forward(self, node_feat, m_ids, m_gs, pos):
        x = self.encode(node_feat)
        x = self.process(x, m_ids, m_gs, pos)
        delta = self.decode(x)
        return node_feat[..., :delta.shape[-1]] + delta
```

(`Graph` provides `get_sparse_adj_mat`, `bfs_dist`, `clusters`; `pool_edge` restricts the squared self-looped adjacency to the kept indices and reindexes them to 0..|kept|-1. A full training setup wraps this path with input/target normalizers, masks invalid nodes, predicts a normalized increment, inverts the target normalizer, and integrates the increment.)
