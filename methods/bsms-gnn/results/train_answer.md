The goal is to learn a fast surrogate for a single time step of a mesh-based physical simulation: given per-node state at time t, predict the state at t+Δt, then roll the model forward autoregressively. On small meshes a flat encode–process–decode graph network already works well enough. The real difficulty is scale. Information in a PDE propagates across the whole domain, so the network must couple distant nodes. In a message-passing graph neural network, signal travels one edge per round, so connecting nodes D hops apart needs on the order of D rounds. As the mesh grows, both the number of nodes N and the graph diameter D grow, which makes training cost super-linear and, worse, progressively smears the signal because graph convolution is a low-pass filter. Deep stacks project the node features onto the smooth eigenvectors of the graph and erase the sharp, high-frequency structures that simulations need. Simply adding more layers makes both problems worse.

A multi-resolution hierarchy is the natural escape: coarse levels give long-range coupling in a handful of hops, so few rounds suffice and the signal is never smoothed to death. But that only moves the difficulty into the coarsening operation. A usable coarsening must preserve connectivity at every coarse level, never invent edges that jump across geometric boundaries, work for arbitrary mesh types, and run automatically without hand-drawn coarse meshes. Learnable top-k pooling fails the connectivity guarantee and generalizes poorly to unseen geometries. Spatial-proximity coarsening keeps things connected but draws edges across gaps and near-contact regions that are physically disconnected. Hand-drawn meshes are accurate but not scalable. What is needed is a deterministic, topology-only pooling rule that is provably connection-conservative.

I propose BSMS-GNN, the Bi-Stride Multi-Scale Graph Neural Network. The core idea is bi-stride pooling. Run breadth-first search from a seed in each connected component and label every node by its geodesic hop distance. Then keep all nodes on every other BFS frontier, choosing the smaller of the even and odd parity sets so roughly half the nodes survive at each level. The BFS layering guarantees that every dropped node has at least one kept material neighbor: a dropped odd-depth node has an even-depth BFS parent, and a dropped even-depth node has an odd-depth parent, with the seed handled as a special case. Therefore every dropped node is only one hop away from a survivor. This makes the pooling second-order connection-conservative, or 2-CC, by construction: after rebuilding the coarse edges with the squared self-looped adjacency, no survivor is stranded and no node is more than two hops from the kept set. K=2 is the smallest useful enhancement order; K=1 leaves kept-dropped-kept paths severed, while larger K over-connects and approaches the all-ones matrix, which is maximal over-smoothing. Because the rule uses only graph topology, it never consults spatial distance and therefore cannot draw a boundary-crossing edge. It is automatic, general across triangles, tetrahedra, and surfaces, and provably connected.

The coarse graph is built from the self-looped material adjacency tilde A_l = A_l + I as A_{l+1} = clear_diag((tilde A_l tilde A_l)[kept, kept]). The diagonal before squaring preserves direct survivor-survivor edges as well as two-hop bridges; the final diagonal clear removes self-loops introduced by the square. Contact edges, which are legitimate spatial-proximity interactions for collision and self-contact, are kept in a separate matrix A^C and carried to coarser levels by tilde A_l A^C_l tilde A_l restricted to the survivors. Each endpoint of a fine contact either survives and uses its self-loop or steps one material hop to a kept neighbor, so no contact is lost. Since pooled and unpooled nodes remain directly connected on the fine graph, a single message-passing round per level is enough to move information across the hierarchy.

The architecture is a lightweight U-Net. The encoder and decoder live only at the finest level. The processor runs one graph message-passing round at each level, using the relative edge offset and its norm as the only geometric input so no persistent edge latent is needed. Transitions between levels are non-parametric and weighted: each node carries a weight initialized to one at the finest level and aggregated down. The downsample forms a contribution table by row-normalizing the adjacency, weighting by sender weights, and normalizing over incoming edges so the shares sum to one. The upsample uses the transpose of the same contribution table, spreading coarse values back to finer nodes. This avoids the mosaic artifacts of zero-filled unpooling and the over-smoothing of an unweighted neighborhood average on irregular element sizes. Training uses a single-step L2 loss with Gaussian noise injected into the inputs so the model learns to correct its own rollout drift.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm

_INF = 1 + 1e10

def scatter_sum(src, index, dim, dim_size):
    shape = list(src.shape)
    shape[dim] = dim_size
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
            mods += [Linear(d, latent_dim), ReLU()]
            d = latent_dim
        mods += [Linear(d, out_dim)]
        if layer_norm:
            mods += [LayerNorm(out_dim, elementwise_affine=False)]
        self.seq = Seq(*mods)
    def forward(self, x):
        return self.seq(x)

def nearest_center_seed(pos, clusters):
    seeds = []
    for c in clusters:
        center = pos[c].mean(0)
        seeds.append(c[int(np.argmin(np.linalg.norm(pos[c] - center, axis=-1)))])
    return seeds

def bstride_selection(graph, pos, num_nodes):
    """BFS-parity pooling -> 2-CC; rebuild coarse edges with (A+I)^2."""
    adj = graph.get_sparse_adj_mat()
    adj.setdiag(1)
    seeds = nearest_center_seed(pos, graph.clusters)
    kept = set()
    for seed in seeds:
        dist = graph.bfs_dist(seed)
        even = {i for i, d in enumerate(dist) if d != _INF and d % 2 == 0}
        odd = {i for i, d in enumerate(dist) if d != _INF and d % 2 == 1}
        kept |= even if (len(even) <= len(odd) or not odd) else odd
    kept = sorted(kept)
    adj = adj.tocsr().astype(float)
    adj = adj @ adj
    adj.setdiag(0)
    return kept, pool_edge(adj, num_nodes, kept)

class GMP(nn.Module):
    def __init__(self, latent_dim, hidden_layer, pos_dim):
        super().__init__()
        self.mlp_edge = MLP(2 * latent_dim + pos_dim + 1, latent_dim, latent_dim, hidden_layer)
        self.mlp_node = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer)
    def forward(self, x, g, pos):
        i, j = g[0], g[1]
        offset = pos[i] - pos[j]
        fiber = torch.cat([offset, offset.norm(dim=-1, keepdim=True)], dim=-1)
        e = self.mlp_edge(torch.cat([fiber, x[i], x[j]], dim=-1))
        aggr = scatter_sum(e, j, dim=-2, dim_size=x.shape[-2])
        return x + self.mlp_node(torch.cat([x, aggr], dim=-1))

class WeightedEdgeConv(nn.Module):
    @torch.no_grad()
    def cal_ew(self, w, g):
        i, j = g[0], g[1]
        normed_w = w.squeeze(-1) / degree(i, w.shape[0])
        w_to_send = normed_w[i]
        aggr_w = scatter_sum(w_to_send, j, dim=-1, dim_size=normed_w.size(0)) + 1e-12
        return w_to_send / aggr_w[j], aggr_w
    def forward(self, x, g, ew, aggregating=True):
        i, j = g[0], g[1]
        src = x[i] if aggregating else x[j]
        msg = src * ew.unsqueeze(-1)
        tgt = j if aggregating else i
        return scatter_sum(msg, tgt, dim=-2, dim_size=x.shape[-2])

class Unpool(nn.Module):
    def forward(self, h, pre_num_nodes, idx):
        new_h = h.new_zeros([pre_num_nodes, h.shape[-1]])
        new_h[idx] = h
        return new_h

class BSGMP(nn.Module):
    def __init__(self, depth, latent_dim, hidden_layer, pos_dim):
        super().__init__()
        self.depth = depth
        self.down_gmps = nn.ModuleList(GMP(latent_dim, hidden_layer, pos_dim) for _ in range(depth))
        self.up_gmps = nn.ModuleList(GMP(latent_dim, hidden_layer, pos_dim) for _ in range(depth))
        self.unpools = nn.ModuleList(Unpool() for _ in range(depth))
        self.bottom_gmp = GMP(latent_dim, hidden_layer, pos_dim)
        self.edge_conv = WeightedEdgeConv()
    def forward(self, h, m_ids, m_gs, pos):
        down_outs, down_ps, cts = [], [], []
        w = pos.new_ones((pos.shape[-2], 1))
        for i in range(self.depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h)
            down_ps.append(pos)
            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h = self.edge_conv(h, m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)
            h, pos, w = h[m_ids[i]], pos[m_ids[i]], w[m_ids[i]]
        h = self.bottom_gmp(h, m_gs[self.depth], pos)
        for i in range(self.depth):
            d = self.depth - i - 1
            h = self.unpools[i](h, down_outs[d].shape[-2], m_ids[d])
            h = self.edge_conv(h, m_gs[d], cts[d], aggregating=False)
            h = self.up_gmps[i](h, m_gs[d], down_ps[d])
            h = h + down_outs[d]
        return h

class BSMS_Simulator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encode = MLP(cfg.out_dim + 1, cfg.latent_dim, cfg.latent_dim, cfg.hidden_layer, True)
        self.process = BSGMP(cfg.unet_depth, cfg.latent_dim, cfg.hidden_layer, cfg.pos_dim)
        self.decode = MLP(cfg.latent_dim, cfg.latent_dim, cfg.out_dim, cfg.hidden_layer, False)
    def forward(self, node_feat, m_ids, m_gs, pos):
        x = self.encode(node_feat)
        x = self.process(x, m_ids, m_gs, pos)
        delta = self.decode(x)
        return node_feat[..., :delta.shape[-1]] + delta
```
