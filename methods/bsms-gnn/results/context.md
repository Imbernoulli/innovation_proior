# Context

## Research question

Given a physical system discretized on a mesh, can we learn a fast surrogate for one time step of its evolution — a map from the state at time *t* to the state at *t+Δt* — that runs orders of magnitude faster than a numerical solver, and that scales to large meshes with complex, irregular geometry (3D surfaces, tetrahedral solids, self-contact)?

A graph network already learns the local update on small meshes. The setting that this question is about is *scale*. Information in a PDE propagates across the whole domain, so a surrogate must move signal from any node to any other. On a mesh of N nodes with diameter D, a graph network that passes messages only along edges needs on the order of D rounds to connect the far corners, and both N and D grow with the mesh. Two effects follow:

1. **Cost.** The number of nodes to process and the number of message-passing rounds both grow with mesh size, so the time and memory of the unrolled computational graph grow super-linearly.
2. **Spectral behavior.** A graph convolution acts as a low-pass filter on the node signal. Stacking many rounds repeatedly projects the signal onto the low-frequency eigenvectors of the graph and attenuates high-frequency components.

A multi-resolution (U-Net-like) hierarchy addresses both at once: coarse levels give long-range coupling in a handful of hops, so few rounds suffice. This raises the question this work takes up: **how do you build the coarse levels of an arbitrary mesh** — a coarsening operation that runs automatically on any mesh type (2D triangles, 3D tetrahedra, 3D surfaces)?

## Background

**Learned mesh simulation as a graph problem.** A mesh becomes a graph: mesh vertices are nodes, mesh edges are graph edges, and per-node physical fields (velocity, density, position, node type) are node features. One step of simulation is one forward pass that predicts a per-node output — usually a time derivative or increment — which is integrated to give the next state. Rolling the map out autoregressively produces a trajectory. The dominant recipe is **encode–process–decode**: per-node and per-edge MLP encoders lift inputs into a latent space; a *processor* runs several rounds of message passing that update edge and node embeddings; an MLP decoder reads out the prediction. A message-passing round has the form

- edge update: e_ij ← f_E(e_ij, v_i, v_j)
- node update: v_i ← v_i + f_V(v_i, Σ_j e_ij)

with residual, LayerNorm MLPs. Because rollouts drift, the standard training trick is to supervise a single step with an L2 loss while injecting Gaussian noise into the inputs, so the model learns to correct the kind of error it will later feed itself.

**The low-pass / over-smoothing view.** Treating message passing as repeated multiplication by a normalized adjacency, the iteration converges toward the dominant (smooth) eigenvector; high-frequency components decay geometrically. This is the analytic lens through which a flat stack of rounds is understood.

**Coarsening operators known for building a hierarchy.** Two families existed. (i) **Graph-information-only** coarsening selects a subset of nodes from the graph itself. Dropping nodes can disconnect the survivors. A known mitigation is *adjacency enhancement* by graph powers: replace the adjacency A by A^K, where A^K(i,j), read as a boolean, is 1 exactly when j is reachable from i within K hops; raising K rebuilds links between survivors that a drop would have severed. (ii) **Geometry-based** coarsening places coarse nodes on a grid or by spatial clustering and connects them by Euclidean proximity.

**Adjacency powers as enhancement.** As K grows, the booleanized A^K fills in: in the limit it becomes the all-ones matrix, i.e. a fully connected graph, and then a single convolution averages every node into the same vector. K=1 leaves only original survivor-survivor edges; on a path kept-dropped-kept, the two survivors are disconnected. So a small power bridges across one removed node, while larger K connects more broadly. The graph-information methods used A^2 in this regime.

## Baselines

**MeshGraphNets (Pfaff et al., 2020).** The single-scale backbone. Builds a graph from the mesh, adds dynamic *world edges* for contact/collision, and runs encode–process–decode with K (typically 5–20) message-passing rounds on a persistent edge-and-node latent; the decoder outputs a per-node acceleration/increment integrated to the next state; trained on single-step L2 with input noise. It is flat: connecting distant parts of a large mesh is done by message passing along edges over many rounds. It is the reference point on scale.

**Graph U-Nets (Gao & Ji, 2019).** Brought the U-Net (encode → pool → … → unpool → decode) to graphs. Pooling (*gPool*) learns a projection vector p, scores each node as x_i·p/‖p‖, and keeps the top-k by score; *gUnpool* restores dropped positions with zeros. It replaces A with A^2 between levels to reconnect survivors after top-k. The original implementation used dense adjacency multiplications on ~100-node graphs.

**Spatial-proximity multi-scale GNNs (Lino et al., 2021/2022; Liu et al., 2021).** Build coarse levels by laying down coarser nodes (a rasterized grid or spatial clustering) and connecting them by Euclidean radius, with learnable down/up transitions between levels and several message-passing rounds per level. Per-case knobs include grid resolution and the inflation rate between levels.

**Manually drawn coarse meshes (Liu et al., 2021; Fortunato et al., 2022).** Two- or multi-level message-passing networks whose coarse meshes are drawn by hand in CAE software, with a learnable per-level transition; one coarse mesh per trajectory.

**Guillard coarsening / multipole (Lino et al., 2022; Li et al., 2020).** Guillard node-nesting coarsens 2D triangle meshes; the multipole approach pools nodes randomly and learns coarse kernels by multi-level matrix factorization.

## Evaluation settings

**Datasets.** Three public mesh-simulation trajectory sets — a 2D incompressible flow past a cylinder, a 2D compressible flow past an airfoil, a 3D hyperelastic plate deformed by an actuator (tetrahedral, with contact) — plus a more demanding 3D set of inflating enclosed elastic surfaces with self-contact (triangular surface meshes, up to roughly 47k nodes, with many contact edges). Each set is split 1000/200/200 into train/validation/test trajectories. Meshes span triangles, tetrahedra, and surfaces, Eulerian and Lagrangian systems, with per-node fields such as velocity, density, world/material position, and node type, and dynamically built contact edges where collisions occur.

**Protocol.** Train by supervising the single-step L2 loss between predicted and ground-truth nodal output, with Gaussian input noise per epoch; evaluate by autoregressive rollout. All variables normalized to zero mean and unit variance.

**Metrics.** Rollout RMSE at several horizons (one step, fifty steps, full trajectory); and efficiency: training time per step, inference time per step, total training cost (wall-clock hours and epochs to a target accuracy), and peak training and inference memory across batch sizes. Hardware is a single consumer GPU. Baselines for comparison are the flat backbone, a spatial-proximity hierarchy, and a learnable-pooling hierarchy.

## Code framework

The primitives that already exist: a mesh-to-graph conversion, a graph data structure with BFS, residual LayerNorm MLPs, a scatter-add for message aggregation, an autoregressive rollout, and a single-step L2 training loop with input noise. The coarsening operation and the processor that uses it are still open slots.

```python
import torch
import torch.nn as nn
import numpy as np

# ---- existing primitives ----------------------------------------------------

def scatter_sum(src, index, dim, dim_size):
    """Sum messages into their target nodes (already available)."""
    out_shape = list(src.shape); out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    return out.index_add_(dim, index, src)

class MLP(nn.Module):
    """Residual-friendly MLP with optional output LayerNorm (already available)."""
    def __init__(self, in_dim, latent_dim, out_dim, hidden_layers, layer_norm=True):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(hidden_layers):
            mods += [nn.Linear(d, latent_dim), nn.ReLU()]; d = latent_dim
        mods += [nn.Linear(d, out_dim)]
        if layer_norm:
            mods += [nn.LayerNorm(out_dim, elementwise_affine=False)]
        self.seq = nn.Sequential(*mods)
    def forward(self, x):
        return self.seq(x)

class MessagePassing(nn.Module):
    """One round of edge->node message passing on a single graph (already available)."""
    def __init__(self, latent_dim, hidden_layers, pos_dim):
        super().__init__()
        self.edge_mlp = MLP(2 * latent_dim + pos_dim + 1, latent_dim, latent_dim, hidden_layers)
        self.node_mlp = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layers)
    def forward(self, x, edges, pos):
        i, j = edges[0], edges[1]
        offset = pos[i] - pos[j]
        fiber = torch.cat([offset, offset.norm(dim=-1, keepdim=True)], dim=-1)
        e = self.edge_mlp(torch.cat([fiber, x[i], x[j]], dim=-1))
        aggr = scatter_sum(e, j, dim=-2, dim_size=x.shape[-2])
        return x + self.node_mlp(torch.cat([x, aggr], dim=-1))

# ---- open design slots -------------------------------------------------------

def build_coarse_levels(flat_edges, num_levels, num_nodes, pos):
    """Turn one mesh-graph into a stack of progressively coarser graphs.

    Must return, per level: the node subset kept and the (re-enhanced,
    re-indexed) edges of the coarser graph. The selection rule and the
    edge rebuild are the open problem.
    """
    # TODO: choose which nodes survive at each coarser level
    # TODO: rebuild coarse-level edges
    raise NotImplementedError

class LevelTransition(nn.Module):
    """Move latent features between two adjacent levels (down and up)."""
    def __init__(self):
        super().__init__()
    def down(self, x, edges):
        # TODO: carry fine-level information onto the coarser level
        raise NotImplementedError
    def up(self, x, edges, pre_num_nodes, kept_idx):
        # TODO: return coarse-level information to the finer level
        raise NotImplementedError

class HierarchicalProcessor(nn.Module):
    """The multi-level processor that replaces a deep flat message-passing stack."""
    def __init__(self, depth, latent_dim, hidden_layers, pos_dim):
        super().__init__()
        # TODO: per-level message passing + transitions arranged as an encoder/decoder hierarchy
        raise NotImplementedError
    def forward(self, x, kept_ids, level_edges, pos):
        # TODO: down pass (process, transition down, pool), bottom, up pass (unpool, transition up, process, skip)
        raise NotImplementedError

class Simulator(nn.Module):
    """Encode -> hierarchical process -> decode, integrated to the next state."""
    def __init__(self, cfg):
        super().__init__()
        self.encode = MLP(cfg.in_dim, cfg.latent_dim, cfg.latent_dim, cfg.hidden_layers, True)
        self.process = HierarchicalProcessor(cfg.depth, cfg.latent_dim, cfg.hidden_layers, cfg.pos_dim)
        self.decode = MLP(cfg.latent_dim, cfg.latent_dim, cfg.out_dim, cfg.hidden_layers, False)
    def forward(self, node_in, kept_ids, level_edges, pos):
        x = self.encode(node_in)
        x = self.process(x, kept_ids, level_edges, pos)
        return self.decode(x)

# ---- existing training / rollout harness ------------------------------------

def train_step(model, batch, optim):
    node_in, target, mask, kept_ids, level_edges, pos = batch
    node_in = node_in + torch.randn_like(node_in) * NOISE_SCALE   # rollout-noise trick
    pred = model(node_in, kept_ids, level_edges, pos)
    loss = (((pred - target) * mask) ** 2).mean()
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.item()
```
