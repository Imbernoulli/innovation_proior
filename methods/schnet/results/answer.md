# SchNet, distilled

SchNet is a rotation/translation/permutation-invariant deep network for atomistic systems
built around the **continuous-filter convolution (cfconv)**: instead of a filter tensor
indexed by integer grid offsets, a small neural network *generates* the convolution filter as
a function of the continuous interatomic distance, so atoms at arbitrary positions can be
convolved without discretizing space. Stacked in residual interaction blocks with a smooth
(C^∞) activation, it produces a scalar energy whose gradient gives curl-free, energy-
conserving, equivariant forces — and the same encoder, fed node feature vectors and a spatial
neighbor graph, produces invariant per-node and per-graph embeddings for other geometric
tasks such as protein structure representation.

## Problem it solves

Predict the energy `E(r_1,…,r_n)` of a molecule (atom types `Z_i`, positions `r_i ∈ R^3`) and
its forces `F_i = -∂E/∂r_i` from the same model, generalizing across chemical and
conformational variation, while respecting three hard constraints: (1) energy invariant to
rotation/translation/atom-indexing and forces equivariant; (2) the force field curl-free /
energy-conserving; (3) the potential-energy surface smooth and twice-differentiable (so
forces and a force-based loss are well-defined).

## Key ideas

- **Continuous-filter convolution.** Replace the grid-indexed filter `W[Δ]` by a filter-
  generating network `W(·)` of the continuous offset, and convolve over neighbors:
  `x_i^{l+1} = Σ_j x_j^l ∘ W(r_i - r_j)`, with `∘` an elementwise (feature-wise) product. No
  grid, so an atom's contribution moves continuously as it moves — no binning discontinuity.
  Feature-wise filtering keeps it `O(F)` per edge; cross-feature mixing is done by separate
  atom-wise dense layers.
- **Rotational invariance via distance.** Feed the filter network only `d_ij = ‖r_i - r_j‖`
  (invariant), not the raw displacement. Then `E` is invariant and `F = -∇E` is equivariant
  and curl-free by construction.
- **Gaussian RBF distance expansion.** A near-linear filter net at initialization maps the
  single scalar `d` to `F` nearly-identical channels, causing an early-training plateau.
  Expand `d` first in a bank of Gaussians `e_k(d) = exp(-γ(d - μ_k)^2)`, centers `μ_k` on a
  uniform grid `0..r_cut`, width matched to the spacing (`γ = 1/(2Δ^2)`). This decorrelates
  the channels (different channels respond to different distance ranges) and removes the
  plateau. #centers = filter resolution; range of centers = filter size.
- **Shifted softplus.** `ssp(x) = softplus(x) - ln 2 = ln(0.5 e^x + 0.5)`. `C^∞` everywhere
  (so the twice-differentiability constraint holds throughout the energy network), and
  `ssp(0) = 0` (centering, better convergence). Used in the filter net and all atom-wise
  layers.
- **Residual interaction blocks, not weight-shared.** Each block is a residual `x_i + v_i`
  with `v_i` = atom-wise → cfconv → atom-wise → ssp → atom-wise; blocks have independent
  weights. Stacking radial pairwise filters builds many-body representations (after one block
  a neighbor's features already encode *its* neighbors), while every filter stays radial so
  invariance is never spent.
- **Smooth cosine cutoff.** A hard neighbor cutoff makes a neighbor's contribution jump to
  zero as it crosses `r_cut` during dynamics. Multiply the filter by a Behler-style cosine
  cutoff `f_cut(d) = 0.5[1 + cos(π d / r_cut)]` (value and slope both zero at `r_cut`) so
  neighbors fade in/out smoothly and the PES stays `C^1`. This appears in the canonical
  implementation (SchNetPack / PyG `CFConv`).
- **Energy-conserving forces.** `F̂_i = -∂Ê/∂r_i` via autodiff: gradient of a scalar, hence
  curl-free and energy-conserving; equivariant because `Ê` is invariant.

## Architecture

```
x_i^0 = a_{Z_i}                          # learnable per-type embedding (or LazyLinear over node features)
for each interaction block l (independent weights):
    e(d_ij) = [exp(-γ(d_ij - μ_k)^2)]_k                  # Gaussian RBF expansion
    W(d_ij) = filter_net(e(d_ij)) · f_cut(d_ij)          # cosine-cut learned filter
    m_i     = Σ_j (lin1 x_j) ∘ W(d_ij)                   # cfconv: feature-wise, summed
    u_i     = lin2(m_i)
    x_i    <- x_i + lin(ssp(u_i))                        # residual interaction update
# output network:
y_i = lin2( ssp( lin1(x_i) ) )                           # per-atom / per-node contribution
E   = Σ_i y_i      (sum: extensive)  or  mean (intensive)   # readout
F_i = -∂E/∂r_i                                            # forces by autodiff
```

Defaults (canonical implementation): `hidden_channels=128`, `num_filters=128`,
`num_interactions=6`, `num_gaussians=50`, `cutoff=10 Å`, `max_num_neighbors=32`,
`readout="add"`. The protein wrapper keeps the interaction stack and readout pattern, replaces
the atom-type embedding with `LazyLinear(hidden_channels)` over node features, keeps the PyG
`lin1: hidden_channels -> hidden_channels // 2`, and replaces `lin2` with
`LazyLinear(out_dim)`.

## Training (energy + force model)

Loss balancing energies and forces:

```
ℓ = ρ ‖E - Ê‖^2 + (1/n) Σ_i ‖ F_i - ( -∂Ê/∂r_i ) ‖^2,   ρ ≈ 0.01
```

(energies and forces differ in scale, so `ρ` down-weights the energy term). Twice-
differentiable network is required; the force model is ~2× the energy model's depth. Trained
with Adam, initial lr `1e-3` with exponential decay, small molecule minibatches, and an EMA
over weights for the test-time model.

## Relation to prior methods

- **Behler–Parrinello high-dim NN potentials / ACSF**: same locality, cosine cutoff, and
  `E = Σ_i E_i` energy-conserving structure, but SchNet *learns* the environment
  descriptors (continuous filters) instead of using fixed hand-crafted symmetry functions.
- **DTNN**: same learnable embeddings and Gaussian distance expansion, but replaces DTNN's
  factorized bilinear *tensor* interaction with a continuous-filter *convolution* (cleaner,
  more accurate, unshared blocks).
- **MPNN / enn-s2s**: SchNet's cfconv message `x_j ∘ W(d_ij)` is the continuous-distance,
  diagonal (feature-wise) special case of the edge-network message `A(e_{vw}) h_w`; using a
  continuous distance instead of one-hot bond types is what gives a smooth PES and enables
  force training / MD, which enn-s2s cannot do.
- **GDML**: both yield energy-conserving forces, but GDML's kernel scales quadratically in
  atoms and examples; SchNet scales linearly with system size and to large datasets.
- **Dynamic filter networks / depthwise-separable conv / ResNet**: the filter-generating idea
  (generalized off the grid), the feature-wise-then-mix factorization, and the residual block.

## Working code (continuous-filter encoder)

Faithful to the canonical PyTorch Geometric SchNet, specialized to the geometric encoder used
for protein structure (node-feature embedding + provided spatial graph; returns node and graph
embeddings).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class GaussianSmearing(nn.Module):
    """e_k(d) = exp(-(d - mu_k)^2 / (2 Delta^2)); centers on a uniform grid 0..cutoff."""
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist.pow(2))


class ShiftedSoftplus(nn.Module):
    """ssp(x) = softplus(x) - ln 2; C-infinity, ssp(0) = 0."""
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    """Continuous-filter convolution. Message j->i: x_j (projected) (*) W(d_ij),
    W(d) = filter_net(e(d)) * cosine_cutoff(d), summed over neighbors."""
    def __init__(self, in_channels, out_channels, num_filters, filter_net, cutoff):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.filter_net = filter_net
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)   # smooth cutoff
        W = self.filter_net(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.filter_net, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class ProteinEncoder(nn.Module):
    """SchNet continuous-filter encoder over alpha-carbon coordinates."""
    def __init__(self, hidden_channels=128, out_dim=1, num_layers=6,
                 num_filters=128, num_gaussians=50, cutoff=10.0, readout="add"):
        super().__init__()
        self.cutoff = cutoff
        self.readout = readout
        self.embedding = nn.LazyLinear(hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_layers)
        ])
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.LazyLinear(out_dim)

    def forward(self, batch):
        edge_index = batch.edge_index
        u, v = edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        h = self.embedding(batch.x)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        h = self.lin1(h)
        h = self.act(h)
        node_emb = self.lin2(h)
        graph_emb = scatter(node_emb, batch.batch, dim=0, reduce=self.readout)
        return {"node_embedding": node_emb, "graph_embedding": graph_emb}
```

For the molecular energy/force model, replace the lazy node-feature embedding with a per-type
`nn.Embedding`, build the graph with a radius graph at `cutoff`, keep the scalar output head
`hidden_channels -> hidden_channels // 2 -> 1`, pool with sum/mean to obtain `E`, and obtain
forces as `-torch.autograd.grad(E, pos)`.
