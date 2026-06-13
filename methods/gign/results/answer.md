# GIGN, distilled

GIGN (Geometric Interaction Graph Neural Network) predicts protein-ligand binding affinity
(`-logKd/Ki`) from the 3D structure of the bound complex. Its contribution is a single
**heterogeneous interaction layer (HIL)** that runs distance-driven message passing over the
covalent (intramolecular) and noncovalent (intermolecular) edge sets *in the same step*, each
through its own learned transform, and fuses the two per node. Geometry enters only through the
rotation/translation-invariant interatomic distance, so the whole network is `E(3)`-invariant by
construction — no data augmentation needed.

## Problem it solves

Predict binding affinity from a protein-ligand complex given per-atom features, 3D coordinates,
the intramolecular bond graph, and the intermolecular contact graph (ligand-pocket atom pairs
within 5 angstroms). Binding affinity is set by physical interactions of two distinct kinds —
covalent bonds inside each molecule and noncovalent contacts across the interface — all of which
are determined by interatomic geometry, and it must be invariant to any rigid motion of the whole
complex.

## Key idea

Carry two interaction channels through one message-passing layer and fuse them per node:

- **Distance-only geometry => `E(3)` invariance.** For a pair of atoms, the distance
  `d_ij = ||pos_i - pos_j||` is the complete scalar invariant this layer needs: translation
  cancels in the difference, and rotation/reflection cancels in the norm since `Q^T Q = I`.
  Geometry enters only through `d_ij`, so every message, node update, the pooled vector, and the
  prediction are invariant. Positions are a fixed input read for distances; there is no
  equivariant/coordinate channel (unlike EGNN), because the target is an invariant scalar.
- **Gaussian RBF distance expansion.** A near-linear filter net at init maps the scalar `d` to
  nearly identical, correlated channels (an early plateau). Expand first in a bank of Gaussians
  `rbf(d)_k = exp(-((d - mu_k)/sigma)^2)`, centers `mu_k` on a uniform grid `[0, 6]` angstroms
  with `K = 9`, width `sigma = (D_max - D_min)/K = 6/9`, decorrelating the channels.
- **cfconv-style gated message.** Generate a per-edge filter `radial = SiLU(W_coord · rbf(d))` of
  feature width, and gate the neighbor: `m_ij = x_j (*) radial_ij` (elementwise), summed over
  neighbors.
- **Heterogeneous = separate transforms per interaction type.** Covalent edges (stiff bonds,
  ~1.5 angstroms) and noncovalent edges (soft contacts, up to 5 angstroms) occupy different
  distance regimes, so each gets its own coordinate net (`W_coord_cov`, `W_coord_ncov`) and its
  own node MLP. One shared transform would blur the two physical regimes.
- **Per-node fusion with a residual self-loop.** Aggregate each channel
  (`out_cov_i`, `out_ncov_i`), keep the node's own state, transform each branch, and sum:

  ```
  out_i = MLP_cov(x_i + out_cov_i) + MLP_ncov(x_i + out_ncov_i)
  ```

  Additive fusion treats the two interaction contributions as co-equal terms; the self-loop
  `x_i + out_s_i` preserves the atom's identity when a channel has few neighbors.

## Architecture

```
x_i^0 = SiLU(Linear_{35->H}(atom_features_i))             # embed ligand & pocket atoms
for layer l = 1..3 (HIL, independent weights):
    # covalent (intra) channel
    d_cov   = ||pos_i - pos_j||                  for (i,j) in E_intra
    r_cov   = SiLU(W_coord_cov · rbf(d_cov))                # rbf: 9 Gaussians on [0,6] A
    out_cov_i = sum_{j in N_intra(i)} x_j (*) r_cov_ij
    # noncovalent (inter) channel (own coord net)
    d_ncov  = ||pos_i - pos_j||                  for (i,j) in E_inter
    r_ncov  = SiLU(W_coord_ncov · rbf(d_ncov))
    out_ncov_i = sum_{j in N_inter(i)} x_j (*) r_ncov_ij
    # fuse (residual self-loop + own MLP per branch, summed)
    x_i = MLP_cov(x_i + out_cov_i) + MLP_ncov(x_i + out_ncov_i)
g     = sum_i x_i              (global_add_pool: affinity is extensive)
y_hat = FC(g)                  (3 hidden blocks -> final Linear -> 1)
```

`MLP_s = [Linear(H->H), Dropout(0.1), LeakyReLU, BatchNorm1d]`. With `n_FC_layer=3`, the FC head
has an input hidden block, two hidden-width blocks, then a final scalar linear layer. SiLU (smooth)
on the geometric filter; LeakyReLU + BatchNorm + Dropout in the feature MLPs and head for
regression stability / regularization on a modest dataset.

## E(3) invariance

The only geometric input is `d_ij`. Under `pos -> Q pos + t` with `Q^T Q = I`:

```
||(Q pos_i + t) - (Q pos_j + t)|| = ||Q(pos_i - pos_j)||
   = sqrt((pos_i - pos_j)^T Q^T Q (pos_i - pos_j)) = ||pos_i - pos_j||,
```

so every `d_ij`, `rbf(d_ij)`, `radial`, message, node update, pooled vector, and the prediction
are unchanged. Invariance is exact and built-in; no rotation augmentation is needed.

## Defaults and why

`node_dim = 35`, `hidden_dim = 256`, `3` HIL layers; RBF `D_min=0, D_max=6, D_count=9`; inter-edge
cutoff `5` angstroms; dropout `0.1`. Trained with Adam `lr=5e-4`, `weight_decay=1e-6`, MSE loss,
batch size `128`, up to `800` epochs with early stopping (patience `100`).

- 9 RBF centers over `[0,6]` angstroms: sub-angstrom resolution spanning bond lengths (~1.5) to
  the noncovalent cutoff (5), with `K` small.
- 3 layers: enough hops to mix covalent and noncovalent context a few bonds deep without
  oversmoothing.
- Sum-pool: affinity grows with the number of favorable interacting atoms (extensive); mean would
  wash out complex-size effects.

## Relation to prior methods

- **SchNet** (Schütt et al. 2017): GIGN reuses the continuous-filter, Gaussian-RBF,
  distance-only-invariant message `x_j (*) W(d)`, but SchNet is homogeneous (one edge type); GIGN
  splits the message into a covalent and a noncovalent channel with separate filters.
- **EGNN** (Satorras et al. 2021): GIGN is the invariant-only specialization — positions held
  fixed, EGNN's equivariant coordinate update dropped (the target is a scalar) — and again split
  into two heterogeneous interaction channels.
- **IGN — InteractionGraphNet** (Jiang et al. 2021): IGN also separates covalent and noncovalent
  interactions but in two *sequential, independent* graph-conv modules; GIGN unifies them into one
  layer that fuses both per node within the same step, and feeds the interaction geometry through
  the distance-invariant filter.
- **MPNN** (Gilmer et al. 2017): the underlying message-passing scheme
  `m_ij = phi(h_i,h_j,a_ij); m_i = sum_j m_ij; h_i' = phi_h(h_i,m_i)`.

## Working code

Faithful to the canonical PyTorch-Geometric implementation (`HIL.py` / `GIGN.py`).

```python
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """Gaussian RBF embedding of distances along a new last axis (D_count channels)."""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))


class HIL(MessagePassing):
    """Heterogeneous interaction layer: covalent (intra) + noncovalent (inter)
    message passing in one step, each with its own distance filter, fused per node."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(0.1),
            nn.LeakyReLU(), nn.BatchNorm1d(out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Dropout(0.1),
            nn.LeakyReLU(), nn.BatchNorm1d(out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]
        radial_cov = self.mlp_coord_cov(
            _rbf(torch.norm(coord_diff_cov, dim=-1),
                 D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(
            edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        radial_ncov = self.mlp_coord_ncov(
            _rbf(torch.norm(coord_diff_ncov, dim=-1),
                 D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(
            edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)

        out_node = self.mlp_node_cov(x + out_node_intra) \
            + self.mlp_node_ncov(x + out_node_inter)
        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        x = x_j * radial               # elementwise distance-filter gating (cfconv-style)
        return x


class FC(nn.Module):
    """Regression head."""

    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


class GIGN(nn.Module):
    """Geometric Interaction Graph Neural Network for binding-affinity regression."""

    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv1 = HIL(hidden_dim, hidden_dim)
        self.gconv2 = HIL(hidden_dim, hidden_dim)
        self.gconv3 = HIL(hidden_dim, hidden_dim)
        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos = \
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos
        x = self.lin_node(x)
        x = self.gconv1(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv2(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv3(x, edge_index_intra, edge_index_inter, pos)
        x = global_add_pool(x, data.batch)
        x = self.fc(x)
        return x.view(-1)


# build + train (MSE on -logKd/Ki)
# model = GIGN(node_dim=35, hidden_dim=256)
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
# criterion = torch.nn.MSELoss()
```
