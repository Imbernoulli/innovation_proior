Protein-ligand binding affinity prediction asks for the strength with which a small molecule sticks to its target protein, given the 3D structure of the bound complex. The quantity to predict is `-logKd/Ki`, and it is determined almost entirely by the physical interactions between atoms: covalent bonds inside the ligand and inside the protein, and noncovalent contacts across the interface. Because the interface is a spatial object, a useful model must consume actual 3D coordinates. But a fundamental symmetry makes this tricky: the binding energy does not change if the entire complex is rigidly translated or rotated. Any predictor that gives different answers for the same complex viewed from a different angle is predicting something physically meaningless, so the model must be exactly invariant to translations and rotations, and it should get that invariance by construction rather than by expensive data augmentation.

Earlier methods fall short in one of two ways. Distance-only geometric networks such as SchNet and EGNN respect the rigid-motion symmetry by feeding only interatomic distances into message passing, but they were designed for homogeneous molecules and process every edge with the same function. A covalent bond at 1.5 angstroms and a noncovalent contact at 3–5 angstroms are very different physical interactions, and a single shared filter must blur them. InteractionGraphNet recognizes the covalent/noncovalent distinction, but it separates the two interaction types into sequential modules that never let an atom integrate both kinds of neighbors in the same step. What is missing is a single layer that is simultaneously E(3)-invariant and heterogeneous.

The method I propose is GIGN, the Geometric Interaction Graph Neural Network. Its core is a single heterogeneous interaction layer (HIL) that runs message passing over the covalent and noncovalent edge sets in parallel and fuses the two per node. Geometry enters only through the rotation- and translation-invariant interatomic distance `d_ij = ||pos_i - pos_j||`. Because distances are unchanged under rigid motions, every message, node update, pooled vector, and final prediction is automatically E(3)-invariant. The scalar distance is first expanded in a small bank of Gaussian radial basis functions on a uniform grid from 0 to 6 angstroms; this decorrelates the filter channels at initialization and avoids the near-linear plateau that stalls single-scalar distance filters. A small learned coordinate network then turns the RBF expansion into a per-edge filter of the same width as the node features.

To respect the heterogeneity of the complex, covalent edges and noncovalent edges each get their own coordinate network, since bonds and contacts occupy different distance regimes and should not be forced through the same map. Inside the layer, each edge type produces a gated message: the neighbor's features are multiplied elementwise by the distance-derived filter, and messages are summed over neighbors. This gives two neighborhood summaries per atom, one covalent and one noncovalent. Each summary is added to the atom's own current features to preserve identity, then passed through its own node MLP, and the two transformed branches are summed. Stacking three such layers lets a ligand atom feel the protein atoms it contacts while also propagating information along its own bonded skeleton, and vice versa. The final per-atom representations are summed over the whole complex because affinity is an extensive-like quantity built from many local interactions, and then a small fully-connected head regresses to a single scalar.

GIGN therefore keeps the geometry-driven invariance of SchNet and EGNN, keeps the acknowledgement of heterogeneous interaction types from InteractionGraphNet, but integrates both interaction channels within one message-passing step rather than in separate sequential stages. It is trained end to end with mean-squared error on the measured `-logKd/Ki`, using Adam with a small learning rate and weight decay.

```python
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """Gaussian RBF expansion of distances along a new last axis."""
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
        radial_cov = self.mlp_coord_cov(
            _rbf(torch.norm(pos[row_cov] - pos[col_cov], dim=-1),
                 D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(
            edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        row_ncov, col_ncov = edge_index_inter
        radial_ncov = self.mlp_coord_ncov(
            _rbf(torch.norm(pos[row_ncov] - pos[col_ncov], dim=-1),
                 D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(
            edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)

        out_node = self.mlp_node_cov(x + out_node_intra) \
            + self.mlp_node_ncov(x + out_node_inter)
        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        return x_j * radial


class FC(nn.Module):
    """Regression head."""

    def __init__(self, d_in, d_hidden, n_layers, dropout, n_out):
        super().__init__()
        layers = []
        for j in range(n_layers):
            if j == 0:
                layers += [nn.Linear(d_in, d_hidden), nn.Dropout(dropout),
                           nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
            if j == n_layers - 1:
                layers.append(nn.Linear(d_hidden, n_out))
            else:
                layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
                           nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
        self.layers = nn.ModuleList(layers)

    def forward(self, h):
        for layer in self.layers:
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
```
