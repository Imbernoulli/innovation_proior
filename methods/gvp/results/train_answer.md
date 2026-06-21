Protein design from backbone structure asks us to read the three-dimensional arrangement of a protein chain and predict, for every position, which of the twenty standard amino acids belongs there. This inverse-folding problem has two inseparable faces. One is geometric: residues point in particular directions, backbone segments bend and twist, and neighboring atoms occupy specific locations in space. The other is relational: which residues contact one another, how interactions are wired across the chain, and the sequential order of residues all matter. Any useful model must honor both. Worse, the correct answer must not change if the entire molecule is rotated or reflected, so the final per-residue predictions must be invariant to rigid motions.

Existing architectures each capture only one face and pay a price for it. Voxel-based three-dimensional convolutions reason directly about spatial shape, but they tie the representation to a fixed grid, discard the residue-residue graph, and are not naturally rotation-invariant without expensive augmentation. Graph neural networks with invariant scalar geometry represent the residue graph naturally, but to stay invariant they project every directional quantity into local coordinate frames at the input, freezing all geometry into scalars before the network can manipulate it as geometric objects. The SO(3)-equivariant irreducible-representation networks keep geometry live at every layer, but their spherical-harmonic and tensor-product machinery is too costly for proteins with hundreds of residues. What is needed is a graph method whose directional features remain live and manipulable through depth, while its scalar readouts stay invariant and its cost scales linearly with the number of residues.

The method I propose is the Geometric Vector Perceptron, or GVP. It is a drop-in replacement for a dense layer that operates on a tuple of scalar and vector features. Scalars are ordinary channels that must be invariant under rotation; vectors are channels in R^3 that must rotate equivariantly. The key observation is that only three operations on vector channels commute with rotation and reflection: bias-free linear mixing over the channel dimension, taking the L2 norm, and rescaling a vector by a scalar function of its own norm. A vector bias is forbidden because no nonzero constant vector is rotation-invariant, and a coordinate-wise nonlinearity is forbidden because coordinate axes are arbitrary. The GVP applies a bias-free map W_h to the input vectors, feeds the row-wise norms into the scalar path alongside the input scalars, and applies a standard biased linear layer plus scalar nonlinearity to produce new scalars. For the new vectors it applies a second bias-free map W_mu and rescales each output vector by a nonlinear function of its norm, so the scalar output is invariant while the vector output is equivariant. The split between W_h and W_mu decouples the number of geometric invariants extracted for the scalars from the number of directional vectors propagated forward.

Stacking GVPs into a message-passing graph neural network yields the GVP-GNN. Each node and edge carries a scalar-vector tuple. Messages are formed by concatenating source-node, edge, and target-node tuples and running them through a short GVP stack, with the final GVP using identity activations so the summed message stays expressive. Incoming messages are aggregated by averaging over the k nearest neighbors, and a residual node update plus an equivariant LayerNorm refines the representation. The equivariant LayerNorm rescales all vector channels by their root-mean-square norm with no per-coordinate parameters, and vector dropout zeros out entire vector channels rather than individual coordinates to preserve equivariance. Because every layer updates both scalars and vectors, the directional state stays alive and gets refined through depth, unlike scalar-only methods that freeze geometry at the input.

For the protein backbone we build a k-nearest-neighbor graph over Cα atoms with k=30. Each node is initialized with six scalar dihedral features from sin and cos of phi, psi, and omega plus three vector directions: the unit vectors from Cα to N, from Cα to C, and the imputed Cβ direction. Each edge carries Gaussian radial basis functions of the Cα-Cα distance, a sinusoidal encoding of the residue offset, and a single direction vector from the center Cα to the neighbor Cα. Input projection GVPs lift nodes to 100 scalars and 16 vectors and edges to 32 scalars and 1 vector. Three encoder layers propagate these tuples. The final node scalars are projected to a hidden dimension and fed to a small perceptron head that outputs per-residue log-probabilities over the twenty amino acids. Training minimizes masked per-residue cross-entropy against the native sequence.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_AA = 20


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def knn_graph(X_ca, mask, k=30):
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
    dX = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)
    D = (1.0 - mask_2D) * 10000.0 + mask_2D * torch.sqrt((dX ** 2).sum(-1) + 1e-6)
    D_max, _ = D.max(-1, keepdim=True)
    D_adjust = D + (1.0 - mask_2D) * (D_max + 1.0)
    D_neighbors, E_idx = torch.topk(D_adjust, min(k, D_adjust.shape[-1]), dim=-1, largest=False)
    return E_idx, D_neighbors


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, 1, 1, -1)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)


def _dihedrals(X, eps=1e-7):
    X = torch.reshape(X, [X.shape[0] * X.shape[1], 4, 3])
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-1, :]
    u_1 = U[:, 1:, :]
    u_0 = U[:, 2:, :]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
    cos_angle = (n_2 * n_1).sum(-1)
    sin_angle = (torch.cross(n_2, u_1, dim=-1) * n_1).sum(-1)
    D = torch.stack([sin_angle, cos_angle], -1)
    D = F.pad(D, (0, 0, 1, 2))
    D = torch.reshape(D, [D.shape[0] // X.shape[0], X.shape[0], 6])
    return D


def gather_nodes_gvp(h_V, E_idx):
    B, L, K = E_idx.shape
    D = h_V.shape[-1]
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


def gather_vectors(V, E_idx):
    B, L, K = E_idx.shape
    nv = V.shape[2]
    E_idx_v = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, nv, 3)
    V_expand = V.unsqueeze(2).expand(-1, -1, K, -1, -1)
    return torch.gather(V_expand, 1, E_idx_v)


class GVPModule(nn.Module):
    def __init__(self, in_dims, out_dims, activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations

        if self.vi:
            self.h_dim = max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

    def forward(self, s, v=None):
        if self.vi and v is not None:
            v_t = v.transpose(-1, -2)
            vh = self.wh(v_t)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v_out = self.wv(vh).transpose(-1, -2)
                gate = self.wsv(self.scalar_act(s) if self.scalar_act else s)
                v_out = v_out * torch.sigmoid(gate).unsqueeze(-1)
            else:
                v_out = None
        else:
            s = self.ws(s)
            v_out = None

        if self.scalar_act:
            s = self.scalar_act(s)
        return s, v_out


class GVPLayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s_dim, self.v_dim = dims
        self.norm_s = nn.LayerNorm(self.s_dim)

    def forward(self, s, v=None):
        s = self.norm_s(s)
        if v is not None and self.v_dim > 0:
            vn = _norm_no_nan(v, axis=-1, keepdims=True)
            v = v / vn.clamp(min=1e-5)
            v = v * vn
        return s, v


class GVPConvLayer(nn.Module):
    def __init__(self, node_dims, edge_dims, drop_rate=0.1):
        super().__init__()
        self.node_s, self.node_v = node_dims
        self.edge_s, self.edge_v = edge_dims

        msg_in_s = self.edge_s + 2 * self.node_s
        msg_in_v = self.edge_v + 2 * self.node_v
        self.msg_gvp = nn.Sequential(
            GVPModule((msg_in_s, msg_in_v), (self.node_s, self.node_v)),
            GVPModule((self.node_s, self.node_v), (self.node_s, self.node_v),
                      activations=(None, None)),
        )

        self.ff_gvp = nn.Sequential(
            GVPModule((self.node_s, self.node_v), (self.node_s * 4, self.node_v)),
            GVPModule((self.node_s * 4, self.node_v), (self.node_s, self.node_v),
                      activations=(None, None)),
        )

        self.norm1 = GVPLayerNorm(node_dims)
        self.norm2 = GVPLayerNorm(node_dims)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, h_s, h_v, e_s, e_v, E_idx, mask, mask_attend):
        B, L, K = E_idx.shape

        h_s_j = gather_nodes_gvp(h_s, E_idx)
        h_s_i = h_s.unsqueeze(2).expand(-1, -1, K, -1)
        msg_s = torch.cat([h_s_i, e_s, h_s_j], dim=-1)

        if h_v is not None:
            h_v_j = gather_vectors(h_v, E_idx)
            h_v_i = h_v.unsqueeze(2).expand(-1, -1, K, -1, -1)
            msg_v = torch.cat([h_v_i, e_v, h_v_j], dim=-2) if e_v is not None else torch.cat([h_v_i, h_v_j], dim=-2)
        else:
            msg_v = e_v

        for layer in self.msg_gvp:
            msg_s, msg_v = layer(msg_s, msg_v)

        mask_expand = mask_attend.unsqueeze(-1)
        msg_s = msg_s * mask_expand
        if msg_v is not None:
            msg_v = msg_v * mask_expand.unsqueeze(-1)

        num_neighbors = mask_attend.sum(dim=-1, keepdim=True).clamp(min=1)
        agg_s = msg_s.sum(dim=2) / num_neighbors
        agg_v = msg_v.sum(dim=2) / num_neighbors.unsqueeze(-1) if msg_v is not None else None

        h_s_res, h_v_res = self.norm1(h_s + self.drop(agg_s),
                                      h_v + self.drop(agg_v) if h_v is not None and agg_v is not None else h_v)

        ff_s, ff_v = h_s_res, h_v_res
        for layer in self.ff_gvp:
            ff_s, ff_v = layer(ff_s, ff_v)

        h_s_out, h_v_out = self.norm2(h_s_res + self.drop(ff_s),
                                      h_v_res + self.drop(ff_v) if h_v_res is not None and ff_v is not None else h_v_res)

        h_s_out = h_s_out * mask.unsqueeze(-1)
        if h_v_out is not None:
            h_v_out = h_v_out * mask.unsqueeze(-1).unsqueeze(-1)
        return h_s_out, h_v_out


class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf

        self.node_s_in = 6
        self.node_v_in = 3
        self.node_s_h = 100
        self.node_v_h = 16
        self.edge_s_in = num_rbf + 16
        self.edge_v_in = 1
        self.edge_s_h = 32
        self.edge_v_h = 1

        self.W_v = GVPModule((self.node_s_in, self.node_v_in),
                             (self.node_s_h, self.node_v_h),
                             activations=(None, None))
        self.norm_v = GVPLayerNorm((self.node_s_h, self.node_v_h))

        self.W_e = GVPModule((self.edge_s_in, self.edge_v_in),
                             (self.edge_s_h, self.edge_v_h),
                             activations=(None, None))
        self.norm_e = GVPLayerNorm((self.edge_s_h, self.edge_v_h))

        self.encoder_layers = nn.ModuleList([
            GVPConvLayer((self.node_s_h, self.node_v_h),
                         (self.edge_s_h, self.edge_v_h), drop_rate=dropout)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(self.node_s_h, hidden_dim)

    def forward(self, X, mask):
        B, L = int(X.shape[0]), int(X.shape[1])
        X_ca = X[:, :, 1, :]

        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        K = int(E_idx.shape[2])

        node_s = _dihedrals(X)

        N_pos, CA_pos, C_pos, O_pos = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
        v_cn = F.normalize(N_pos - CA_pos, dim=-1)
        v_cc = F.normalize(C_pos - CA_pos, dim=-1)
        v_co = F.normalize(O_pos - CA_pos, dim=-1)
        node_v = torch.stack([v_cn, v_cc, v_co], dim=2)

        rbf = _rbf(D_neighbors, device=X.device)
        residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = residue_idx.unsqueeze(2) - torch.gather(
            residue_idx.unsqueeze(2).expand(-1, -1, K), 1, E_idx.clamp(0, L - 1))
        pe_dim = 16
        freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) *
                         -(np.log(10000.0) / pe_dim))
        angles = offset.unsqueeze(-1).float() * freq
        pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        edge_s = torch.cat([rbf, pos_enc], dim=-1)

        CA_neighbors = gather_nodes_gvp(CA_pos, E_idx)
        edge_dir = F.normalize(CA_neighbors - CA_pos.unsqueeze(2), dim=-1)
        edge_v = edge_dir.unsqueeze(3)

        h_s, h_v = self.W_v(node_s, node_v)
        h_s, h_v = self.norm_v(h_s, h_v)

        e_s, e_v = self.W_e(edge_s, edge_v)
        e_s, e_v = self.norm_e(e_s, e_v)

        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1, E_idx.clamp(0, L - 1))
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_s, h_v = layer(h_s, h_v, e_s, e_v, E_idx, mask, mask_attend)

        h_V = self.out_proj(h_s)
        return h_V


class InverseFoldingModel(nn.Module):
    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, k_neighbors, dropout, num_rbf)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA),
        )

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        return F.log_softmax(self.decoder(h_V), dim=-1)
```
