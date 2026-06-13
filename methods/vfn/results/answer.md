**Problem.** Frame-aware protein encoders model the geometric relationship between two residues' atom
clouds, but Invariant Point Attention (and distance-only encoders like PiFold) pool that relationship
into a *scalar* — a squared-distance sum — before anything learnable touches it, discarding the
*direction* of the point-to-point geometry. This atom-representation bottleneck caps how expressively
frames can be modeled.

**Key idea (Vector Field Network).** Keep the geometry as *vectors* and reduce to scalars only at the
end. Give each residue learnable **virtual atoms** anchored in its local frame; express the center's and
a neighbor's atoms in the common center frame; run a **vector field operator** — a learnable linear
combination over the atom *vectors*, `h⃗_k = Σ_n w^a_{k,n} q⃗_n + Σ_n w^b_{k,n} k⃗_n` (a linear layer
whose channel values are 3D vectors) — then reduce each output vector to its unit direction *and* an RBF
of its magnitude, `g = concat_k(h⃗_k/‖h⃗_k‖, RBF(‖h⃗_k‖))`. This invariant geometric feature carries
`d_out` learnable directions plus magnitudes, where IPA kept one scalar.

**Why it works.** The vector field operator commutes with rotation of the common frame (scalar weights
times whole vectors), so the output vectors are equivariant; reducing to unit-direction + norm makes the
feature invariant while *retaining* the directional structure IPA threw away. Virtual atoms give a
transferable, side-chain-like geometric probe (the side chain is absent from the backbone input). The
result models frames with the capacity of a vector-valued linear layer, not a scalar distance-sum.

**Architecture.** Per-residue frames `(R, t)` via AlphaFold rigidFrom3Points from `N, CA, C`; shared
learnable virtual atoms updated per residue via a node-feature update `Q_i ← Linear(s_i)`; each
layer = vector field operator → MLP-scored neighbor attention (`a = softmax(MLP(s_i, s_j, g, e))`, value
`MLP(s_j, g, e)`) → edge update from refreshed endpoints → virtual-atom update; 15 layers; one-shot
linear decoder under per-residue cross-entropy. Strictly SE(3)-invariant (verified),
`O(L·k·d_q·d_out)`.

**Hyperparameters.** `hidden_dim=128`, `num_layers=15`, `k_neighbors=30`, `num_virtual=4`, `d_vec=32`,
`num_rbf=16`, `num_heads=4`, `dropout=0.1`; node features = backbone dihedrals (invariant); edge
features = CA-CA RBF + sinusoidal sequence offset.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_AA = 20


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((torch.unsqueeze(D, -1) - D_mu) / D_sigma) ** 2)


def rigid_frames(X, eps=1e-6):
    """AlphaFold2 rigidFrom3Points: t = CA; R = [e1, e2, e3] from N, CA, C."""
    N, CA, C = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
    e1 = F.normalize(C - CA, dim=-1, eps=eps)
    v2 = N - CA
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)   # columns are frame axes
    return R, CA


def gather_nodes_vfn(h_V, E_idx):
    B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
    D = int(h_V.shape[-1])
    return torch.gather(h_V.unsqueeze(2).expand(-1, -1, K, -1), 1,
                        E_idx.unsqueeze(-1).expand(-1, -1, -1, D))


class VectorFieldOperator(nn.Module):
    """VFN Eqs. 1-3: learnable linear combination over frame-anchored virtual-atom
    VECTORS, reduced to invariant unit-direction + RBF(norm)."""

    def __init__(self, num_virtual, d_out, num_rbf=16):
        super().__init__()
        self.w_a = nn.Parameter(torch.randn(d_out, num_virtual) / np.sqrt(num_virtual))
        self.w_b = nn.Parameter(torch.randn(d_out, num_virtual) / np.sqrt(num_virtual))
        self.d_out = d_out

    def forward(self, q_i, k_j):
        # q_i, k_j: (B, L, K, num_virtual, 3) center + neighbor atoms in center frame
        h = torch.einsum('on,blknd->blkod', self.w_a, q_i) \
            + torch.einsum('on,blknd->blkod', self.w_b, k_j)   # (B, L, K, d_out, 3)
        norm = torch.sqrt((h ** 2).sum(-1) + 1e-8)
        unit = h / norm.unsqueeze(-1).clamp(min=1e-6)
        rbf = _rbf(norm, device=h.device)
        B, L, K = h.shape[0], h.shape[1], h.shape[2]
        return torch.cat([unit.reshape(B, L, K, -1), rbf.reshape(B, L, K, -1)], dim=-1)


class VFNLayer(nn.Module):
    """Geometric vector feature -> attention node update + edge update + virtual-atom update."""

    def __init__(self, hidden_dim, num_virtual=4, d_vec=32, num_rbf=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim, self.num_virtual = hidden_dim, num_virtual
        self.num_heads, self.d_head = num_heads, hidden_dim // num_heads
        self.vfo = VectorFieldOperator(num_virtual, d_vec, num_rbf)
        g_dim = d_vec * (3 + num_rbf)
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + g_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_heads))
        self.val_mlp = nn.Sequential(
            nn.Linear(hidden_dim + g_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.o_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + g_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.va_update = nn.Linear(hidden_dim, num_virtual * 3)
        self.norm_node = nn.LayerNorm(hidden_dim)
        self.norm_edge = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                                 nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, Q, R, t, E_idx, mask, mask_attend):
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        nv = self.num_virtual
        # neighbor virtual atoms expressed in the CENTER frame: K_j = R_i^T (R_j Q_j + t_j - t_i)
        Q_world = torch.einsum('blij,blnj->blni', R, Q) + t.unsqueeze(2)
        idx_n = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, nv, 3)
        Q_world_j = torch.gather(Q_world.unsqueeze(2).expand(-1, -1, K, -1, -1), 1, idx_n)
        rel = Q_world_j - t.unsqueeze(2).unsqueeze(3)
        k_j = torch.einsum('blji,blknj->blkni', R, rel)        # R^T rel
        q_i = Q.unsqueeze(2).expand(-1, -1, K, -1, -1)
        g = self.vfo(q_i, k_j)

        h_V_j = gather_nodes_vfn(h_V, E_idx)
        h_V_i = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        w = self.att_mlp(torch.cat([h_V_i, h_V_j, g, h_E], dim=-1))
        w = w.view(B, L, K, self.num_heads, 1) / np.sqrt(self.d_head)
        if mask_attend is not None:
            w = w + (1.0 - mask_attend.unsqueeze(-1).unsqueeze(-1)) * (-1e9)
        a = torch.softmax(w, dim=2)
        v = self.val_mlp(torch.cat([h_V_j, g, h_E], dim=-1)).view(B, L, K, self.num_heads, self.d_head)
        o = (a * v).sum(dim=2).reshape(B, L, self.hidden_dim)
        h_V = self.norm_node(h_V + self.dropout(self.o_mlp(o)))
        h_V = self.norm_ffn(h_V + self.dropout(self.ffn(h_V)))
        h_V = h_V * mask.unsqueeze(-1)

        h_V_i = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_V_j = gather_nodes_vfn(h_V, E_idx)
        h_E = self.norm_edge(h_E + self.dropout(
            self.edge_mlp(torch.cat([h_V_i, h_V_j, g, h_E], dim=-1))))

        Q = self.va_update(h_V).view(B, L, nv, 3)              # Eq. 8: Q_i <- Linear(s_i)
        return h_V, h_E, Q


class VFNEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=15, k_neighbors=30, dropout=0.1,
                 num_rbf=16, num_virtual=4, d_vec=32):
        super().__init__()
        self.k_neighbors, self.num_virtual, self.hidden_dim = k_neighbors, num_virtual, hidden_dim
        self.virtual_init = nn.Parameter(torch.randn(num_virtual, 3) * 0.5)
        self.node_embed = nn.Linear(6, hidden_dim)            # dihedrals (invariant)
        self.edge_embed = nn.Linear(num_rbf + 16, hidden_dim)
        self.layers = nn.ModuleList([
            VFNLayer(hidden_dim, num_virtual, d_vec, num_rbf, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, X, mask, dihedrals, E_idx, D_neighbors):
        B, L, K = X.shape[0], X.shape[1], E_idx.shape[2]
        R, t = rigid_frames(X)
        Q = self.virtual_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).contiguous()
        h_V = self.node_embed(dihedrals)
        rbf = _rbf(D_neighbors, device=X.device)
        ridx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = ridx.unsqueeze(2) - torch.gather(ridx.unsqueeze(2).expand(-1, -1, K), 1,
                                                   E_idx.clamp(0, L - 1))
        freq = torch.exp(torch.arange(0, 16, 2, dtype=torch.float32, device=X.device)
                         * -(np.log(10000.0) / 16))
        ang = offset.unsqueeze(-1).float() * freq
        h_E = self.edge_embed(torch.cat([rbf, torch.cos(ang), torch.sin(ang)], dim=-1))
        ma = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1, E_idx.clamp(0, L - 1))
        ma = mask.unsqueeze(-1) * ma
        for layer in self.layers:
            h_V, h_E, Q = layer(h_V, h_E, Q, R, t, E_idx, mask, ma)
            h_V = h_V * mask.unsqueeze(-1)
        return h_V


class VFNDesign(nn.Module):
    """One-shot inverse-folding model: VFN encoder + linear decoder."""

    def __init__(self, hidden_dim=128, num_layers=15, k_neighbors=30, dropout=0.1):
        super().__init__()
        self.encoder = VFNEncoder(hidden_dim, num_layers, k_neighbors, dropout)
        self.decoder = nn.Linear(hidden_dim, NUM_AA)

    def forward(self, X, mask, dihedrals, E_idx, D_neighbors):
        h_V = self.encoder(X, mask, dihedrals, E_idx, D_neighbors)
        return F.log_softmax(self.decoder(h_V), dim=-1)
```
