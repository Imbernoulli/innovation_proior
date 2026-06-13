**Problem (from step 3).** PiFold topped the ladder (0.4648 / 0.4772 / 0.5228 recovery), and its lever
was *learnable virtual atoms read as distances*. But a distance is a lossy summary of the geometric
relationship between two frame-anchored point clouds — it keeps magnitude and discards *direction*. That
is the scalar-collapse disease one notch finer than the rest of the ladder.

**Key idea (Vector Field Network).** Read the frame-anchored virtual atoms as *vectors*, not distances.
Per residue, place learnable virtual atoms `Q_i` in its AlphaFold frame `(R_i, t_i)`; express a
neighbor's atoms in the center frame `K_j = R_i^T(R_j Q_j + t_j − t_i)`; run a **vector field operator** —
a learnable linear combination over the atom *vectors*, `h⃗_k = Σ_n w^a_{k,n} q⃗_n + Σ_n w^b_{k,n} k⃗_n`
(a linear layer whose channel values are 3D vectors) — then reduce each output vector to its unit
direction *and* an RBF of its magnitude, `g = concat_k(h⃗_k/‖h⃗_k‖, RBF(‖h⃗_k‖))`. Where PiFold kept a few
scalar distances, this keeps `d_out` learnable directions plus magnitudes — the direction PiFold threw
away, learnable rather than a fixed atom-pair list.

**Why it works.** Scalar weights times whole vectors commute with rotation of the common frame, so the
output vectors are equivariant; reducing to unit-direction + norm in the canonical center frame makes the
feature invariant while *retaining* direction. Verified strictly SE(3)-invariant (random transform moves
the output log-probabilities by ~1e-6, float noise).

**Why it fits this task (and why it is the natural step past PiFold).** PiFold already chose a one-shot
linear decoder and put all the work in the encoder, so it lost nothing to the harness's amputation of the
autoregressive decoder; VFN inherits exactly that decision. So this is purely an *encoder* improvement
layered on PiFold's decoding philosophy: `forward(X, mask)`, KNN graph, dense `(B, L, K)` tensors, the
harness mask, a one-shot linear head, and `CONFIG_OVERRIDES = {'num_encoder_layers': 15, 'batch_size': 8}`
for the deeper stack — no sequence input, no decoder loop, no second optimizer.

**Hyperparameters.** `hidden_dim=128`, `num_encoder_layers=15`, `k_neighbors=30`, `num_virtual=4`,
`d_vec=32`, `num_rbf=16`, `num_heads=4`, `dropout=0.1`; node features = backbone dihedrals (strictly
invariant); edge features = CA-CA RBF + sinusoidal sequence offset; virtual atoms shared/learnable, set
per residue per layer by `Q_i ← Linear(s_i)`; one-shot linear decoder; masked per-residue cross-entropy.

**The bar (no feedback — this is the endpoint).** It must beat PiFold's real 0.4648 / 0.4772 / 0.5228
recovery and 5.2943 / 5.1294 / 4.5513 perplexity on all three benchmarks; the falsifiable claim is that
"direction adds information over distance" should show *cleanest on perplexity* (the smoother signal) — if
VFN does not beat PiFold there, the vector field operator is just extra capacity, not extra geometry. To
validate: an ablation swapping only the vector field operator for PiFold's distance readout (attention,
virtual-atom count, depth fixed), and a check that the per-layer virtual-atom motion helps without
destabilizing training.

```python
# =====================================================================
# EDITABLE SECTION START — VFN (Vector Field Network) finale
# =====================================================================

import numpy as np


def _rigid_frames(X, eps=1e-6):
    """Per-residue local frame via the AlphaFold2 rigidFrom3Points Gram-Schmidt.
    Frame translation t = CA; rotation R = [e1, e2, e3] with
    e1 = norm(C - CA), e2 = norm((N - CA) - (e1.(N-CA)) e1), e3 = e1 x e2.
    Returns R (B, L, 3, 3) and t (B, L, 3).
    """
    N, CA, C = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
    v1 = C - CA
    v2 = N - CA
    e1 = F.normalize(v1, dim=-1, eps=eps)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # (B, L, 3, 3), columns are the frame axes
    t = CA
    return R, t


class VectorFieldOperator(nn.Module):
    """Vector field operator (VFN Eqs. 1-3).

    Given the center residue's virtual atoms Q_i (in i's local frame) and the
    neighbor's virtual atoms expressed in i's frame K_j = T_{i<-j} . Q_j, compute
    d_out feature vectors as learnable linear combinations of both atom sets:
        h_k = sum_l w^a_{k,l} q_l + sum_l w^b_{k,l} k_l            (Eq. 2)
    then split each output vector into a unit direction and an RBF of its norm:
        g = concat_k( h_k / ||h_k||,  RBF(||h_k||) )               (Eq. 3)
    Output scalar feature dim = d_out * (3 + num_rbf).
    """

    def __init__(self, num_virtual, d_out, num_rbf=16):
        super().__init__()
        self.num_virtual = num_virtual
        self.d_out = d_out
        self.num_rbf = num_rbf
        # learnable weights over the 2*num_virtual input vectors (center + neighbor)
        self.w_a = nn.Parameter(torch.randn(d_out, num_virtual) * (1.0 / np.sqrt(num_virtual)))
        self.w_b = nn.Parameter(torch.randn(d_out, num_virtual) * (1.0 / np.sqrt(num_virtual)))

    def forward(self, q_i, k_j):
        """
        q_i: (B, L, K, num_virtual, 3) center virtual atoms (broadcast over K)
        k_j: (B, L, K, num_virtual, 3) neighbor virtual atoms in center frame
        Returns g: (B, L, K, d_out * (3 + num_rbf))
        """
        # h_k = sum_n w_a[k,n] q_i[n] + sum_n w_b[k,n] k_j[n]   -> (B, L, K, d_out, 3)
        h = torch.einsum('on,blknd->blkod', self.w_a, q_i) \
            + torch.einsum('on,blknd->blkod', self.w_b, k_j)
        norm = torch.sqrt((h ** 2).sum(-1) + 1e-8)               # (B, L, K, d_out)
        unit = h / norm.unsqueeze(-1).clamp(min=1e-6)            # (B, L, K, d_out, 3)
        rbf = _rbf(norm, device=h.device)                       # (B, L, K, d_out, num_rbf)
        B, L, K = h.shape[0], h.shape[1], h.shape[2]
        g = torch.cat([unit.reshape(B, L, K, -1), rbf.reshape(B, L, K, -1)], dim=-1)
        return g


class VFNLayer(nn.Module):
    """One VFN layer (Eqs. 4-8): geometric features -> attention node update,
    edge update, and a node-feature-based virtual-atom update."""

    def __init__(self, hidden_dim, num_virtual=4, d_vec=32, num_rbf=16,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_virtual = num_virtual
        self.num_heads = num_heads
        self.d_head = hidden_dim // num_heads

        self.vfo = VectorFieldOperator(num_virtual, d_vec, num_rbf)
        g_dim = d_vec * (3 + num_rbf)

        # attention logits from [s_i, s_j, g_ij, e_ij]   (Eq. 4)
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + g_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        # value from [s_j, g_ij, e_ij]                    (Eq. 5)
        self.val_mlp = nn.Sequential(
            nn.Linear(hidden_dim + g_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.o_mlp = nn.Linear(hidden_dim, hidden_dim)
        # edge update from [s_i, s_j, g_ij, e_ij]         (Eq. 7)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + g_dim + hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # node-feature-based virtual atom update          (Eq. 8)
        self.va_update = nn.Linear(hidden_dim, num_virtual * 3)

        self.norm_node = nn.LayerNorm(hidden_dim)
        self.norm_edge = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, Q, R, t, E_idx, mask, mask_attend):
        """
        h_V: (B, L, D) node feats; h_E: (B, L, K, D) edge feats
        Q: (B, L, num_virtual, 3) virtual atoms in each residue's local frame
        R: (B, L, 3, 3); t: (B, L, 3); E_idx: (B, L, K)
        """
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        nv = self.num_virtual

        # Neighbor virtual atoms in WORLD coords: world = R_j @ Q_j + t_j
        Q_world = torch.einsum('blij,blnj->blni', R, Q) + t.unsqueeze(2)  # (B, L, nv, 3)
        # gather neighbor world atoms and neighbor frames
        idx_n = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, nv, 3)
        Q_world_j = torch.gather(
            Q_world.unsqueeze(2).expand(-1, -1, K, -1, -1), 1, idx_n)     # (B, L, K, nv, 3)
        # transform neighbor world atoms into center i's local frame:
        #   K_j = R_i^T (world_j - t_i)
        rel = Q_world_j - t.unsqueeze(2).unsqueeze(3)                     # (B, L, K, nv, 3)
        k_j = torch.einsum('blji,blknj->blkni', R, rel)                  # R^T via index swap
        # center virtual atoms in its own frame are simply Q_i (broadcast over K)
        q_i = Q.unsqueeze(2).expand(-1, -1, K, -1, -1)                   # (B, L, K, nv, 3)

        g = self.vfo(q_i, k_j)                                           # (B, L, K, g_dim)

        h_V_j = gather_nodes_vfn(h_V, E_idx)                            # (B, L, K, D)
        h_V_i = h_V.unsqueeze(2).expand(-1, -1, K, -1)

        # attention (Eqs. 4-5)
        att_in = torch.cat([h_V_i, h_V_j, g, h_E], dim=-1)
        w = self.att_mlp(att_in).view(B, L, K, self.num_heads, 1) / np.sqrt(self.d_head)
        if mask_attend is not None:
            w = w + (1.0 - mask_attend.unsqueeze(-1).unsqueeze(-1)) * (-1e9)
        a = torch.softmax(w, dim=2)                                      # (B, L, K, H, 1)
        val_in = torch.cat([h_V_j, g, h_E], dim=-1)
        v = self.val_mlp(val_in).view(B, L, K, self.num_heads, self.d_head)
        o = (a * v).sum(dim=2).reshape(B, L, self.hidden_dim)           # (B, L, D)
        h_V = self.norm_node(h_V + self.dropout(self.o_mlp(o)))
        h_V = self.norm_ffn(h_V + self.dropout(self.ffn(h_V)))
        h_V = h_V * mask.unsqueeze(-1)

        # edge update (Eq. 7) with refreshed nodes
        h_V_i = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_V_j = gather_nodes_vfn(h_V, E_idx)
        edge_in = torch.cat([h_V_i, h_V_j, g, h_E], dim=-1)
        h_E = self.norm_edge(h_E + self.dropout(self.edge_mlp(edge_in)))

        # virtual-atom update (Eq. 8): Q_i <- Linear(s_i)
        Q = self.va_update(h_V).view(B, L, nv, 3)
        return h_V, h_E, Q


def gather_nodes_vfn(h_V, E_idx):
    B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
    D = int(h_V.shape[-1])
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


class StructureEncoder(nn.Module):
    """VFN structure encoder: frame-anchored virtual atoms + vector field operator."""

    def __init__(self, hidden_dim=128, num_layers=15, k_neighbors=30, dropout=0.1, num_rbf=16,
                 num_virtual=4, d_vec=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf
        self.num_virtual = num_virtual

        # shared learnable virtual-atom coordinates in the local frame (one set, broadcast
        # to every residue; each residue's atoms are then set per layer by Q_i <- Linear(s_i))
        self.virtual_init = nn.Parameter(torch.randn(num_virtual, 3) * 0.5)

        # initial node/edge embeddings from strictly invariant scalars
        node_in = 6                  # backbone dihedrals (sin/cos of phi, psi, omega)
        edge_in = num_rbf + 16       # CA-CA RBF + sinusoidal offset
        self.node_embed = nn.Linear(node_in, hidden_dim)
        self.edge_embed = nn.Linear(edge_in, hidden_dim)

        self.layers = nn.ModuleList([
            VFNLayer(hidden_dim, num_virtual=num_virtual, d_vec=d_vec, num_rbf=num_rbf,
                     dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, mask):
        B, L = int(X.shape[0]), int(X.shape[1])
        X_ca = X[:, :, 1, :]
        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        K = int(E_idx.shape[2])

        R, t = _rigid_frames(X)

        # per-residue virtual atoms in the local frame (shared params, broadcast)
        Q = self.virtual_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).contiguous()

        # node scalars: backbone dihedrals only (strictly invariant; all directional
        # geometry enters through the frame-anchored vector field operator below)
        node_feat = _dihedrals(X)
        h_V = self.node_embed(node_feat)

        # edge scalars: CA-CA RBF + sinusoidal sequence offset
        rbf = _rbf(D_neighbors, device=X.device)
        residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = residue_idx.unsqueeze(2) - torch.gather(
            residue_idx.unsqueeze(2).expand(-1, -1, K), 1, E_idx.clamp(0, L - 1))
        pe_dim = 16
        freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device)
                         * -(np.log(10000.0) / pe_dim))
        ang = offset.unsqueeze(-1).float() * freq
        pos_enc = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
        edge_feat = torch.cat([rbf, pos_enc], dim=-1)
        h_E = self.edge_embed(edge_feat)

        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1, E_idx.clamp(0, L - 1))
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.layers:
            h_V, h_E, Q = layer(h_V, h_E, Q, R, t, E_idx, mask, mask_attend)
            h_V = h_V * mask.unsqueeze(-1)

        return h_V


class InverseFoldingModel(nn.Module):
    """VFN inverse folding model with a one-shot linear decoder (PiFold-style)."""

    def __init__(self, hidden_dim=128, num_encoder_layers=15, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim, num_layers=num_encoder_layers,
            k_neighbors=k_neighbors, dropout=dropout, num_rbf=num_rbf,
        )
        self.decoder = nn.Linear(hidden_dim, NUM_AA)

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```

The late-file edit accompanying this block sets the deeper-stack training overrides (it replaces the
scaffold's `CONFIG_OVERRIDES = {}` line inside `main()`):

```python
CONFIG_OVERRIDES = {'num_encoder_layers': 15, 'batch_size': 8}
```
