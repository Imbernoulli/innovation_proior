Inverse folding asks for a per-residue distribution over the 20 amino acids that would fold into a given backbone, and because a protein's identity is unchanged under a proper rigid motion of the whole molecule, that distribution must be invariant to rotation and translation. The piece that decides accuracy is the structure encoder, the module that turns backbone geometry into per-residue embeddings, and the best encoders attach a local coordinate frame to every residue — translation at $C\alpha$, rotation built from $N, C\alpha, C$ by Gram–Schmidt — and reason about the relative pose of neighboring frames. The question I care about is how *expressively* an encoder uses those frames, and here every existing option leaves capacity on the table. Invariant Point Attention, the dominant frame-modeling primitive, has each residue emit query and key points in its own frame and then forms the geometric attention term from the *squared distance* between the center's query points and a neighbor's key points after both are brought into a common frame. That is invariant and effective, but it collapses the relationship between two residues' atom clouds into a single scalar — a sum of squared point distances — before anything learnable touches it, and a scalar distance-sum is blind to *direction*: two neighbors whose atom clouds sit at the same distances but rotated differently relative to me look almost identical to it, even though they are different environments. PiFold keeps virtual atoms but reads only their pairwise distances, which is the same scalar collapse one notch down. GVP restricts its vector path to a narrow toolkit of rotation-commuting operations with no general learnable map between arbitrary frame-anchored points, and tensor-field networks pay the heavy irreps tax. The gap is therefore a learnable, invariant, high-capacity vector computation between frame-anchored points — modeling frames the way a linear layer models scalars, but with 3D vectors as the channel values. I call this missing capacity the atom-representation bottleneck, and breaking it is the whole job.

I propose the Vector Field Network. The principle is to keep the geometry as *vectors* and reduce to invariant scalars only at the very end, richly rather than once. First I give each residue a set of learnable virtual atoms: points $Q$ whose coordinates are parameters expressed in the local frame and shared across residues. Because they live in the frame they rotate and translate with the residue, so any frame-relative quantity computed from them is invariant; and sharing their coordinates forces the optimizer to learn a *transferable* geometric probe — a consensus set of points in the residue frame that best discriminate identity — which is exactly the kind of side-chain-like information missing from the fixed-backbone input. To relate the center residue $i$ to a neighbor $j$ I bring both atom sets into the common center frame: the neighbor's atoms go to world coordinates $R_j Q_j + t_j$ and then into $i$'s frame as $K_j = R_i^{\top}(R_j Q_j + t_j - t_i)$, which is precisely $T_{i\leftarrow j}\circ Q_j$, while the center's own atoms are just $Q_i$. Under a global motion $X \to R_g X + t_g$ the frames transform as $R_i \to R_g R_i$ and $t_i \to R_g t_i + t_g$, and the global rotation cancels through $R_i^{\top} R_g^{\top} R_g(\cdot) = R_i^{\top}(\cdot)$, so anything computed from $Q_i$ and $K_j$ in this common frame is invariant. That is the substrate.

The core operation on these two sets of vectors is the vector field operator, and the key move is to treat it like a linear layer whose channel values are 3D vectors instead of scalars. An ordinary linear layer maps scalar inputs to $y_k = \sum_l w_{k,l} x_l$; I do the same, but the inputs are now the virtual-atom *vectors*, so the outputs are themselves vectors,
$$\vec h_k = \sum_n w^a_{k,n}\,\vec q_n + \sum_n w^b_{k,n}\,\vec k_n,$$
with learnable scalar weights $w^a, w^b$ mixing the $\text{num\_virtual}$ center atoms and neighbor atoms into $d_{\text{out}}$ output vectors in $\mathbb{R}^3$. Because the weights are scalars multiplying whole vectors, this operation commutes with rotation of the common frame: rotate $Q_i, K_j$ by some $U$ and every $\vec h_k$ rotates by the same $U$, since $\sum w(vU) = (\sum w v)U$. The output vectors are therefore equivariant in the canonical center frame, which is exactly what I want before reducing. Then I reduce *late* and *richly*: for each output vector I keep both its unit direction $\vec h_k/\lVert \vec h_k\rVert$ — three numbers that survive as invariants because the center frame is canonical — and its magnitude lifted into a radial basis $\mathrm{RBF}(\lVert \vec h_k\rVert)$, and I concatenate across channels,
$$g_{i,j} = \operatorname{concat}_k\!\Big(\,\vec h_k/\lVert \vec h_k\rVert,\ \mathrm{RBF}(\lVert \vec h_k\rVert)\,\Big),$$
an invariant geometric feature of length $d_{\text{out}}\cdot(3 + \text{num\_rbf})$. Where IPA kept the single scalar $\sum\lVert\cdot\rVert^2$, this keeps $d_{\text{out}}$ learnable directions *and* their RBF-coded magnitudes — many more geometric numbers, and the directions IPA discarded are back. Keeping the RBF of the magnitude rather than the raw norm turns "this length is about $\mu_n$" into a soft one-hot the downstream linear layers can read like a lookup; reducing only to the norms, by contrast, would re-collapse direction into magnitude and put me right back at IPA.

The rest of the layer is an ordinary attention-plus-update block, except that the attention and value functions now get to see this rich $g_{i,j}$. The attention weight is $a_{i,j} = \operatorname{softmax}_j\big(\mathrm{MLP}(s_i, s_j, g_{i,j}, e_{i,j})\big)$, softmax over the center's neighbors and scaled by $1/\sqrt{d_{\text{head}}}$ for the usual saturation reason; the value is $v_{i,j} = \mathrm{MLP}(s_j, g_{i,j}, e_{i,j})$; I aggregate $o_i = \sum_j a_{i,j} v_{i,j}$ and do a residual update $s_i \leftarrow s_i + \mathrm{MLP}(o_i)$ followed by a wide feed-forward, each with a LayerNorm. I update the edges too, re-deriving each from its refreshed endpoints and geometry, $e_{i,j} \leftarrow e_{i,j} + \mathrm{MLP}(s_i, s_j, g_{i,j}, e_{i,j})$, because leaving the most geometry-laden channel static wastes it. Finally I let the virtual atoms drift through depth so the probe can specialize: at the end of each layer I re-predict them directly from the node state, $Q_i \leftarrow \mathrm{Linear}(s_i)$, the lighter and more stable of the two possible updates (the alternative pulls neighbor atoms in via $Q_i \leftarrow \text{V-MLP}(Q_i, \sum_j a_{i,j} K_j)$), and because this predicts coordinates in the local frame from invariant node features, the atoms stay frame-anchored. End to end the model is SE(3)-invariant by construction — $g_{i,j}$ is invariant, the node and edge scalar features are invariant, the attention is a softmax over a symmetric neighbor set, the virtual-atom update predicts frame-local coordinates from invariant features, and the readout is a linear map of invariant embeddings — and the per-layer cost is $O(L\cdot k\cdot d_q\cdot d_{\text{out}})$, linear in length. For the design head I follow the one-shot route rather than an autoregressive one: I make the encoder strong enough that a single parallel forward pass through fifteen VFN layers and a linear decoder suffices, $p(s_i\mid X) = \log\operatorname{softmax}(W h_i)$, trained under per-residue cross-entropy. Fifteen layers, deeper than the distance-only baseline's ten, is the depth at which the richer per-layer geometric feature pays off across more rounds of refinement. Removing any single piece regresses to a known weaker method — drop the vector field operator and I am back to distance-only PiFold; drop the virtual atoms and the probe loses its side-chain-like capacity; keep only the norms and I am back to IPA; freeze the atoms and the probe cannot specialize through depth — and the central piece, the learnable vector combination of frame-anchored points reduced to invariant directions-plus-magnitudes only at the end, is exactly what breaks the atom-representation bottleneck.

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
