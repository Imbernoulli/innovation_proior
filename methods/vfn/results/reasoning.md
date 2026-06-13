Let me start from the thing that actually limits me. I have a per-residue frame `T_i = (R_i, t_i)` built
from the backbone — translation at `Cα`, rotation from `N, Cα, C` by Gram–Schmidt — and I want a layer
that reads the geometric relationship between two residues' frames and turns it into features that update
the residue representation. The dominant primitive for this is invariant point attention: each residue
emits query and key *points* in its own frame, and the geometric attention term is the squared distance
between `i`'s query points and `j`'s key points after both are brought into a common frame. It works, it
is invariant, and I see precisely where it hurts. The geometry between the two residues is a relationship
between two little clouds of points in 3D, and IPA collapses that relationship into a *scalar* — a sum of
squared point distances — before anything learnable touches it. A scalar distance-sum is blind to
*direction*: two neighbors whose atom clouds sit at the same set of distances but rotated differently
relative to me produce nearly the same scalar, and yet they are different environments. The point cloud
carries directional structure and IPA throws it away at the door. I will call this the atom-
representation bottleneck, and breaking it is the whole job.

So what would "not collapse to a scalar" look like while staying invariant? The instinct is to keep the
geometry as *vectors* and only reduce to scalars at the very end. Let me think about what object I should
be computing vectors *from*. Each residue carries a set of points anchored in its frame. Real backbone
atoms are an obvious set, but they are few and fixed, and the side chain — which is what actually
distinguishes amino acids — is not in the input. So let me give each residue a set of *virtual atoms*:
learnable points whose coordinates are parameters expressed in the local frame, shared across residues.
Because they live in the frame, they rotate and translate with the residue, so any frame-relative
quantity computed from them is invariant. Shared coordinates force the optimizer to learn a *transferable*
geometric probe — a consensus set of points in the residue frame that best help discriminate identity —
rather than arbitrary per-example points. PiFold already has such virtual atoms, but it reads only their
pairwise distances, which is the same scalar collapse one notch down. I want to read *vectors* between
them.

Now make the relationship between two residues concrete. Residue `i` carries virtual atoms `Q_i` in its
own frame; residue `j` carries `Q_j` in *its* frame. To relate them I must put them in a common frame —
the center residue `i`'s frame is the natural choice. The neighbor's atoms in world coordinates are
`R_j Q_j + t_j`; bringing those into `i`'s frame gives `K_j = R_i^T(R_j Q_j + t_j − t_i)`, which is
exactly `T_{i←j} ∘ Q_j` with `T_{i←j} = T_i^{-1} ∘ T_j`. The center's own atoms in its frame are just
`Q_i`. So in `i`'s frame I now have two sets of 3D points — `Q_i` (the center, `d_q` points) and `K_j`
(the neighbor, `d_q` points) — and the global rotation has cancelled, because under `X → R_g X + t_g` the
frames become `R_i → R_g R_i`, `t_i → R_g t_i + t_g`, and `R_i^T R_g^T R_g (·) = R_i^T (·)`. Whatever I
compute from `Q_i` and `K_j` is invariant. Good — that is the substrate.

What computation do I run on two sets of vectors? Here is the key move: treat it like a *linear layer*,
but with 3D vectors as the channel values instead of scalars. An ordinary linear layer takes scalar
inputs `x_l` and outputs `y_k = Σ_l w_{k,l} x_l`. I do the same, but the `x_l` are now the virtual-atom
*vectors*, so the output channels are themselves vectors:
`h⃗_k = Σ_l w^a_{k,l} q⃗_l + Σ_l w^b_{k,l} k⃗_l`, with learnable scalar weights `w^a, w^b` mixing the
center atoms and the neighbor atoms into `d_out` output *vectors* in `R^3`. This is the vector field
operator: it is a learnable linear combination over the input atom vectors, exactly like a linear layer,
except every channel carries an `R^3` vector. Crucially the weights are *scalars* multiplying whole
vectors, so the operation commutes with rotation of the common frame — if I rotated `Q_i, K_j` by some
`U`, every `h⃗_k` rotates by the same `U`, because `Σ w (vU) = (Σ w v)U`. So the output vectors are
equivariant in the common frame, which is what I want before I reduce.

Now reduce to invariant scalars, but reduce *late* and reduce *richly*. The norm of a vector is
invariant; so is the unit direction *relative to the (already-canonical) center frame*. So for each
output vector I keep both: the unit direction `h⃗_k / ‖h⃗_k‖` (three numbers that survive because the
center frame is canonical) and the magnitude lifted into an RBF basis `RBF(‖h⃗_k‖)`. Concatenate over the
`d_out` channels: `g_{i,j} = concat_k( h⃗_k/‖h⃗_k‖ , RBF(‖h⃗_k‖) )`. This is the geometric feature for
the edge `(i, j)`, an invariant vector of length `d_out · (3 + num_rbf)`. Compare to IPA: IPA kept only
`Σ ‖·‖²` (one scalar), I keep `d_out` learnable directions *and* their RBF-coded magnitudes — many more
geometric numbers, and the *directions* that IPA discarded are back. That is the bottleneck broken. The
RBF on the magnitude (rather than the raw norm) is the same trick the rest of the field uses: it turns
"this length is about `μ_n`" into a soft one-hot the downstream linear layer can act on like a lookup.

With the geometric feature in hand, the layer is an ordinary attention-plus-update block, except the
attention and value functions get to see `g_{i,j}`. Attention weight from the center node, neighbor
node, geometric feature, and edge feature: `a_{i,j} = softmax_j( MLP(s_i, s_j, g_{i,j}, e_{i,j}) )`,
softmax over the center's neighbors, scaled by `1/√d_head` for the usual saturation reason. Value from
the neighbor, geometry, and edge: `v_{i,j} = MLP(s_j, g_{i,j}, e_{i,j})`; aggregate `o_i = Σ_j a_{i,j}
v_{i,j}` and residual-update `s_i ← s_i + MLP(o_i)`, followed by a wide feed-forward, each with a norm.
Update the edges too, because a static edge wastes the most geometry-laden channel — re-derive each edge
from the refreshed endpoints and its geometry, `e_{i,j} ← e_{i,j} + MLP(s_i, s_j, g_{i,j}, e_{i,j})`.

There is one more thing to decide: the virtual atoms are fixed parameters so far, but they could *move*
as the representation refines, which would let the geometric probe adapt per residue through depth. The
paper offers two updates. The node-feature-based update simply re-predicts the atoms from the current
node state, `Q_i ← Linear(s_i)` — cheap, and it lets each residue's probe specialize to what its
embedding now knows. The coordinate-aggregating update instead pulls neighbor atoms in,
`Q_i ← V-MLP(Q_i, Σ_j a_{i,j} K_j)`. The node-feature update is the lighter, more stable choice and is
what I will lean on: predict the virtual-atom coordinates directly from the node features at the end of
each layer, so the atoms stay frame-anchored while specializing to the current residue state.

Let me check the whole thing is invariant end to end, because that is the property everything rides on.
The geometric feature `g_{i,j}` is computed from `Q_i` and `K_j`, both expressed in the canonical center
frame, through scalar-weighted vector combinations and then norm/unit-direction reductions — invariant,
as shown. The node and edge scalar features feeding the MLPs are themselves invariant (dihedrals, RBF
distances, sequence offsets). The attention is a softmax over a symmetric neighbor set. The virtual-atom
update predicts atoms in the local frame from invariant node features, so the atoms stay frame-anchored.
The final readout is a linear map of the invariant node embeddings. So the output distribution does not
move when the whole protein is rotated or translated — SE(3)-invariant by construction. And the
cost is `O(L·k·d_q·d_out)` per layer, linear in length; with modest `d_q` and `d_out` it scales to whole
proteins.

For the design task itself I follow PiFold's decoding decision rather than an autoregressive one: make
the encoder strong enough that a *one-shot* linear decoder suffices, `p(s_i | X) = log_softmax(W h_i)`,
trained with per-residue cross-entropy. The autoregressive factorization models residue-residue
dependencies through generated tokens, but most of those dependencies are already induced by the shared
structure the encoder now reads with high geometric fidelity — so a single parallel forward pass through
fifteen VFN layers and a linear head is enough, and it is fast. Fifteen layers is the depth at which the
vector-field machinery pays off; deeper than PiFold's ten because each layer's geometric feature is
richer and benefits from more rounds of refinement.

Let me sanity-check by removing each piece. Drop the vector field operator and keep only scalar
distances between virtual atoms, and I am back to PiFold's distance-only reading — the bottleneck
returns. Drop the virtual atoms and use only real backbone atoms, and the probe loses the side-chain-
like capacity that distinguishes residues. Reduce the geometric feature to just the norms (drop the unit
directions), and I have re-collapsed direction into magnitude — IPA again. Freeze the virtual atoms
(drop the node-feature update), and the probe cannot specialize through depth. Each piece prevents a
specific regression, and the central one — the learnable vector combination of frame-anchored points,
reduced to invariant directions-plus-magnitudes only at the end — is exactly what breaks the atom-
representation bottleneck that scalar pooling imposes.

So the causal chain, end to end. Frame-aware encoders model the geometry between two residues' atom
clouds, but IPA (and distance-only PiFold) pool that relationship into a scalar before anything learnable
sees it, discarding direction. Keep the geometry as vectors instead: express the center's and neighbor's
virtual atoms in the common center frame, run a learnable *linear-layer-over-vectors* (the vector field
operator) to produce `d_out` output vectors, and reduce to invariant scalars only at the end by keeping
each output vector's unit direction *and* its RBF-coded magnitude. Feed that rich geometric feature into
an attention-plus-edge-update block, let the frame-anchored virtual atoms drift via a node-feature
update, stack fifteen such layers, and decode one-shot with a linear head under cross-entropy. The
result models frames with the capacity of a vector-valued linear layer rather than a scalar distance-sum,
breaking the bottleneck while staying exactly invariant and linear in length.

Here is the method as code, the slots filled in (a dense, frame-anchored implementation over padded
`(B, L, K)` tensors with the harness mask).

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
    """Vector field operator (VFN Eqs. 1-3): learnable linear combination over
    frame-anchored virtual-atom VECTORS, reduced to invariant unit-direction + RBF(norm)."""

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
        # neighbor virtual atoms expressed in the CENTER frame
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
```

So the final shape: per-residue frames and shared learnable virtual atoms; a vector field operator that
runs a learnable linear combination over frame-anchored atom *vectors* and reduces to invariant
unit-directions plus RBF-coded magnitudes (breaking IPA's scalar bottleneck); an attention-plus-edge-
update block that consumes those rich geometric features; a node-feature virtual-atom update so the probe
specializes through depth; fifteen layers; a one-shot linear decoder under per-residue cross-entropy.
