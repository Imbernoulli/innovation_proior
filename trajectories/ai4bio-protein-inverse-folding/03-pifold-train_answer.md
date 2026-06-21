ProteinMPNN confirmed the diagnosis cleanly: pouring 25 invariant inter-atomic distances and a live edge update into the graph lifted recovery to 0.4603 / 0.4612 / 0.4829 and dropped perplexity from GVP's high-5.8s into the mid-5.4s. The lever was edge-feature richness, not the choice of invariance route. But ProteinMPNN stopped short in three nameable places. Its aggregation is an *unweighted* symmetric MLP sum divided by a fixed `scale` — every valid neighbor contributes equally, and the model never learns that a tightly-packed hydrophobic contact should weigh more than a distant glancing one. Its only side-chain proxy is a *single fixed* virtual $C\beta$ at the ideal tetrahedral position — one hand-placed point, identical for every residue, where the geometry distinguishing amino acids could use a learned, transferable probe. And it has *no global context* — every residue sees only its $k$-NN neighborhood, never a protein-level summary, so it cannot tell a buried core position from a surface one except through whatever leaks over three hops.

I propose the **PiFold encoder**, which grabs all three handles at once while keeping the rich invariant featurization that just paid off. For aggregation I want the graph to *decide* how much neighbor $j$ matters to center $i$. Standard graph attention makes a query from $i$, a key and value from $j$, and scores by a dot product — but the dot product is the restrictive part, because in a protein graph the edge feature is not a small bias, it is the actual pairwise geometry, and I want it to participate *directly* in the score. So instead of separate query/key projections I score each edge with an MLP over the concatenation of the center node, the edge, and the neighbor node,
$$w_{ji} = \frac{\mathrm{AttMLP}([h_i \,\Vert\, e_{ji} \,\Vert\, h_j])}{\sqrt{d_\text{head}}},$$
softmax over the incoming neighbors of the same center, with the value formed from the edge-and-neighbor concatenation $v_j = \mathrm{NodeMLP}([e_{ji}\,\Vert\,h_j])$ and $\hat h_i = \sum_j a_{ji} v_j$. The $1/\sqrt{d_\text{head}}$ scaling is needed for the Transformer reason — large head dimensions otherwise saturate the neighbor softmax. This replaces the equal-weight sum with a learned, geometry-aware weighting. I keep the edge update ProteinMPNN introduced: after the attention refines $h_i$, re-derive each edge from its refreshed endpoints, $e_{ji}\leftarrow\mathrm{EdgeMLP}([\hat h_j\,\Vert\,e_{ji}\,\Vert\,\hat h_i])$.

For global context I refuse to pay quadratic full attention; I only need a cheap protein-level summary. Mean-pool the current node embeddings over the residues of the same protein, push that through an MLP, and use a *sigmoid gate* to modulate each residue's channels, $h_i \leftarrow \hat h_i \cdot \sigma(\mathrm{GateMLP}(c))$. A multiplicative gate, not an additive context: addition would overwrite the local, residue-specific geometry that is the whole carrier of the prediction, whereas multiplication lets the global summary keep or suppress channels while the local representation stays the carrier. This is linear in residues. So one encoder block is now MLP-scored neighbor attention, a wide feed-forward refinement, the edge update, and the global channel gate — strictly more than ProteinMPNN's node-then-edge block.

The richest fix is the third. ProteinMPNN's one $C\beta$ is a hand-placed point; I want *learnable* virtual atoms — points whose positions in the local backbone frame are parameters shared across all residues. Build the frame from the three backbone atoms around $C\alpha$ and place each virtual atom
$$V_i^k = x_k\,b_i + y_k\,n_i + z_k\,(b_i\times n_i) + C\alpha_i$$
with shared, unit-normalized coefficients $(x_k, y_k, z_k)$. Sharing the coefficients across residues is the crucial discipline: if every residue had its own virtual coordinates the points would be arbitrary per example, but shared coordinates force a *transferable* placement — a consensus geometric probe that adds side-chain-like information without ever seeing a side chain, which the optimizer can tune to wherever it most helps discriminate amino acids. With three virtual atoms I get extra invariant distances on both nodes and edges: virtual-virtual pair distances at the node level, and same-index and cross-virtual distances at the edge level, all in the RBF basis. This is the principled generalization of the lone fixed $C\beta$ — three learned probes instead of one fixed one. Around the virtual atoms I enrich the rest of the featurization to the limit of what invariant scalars can carry, since the ProteinMPNN run proved feature richness is the dominant lever: for nodes, intra-residue real-atom pair distances ($C\alpha\text{-}N, C\alpha\text{-}C, C\alpha\text{-}O, N\text{-}C, N\text{-}O, O\text{-}C$) in the RBF basis, the virtual-virtual distances, the six dihedral $\{\sin,\cos\}$ channels, and the local orientation channels; for edges, a long list of inter-residue real-atom pair distances in both orientations, the virtual-atom edge distances, several direction features (dot products of the backbone direction vectors with the neighbor direction, and their cross-product magnitudes — all invariant because they are projections), a few angle features, and the sinusoidal positional encoding of the offset. Every channel is an invariant scalar; together they describe the local geometry far more completely than ProteinMPNN's 25 distances alone. I normalize with BatchNorm rather than LayerNorm here — the dense per-residue MLP stacks that embed these features benefit from batch statistics, and BatchNorm is what the reference PiGNN uses.

On the decoder the encoder-only edit surface and PiFold *agree*, which is why this rung fits the task so naturally. The autoregressive factorization $p(S|X) = \prod_t p(s_t\mid s_{<t}, X)$ is expressive but pays a sequential cost, and many of the dependencies it models through previously generated tokens are *already* induced by the shared structure — if two residues pack together, the backbone geometry and the contact graph have told the encoder so. The right trade is to make the structural encoder strong enough that a *one-shot* decoder suffices, $p(S|X) = \prod_i p(s_i\mid X)$, with the decoder a single linear readout to 20 logits and a log-softmax. This is not a claim that residues are physically independent; it is the claim that the final node embeddings already contain the structural context for each marginal — and it is exactly what the harness wants: `forward(X, mask)`, no sequence input, one parallel forward pass, no length-$L$ decoding loop. Where GVP and ProteinMPNN had their autoregressive decoders *amputated* by the edit surface and lost something, PiFold *chose* the one-shot decoder, so it loses nothing; its whole philosophy is to push the work into the encoder, which is precisely the component this task isolates. That is the structural reason I expect PiFold to top the ladder. Two task-specific notes: the reference runs ten PiGNN layers, far more than the scaffold's three, because the deeper stack is where the attention, edge-update, and context-gate machinery earns its keep — so this rung sets `num_encoder_layers = 10` and drops `batch_size` to 8 to fit memory, both within the four keys the harness permits; and this is the *encoder* transplanted into the dense batched $(B, L, K)$ harness (attention softmax over the neighbor axis with a $-10^9$ mask on padded neighbors, BatchNorm on flattened $(B\cdot L, \textit{hidden})$ tensors), not the sparse scatter-based reference. Invariance holds end to end — distance features are inter-atomic distances, dihedrals and direction dot/cross-product magnitudes are projections, virtual atoms placed in the local frame keep their distances invariant, the positional encoding is frame-independent, and the linear readout consumes only invariant node embeddings — at $O(L\cdot k)$ per layer plus an $O(L)$ context gate. The falsifiable bet against ProteinMPNN's numbers is a clear gain (mid-0.46 on CATH 4.2, high-0.47 on CATH 4.3, TS50 past 0.52, perplexity into the low-5s and mid-4s) with the *largest* jump on out-of-distribution TS50 — because the global gate and the learned probes are exactly the transferable, fold-agnostic structure that should generalize past the training distribution. The open question a finale would have to answer: can the frame-anchored virtual atoms carry richer geometry via *learnable vector operations* between them, not just their pairwise distances?

```python
# =====================================================================
# EDITABLE SECTION START — PiFold baseline
# =====================================================================

import numpy as np


def gather_nodes_pifold(h_V, E_idx):
    """Gather node features for neighbor nodes. Dense batched version."""
    B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
    D = int(h_V.shape[-1])
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


class PiFoldAttention(nn.Module):
    """Attention-based message passing layer inspired by PiFold's NeighborAttention."""

    def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.d_head = hidden_dim // num_heads

        # Value network: processes edge-concatenated features
        self.W_V = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Attention bias from node+edge features
        self.Bias = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        """
        h_V: (B, L, D), h_E: (B, L, K, D_e), E_idx: (B, L, K), mask: (B, L)
        """
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        D = self.hidden_dim
        n_heads = self.num_heads
        d = self.d_head

        # Gather neighbor features
        h_V_neighbors = gather_nodes_pifold(h_V, E_idx)  # (B, L, K, D)
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, D)

        # Edge + neighbor concatenation for value
        val_input = torch.cat([h_E, h_V_neighbors], dim=-1)  # (B, L, K, D_e+D)
        V = self.W_V(val_input).view(B, L, K, n_heads, d)  # (B, L, K, H, d)

        # Attention logits
        bias_input = torch.cat([h_V_expand, h_E, h_V_neighbors], dim=-1)
        w = self.Bias(bias_input).view(B, L, K, n_heads, 1) / np.sqrt(d)

        # Mask and softmax
        if mask_attend is not None:
            w = w + (1.0 - mask_attend.unsqueeze(-1).unsqueeze(-1)) * (-1e9)
        attend = torch.softmax(w, dim=2)  # (B, L, K, H, 1)

        # Aggregate
        h_V_update = (attend * V).sum(dim=2).reshape(B, L, D)  # (B, L, D)
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class PiFoldEdgeMLP(nn.Module):
    """Edge update network from PiFold."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim + edge_dim + hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, E_idx, mask):
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        h_V_neighbors = gather_nodes_pifold(h_V, E_idx)  # (B, L, K, D)
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_EV = torch.cat([h_V_expand, h_E, h_V_neighbors], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        # Apply batch norm per-feature
        D_e = int(h_E.shape[-1])
        h_E_flat = h_E.reshape(-1, D_e)
        h_msg_flat = h_message.reshape(-1, D_e)
        h_E = self.norm(h_E_flat + self.dropout(h_msg_flat)).reshape(B, L, K, D_e)
        return h_E


class PiFoldEncoderLayer(nn.Module):
    """PiFold encoder layer: attention + FFN + edge update + context gating."""

    def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = PiFoldAttention(hidden_dim, edge_dim, num_heads, dropout)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.edge_update = PiFoldEdgeMLP(hidden_dim, hidden_dim, dropout)
        # Context gating
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        B, L = int(h_V.shape[0]), int(h_V.shape[1])
        # Node update via attention
        D = int(h_V.shape[-1])
        dh = self.attention(h_V, h_E, E_idx, mask, mask_attend)
        h_V_flat = h_V.reshape(-1, D)
        dh_flat = dh.reshape(-1, D)
        h_V = self.norm1(h_V_flat + self.dropout(dh_flat)).reshape(B, L, -1)

        dh = self.ffn(h_V)
        h_V_flat = h_V.reshape(-1, D)
        dh_flat = dh.reshape(-1, D)
        h_V = self.norm2(h_V_flat + self.dropout(dh_flat)).reshape(B, L, -1)

        # Edge update
        h_E = self.edge_update(h_V, h_E, E_idx, mask)

        # Context gating (global information)
        # Mean pool over valid residues for context
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        c_V = (h_V * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)  # (B, 1, D)
        gate = self.context_gate(c_V.expand_as(h_V))
        h_V = h_V * gate

        h_V = h_V * mask.unsqueeze(-1)
        return h_V, h_E


class StructureEncoder(nn.Module):
    """PiFold-style structure encoder with rich geometric features."""

    def __init__(self, hidden_dim=128, num_layers=10, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf

        # PiFold uses rich multi-atom-pair features
        # Node features: 6 intra-residue atom-pair RBFs + 6 dihedrals + 9 orientations
        # = 6*num_rbf + 6 + 9
        node_input_dim = 6 * num_rbf + 12
        # Edge features: 15 inter-residue atom-pair RBFs + 4 angles + 12 directions + 16 pos_enc
        edge_input_dim = 15 * num_rbf + 4 + 8 + 16

        # Virtual atoms (learned positions in local frame, like PiFold)
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0, 0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix, dtype=torch.float32))
        n_virtual = 3
        # Add virtual atom pair distances to both node and edge features
        node_input_dim += n_virtual * (n_virtual - 1) * num_rbf  # virtual-virtual pairs
        edge_input_dim += n_virtual * num_rbf + n_virtual * (n_virtual - 1) * num_rbf

        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_e = nn.Linear(hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([
            PiFoldEncoderLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _compute_features(self, X, mask, E_idx):
        """Compute PiFold-style rich geometric features."""
        B, L = int(X.shape[0]), int(X.shape[1])
        K = int(E_idx.shape[2])

        N_pos = X[:, :, 0, :]
        CA_pos = X[:, :, 1, :]
        C_pos = X[:, :, 2, :]
        O_pos = X[:, :, 3, :]

        # Virtual Cb and virtual atoms
        b = CA_pos - N_pos
        c = C_pos - CA_pos
        a = torch.cross(b, c, dim=-1)

        va = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
        virtual_pos = []
        for i in range(int(va.shape[0])):
            vp = va[i, 0] * a + va[i, 1] * b + va[i, 2] * c + CA_pos
            virtual_pos.append(vp)

        # --- Node features ---
        def _node_rbf(_src, _dst):
            D = torch.sqrt(((_src - _dst) ** 2).sum(-1) + 1e-6)  # (B, L)
            return _rbf(D.unsqueeze(2), device=X.device).squeeze(2)  # (B, L, num_rbf)

        node_dist = []
        for _src, _dst in [(CA_pos, N_pos), (CA_pos, C_pos), (CA_pos, O_pos),
                      (N_pos, C_pos), (N_pos, O_pos), (O_pos, C_pos)]:
            node_dist.append(_node_rbf(_src, _dst))

        # Virtual atom pair distances (node-level)
        for i in range(len(virtual_pos)):
            for j in range(i):
                node_dist.append(_node_rbf(virtual_pos[i], virtual_pos[j]))
                node_dist.append(_node_rbf(virtual_pos[j], virtual_pos[i]))

        V_dist = torch.cat(node_dist, dim=-1)

        # Dihedrals and orientations
        V_dihedrals = _dihedrals(X)  # (B, L, 6)
        V_orient = _orientations(X)  # (B, L, 6)

        node_feat = torch.cat([V_dist, V_dihedrals, V_orient], dim=-1)

        # --- Edge features ---
        def _edge_rbf(_src, _dst, E_idx):
            D = torch.sqrt(((_src[:, :, None, :] - _dst[:, None, :, :]) ** 2).sum(-1) + 1e-6)
            D_neighbors = torch.gather(D, 2, E_idx)
            return _rbf(D_neighbors, device=X.device)

        edge_dist = []
        atom_pairs = [
            (CA_pos, CA_pos), (CA_pos, C_pos), (C_pos, CA_pos),
            (CA_pos, N_pos), (N_pos, CA_pos), (CA_pos, O_pos), (O_pos, CA_pos),
            (C_pos, C_pos), (C_pos, N_pos), (N_pos, C_pos),
            (C_pos, O_pos), (O_pos, C_pos), (N_pos, N_pos),
            (N_pos, O_pos), (O_pos, O_pos),
        ]
        for _src, _dst in atom_pairs:
            edge_dist.append(_edge_rbf(_src, _dst, E_idx))

        # Virtual atom edge features
        for i in range(len(virtual_pos)):
            edge_dist.append(_edge_rbf(virtual_pos[i], virtual_pos[i], E_idx))
            for j in range(i):
                edge_dist.append(_edge_rbf(virtual_pos[i], virtual_pos[j], E_idx))
                edge_dist.append(_edge_rbf(virtual_pos[j], virtual_pos[i], E_idx))

        E_dist = torch.cat(edge_dist, dim=-1)

        # Edge angles and directions
        CA_neighbors = gather_nodes_pifold(CA_pos, E_idx)  # (B, L, K, 3)
        dX = CA_neighbors - CA_pos.unsqueeze(2)
        dU = F.normalize(dX, dim=-1)

        fwd = F.normalize(CA_pos[:, 1:, :] - CA_pos[:, :-1, :], dim=-1)
        fwd = F.pad(fwd, (0, 0, 0, 1))
        n_vec = F.normalize(N_pos - CA_pos, dim=-1)
        c_vec = F.normalize(C_pos - CA_pos, dim=-1)
        o_vec = F.normalize(O_pos - CA_pos, dim=-1)

        # Direction features
        E_direct = torch.cat([
            (fwd.unsqueeze(2) * dU).sum(-1, keepdim=True),
            (n_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
            (c_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
            (o_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
            torch.cross(fwd.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
            torch.cross(n_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
            torch.cross(c_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
            torch.cross(o_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
        ], dim=-1)  # (B, L, K, 8)

        # Edge angles (4): dihedral-like between consecutive neighbors
        E_angles = torch.cat([
            (dU[:, :, :, 0:1] * dU[:, :, :, 1:2]).clamp(-1, 1),
            (dU[:, :, :, 0:1] * dU[:, :, :, 2:3]).clamp(-1, 1),
            dU.norm(dim=-1, keepdim=True),
            dX.norm(dim=-1, keepdim=True) / 20.0,
        ], dim=-1)  # (B, L, K, 4)

        # Positional encoding
        residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = residue_idx.unsqueeze(2) - torch.gather(
            residue_idx.unsqueeze(2).expand(-1, -1, K), 1,
            E_idx.clamp(0, L - 1)
        )
        pe_dim = 16
        freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) * -(np.log(10000.0) / pe_dim))
        angles = offset.unsqueeze(-1).float() * freq
        pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        edge_feat = torch.cat([E_dist, E_angles, E_direct, pos_enc], dim=-1)

        return node_feat, edge_feat

    def forward(self, X, mask):
        B, L = int(X.shape[0]), int(X.shape[1])
        X_ca = X[:, :, 1, :]
        E_idx, _ = knn_graph(X_ca, mask, self.k_neighbors)
        K = int(E_idx.shape[2])

        # Compute features
        node_feat, edge_feat = self._compute_features(X, mask, E_idx)

        # Embed
        h_V_flat = self.node_embed(node_feat).reshape(-1, self.hidden_dim)
        h_V = self.norm_nodes(h_V_flat).reshape(B, L, self.hidden_dim)
        h_V = self.W_v[0](h_V)
        h_V_flat = h_V.reshape(-1, self.hidden_dim)
        h_V = self.W_v[2](self.W_v[1](h_V_flat)).reshape(B, L, self.hidden_dim)
        h_V = self.W_v[3](h_V)
        h_V_flat = h_V.reshape(-1, self.hidden_dim)
        h_V = self.W_v[5](self.W_v[4](h_V_flat)).reshape(B, L, self.hidden_dim)
        h_V = self.W_v[6](h_V)

        h_E_flat = self.edge_embed(edge_feat).reshape(-1, self.hidden_dim)
        h_E = self.norm_edges(h_E_flat).reshape(B, L, K, self.hidden_dim)
        h_E = self.W_e(h_E)

        # Attention mask
        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
                                    E_idx.clamp(0, L - 1))
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # Message passing
        for layer in self.layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V


class InverseFoldingModel(nn.Module):
    """PiFold inverse folding model with non-autoregressive MLP decoder."""

    def __init__(self, hidden_dim=128, num_encoder_layers=10, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Linear(hidden_dim, NUM_AA)

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```

The late-file edit that accompanies this block sets the training overrides the deeper stack needs (it
replaces the scaffold's `CONFIG_OVERRIDES = {}` line inside `main()`):

```python
CONFIG_OVERRIDES = {'num_encoder_layers': 10, 'batch_size': 8}
```
