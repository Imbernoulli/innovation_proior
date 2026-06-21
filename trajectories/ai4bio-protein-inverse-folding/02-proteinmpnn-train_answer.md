GVP landed at the bottom of the ladder — 0.4310 / 5.8996 on CATH 4.2, 0.4337 / 5.8209 on CATH 4.3, 0.4602 / 5.4831 on TS50 — and the failure was not its invariance, which was provable and trained stably, but *information starvation on the edges*. Each edge carried one $C\alpha\!\to\!C\alpha$ direction and one $C\alpha$-only distance, which under-determines the thing the prediction most depends on: the relative pose of two residues. Two neighbors at the same $C\alpha$ separation can pack against me from completely different angles and present completely different chemical environments, and GVP's thin edge tuple barely distinguishes them. So GVP paid for equivariant-vector machinery and then starved it of input. The diagnosis is sharp: I do not need *more* invariance machinery, I need *richer invariant features* poured into the graph, and I can drop the equivariant bookkeeping that bought me little.

I propose the **ProteinMPNN encoder**, built on a principle GVP left unused. Distances between atoms are invariant to rotation and translation *by construction* — the cleanest invariance available, with zero frame bookkeeping. The reason GVP needed vectors was the Ingraham diagnostic that a single distance is not locally informative, but that diagnostic has an escape hatch: orientation is recoverable from distances if I use *enough* of them. One $C\alpha\text{–}C\alpha$ distance is blind to which side a neighbor sits on, but if for each edge I take the pairwise distances among a small set of backbone atoms on *both* residues, the relative rigid-body pose becomes a little distance-geometry problem — the cross-distances between the two atom triads encode the relative rotation implicitly. So I can have the locality information GVP fought for *and* keep everything as invariant scalars, simply by being generous with how many inter-atomic distances I feed the edge.

Which atoms? I have $N, C\alpha, C, O$, but not the side chain — which is exactly what distinguishes many residues. The *direction* a side chain would point is a strong cue (buried vs. exposed, toward a neighbor or away), and I get a proxy for free by placing a virtual $C\beta$ at the ideal tetrahedral position. With $b = N-C\alpha$, $c = C-C\alpha$, $a = b\times c$,
$$C\beta = -0.58273431\,a + 0.56802827\,b - 0.54067466\,c + C\alpha,$$
the constants being the tetrahedral geometry that puts $C\beta$ where it sits in a real residue. Now there are five atoms per residue. For an edge $(i,j)$ I take the 25 *ordered* atom pairs — the same-atom pairs $C\alpha\text{-}C\alpha, N\text{-}N, C\text{-}C, O\text{-}O, C\beta\text{-}C\beta$ plus the cross pairs in *both* orientations, because $C\alpha\text{-}N$ and $N\text{-}C\alpha$ are different facets of the local-frame pattern and keeping both directions helps the network read relative pose. Each of the 25 distances is lifted into the RBF basis (centers 2–22 Å, the contact-to-neighborhood range), giving $25\times\textit{num\_rbf}$ distance features per edge. That is the concrete answer to GVP's starvation: where GVP saw one direction, ProteinMPNN sees twenty-five invariant distances that together reconstruct the pose. I add a sinusoidal encoding of the *relative* sequence offset $i-j$ (the chain has a direction but no canonical origin, so the offset, not the absolute index, is what is frame-independent and meaningful), concatenate it onto the 25-pair RBF block, and embed to `hidden_dim` with a bias-free linear and a LayerNorm. The nodes keep only a thin geometric summary — a forward orientation vector and a side-chain-direction proxy, six channels — because the heavy geometry now lives on the edges; the reference ProteinMPNN goes further and zeros the node features entirely, forcing all geometry through the edges, and this rung keeps the same spirit with the edges as the carrier.

The second improvement over GVP and the scaffold default is that the edges *update* every layer. The Structured Transformer, the scaffold default, and GVP all refine only the node embeddings and freeze the edge features at featurization — but a contact between two residues *means* different things depending on the rest of the local environment, and a static edge feature cannot express that. So each encoder layer does two things in order. First the node update: gather for each edge the triple $[h_i, h_j, e_{ij}]$, run a three-linear MLP with GELU, sum the per-neighbor messages, divide by a fixed $\textit{scale}=30$ (a degree-stable approximate mean — a literal sum would grow with the variable neighbor count and fight LayerNorm), residual-add and LayerNorm, then a wide position-wise feed-forward with its own residual and LayerNorm. Then the edge update the prior methods lacked: recompute the edge's input triple from the *refreshed* node states, run a second three-linear MLP, residual-add into the edge with a LayerNorm. So each round, nodes aggregate from neighbors and edges, then edges are re-derived from the freshly updated nodes — "what this contact means" gets refined through depth instead of frozen. Three layers reach roughly three-hop neighborhoods, enough for the local geometry that dominates and fast enough for the budget; hidden 128, GELU throughout, Xavier init.

The famous ProteinMPNN is much more than an encoder — autoregressive *order-agnostic* random decoding (a lower-triangular order-space mask conjugated through a permutation matrix, giving a BERT-like denser signal and letting a designer fix arbitrary positions), sequence embeddings flowing in only from already-decoded neighbors, backbone-coordinate Gaussian noise `augment_eps` as a regularizer against the crystal-vs-predicted shift, chain encoding for multi-chain design, label smoothing, and a Noam warmup schedule. None of that survives the edit surface: the harness gives `forward(X, mask)` with no sequence input, a fixed MLP decoder head, one chain, and a fixed `AdamW`/`OneCycleLR`/cross-entropy loop, so there is no decoder to make autoregressive, no sequence to mask, no second optimizer to schedule, and `augment_eps` stays at 0. So this rung is the ProteinMPNN *encoder* transplanted whole — the 25-pair all-atom RBF featurizer with virtual $C\beta$ and positional encoding, the `EncLayer` with node *and* edge updates, zero/thin node init, three layers — feeding the scaffold's MLP decoder and one-shot log-softmax over 20 amino acids. Invariance rides on the dominant channel: the edge features are pure inter-atomic distances (invariant) plus the relative sequence offset (frame-independent), and zeroing or thinning the node channel is exactly why the reference leans on edges. Every layer's message is an MLP over node and edge states summed symmetrically over neighbors, so permutation invariance is free, at $O(L\cdot k)$ per layer. The bet against GVP's numbers is precise: 25 invariant inter-atomic distances plus a live edge update should beat one direction vector plus a frozen edge, lifting recovery into the mid-0.46s on CATH with perplexity dropping toward the mid-5s — and the gap that remains (an unweighted symmetric sum with no learned neighbor weighting, one fixed $C\beta$, no global context) points the next rung at attention, learned virtual atoms, and a context gate.

```python
# =====================================================================
# EDITABLE SECTION START — ProteinMPNN baseline
# =====================================================================

import numpy as np

class ProteinFeatures(nn.Module):
    """Extract protein structural features: all-atom pairwise RBFs + positional.

    Computes 25 pairwise RBF distance matrices between all backbone atom
    pairs (N, CA, C, O, Cb) following the reference ProteinMPNN implementation.
    Total edge features = 25 * num_rbf + num_pos_emb.
    """
    def __init__(self, edge_features, node_features, num_pos_emb=16, num_rbf=16,
                 top_k=30, augment_eps=0.0):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_pos_emb = num_pos_emb

        # 25 pairwise RBFs + positional encoding
        edge_in = num_pos_emb + num_rbf * 25
        node_in = 6  # forward + side-chain orientation vectors

        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)

    def _pos_enc(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view(1, -1, 1)
        d = (E_idx.float() - ii).unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.num_pos_emb, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_pos_emb)
        )
        angles = d * frequency.view(1, 1, 1, -1)
        return torch.cat([torch.cos(angles), torch.sin(angles)], -1)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = (1. - mask_2D) * 10000 + mask_2D * torch.sqrt((dX ** 2).sum(3) + eps)
        D_max, _ = D.max(-1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]),
                                         dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf_fn(self, D):
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        """Compute pairwise distances between atoms A and B, gather for neighbors, apply RBF."""
        D_AB = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        # Gather neighbor distances
        B_size, L, K = E_idx.shape
        E_idx_expand = E_idx.unsqueeze(-1)  # (B, L, K, 1)
        D_AB_expand = D_AB.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, L)
        # For each node i and neighbor j = E_idx[i,k], get D_AB[i, j]
        D_AB_neighbors = torch.gather(D_AB_expand, 3, E_idx_expand).squeeze(-1)  # (B, L, K)
        return self._rbf_fn(D_AB_neighbors)

    def _orientations(self, X):
        fwd = F.normalize(X[:, 1:, :] - X[:, :-1, :], dim=-1)
        fwd = F.pad(fwd, (0, 0, 0, 1))
        return fwd

    def _sidechains(self, X):
        n, ca, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        u = F.normalize(n - ca, dim=-1)
        v = F.normalize(c - ca, dim=-1)
        return F.normalize(u - v, dim=-1)

    def forward(self, X, mask, residue_idx=None, chain_encoding=None):
        B, L = X.shape[0], X.shape[1]
        N = X[:, :, 0, :]   # N atoms
        Ca = X[:, :, 1, :]  # CA atoms
        C = X[:, :, 2, :]   # C atoms
        O = X[:, :, 3, :]   # O atoms

        # Virtual Cb (beta carbon from N-CA-C geometry)
        b = N - Ca
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # KNN based on CA distances
        D_neighbors, E_idx = self._dist(Ca, mask)

        # All 25 pairwise RBF distances (matching reference ProteinMPNN)
        RBF_all = []
        RBF_all.append(self._rbf_fn(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))
        RBF_all.append(self._get_rbf(C, C, E_idx))
        RBF_all.append(self._get_rbf(O, O, E_idx))
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
        RBF_all.append(self._get_rbf(Ca, N, E_idx))
        RBF_all.append(self._get_rbf(Ca, C, E_idx))
        RBF_all.append(self._get_rbf(Ca, O, E_idx))
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
        RBF_all.append(self._get_rbf(N, C, E_idx))
        RBF_all.append(self._get_rbf(N, O, E_idx))
        RBF_all.append(self._get_rbf(N, Cb, E_idx))
        RBF_all.append(self._get_rbf(Cb, C, E_idx))
        RBF_all.append(self._get_rbf(Cb, O, E_idx))
        RBF_all.append(self._get_rbf(O, C, E_idx))
        RBF_all.append(self._get_rbf(N, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, Ca, E_idx))
        RBF_all.append(self._get_rbf(O, Ca, E_idx))
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, N, E_idx))
        RBF_all.append(self._get_rbf(O, N, E_idx))
        RBF_all.append(self._get_rbf(Cb, N, E_idx))
        RBF_all.append(self._get_rbf(C, Cb, E_idx))
        RBF_all.append(self._get_rbf(O, Cb, E_idx))
        RBF_all.append(self._get_rbf(C, O, E_idx))
        RBF_all = torch.cat(RBF_all, dim=-1)  # (B, L, K, 25*num_rbf)

        # Positional encoding
        O_pos = self._pos_enc(E_idx)  # (B, L, K, num_pos_emb)

        # Edge features: positional + all-atom RBFs
        E = torch.cat([O_pos, RBF_all], dim=-1)

        # Node features: forward + side-chain orientation vectors
        O_fwd = self._orientations(Ca)
        O_sc = self._sidechains(X)
        V = torch.cat([O_fwd, O_sc], dim=-1)

        V = self.norm_nodes(self.node_embedding(V))
        E = self.norm_edges(self.edge_embedding(E))
        return V, E, E_idx


def gather_nodes(h_V, E_idx):
    """Gather node features for neighbor nodes."""
    B, L, K = E_idx.shape
    D = h_V.shape[-1]
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
    """Concatenate neighbor node features with edge features."""
    h_V_neighbors = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_edges, h_V_neighbors], dim=-1)


class EncLayer(nn.Module):
    """ProteinMPNN encoder layer with node and edge updates."""
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.GELU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = h_message.sum(-2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask is not None:
            h_V = mask.unsqueeze(-1) * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class StructureEncoder(nn.Module):
    """ProteinMPNN-style structure encoder with all-atom pairwise features."""

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        self.features = ProteinFeatures(
            hidden_dim, hidden_dim, top_k=k_neighbors, augment_eps=0.0, num_rbf=num_rbf
        )
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_layers)
        ])

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, mask):
        V, E, E_idx = self.features(X, mask)

        # Start with zero node features (per reference ProteinMPNN)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V


class InverseFoldingModel(nn.Module):
    """ProteinMPNN inverse folding model."""

    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA),
        )

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```
