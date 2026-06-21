## Research question

Protein inverse folding (fixed-backbone design): given the backbone coordinates `X (B, L, 4, 3)` — the `N, CA, C, O` positions for each of `L` residues — predict a per-residue distribution over the 20 standard amino acids that would fold into that shape. The target is `p(s | X)`, not a deterministic inverse, and the prediction must be invariant to rigid motion. The design focus is the **structure encoder**, the GNN module that turns backbone geometry into per-residue embeddings `h_V (B, L, hidden_dim)`. The decoder head, training loop, loss, and evaluation are fixed by the scaffold.

## Prior art / Background / Baselines

Current approaches encode structure in different ways and leave different gaps.

- **Voxelized 3D-CNN models.** Rasterize atoms into a 3D occupancy grid and apply 3D convolutions. Gap: the grid is not orientation-invariant without augmentation, resolution trades against memory, and the residue-residue graph structure is discarded.
- **Sequential / hand-crafted-feature models.** Summarize each residue's 3D environment with hand-built features and feed a 1D sequence model. Gap: geometry not captured by the chosen features is unavailable to the network.
- **Message passing on a proximity graph.** Form messages over a `k`-nearest-neighbor graph and aggregate at each node. Gap: relational reasoning and permutation invariance are cheap, but the design of invariant node/edge features and update rules remains open.
- **Graph encoders with invariant scalar geometry.** Pin a local frame per node and encode edges with distance, normalized relative direction, and relative orientation. Gap: geometry is collapsed into invariant scalars before message passing, so later layers cannot update the representation of direction.

## Fixed substrate / Code framework

The inverse-folding harness is frozen and must not be changed. It supplies: datasets and structure-based train/validation/test splits; padding/masking of variable-length proteins into `X (B, L, 4, 3)`, sequence indices `S (B, L)`, and a residue `mask (B, L)`; the training loop (`AdamW`, `betas=(0.9, 0.98)`, `OneCycleLR`, gradient-norm clip 1.0, per-residue masked cross-entropy); NaN-coordinate handling; and evaluation reporting **recovery** (argmax-correct fraction) and **perplexity** (`exp` of the mean per-residue NLL).

Reusable geometric helpers above the editable slot:

- `_rbf(D, ...)` — lift distances into 16 Gaussian radial basis functions (centers 0–20 Å).
- `_dihedrals(X)` — backbone dihedrals `(φ, ψ, ω)` as `{sin, cos}`, `(B, L, 6)`.
- `_orientations(X)` — local forward + binormal unit vectors, `(B, L, 6)`.
- `knn_graph(X_ca, mask, k)` — build the `k`-nearest-neighbor graph from `CA` coordinates, returning `E_idx (B, L, K)` and `D_neighbors (B, L, K)`.

## Editable interface

Only the `StructureEncoder` and `InverseFoldingModel` classes are editable, plus an optional `CONFIG_OVERRIDES` dict that may set only `learning_rate`, `dropout`, `num_encoder_layers`, and `batch_size`. The contract is fixed: `StructureEncoder.forward(X, mask)` returns `h_V (B, L, hidden_dim)`; `InverseFoldingModel.forward` returns `log_probs (B, L, 20)`. The scaffold does not provide autoregressive decoding, sequence input, multi-chain encoding, or coordinate-noise augmentation: the model sees only backbone geometry and must produce all 20 marginals in one shot.

The starting fill is a plain MPNN encoder (KNN graph, dihedral + orientation node features, RBF + direction edge features, three message-passing layers) feeding a two-layer MLP decoder.

```python
# EDITABLE region of custom_invfold.py — default fill (plain MPNN encoder + MLP decoder)
class MPNNEncoderLayer(nn.Module):
    """Message Passing Neural Network layer for protein graphs."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_msg = nn.Sequential(                       # edge message network
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_node = nn.Sequential(                      # node update network
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, E_idx, mask):
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        D = int(h_V.shape[-1])
        h_V_neighbors = torch.gather(
            h_V.unsqueeze(2).expand(-1, -1, K, -1), 1,
            E_idx.unsqueeze(-1).expand(-1, -1, -1, D))    # (B, L, K, D)
        h_V_expand = h_V.unsqueeze(2).expand_as(h_V_neighbors)
        msg_input = torch.cat([h_V_expand, h_V_neighbors, h_E], dim=-1)
        messages = self.W_msg(msg_input)                  # (B, L, K, D)
        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
                                   E_idx.clamp(0, L - 1)).unsqueeze(-1)
        messages = messages * mask_attend
        agg = messages.sum(dim=2) / (mask_attend.sum(dim=2).clamp(min=1))
        h_V = self.norm1(h_V + self.dropout(agg))
        h_V_upd = self.W_node(torch.cat([h_V, agg], dim=-1))
        h_V = self.norm2(h_V + self.dropout(h_V_upd))
        h_V = h_V * mask.unsqueeze(-1)
        return h_V


class StructureEncoder(nn.Module):
    """GNN encoder: dihedral+orientation node features, RBF+direction edge features."""

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf
        node_input_dim = 12                               # dihedrals (6) + orientation (6)
        edge_input_dim = num_rbf + 3                      # RBF distance + direction
        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNEncoderLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, X, mask):
        B, L = X.shape[0], X.shape[1]
        X_ca = X[:, :, 1, :]
        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        K = E_idx.shape[2]
        dihedrals = _dihedrals(X)                         # (B, L, 6)
        orientations = _orientations(X)                   # (B, L, 6)
        node_feat = torch.cat([dihedrals, orientations], dim=-1)
        rbf = _rbf(D_neighbors, device=X.device)          # (B, L, K, num_rbf)
        X_ca_neighbors = torch.gather(
            X_ca.unsqueeze(2).expand(-1, -1, K, -1), 1,
            E_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        direction = F.normalize(X_ca_neighbors - X_ca.unsqueeze(2), dim=-1)
        edge_feat = torch.cat([rbf, direction], dim=-1)   # (B, L, K, num_rbf+3)
        h_V = self.node_embed(node_feat)
        h_E = self.edge_embed(edge_feat)
        for layer in self.layers:
            h_V = layer(h_V, h_E, E_idx, mask)
        return h_V


class InverseFoldingModel(nn.Module):
    """Encoder (editable) + a two-layer MLP decoder to amino-acid log-probabilities."""

    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(hidden_dim=hidden_dim, num_layers=num_encoder_layers,
                                        k_neighbors=k_neighbors, dropout=dropout, num_rbf=num_rbf)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA))

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```

## Evaluation settings

Three benchmarks: **CATH 4.2** (~18k train / 608 test, structure-split), **CATH 4.3** (~21k train / 1120 test), and **TS50** (50 de novo designed proteins, trained on CATH 4.2 with overlapping training proteins removed). Primary metric: **recovery** (argmax-correct fraction). Secondary metric: **perplexity**. One seed (42). Fixed compute/time budget per benchmark.
