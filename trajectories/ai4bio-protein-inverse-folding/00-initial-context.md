## Research question

Protein inverse folding (fixed-backbone design): given a backbone — for each of `L` residues the
3D coordinates of its four backbone atoms `N, CA, C, O` — predict a per-residue distribution over the
20 standard amino acids that would fold into that shape. The map is degenerate (many sequences fold to
nearly the same backbone), so the target is `p(s | X)`, not a deterministic inverse; and because a
protein's identity is unchanged when the whole molecule is rotated, reflected, or translated, the
prediction must be **invariant** to rigid motion. The single component being designed is the
**structure encoder**: the GNN module that turns backbone geometry into per-residue embeddings
`h_V` of shape `(B, L, hidden_dim)`. Everything downstream — a decoder head to amino-acid logits, the
training loop, the loss, the evaluation — is fixed by the scaffold. The encoder is the critical
component because all methods share the same input (backbone coordinates) and output (amino-acid
log-probabilities) and differ only in how they transform structure into sequence-informative
representations.

## Prior art before the first rung (the structure-graph lineage)

The first rung reacts to a line of structure-encoding methods that each capture one face of the problem
and surrender the other. The two faces are **geometry** (which way a residue points, how the backbone
curves, whether two neighbors sit on the same side of me or opposite sides — directions and angles in
space) and **relation** (who contacts whom, the connectivity pattern, the order along the chain). The
ancestors below are what the ladder is built against.

- **Voxelized 3D-CNN models (Anand et al. 2020; quality-assessment Ornate, 2019).** Rasterize atoms
  into a 3D occupancy grid and run a 3D convolution; filters latch onto pocket shapes and motifs
  directly in space. Gap: the grid is not invariant to how the molecule is oriented (one re-orients by
  augmentation), resolution trades memory against geometric precision, and the residue-residue graph is
  thrown away, so relational reasoning has to be re-learned through dense volumetric filters.
- **Sequential / hand-crafted-feature models (SPIN2, O'Connell et al. 2018).** Summarize each residue's
  3D environment as a hand-built feature vector and feed the protein to a 1D-CNN/RNN/dense net. Gap: the
  structure is represented only indirectly, through whatever the features happen to capture; geometry
  the features discard is unrecoverable downstream.
- **Message passing on a proximity graph (Gilmer et al. 2017).** The substrate everything here is built
  on: form a message per directed edge `m_{j→i} = g(h_i, h_j, e_{j→i})`, aggregate the incoming
  messages at each node, update `h_i`. The neighbor sum is permutation-symmetric, so graph-index
  invariance is free; relational reasoning is native and `O(L·k)` cheap on a `k`-nearest-neighbor graph.
  What it leaves open — and what every method below answers differently — is *what the node/edge
  features carry* and *how a step transforms them while staying invariant to rigid motion*.
- **Graph encoders with invariant scalar geometry (Structured Transformer, Ingraham et al. 2019).** The
  CPD state of the art before the ladder. Diagnosed that distance-only edges are *not locally
  informative* — two neighbors at equal distance can be on the same or opposite side and the scalar
  can't tell them apart — and fixed it by pinning a local frame `O_i = [b_i, n_i, b_i×n_i]` per node and
  encoding each edge as `(RBF(‖x_j−x_i‖), O_i^T(x_j−x_i)/‖x_j−x_i‖, q(O_i^T O_j))`. Gap: all geometry is
  collapsed into invariant scalars *at the input*; after that first projection the network can no longer
  touch the directional quantities as geometric objects, and orientation is stored redundantly (once per
  neighbor frame rather than once, absolutely, per node).

So the table is set with a tension: CNNs reason geometrically but discard the graph and the invariance;
graph encoders reason relationally and cheaply but, to stay invariant, freeze the geometry into scalars
at the door. The ladder is the search for an encoder that reasons geometrically *and* relationally,
invariantly, at a cost that scales to whole proteins.

## The fixed substrate

A self-contained inverse-folding harness is frozen and must not be touched. It supplies: the datasets
and structure-based train/validation/test splits; padding/masking of variable-length proteins into
`X (B, L, 4, 3)`, sequence indices `S (B, L)`, and a residue `mask (B, L)`; the training loop
(`AdamW`, `betas=(0.9, 0.98)`, `OneCycleLR`, gradient-norm clip 1.0, per-residue masked cross-entropy);
NaN-coordinate handling (masked out); and the evaluation that reports **recovery** (argmax-correct
fraction) and **perplexity** (`exp` of the mean per-residue NLL). The harness also exposes four
geometric helpers in the fixed region above the editable slot, which any encoder may reuse:

- `_rbf(D, ...)` — lift distances into 16 Gaussian radial basis functions (centers 0–20 Å).
- `_dihedrals(X)` — backbone dihedrals `(φ, ψ, ω)` as `{sin, cos}`, `(B, L, 6)`.
- `_orientations(X)` — local forward + binormal unit vectors, `(B, L, 6)`.
- `knn_graph(X_ca, mask, k)` — build the `k`-nearest-neighbor graph from `CA` coordinates, returning
  `E_idx (B, L, K)` and `D_neighbors (B, L, K)`.

## The editable interface

Exactly one region is editable: the `StructureEncoder` and `InverseFoldingModel` classes (and an
optional `CONFIG_OVERRIDES` dict that may set only `learning_rate`, `dropout`, `num_encoder_layers`,
`batch_size`). The contract is fixed: `StructureEncoder.forward(X, mask)` takes `X (B, L, 4, 3)` and
`mask (B, L)` and returns per-residue embeddings `h_V (B, L, hidden_dim)`; `InverseFoldingModel.forward`
wraps the encoder with a decoder head and returns `log_probs (B, L, 20)`. Every rung is a fill of this
same contract. Note one thing the scaffold does *not* expose, which constrains every method here:
there is **no autoregressive decoder, no sequence input to the model, no chain/multi-chain encoding,
and no coordinate-noise augmentation** — the model sees only backbone geometry and must produce all 20
marginals in **one shot** from the encoder embeddings through a small decoder head.

The starting point is the scaffold default: a plain MPNN encoder (KNN graph, dihedral + orientation
node features, RBF + direction edge features, three message-passing layers) feeding a two-layer MLP
decoder.

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

Three benchmarks. **CATH 4.2** — the standard single-chain design benchmark, structure-split so held-out
folds are dissimilar to training (~18k train / 608 test). **CATH 4.3** — an updated, more diverse CATH
(~21k train / 1120 test). **TS50** — 50 de novo designed proteins for out-of-distribution
generalization, trained on CATH 4.2 with TS-overlapping training proteins removed. Primary metric:
**recovery**, the fraction of residues whose argmax prediction matches the native amino acid (higher is
better). Secondary metric: **perplexity**, `exp` of the mean per-residue cross-entropy (lower is
better). One seed (42). Each method is trained under a fixed compute/time budget per benchmark.
