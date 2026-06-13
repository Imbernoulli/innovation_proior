**Problem (from step 3).** SEAL bought AUC and large-graph Hits but *regressed* the small-graph
ranking metrics (Cora MRR 31.2→27.5, Hits@20 70.3→61.2) — because its "structural" signal is the
absolute-difference of *learned embeddings*, not a measurement of the graph. Every rung so far scores a
pair only through the geometry of two independently-encoded points; none ever counts a shared neighbor.
A plain GNN cannot count triangles, so the common-neighbor family — the dominant link signal on these
graphs for two decades — has been structurally unrepresentable. That is the ceiling.

**Key idea (BUDDY-inspired, exact overlap).** Break the all-learned scoring: compute the canonical
neighborhood-overlap heuristics *exactly* against the live adjacency and hand them to the decoder.
Common neighbors CN = |N(u) ∩ N(v)| (= the A[1,1] distance-label count / triangle count), Adamic–Adar
AA = Σ_{w ∈ N(u)∩N(v)} 1/log deg(w), resource allocation RA = Σ 1/deg(w). These are pair-relative, so
they separate the automorphic-node links the embedding-only decoders could not. Project [CN, AA, RA] to
the hidden width and concatenate with the two embeddings:
$h = [z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,\text{proj}(\text{CN},\text{AA},\text{RA})]$, then MLP
$3H\to H\to H\to 1$. The MLP learns *which* degree-discounting the data prefers.

**What the harness omits vs. the full method.** No MinHash / HyperLogLog sketching and no multi-distance
count table A[d_u, d_v]: at ≤236k nodes the overlaps are computed *exactly* with scipy sparse (CSR row
slices, sparse elementwise product → CN/AA/RA), which is feasible and tighter than sketch estimates. The
sketches exist in the full method to make the *same* counts scale to millions of nodes — dropped here as
unnecessary. Fusion is concat+projection rather than the full method's Hadamard edge-pooling, because
the counts are a different kind of quantity than learned coordinates. Overlap runs under `no_grad`
(fixed structural measurements); the harness passes the correct adjacency per phase (train-only at
validation, train+val at test, OGB protocol).

**Why it should clear SEAL.** CN/AA/RA are exactly the top-of-list signal SEAL's embedding-difference
feature only approximated, so the regressed small-graph MRR/Hits@20 should recover and pass gcn_dot, and
ogbl-collab Hits@50 should pass SEAL's 57.88 into the low-mid sixties — CN/AA/RA being *the* dominant
signal on a dense collaboration graph. AUC should hold (never the failing metric); a drop on
Cora/CiteSeer would warn the structural features are overwhelming the node features.

**Hyperparameters.** `hidden_channels=256`, `num_layers=2`, `dropout=0.0`, struct features = 3
(CN, AA, RA) → `struct_proj` to H; decoder $3H\to H\to H\to 1$, BatchNorm on all encoder layers, Adam
`lr=0.01`, BCE, 200 epochs, patience 20.

```python
class StructuralFeatureComputer:
    """Precomputes structural pairwise features (approximating BUDDY sketches)."""

    @staticmethod
    @torch.no_grad()
    def compute_cn_features(edge_index, num_nodes, edge_label_index):
        """Compute CN/AA/RA features using scipy sparse (memory-efficient)."""
        import scipy.sparse as sp
        device = edge_label_index.device

        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        adj = sp.csr_matrix((np.ones(len(row)), (row, col)),
                            shape=(num_nodes, num_nodes))

        src = edge_label_index[0].cpu().numpy()
        dst = edge_label_index[1].cpu().numpy()

        # Sparse row extraction + element-wise multiply stays sparse
        src_rows = adj[src]   # [batch, N] sparse
        dst_rows = adj[dst]   # [batch, N] sparse
        common = src_rows.multiply(dst_rows)  # sparse intersection

        deg = np.asarray(adj.sum(axis=1)).flatten().clip(min=1)
        cn = np.asarray(common.sum(axis=1)).flatten()
        aa = np.asarray(common.multiply(1.0 / np.log(deg).clip(min=1.0))
                        .sum(axis=1)).flatten()
        ra = np.asarray(common.multiply(1.0 / deg).sum(axis=1)).flatten()

        return torch.tensor(np.stack([cn, aa, ra], axis=1),
                            dtype=torch.float32, device=device)


class LinkPredictor(nn.Module):
    """BUDDY-inspired link predictor.

    Combines GCN node embeddings with precomputed structural features
    (common neighbors, Adamic-Adar, resource allocation) via an MLP decoder.
    This approximates BUDDY's subgraph sketching approach.

    The new decode interface takes `edge_label_index` (original node
    indices) and the full embedding table `z` directly, so we no longer
    need to recover indices via hashing/argmax.  The training graph
    `edge_index` is also passed through, enabling exact CN/AA/RA
    computation against whichever adjacency is in use (train-only during
    validation, train+val during final test, as OGB prescribes).
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN encoder
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        # Structural feature dimension: CN, AA, RA = 3
        struct_dim = 3
        self.struct_proj = nn.Linear(struct_dim, hidden_channels)

        # MLP decoder: node features + structural features
        dec_in = hidden_channels * 2 + hidden_channels  # src, dst, struct
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Cached context set at encode-time so decode() has sensible
        # defaults when the caller does not pass edge_index explicitly.
        self._edge_index = None
        self._num_nodes = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self._edge_index = edge_index
        self._num_nodes = x.size(0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        # Resolve the adjacency to use for structural features.
        ei = edge_index if edge_index is not None else self._edge_index
        N = num_nodes if num_nodes is not None else (
            self._num_nodes if self._num_nodes is not None else z.size(0))

        with torch.no_grad():
            struct_feats = StructuralFeatureComputer.compute_cn_features(
                ei, N, edge_label_index)
        struct_h = self.struct_proj(struct_feats.float())

        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        h = torch.cat([z_src, z_dst, struct_h], dim=-1)
        return self.decoder(h).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z,
                           edge_index=edge_index, num_nodes=x.size(0))
```
