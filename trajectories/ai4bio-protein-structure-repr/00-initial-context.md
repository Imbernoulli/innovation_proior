## Research question

Protein function follows from the folded 3D shape, not from the bare sequence. I am handed, for each residue, a single alpha-carbon coordinate in space and the residue identity, and I must produce informative embeddings — one vector per residue and one vector per protein — that a fixed downstream classifier head can read to predict enzyme commission number, gene-ontology biological process, and fold. The single thing being designed is the **geometric GNN encoder**: the map from `(pos, node_feat, batch)` to `(node_emb, graph_emb)`. Everything else — how the node features are computed from coordinates, how proteins are batched, the classifier heads, the training and evaluation loops, and the metrics — is fixed and must not be touched.

The encoder must be **invariant to rigid motion**: translate, rotate, or reflect a protein and it is the same protein, so the embeddings must not change. The encoder must also **use the 3D arrangement** — the distances and angles between residues — and capture both local backbone geometry and the global fold, rather than collapsing to who-is-bonded-to-whom along the chain.

## Prior art / Background / Baselines

- **Hand-crafted symmetry functions (Behler & Parrinello 2007).** The energy is written as a sum over atoms, each atom energy given by a small net on a fixed vector of radial and angular descriptors of its local environment, smoothed by a cutoff.
- **Deep tensor / message-passing nets (Schütt et al. 2017; Gilmer et al. 2017).** Per-node embeddings are refined by summed messages with geometry entering through a learned coupling.
- **Steerable / higher-order equivariant nets (Tensor Field Networks; SE(3)-Transformer).** Type-1 (and higher) features are carried through every layer with spherical harmonics and Clebsch–Gordan coefficients, so the layer commutes with rotation by construction and can emit vectors as well as scalars.

## Fixed substrate / Code framework

The pipeline around the encoder is frozen. `compute_node_features(pos, aa_idx, batch)` produces the 28-dim scalar node features once, from coordinates: a 20-dim amino-acid one-hot, a 2-dim sin/cos sequence positional encoding, and 6 pseudo-dihedral features (cosine/sine of the dihedral, and the two backbone segment lengths and the two bond-angle cosines around each residue). `SCALAR_NODE_DIM = 28`. The dataset construction reads alpha-carbon positions and residue indices into PyG `Data` objects; batching is `Batch.from_data_list`; the classifier is a fixed three-layer MLP head on the *graph* embedding (`out_dim → 256 → 256 → num_classes`); the optimizer is Adam (`lr=1e-3`, `weight_decay=1e-4`), cosine-annealed, with global grad-norm clip 1.0, batch size 32, 50 epochs, and the best-val checkpoint is scored on test. Loss and metric are chosen by task type (cross-entropy / accuracy for the multiclass EC and Fold tasks; BCE-with-logits / threshold-swept `f1_max` for the multilabel GO-BP task).

The loop also supplies, already imported in the module's fixed header, the geometry helpers an encoder may use: `knn_graph`, `radius_graph`, `global_mean_pool`, `global_add_pool` from `torch_geometric.nn`; `scatter_mean`, `scatter_add` from `torch_scatter`; and `add_self_loops`. The encoder is handed raw coordinates, so it builds its own edges from `pos` and `batch`.

## Editable interface

Exactly one region is editable — the block between `EDITABLE SECTION START` and `EDITABLE SECTION END` in `custom_protein_encoder.py`. It must define a `ProteinEncoder` class with the contract

```
__init__(self, input_dim=SCALAR_NODE_DIM, hidden_dim=256, out_dim=128,
         num_layers=6, dropout=0.1, cutoff=10.0, max_neighbors=16)
forward(self, pos, node_feat, batch) -> (node_emb, graph_emb)
    pos:       (N, 3) alpha-carbon coordinates
    node_feat: (N, 28) fixed scalar node features
    batch:     (N,) batch assignment
    returns    node_emb (N, out_dim), graph_emb (B, out_dim)
```

Any helper layers/classes may be defined inside the region. The training loop constructs the encoder with the fixed signature, so any method-specific widths are set *inside* `__init__` (overriding the defaults), not via new constructor arguments. The starting point is the scaffold default: a basic invariant message-passing GNN — a kNN graph with distance+direction edge features, an edge MLP that mixes the two endpoints with the edge feature, mean aggregation, a residual node MLP with LayerNorm, and mean pooling for the graph embedding. Each rung replaces exactly this region.

```python
# EDITABLE region of custom_protein_encoder.py — default fill (basic invariant message passing)
class MessagePassingLayer(nn.Module):
    """Basic invariant message passing layer for protein graphs."""

    def __init__(self, hidden_dim, edge_dim=EDGE_FEAT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msg = self.edge_mlp(edge_input)
        agg = scatter_mean(msg, dst, dim=0, dim_size=h.size(0))      # invariant aggregation
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_new)                                     # residual + norm
        return h


class ProteinEncoder(nn.Module):
    """Geometric GNN encoder over alpha-carbon graphs (basic invariant message passing)."""

    def __init__(self, input_dim: int = SCALAR_NODE_DIM, hidden_dim: int = 256,
                 out_dim: int = 128, num_layers: int = 6, dropout: float = 0.1,
                 cutoff: float = 10.0, max_neighbors: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, EDGE_FEAT_DIM) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def _build_edges(self, pos, batch):
        edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
        src, dst = edge_index
        diff = pos[dst] - pos[src]
        dist = diff.norm(dim=-1, keepdim=True)
        direction = diff / (dist + 1e-8)
        edge_attr = torch.cat([dist, direction], dim=-1)            # (E, 4): distance + unit direction
        return edge_index, edge_attr

    def forward(self, pos, node_feat, batch):
        edge_index, edge_attr = self._build_edges(pos, batch)
        h = self.node_embed(node_feat)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)
        node_emb = self.out_proj(h)
        graph_emb = global_mean_pool(node_emb, batch)
        return node_emb, graph_emb
```

## Evaluation settings

Three downstream benchmarks, each a single fixed seed (42): **EC reaction** (384-class multiclass, metric top-1 **accuracy**), **GO biological process** (1943-class multilabel, metric **f1_max** — the maximum F1 over a sweep of decision thresholds), and **Fold classification** (1195-class multiclass, metric top-1 **accuracy**). Higher is better for all three; the scoring code also records test loss. The same encoder architecture is trained and evaluated independently on each benchmark.
