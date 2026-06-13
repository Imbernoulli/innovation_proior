**Problem.** Encode a protein from alpha-carbon coordinates into invariant per-residue and per-protein
embeddings. The scaffold default is invariant only by the head's good graces (its edge feature carries a
unit direction vector); I want the cheapest encoder whose invariance is structural and whose geometry
filter is *learned*, as the honest floor of the ladder.

**Key idea (continuous-filter convolution).** Feed messages only the rotation/translation/reflection-
invariant distance `d_ij = ‖pos_i - pos_j‖`. Replace any binned distance with a *learned* filter
network of the continuous distance: expand `d_ij` in a Gaussian RBF bank (to decorrelate the filter
channels and kill the init plateau), pass it through a small net with a smooth shifted-softplus
activation, fade it with a cosine cutoff (value and slope zero at the cutoff), and convolve feature-wise
over neighbors. Stack six **unshared residual** interaction blocks so strictly radial pairwise filters
compose into many-body structure; sum-pool for the graph embedding.

**Why it works.** Distance-only geometry makes the encoder E(3)-invariant by construction (no
equivariant channel, nothing to prove per layer). The learned continuous filter beats hand-built radial
descriptors; the Gaussian expansion gives the filter net diverse channels at init; depth turns pairwise
filters into multi-scale structure. It is a learned, invariant geometric encoder — strictly better than
the default's hope-based invariance.

**This task's edit, not the molecular model.** Uses the library's `InteractionBlock`, `GaussianSmearing`,
`ShiftedSoftplus`. No energy and no forces — the encoder returns embeddings, so the
twice-differentiability motivation for the smooth activation is absent; I keep `ssp` as the canonical
choice. Builds its own **kNN graph** (`k = max_neighbors = 32`), not a radius graph. Node-feature
embedding is `Linear(28, hidden)` (not a per-element table). Output head is `lin1 (hidden→hidden) → ssp →
lin2 (hidden→out_dim)`; graph embedding is sum-scatter.

**Hyperparameters.** `hidden_channels=512`, `num_filters=128`, `num_gaussians=50`, `cutoff=10.0`,
`max_num_neighbors=32`, `num_layers=6`, `readout="add"`.

**What to watch.** A radial encoder is blind to direction — geometries with the same neighbor-distance
multiset are indistinguishable. Expect decent EC (sequence/active-site correlated), middling GO-BP
(coarse f1_max), and the *lowest* number on Fold, where directional fold discrimination is needed. That
Fold ceiling is what forces an equivariant directional channel at the next rung.

```python
# =====================================================================
# EDITABLE SECTION START — SchNet encoder (ported from ProteinWorkshop)
# =====================================================================

# Import PyG SchNet components used by the reference implementation
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing, ShiftedSoftplus

class ProteinEncoder(nn.Module):
    """SchNet-based protein structure encoder.

    Ported directly from ProteinWorkshop SchNetModel.
    Uses continuous-filter convolutions with Gaussian RBF distance expansion.
    Invariant to rotations and translations by design.

    Reference hyperparameters (from proteinworkshop/config/encoder/schnet.yaml):
      hidden_channels=512, num_filters=128, num_gaussians=50, cutoff=10.0,
      max_num_neighbors=32, readout="add"
    """
    def __init__(
        self,
        input_dim: int = SCALAR_NODE_DIM,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        cutoff: float = 10.0,
        max_neighbors: int = 16,
    ):
        super().__init__()
        # Override with ProteinWorkshop reference hyperparameters
        hidden_channels = 512
        num_filters = 128
        num_gaussians = 50
        self.cutoff = cutoff
        max_num_neighbors = 32
        readout = "add"

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout

        # Overwrite embedding to accept arbitrary input features (matching reference LazyLinear)
        self.embedding = nn.Linear(input_dim, hidden_channels)

        # Gaussian RBF distance expansion (from PyG SchNet)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Stack of InteractionBlocks (from PyG SchNet)
        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # Output MLP: lin1 -> act -> lin2 (matching reference)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels, out_dim)

    def _build_edges(self, pos, batch):
        """Build kNN graph and compute edge weights + RBF features."""
        edge_index = knn_graph(
            pos, k=self.max_num_neighbors, batch=batch, loop=False
        )
        u, v = edge_index
        edge_weight = (pos[u] - pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        return edge_index, edge_weight, edge_attr

    def forward(self, pos, node_feat, batch):
        """Forward pass matching ProteinWorkshop SchNetModel.

        Args:
            pos: (N, 3) alpha-carbon coordinates
            node_feat: (N, input_dim) node scalar features
            batch: (N,) batch index

        Returns:
            node_emb: (N, out_dim) per-node embeddings
            graph_emb: (B, out_dim) per-graph embeddings
        """
        edge_index, edge_weight, edge_attr = self._build_edges(pos, batch)

        # Project input features to hidden dimension
        h = self.embedding(node_feat)

        # Message passing with residual connections (matching reference exactly)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Output projection: lin1 -> act -> lin2 (matching reference)
        h = self.lin1(h)
        h = self.act(h)
        node_emb = self.lin2(h)

        # Graph-level readout via scatter (matching reference readout="add")
        graph_emb = scatter_add(node_emb, batch, dim=0)

        return node_emb, graph_emb

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
```
