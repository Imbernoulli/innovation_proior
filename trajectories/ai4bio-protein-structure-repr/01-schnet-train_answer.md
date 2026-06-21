I begin the ladder from the scaffold's default fill — a basic invariant message-passing GNN whose edge feature carries the unit direction $(\mathrm{pos}_{dst}-\mathrm{pos}_{src})/d$ — and the way it is invariant is exactly what I want to fix. A direction is *not* rotation-invariant; rotate the protein and it rotates with it. The default survives only because the downstream classifier could, in principle, learn to ignore the directional channels — invariance by hope, not by construction, the same "learn the symmetry from data" trap that has no place at the floor of the ladder. So my goal for the first rung is the cheapest encoder whose invariance is *structural*, where every geometric quantity that touches a message is invariant on its own, and where the filter that reads geometry is *learned* rather than hand-built.

I propose the continuous-filter encoder, **SchNet**. Two facts about rigid motion fix the whole design. An orthogonal matrix $Q$ preserves inner products, hence norms: $(Qa)\cdot(Qb)=a\cdot b$. A shared translation cancels in a difference: $(a+g)-(b+g)=a-b$. Put them on a pair of residues: the relative difference $\mathrm{pos}_i-\mathrm{pos}_j$ is translation-invariant but rotates with $Q$, while its norm, the distance $d_{ij}=\lVert\mathrm{pos}_i-\mathrm{pos}_j\rVert$, is invariant to translation, rotation, *and* reflection — a genuine E(3)-invariant scalar built purely from geometry. So if the only geometric input any message ever sees is $d_{ij}$, the whole encoder is invariant by construction, with no equivariant coordinate channel and nothing to prove per layer. That is the cheap route, and for the floor I take it without hesitation: feed messages only distances.

What makes this more than a plain distance-GNN is *how* the distance enters the message. The blunt option is to bin it — a lookup table indexed by which distance bucket $d_{ij}$ falls into, the molecular cousin of a voxel grid or a one-hot bond type — but a table over a discrete index reintroduces exactly the discreteness I am avoiding: as a residue drifts and $d_{ij}$ crosses a bin boundary, its contribution snaps from one tap to the next and the encoder's function of geometry has a jump. So I replace the table with a *function* of the continuous distance: a small neural network $W(\cdot)$ that maps the scalar $d_{ij}$ to a vector of filter values, convolved over neighbors by an elementwise (feature-wise) product,
$$m_i=\sum_{j}(\mathrm{lin}_1\,h_j)\circ W(d_{ij}).$$
There is no grid anymore; a residue at any distance contributes through $W$ evaluated at its exact $d_{ij}$, and as it moves, $W(d_{ij})$ moves continuously. This is the continuous-filter convolution, the conceptual core of the rung. I make the filter feature-wise rather than a full matrix so it costs $O(\text{num\_filters})$ per edge instead of $O(F^2)$, leaving cross-channel mixing to ordinary per-node linear layers around it — geometry in the filter, feature recombination in the dense layers.

There is a sharp practical failure to design around before this trains at all. The filter network takes the single scalar $d_{ij}$ and must emit many filter channels; at initialization a net is nearly linear, so each output channel is approximately the *same* linear ramp in $d_{ij}$, just scaled — the channels are almost identical, carrying one effective degree of freedom instead of many, a flat plateau with no filter diversity to exploit. The fix is to lift the scalar into a representation where different channels naturally see different things: expand the distance in a bank of Gaussians,
$$e_k(d)=\exp\!\big(-\gamma\,(d-\mu_k)^2\big),$$
with centers $\mu_k$ on a uniform grid from $0$ to the cutoff and width matched to the spacing ($\gamma=1/(2\Delta^2)$). Now a given distance lights up the few Gaussians whose centers are near it and leaves the rest near zero, so even a near-linear filter net produces diverse filters — different channels latch onto different distance ranges. The number of centers is the filter's resolution, the span of the centers its size, and the expansion is itself smooth, so no discontinuity is reintroduced.

The nonlinearity matters more than usual. The cleanest activation for a geometric feature network is one smooth to all orders, since the function may be differentiated through and I do not want kinks in the geometry response. The shifted softplus, $\mathrm{ssp}(x)=\mathrm{softplus}(x)-\ln 2=\ln(0.5\,e^x+0.5)$, is the $C^\infty$ cousin of ReLU: it bends instead of cornering, and $\mathrm{ssp}(0)=0$ so zero pre-activations map to zero and the activations stay centered. I use it in the filter net and in the per-node layers. I also fold a smooth cosine cutoff into the filter, $f_\text{cut}(d)=0.5\,[1+\cos(\pi d/\text{cutoff})]$, whose value *and* slope are both zero at the cutoff so a neighbor crossing the boundary fades out without a jump; the effective filter is $W(d)=\text{filter\_net}(e(d))\cdot f_\text{cut}(d)$.

Depth is what turns strictly pairwise radial filters into genuinely many-body structure, and it is the payoff that makes a deep stack worth it. One convolution lets residue $i$ feel each neighbor $j$ individually — pairwise. But after one block, $h_j$ has already absorbed information about $j$'s own neighbors $k$, so in the next block, when $i$ pulls in $h_j$, it implicitly pulls in something that knows about $k$ — $i$ feels the $(i,j,k)$ triple. A few blocks and a residue's representation reflects its spatial environment several hops out, all while every individual filter only ever looked at a single invariant distance. The depth manufactures the many-body character; the invariance is never spent, because each filter is radial. I wrap each block in a residual connection, $h\leftarrow h+\text{InteractionBlock}(h)$, so the deep stack stays trainable and early local features survive to the output, and I give each block its own weights (unshared) so earlier blocks build short-range structure and later blocks build on it.

Made concrete on this task's edit surface, the implementation departs from the generic continuous-filter model in ways worth stating, because the harness fixes choices the molecular version leaves open. I do not hand-roll the convolution; I use the geometry library's components — the interaction block (continuous-filter convolution plus shifted softplus plus a linear), the Gaussian smearing, and the shifted-softplus activation — wired in the same residual pattern. There are no forces and no energy here: this is an *encoder*, so the landing artifact is the node and graph embeddings, not a scalar with an autograd gradient. The whole twice-differentiable, force-conserving story that originally motivated the smooth activation is therefore absent; I keep $\mathrm{ssp}$ anyway as the canonical choice that costs nothing. The encoder is handed raw $\mathrm{pos}$ and $\mathrm{batch}$ and builds its own edges, so I build a **kNN graph** with $k=\text{max\_neighbors}$, not a radius graph and not a provided adjacency; the edge weight is the Euclidean distance on those kNN edges and the edge attribute is the Gaussian expansion of that distance. I override the constructor defaults to the reference configuration — $\text{hidden\_channels}=512$, $\text{num\_filters}=128$, $\text{num\_gaussians}=50$, $\text{cutoff}=10.0$, $\text{max\_num\_neighbors}=32$, six interaction layers, and an `add` (sum) readout. The output head is a linear from hidden to hidden, a shifted softplus, then a linear to $\text{out\_dim}$, and the graph embedding is the sum-scatter of the node embeddings over the batch. The per-element embedding table of the molecular model becomes a plain `Linear(input_dim, hidden_channels)` over the 28-dim node features, because the input is a feature vector rather than a single atom type.

I expect this floor to earn real traction as a learned, invariant geometric encoder, but to be capped by its one structural blindness: its entire signal is scalar-radial, so it cannot see *direction*. Two geometries presenting the same multiset of neighbor distances are indistinguishable to it — a single ring and two smaller rings with matching bond lengths look identical from every node — and in a folded protein that degeneracy is exactly the kind that matters, since whether a residue's contacts go off together or apart is a distance-blind quantity that helps separate folds. So I expect decent EC (active-site and sequence-composition correlated), a middling-but-real GO-BP (coarse threshold-swept f1\_max), and the *lowest* number on Fold, where fine directional discrimination is needed. That Fold ceiling is the radial blindness made visible, and it is the bar the next rung's equivariant directional channel must clear.

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
