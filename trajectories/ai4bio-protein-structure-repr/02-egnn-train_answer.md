The continuous-filter floor told me its ceiling in three numbers. EC came in at 0.589 accuracy — respectable, since the enzyme class correlates with active-site geometry and the amino-acid composition the 28-dim node features already carry. GO-BP landed at 0.245 f1\_max — a real but middling number from a coarse threshold-swept multilabel metric. But Fold came in at 0.184, far the lowest of the three, and that is the tell: distinguishing among ~1200 folds is precisely the task where two folds can present the same multiset of neighbor distances and differ only in *direction*, and a distance-only encoder cannot tell them apart no matter how deep the stack. SchNet's whole signal is scalar-radial; the Fold number is the radial ceiling made visible. The diagnosis is not a width or depth problem — it is that the encoder has no channel that carries and *transforms* directional information through its layers. The fix has to add exactly that channel without giving up the invariance that bought SchNet its honesty.

I propose **EGNN**, E(n)-equivariant message passing. Return to the two facts the rigid-motion constraint gives. The relative difference $\mathrm{pos}_i-\mathrm{pos}_j$ is translation-invariant and rotates with an orthogonal $Q$ — it transforms as a type-1 vector, exactly the way I want a directional quantity to transform. SchNet collapsed that difference immediately to its norm $d_{ij}$, and from that moment the type-1 structure was gone forever — which is *why* it is capped on Fold. So the move is to stop collapsing: keep a vector channel alive through the layers and update it equivariantly, while keeping the feature channel invariant the way SchNet did. I want both regimes on the same graph — invariant features $h$ (rotate the protein, $h$ is unchanged) and an equivariant coordinate channel (rotate the protein, the coordinates rotate with it).

The expensive way to carry type-1 information is the steerable route — higher-order steerable types, spherical harmonics, Clebsch–Gordan coefficients — but the harmonics are heavy to recompute per geometry and welded to three dimensions. I do not need that machinery; I need something equivariant the way the relative-difference vector itself is. Stare at that vector again. If I take a weighted sum $\sum_j w_{ij}\,(\mathrm{pos}_i-\mathrm{pos}_j)$ and rotate the input, it becomes $\sum_j w_{ij}\,Q(\mathrm{pos}_i-\mathrm{pos}_j)=Q\sum_j w_{ij}\,(\mathrm{pos}_i-\mathrm{pos}_j)$ — $Q$ factors cleanly out of the sum, *provided the weights $w_{ij}$ do not themselves change under rotation*. That proviso is the entire trick: if each weight is an invariant scalar, the combination of difference vectors is equivariant. And I already have an invariant scalar on every edge — the message $m_{ij}$, built SchNet-style from invariant features and the invariant distance. So let the coordinate weight be a scalar function of the message, $\varphi_x(m_{ij})$, and update
$$\mathrm{pos}_i\leftarrow \mathrm{pos}_i+\operatorname*{mean}_{j}\;\frac{\mathrm{pos}_i-\mathrm{pos}_j}{\lVert\mathrm{pos}_i-\mathrm{pos}_j\rVert+1}\,\varphi_x(m_{ij}).$$
Equivariance lives in the difference vectors, learning lives in the invariant scalar weight, and they meet only through a product (vector)·(scalar). The weight *must* be a scalar — if $\varphi_x$ emitted anything with directional structure, that thing would itself transform under $Q$ and the clean factoring would break, dragging me back toward the steerable machinery I am avoiding.

So the layer carries two updates. The message is the SchNet-style invariant edge function — concatenate the endpoint features with the invariant distance and pass through an edge MLP, $m_{ij}=\varphi_e\big(h_i,h_j,\lVert\mathrm{pos}_i-\mathrm{pos}_j\rVert\big)$. The feature update aggregates the messages and runs a node MLP with a residual, $h_i\leftarrow h_i+\varphi_h\big(h_i,\sum_j m_{ij}\big)$ — invariant in, invariant out, so the feature channel stays invariant exactly as in SchNet. The coordinate update is the new equivariant piece, and the two consistencies close into an induction: if $h$ is invariant entering the layer, the distance is invariant, so $m_{ij}$ is invariant, so $\varphi_x(m_{ij})$ is an invariant scalar, so the coordinate update is equivariant and the new $h$ is again invariant — a whole stack is invariant on features and equivariant on coordinates. That is the channel SchNet was missing, and it costs one scalar MLP and a weighted sum of vectors, not a basis of spherical harmonics.

Two stability details, because the difference vectors can misbehave. A bare sum over neighbors grows with degree, so the per-step displacement scale would depend on graph size; I aggregate the coordinate displacements with a *mean* rather than a sum to keep the displacement $O(1)$ regardless of degree (the feature messages keep the plain sum). And the raw difference $(\mathrm{pos}_i-\mathrm{pos}_j)$ can be large for distant pairs, so I normalize it by its own length plus one, $(\mathrm{pos}_i-\mathrm{pos}_j)/(\lVert\mathrm{pos}_i-\mathrm{pos}_j\rVert+1)$, so the learned weight scales essentially a bounded direction and the magnitude is governed by the weight, not by how far apart the residues happen to be. Dividing by a scalar function of the invariant distance does not touch equivariance — it is still $Q$ times a direction times an invariant scalar. Every nonlinearity lives on the invariant channels (inside $\varphi_e,\varphi_x,\varphi_h$, all consuming and producing invariant quantities); I never apply a pointwise nonlinearity to the coordinate, because a pointwise nonlinearity does not commute with $Q$ and would silently break equivariance.

Made concrete on this task's edit surface, several differences from the generic equivariant net are load-bearing. This task builds its own **kNN graph** from $\mathrm{pos}$ ($k=\text{max\_neighbors}=16$), so the equivariant layer runs on kNN edges, not a fully-connected point cloud and not a provided adjacency. The widths are overridden to the reference configuration — $\text{emb\_dim}=512$, six layers, ReLU activation, dropout 0.1, **batch normalization inside every MLP** ($\text{norm}=\text{batch}$), message aggregation $\text{sum}$, residual on both the feature channel and the coordinate channel. Several pieces of the general machinery are *omitted* by the harness and I do not import their story: there is no soft attention edge gate, no velocity channel, no tanh bound on the coordinate weight, and no special tiny initialization for the coordinate MLP — the layer is the plain message-passing port, the coordinate displacement weighted by `mlp_pos(msg)` on the normalized direction and mean-aggregated. The single biggest thing to be honest about: although the coordinates are updated equivariantly layer by layer, the **readout uses only the invariant feature channel** — the node embedding is `out_proj(h)` and the graph embedding is the *mean* pool of those node embeddings ($\text{pool}=\text{mean}$, not SchNet's sum). The updated coordinates are computed and propagated but never read out as the embedding. So the role of the equivariant coordinate channel here is not to emit a vector at the end; it is to let geometry and features exchange information richly inside the stack — the coordinate update feeds the next layer's distances, which feed the next messages, which feed the features that are finally read out. That is the mechanism by which directional, relative-geometry structure reaches the invariant embedding that SchNet's pure-radial layer could never give it.

I expect the delta from SchNet to show most clearly on **Fold**: equivariance should help most where directional fold discrimination matters most, so Fold should rise substantially above 0.184 — if it does not move, the equivariant channel is not actually injecting usable directional information through the kNN distances and the whole diagnosis is wrong. **EC** should also rise above 0.589 — active-site geometry benefits from directional structure too — but by less, because SchNet already did well there on radial-plus-sequence signal. **GO-BP** is the one I am least sure about: it is a coarse multilabel metric, and the extra coordinate machinery with batch-norm-heavy MLPs could fail to help or marginally regress if the GO signal is dominated by the sequence-composition node features rather than fine geometry — so I will watch whether GO-BP holds near or slightly below 0.245 rather than assuming it climbs. And if EGNN clears SchNet on Fold and EC but a single scalar weight per edge still cannot model how *two* edges of the same residue relate, the next rung is already implied: type the edges by what they mean and let relational structure, not just per-edge geometry, into the encoder.

```python
# =====================================================================
# EDITABLE SECTION START — EGNN encoder (ported from ProteinWorkshop)
# =====================================================================

import torch_scatter
from torch.nn import Linear, Dropout, Sequential
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    """E(n) Equivariant GNN Layer.

    Ported directly from ProteinWorkshop:
      proteinworkshop/models/graph_encoders/layers/egnn.py

    Paper: E(n) Equivariant Graph Neural Networks, Satorras et al. (ICML 2021)
    """
    def __init__(self, emb_dim, activation='relu', norm='batch', aggr='sum', dropout=0.1):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim

        # Normalization layer (matching reference)
        norm_cls = {
            'layer': nn.LayerNorm,
            'batch': nn.BatchNorm1d,
        }[norm]

        # Helper to create fresh activation instances
        def _make_act():
            if activation == 'relu':
                return nn.ReLU()
            elif activation in ('silu', 'swish'):
                return nn.SiLU()
            elif activation == 'elu':
                return nn.ELU()
            return nn.ReLU()

        # MLP psi_h for computing messages m_ij (matching reference exactly)
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 1, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
            Linear(emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
        )
        # MLP psi_x for computing coordinate displacement weights
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
            Linear(emb_dim, 1),
        )
        # MLP phi for computing updated node features
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
            Linear(emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
        )

    def forward(self, h, pos, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (2, e) - edge indices
        Returns:
            msg_aggr: (n, d) - updated node features delta
            pos_aggr: (n, 3) - coordinate displacement
        """
        msg_aggr, pos_aggr = self.propagate(edge_index, h=h, pos=pos)
        msg_aggr = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        return msg_aggr, pos_aggr

    def message(self, h_i, h_j, pos_i, pos_j):
        """Compute messages (matching reference exactly)."""
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1, keepdim=True)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale displacement vector by learned weight
        pos_diff = pos_diff / (dists + 1) * self.mlp_pos(msg)
        return msg, pos_diff

    def aggregate(self, inputs, index):
        """Aggregate messages and position displacements separately (matching reference)."""
        msgs, pos_diffs = inputs
        # Aggregate messages using configured aggr (sum in reference config)
        msg_aggr = torch_scatter.scatter(
            msgs, index, dim=self.node_dim, reduce=self.aggr
        )
        # Aggregate displacement vectors always with mean (matching reference)
        pos_aggr = torch_scatter.scatter(
            pos_diffs, index, dim=self.node_dim, reduce="mean"
        )
        return msg_aggr, pos_aggr

    def __repr__(self):
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class ProteinEncoder(nn.Module):
    """EGNN-based protein structure encoder.

    Ported directly from ProteinWorkshop EGNNModel.
    E(n)-equivariant: jointly updates node features and coordinates.
    Uses residual connections on both features and coordinates.

    Reference hyperparameters (from proteinworkshop/config/encoder/egnn.yaml):
      num_layers=6, emb_dim=512, activation=relu, norm=batch, aggr=sum,
      pool=mean, residual=True, dropout=0.1
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
        emb_dim = 512
        activation = 'relu'
        norm = 'batch'
        aggr = 'sum'
        residual = True

        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.residual = residual

        # Embedding lookup for initial node features (matching reference LazyLinear)
        self.emb_in = nn.Linear(input_dim, emb_dim)

        # Stack of EGNN layers (matching reference)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr, dropout))

        # Global pooling/readout: mean (matching reference config)
        self.pool = global_mean_pool

        # Output projection to match expected out_dim
        self.out_proj = nn.Linear(emb_dim, out_dim)

    def _build_edges(self, pos, batch):
        """Build kNN graph for message passing."""
        edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
        return edge_index

    def forward(self, pos, node_feat, batch):
        """Forward pass matching ProteinWorkshop EGNNModel.

        Args:
            pos: (N, 3) alpha-carbon coordinates
            node_feat: (N, input_dim) node scalar features
            batch: (N,) batch index

        Returns:
            node_emb: (N, out_dim) per-node embeddings
            graph_emb: (B, out_dim) per-graph embeddings
        """
        edge_index = self._build_edges(pos, batch)

        h = self.emb_in(node_feat)  # (n, input_dim) -> (n, emb_dim)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, edge_index)

            # Update node features with residual (matching reference)
            h = h + h_update if self.residual else h_update

            # Update node coordinates with residual (matching reference)
            pos = pos + pos_update if self.residual else pos_update

        # Project to output dimension
        node_emb = self.out_proj(h)
        graph_emb = self.pool(node_emb, batch)

        return node_emb, graph_emb

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
```
