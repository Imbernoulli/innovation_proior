**Problem (from step 2).** EGNN's equivariant channel won EC and Fold but ran one shared transform over a
single kNN graph that blends sequential and spatial edges, with a single scalar weight per edge and no
edge type — and it slightly *regressed* GO-BP (0.238 < SchNet's 0.245) by piling equivariant machinery on
a coarse, sequence-correlated task. Wanted: type the edges by what they mean and let relational structure,
not just per-edge geometry, into the encoder.

**Key idea (geometry-aware relational convolution).** Build a multi-relational residue graph and give each
edge type its own kernel: `Σ_r W_r Σ_{j∈N_r(i)} (·)`. Seven types — five sequential offsets `{-2,-1,0,1,2}`
(direction and offset are meaningful; `0` is the self relation), one radius relation, one kNN relation
(radius restores density, kNN floors the degree; complementary failure modes). Read an invariant edge
feature (endpoint features + relation one-hot + sequence separation + distance), concat all six layers'
hidden states for the node embedding, and **sum**-pool for the graph embedding.

**Why it works.** The kernel count scales with the number of *types* (a handful), not edges, so it is
cheap and memory-flat. Typing separates backbone from tertiary contacts that a shared kNN transform
blended; multi-scale concat gives the head local and global structure; all geometry is distances, so the
encoder stays E(3)-invariant.

**This task's edit, not full GearNet.** **No line graph, no edge-to-edge messages, no angle bins** — the
harness omits the edge-enhanced (IEConv-style) branch entirely, so this is the *plain* relational node
convolution. Builds its own 7-relation graph from `pos`/`batch` (radius `radius_graph` at 10.0, kNN
`knn_graph` at `k=16`). Edge feature gates the message *multiplicatively via sigmoid*
(`h[src] * sigmoid(edge_linear(edge_feat))`) where canonical GearNet *adds* a projected edge feature.
BatchNorm is a separate post-conv module; residual short-cut, dropout, ReLU; `concat_hidden=True`;
`global_add_pool`.

**Hyperparameters.** `hidden_dim=512`, `num_layers=6`, `num_relation=7`, `cutoff=10.0`, `max_neighbors=16`,
`short_cut=True`, `concat_hidden=True`, `batch_norm=True`, `dropout=0.1`, pool=sum.

**What to watch.** Expect GO-BP to recover above 0.245 and clear EGNN's 0.238; EC the strongest of the
ladder (above 0.747). Fold is the cautious one: removing the explicit coordinate update without restoring
the omitted angle branch may leave Fold near or slightly *below* EGNN's 0.331 — the relational graph sees
which contacts exist and their type, but not the angle between two contacts of a residue.

```python
# =====================================================================
# EDITABLE SECTION START — GearNet encoder
# =====================================================================

class GeometricRelationalConv(nn.Module):
    """Geometric relational graph convolution layer from GearNet.

    Handles multiple edge types (relation types) via separate weight matrices
    and incorporates edge features.
    """
    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None,
                 batch_norm=True, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation

        # Per-relation linear transforms
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)

        if edge_input_dim is not None:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

        self.batch_norm_layer = nn.BatchNorm1d(output_dim) if batch_norm else None

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, h, edge_index, edge_type, edge_feat, num_nodes):
        """
        Args:
            h: (N, input_dim) node features
            edge_index: (2, E) edge indices
            edge_type: (E,) relation type per edge
            edge_feat: (E, edge_input_dim) or None
            num_nodes: total number of nodes
        Returns:
            out: (N, output_dim) updated node features
        """
        src, dst = edge_index

        # Edge-modulated messages
        msg = h[src]
        if self.edge_linear is not None and edge_feat is not None:
            msg = msg * torch.sigmoid(self.edge_linear(edge_feat))

        # Per-relation aggregation
        # Use edge_type to index into relation-specific buckets
        node_out = dst * self.num_relation + edge_type
        update = scatter_add(msg, node_out, dim=0,
                           dim_size=num_nodes * self.num_relation)
        update = update.view(num_nodes, self.num_relation * self.input_dim)
        update = self.linear(update)

        # Self-loop
        out = update + self.self_loop(h)
        out = self.activation(out)

        if self.batch_norm_layer is not None:
            out = self.batch_norm_layer(out)

        return out


class ProteinEncoder(nn.Module):
    """GearNet-based protein structure encoder.

    Geometry-Aware Relational Graph Neural Network that uses multiple
    edge types (sequential bonds, spatial proximity, k-nearest neighbors)
    with relational convolutions and optional short-cut connections.

    Reference hyperparameters (from proteinworkshop/config/encoder/gear_net.yaml
    and the GearNet paper, Zhang et al. 2022, arXiv:2203.06125):
      num_layers=6, emb_dim=512, activation=relu, short_cut=True,
      concat_hidden=True, batch_norm=True, pool=sum, num_relation=7
      (5 sequential offsets {-2,-1,0,1,2} + 1 spatial radius + 1 kNN).
    """
    def __init__(
        self,
        input_dim: int = SCALAR_NODE_DIM,
        hidden_dim: int = 512,
        out_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        cutoff: float = 10.0,
        max_neighbors: int = 16,
        num_relation: int = 7,
        short_cut: bool = True,
        concat_hidden: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # Build layer dimensions
        dims = [input_dim] + [hidden_dim] * num_layers
        edge_input_dim = input_dim * 2 + num_relation + 2  # node_i, node_j, rel_onehot, seq_dist, spatial_dist

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        for i in range(num_layers):
            self.layers.append(
                GeometricRelationalConv(
                    dims[i], dims[i + 1], num_relation,
                    edge_input_dim=edge_input_dim,
                    batch_norm=False,
                    activation='relu',
                )
            )
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        # Output projection
        if concat_hidden:
            total_dim = sum(dims[1:])
        else:
            total_dim = dims[-1]
        self.out_proj = nn.Linear(total_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_multi_relational_edges(self, pos, node_feat, batch):
        """Build edges with 7 relation types matching GearNet (Zhang et al. 2022):
        0..4: sequential edges with offsets {-2,-1,0,1,2}
              (offset 0 corresponds to a self-loop relation in sequential space)
        5:    spatial proximity (within cutoff radius)
        6:    k-nearest neighbors (k = max_neighbors)
        """
        device = pos.device
        N = pos.size(0)

        all_src, all_dst, all_type = [], [], []

        # Relations 0..4: sequential edges with offsets {-2, -1, 0, 1, 2}
        # Offsets are within the same protein (same batch index).
        # Bidirectionality is naturally produced by including both negative
        # and positive offsets as distinct relation types.
        seq_offsets = [-2, -1, 0, 1, 2]
        num_graphs = int(batch.max().item()) + 1
        for b in range(num_graphs):
            mask = (batch == b).nonzero(as_tuple=True)[0]
            n_b = len(mask)
            if n_b == 0:
                continue
            for r_idx, off in enumerate(seq_offsets):
                if off == 0:
                    # self-loop sequential relation
                    src = mask
                    dst = mask
                elif off > 0:
                    if n_b <= off:
                        continue
                    src = mask[:-off]
                    dst = mask[off:]
                else:  # off < 0
                    k = -off
                    if n_b <= k:
                        continue
                    src = mask[k:]
                    dst = mask[:-k]
                if len(src) == 0:
                    continue
                all_src.append(src)
                all_dst.append(dst)
                all_type.append(torch.full((len(src),), r_idx, dtype=torch.long, device=device))

        # Relation 5: spatial proximity within cutoff radius
        rad_edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=False,
                                      max_num_neighbors=512)
        rad_src, rad_dst = rad_edge_index
        all_src.append(rad_src)
        all_dst.append(rad_dst)
        all_type.append(torch.full((rad_src.numel(),), 5, dtype=torch.long, device=device))

        # Relation 6: k-nearest neighbors
        knn_edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
        knn_src, knn_dst = knn_edge_index
        all_src.append(knn_src)
        all_dst.append(knn_dst)
        all_type.append(torch.full((knn_src.numel(),), 6, dtype=torch.long, device=device))

        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
        edge_type = torch.cat(all_type)

        # Edge features: [node_feat_src, node_feat_dst, rel_onehot, seq_dist, spatial_dist]
        src, dst = edge_index
        ef_node_src = node_feat[src]
        ef_node_dst = node_feat[dst]
        ef_rel = F.one_hot(edge_type, self.num_relation).float()
        ef_seq_dist = torch.abs(src.float() - dst.float()).unsqueeze(-1)
        ef_spatial_dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)
        edge_feat = torch.cat([ef_node_src, ef_node_dst, ef_rel, ef_seq_dist, ef_spatial_dist], dim=-1)

        return edge_index, edge_type, edge_feat

    def forward(self, pos, node_feat, batch):
        N = pos.size(0)
        edge_index, edge_type, edge_feat = self._build_multi_relational_edges(pos, node_feat, batch)

        hiddens = []
        h = node_feat  # start from raw features (input_dim)

        for i, layer in enumerate(self.layers):
            hidden = layer(h, edge_index, edge_type, edge_feat, N)
            if self.short_cut and hidden.shape == h.shape:
                hidden = hidden + h
            if self.batch_norms is not None:
                hidden = self.batch_norms[i](hidden)
            hidden = self.dropout(hidden)
            hiddens.append(hidden)
            h = hidden

        if self.concat_hidden:
            node_feat_out = torch.cat(hiddens, dim=-1)
        else:
            node_feat_out = hiddens[-1]

        node_emb = self.out_proj(node_feat_out)
        # Sum pooling matches reference gear_net.yaml (pool=sum) and the
        # GearNet paper (Zhang et al. 2022, arXiv:2203.06125).
        graph_emb = global_add_pool(node_emb, batch)

        return node_emb, graph_emb

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
```
