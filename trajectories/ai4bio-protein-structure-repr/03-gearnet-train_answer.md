The equivariant rung did almost exactly what I predicted, and the one place it did not is the seam to pull on. Fold jumped from SchNet's 0.184 to 0.331 — the directional-discrimination gain I argued for, equivariance helping most precisely where two folds share distance statistics and differ in direction. EC rose from 0.589 to 0.747, a large gain too, because active-site geometry benefits from the directional structure the coordinate channel injects into the distances. But GO-BP went the *wrong way*: 0.238, down from SchNet's 0.245 — I flagged that risk, and the regression confirms it: piling batch-norm-heavy equivariant machinery onto a coarse, sequence-correlated task hurt it slightly. So EGNN is the stronger baseline overall but not uniformly better, and the regression plus the *shape* of the EGNN layer says what it is still missing. The equivariant coordinate update is a single scalar weight per edge: $\varphi_x(m_{ij})$ reads one edge's geometry and moves the coordinate along that one difference vector. The neighbors never talk to each other, and the encoder never asks what *kind* of edge each one is. Edges that mean different things are all transformed by the same machinery, and the relational structure between a residue's contacts is invisible.

I propose **GearNet**, a geometry-aware relational convolution, and I take the edge-type problem first because it is the more basic structural error and the one most likely to recover GO-BP while holding the gains. A residue's edges mean different things in a folded protein: some are "these two are adjacent along the chain" — sequential, backbone, local — and some are "these two are far apart in the chain but close in space" — a tertiary contact, the whole point of folding. Those two kinds carry completely different geometric information, yet SchNet and EGNN both build a single kNN graph and run one shared transform over every edge; a kNN graph in particular *mixes* the two, since a residue's $k$ nearest neighbors are an indistinguishable blend of sequential neighbors and spatial contacts. That is the same error as one global learning rate for parameters at different scales. So the move is to type the edges and let the transform depend on the type — a relational graph convolution with one learnable kernel $W_r$ per edge type $r$, shared across all edges of that type, combining the per-type neighbor sums as
$$h_i' = \sum_r W_r \sum_{j\in\mathcal N_r(i)} (\cdot).$$
The key accounting fact is that the number of kernels is the number of edge *types*, not edges — a handful, independent of how many edges any protein has. That matters because the database is structurally diverse and proteins vary wildly in size: a per-edge tailored kernel would blow up in memory, a single shared kernel cannot tell types apart, and the relational convolution sits exactly between — capacity scaling in the number of types, memory not scaling with the number of edges at all.

I design the types from the geometry deliberately. For the sequential structure I do not lump all backbone neighbors into one type, because direction and exact offset along the chain are meaningful — the relationship $i\to i{+}1$ is not the relationship $i\to i{-}2$ — so I type each sequential edge by its relative position with offsets $\{-2,-1,0,1,2\}$, five sequential relation types, including the $0$ self-relation that gives a clean place for self-information inside the relational machinery and positive/negative offsets kept distinct so direction is preserved. For the spatial structure I use *two* complementary types, and the reason is the failure mode of each rule alone. A radius rule — connect $i,j$ when $\lVert\mathrm{pos}_i-\mathrm{pos}_j\rVert<\text{cutoff}$ — gives density information in crowded regions but leaves loosely-packed proteins near-edgeless, since no single fixed radius fits every structure. A kNN rule gives every residue a guaranteed degree but flattens away the density variation in packed regions, capping everyone at $k$. The two are complementary — radius restores density, kNN puts a floor under the degree so no protein collapses — so I add both as separate edge types, one radius relation and one kNN relation. Seven relation types in total, seven kernels, that is all.

Made concrete on this task's edit surface, the honest accounting matters more here than anywhere on the ladder, because this is a *stripped-down* version of the full relational-with-edge-messages design and I must build the version the harness actually exposes. The biggest omission: there is **no line graph, no edge-to-edge message passing, and no angle binning**. The richest version of this idea would build a graph whose nodes are the *edges* of the residue graph, connect two edges that share a residue, type that connection by the binned angle between them, and run a second relational convolution on it — that is what would let a residue's contacts *relate* to each other and deliver the between-edge directional information the node-only layer is blind to. The harness does not expose that branch, so this rung gets the relational *node* convolution and the seven-type graph but **not** the edge-enhanced angle machinery, and that bounds what I can expect.

The construction. The encoder builds the seven-relation graph itself from $\mathrm{pos}$, $\mathrm{node\_feat}$, and $\mathrm{batch}$: five sequential offset relations enumerated per protein within each batch element (the $0$ offset as a self-loop sequential relation, positive/negative offsets distinct), one radius relation (`radius_graph` at $\text{cutoff}=10.0$ with a generous max-neighbor budget so dense regions are not truncated), and one kNN relation (`knn_graph` at $k=\text{max\_neighbors}=16$). The per-relation aggregation is done in a single pass with the scatter trick: scatter every edge message into a bucket keyed by $\text{dst}\cdot\text{num\_relation}+r$ into an array of size $\text{num\_nodes}\cdot\text{num\_relation}$, reshape to $(\text{num\_nodes},\,\text{num\_relation}\cdot\text{input\_dim})$, and apply one `Linear(num_relation * input_dim, output_dim)` — that single weight matrix *is* the seven $W_r$ stacked side by side, so all relations are one matmul. The layer also reads an edge feature: for each edge it concatenates the two endpoint node features, a one-hot of the relation type, the absolute sequence separation $|i-j|$, and the spatial distance $\lVert\mathrm{pos}_i-\mathrm{pos}_j\rVert$ — every geometric quantity in there a distance, so the encoder stays E(3)-invariant. The harness modulates each message by this edge feature *multiplicatively through a sigmoid gate*, $\text{message}=h_{\text{src}}\cdot\sigma\big(\text{edge\_linear}(\text{edge\_feat})\big)$, a detail to get right because the canonical relational layer *adds* a projected edge feature rather than gating — this task gates. The layer applies the relation-combining linear, then (as a separate module after the conv, not inside it) a batch norm, a residual short-cut when dimensions match, dropout, and ReLU. Six layers, hidden width 512.

Two readout choices follow the relational design and both differ from EGNN. First, `concat_hidden`: the per-node representation is the concatenation of *all six* layers' hidden states, not just the last, because the early layers hold local backbone geometry and the later layers hold the propagated global fold, and the downstream head benefits from seeing every scale — a multi-scale readout EGNN did not have. Second, the graph embedding is a **sum** pool (`global_add_pool`) over those concatenated node embeddings, not EGNN's mean: summing keeps a notion of the total signal and the protein's size, which the classification heads can use and which mean would normalize away.

I expect the delta from EGNN to show most clearly on **GO-BP**: typing the edges and the multi-scale concat readout should recover and exceed the regression — GO-BP should climb back above SchNet's 0.245 and clear EGNN's 0.238, because the relational structure lets the cheap, sequence-correlated signal through cleanly rather than drowning it in equivariant coordinate machinery; if it does not recover, edge-typing is not the lever I think it is. **EC** should rise above EGNN's 0.747 — distinguishing sequential from spatial contacts is exactly what active-site reasoning wants, and the multi-scale readout helps — so I expect the strongest EC of the ladder here. **Fold** is the prediction I am most cautious about: EGNN's equivariant coordinate channel gave it a genuine directional edge (0.331), and this rung *removed* the explicit coordinate update and did *not* replace it with the line-graph angle structure that would supply between-edge direction, so Fold could plausibly land slightly *below* EGNN even as EC and GO-BP rise. If Fold lands near or just under EGNN while EC and GO-BP clearly lead, that is the signature of this exact trade — the node-relational encoder wins on contact-type and sequence-correlated structure and pays a little on the pure directional fold cue it cannot see without the omitted edge branch — and that omitted branch is the obvious thing a further rung would restore.

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

    Reference hyperparameters (from proteinworkshop/config/encoder/gear_net.yaml):
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
        """Build edges with 7 relation types:
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
        # Sum pooling matches reference gear_net.yaml (pool=sum).
        graph_emb = global_add_pool(node_emb, batch)

        return node_emb, graph_emb

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
```
