I need an encoder that turns a single protein structure into fixed vectors: one per residue and one for the whole chain. The input is sparse — just an alpha-carbon coordinate and a residue identity per node — and the output must be useful for function and fold prediction. The hard constraints are rigid-motion invariance and data scarcity. Any quantity the network sees must be unchanged by translation, rotation, or reflection, and the model must be light enough to train on the relatively small set of experimentally solved structures.

The baselines each leave a gap. A plain graph convolution treats a backbone bond and a long-range tertiary contact as the same kind of edge, which is structurally wrong. Relational convolutions fix that in principle, but standard R-GCN expects relation types to be given; it does not tell us which geometric relations a protein should expose. SchNet is cleanly invariant because it uses only distances, but it is blind to direction: two different local geometries can present the same set of neighbor distances. EGNN adds direction by carrying and updating coordinates equivariantly, yet that is heavier than necessary when we only want an invariant embedding, and it still runs one shared transform over a single kNN graph that blends sequential and spatial information. IEConv and DimeNet are expressive but pay with per-edge kernels or expensive spherical bases that are awkward at the residue level. What is missing is a design that is invariant, directional, relationally structured, and cheap.

The method I propose is GearNet, a geometry-aware relational graph neural network. It builds a multi-relational residue graph and runs a relational convolution with one kernel per edge type. To recover directional information without equivariant coordinates, it also updates edge states on a sparse angle-typed line graph and folds those edge states back into node messages. Every quantity the network consumes is a distance or an angle, so the entire encoder is E(3)-invariant by construction, and the number of learnable kernels scales with the number of edge types rather than the number of edges.

The residue graph has one node per alpha-carbon. Sequential edges are typed by their sequence offset, because backbone direction and exact separation matter: with offsets in {-2, -1, 0, 1, 2} there are five sequential relation types, where offset 0 is a self relation. Spatial edges come in two complementary forms: a radius graph captures genuine density in packed regions, while a k-nearest-neighbor graph guarantees a minimum degree so loosely packed structures do not collapse to an edgeless graph. Using only one of these fails in opposite ways, so GearNet uses both. To keep spatial edges focused on tertiary contacts, edges with small sequence separation are dropped. That gives seven relation types in total, each with its own kernel.

The node update is a relational convolution. For each node and relation, the messages from neighbors of that relation are summed; the results for all relations are concatenated and passed through one linear layer that is equivalent to a stack of per-relation kernels. Batch normalization and ReLU are applied to the aggregated message, and the result is added residually to the previous node state. Stacking six layers of width 512 lets local backbone features propagate into global fold features.

The missing piece is the angle between two contacts at a residue. A node-only layer cannot distinguish whether two incident edges point in nearly the same direction or in opposite directions, even though that is exactly the geometry that separates one local arrangement from another. GearNet fixes this with a sparse line graph. Each directed edge of the residue graph becomes a node in the line graph, and two such edge-nodes are connected when they share a middle residue. The line-graph relation is the unsigned angle at that middle residue, binned into eight discrete types. The same cheap relational convolution then updates edge states on this line graph. Before each node layer, the corresponding edge layer produces a message for every residue-graph edge; that message is added to the neighbor feature carried by the same edge, so the node update sees not only who the neighbor is but how that contact is oriented relative to the other contacts at the residue.

For the readout, all six hidden layers are concatenated for each residue so the downstream head sees both local and global scales. The graph-level embedding is a sum pool over those concatenated node vectors; summing preserves size information that averaging would discard.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_cluster import radius_graph, knn_graph


class RelationalGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_relation, batch_norm=True):
        super().__init__()
        self.num_relation = num_relation
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim) if batch_norm else None

    def forward(self, x, edge_index, edge_type, num_nodes, edge_input=None):
        src, dst = edge_index
        msg = x[src]
        if edge_input is not None:
            msg = msg + edge_input
        bucket = dst * self.num_relation + edge_type
        update = scatter_add(msg, bucket, dim=0,
                             dim_size=num_nodes * self.num_relation)
        update = update.view(num_nodes, self.num_relation * x.size(-1))
        update = self.linear(update)
        if self.bn is not None:
            update = self.bn(update)
        return F.relu(update)


class GearNet(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=512, num_layers=6,
                 num_relation=7, num_angle_bin=8, dropout=0.0):
        super().__init__()
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.dims = [input_dim] + [hidden_dim] * num_layers
        self.edge_dims = [input_dim] + self.dims[:-1]

        self.layers = nn.ModuleList([
            RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation)
            for i in range(num_layers)
        ])
        self.edge_layers = nn.ModuleList([
            RelationalGraphConv(self.edge_dims[i], self.edge_dims[i + 1],
                                num_angle_bin)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def build_edges(self, pos, batch, seq_offsets=(-2, -1, 0, 1, 2),
                    radius=10.0, k=10, min_seq_dist=5):
        device = pos.device
        srcs, dsts, types = [], [], []
        num_graphs = int(batch.max().item()) + 1

        for g in range(num_graphs):
            mask = (batch == g).nonzero(as_tuple=True)[0]
            n = mask.numel()
            for r, off in enumerate(seq_offsets):
                if off == 0:
                    s, d = mask, mask
                elif off > 0:
                    if n <= off:
                        continue
                    s, d = mask[:-off], mask[off:]
                else:
                    if n <= -off:
                        continue
                    s, d = mask[-off:], mask[:off]
                srcs.append(s)
                dsts.append(d)
                types.append(torch.full_like(s, r))

        rad = radius_graph(pos, r=radius, batch=batch, loop=False,
                           max_num_neighbors=512)
        s, d = rad
        keep = torch.abs(s.float() - d.float()) >= min_seq_dist
        srcs.append(s[keep])
        dsts.append(d[keep])
        types.append(torch.full_like(s[keep], len(seq_offsets)))

        knn = knn_graph(pos, k=k, batch=batch, loop=False)
        s, d = knn
        keep = torch.abs(s.float() - d.float()) >= min_seq_dist
        srcs.append(s[keep])
        dsts.append(d[keep])
        types.append(torch.full_like(s[keep], len(seq_offsets) + 1))

        edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)
        edge_type = torch.cat(types)
        return edge_index, edge_type

    def build_line_graph(self, edge_index, pos):
        src, dst = edge_index
        E = edge_index.size(1)
        device = pos.device
        num_nodes = int(dst.max().item()) + 1

        order = torch.argsort(dst)
        in_deg = torch.bincount(dst, minlength=num_nodes)
        in_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                            in_deg.cumsum(0)])

        deg_e2 = in_deg[src]
        e2_idx = torch.arange(E, device=device).repeat_interleave(deg_e2)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                             deg_e2.cumsum(0)[:-1]])
        local = (torch.arange(deg_e2.sum(), device=device) -
                 offsets.repeat_interleave(deg_e2))
        e1_idx = order[in_ptr[src[e2_idx]] + local]

        mask = src[e1_idx] != dst[e2_idx]
        e1_idx, e2_idx = e1_idx[mask], e2_idx[mask]

        i, j, k = src[e1_idx], dst[e1_idx], dst[e2_idx]
        v1 = pos[i] - pos[j]
        v2 = pos[k] - pos[j]
        angle = torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1),
                            (v1 * v2).sum(dim=-1))
        rel = (angle / math.pi * self.num_angle_bin).long().clamp(
            max=self.num_angle_bin - 1)

        return torch.stack([e1_idx, e2_idx], dim=0), rel

    def forward(self, graph, node_feature):
        pos, batch = graph.pos, graph.batch
        num_nodes = node_feature.size(0)
        edge_index, edge_type = self.build_edges(pos, batch)
        line_edge_index, line_edge_type = self.build_line_graph(edge_index, pos)

        hiddens = []
        h = node_feature
        edge_h = node_feature[edge_index[0]]

        for layer, edge_layer in zip(self.layers, self.edge_layers):
            edge_h = edge_layer(edge_h, line_edge_index, line_edge_type,
                                edge_h.size(0))
            hidden = layer(h, edge_index, edge_type, num_nodes,
                           edge_input=edge_h)
            hidden = self.dropout(hidden)
            if hidden.shape == h.shape:
                hidden = hidden + h
            hiddens.append(hidden)
            h = hidden

        node_feature_out = torch.cat(hiddens, dim=-1)
        graph_feature = scatter_add(node_feature_out, batch, dim=0)
        return {"node_feature": node_feature_out,
                "graph_feature": graph_feature}
```
