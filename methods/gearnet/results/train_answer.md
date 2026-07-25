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


class GeometricRelationalGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_relation, edge_input_dim=None,
                 batch_norm=True, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.edge_linear = nn.Linear(edge_input_dim, input_dim) if edge_input_dim else None
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def message(self, graph, input, edge_input=None):
        node_in = graph.edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear is not None:
            message = message + self.edge_linear(graph.edge_feature.float())
        if edge_input is not None:
            assert edge_input.shape == message.shape
            message = message + edge_input
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        dst = graph.edge_list[:, 1]
        relation = graph.edge_list[:, 2]
        bucket = dst * self.num_relation + relation
        weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * weight, bucket, dim=0,
                             dim_size=graph.num_node * self.num_relation)
        return update.view(graph.num_node, self.num_relation * self.input_dim)

    def combine(self, update):
        update = self.linear(update)
        if self.batch_norm is not None:
            update = self.batch_norm(update)
        if self.activation is not None:
            update = self.activation(update)
        return update

    def forward(self, graph, input, edge_input=None):
        message = self.message(graph, input, edge_input)
        update = self.aggregate(graph, message)
        return self.combine(update)


class SpatialLineGraph(nn.Module):
    def __init__(self, num_angle_bin=8):
        super().__init__()
        self.num_angle_bin = num_angle_bin

    def forward(self, graph):
        line_graph = graph.line_graph()
        node_in, node_out = graph.edge_list[:, :2].t()
        prev_edge, next_edge = line_graph.edge_list.t()

        # line_graph enumerates consecutive directed edges src -> mid -> dst.
        src = node_in[prev_edge]
        mid = node_out[prev_edge]
        dst = node_out[next_edge]

        v1 = graph.node_position[src] - graph.node_position[mid]
        v2 = graph.node_position[dst] - graph.node_position[mid]
        angle = torch.atan2(torch.cross(v1, v2).norm(dim=-1), (v1 * v2).sum(dim=-1))
        relation = (angle / math.pi * self.num_angle_bin).long()
        relation = relation.clamp(max=self.num_angle_bin - 1)
        edge_list = torch.cat([line_graph.edge_list, relation.unsqueeze(-1)], dim=-1)
        return type(line_graph)(edge_list, num_nodes=line_graph.num_nodes,
                                offsets=line_graph._offsets, num_edges=line_graph.num_edges,
                                num_relation=self.num_angle_bin, meta_dict=line_graph.meta_dict,
                                **line_graph.data_dict)


class SumReadout(nn.Module):
    def forward(self, graph, node_feature):
        return scatter_add(node_feature, graph.node2graph, dim=0, dim_size=graph.batch_size)


class GearNet(nn.Module):
    def __init__(self, input_dim=21, hidden_dims=(512, 512, 512, 512, 512, 512),
                 num_relation=7, edge_input_dim=59, batch_norm=True, concat_hidden=True,
                 short_cut=True, readout="sum", dropout=0, num_angle_bin=8):
        super().__init__()
        self.num_relation = num_relation
        self.concat_hidden = concat_hidden
        self.short_cut = short_cut
        self.num_angle_bin = num_angle_bin
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]

        self.layers = nn.ModuleList([
            GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                         None, batch_norm, "relu")
            for i in range(len(self.dims) - 1)
        ])
        self.dropout = nn.Dropout(dropout)

        if num_angle_bin:
            self.spatial_line_graph = SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList([
                GeometricRelationalGraphConv(self.edge_dims[i], self.edge_dims[i + 1],
                                             num_angle_bin, None, batch_norm, "relu")
                for i in range(len(self.edge_dims) - 1)
            ])

        if readout != "sum":
            raise ValueError("This configuration uses sum readout")
        self.readout = SumReadout()

    def forward(self, graph, input):
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_hidden = line_graph.node_feature.float()
        else:
            edge_hidden = None

        for i, layer in enumerate(self.layers):
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_hidden)
            hidden = layer(graph, layer_input, edge_hidden)
            hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_feature = torch.cat(hiddens, dim=-1) if self.concat_hidden else hiddens[-1]
        graph_feature = self.readout(graph, node_feature)
        return {"graph_feature": graph_feature, "node_feature": node_feature}
```
