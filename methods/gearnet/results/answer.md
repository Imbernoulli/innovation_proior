# GearNet, distilled

GearNet is a structure-based protein encoder. It represents a protein as a residue-level
relational graph, runs a relational graph convolution with one kernel per edge type, and in the
edge-enhanced version updates edge states on a sparse angle-typed line graph before folding those
edge states back into node messages. All geometric inputs are distances or angles, so the encoder
is invariant to translation, rotation, and reflection.

## Final Construction

The residue graph has one node per alpha-carbon. Node features are residue identities
(`input_dim = 21` in the downstream configuration). Directed edge types are:

- sequential offsets with `|j - i| < d_seq`, `d_seq = 3`, so
  `d = j - i in {-2, -1, 0, 1, 2}` gives five sequential relation types, including `d = 0`;
- one radius relation with cutoff `10.0`;
- one kNN relation with `k = 10`;
- spatial edges are kept for long-range contacts with `min_distance = 5`.

Thus the node graph has `|R| = (2 * d_seq - 1) + 2 = 2 * d_seq + 1 = 7` relation types.
The implementation edge feature is the concatenation of endpoint residue types, edge-type
one-hot, clamped sequence-distance one-hot, and Euclidean distance; the downstream edge-enhanced
configuration uses `edge_input_dim = 59`.

The node relational convolution is

```text
h_i^(0) = f_i
u_i^(l) = sigma(BN(sum_{r in R} W_r sum_{j in N_r(i)} h_j^(l-1)))
h_i^(l) = h_i^(l-1) + u_i^(l)     # residual short-cut when dimensions match
```

The implementation computes the aggregation by scattering every edge message into bucket
`dst * num_relation + r`, reshaping the result to `(num_node, num_relation * input_dim)`, and
applying one `Linear(num_relation * input_dim, output_dim)`. That single linear layer is the
stack of the relation-specific kernels `W_r`. BatchNorm and ReLU are applied after this linear
aggregation.

For the edge-enhanced model, the line graph has one node per directed edge of the residue graph.
It connects directed edge `(i -> j)` to directed edge `(j -> k)` when the edges share residue `j`
and `i != k`. The line-graph relation is the binned angle at `j`:

```text
v1 = x_i - x_j
v2 = x_k - x_j
angle = atan2(||v1 x v2||, v1 . v2)
bin = clamp(floor(angle / pi * 8), max=7)
```

A second relational convolution runs on this line graph with `num_angle_bin = 8`. In the
implementation, edge layer `l` outputs the input dimension of node layer `l`, so the learned
linear map on `m_(j,i,r)^l` is absorbed into the edge convolution output dimension and the
node-layer message is a direct addition:

```text
message_(j,i,r)^l = h_j^(l-1) + m_(j,i,r)^l
```

The model stacks six 512-wide layers. With `concat_hidden = True`, the final node representation
is `Cat(h^(1), ..., h^(L))`; the graph representation is a sum readout over those node
representations.

## Implementation Core

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

## Downstream Edge Configuration

```yaml
task:
  class: MultipleBinaryClassification
  model:
    class: GearNet
    input_dim: 21
    hidden_dims: [512, 512, 512, 512, 512, 512]
    batch_norm: True
    concat_hidden: True
    short_cut: True
    readout: 'sum'
    num_relation: 7
    edge_input_dim: 59
    num_angle_bin: 8
  graph_construction_model:
    class: GraphConstruction
    node_layers:
      - class: AlphaCarbonNode
    edge_layers:
      - class: SequentialEdge
        max_distance: 2
      - class: SpatialEdge
        radius: 10.0
        min_distance: 5
      - class: KNNEdge
        k: 10
        min_distance: 5
    edge_feature: gearnet
```
