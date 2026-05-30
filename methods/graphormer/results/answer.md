# Graphormer

## Problem

A standard Transformer encoder, applied to a graph, treats the nodes as an unordered set of feature vectors: the attention score $A_{ij}=(h_iW_Q)(h_jW_K)^\top/\sqrt{d}$ depends only on node features and is permutation-equivariant, so it sees no node degrees, no inter-node distances, and no edges. Graphormer keeps the standard Transformer block intact and makes it competitive on graph-level prediction by injecting graph structure through three simple encodings, plus a virtual node for readout.

## Key idea

Match each graph-native structural signal to the part of attention with the right arity:

1. **Centrality encoding (per-node → input).** Add learnable embeddings indexed by node in/out-degree to the input node features, so $Q$ and $K$ carry node-importance information:
$$h_i^{(0)} = x_i + z^-_{\deg^-(v_i)} + z^+_{\deg^+(v_i)},\qquad z^-,z^+\in\mathbb{R}^d.$$
(Undirected graphs use a single $\deg(v_i)$ term.)

2. **Spatial encoding (per-pair → attention score).** Let $\phi(v_i,v_j)$ be the shortest-path distance (a special value for disconnected pairs). Add a learnable scalar bias, indexed by $\phi$ and shared across all layers:
$$A_{ij}=\frac{(h_iW_Q)(h_jW_K)^\top}{\sqrt{d}} + b_{\phi(v_i,v_j)}.$$
This preserves the global receptive field while letting the model learn locality (a decreasing $b_\phi$) or any other distance-attention profile.

3. **Edge encoding (per-pair → attention score).** For the shortest path $\mathrm{SP}_{ij}=(e_1,\dots,e_N)$ from $i$ to $j$, average the dot-products of each edge feature with a per-path-position learnable weight, and add it as a bias:
$$A_{ij}=\frac{(h_iW_Q)(h_jW_K)^\top}{\sqrt{d}} + b_{\phi(v_i,v_j)} + c_{ij},\qquad c_{ij}=\frac{1}{N}\sum_{n=1}^{N} x_{e_n}(w^E_n)^\top.$$
Averaging (not summing) over the path keeps $c_{ij}$ about *what kind* of edges connect the pair, leaving *how far* to $b_\phi$.

**[VNode] readout.** A special virtual node is connected to all nodes and processed like any node; its final-layer representation is the graph vector $h_G$ (a learned readout, like [CLS]). Because the VNode's connections are virtual, its spatial biases $b_{\phi(\text{VNode},\cdot)}$ and $b_{\phi(\cdot,\text{VNode})}$ are reset to a distinct learnable scalar so artificial links are not confused with real edges.

**Block.** A standard pre-LN Transformer layer:
$$h'^{(l)}=\mathrm{MHA}(\mathrm{LN}(h^{(l-1)}))+h^{(l-1)},\qquad h^{(l)}=\mathrm{FFN}(\mathrm{LN}(h'^{(l)}))+h'^{(l)},$$
with the FFN inner width set to $d$ (rather than $4d$) to save parameters on small graphs.

## Expressiveness

With the spatial and edge encodings, a single Graphormer layer can represent the AGGREGATE–COMBINE step of popular GNNs, so **GCN, GraphSAGE, and GIN are special cases**:

- **MEAN aggregate** (GCN): set $b_\phi=0$ for $\phi=1$, $-\infty$ otherwise; $W_Q=W_K=0$; $W_V=I$. Softmax restricts to one-hop neighbors with uniform weights → average over $\mathcal N(v_i)$. For a self-inclusive mean, keep $\phi=0$ unmasked too, or handle the self state with the COMBINE head.
- **SUM aggregate** (GIN): $\text{SUM}=\deg(v_i)\cdot\text{MEAN}$. Read the degree from the centrality encoding (extra head) and let a sufficiently wide FFN multiply the mean by the degree.
- **MAX aggregate** (GraphSAGE): per dimension $t$, set $b_\phi=0$ for $\phi=1$ else $-\infty$; $W_K=e_t$, $W_Q=0$ with $Q$-bias $T\mathbf{1}$ (large temperature $T$), $W_V=e_t$ → softmax approximates hard max over neighbors.
- **COMBINE**: an extra head with $b_\phi=0$ for $\phi=0$ else $-\infty$, $W_Q=W_K=0$, $W_V=I$ returns the node's own feature; a sufficiently wide FFN fuses it with the aggregate.

A vanilla layer (no extra encodings) can also represent a **MEAN READOUT** (on a bias-free head, $W_Q=W_K=0$ with equal query/key biases → every score in a row equal → uniform attention over all nodes; $W_V=I$), which explains why self-attention subsumes the virtual-node/readout heuristic. (A large constant score cannot itself "wash out" a non-constant spatial bias, since softmax is invariant to row-wise constants — hence the bias-free head.) There exist graph pairs whose 1-WL colors match but whose node-wise shortest-path-distance profiles differ; the spatial encoding exposes exactly that missing signal, so Graphormer goes beyond the 1-WL ceiling of message-passing GNNs.

## Code

Per-graph preprocessing computes all-pairs shortest paths and the edges along them; the model adds centrality to the node inputs and the spatial/edge biases to the attention scores.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphormer.data import algos


# ---- Per-graph preprocessing: structural quantities (Floyd-Warshall) ----
def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(
        0, feature_num * offset, offset, dtype=torch.long, device=x.device
    )
    return x + feature_offset


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    if edge_attr.dim() == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1
    # all-pairs shortest path distance (phi) and edges along each path
    spd, path = algos.floyd_warshall(adj.numpy())       # SPD matrix, predecessor table
    edge_input = algos.gen_edge_input(spd.max(), path, attn_edge_type.numpy())
    item.x = x
    item.spatial_pos = torch.from_numpy(spd).long()     # phi(i, j)
    item.attn_edge_type = attn_edge_type
    item.edge_input = torch.from_numpy(edge_input).long()
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree                    # undirected
    item.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # + VNode row/col
    return item


def pad_1d_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos,
         item.in_degree, item.out_degree, item.x,
         item.edge_input[:, :, :multi_hop_max_dist, :], item.y)
        for item in items
    ]
    (idxs, attn_biases, attn_edge_types, spatial_poses,
     in_degrees, out_degrees, xs, edge_inputs, ys) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    return {
        "idx": torch.LongTensor(idxs),
        "attn_bias": torch.cat([
            pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases
        ]),
        "attn_edge_type": torch.cat([
            pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types
        ]),
        "spatial_pos": torch.cat([
            pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses
        ]),
        "in_degree": torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]),
        "out_degree": torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]),
        "x": torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]),
        "edge_input": torch.cat([
            pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist)
            for i in edge_inputs
        ]),
        "y": torch.cat(ys),
    }


# ---- Centrality encoding + node features + [VNode] ----
class GraphNodeFeature(nn.Module):
    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim):
        super().__init__()
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)          # [VNode]

    def forward(self, batched_data):
        x = batched_data["x"]
        in_degree, out_degree = batched_data["in_degree"], batched_data["out_degree"]
        n_graph = x.size(0)
        node_feature = self.atom_encoder(x).sum(dim=-2)         # raw node features
        node_feature = (node_feature                           # h_i^(0) = x_i
                        + self.in_degree_encoder(in_degree)    #   + z^-_{deg^-}
                        + self.out_degree_encoder(out_degree)) #   + z^+_{deg^+}
        graph_token = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        return torch.cat([graph_token, node_feature], dim=1)   # prepend [VNode]


# ---- Spatial + edge encoding -> additive attention bias ----
class GraphAttnBias(nn.Module):
    def __init__(self, num_heads, num_edges, num_spatial, num_edge_dis,
                 multi_hop_max_dist=20):
        super().__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = "multi_hop"
        self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(self, batched_data):
        attn_bias, spatial_pos = batched_data["attn_bias"], batched_data["spatial_pos"]
        edge_input = batched_data["edge_input"]
        attn_edge_type = batched_data["attn_edge_type"]
        x = batched_data["x"]
        n_graph, n_node = x.size()[:2]
        gb = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # spatial encoding: A_ij += b_{phi(i,j)}
        spatial_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        gb[:, :, 1:, 1:] = gb[:, :, 1:, 1:] + spatial_bias

        # reset VNode connections to a distinct learnable scalar
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        gb[:, :, 1:, 0] = gb[:, :, 1:, 0] + t
        gb[:, :, 0, :] = gb[:, :, 0, :] + t

        # edge encoding: average of (edge feature . learnable weight) along SP_ij
        if self.edge_type == "multi_hop":
            sp = spatial_pos.clone()
            sp[sp == 0] = 1
            sp = torch.where(sp > 1, sp - 1, sp).clamp(0, self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2)    # embed each edge
            max_dist = edge_input.size(-2)
            eib = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            eib = torch.bmm(
                eib,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist],
            )
            edge_input = eib.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / sp.float().unsqueeze(-1)
            ).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        gb[:, :, 1:, 1:] = gb[:, :, 1:, 1:] + edge_input
        return gb + attn_bias.unsqueeze(1)


class MultiheadAttention(nn.Module):
    def __init__(self, d, num_heads, dropout=0.1):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        length, batch, d = x.size()
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        def shape(t):
            return (
                t.contiguous()
                .view(length, batch * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        q, k, v = shape(q), shape(k), shape(v)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.reshape(
                batch * self.num_heads, length, length
            )
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch, self.num_heads, length, length)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )
            attn_weights = attn_weights.view(batch * self.num_heads, length, length)
        attn_probs = F.dropout(
            torch.softmax(attn_weights, dim=-1),
            p=self.dropout,
            training=self.training,
        )
        attn = torch.bmm(attn_probs, v)
        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(length, batch, d)
        )
        return self.out_proj(attn), attn_probs


# ---- Pre-LN Transformer block; structure enters as one bias on the scores ----
class GraphormerLayer(nn.Module):
    def __init__(self, d, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d, num_heads, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, ffn_dim), nn.Linear(ffn_dim, d)
        self.ffn_ln = nn.LayerNorm(d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias, padding_mask=None):
        h = self.attn_ln(x)
        h, _ = self.self_attn(h, attn_bias=attn_bias, key_padding_mask=padding_mask)
        x = x + self.dropout(h)
        h = self.fc2(self.dropout(self.act(self.fc1(self.ffn_ln(x)))))
        return x + self.dropout(h)


class Graphormer(nn.Module):
    def __init__(self, num_atoms, num_in_degree, num_out_degree, num_edges,
                 num_spatial, num_edge_dis, d=768, ffn_dim=768, num_heads=32,
                 n_layers=12, out_dim=1):
        super().__init__()
        self.node_feature = GraphNodeFeature(num_atoms, num_in_degree,
                                             num_out_degree, d)
        self.attn_bias = GraphAttnBias(num_heads, num_edges, num_spatial, num_edge_dis)
        self.layers = nn.ModuleList(
            [GraphormerLayer(d, ffn_dim, num_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d, out_dim)

    def forward(self, batched_data):
        padding_mask = batched_data["x"][:, :, 0].eq(0)
        padding_mask = torch.cat(
            [padding_mask.new_zeros(padding_mask.size(0), 1), padding_mask],
            dim=1,
        )
        x = self.node_feature(batched_data)        # features + centrality + [VNode]
        bias = self.attn_bias(batched_data)        # b_phi + c_ij + VNode reset
        x = x.transpose(0, 1)                       # [N+1, B, d]
        for layer in self.layers:
            x = layer(x, bias, padding_mask=padding_mask)
        return self.out(x[0])                       # [VNode] is the graph readout
```
