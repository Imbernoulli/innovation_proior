The problem is to make a standard Transformer encoder competitive for graph-level prediction. A Transformer works on sequences by attending over tokens, but a graph has no canonical node order and no one-dimensional offsets. If you simply feed a graph's node features into self-attention, the model sees only an unordered bag of vectors. It does not know which nodes are connected, how far apart two nodes are, or how central a node is. That is why plain Transformers lose to message-passing GNNs on graph benchmarks, and why most prior graph-Transformers either degenerate into GNNs with attention-shaped aggregators or rely on spectral positional encodings that are not native to the graph. The real question is what minimal structural information to give a standard Transformer encoder so that it can match and exceed GNNs.

GNNs have another limitation: they aggregate only over one-hop neighbors per layer, so their receptive field grows only with depth, they tend to over-smooth, and their discriminative power is bounded by the 1-WL test. A Transformer already gives every node a global receptive field in one layer. What it lacks is the structure-aware bias to use that receptive field well. The right fix is to feed the graph's pre-model structural facts into the attention computation exactly where their arities match: per-node signals go into the node embeddings, and per-pair signals go into the attention scores.

The method is Graphormer. It keeps the Transformer block essentially intact and adds three graph-native encodings plus a virtual readout node.

Centrality encoding adds a learnable embedding indexed by each node's in-degree and out-degree to the initial node feature. The intuition is that degree is a direct measure of node importance, and since attention scores are built from queries and keys, which are linear projections of the input, the only way for the score to condition on importance is to put it in the input embedding. The update is h_i^(0) = x_i + z^-_{deg^-(v_i)} + z^+_{deg^+(v_i)}, with a single degree term for undirected graphs.

Spatial encoding injects the shortest-path distance phi(v_i, v_j) into the attention score as a learned scalar bias. This is the natural analogue of relative positional bias in sequence Transformers, but instead of indexing by index offset, we index by graph distance. The score becomes A_ij = (h_i W_Q)(h_j W_K)^T / sqrt(d) + b_{phi(v_i, v_j)}. The bias is shared across layers and can learn any relationship between distance and attention: it can enforce locality by down-weighting far nodes, or it can attend globally when that helps. Because it is added to the score, the global receptive field is preserved while structure modulates it.

Edge encoding adds the edge features along the shortest path from i to j. The relevant edge information for a pair is the chain of edges connecting them, not just the edges incident to each endpoint. For shortest path SP_ij = (e_1, ..., e_N), we embed each edge feature, take a learnable dot-product with a per-path-position weight, and average over the path: c_ij = (1/N) sum_{n=1}^N x_{e_n} (w^E_n)^T. Averaging normalizes by path length so that this term captures what kind of edges connect the pair, while spatial encoding captures how far apart they are. This is also added to A_ij.

For readout, Graphormer prepends a special virtual node [VNode] connected to every real node, similar to [CLS]. It is processed through all layers and its final representation becomes the whole-graph vector. Because self-attention is already global, the network can learn an adaptive pooling rather than a fixed mean or sum. The spatial biases on VNode connections are reset to a distinct learnable scalar so the model does not confuse virtual links with real bonds.

The Transformer block itself is standard pre-LN: layer norm before multi-head attention and before the feed-forward network, with residual connections around each. The FFN inner dimension is kept at d rather than 4d, which saves parameters on the small graphs common in molecular benchmarks and keeps the capacity focused on the structure-aware attention.

Graphormer is provably at least as expressive as popular GNNs. By setting the spatial bias to allow only one-hop neighbors and zeroing the query-key contribution, one attention head can compute uniform mean aggregation, matching GCN. With the degree available from centrality encoding, sum aggregation and hence GIN is recoverable. By using a high-temperature softmax per dimension, max aggregation and hence GraphSAGE become special cases. A separate self-attention head can return the node's own feature for COMBINE. At the same time, because spatial encoding gives every node its full shortest-path-distance profile, Graphormer can distinguish graphs that 1-WL cannot, so it exceeds the GNN ceiling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphormer.data import algos


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
    spd, path = algos.floyd_warshall(adj.numpy())
    edge_input = algos.gen_edge_input(spd.max(), path, attn_edge_type.numpy())
    item.x = x
    item.spatial_pos = torch.from_numpy(spd).long()
    item.attn_edge_type = attn_edge_type
    item.edge_input = torch.from_numpy(edge_input).long()
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree
    item.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
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
        "attn_bias": torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]),
        "attn_edge_type": torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, i.size(-2)) for i in attn_edge_types]),
        "spatial_pos": torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]),
        "in_degree": torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]),
        "out_degree": torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]),
        "x": torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]),
        "edge_input": torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]),
        "y": torch.cat(ys),
    }


class GraphNodeFeature(nn.Module):
    def __init__(self, num_atoms, num_in_degree, num_out_degree, hidden_dim):
        super().__init__()
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(self, batched_data):
        x = batched_data["x"]
        in_degree, out_degree = batched_data["in_degree"], batched_data["out_degree"]
        n_graph = x.size(0)
        node_feature = self.atom_encoder(x).sum(dim=-2)
        node_feature = (node_feature
                        + self.in_degree_encoder(in_degree)
                        + self.out_degree_encoder(out_degree))
        graph_token = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        return torch.cat([graph_token, node_feature], dim=1)


class GraphAttnBias(nn.Module):
    def __init__(self, num_heads, num_edges, num_spatial, num_edge_dis, multi_hop_max_dist=20):
        super().__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(self, batched_data):
        attn_bias = batched_data["attn_bias"]
        spatial_pos = batched_data["spatial_pos"]
        edge_input = batched_data["edge_input"]
        x = batched_data["x"]
        n_graph, n_node = x.size()[:2]
        gb = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        spatial_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        gb[:, :, 1:, 1:] = gb[:, :, 1:, 1:] + spatial_bias

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        gb[:, :, 1:, 0] = gb[:, :, 1:, 0] + t
        gb[:, :, 0, :] = gb[:, :, 0, :] + t

        sp = spatial_pos.clone()
        sp[sp == 0] = 1
        sp = torch.where(sp > 1, sp - 1, sp).clamp(0, self.multi_hop_max_dist)
        edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
        edge_input = self.edge_encoder(edge_input).mean(-2)
        max_dist = edge_input.size(-2)
        eib = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        eib = torch.bmm(
            eib,
            self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist],
        )
        edge_input = eib.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
        edge_input = (edge_input.sum(-2) / sp.float().unsqueeze(-1)).permute(0, 3, 1, 2)
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
            return t.contiguous().view(length, batch * self.num_heads, self.head_dim).transpose(0, 1)

        q, k, v = shape(q), shape(k), shape(v)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.reshape(batch * self.num_heads, length, length)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch, self.num_heads, length, length)
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            attn_weights = attn_weights.view(batch * self.num_heads, length, length)
        attn_probs = F.dropout(torch.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(length, batch, d)
        return self.out_proj(attn), attn_probs


class GraphormerLayer(nn.Module):
    def __init__(self, d, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d, num_heads, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d)
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
        self.node_feature = GraphNodeFeature(num_atoms, num_in_degree, num_out_degree, d)
        self.attn_bias = GraphAttnBias(num_heads, num_edges, num_spatial, num_edge_dis)
        self.layers = nn.ModuleList([GraphormerLayer(d, ffn_dim, num_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d, out_dim)

    def forward(self, batched_data):
        padding_mask = batched_data["x"][:, :, 0].eq(0)
        padding_mask = torch.cat([padding_mask.new_zeros(padding_mask.size(0), 1), padding_mask], dim=1)
        x = self.node_feature(batched_data)
        bias = self.attn_bias(batched_data)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, bias, padding_mask=padding_mask)
        return self.out(x[0])
```
