The problem is to give graph-structured data the same multi-scale encoder-decoder behavior that image U-Nets provide for grids. In node-level tasks the right label for a node often depends on context several hops away, so a model needs both local detail and a coarse global view. Flat graph convolutional networks keep every layer at the original node resolution; stacking more layers only increases the receptive field at the cost of over-smoothing and does not create compact region-level representations. Global pooling collapses the whole graph into one vector and therefore cannot feed a node-level decoder. Feature-wise k-max pooling selects scalar activations from arbitrary nodes and breaks the node-to-row correspondence that an adjacency matrix requires. Soft-assignment pooling is permutation invariant and differentiable, but it pays for that flexibility with a dense assignment matrix, a second pooling network, fixed cluster counts, and often auxiliary losses. What is missing is a lightweight, learned way to drop whole nodes, keep a coherent smaller graph, and later restore the original node identities so that skip connections can align row-for-row.

I propose Graph U-Nets, a U-shaped graph encoder-decoder built around two new operations called gPool and gUnpool. At each contracting step gPool scores every node with a trainable normalized projection of its feature vector onto a learned direction, keeps the top-k nodes, gates their feature rows by the sigmoid of their scores, and forms the next adjacency from the second graph power restricted to the surviving nodes. The top-k choice is discrete and therefore not directly differentiable, but the sigmoid gate puts the projection vector into the continuous forward values, so gradients flow back to the scorer and the pooling direction can be learned. Squaring the adjacency before restriction repairs local connectivity that would otherwise be lost when intermediate nodes are dropped: a self-looped adjacency squared contains both direct edges and length-two shortcuts, so two surviving nodes that were previously two hops apart remain neighbors. On the expansive side gUnpool simply scatters the coarse rows back into an all-zero matrix sized for the previous level at the saved original indices. This is parameter-free, preserves node identity, and makes additive skip connections trivial because encoder and decoder rows correspond to the same nodes.

The full architecture starts with an embedding GCN that maps raw node features into a hidden dimension. A series of down blocks then applies a GCN, saves the adjacency and features, and pools. After a bottleneck GCN processes the smallest graph, matched up blocks reverse the order: unpool to the previous resolution, apply a GCN, and add the saved encoder features from the same level. A final additive skip from the embedded input restores the finest detail. The GCN itself uses an increased self-loop weight before normalization so a node's own feature counts more heavily than any single neighbor, which is helpful for node-level predictions. The body returns the restored node features for a node-level head, or all intermediate states for a graph-level readout.

```python
import torch
import torch.nn as nn


def norm_g(g):
    degrees = torch.sum(g, 1)
    return g / degrees


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        return self.act(h)


class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        z = self.drop(h)
        weights = self.proj(z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)

    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


class Unpool(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super().__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h

        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)

        h = self.bottom_gcn(g, h)

        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)

        h = h.add(org_h)
        hs.append(h)
        return hs
```

Graph U-Nets add only one scalar scorer per pooling layer instead of a dense assignment matrix, so they stay much lighter than soft-assignment alternatives. The gate on selected rows is not merely a scaling convenience; it is the differentiable path that makes the pooling direction trainable. The second graph power is a targeted connectivity repair rather than an all-pairs densification, and scatter unpooling keeps node identity intact so that encoder and decoder features can be fused directly. Together these pieces give a graph-native encoder-decoder that combines local message passing with learned multi-resolution contraction and expansion.
