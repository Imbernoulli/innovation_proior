# Graph U-Nets: gPool, gUnpool, and a U-Shaped Graph Encoder-Decoder

## Problem

Graph convolution gives a local message-passing layer, but an image-style encoder-decoder also needs graph down-sampling, graph up-sampling, and aligned skip connections. The goal is a multi-resolution graph network that can contract a graph to encode higher-order features and then restore the original node resolution.

## Method

**gPool.** For node features $X^\ell\in\mathbb R^{N\times C}$ and a trainable projection vector $\mathbf p^\ell\in\mathbb R^C$, compute the normalized scalar projection
$$\mathbf y = X^\ell\mathbf p^\ell/\|\mathbf p^\ell\|_2.$$
Keep the $k$ nodes with largest entries in $\mathbf y$, extract their feature rows and adjacency rows/columns, and gate the kept feature rows by $\mathrm{sigmoid}(\mathbf y(\mathrm{idx}))$:
$$
\begin{aligned}
\mathrm{idx} &= \mathrm{rank}(\mathbf y,k),\\
\tilde{\mathbf y} &= \mathrm{sigmoid}(\mathbf y(\mathrm{idx})),\\
\tilde X^\ell &= X^\ell(\mathrm{idx},:),\\
X^{\ell+1} &= \tilde X^\ell \odot (\tilde{\mathbf y}\mathbf 1_C^\top).
\end{aligned}
$$
The rank operation selects whole nodes; the gate puts $\mathbf p$ into the continuous forward values, so $\mathbf p$ receives gradient even though the top-$k$ index choice is discrete.

**Graph connectivity augmentation.** Selecting nodes can delete intermediary nodes and disconnect surviving nodes. Before restricting the adjacency, square the current self-looped adjacency so direct and two-hop reachability are present in the Boolean support:
$$A^2=A^\ell A^\ell,\qquad A^{\ell+1}=A^2(\mathrm{idx},\mathrm{idx}).$$
The self-loop qualification matters: without self-loops, $A^2$ marks exact length-two walks rather than all paths of length at most two.

**gUnpool.** Store the selected indices during pooling. During decoding, scatter the current $k$ rows back into an $N\times C$ zero matrix at those original positions:
$$X^{\ell+1}=\mathrm{distribute}(0_{N\times C},X^\ell,\mathrm{idx}).$$
This is parameter-free and preserves row identity, so same-level encoder features can be added back into the decoder.

**Improved GCN.** The GCN equation uses a larger self-loop weight,
$$\hat A=A+2I,\qquad X_{\ell+1}=\sigma(\hat D^{-1/2}\hat A\hat D^{-1/2}X_\ell W_\ell),$$
so a node's own feature has more pre-normalization weight than any single neighbor.

**Architecture.** Embed raw node features with a GCN, run repeated down blocks, process the bottleneck graph, then run matched up blocks in reverse. The implementation skeleton is:
GCN before each pool, save adjacency/features/indices, pool; bottleneck GCN; reverse unpool, GCN, additive same-level skip; final additive skip from the embedded input. A node-level head can consume the final restored node features, while a graph-level head can read out all returned states.

## Code

The snippet below mirrors the dense `GraphUnet` body: the equations above state the normalized-projection form, while this implementation realizes the scalar scorer with `Linear(C,1)`, gates with sigmoid scores, squares the Boolean adjacency before restriction, scatters rows in unpooling, and uses additive skips.

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

## Properties

- gPool adds one projection vector per pooling layer and no dense assignment matrix.
- The gate is essential for backpropagation into the projection direction.
- gUnpool is a scatter by saved indices, so skip connections align by original node identity.
- The second graph power is a local connectivity repair, not an all-pairs densification.
- Additive skips match the fixed hidden width of the dense implementation.
