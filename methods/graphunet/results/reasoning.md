Let me start from the shape of the task. I have a graph, every node may need its own prediction or embedding, and the right representation for a node depends on context that can live several hops away. On images I already know the architectural answer to that shape: shrink the grid while convolving so the deep representation sees large regions, then expand back to full resolution, copying high-resolution features across the bottleneck so the decoder can localize. The convolution part has a graph analogue. The shrinking and expanding parts do not.

The missing piece is not just an implementation nuisance. On a grid, pooling is possible because the units have an order and local neighborhoods come in fixed windows. I can say "take every $2\times2$ block" before I look at the image. On a graph there is no canonical first, second, third node, and no fixed rectangle around a node. If I impose an arbitrary node ordering and pool by chunks of indices, the result depends on the ordering rather than the graph. If I cluster the graph before training, the hierarchy cannot adapt to the features or the task. If I pool globally, I get one vector and lose the possibility of a decoder. So the operation has to be learned, graph-structured, and still cheap.

A flat GCN is the first thing to test mentally, because it already propagates information. With the usual rule
$$X_{\ell+1}=\sigma(\hat D^{-1/2}\hat A\hat D^{-1/2}X_\ell W_\ell),\qquad \hat A=A+I,$$
one layer mixes each node with its one-hop neighborhood; $r$ layers reach $r$ hops. But this keeps every layer at the original resolution. A node after many flat layers has seen farther away, yet there is never a coarser unit representing a larger region. Repeated averaging also pushes node features toward one another, so depth alone can blur away the distinctions that node prediction needs. I need an actual graph contraction, not only a larger hop count.

What should a contraction keep? If I borrow $k$-max pooling naively, I select the largest activations in feature maps. On a graph that means the selected units can come from different channels of different nodes. Then I no longer know which nodes survived, and I cannot form a coherent adjacency for the next graph-convolution layer. That failure gives me a constraint: the selectable object must be an entire node row, not a scalar activation inside a node. Soft cluster assignment is another route: learn $S$, form $X'=S^\top Z$ and $A'=S^\top A S$, and let each coarse node be a weighted mixture of original nodes. That is differentiable, but it pays with a dense assignment matrix, another GNN to produce the assignments, fixed coarse sizes, and often side losses to keep the assignments sensible. I want the lighter object: choose real nodes, remember their indices, and let the task loss train the choice.

So I need one score per node. The score has to be computed from the whole feature vector $\mathbf x_i\in\mathbb R^C$, be permutation-compatible across nodes, and add very few parameters. A single learned direction does exactly that. Let $\mathbf p\in\mathbb R^C$ be trainable and measure the signed scalar projection of each node feature onto that direction:
$$y_i=\frac{\mathbf x_i^\top \mathbf p}{\|\mathbf p\|_2}.$$
The denominator matters. Without it, the length of $\mathbf p$ changes the scale of the scores even though the ranking only needs the direction; with it, $y_i$ is the length of the shadow of $\mathbf x_i$ on the unit vector $\mathbf p/\|\mathbf p\|_2$. Now "keep the largest scores" has a clean meaning: keep the nodes whose feature vectors retain the most content along the learned direction. For a feature matrix $X^\ell$, this gives
$$\mathbf y=X^\ell\mathbf p^\ell/\|\mathbf p^\ell\|_2,\qquad \mathrm{idx}=\mathrm{rank}(\mathbf y,k),$$
and the smaller graph can use the selected rows $X^\ell(\mathrm{idx},:)$ and selected rows and columns of the adjacency. This is cheap: one vector per pooling layer.

Then I hit the real training problem. The top-$k$ index set is discrete. If $\mathbf p$ changes a little and the same nodes remain in the top $k$, the selected rows are unchanged; if the set changes, the operation jumps. Either way, the ranking itself does not give a useful ordinary gradient to $\mathbf p$. If I stop here, the only trainable object in the pooling layer is not really trainable by backpropagation.

The score has to affect a continuous value after the discrete choice. I can keep the top-$k$ indices, extract their scores, and use those scores as gates on the surviving rows:
$$\tilde{\mathbf y}=\mathrm{sigmoid}(\mathbf y(\mathrm{idx})),\qquad
X^{\ell+1}=X^\ell(\mathrm{idx},:)\odot(\tilde{\mathbf y}\mathbf 1_C^\top).$$
Now $\mathbf p$ enters the forward values through $\mathbf y$ and the sigmoid. For a selected row $i$, the derivative has the form
$$\frac{\partial L}{\partial \mathbf p}
\supset
\left\langle \frac{\partial L}{\partial X^{\ell+1}_i},\,X^\ell_i\right\rangle
\sigma(y_i)(1-\sigma(y_i))\,\frac{\partial y_i}{\partial \mathbf p},$$
where
$$\frac{\partial y_i}{\partial \mathbf p}
=\frac{\mathbf x_i}{\|\mathbf p\|_2}
-\frac{(\mathbf x_i^\top\mathbf p)\mathbf p}{\|\mathbf p\|_2^3}.$$
That derivative is exactly the derivative of a normalized projection: move $\mathbf p$ in a direction that changes the unit direction, not merely its length. The sigmoid is bounded and positive, so the gate attenuates or preserves a selected node without flipping feature signs or letting score magnitude explode. The discrete ranking still chooses which rows survive, but the selected scores create the gradient path that trains the projection direction.

The adjacency needs just as much care as the features. If I select nodes and take only $A^\ell(\mathrm{idx},\mathrm{idx})$, every path through a dropped node disappears. Two selected nodes that were two hops apart through one discarded node become disconnected. The next GCN would then have less information flow, and a selected node could even become isolated. Graph powers are the natural repair. For an adjacency with self-loops, the Boolean pattern of $A^2$ marks pairs at distance at most two: the diagonal/self-loop terms preserve the node itself, direct edges survive through the self-loop cross terms, and length-two walks add the missing shortcut edges. Without self-loops, $A^2$ only counts exact length-two walks, so the adjacency being squared must be the current self-looped graph, or the graph-power convention must explicitly include shorter paths. With that understood, the pooled adjacency becomes
$$A^2=A^\ell A^\ell,\qquad A^{\ell+1}=A^2(\mathrm{idx},\mathrm{idx}).$$
Why stop at two? Each pooling decision is made after a GCN has already mixed one-hop information into the node features. The extra risk created by dropping nodes is that selected nodes linked through one removed intermediary lose contact. The second power repairs that local break. Higher powers would densify the graph more aggressively and weaken locality.

Now the inverse operation almost writes itself because the down-sampling kept real nodes and recorded their positions. Suppose the current coarse feature matrix has $k$ rows and the matching earlier graph had $N$ rows. I allocate an $N\times C$ zero matrix and scatter the $k$ rows back to their saved indices:
$$X^{\ell+1}=\mathrm{distribute}(0_{N\times C},X^\ell,\mathrm{idx}).$$
Rows that were selected receive their current coarse features; rows that were dropped remain zero until the following graph convolution and skip connection provide information. This is the graph analogue of unpooling with switches. It is parameter-free, and it preserves node identity, which is the property the decoder needs for row-wise skip connections.

The convolution itself can be nudged for node prediction. In the usual self-looped GCN, a node's own feature and each neighbor's feature enter with the same unnormalized weight before degree normalization. But for predicting a node's own label or building its own embedding, its current feature should have a larger prior weight than any single neighbor. The smallest change is to use
$$\hat A=A+2I$$
before normalization. The graph operation is still the same first-order message passing, but the self feature receives double the adjacency weight.

Putting the pieces together, I want a U-shaped graph network. First, an embedding GCN maps high-dimensional raw node features into a working hidden dimension. At each resolution, a GCN updates node features before pooling so the score for a node can depend on its local topology through the mixed representation. Then the pooling layer ranks whole nodes by normalized scalar projection, gates the selected rows, and builds the next adjacency from the second graph power restricted to those rows and columns. After the last contraction, a bottleneck GCN processes the smallest graph. The decoder walks the saved levels in reverse: restore the previous node count by scattering rows to their recorded indices, run a GCN at that finer graph, and add the saved encoder features from the same level. Addition is the clean skip operation when every hidden layer keeps the same width; concatenation is possible architecturally, but fixed width lets the decoder tensor and saved encoder tensor align without adding another projection. After the final decoder level, a long skip adds the embedded input representation back in. The body can return the restored node features for a node-level head, or all restored states for a graph-level readout.

When I turn this into code, I need the implementation to keep the same moving parts: a one-scalar-per-node scorer, sigmoid-gated top-k node selection, Boolean second graph power before restriction, scatter unpooling, and the down/save/pool/bottom/up/add path. I keep the normalized projection as the mathematical definition of the scorer; in a compact module, a one-output linear layer carries that scalar scorer, and its sigmoid output can be used both for ranking and for the continuous gate because sigmoid is monotone.

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

The chain now has no loose part. Whole-node top-$k$ gives a coherent coarsened graph. Normalized scalar projection gives the ranking a learnable direction rather than a scale trick. The sigmoid gate is the differentiable route by which the projection vector learns. The second graph power repairs local connectivity lost by deleting intermediary nodes. Scatter unpooling restores node identity. Additive skips bring back the high-resolution features that pooling necessarily dropped. The result is an encoder-decoder for graphs built from ordinary graph convolutions plus a small, trainable, invertible node-selection mechanism.
