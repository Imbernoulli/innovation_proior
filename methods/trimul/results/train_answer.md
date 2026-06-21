The pair representation $z_{ij} \in \mathbb{R}^{c_z}$ (with $c_z = 128$) is an edge feature on the complete graph whose nodes are a protein's residues, and it is the quantity the network reads out into a distogram and ultimately into atom coordinates. The trouble is that an edge is not free. A single pairwise distance is cheap to guess in isolation, but a *collection* of pairwise distances is physically meaningful only if it is mutually realizable in three dimensions, and metric geometry says an arbitrary symmetric matrix of "distances" is almost never embeddable in $\mathbb{R}^3$. The first and cleanest necessary condition is the triangle inequality: for every triple of residues $(i, j, k)$, $d_{ij} \le d_{ik} + d_{kj}$, with stricter higher-order Euclidean distance-matrix constraints beyond that. So $d_{ij}$ is squeezed by the two other sides of every triangle $\{i, j, k\}$, and the edge $(i, j)$ sits in $N-2$ such triangles, one per third residue $k$. Any update that treats the pair map as $N^2$ independent cells — a per-pixel MLP, a 2D convolution on the distance map, or axial attention — is geometrically blind to this. The convolution is the worst offender: locality in index space $(i, j)$ has nothing to do with geometric adjacency. Axial attention is the strong baseline and the one worth dissecting: for a fixed row $i$ it lets $z_{ij}$ attend over $z_{ik}$ for all $k$, and for a fixed column $j$ it lets $z_{ij}$ attend over $z_{kj}$ for all $k$, at $O(N)$ interactions per axis. But stare at what each operation binds together. Row attention pairs $(i, j)$ with $(i, k)$; column attention pairs $(i, j)$ with $(k, j)$. The two sides that constrain $(i, j)$ through a shared third residue — $(i, k)$ and $(k, j)$ — are exactly the two that never enter the *same* message indexed by the same $k$. Axial attention can route along the sides of a triangle one side at a time, but it never closes the triangle. That is the gap.

I propose the triangular multiplicative update together with triangular self-attention — two complementary triangle-shaped pair operators that, for target edge $(i, j)$, build a message out of the other two sides of every triangle and aggregate it over the third residue $k$. The shape of the operation is essentially forced: walk every $k$, grab the two directed entries that represent the other two sides, combine them, and sum over $k$. This is message passing on a triangle, an edge receiving messages routed through a third node, and it costs $O(N^3)$ because the "for all $k$" adds a factor of $N$ beyond the $N^2$ edges, so the only freedom left is the combine function and the aggregation, and both must be cheap. The frugal choice for the first operator is a bilinear contraction rather than attention, because at this stage I only need the two sides to *meet*, not to be content-selected. From the (normalized) edge I form two learned projections and take their gated elementwise product, summed over the shared node. Because the pair map is directed — $z_{ij} \ne z_{ji}$ in general — which index plays the shared apex genuinely matters, and there are two distinct, complementary contractions. The outgoing version shares the *target* $k$,
$$\tilde z_{ij} = g_{ij} \odot \mathrm{Linear}\!\left(\mathrm{LayerNorm}\!\left(\textstyle\sum_k a_{ik} \odot b_{jk}\right)\right), \qquad \text{einsum } ikc, jkc \to ijc,$$
where $i$ and $j$ are the two sources pointing at a common destination $k$; the incoming version shares the *source* $k$, summing $a_{ki} \odot b_{kj}$ (einsum $kjc, kic \to ijc$), where $i$ and $j$ are the two targets emitted from a common origin $k$. Outgoing reads from the $(i, \cdot)$ and $(j, \cdot)$ rows, incoming from the $(\cdot, i)$ and $(\cdot, j)$ columns; neither alone covers both orientations of the triangle, so the block runs both, outgoing then incoming. Each component earns its place. The left and right projections $a_{ij} = \sigma(\mathrm{Linear}(z_{ij})) \odot \mathrm{Linear}(z_{ij})$ and $b_{ij} = \sigma(\mathrm{Linear}(z_{ij})) \odot \mathrm{Linear}(z_{ij})$ carry a per-channel sigmoid gate because the "for all $k$" is indiscriminate: the hard padding mask removes nonexistent residues but cannot know which real pairs carry useful evidence, so the gate is the soft, learnable selection knob that damps edges that should not contribute to the product. There are two normalizations, and each is load-bearing. The input LayerNorm conditions $z_{ij}$ before projection. The output LayerNorm on the summed result is what tames the magnitude of an $N$-term sum: the standard deviation of $\sum_k$ grows like $\sqrt{N}$ (or the mean like $N$ if the products are not centered), so without renormalization the message scale would ride on sequence length; the center norm restores a controlled magnitude regardless of how many $k$ contributed. Finally the output gate $g_{ij} = \sigma(\mathrm{Linear}(z_{ij})) \in \mathbb{R}^{c_z}$ multiplies the projected message so the target edge decides how much triangle-routed update it absorbs — confident edges with strong coevolution signal set $g$ low and refuse to be stomped on, uncertain edges open up — which is exactly the right knob given that the whole thing is added residually. This multiplicative update is the wide, cheap workhorse and runs at hidden width $c = 128$.

The multiplicative update has the right inputs but a fixed, content-blind path: every $k$ contributes through the gated Hadamard product, and the only down-weighting is the per-edge gate on $a$ and $b$, which is a function of those edges alone, never of how well edge $(i, k)$ actually matches $(i, j)$. For triangles where a few $k$ are geometrically decisive and the rest are noise, I want query-dependent selection, which is exactly attention. So the second operator lets $(i, j)$ attend over the third node while still feeling the third side. Plain row attention already lets $(i, j)$ attend over $(i, k)$ with the query from the central edge and keys/values from the left edges — but that is the axial baseline again, coupling only the two sides $(i, j)$ and $(i, k)$. The missing side is $(j, k)$, the edge between the two other endpoints, and the trick is to inject it as an additive per-head scalar bias on the attention logit, projecting $z_{jk}$ to a scalar and adding it to the $q\cdot k$ affinity. For the starting-node version, sharing residue $i$,
$$a^h_{ijk} = \mathrm{softmax}_k\!\left(\tfrac{1}{\sqrt{c}}\, q^h_{ij}\cdot k^h_{ik} + b^h_{jk}\right), \qquad o^h_{ij} = g^h_{ij} \odot \textstyle\sum_k a^h_{ijk}\, v^h_{ik}, \qquad \tilde z_{ij} = \mathrm{Linear}\big(\mathrm{concat}_h\, o^h_{ij}\big).$$
The $q^h_{ij}\cdot k^h_{ik}$ term brings in sides $ij$ and $ik$; the $+\,b^h_{jk}$ injects the missing side $jk$, so the affinity depends on all three sides of the triangle. This is the whole point and what makes it *triangular* rather than vanilla axial attention with a gratuitous bias: ordinary self-attention would compute a logit $q_{ij}\cdot k_{ik}$ that is a function of two sides only, and could not distinguish two third residues $k$ with identical $ik$ edges but wildly different $jk$ edges — geometrically completely different triangles. Adding $b^h_{jk}$ breaks that degeneracy. The $1/\sqrt{c}$ is the usual dot-product temperature so the softmax does not saturate at the per-head width $c = 32$ with $4$ heads, and $g^h_{ij}$ is the same sigmoid output gate controlling absorption as in the multiplicative case. The ending-node sibling shares residue $j$ instead, with logit $\mathrm{softmax}_k\!\big(\tfrac{1}{\sqrt{c}}\, q^h_{ij}\cdot k^h_{kj} + b^h_{ki}\big)$ and values $v^h_{kj}$; as before one orientation of the shared node is not enough on a directed pair map, so the block does starting-node then ending-node. Implementationally the bias $b^h_{jk}$ is read from the $j$-th row and $k$-th column of a per-head scalar projection of $z$, broadcasting over the starting node $i$ with shape $[*, 1, H, J, K]$; the ending-node version is the same row-attention machinery applied after a transpose of the two residue axes and transposed back. I keep both families rather than only the richer attention because they trade off: the multiplicative update is the cheap, fully symmetric, content-blind contraction that gives every block a broad triangle-shaped mixing pass at low constant cost and so earns the wide width $c = 128$, while the attention is the more expensive content-routed refinement that sharpens onto the decisive $k$ and so runs narrow at $c = 32$ with $4$ heads. The order within a block is deliberate — two multiplicative updates, then two attentions, then a $4\times$-wide transition MLP — so the broad symmetric consistency pass comes first, content routing next, and the transition digests the result. The first three residual sub-layers take rowwise dropout; the ending-node attention takes columnwise dropout in the original orientation. Each block thereby nudges the pair representation toward a geometry that can actually be built in 3D rather than $N^2$ independently guessed numbers.

```python
import torch
import torch.nn as nn
from functools import partialmethod

# Linear, LayerNorm, Attention, DropoutRowwise, DropoutColumnwise,
# PairTransition, permute_final_dims, and is_fp16_enabled are trunk primitives.


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)
        self.linear_a_p = Linear(c_z, c_hidden)
        self.linear_a_g = Linear(c_z, c_hidden, init="gating")
        self.linear_b_p = Linear(c_z, c_hidden)
        self.linear_b_g = Linear(c_z, c_hidden, init="gating")
        self.linear_z = Linear(c_hidden, c_z, init="final")
        self.linear_g = Linear(c_z, c_z, init="gating")
        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a, b):
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))   # [*, c, i, k]
            b = permute_final_dims(b, (2, 1, 0))   # [*, c, k, j]
        else:
            a = permute_final_dims(a, (2, 1, 0))   # [*, c, k, i]
            b = permute_final_dims(b, (2, 0, 1))   # [*, c, k, j]
        p = torch.matmul(a, b)
        return permute_final_dims(p, (1, 2, 0))    # [*, i, j, c]

    def forward(self, z, mask=None):
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask * self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)
        b = mask * self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)

        if is_fp16_enabled():
            a_std = a.std()
            b_std = b.std()
            if a_std != 0.0 and b_std != 0.0:
                a = a / a_std
                b = b / b_std
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        return x * g


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        super().__init__()
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(c_in)
        self.linear = Linear(c_in, no_heads, bias=False, init="normal")
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    def forward(self, x, mask=None):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        x = self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])

        if not self.starting:
            x = x.transpose(-2, -3)
        return x


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)


class PairStack(nn.Module):
    def __init__(self, c_z, pair_dropout=0.25):
        super().__init__()
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, 128)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, 128)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, 32, 4)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, 32, 4)
        self.pair_transition = PairTransition(c_z, n=4)
        self.dropout_row = DropoutRowwise(pair_dropout)
        self.dropout_col = DropoutColumnwise(pair_dropout)

    def forward(self, z, pair_mask):
        z = z + self.dropout_row(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.dropout_row(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.dropout_row(self.tri_att_start(z, mask=pair_mask))
        z = z + self.dropout_col(self.tri_att_end(z, mask=pair_mask))
        z = z + self.pair_transition(z, mask=pair_mask)
        return z
```
