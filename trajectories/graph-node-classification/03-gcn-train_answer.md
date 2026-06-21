Attention did what I predicted on Cora and missed what I predicted on CiteSeer, and that gap is the whole lesson. Cora rose to 0.8260, clearly past the mean aggregator's 0.7923, exactly as the dense-homophilous story said; PubMed held at 0.7777, a hair above 0.7693, clean neighborhoods leaving little for learned weighting to gain. But CiteSeer came in at 0.7077 with seeds {0.711, 0.696, 0.716}: the mean *went up* from 0.6603, yet the variance did not collapse the way I claimed — a ~2-point swing remains and the worst seed barely beats the mean aggregator's best. Learned per-neighbor weighting bought Cora a lot, CiteSeer a little and shakily, PubMed nothing, and that asymmetry makes me doubt whether the *learned* part is pulling its weight here. Attention spends a great deal — eight heads, two attention vectors each, a LeakyReLU and a softmax and a dropout over every edge — all fit from ~20 labels per class. But citation graphs are strongly homophilous: a citation usually links same-topic papers, so the "right" answer for most neighbors is just "count this neighbor, with a sensible degree-based discount." A *fixed* weighting that already encodes "discount high-degree neighbors" might match what attention is straining to learn, at none of the parameter cost and none of the seed variance — and CiteSeer's stubborn swing is consistent with exactly that: the scorer is overfitting the tiny label set, so which weighting it lands on depends on the seed.

So I throw away the learned weight and go back to a *principled fixed* weight — but a better-founded one than the floor's plain unnormalized mean. To find the right one I have to ask what "convolution on a graph" even means. An ordinary convolution is built on *translation* — shift a filter by one position, multiply-and-sum, shift again — but on a graph "shift by one" is undefined: node 7 has three neighbors, node 8 has nine hundred, there is no canonical next node. The only translation-free definition goes through the convolution theorem: convolution in space is multiplication in the Fourier domain. The graph's Fourier basis comes from the symmetric normalized Laplacian $\mathbf L=\mathbf I-\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$, which is real and symmetric, so it diagonalizes with an orthonormal eigenbasis $\mathbf L=\mathbf U\boldsymbol\Lambda\mathbf U^\top$ — exactly what a Fourier basis must be — with a bounded spectrum $\lambda\in[0,2]$ that I will lean on twice. A spectral filter is anything diagonal in that basis, $g_\theta\star\mathbf x=\mathbf U\,\mathrm{diag}(\theta)\,\mathbf U^\top\mathbf x$, but a free diagonal $\theta$ has $O(N)$ parameters, is non-localized, needs an $O(N^3)$ eigendecomposition, and applies as a dense $O(N^2)$ multiply — hopeless against 20 labels on a 20,000-node graph. The escape is to write the filter as a smooth function of the eigenvalues and approximate it by a degree-$K$ Chebyshev polynomial, $g_\theta(\boldsymbol\Lambda)\approx\sum_k\theta_k T_k(\tilde{\boldsymbol\Lambda})$. Pushing the polynomial back through the eigenbasis, the $\mathbf U$'s telescope — $(\mathbf U\boldsymbol\Lambda\mathbf U^\top)^k=\mathbf U\boldsymbol\Lambda^k\mathbf U^\top$ — so $g_\theta\star\mathbf x=\sum_k\theta_k T_k(\tilde{\mathbf L})\mathbf x$ with no eigenvectors anywhere, sparse, $O(K|E|)$, $K$-hop localized, $K{+}1$ coefficients, graph-independent. All four walls fall.

I propose the renormalized symmetric graph convolution — GCN — the leanest fill of this contract. The move that lands me on the principled fixed weight is to notice $K$ in the polynomial does double duty, setting both the filter's expressiveness and the receptive field. I untangle them by driving $K$ to 1 and letting *depth* supply the receptive field by composition. With $K=1$, $g_\theta\star\mathbf x\approx\theta_0\mathbf x+\theta_1\mathbf L\mathbf x$; approximating $\lambda_{\max}\approx2$ (justified by the $[0,2]$ bound, with the trainable weights free to absorb the error) collapses $\tilde{\mathbf L}=\mathbf L-\mathbf I=-\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}$, leaving $\theta_0\mathbf x-\theta_1\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}\mathbf x$. Tie the coefficients, $\theta=\theta_0=-\theta_1$, to cut parameters and regularize, and the operator becomes $\theta(\mathbf I+\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2})$ — "yourself plus your symmetric-normalized neighbors."

The choice of normalization is the entire point of this rung. The obvious alternative is the random-walk form $\mathbf D^{-1}\mathbf A$, whose row $i$ is literally the average of $i$'s neighbors — the floor's plain mean. The symmetric form weights edge $(i,j)$ by $1/\sqrt{d_i d_j}$, the geometric mean of the two degrees, and that is exactly the degree damping the mean lacked: it down-weights edges to high-degree neighbors more aggressively, so a citation to a paper everyone cites carries less than one to an obscure paper. That is the homophilous-graph prior — "discount the popular neighbor" — baked into a fixed weight with no parameters and nothing to overfit, and it is, I claim, precisely the useful part of what attention was straining to relearn. The symmetric form is also the one the spectral story requires: $\mathbf D^{-1}\mathbf A$ is not symmetric, has no orthonormal eigenbasis, and breaks the Fourier picture, whereas $\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}=\mathbf I-\mathbf L$ shares $\mathbf L$'s spectrum.

There is a stability catch, the second use of the $[0,2]$ bound. The operator $\mathbf I+\mathbf D^{-1/2}\mathbf A\mathbf D^{-1/2}=2\mathbf I-\mathbf L$ has top eigenvalue $\approx2$, because the self-loop's unit mass was bolted on *after* normalizing the neighbor part and so is not accounted for in the degree denominators; stacking an operator of spectral radius 2 amplifies signals like $2^{\text{depth}}$. The fix is the renormalization trick: fold the self-loop in *before* normalizing. Add a self-loop everywhere, $\tilde{\mathbf A}=\mathbf A+\mathbf I$, recompute degrees $\tilde{\mathbf D}=\mathbf D+\mathbf I$, and symmetric-normalize that,
$$\tilde{\mathbf P}=\tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\tilde{\mathbf D}^{-1/2}.$$
Now $\tilde{\mathbf P}$ is the normalized adjacency of the self-looped graph, its spectral radius is back to 1, and the node and its neighbors sit on the same footing — edge $(i,j)$ gets $1/\sqrt{\tilde d_i\tilde d_j}$, the self-loop $1/\tilde d_i$, all from one $\tilde{\mathbf D}$ — so stacking no longer blows up. This is exactly the scaffold's default layer, the one I have been replacing for two rungs, and landing back on it is the right thing rather than a retreat: the principled fixed weight, with proper degree normalization and a consistently-normalized self-loop, is a strong *low-variance* baseline on homophilous citation graphs precisely because it has no learned edge weights to overfit the tiny label set that destabilized CiteSeer. A cross-check from an unrelated direction confirms the operator: read node-wise, $\mathbf h_i'=\sigma\big(\sum_{j\in N(i)\cup\{i\}}\frac{1}{\sqrt{\tilde d_i\tilde d_j}}\mathbf W\mathbf h_j\big)$ is the 1-Weisfeiler–Lehman vertex-refinement step with the hash replaced by a differentiable normalized map — and the normalization forces the same $\sqrt{\tilde d_i\tilde d_j}$ constant the spectral derivation produced. Two unrelated motivations landing on one operator is exactly the sign I am at the right fixed weight.

So the layer is the leanest on the ladder: one shared $\mathbf W$ per layer, propagate through $\tilde{\mathbf P}$, add a bias — no attention vectors, no per-edge softmax, no head dimension, far fewer parameters than the eight-head attention layer, which on a 20-label budget is itself a regularizer. Two layers, ReLU and dropout 0.5 between, give the two-hop field. I expect PubMed to hold or slightly exceed gat's 0.7777 (clean large neighborhoods where degree damping is the whole story), CiteSeer's mean to move past 0.7077 toward ~0.715 with its {0.711, 0.696, 0.716} spread collapsing as the learned weight that was overfitting it disappears, and Cora close to gat around 0.82, possibly a touch under — the one place a fixed weight risks giving back attention's learned sharpening.

```python
# EDITABLE region of custom_nodecls.py — step 3: renormalized symmetric graph convolution
class CustomMessagePassingLayer(MessagePassing):
    """Standard graph convolutional layer (symmetric normalization, renormalization trick)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.lin(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))   # self-loop folded in first
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]            # 1/sqrt(d~_i d~_j): degree damping
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    """2-layer renormalized graph convolution with ReLU and dropout."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```
