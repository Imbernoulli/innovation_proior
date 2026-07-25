A convolution earns its keep on a grid because the grid gives every pixel the same fixed local geometry — "the pixel above," "the pixel to the upper-left" — so one small set of weights can be slid across every position and reused everywhere. Graphs have no such geometry: a node in a citation network, a protein-interaction graph, a mesh, has some number of neighbors that varies from node to node, and there is no canonical ordering among them. Any layer I want has to accept a variable-sized, unordered neighborhood and still share its parameters the way a CNN filter does. The classical answer is spectral: diagonalize the normalized Laplacian $L = I - D^{-1/2} A D^{-1/2} = U \Lambda U^T$ and filter in that eigenbasis, $g_\theta \star x = U g_\theta(\Lambda) U^T x$. But the filter is then a function of one particular graph's eigenvectors — it is not even defined on a different graph, so this route buys nothing inductive, and the naive form costs an $O(N^3)$ eigendecomposition and an $O(N^2)$ forward pass. Chebyshev truncation ($g_\theta(\Lambda) \approx \sum_k \theta_k T_k(\tilde\Lambda)$) fixes the cost and locality by working with powers of $L$ directly, and Kipf and Welling push this to first order to get GCN, $H^{(l+1)} = \sigma(\tilde D^{-1/2}\tilde A \tilde D^{-1/2} H^{(l)} W^{(l)})$ with self-loops $\tilde A = A+I$. That layer is cheap and strong, but look at exactly what weight it gives neighbor $j$ of node $i$: $1/\sqrt{\tilde d_i \tilde d_j}$, a number fixed by the two degrees alone, identical for every model ever trained on that graph. It cannot say "trust this neighbor more than that one." The spatial alternatives that try to avoid the eigenbasis all buy their locality with a different tax: a separate weight matrix per node degree, or fixed-size ordered neighborhood patches imposed on a set that has no order, or pseudo-coordinates that are themselves structural functions of degree. GraphSAGE gets closer by learning aggregator functions of features rather than a per-node embedding table, which does make it inductive — but its neighborhoods are a fixed-size uniform *sample*, never the whole neighborhood, its mean/GCN-style aggregators still pool neighbors equally, and its LSTM aggregator has to paper over feeding an unordered set through a sequence model by shuffling the order every time. None of these methods gives a node a learned, feature-dependent weight on each of its neighbors while staying inductive and using the whole neighborhood.

I propose Graph Attention Networks (GAT): let each node be a query that attends over its own neighborhood, so the weight on each neighbor is learned from the current features rather than fixed by degree. Concretely, every node's features $h_i \in \mathbb{R}^F$ first go through a shared linear transform $W \in \mathbb{R}^{F'\times F}$, exactly as a CNN filter is shared across positions — this is what makes the layer inductive, since $W$ and everything built from it is a function of features, not a table indexed by node identity. The importance of neighbor $j$ to node $i$ is then scored by a single-layer additive network rather than a dot product: $e_{ij} = \mathrm{LeakyReLU}\big(a^T [Wh_i \,\|\, Wh_j]\big)$, with a trainable vector $a \in \mathbb{R}^{2F'}$ and slope $0.2$ on the LeakyReLU. The additive form matters and is not an arbitrary choice: after the shared $W$, a plain dot product $(Wh_i)\cdot(Wh_j)$ is a fixed, symmetric bilinear form with no further trainable capacity, whereas splitting $a = [a_l; a_r]$ gives $e_{ij} = a_l^T(Wh_i) + a_r^T(Wh_j)$, which is generally different from $e_{ji} = a_l^T(Wh_j) + a_r^T(Wh_i)$ — the scorer can decide $j$ matters to $i$ more than $i$ matters to $j$, something a symmetric dot product can never express. LeakyReLU rather than plain ReLU keeps a gradient flowing on the negative branch, which matters because a low pre-softmax score is the model's way of saying "this neighbor is unimportant," and it needs to be able to learn to push a coefficient further down rather than have that branch go dead. Structure enters only as a mask: I compute $e_{ij}$ only for $j \in N_i$ (the first-order neighbors, with a self-loop so a node also attends to itself), then normalize with a softmax restricted to that neighborhood,
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in N_i}\exp(e_{ik})},$$
so a degree-7 node and a degree-200 node both produce a convex combination over their own neighbors — the coefficients are comparable regardless of degree, which a scheme like GCN's degree-normalized weight has to bake in by construction rather than learn. The output is the attention-weighted sum of the transformed neighbor features passed through a nonlinearity,
$$h'_i = \sigma\Big(\sum_{j\in N_i}\alpha_{ij} Wh_j\Big).$$
Because $a^T[Wh_i\|Wh_j]$ is linear in the concatenation, it factors as $a_l^T(Wh_i) + a_r^T(Wh_j)$: I compute one source score per node and one target score per node, each an $O(|V|F')$ pass, and broadcast-add them rather than ever materializing an $N\times N$ concatenation, so the whole layer costs $O(|V|FF' + |E|F')$ per head — on par with GCN, with no eigendecomposition, no matrix inversion, and no imposed ordering. This construction contains GCN-style averaging as a degenerate case: if the scorer is held constant, $a(x,y)=1$, the masked softmax over a neighborhood of size $|N_i|$ collapses to $\alpha_{ij} = 1/|N_i|$ for every neighbor, i.e. uniform averaging over the self-loop-augmented neighborhood — not numerically identical to GCN's $1/\sqrt{\tilde d_i \tilde d_j}$, since that normalization is symmetric across an edge while row-wise averaging is not, but structurally the same family, with GCN sitting at the corner where the scorer is forbidden from depending on features. Turning the scorer on is exactly the added capacity. Self-attention is a noisy, high-variance estimator on its own, so I follow the multi-head recipe: $K$ independent heads, each with its own $W^k, a^k$, and in hidden layers their outputs are concatenated, $h'_i = \|_{k=1}^K \sigma(\sum_{j\in N_i}\alpha_{ij}^k W^k h_j)$, giving $K\!\cdot\!F'$ features and keeping each head's distinct notion of relevance as a separate channel rather than collapsing it. Concatenating cannot be right on the final, class-scoring layer, though — $K$ independent $C$-dimensional logit vectors concatenated into $KC$ numbers have no sensible meaning — so there I average the heads and delay the final nonlinearity until after the average, $h'_i = \sigma\big(\frac{1}{K}\sum_{k=1}^K\sum_{j\in N_i}\alpha_{ij}^k W^k h_j\big)$, which keeps the output $C$-dimensional and ensembles the heads in logit space. With only twenty labeled nodes per class in the transductive citation setting, this model would overfit immediately without help, so beyond ordinary input dropout I apply dropout directly to the normalized attention coefficients $\alpha_{ij}$: dropping a fraction of them after the softmax exposes each node to a stochastically sampled neighborhood at every step, a regularizer that falls directly out of having attention weights to perturb in the first place. I use ELU (not ReLU) as the hidden nonlinearity, since the attention-weighted sums can go negative and ELU keeps a smooth, nonzero response there, and I initialize $W$ and $a$ with Glorot/Xavier scaling so early-training scores are not saturated. Stacking $L$ layers reaches the $L$-hop neighborhood, exactly as in GCN, and deeper stacks can take residual connections the way image networks do; and because the dense $e_{ij}$ scoring only actually needs to touch existing edges, a large single graph can instead scatter the $|E|$ edge scores into a per-node sparse softmax and a sparse matrix multiply for the weighted sum, bringing storage down to $O(|V|+|E|)$ while computing the identical layer.

Here is the layer and the network built from it, exactly as I use them: a single dense attention head, and the stack that concatenates $K$ heads in the hidden layer and averages heads on the prediction layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """One graph attention head: h (N x F_in) -> h' (N x F_out), each neighbor
    weighted by a learned, feature-dependent attention coefficient."""

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super().__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.concat = concat                                   # hidden layer (ELU) vs last (raw)

        self.W = nn.Parameter(torch.empty(in_features, out_features))   # shared transform
        nn.init.xavier_uniform_(self.W, gain=1.414)
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))         # attention vector
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)                   # negative slope 0.2

    def forward(self, h, adj):
        Wh = h @ self.W                                        # Wh_i: (N, F_out)

        # e_ij = LeakyReLU(a^T[Wh_i || Wh_j]) = LeakyReLU(a_l^T Wh_i + a_r^T Wh_j)
        Wh_src = Wh @ self.a[: self.out_features, :]           # (N, 1)
        Wh_tgt = Wh @ self.a[self.out_features :, :]           # (N, 1)
        e = self.leakyrelu(Wh_src + Wh_tgt.T)                  # (N, N) pairwise scores

        e = e.masked_fill(adj <= 0, float("-inf"))            # mask non-neighbors
        alpha = F.softmax(e, dim=1)                            # alpha_ij = softmax over N_i
        alpha = F.dropout(alpha, self.dropout, training=self.training)  # stochastic neighborhood

        h_prime = alpha @ Wh                                   # h'_i = sum_j alpha_ij Wh_j
        return F.elu(h_prime) if self.concat else h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, out_heads=1, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        # hidden layer: K heads, outputs CONCATENATED -> nhid*nheads features
        self.heads = nn.ModuleList(
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        )
        # output layer: heads produce class-score logits, then get AVERAGED
        self.out_heads = nn.ModuleList(
            GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
                                alpha=alpha, concat=False)
            for _ in range(out_heads)
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([head(x, adj) for head in self.heads], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        logits = torch.stack([head(x, adj) for head in self.out_heads], dim=0).mean(dim=0)
        return F.log_softmax(logits, dim=1)
```

For the transductive citation networks I use a two-layer model, eight heads of eight features each in the hidden layer (concatenated to 64), and a single output head over the class logits with softmax; dropout $p=0.6$ on both the inputs and the attention coefficients, $L_2$ weight decay, Adam, and early stopping on a validation metric. On the inductive PPI graphs, where the test graphs are never seen during training, the same layer applies unchanged because nothing in it references a fixed node identity or a fixed graph's eigenbasis — only the mask changes from graph to graph.
