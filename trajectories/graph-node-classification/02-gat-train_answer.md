The mean aggregator landed where I predicted its weakest point would be, and the shape of the failure is the argument for what comes next. Cora held at 0.7923 with almost no seed spread, PubMed held at 0.7693, but CiteSeer collapsed to 0.6603 and its seeds fanned out — {0.679, 0.674, 0.628}, a five-point swing, with macro-F1 cratering to 0.5811 on the worst seed. That variance *is* the diagnosis. CiteSeer is the sparsest, least-homophilous graph; the plain mean counts every neighbor equally with no degree damping, so it blurs an off-topic neighbor into the average at full weight, and a poorly-connected node's representation is then at the mercy of which particular neighbors it happens to have. The mean has no knob for "this neighbor matters more than that one," and neither does the scaffold default — the default's $1/\sqrt{\tilde d_i\tilde d_j}$ is *fixed* by degrees and the mean's $1/|N(i)|$ is *uniform*, both pinned by graph structure, neither able to decide that for predicting node $i$'s class one neighbor matters more than another. The fix is not a different aggregation *shape* — separating self from neighbor was the right backbone and I keep it — but to make the neighbor weight a *learned function of the features*.

I propose masked multi-head graph attention — GAT. Spell out what I need and it is exactly attention: a node is the query, its neighbors are the items, each neighbor is scored, the scores are normalized over the neighborhood, and the output is the score-weighted sum. The mean is the special case where every score is equal; attention is the generalization that can make them unequal, and a softmax-weighted sum over a set stays permutation-invariant, so it keeps the one property the mean had right. I build the layer so each piece falls out of a need. First a shared linear transform on every node, $\mathbf W\mathbf h_i$ — the minimum that gives the layer expressive power and lets it change dimension, the same mechanism everywhere. Then a shared scorer for how much neighbor $j$ matters to node $i$, $e_{ij}=a(\mathbf W\mathbf h_i,\mathbf W\mathbf h_j)$. Here is a fork I have to settle. The maximally flexible option is full self-attention — let every node attend over every other node, dropping the graph and rediscovering relationships — but that is $O(N^2)$ and, worse, throws away the graph I trust: the off-topic-neighbor problem is about *weighting real neighbors*, not discovering new edges. So I inject the graph by *masking* — compute $e_{ij}$ only for $j\in N(i)$, the actual edges, plus a self-loop so the node keeps its own features, the same reason the convolution folds in $\mathbf A+\mathbf I$. Multi-hop reach comes from stacking the same two layers as before.

Scores from a node with three neighbors and a node with three hundred must be made comparable, so I normalize over each neighborhood with a softmax,
$$\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k\in N(i)}\exp(e_{ik})},$$
which does three things at once: it turns raw scores into a proper distribution over the neighborhood regardless of its size, it is permutation-invariant over the set, and it is differentiable with well-behaved gradients. For the scorer $a$ I take the cheapest learnable form — a single weight vector over the two transformed endpoints with a nonlinearity, so the score is not merely bilinear. The concatenation $\vec a^\top[\mathbf W\mathbf h_i\Vert\mathbf W\mathbf h_j]$ splits *additively* into a source score and a destination score, $\vec a_s^\top\mathbf W\mathbf h_i+\vec a_d^\top\mathbf W\mathbf h_j$, and this split is not cosmetic — it is the efficient implementation. I compute one scalar per node for each half and the edge score becomes a broadcast sum over the edge list, never a materialized $2F'$ concatenation per edge. The full normalized weight is
$$\alpha_{ij}=\mathrm{softmax}_j\!\big(\mathrm{LeakyReLU}(\vec a_s^\top\mathbf W\mathbf h_i+\vec a_d^\top\mathbf W\mathbf h_j)\big),\qquad \mathbf h_i'=\sigma\!\Big(\sum_{j\in N(i)}\alpha_{ij}\,\mathbf W\mathbf h_j\Big).$$

The nonlinearity must be LeakyReLU rather than plain ReLU, and the reason connects straight back to CiteSeer. The pre-activation is routinely negative, and I need the *ordering* of scores among neighbors to survive into the softmax. Plain ReLU clamps every negative score to exactly zero — a whole range of "this neighbor is somewhat less relevant" collapses to one value and the gradient there dies — so the scorer could never learn to discriminate among the down-weighted neighbors, which is exactly what CiteSeer needs: the off-topic citation has to be pushed *below* the on-topic ones, not flattened to zero alongside them. LeakyReLU keeps a small negative slope (0.2) and preserves both the ordering and the gradient.

Attention trained from scratch on ~20 labels per class can latch onto one bad scoring pattern early — which is a precise restatement of the seed-to-seed instability the mean aggregator showed on CiteSeer — so I do not bet everything on a single head. I run $K$ independent heads in parallel, each with its own $\mathbf W^k$ and $\vec a^k$, and combine them: in the hidden layer I **concatenate** the heads, keeping every head's view (8 heads of 8 features into the 64 hidden width), and at the **output** layer I **average** a single head's class scores, because concatenating heads at the output would give $K\times C$ numbers instead of a class vector. Averaging several independent attempts is itself the variance reducer the mean could not supply — the consensus of several weightings rather than one fragile one. The small-label regime also wants aggressive regularization, and here there is a graph-specific lever: I apply dropout at rate 0.6 to the *normalized attention coefficients* themselves, which zeros some $\alpha_{ij}$ each step so that each node sees a stochastically sampled subset of its neighborhood every iteration — augmentation on the connectivity itself, tailored to this exact layer. Heavy feature dropout (0.6) before each conv rounds it out, and I use ELU for the activation — smooth, allowing negative outputs so activations stay near zero-mean, which converges faster in this moderately deep attentional stack.

The cost stays linear: applying $\mathbf W$ to all nodes is $O(|V|FF')$, the scores and weighted sums over edges are $O(|E|F')$, so one head is $O(|V|FF'+|E|F')$ — on par with the mean layer, with no eigendecomposition and no inversion. Multi-head multiplies parameters by $K$, but the heads are independent and run in parallel. I am paying more parameters than the floor — eight heads, two attention vectors each — but buying a learned, feature-driven per-neighbor weight, and I expect the gain concentrated where uniform weighting failed: Cora clearly up past 0.82 (dense and homophilous, attention can sharpen already-good neighborhoods), CiteSeer's floor lifted above 0.6603 and its seed spread tightened by the learned weights plus head-averaging plus attention-dropout, and PubMed roughly holding near 0.7693 since its clean large neighborhoods left little for attention to gain.

```python
# EDITABLE region of custom_nodecls.py — step 2: multi-head graph attention
class CustomMessagePassingLayer(MessagePassing):
    """Graph attention layer with multi-head additive attention."""

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 8, concat: bool = True, negative_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        if concat:
            assert out_channels % heads == 0
            self.head_dim = out_channels // heads
        else:
            self.head_dim = out_channels
        self.lin = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))   # a_s (source score half)
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))   # a_d (destination half)
        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        H, D = self.heads, self.head_dim
        x = self.lin(x).view(-1, H, D)
        alpha_src = (x * self.att_src).sum(dim=-1)             # per-node source scalar (per head)
        alpha_dst = (x * self.att_dst).sum(dim=-1)             # per-node destination scalar (per head)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, alpha_src=alpha_src, alpha_dst=alpha_dst)
        out = out.view(-1, H * D) if self.concat else out.mean(dim=1)
        return out + self.bias

    def message(self, x_j: Tensor, alpha_src_i: Tensor, alpha_dst_j: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_src_i + alpha_dst_j                      # additive score = a_s.Wh_i + a_d.Wh_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)            # normalize over the neighborhood
        alpha = F.dropout(alpha, p=0.6, training=self.training)  # dropout on attention coefficients
        return x_j * alpha.unsqueeze(-1)


class CustomGNN(nn.Module):
    """Multi-head attention GNN: 8 concat heads then 1 averaged head."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels, heads=8, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels, heads=8, concat=True))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
```
