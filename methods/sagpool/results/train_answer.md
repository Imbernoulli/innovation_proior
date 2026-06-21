The task is graph classification: one label for a whole graph — is this protein an enzyme, is this molecule a mutagen — from graphs whose node count and wiring vary from one example to the next. A message-passing backbone already solves the per-node half of the problem. Stacking graph-convolution layers of the form $h^{(l+1)} = \sigma\!\left(\tilde D^{-1/2}\tilde A\,\tilde D^{-1/2} h^{(l)}\Theta\right)$, with $\tilde A = A + I$ and $\tilde D$ its degree matrix, lets every node mix a normalized blend of its own and its neighbors' features through a learned $\Theta$, so after a few layers each node carries an embedding of its local structural neighborhood. What is missing is the last step: a classifier needs one fixed-size vector per *graph*, and what we are holding is a variable-size pile of node vectors — thirty for a small molecule, several thousand for a large protein. That collapse — the readout, the pooling — is the unsettled operation, and it must be permutation-invariant because the node numbering is arbitrary.

The crude option is a global readout: sum, mean, or coordinatewise max over all node vectors at once. These are permutation-invariant and fixed-size, but they throw structure away. A mean weights an uninformative leaf exactly as heavily as a hub that determines the graph's class; a sum lets graph size leak into the magnitude; a max keeps one salient coordinate and forgets the rest. None of them consults the *topology* when deciding how much a node should count, and none builds anything multi-scale. A CNN does not classify an image by averaging its pixels — it pools in stages, downsampling a little at a time, building a pyramid that preserves composition. The graph analogue should coarsen in stages too. So we want learned, hierarchical coarsening. DiffPool was the first end-to-end learned hierarchical pooler: at layer $l$ a GNN emits a soft assignment $S^{(l)} = \mathrm{softmax}(\mathrm{GNN}_l(A^{(l)}, X^{(l)})) \in \mathbb{R}^{n_l \times n_{l+1}}$, then coarsens both features and wiring, $X^{(l+1)} = S^{(l)\top} Z^{(l)}$ and $A^{(l+1)} = S^{(l)\top} A^{(l)} S^{(l)}$. It is fully differentiable and genuinely learns a hierarchy, but the bookkeeping is fatal. $S$ is dense, and $S^\top A S$ produces a *dense* coarsened adjacency even when $A$ was sparse, so storage runs to $O(k|V|^2)$ and the big protein graphs fall over. Worse and more structural: $n_{l+1}$ is fixed when the model is built, so the parameter count couples to the provisioned node count and one cluster size is forced onto a dataset whose graphs span two orders of magnitude — on D&D from thirty nodes to nearly six thousand — so a fraction sensible for the median graph actually *expands* the small ones. gPool / Graph U-Nets buys back the complexity by replacing soft assignment with hard top-$k$ node selection driven by one projection vector $p$: score $y = Xp/\lVert p\rVert$, keep the top-$k$ indices, gate them with a sigmoid, and set $X^{(l+1)} = X(\mathrm{idx},:)\odot \sigma(y(\mathrm{idx}))$, $A^{(l+1)} = A(\mathrm{idx},\mathrm{idx})$. Indexing the adjacency keeps it sparse at $O(|V|+|E|)$, and the lone vector $p$ makes the parameter count independent of graph size. But its score $y_i = x_i\cdot p/\lVert p\rVert$ is a function of node $i$'s own features and nothing else — the adjacency never enters. A graph pooler decides what to survive while structurally blind to the graph: two nodes with identical features but utterly different roles, a peripheral leaf and a central hub, get identical scores. That is the gap to close.

I propose SAGPool — Self-Attention Graph Pooling. The defining move is to keep gPool's entire sparse top-$k$ machinery but replace its topology-blind score with one that sees the graph. I want a per-node scalar — one importance number per node, the same shape as $y$, so $\mathrm{rank}$, select, gate, and index all carry over untouched — but a scalar that depends on the node's neighborhood, not just on the node. The source of such a quantity is already running: the convolution's $\tilde D^{-1/2}\tilde A\,\tilde D^{-1/2}$ term is *exactly* the first-order graph-Laplacian operator that injects topology into a per-node value. Convolution uses it with $\Theta \in \mathbb{R}^{F\times F'}$ to produce an $F'$-dimensional embedding, but nothing forces width $F'$. Setting the output width to one, with $\Theta_{att} \in \mathbb{R}^{F\times 1}$, the same operator emits a single scalar per node already mixed across the neighborhood:

$$Z = \sigma\!\left(\tilde D^{-1/2}\tilde A\,\tilde D^{-1/2} X\,\Theta_{att}\right),\qquad Z \in \mathbb{R}^{N\times 1},\quad \Theta_{att}\in\mathbb{R}^{F\times 1}.$$

This is the score I wanted. It has gPool's shape, so it slots straight into the existing plumbing, but unlike $x_i\cdot p/\lVert p\rVert$ it is not a function of node $i$ alone: the normalized-adjacency multiply makes $Z_i$ depend on $i$'s features *and* all of $i$'s neighbors' features, weighted by the wiring. It is the self-attention idea — let the input supply the criterion for scoring its own parts — except the input is the graph and the scoring respects the edges. The cost is one length-$F$ vector per pooling layer, independent of $N$, and the operator is the same sparse normalized adjacency the block's convolution already uses, so we get DiffPool's structure-aware scoring at gPool's sparse, size-independent price. (Nothing forces precisely the Kipf-Welling GCN here — the only requirement is a GNN mapping $(X,A)$ to a per-node scalar, so Chebyshev, GraphSAGE, or GAT drop in equally; the GCN is the cheapest topology-aware default.)

With the score reintroducing a hard top-$k$ I must handle the differentiability trap carefully, because it is the same one that explains gPool's gate. The selection $\mathrm{idx} = \mathrm{top\text{-}rank}(Z, \lceil kN\rceil)$ is an $\mathrm{argsort}$: a piecewise-constant function of the scores, zero derivative almost everywhere and a discontinuous jump elsewhere. Selecting and passing $X(\mathrm{idx},:)$ forward would make nothing downstream depend smoothly on $Z$, so $\Theta_{att}$ would get no gradient and never train — selection silently severs the parameter from the loss. The fix is to make the kept features depend continuously on their scores by multiplying each survivor's features by a continuous function of its own score. Writing $\mathrm{idx} = \mathrm{top\text{-}rank}(Z, \lceil kN\rceil)$ and $Z_{\mathrm{mask}} = Z_{\mathrm{idx}}$,

$$X_{out} = X_{\mathrm{idx},:}\odot Z_{\mathrm{mask}},\qquad A_{out} = A_{\mathrm{idx},\mathrm{idx}},$$

where $\odot$ broadcasts the per-node scalar across feature channels. Now the forward features carry $Z$, the loss depends continuously on $Z$ and hence on $\Theta_{att}$, and the projection trains; and $A_{\mathrm{idx},\mathrm{idx}}$ is sparse indexing — keep an edge only if both endpoints survive — so the coarsened adjacency stays $O(|V|+|E|)$ with no $S^\top A S$ blowup. The gate $\sigma$ I take as $\tanh$ rather than gPool's sigmoid. As a gradient bridge any smooth monotone function works; as a multiplicative gate it should be bounded so a wild score cannot blow a node up to dominate the readout — both $\tanh$ and sigmoid are bounded. The difference is range: sigmoid lives in $(0,1)$ and can never flip a sign, while $\tanh \in (-1,1)$ is centered at zero, so a node the network has learned to score negatively gets its features sign-flipped and a node scored near zero is nearly erased — the score becomes not just "how much" but "in which direction," and boundary nodes of low importance are suppressed toward nothing. Because $\tanh$ is monotone, ordering by the raw score equals ordering by $\tanh(\text{score})$, which lets me rank on the raw pre-activation score and apply $\tanh$ only inside the feature gate, saving one elementwise activation on every node before the cut. One further choice: I keep a *ratio* $k\in(0,1]$, retaining $\lceil kN\rceil$ nodes, rather than a fixed count. A fixed $K$ is the exact disease that sank DiffPool — far too many for a thirty-node molecule, far too few for a six-thousand-node protein — whereas a fraction of *this* graph's node count adapts to each input, and the ceiling guarantees at least one survivor. Since $k$ is a hyperparameter, not a learned matrix, nothing about the parameter count depends on $N$.

One SAGPool layer is therefore: score every node with the scalar graph convolution, keep the top $\lceil kN\rceil$ by score, gate the survivors by $\tanh(\text{score})$, and index the adjacency. Stacked the way CNN pooling stacks, a block is one graph-convolution layer followed by one SAGPool layer, and three blocks give a genuine pyramid — convolve-then-coarsen three times, each pool keeping a fraction of the previous level. After each block I read the coarser node set out with a Jumping-Knowledge-style summary, concatenating mean and max over the current node set, $s = (\tfrac{1}{N}\sum_i x_i)\,\|\,\max_i x_i$ — mean for the average signal across survivors, max for the single most salient coordinate, fixed-size regardless of how many nodes survived — and then *sum* the per-block readouts so the coarse summaries (after one and two poolings) and the fine summary all feed the linear classifier. Both reductions are permutation-invariant, so the whole graph vector is invariant to node numbering. The single idea — change a graph convolution's output width to one and rank by it — also yields free two-hop variants ($Z = \sigma(\mathrm{GNN}(X, A + A^2))$, since $A^2$ connects two-hop neighbors, or two stacked scoring convs) and a multi-head consensus ($Z = \tfrac{1}{M}\sum_m \sigma(\mathrm{GNN}_m(X,A))$); the base method is the single GCN-scored top-$k$. The implementation leans on existing utilities: a batched top-$k$ that computes $\lceil kN_g\rceil$ per graph and an adjacency filter that keeps an edge only if both endpoints survived and relabels indices.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool.topk_pool import topk, filter_adj  # per-graph top-k + sparse edge filter


class SAGPool(nn.Module):
    """Self-attention graph pooling layer (Lee et al., 2019). Scores nodes with a
    single-output graph convolution (features + topology), keeps the top
    ceil(ratio*N) per graph, gates kept features by tanh(score), indexes adjacency."""

    def __init__(self, in_channels, ratio=0.5, Conv=GCNConv, non_linearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        # Scalar GCN score; paper notation applies sigma to obtain Z.
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()                 # raw scalar score
        perm = topk(score, self.ratio, batch)                            # same order as tanh(score)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)        # tanh gate
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(                              # A[idx, idx], stays sparse
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm


class GraphReadout(nn.Module):
    """Hierarchical SAGPool readout: 3 x (GCN conv -> SAGPool), each block summarized
    by max||mean in the official-code order; block summaries are summed."""

    def __init__(self, hidden_dim, num_layers, ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.pool1 = SAGPool(hidden_dim, ratio=ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPool(hidden_dim, ratio=ratio)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = SAGPool(hidden_dim, ratio=ratio)
        self.output_dim = hidden_dim * 2          # max||mean per block, summed across blocks

    def _readout(self, x, batch):
        # Official code uses max||mean; the paper equation writes mean||max.
        return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=-1)

    def forward(self, x, edge_index, batch, layer_outputs):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        s1 = self._readout(x, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        s2 = self._readout(x, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        s3 = self._readout(x, batch)

        return s1 + s2 + s3
```
