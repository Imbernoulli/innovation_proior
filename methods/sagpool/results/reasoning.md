Let me start from what actually hurts. I have a graph and I want one label for the whole thing — enzyme or not, mutagen or not — and the message-passing backbone has already done its job: after stacking a few graph-convolution layers, every node carries a good embedding of its local structural neighborhood. Each layer does h^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} h^{(l)} Θ), with Ã = A + I and D̃ its degree matrix, so a node's new feature is a normalized blend of its own and its neighbors' features pushed through a learned Θ. That part works. What I don't have is the last step: the classifier needs one fixed-size vector per graph, and what I'm holding is a variable-size pile of node vectors — thirty of them for a small molecule, a few thousand for a big protein. Collapsing that pile into a single vector is the readout, the pooling, and that's the operation that isn't settled. So the whole game is: out of a bag of node embeddings of unknown size, with the graph's wiring sitting right there, manufacture one good graph-level vector — and I'd better keep it invariant to how the nodes happen to be numbered, because that numbering is arbitrary.

The crude thing I could do is just reduce all the node vectors at once — sum them, or average them, or take a coordinatewise max. That's a global readout, it's permutation-invariant, it gives a fixed size, done. But stare at what it throws away. A mean weights a leaf node that touches nothing exactly as heavily as a hub that determines the whole graph's class; a sum lets graph size leak into the magnitude; a max keeps one salient coordinate and forgets the rest. None of them looks at the *structure* when deciding how much each node should count, and none of them builds anything multi-scale. A CNN doesn't classify an image by averaging all the pixels — it pools in stages, downsampling a little at a time, building a pyramid of coarser and coarser representations so that composition survives. The graph analogue should coarsen the graph in stages too: shrink it, look again, shrink it again. Flat one-shot pooling can't represent the structural composition that distinguishes one graph from another. So I want hierarchical, learned coarsening, not a single global squash.

Who's tried this? DiffPool is the obvious ancestor — Ying and colleagues' differentiable pooling, the first end-to-end learned hierarchical pooler. Its idea is soft clustering: at layer l a GNN spits out an assignment matrix S^{(l)} = softmax(GNN_l(A^{(l)}, X^{(l)})) ∈ R^{n_l × n_{l+1}}, where row i is a probability distribution over the n_{l+1} clusters of the next, coarser graph. Then it coarsens both the features and the wiring with that matrix: X^{(l+1)} = S^{(l)T} Z^{(l)} and A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)}. Elegant — it's fully differentiable, every node is softly assigned to every cluster, and it genuinely learns a hierarchy. I want to like it. But now I do the bookkeeping. S is dense: every one of n_l nodes gets a weight on every one of n_{l+1} clusters, so S alone is n_l × n_{l+1}. Worse, A^{(l+1)} = S^T A S is a *dense* matrix even when A was sparse — the soft contraction connects everything to everything — so the coarsened adjacency is n_{l+1} × n_{l+1} of nonzeros, and storage runs to O(k|V|^2). On the big protein graphs that's hundreds of thousands of nodes squared; it falls over. And there's a second problem that bothers me more because it's structural, not just a constant factor: n_{l+1}, the number of clusters, has to be fixed when I build the model. So the parameter count depends on the number of nodes I provision for, and I have to commit to one cluster size for the whole dataset. But these datasets have graphs spanning two orders of magnitude in size — on D&D the smallest graph has thirty nodes and the largest nearly six thousand. If I pick a cluster size that's a sensible fraction of the median graph, it's far larger than most small graphs entirely, so "pooling" them actually expands them. One fixed cluster size cannot be right for a dataset whose graphs don't share a scale. Wall: the soft, dense assignment is what gives DiffPool its differentiability and its hierarchy, and it's also exactly what makes it quadratic and size-coupled.

So how do I keep the hierarchy and the end-to-end learning but ditch the dense assignment? The move that suggests itself is to stop assigning softly to clusters and instead just *select* a subset of the existing nodes — drop the unimportant ones, keep the important ones, and let the kept nodes (with their existing wiring restricted to them) be the coarser graph. No new cluster centroids, no S^T A S contraction. If I select node indices idx, the coarser feature matrix is just X(idx, :) and the coarser adjacency is just A(idx, idx) — I'm *indexing* the sparse adjacency, not contracting it, so it stays sparse. Storage drops to O(|V|+|E|). And if the rule that picks idx is governed by a small fixed-length parameter rather than an n_l × n_{l+1} matrix, the parameter count stops depending on graph size. This is exactly the route gPool takes — Gao and Ji's Graph U-Nets, and Cangea and colleagues' sparse hierarchical classifier. Concretely gPool learns a single projection vector p, scores each node by how far its features point along p,

  y = X p / ||p||,   idx = rank(y, k),   X^{(l+1)} = X(idx,:) ⊙ (sigmoid(y(idx)) 1^T),   A^{(l+1)} = A(idx,idx),

keeping the top-k by score. That's the complexity fix I wanted: sparse, size-independent parameters, hierarchical.

But before I adopt it I have to understand one thing in it that looks like a wart and is actually load-bearing: why the sigmoid gate? The selection step idx = rank(y, k) is a hard top-k — it returns indices, an argsort. That's a piecewise-constant function of the scores: nudge a score a hair and almost always the same nodes are selected, so the derivative of "which nodes" with respect to y is zero almost everywhere, and where it isn't zero it's a discontinuous jump. So if I select and then just pass X(idx,:) forward, nothing downstream depends *smoothly* on y, which means nothing depends smoothly on p, which means p gets no gradient and never learns. The selection has silently severed the parameter from the loss. The fix is to make the kept features depend continuously on their scores: multiply each kept node's features by a continuous function of its own score, X(idx,:) ⊙ gate(y(idx)). Now the forward value carries y through, gradient flows back into p through the gate, and the projection trains. That's the whole reason the gate is there — it's not decoration, it's the differentiable bridge across the non-differentiable top-k. Good, I'll need that same trick whatever scoring I use.

So gPool's machinery is right. But now look hard at *what it scores by*, because that's where it disappoints me. y_i = x_i · p / ||p|| is the scalar projection of node i's own feature vector onto a learned direction p. It's a function of x_i and nothing else. The adjacency matrix never enters the score. Read that again — this is a *graph* pooling method, and the decision of which nodes survive the coarsening is made while completely blind to the graph. Two nodes with identical feature vectors but utterly different roles — one a peripheral leaf, one a central hub bridging two communities — get identical scores and are kept or dropped together. The topology, the one thing that makes this data a graph rather than a feature table, is ignored at exactly the moment it should matter most. That's the gap. DiffPool at least fed A into its assignment GNN; gPool bought sparsity by scoring on features alone and threw the structure away.

What would it take to score a node by its features *and* its place in the graph? I want a per-node scalar — one importance number per node, same shape as gPool's y, so the rest of the machinery (rank, select, gate, index) carries over untouched — but I want that scalar to depend on the node's neighborhood, not just on the node. Let me think about where a per-node quantity that already mixes in the neighborhood could come from. … I have one sitting right in front of me. The convolution layer I'm already running does h^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} h^{(l)} Θ). That D̃^{-1/2} Ã D̃^{-1/2} multiply is *exactly* the operation that makes a node's output depend on its neighbors — it's the first-order graph-Laplacian term, the thing that injects topology into a per-node value. The convolution uses it with Θ ∈ R^{F×F'} to produce an F'-dimensional embedding per node. But there's nothing forcing the output to be F'-dimensional. If I set the output width to one — take Θ_att ∈ R^{F×1} — the very same operator produces a single scalar per node, and that scalar has already been mixed across the neighborhood by D̃^{-1/2} Ã D̃^{-1/2}. So:

  Z = σ(D̃^{-1/2} Ã D̃^{-1/2} X Θ_att),   Z ∈ R^{N×1},   Θ_att ∈ R^{F×1}.

That's it — that's the score I wanted. It has gPool's shape (one number per node) so I can slot it straight into rank/select/gate/index, but unlike x_i · p / ||p|| it is *not* a function of node i alone: the normalized-adjacency multiply means Z_i depends on i's own features and on all of i's neighbors' features, weighted by the graph's wiring. Features and topology, both, in the importance score. And it's the same self-attention idea that's everywhere now — let the input itself supply the criterion for scoring its own parts — except the "input" here is the graph, and the scoring respects the edges. The contribution is one number: change gPool's output dimension on a graph convolution from F to 1 and use it to rank.

Let me check the cost of this, because the whole point was to not pay DiffPool's price. The extra parameters are Θ_att ∈ R^{F×1} — a single F-vector per pooling layer. That's independent of the number of nodes, same as gPool's p, far better than DiffPool's size-dependent assignment matrix. And the D̃^{-1/2} Ã D̃^{-1/2} term I need is not a new dense object; it is the same sparse normalized-adjacency operator already used by the graph-convolution layer in the same block. Mathematically I am applying the same graph Laplacian approximation with output width one. In a concrete library call, a separate scalar GCN layer may rebuild or cache that normalization internally, so the faithful implementation point is "same sparse operator," not "literally reuse a cached matrix." The topology-awareness is still cheap: one scalar GCN with a length-F weight, not a dense assignment matrix. So I get DiffPool's structure-aware scoring at gPool's sparse, size-independent cost. That's the trade I was looking for.

Now I have to be careful about the same differentiability trap, because I've reintroduced a hard top-k. I'll select idx = top-rank(Z, ⌈kN⌉) and that's an argsort again — zero gradient to Θ_att through the selection. So I must gate: when I keep the selected node features, I multiply them by their own scores. Write the masking step out. Let X' = X_{idx,:} be the kept feature rows, Z_mask = Z_{idx} the kept scores, and form

  X_out = X' ⊙ Z_mask,   A_out = A_{idx,idx},

where ⊙ broadcasts the per-node scalar Z_mask across that node's feature channels. Now the forward features carry Z, so the loss depends continuously on Z and therefore on Θ_att, and the projection trains exactly the way gPool's p trains through its sigmoid gate. The A_{idx,idx} is the sparse indexing — keep only edges whose both endpoints survived — so the coarsened adjacency stays sparse, O(|V|+|E|), no S^T A S blowup.

What should σ be? gPool used sigmoid. Let me think about what the gate has to do. It plays two roles at once and they pull a little differently. As a *gradient bridge* it just needs to be a smooth, monotone function of the score so gradient flows — sigmoid, tanh, anything monotone works there. As a *multiplicative gate on the features* it should be bounded, so a wildly large score can't blow a kept node's features up to dominate the readout — both sigmoid and tanh are bounded, good. The difference is the range: sigmoid lives in (0,1), so every kept node's features are scaled down by a positive factor and the gate can never flip a sign; tanh lives in (−1,1) and is centered at zero, so a node the network has learned to score *negatively* gets its features sign-flipped, and a node scored near zero is nearly erased. I like that tanh keeps the sign information — the score isn't just "how much" but can be "in which direction," and zero-importance nodes near the selection boundary get suppressed toward nothing rather than retained at half strength. Centered at zero also means the ranking key and the signed gate value are consistent: tanh is monotone, so ordering by the raw score is the same as ordering by tanh(score). So σ = tanh. One simplification falls out of that monotonicity: I can rank on the raw pre-activation score and apply tanh only inside the feature gate — the ordering is identical either way, so the top-k set is the same whether I sort by Z = tanh(score) or by score itself, and I save one elementwise tanh on every node before the cut.

So one SAGPool layer is: score every node with a scalar graph convolution Z = tanh(D̃^{-1/2} Ã D̃^{-1/2} X Θ_att); keep the top ⌈kN⌉ by Z; gate the kept features by their scores and index the adjacency:

  Z = tanh(D̃^{-1/2} Ã D̃^{-1/2} X Θ_att),
  idx = top-rank(Z, ⌈kN⌉),   Z_mask = Z_idx,
  X_out = X_{idx,:} ⊙ Z_mask,   A_out = A_{idx,idx}.

A couple of choices to pin down. Why a *ratio* k rather than a fixed number of kept nodes? Because the graphs don't share a size. If I kept a fixed K nodes, K would be far too many for a thirty-node molecule and far too few for a six-thousand-node protein — that's the exact disease that sank DiffPool's fixed cluster count. Keeping ⌈kN⌉, a fraction of *this* graph's node count, means the coarsening adapts to each input: a half-ratio halves a small graph and halves a big one. The ceiling guarantees at least one node survives even a tiny graph. And because k is a hyperparameter, not a learned matrix, nothing about the parameter count depends on N. (I notice a limitation while I'm here: k is fixed across all graphs and I can't learn a per-graph optimal ratio — the selection is a hard threshold on a continuous score, and there's no obvious differentiable knob that says "this particular graph deserves a different fraction." I could imagine recasting the keep/drop as a per-node binary decision to make the count adaptive, but that doesn't cleanly solve it; I'll keep the fixed-ratio top-k and note the gap.)

Now stack it into the network the way CNN pooling stacks. Hierarchical architecture: a block is one graph-convolution layer followed by one SAGPool layer; I stack three blocks, so the graph gets convolved-then-coarsened three times, each pool keeping a fraction of the previous level's nodes — a genuine pyramid. After each block I need to read the (now coarser) node set out to a fixed-size vector and combine the levels. For the readout I'll borrow the Jumping-Knowledge-style summary Cangea used: per block, concatenate the mean and the max over the current node set,

  s = (1/N Σ_i x_i) || (max_i x_i),

mean to capture the average signal across the surviving nodes and max to capture the single most salient coordinate, concatenated so I keep both, and fixed-size regardless of how many nodes survived. Then I *sum* the per-block readouts so the coarse-level summaries (which see the graph after one and two poolings) and the fine-level summary all contribute, and feed that to a linear classifier. Both reductions are permutation-invariant, so the whole graph vector is invariant to node numbering, as required.

Let me also note that nothing in the scoring forced me to use precisely the Kipf-Welling GCN — the only requirement was "a GNN that takes (X, A) and returns a per-node scalar," i.e. Z = σ(GNN(X, A)). Any message-passing layer with output width one qualifies: Chebyshev convolution, GraphSAGE, GAT. The GCN is the natural default because it's the cheapest topology-aware operator and I'm already running it, but the layer is a drop-in. And once I see the score as "a GNN over (X, A)," two ways to widen the receptive field of the *importance* score fall out for free, both reaching two-hop neighbors. One is to augment the edges before scoring: feed A + A^2 instead of A, since A^2 has a nonzero (i,j) exactly when i and j are two hops apart, so a single scoring convolution now mixes two-hop neighbors — Z = σ(GNN(X, A + A^2)). The other is to stack two scoring convolutions, Z = σ(GNN_2(σ(GNN_1(X, A)), A)), where the indirect aggregation of a second layer also pulls in two-hop information, at the cost of more nonlinearity and a few more parameters. And a third axis: average several independent scoring GNNs, Z = (1/M) Σ_m σ(GNN_m(X, A)), so the importance score is a consensus of M differently-initialized heads rather than one — steadier when a single head's scoring is noisy. These are variations on the one idea; the base method is the single GCN-scored top-k.

Let me write it as the code I'd actually ship, filling the readout slot. The selection plumbing — per-graph top-k within a batch, and filtering the edge list to the kept nodes — already exists as utilities (the batched top-k that computes ⌈ratio · N_g⌉ per graph and the adjacency filter that keeps an edge only if both endpoints survived and relabels indices), so I lean on those rather than reinventing them. Two implementation choices to pin down: top-k ranks on the raw scalar GCN score (equivalent to ranking on tanh(score), so I skip the redundant activation before the cut) and the gate is tanh(score) on the survivors; and the per-block readout concatenates max then mean — the order is arbitrary as long as it's fixed, since both halves feed the same linear classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool.topk_pool import topk, filter_adj  # per-graph top-k + sparse edge filter


class SAGPool(nn.Module):
    """One self-attention pooling layer: score every node with a scalar graph
    convolution (so the score sees features AND topology), keep the top-ceil(ratio*N)
    per graph, gate the kept features by tanh(score) so gradient reaches the score
    parameters through the otherwise non-differentiable top-k, and index the
    (sparse) adjacency to the survivors."""

    def __init__(self, in_channels, ratio=0.5, nonlinearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        # the only parameter of the layer: a graph conv with output width 1
        # Z = sigma( D^-1/2 A~ D^-1/2 X Theta_att ),  Theta_att in R^{F x 1}
        self.score_layer = GCNConv(in_channels, 1)
        self.nonlinearity = nonlinearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # raw per-node scalar score; topology enters via the conv's normalized adjacency
        score = self.score_layer(x, edge_index).squeeze()
        # idx = top-rank(score, ceil(ratio*N)) per graph; same ordering as tanh(score)
        perm = topk(score, self.ratio, batch)
        # X_out = X[idx] (.) tanh(score[idx]) -- the continuous gate for the score layer
        x = x[perm] * self.nonlinearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        # A_out = A[idx, idx] -- keep edges with both endpoints retained; stays sparse
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm


class GraphReadout(nn.Module):
    """Hierarchical readout: three (conv -> SAGPool) blocks, each summarized by a
    max||mean readout (fixed concat order); block summaries are summed."""

    def __init__(self, hidden_dim, num_layers, ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.pool1 = SAGPool(hidden_dim, ratio=ratio)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPool(hidden_dim, ratio=ratio)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool3 = SAGPool(hidden_dim, ratio=ratio)
        # each block's readout is max||mean = 2*hidden_dim, summed across blocks
        self.output_dim = hidden_dim * 2

    def _readout(self, x, batch):
        # s = max_i x_i || (1/N sum_i x_i)  -- fixed size, permutation invariant
        return torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=-1)

    def forward(self, x, edge_index, batch, layer_outputs):
        # block 1: convolve, then self-attention coarsen, then read out
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        s1 = self._readout(x, batch)

        # block 2: on the coarsened graph
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        s2 = self._readout(x, batch)

        # block 3: coarsen once more
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        s3 = self._readout(x, batch)

        # sum the multi-scale summaries into one graph-level vector
        return s1 + s2 + s3
```

Now the causal chain, so I can see the whole thing at once. I started needing one fixed-size, permutation-invariant vector per variable-size graph, and global mean/max/sum readouts threw away structure and built nothing multi-scale, so I wanted learned hierarchical coarsening. DiffPool gave me exactly that with a soft assignment matrix, but the matrix is dense, S^T A S makes the coarsened adjacency dense and storage quadratic, and the fixed cluster count couples parameters to graph size and can't fit a dataset of wildly varying sizes. gPool fixed the complexity by replacing soft assignment with hard top-k node selection driven by a single projection vector — sparse, O(|V|+|E|), size-independent parameters — and showed me that the multiplicative gate on the kept scores is what carries gradient across the non-differentiable top-k so the scoring parameters can train. But gPool's score y = x·p/||p|| is a function of each node's features alone; the adjacency never enters, so a graph pooler decides what to keep while blind to the graph. The fix was to notice that the normalized-adjacency term D̃^{-1/2} Ã D̃^{-1/2} in graph convolution is exactly what injects topology into a per-node value, so a graph convolution with output width one, Z = tanh(D̃^{-1/2} Ã D̃^{-1/2} X Θ_att), produces a per-node importance score that depends on features and neighborhood both — a self-attention score over the graph — at the cost of a single length-F vector and the same sparse message-passing operator. Keeping the top ⌈kN⌉ by that score, gating the survivors by tanh(score) to keep Θ_att trainable, and indexing the sparse adjacency gives one coarsening layer; stacking three conv-then-pool blocks with summed max||mean readouts gives the hierarchy. The ratio-based keep adapts to each graph's size and keeps parameters independent of N. Any (X,A)-consuming GNN can supply the score, and two-hop variants (A + A^2, stacked convs) and multi-head averaging fall out as drop-ins on the same one idea.
