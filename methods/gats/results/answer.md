# Graph Attention Networks (GAT), distilled

GAT is a neural layer for graph-structured data built from **masked self-attention** over each
node's neighborhood. Each node computes a new representation as a weighted sum of its neighbors'
transformed features, where the weights are **learned, feature-dependent attention coefficients**
— so different neighbors get different importances, the layer needs no eigendecomposition or
graph-specific spectral parameters, and it works inductively once the neighborhoods of a new
graph are supplied.

## Problem it solves

Generalize the CNN's weight-shared local filter to arbitrary graphs, where a node has a
variable-sized, unordered set of neighbors. A good layer must (1) be a shared function of features
(inductive — transfers to unseen nodes/graphs), (2) use the whole neighborhood, (3) avoid
eigendecompositions / inversions / imposed orderings / degree-indexed weight banks / structural
pseudo-coordinates, and (4) assign **different importances** to different neighbors — the capacity
that spectral GCN (fixed `1/√(d̃_i d̃_j)` neighbor weights) and equal-pooling aggregators lack.

## Key idea

Let each node be a query that **attends over its neighbors**. After a shared linear transform `W`,
score each neighbor with an additive single-layer network, normalize the scores over the
neighborhood with softmax, and output the score-weighted sum of neighbor features. Structure
enters only as a **mask** on which pairs are scored.

## The graph attentional layer

Input node features `h = {h_1,…,h_N}`, `h_i ∈ R^F`. Shared weight `W ∈ R^{F'×F}` and attention
vector `a ∈ R^{2F'}`.

1. Raw coefficient (importance of `j` to `i`), with LeakyReLU (negative slope 0.2):
   `e_ij = LeakyReLU( a^T [ W h_i ‖ W h_j ] )`.
2. **Masked attention**: compute `e_ij` only for `j ∈ N_i` (first-order neighbors, including `i`).
3. **Normalize** over the neighborhood:
   `α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k∈N_i} exp(e_ik)`.
4. **Output**: `h'_i = σ( Σ_{j∈N_i} α_ij W h_j )`.

**Multi-head** (`K` independent heads) — concatenate in hidden layers:
`h'_i = ‖_{k=1}^K σ( Σ_{j∈N_i} α_ij^k W^k h_j )` (gives `K·F'` features);
**average** on the prediction layer (concatenating class logits is meaningless), delaying the
final nonlinearity: `h'_i = σ( (1/K) Σ_{k=1}^K Σ_{j∈N_i} α_ij^k W^k h_j )`.

Cost per head: `O(|V| F F' + |E| F')` — on par with GCN; no eigendecomposition or inversion. The
receptive field after `L` layers is the `L`-hop neighborhood (extend depth with skip connections).

## Why each choice

- **Attention over neighbors** is the one mechanism that gives learned per-neighbor importance
  while natively handling variable-sized, unordered sets and requiring no graph eigenbasis — the
  exact union of properties prior methods each missed.
- **Masked attention** (restrict to `N_i`, set non-neighbor logits to `−∞` before softmax) injects
  the graph as a mask, keeping cost `O(|E|)` instead of all-pairs `O(N^2)`, while still letting the
  weights vary by relevance.
- **softmax over the neighborhood** makes coefficients comparable across nodes of very different
  degree (each node's weights sum to 1) and yields positive, interpretable, alignment-like weights.
- **Additive `a^T[Wh_i ‖ Wh_j]`, not dot-product**: after the shared `W`, a dot product is a fixed
  bilinear form with no extra trainable capacity to shape the comparison and is symmetric; the
  single-layer net has its own weights `a` (split into source/target halves) so the score is
  trainable and **asymmetric** (`e_ij ≠ e_ji`).
- **LeakyReLU (slope 0.2), not ReLU/tanh**: low pre-softmax scores must keep a gradient so the
  layer can learn to push a coefficient down; ReLU would zero the negative branch and its gradient.
- **Multi-head, concat in hidden layers**: stabilizes the otherwise high-variance self-attention
  and keeps each head's distinct view as separate channels; averaging at the output keeps the
  class dimension and ensembles heads in logit space.
- **Dropout on the normalized attention coefficients**: exposes each node to a stochastically
  sampled neighborhood per step — a strong regularizer for small training sets.
- **Constant attention `a(x,y)=1` recovers GCN-style averaging** (`α_ij = 1/|N_i|`), confirming the
  layer *generalizes* fixed-weight pooling by making the weights learned.

## Efficient scoring

Since `a^T[Wh_i ‖ Wh_j]` is linear in the concatenation, split `a = [a_l ; a_r]`, each in `R^{F'}`:
`a^T[Wh_i ‖ Wh_j] = a_l^T (W h_i) + a_r^T (W h_j)`. Compute one source score per node and one
target score per node (`O(|V|F')` each), then broadcast-add — never materializing the `N×N`
concatenations. A sparse implementation scatters the `|E|` edge scores into a per-node sparse
softmax and does a sparse matmul for the weighted sum, giving `O(|V| + |E|)` storage.

## Working code

Dense head and full network (PyTorch), filling the `GraphLayer` slot of the node-classification
harness with the attention rule and mirroring the canonical split-score implementation:

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

Typical transductive settings (citation networks): a 2-layer model, first layer `K=8` heads of
`F'=8` features each (64 total) with ELU, output layer usually one head of `C` features with
softmax; when several prediction heads are used, average their logits before the task
softmax/sigmoid. Use dropout `p=0.6` on inputs and attention coefficients, `L2` weight decay,
Adam, and early stopping on a validation metric.

For large single graphs, score only existing edges and use a sparse row-wise softmax followed by
sparse matrix multiplication; the TensorFlow sparse head does exactly this and keeps storage
linear in `|V| + |E|` for batch size one.
