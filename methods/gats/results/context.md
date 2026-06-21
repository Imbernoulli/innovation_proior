## Research question

Convolutional networks dominate grid-structured data — images, regular time series — because a
small local filter with learnable weights is slid across every position, reusing the same
parameters everywhere. A great deal of interesting data does not live on a grid: 3D meshes,
social networks, telecommunication and biological networks, citation graphs, brain connectomes.
These are naturally graphs: each node has an arbitrary number of neighbors and there is no
canonical ordering or geometry to align a filter against.

The problem is to design a neural layer that operates directly on graph-structured data: a layer
that computes a new representation for each node from its own features and the features of its
neighbors, sharing its parameters across all nodes the way a CNN filter is shared across
positions, while coping with neighborhoods of different sizes.

## Background

A graph is given by nodes with feature vectors `h_i ∈ R^F` and an adjacency structure. Two broad
strategies for generalizing convolution to graphs had crystallized, and both sit at the heart
of the field's thinking.

**Spectral graph convolution.** The classical route defines convolution through the graph Fourier
transform. The normalized graph Laplacian is `L = I − D^{-1/2} A D^{-1/2}`, symmetric
positive-semidefinite, with eigendecomposition `L = U Λ U^T`. The columns of `U` are the graph's
Fourier modes and `U^T x` is the graph Fourier transform of a node signal `x`. A spectral filter
is `g_θ ⋆ x = U g_θ(Λ) U^T x`, where `g_θ(Λ)` is a diagonal operator acting on the spectrum.
The filter is defined as a function of the eigenvalues of a particular graph's Laplacian, and
applying it requires that graph's eigenvectors `U`.

**Non-spectral / spatial convolution.** The alternative defines the operation directly in the
vertex domain, aggregating over spatially close neighbors. The recurring task is to make a single
operator work across neighborhoods of different sizes while keeping CNN-style weight-sharing.
Different proposals approach this differently — learning a separate weight matrix per node degree;
using powers of a transition matrix to define multi-hop neighborhoods with per-hop weights;
extracting and normalizing fixed-size, ordered neighborhood patches so a standard CNN applies; or
unifying these under a parametric family of weighting kernels defined over pseudo-coordinates
assigned to node pairs.

**Transductive vs inductive.** A second axis cuts across the above. Most node-embedding methods
learn one embedding vector per node by matrix-factorization-style objectives; they are
**transductive** — defined on the single fixed graph they were optimized on. An **inductive**
layer instead learns a *function* of features and local structure that applies to new nodes and
new graphs. Production systems operate on evolving graphs and constantly encounter unseen nodes,
and one would like to train on protein-interaction graphs from one organism and transfer to
another.

**Attention.** In sequence modeling, attention had become a near-default tool. Given a decoder
state and a variable-length collection of encoder annotations, an attention mechanism computes a
scalar alignment score for each annotation, normalizes the scores with a softmax, and returns a
score-weighted context vector. Bahdanau et al. (2015) used a small feedforward network for the
alignment score and trained the aligner jointly with the sequence model. When positions in a
single sequence are related to positions in that same sequence, the mechanism is self-attention.
Vaswani et al. (2017) showed that self-attention could replace recurrence and convolution in a
sequence transduction model, and that running several independent attention functions in parallel
("multi-head" attention) could stabilize and enrich the representation.

## Baselines

These are the prior methods a new graph layer would be measured against and react to.

**Spectral networks (Bruna et al., 2014).** Convolution in the Fourier domain via
`g_θ ⋆ x = U g_θ(Λ) U^T x` with free per-eigenvalue parameters. Eigendecomposition costs `O(N^3)`
and a forward pass `O(N^2)`; filters have `O(N)` parameters and live in one graph's eigenbasis.

**Smooth spectral multipliers (Henaff et al., 2015).** Parameterize `g_θ(Λ)` with smooth
coefficients so the spatial filter becomes localized; still expressed in the eigenbasis of a fixed
graph.

**Chebyshev filters (Defferrard et al., 2016).** Approximate `g_θ(Λ) ≈ Σ_{k=0}^K θ_k T_k(Λ̃)`,
`Λ̃ = 2Λ/λ_max − I`, with `T_k(x) = 2x T_{k-1}(x) − T_{k-2}(x)`, `T_0=1`, `T_1=x`. Equivalently
apply `Σ_k θ_k T_k(L̃)`, `L̃ = 2L/λ_max − I`, directly in the vertex domain; since `T_k(L̃)` is a
degree-`k` polynomial in `L`, the filter touches only the `K`-hop neighborhood and needs no
eigendecomposition — only `K` sparse multiplies by `L̃`, giving `O(K|E|)` cost. The operation is a
fixed polynomial of the graph's Laplacian.

**Graph convolutional networks (Kipf & Welling, 2017).** Take the Chebyshev filter to first order
(`K=1`, `λ_max ≈ 2`): `g_θ ⋆ x ≈ θ_0 x − θ_1 D^{-1/2} A D^{-1/2} x`. Constrain `θ = θ_0 = −θ_1`:
`g_θ ⋆ x ≈ θ (I + D^{-1/2} A D^{-1/2}) x`. The matrix `I + D^{-1/2} A D^{-1/2}` has eigenvalues in
`[0, 2]`; a *renormalization trick* adds self-loops `Ã = A + I`, degree `D̃`, and uses
`D̃^{-1/2} Ã D̃^{-1/2}`. The layer becomes
`H^{(l+1)} = σ( D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)} )`, with cost `O(|E|FC)` and strong results on
citation-network node classification. The propagation matrix is a fixed function of the graph: the
weight node `i` gives neighbor `j` is `1/√(d̃_i d̃_j)` for unweighted edges. In the standard
semi-supervised full-graph setup this operator is precomputed from the graph at hand.

**Mixture-model spatial convolution (Monti et al., 2016).** A unifying spatial framework: assign
each node pair a pseudo-coordinate `u(x, y)` and define the patch operator through learnable
weighting kernels `w_j(u)` (e.g. Gaussians) over those coordinates; GCN and diffusion convolutions
fall out as special cases. The pseudo-coordinates are typically structural (functions of node
degrees).

**GraphSAGE (Hamilton et al., 2017).** The inductive baseline. Instead of learning a fixed
embedding per node it learns *aggregator functions*:
`h_v^k = σ( W^k · CONCAT( h_v^{k-1}, AGG_k({ h_u^{k-1} : u ∈ N(v) }) ) )`, where `N(v)` is a
**fixed-size uniform sample** of `v`'s neighbors. Because the aggregators are shared functions of
features, the model applies to unseen nodes and graphs. Aggregators examined: mean (≈ GCN),
max-pooling `max({ σ(W_pool h_u + b) })`, and an LSTM (which treats the neighbor set as a sequence,
fed in random permutations).

**Bahdanau-style additive attention (Bahdanau et al., 2015).** Scores a query against each item
with a single-hidden-layer feedforward network, `e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)`,
normalized by softmax `α_ij = exp(e_ij)/Σ_k exp(e_ik)`, returning `Σ_j α_ij h_j`. Handles
variable-sized inputs; the weights are interpretable.

**Multi-head self-attention (Vaswani et al., 2017).** Self-attention as a complete building block.
Scaled dot-product `softmax(QK^T/√d_k)V` (the `1/√d_k` keeps logits from saturating the softmax),
with several independent heads run in parallel, concatenated and projected.

## Evaluation settings

The natural yardsticks already in use, transductive and inductive:

- **Transductive citation networks** — Cora (2708 nodes, 5429 edges, 7 classes, 1433 BoW features
  per node), Citeseer (3327 / 4732 / 6 / 3703), Pubmed (19717 / 44338 / 3 / 500). Nodes are
  documents, undirected edges are citations, node features are bag-of-words, each node has a class
  label. Standard split (Yang et al., 2016): 20 labeled training nodes per class, 500 validation,
  1000 test; the algorithm sees *all* nodes' feature vectors and the full graph at train time.
  Metric: mean classification accuracy on the test nodes (over many runs).
- **Inductive protein-protein interaction (PPI)** — 24 graphs (human tissues), 20 train / 2
  validation / 2 test, with the **test graphs completely unobserved during training**. ~2372 nodes
  per graph on average; 50 features per node (positional gene sets, motif gene sets, immunological
  signatures); 121 gene-ontology labels per node, multi-label. Metric: micro-averaged F1 on the
  unseen test graphs.
- Protocol: Glorot initialization; cross-entropy loss on labeled nodes; Adam; early stopping on
  validation loss/accuracy (or micro-F1) with a fixed patience. Liberal regularization (L2
  weight decay, dropout) on the small transductive training sets; comparisons against the prior
  spectral, spatial, and inductive baselines above.

## Code framework

The graph layer plugs into the same node-classification harness already used for the baselines: a
feature matrix `H ∈ R^{N×F}`, an adjacency structure `adj`, a stack of graph layers, and a final
classifier trained with masked cross-entropy and Adam. What the layer *does* with a node's
neighbors — how it weights them, whether the weights are fixed or learned, whether it needs the
whole graph — is exactly what is to be designed, so the substrate below is only the generic graph
plumbing that already exists, with one empty slot where the propagation layer will go.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    """One graph propagation layer: maps node features H (N x F_in) to new node
    features (N x F_out), given the adjacency `adj`. How a node combines its own
    features with its neighbors' features is exactly the rule to be designed."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO: the parameters of the propagation rule we will design.

    def forward(self, h, adj):
        # h:   (N, in_features)   node features
        # adj: (N, N)             graph structure (which nodes are neighbors)
        # TODO: produce new node features (N, out_features) by combining each
        #       node with its neighbors according to the rule we will design.
        raise NotImplementedError


class GraphNet(nn.Module):
    """Stack of graph layers for node classification."""

    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.dropout = dropout
        self.layer1 = GraphLayer(nfeat, nhid)
        self.layer2 = GraphLayer(nhid, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.layer1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)


# existing node-classification training loop the layer plugs into
def train(model, optimizer, features, adj, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)                       # forward through the graph net
    loss = F.nll_loss(output[idx_train], labels[idx_train])   # masked cross-entropy
    loss.backward()
    optimizer.step()                                   # Adam update
```

The outer loop supplies node features and the graph; `GraphLayer.forward` is where the
neighbor-combination rule will live.
